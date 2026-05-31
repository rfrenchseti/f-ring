"""Find pairs of images from two PDS3 datasets with matching geometry.

The goal is to enable photometric comparisons between two
mission/instrument datasets (e.g. Cassini ISS vs Voyager ISS at Saturn) by
locating image pairs that:

  1. Use clear filters (broadband, no spectral filter selected).
  2. Show the same target body.
  3. Have the body filling at least a minimum fraction of the field of
     view (so there is enough signal to compare).
  4. Have illumination and viewing geometry as numerically close as
     possible. Seven angles are matched: phase, incidence, emission,
     sub-solar latitude, sub-solar IAU longitude, sub-observer
     latitude, and sub-observer IAU longitude. The two latitude /
     longitude pairs ensure that the same hemisphere of the body is
     viewed under shadows that fall in the same direction, which
     matters for resolved bodies with surface texture.

This avoids the need for a photometric model (which is error-prone for
rough surfaces) because two well-matched images can be compared directly.

Each input "set" represents one mission/instrument and is described by
two collections of PDS3 metadata files:

    --setN-moon  : *_moon_summary.lbl files (per-body geometry)
    --setN-index : *_index.lbl files (per-image filter / instrument info)

Both arguments accept one or more file paths or glob patterns.

Each set can additionally be restricted to a single camera family
(``--set1-camera NAC|WAC|ANY`` and ``--set2-camera NAC|WAC|ANY``). To do a
direct COISS NAC vs COISS WAC photometric cross-check, point both sets at
the same Cassini index files and pass different ``--setN-camera`` values.

Example:

    python3 utilities/find_geometry_matched_images.py \
    --set1-name "Cassini ISS" \
    --set1-moon "/mnt/ganymede/PDS/holdings/metadata/COISS_2xxx/COISS_2*/COISS_2*_moon_summary.lbl" \
    --set1-index "/mnt/ganymede/PDS/holdings/metadata/COISS_2xxx/COISS_2*/COISS_2*_index.lbl" \
    --set2-name "Voyager ISS" \
    --set2-moon "/mnt/ganymede/PDS/holdings/metadata/VGISS_6xxx/VGISS_6*/VGISS_6*_moon_summary.lbl" \
    --set2-index "/mnt/ganymede/PDS/holdings/metadata/VGISS_6xxx/VGISS_6*/VGISS_6*_index.lbl" \
    --targets ENCELADUS MIMAS TETHYS DIONE RHEA \
    --min-fov-fraction 0.05 \
    --max-phase-diff 10 --max-incidence-diff 10 --max-emission-diff 10 \
    --max-sub-solar-lat-diff 30 --max-sub-obs-lat-diff 10 \
    --max-sub-solar-lon-diff 15 --max-sub-obs-lon-diff 15 \
    --top 1 --output /tmp/m_inner_recommended.csv

COISS NAC vs COISS WAC example (same index files on both sides):

    python3 utilities/find_geometry_matched_images.py \
    --set1-name "COISS NAC" --set1-camera NAC \
    --set1-moon "/.../COISS_2*/COISS_2*_moon_summary.lbl" \
    --set1-index "/.../COISS_2*/COISS_2*_index.lbl" \
    --set2-name "COISS WAC" --set2-camera WAC \
    --set2-moon "/.../COISS_2*/COISS_2*_moon_summary.lbl" \
    --set2-index "/.../COISS_2*/COISS_2*_index.lbl" \
    --targets ENCELADUS MIMAS TETHYS DIONE RHEA \
    --min-fov-fraction 0.05 \
    --max-phase-diff 5 --max-incidence-diff 5 --max-emission-diff 5 \
    --max-sub-solar-lat-diff 5 --max-sub-obs-lat-diff 5 \
    --max-sub-solar-lon-diff 5 --max-sub-obs-lon-diff 5 \
    --top 5 --output /tmp/coiss_nac_vs_wac.csv

Requirements:
    pandas
    numpy
    rms-pdstable
"""

import argparse
import csv
import glob
import os
import re
import sys
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pdstable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-instrument image dimensions in pixels (square images).
# Used to convert per-pixel CENTER_RESOLUTION into a full-frame extent so
# that we can compute what fraction of the field of view a given body
# fills.
INSTRUMENT_IMAGE_SIZE_PX = {
    # Cassini ISS narrow- and wide-angle cameras.
    'ISSNA': 1024,
    'ISSWA': 1024,
    # Voyager ISS narrow- and wide-angle cameras (PDS3 indexes report
    # only INSTRUMENT_NAME for VGISS).
    'NARROW ANGLE CAMERA': 800,
    'WIDE ANGLE CAMERA': 800,
}

# Map from raw INSTRUMENT_ID / INSTRUMENT_NAME values to a canonical camera
# family ('NAC' or 'WAC'). Used by the --setN-camera filter so the same set
# of index files can be split into NAC-only and WAC-only sets and compared
# directly against each other.
CAMERA_FAMILY = {
    'ISSNA':               'NAC',
    'NARROW ANGLE CAMERA': 'NAC',
    'ISSWA':               'WAC',
    'WIDE ANGLE CAMERA':   'WAC',
}
CAMERA_CHOICES = ('NAC', 'WAC', 'ANY')

# Mean radius (km) of bodies likely to appear in outer-planet ISS data.
# Used to estimate the fraction of the field of view that a body fills.
# Values are mean radii from the JPL/IAU working group; small bodies are
# approximate. Bodies without a tabulated radius are skipped.
BODY_MEAN_RADIUS_KM = {
    # Saturn system.
    'PAN': 14.0, 'DAPHNIS': 4.0, 'ATLAS': 15.1, 'PROMETHEUS': 43.1,
    'PANDORA': 40.7, 'EPIMETHEUS': 58.2, 'JANUS': 89.5,
    'AEGAEON': 0.3, 'METHONE': 1.6, 'ANTHE': 0.5, 'PALLENE': 2.5,
    'MIMAS': 198.2, 'ENCELADUS': 252.1, 'TETHYS': 531.0,
    'TELESTO': 12.4, 'CALYPSO': 9.5,
    'DIONE': 561.4, 'HELENE': 17.6, 'POLYDEUCES': 1.3,
    'RHEA': 763.5, 'TITAN': 2575.0, 'HYPERION': 135.0,
    'IAPETUS': 734.5, 'PHOEBE': 106.5,
    # Jupiter system.
    'METIS': 21.5, 'ADRASTEA': 8.2, 'AMALTHEA': 83.5, 'THEBE': 49.3,
    'IO': 1821.6, 'EUROPA': 1560.8, 'GANYMEDE': 2634.1,
    'CALLISTO': 2410.3,
    'HIMALIA': 85.0, 'ELARA': 43.0, 'PASIPHAE': 30.0, 'SINOPE': 19.0,
    'LYSITHEA': 18.0, 'CARME': 23.0, 'ANANKE': 14.0, 'LEDA': 10.0,
    'THEMISTO': 4.0,
    # Uranus system (regular satellites and inner moons).
    'CORDELIA': 20.1, 'OPHELIA': 21.4, 'BIANCA': 27.0, 'CRESSIDA': 41.0,
    'DESDEMONA': 32.0, 'JULIET': 47.0, 'PORTIA': 67.6, 'ROSALIND': 36.0,
    'BELINDA': 40.0, 'PUCK': 81.0,
    'MIRANDA': 235.8, 'ARIEL': 578.9, 'UMBRIEL': 584.7,
    'TITANIA': 788.4, 'OBERON': 761.4,
    # Neptune system.
    'NAIAD': 33.0, 'THALASSA': 41.0, 'DESPINA': 75.0, 'GALATEA': 79.0,
    'LARISSA': 97.0, 'PROTEUS': 210.0, 'TRITON': 1353.4, 'NEREID': 170.0,
    # Pluto system.
    'PLUTO': 1188.3, 'CHARON': 606.0, 'NIX': 24.5, 'HYDRA': 21.0,
    'KERBEROS': 6.0, 'STYX': 8.5,
}

# File-spec suffixes added by some PDS3 indexes after the image base
# name; stripping them lets us match an index row to a moon-summary row
# even when the indexes describe different product types of the same
# observation (Voyager has CALIB / RAW / CLEANED / GEOMA / etc.).
PRODUCT_TYPE_SUFFIXES = (
    '_RAW', '_CALIB', '_CLEANED', '_GEOMA', '_GEOMED', '_RESLOC', '_FAR',
    '_UNCALIB',
)

# Voyager INSTRUMENT_NAME often includes wavelength qualifiers; we
# normalize those to one of the recognized names by trimming.
NULL_VAL = -999.0  # PDS3 NULL_CONSTANT for the moon-summary numeric cols.


# ---------------------------------------------------------------------------
# Reading PDS3 tables
# ---------------------------------------------------------------------------

def expand_paths(patterns: Iterable[str]) -> list[str]:
    """Expand one or more glob patterns / paths into a sorted file list."""

    files: list[str] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if not matches and os.path.isfile(pat):
            matches = [pat]
        if not matches:
            print(f'  warning: no files matched "{pat}"', file=sys.stderr)
        files.extend(matches)
    # Preserve order but drop duplicates.
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def read_pds3_table(label_path: str,
                    columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Read a PDS3 ASCII fixed-length table into a pandas DataFrame.

    Only the requested columns (if any) are loaded. Tuple-valued columns
    (e.g. Cassini FILTER_NAME) are joined with commas so they can live
    in a DataFrame cell.
    """

    table = pdstable.PdsTable(label_path, columns=columns)
    data: dict[str, list] = {}
    for name, values in table.column_values.items():
        clean = []
        for v in values:
            if isinstance(v, (tuple, list, np.ndarray)):
                clean.append(','.join(str(x).strip() for x in v))
            elif isinstance(v, (bytes, bytearray)):
                clean.append(v.decode('ascii', errors='replace').strip())
            elif isinstance(v, str):
                clean.append(v.strip())
            else:
                clean.append(v)
        data[name] = clean
    return pd.DataFrame(data)


def base_stem(file_spec: str) -> str:
    """Return a canonical image stem from a PDS3 FILE_SPECIFICATION_NAME.

    Strips any directory, extension, and known product-type suffix so
    that rows in moon-summary and index tables describing the same image
    map to the same key.
    """

    base = os.path.basename(file_spec.strip())
    base, _ = os.path.splitext(base)
    upper = base.upper()
    for suf in PRODUCT_TYPE_SUFFIXES:
        if upper.endswith(suf):
            base = base[: -len(suf)]
            break
    return base.upper()


# ---------------------------------------------------------------------------
# Loading a "set" (mission/instrument) -> single DataFrame
# ---------------------------------------------------------------------------

# Columns we want from the moon_summary tables. Those marked with
# min/max are summarized into mean values for the matching step.
MOON_SUMMARY_COLUMNS = [
    'VOLUME_ID',
    'FILE_SPECIFICATION_NAME',
    'OPUS_ID',
    'TARGET_NAME',
    'MINIMUM_PHASE_ANGLE', 'MAXIMUM_PHASE_ANGLE',
    'MINIMUM_INCIDENCE_ANGLE', 'MAXIMUM_INCIDENCE_ANGLE',
    'MINIMUM_EMISSION_ANGLE', 'MAXIMUM_EMISSION_ANGLE',
    'CENTER_PHASE_ANGLE',
    'CENTER_RESOLUTION',
    'CENTER_DISTANCE',
    'SUB_SOLAR_PLANETOCENTRIC_LATITUDE',
    'SUB_OBSERVER_PLANETOCENTRIC_LATITUDE',
    'SUB_SOLAR_IAU_LONGITUDE',
    'SUB_OBSERVER_IAU_LONGITUDE',
]

# Index columns we attempt to read; not every index has every column.
INDEX_COLUMN_CANDIDATES = [
    # Volume / file identifiers (named differently in COISS vs VGISS).
    'VOLUME_ID', 'VOLUME_NAME',
    'FILE_SPECIFICATION_NAME',
    'PRODUCT_ID', 'PRODUCT_TYPE',
    # Filter / instrument metadata.
    'FILTER_NAME', 'FILTER_NUMBER',
    'INSTRUMENT_ID', 'INSTRUMENT_NAME', 'INSTRUMENT_HOST_NAME',
    'EXPOSURE_DURATION',
    'IMAGE_TIME', 'START_TIME', 'IMAGE_MID_TIME',
]


def load_index_files(index_paths: list[str]) -> pd.DataFrame:
    """Load and concatenate one or more PDS3 *_index.lbl files."""

    frames = []
    for p in index_paths:
        try:
            tbl = pdstable.PdsTable(p)
        except Exception as exc:
            print(f'  warning: failed to read {p}: {exc}', file=sys.stderr)
            continue
        present = [c for c in INDEX_COLUMN_CANDIDATES
                   if c in tbl.column_values]
        df = read_pds3_table(p, columns=present)
        # Normalize VOLUME_ID column.
        if 'VOLUME_ID' not in df.columns and 'VOLUME_NAME' in df.columns:
            df = df.rename(columns={'VOLUME_NAME': 'VOLUME_ID'})
        frames.append(df)
        print(f'  read {len(df):>7d} rows from {os.path.basename(p)}')

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_moon_summary_files(moon_paths: list[str]) -> pd.DataFrame:
    """Load and concatenate one or more PDS3 *_moon_summary.lbl files."""

    frames = []
    for p in moon_paths:
        try:
            tbl = pdstable.PdsTable(p)
        except Exception as exc:
            print(f'  warning: failed to read {p}: {exc}', file=sys.stderr)
            continue
        present = [c for c in MOON_SUMMARY_COLUMNS
                   if c in tbl.column_values]
        df = read_pds3_table(p, columns=present)
        frames.append(df)
        print(f'  read {len(df):>7d} rows from {os.path.basename(p)}')

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def is_clear_filter(filter_name: str) -> bool:
    """True if the filter_name string represents a clear/broadband filter.

    Cassini ISS reports a tuple of two filter wheels (e.g. "CL1,CL2"),
    so the clear case is when both wheels are CLx. Voyager ISS uses the
    string "CLEAR".
    """

    if not isinstance(filter_name, str):
        return False
    s = filter_name.strip().upper()
    if s == 'CLEAR':
        return True
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if not parts:
        return False
    # Cassini-style: both filter wheels must be a clear ("CLn") slot.
    return all(re.fullmatch(r'CL\d+', p) for p in parts)


def load_set(name: str,
             moon_patterns: list[str],
             index_patterns: list[str]) -> pd.DataFrame:
    """Load a single set's moon-summary + index data as a merged DataFrame.

    The returned DataFrame has one row per (image, target body) with
    the geometry columns from moon_summary and the FILTER_NAME /
    instrument columns from the matching image-index row.
    """

    print(f'\nLoading set "{name}"')
    print(' moon_summary files:')
    moon_paths = expand_paths(moon_patterns)
    print(' index files:')
    index_paths = expand_paths(index_patterns)
    if not moon_paths or not index_paths:
        print(f'  ERROR: set "{name}" has no usable files', file=sys.stderr)
        sys.exit(1)

    print(' reading moon_summary tables...')
    moon = load_moon_summary_files(moon_paths)
    print(' reading index tables...')
    idx = load_index_files(index_paths)
    if moon.empty or idx.empty:
        return pd.DataFrame()

    # Build matching key (volume + image stem) on both sides.
    moon['_key_stem'] = moon['FILE_SPECIFICATION_NAME'].map(base_stem)
    idx['_key_stem'] = idx['FILE_SPECIFICATION_NAME'].map(base_stem)
    moon['_key_vol'] = moon['VOLUME_ID'].str.strip().str.upper()
    idx['_key_vol'] = idx['VOLUME_ID'].str.strip().str.upper()

    # Cumulative-index volumes (e.g. COISS_2999, VGISS_6999) repeat
    # rows from per-volume files. Drop those duplicates so each image
    # contributes once per body.
    n_moon_pre = len(moon)
    moon = moon.drop_duplicates(
        subset=['_key_vol', '_key_stem', 'TARGET_NAME'])
    if len(moon) < n_moon_pre:
        print(f'  removed {n_moon_pre - len(moon):>7d} duplicate '
              f'moon-summary rows')

    # An index file may have multiple rows per image (e.g. Voyager has
    # one row per product type). Keep only one row per (volume, stem)
    # by ranking PRODUCT_TYPE so calibrated > raw > geometric > tables.
    if 'PRODUCT_TYPE' in idx.columns:
        product_rank = {
            'CALIBRATED_IMAGE': 0,
            'CLEANED_IMAGE': 1,
            'DECOMPRESSED_RAW_IMAGE': 2,
            'GEOMETRICALLY_CORRECTED_IMAGE': 3,
            'RAW_IMAGE': 4,
            'EDR': 0,
        }
        idx['_rank'] = idx['PRODUCT_TYPE'].map(
            lambda v: product_rank.get(str(v).strip().upper(), 99))
        idx = (idx.sort_values('_rank')
                  .drop_duplicates(subset=['_key_vol', '_key_stem'])
                  .drop(columns='_rank'))
    else:
        idx = idx.drop_duplicates(subset=['_key_vol', '_key_stem'])

    # Carry only the columns we care about into the merge.
    keep_idx = ['_key_vol', '_key_stem']
    for c in ('FILTER_NAME', 'INSTRUMENT_ID', 'INSTRUMENT_NAME',
              'INSTRUMENT_HOST_NAME', 'EXPOSURE_DURATION', 'IMAGE_TIME',
              'IMAGE_MID_TIME', 'START_TIME', 'PRODUCT_ID',
              'FILE_SPECIFICATION_NAME'):
        if c in idx.columns:
            keep_idx.append(c)
    idx_sub = idx[keep_idx].rename(
        columns={'FILE_SPECIFICATION_NAME': 'INDEX_FILE_SPECIFICATION_NAME'})

    merged = moon.merge(idx_sub, on=['_key_vol', '_key_stem'], how='inner')
    print(f'  matched {len(merged):>7d} (moon-summary, index) rows')

    merged['SET_NAME'] = name
    return merged


# ---------------------------------------------------------------------------
# Filtering and matching
# ---------------------------------------------------------------------------

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add MEAN_INCIDENCE_ANGLE, MEAN_EMISSION_ANGLE, FOV_FRACTION, etc.

    Rows with NULL geometry, missing instrument size, or unknown body
    radius end up with NaNs and will be filtered out downstream.
    """

    df = df.copy()
    df['TARGET_NAME'] = df['TARGET_NAME'].str.strip().str.upper()
    df['FILTER_NAME'] = df.get('FILTER_NAME', '').astype(str).str.strip()

    # Replace PDS NULL placeholders with NaN for numeric columns.
    numeric_cols = [
        'MINIMUM_PHASE_ANGLE', 'MAXIMUM_PHASE_ANGLE',
        'MINIMUM_INCIDENCE_ANGLE', 'MAXIMUM_INCIDENCE_ANGLE',
        'MINIMUM_EMISSION_ANGLE', 'MAXIMUM_EMISSION_ANGLE',
        'CENTER_PHASE_ANGLE', 'CENTER_RESOLUTION', 'CENTER_DISTANCE',
        'SUB_SOLAR_PLANETOCENTRIC_LATITUDE',
        'SUB_OBSERVER_PLANETOCENTRIC_LATITUDE',
        'SUB_SOLAR_IAU_LONGITUDE', 'SUB_OBSERVER_IAU_LONGITUDE',
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df.loc[df[c] == NULL_VAL, c] = np.nan

    df['MEAN_PHASE_ANGLE'] = df.get(
        'CENTER_PHASE_ANGLE',
        0.5 * (df['MINIMUM_PHASE_ANGLE'] + df['MAXIMUM_PHASE_ANGLE']))
    df['MEAN_INCIDENCE_ANGLE'] = 0.5 * (
        df['MINIMUM_INCIDENCE_ANGLE'] + df['MAXIMUM_INCIDENCE_ANGLE'])
    df['MEAN_EMISSION_ANGLE'] = 0.5 * (
        df['MINIMUM_EMISSION_ANGLE'] + df['MAXIMUM_EMISSION_ANGLE'])

    # Body radius (km) from lookup table.
    df['BODY_RADIUS_KM'] = df['TARGET_NAME'].map(BODY_MEAN_RADIUS_KM)

    # Image size (pixels) from instrument identifier. Prefer the
    # short ID (Cassini); fall back to instrument name (Voyager).
    def _image_size(row):
        for col in ('INSTRUMENT_ID', 'INSTRUMENT_NAME'):
            if col in row and isinstance(row[col], str):
                v = row[col].strip().upper()
                if v in INSTRUMENT_IMAGE_SIZE_PX:
                    return INSTRUMENT_IMAGE_SIZE_PX[v]
        return np.nan

    df['IMAGE_SIZE_PX'] = df.apply(_image_size, axis=1)

    def _camera_family(row):
        for col in ('INSTRUMENT_ID', 'INSTRUMENT_NAME'):
            if col in row and isinstance(row[col], str):
                v = row[col].strip().upper()
                if v in CAMERA_FAMILY:
                    return CAMERA_FAMILY[v]
        return ''

    df['CAMERA_FAMILY'] = df.apply(_camera_family, axis=1)

    # Fraction of the FOV that the body diameter spans along one axis.
    fov_extent_km = df['IMAGE_SIZE_PX'] * df['CENTER_RESOLUTION']
    df['FOV_FRACTION'] = (2.0 * df['BODY_RADIUS_KM']) / fov_extent_km

    return df


def filter_set(df: pd.DataFrame, *,
               min_fov_fraction: float,
               max_fov_fraction: float = 1.0,
               clear_only: bool = True,
               camera: Optional[str] = None,
               targets: Optional[set[str]] = None) -> pd.DataFrame:
    """Apply per-image quality filters to a set DataFrame.

    ``camera`` is one of ``'NAC'``, ``'WAC'``, or ``None`` (no camera-family
    restriction). The mapping uses the ``CAMERA_FAMILY`` lookup, which
    handles both Cassini INSTRUMENT_ID ('ISSNA' / 'ISSWA') and Voyager
    INSTRUMENT_NAME ('NARROW ANGLE CAMERA' / 'WIDE ANGLE CAMERA').
    """

    n0 = len(df)
    mask = pd.Series(True, index=df.index)

    if clear_only:
        mask &= df['FILTER_NAME'].map(is_clear_filter)
    n_filt = mask.sum()

    if camera:
        mask &= df['CAMERA_FAMILY'] == camera.upper()
    n_cam = mask.sum()

    needed = ['MEAN_PHASE_ANGLE', 'MEAN_INCIDENCE_ANGLE',
              'MEAN_EMISSION_ANGLE', 'CENTER_RESOLUTION',
              'BODY_RADIUS_KM', 'IMAGE_SIZE_PX',
              'SUB_SOLAR_PLANETOCENTRIC_LATITUDE',
              'SUB_OBSERVER_PLANETOCENTRIC_LATITUDE',
              'SUB_SOLAR_IAU_LONGITUDE',
              'SUB_OBSERVER_IAU_LONGITUDE']
    for c in needed:
        mask &= df[c].notna()
    n_geom = mask.sum()

    # Body must not fully envelope the frame and must fill enough of it.
    mask &= df['FOV_FRACTION'] >= min_fov_fraction
    mask &= df['FOV_FRACTION'] <= max_fov_fraction
    n_fov = mask.sum()

    # Day side only (incidence < 90).
    mask &= df['MEAN_INCIDENCE_ANGLE'] < 90.0
    mask &= df['MEAN_EMISSION_ANGLE'] < 90.0
    n_day = mask.sum()

    if targets is not None:
        mask &= df['TARGET_NAME'].isin(targets)

    out = df.loc[mask].copy()
    print(f'  {n0:>7d} rows total')
    print(f'  {n_filt:>7d} after clear-filter')
    if camera:
        print(f'  {n_cam:>7d} after camera={camera.upper()}')
    print(f'  {n_geom:>7d} after geometry/instrument completeness')
    print(f'  {n_fov:>7d} after FOV-fraction in '
          f'[{min_fov_fraction:.3f}, {max_fov_fraction:.3f}]')
    print(f'  {n_day:>7d} after day-side cut')
    print(f'  {len(out):>7d} after target restriction')
    return out


def angular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest unsigned angular separation (deg) between a and b.

    Operates element-wise; both inputs may be broadcastable arrays.
    Returned values lie in [0, 180].
    """

    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def find_matches(df1: pd.DataFrame, df2: pd.DataFrame, *,
                 max_phase_diff: float,
                 max_incidence_diff: float,
                 max_emission_diff: float,
                 max_sub_solar_lat_diff: float,
                 max_sub_obs_lat_diff: float,
                 max_sub_solar_lon_diff: float,
                 max_sub_obs_lon_diff: float,
                 phase_weight: float = 1.0,
                 incidence_weight: float = 1.0,
                 emission_weight: float = 1.0,
                 sub_solar_lat_weight: float = 1.0,
                 sub_obs_lat_weight: float = 1.0,
                 sub_solar_lon_weight: float = 1.0,
                 sub_obs_lon_weight: float = 1.0) -> pd.DataFrame:
    """Compute the cross-set pairs that satisfy the geometry tolerances.

    Matching is done in seven dimensions:

      * mean phase angle (Sun-body-observer geometry),
      * mean incidence and emission angles (overall illumination/viewing),
      * sub-solar planetocentric latitude (where the Sun stands above
        the body), sub-solar IAU longitude (which face is lit),
      * sub-observer planetocentric latitude (which hemisphere is
        viewed) and sub-observer IAU longitude (which face is viewed).

    Latitude differences are simple absolute values; longitude
    differences use the smallest signed angle on the circle so e.g. 359
    and 1 are 2 deg apart. Pairs are scored by a weighted Euclidean
    distance over those seven axes so the user can rank by overall
    geometric similarity.
    """

    bodies = sorted(set(df1['TARGET_NAME']) & set(df2['TARGET_NAME']))
    print(f'\nMatching across {len(bodies)} common bodies: {", ".join(bodies)}')

    rows = []
    for body in bodies:
        a = df1[df1['TARGET_NAME'] == body]
        b = df2[df2['TARGET_NAME'] == body]
        if a.empty or b.empty:
            continue

        # Cross-product via numpy for speed.
        p1 = a['MEAN_PHASE_ANGLE'].to_numpy()
        i1 = a['MEAN_INCIDENCE_ANGLE'].to_numpy()
        e1 = a['MEAN_EMISSION_ANGLE'].to_numpy()
        ssl1 = a['SUB_SOLAR_PLANETOCENTRIC_LATITUDE'].to_numpy()
        sol1 = a['SUB_OBSERVER_PLANETOCENTRIC_LATITUDE'].to_numpy()
        sslon1 = a['SUB_SOLAR_IAU_LONGITUDE'].to_numpy()
        solon1 = a['SUB_OBSERVER_IAU_LONGITUDE'].to_numpy()

        p2 = b['MEAN_PHASE_ANGLE'].to_numpy()
        i2 = b['MEAN_INCIDENCE_ANGLE'].to_numpy()
        e2 = b['MEAN_EMISSION_ANGLE'].to_numpy()
        ssl2 = b['SUB_SOLAR_PLANETOCENTRIC_LATITUDE'].to_numpy()
        sol2 = b['SUB_OBSERVER_PLANETOCENTRIC_LATITUDE'].to_numpy()
        sslon2 = b['SUB_SOLAR_IAU_LONGITUDE'].to_numpy()
        solon2 = b['SUB_OBSERVER_IAU_LONGITUDE'].to_numpy()

        dp = np.abs(p1[:, None] - p2[None, :])
        di = np.abs(i1[:, None] - i2[None, :])
        de = np.abs(e1[:, None] - e2[None, :])
        dsl = np.abs(ssl1[:, None] - ssl2[None, :])
        dol = np.abs(sol1[:, None] - sol2[None, :])
        dslon = angular_diff_deg(sslon1[:, None], sslon2[None, :])
        dolon = angular_diff_deg(solon1[:, None], solon2[None, :])

        ok = ((dp <= max_phase_diff)
              & (di <= max_incidence_diff)
              & (de <= max_emission_diff)
              & (dsl <= max_sub_solar_lat_diff)
              & (dol <= max_sub_obs_lat_diff)
              & (dslon <= max_sub_solar_lon_diff)
              & (dolon <= max_sub_obs_lon_diff))
        if not ok.any():
            print(f'  {body:<12s} {len(a):>5d} x {len(b):>5d} -> {0:>6d} pairs')
            continue
        ii, jj = np.where(ok)
        score = np.sqrt(
            (phase_weight * dp[ii, jj]) ** 2
            + (incidence_weight * di[ii, jj]) ** 2
            + (emission_weight * de[ii, jj]) ** 2
            + (sub_solar_lat_weight * dsl[ii, jj]) ** 2
            + (sub_obs_lat_weight * dol[ii, jj]) ** 2
            + (sub_solar_lon_weight * dslon[ii, jj]) ** 2
            + (sub_obs_lon_weight * dolon[ii, jj]) ** 2)

        a_rec = a.iloc[ii].reset_index(drop=True)
        b_rec = b.iloc[jj].reset_index(drop=True)
        pair = pd.DataFrame({
            'TARGET_NAME': body,
            'set1_name': a_rec['SET_NAME'],
            'set1_volume': a_rec['VOLUME_ID'],
            'set1_file': a_rec['FILE_SPECIFICATION_NAME'],
            'set1_opus_id': a_rec.get(
                'OPUS_ID', pd.Series([''] * len(a_rec))),
            'set1_filter': a_rec['FILTER_NAME'],
            'set1_instrument': a_rec.get(
                'INSTRUMENT_ID',
                a_rec.get('INSTRUMENT_NAME',
                          pd.Series([''] * len(a_rec)))),
            'set1_phase': p1[ii],
            'set1_incidence': i1[ii],
            'set1_emission': e1[ii],
            'set1_sub_solar_lat': ssl1[ii],
            'set1_sub_obs_lat': sol1[ii],
            'set1_sub_solar_lon': sslon1[ii],
            'set1_sub_obs_lon': solon1[ii],
            'set1_resolution_km_px': a_rec['CENTER_RESOLUTION'].to_numpy(),
            'set1_fov_fraction': a_rec['FOV_FRACTION'].to_numpy(),
            'set2_name': b_rec['SET_NAME'],
            'set2_volume': b_rec['VOLUME_ID'],
            'set2_file': b_rec['FILE_SPECIFICATION_NAME'],
            'set2_opus_id': b_rec.get(
                'OPUS_ID', pd.Series([''] * len(b_rec))),
            'set2_filter': b_rec['FILTER_NAME'],
            'set2_instrument': b_rec.get(
                'INSTRUMENT_ID',
                b_rec.get('INSTRUMENT_NAME',
                          pd.Series([''] * len(b_rec)))),
            'set2_phase': p2[jj],
            'set2_incidence': i2[jj],
            'set2_emission': e2[jj],
            'set2_sub_solar_lat': ssl2[jj],
            'set2_sub_obs_lat': sol2[jj],
            'set2_sub_solar_lon': sslon2[jj],
            'set2_sub_obs_lon': solon2[jj],
            'set2_resolution_km_px': b_rec['CENTER_RESOLUTION'].to_numpy(),
            'set2_fov_fraction': b_rec['FOV_FRACTION'].to_numpy(),
            'd_phase': dp[ii, jj],
            'd_incidence': di[ii, jj],
            'd_emission': de[ii, jj],
            'd_sub_solar_lat': dsl[ii, jj],
            'd_sub_obs_lat': dol[ii, jj],
            'd_sub_solar_lon': dslon[ii, jj],
            'd_sub_obs_lon': dolon[ii, jj],
            'score': score,
        })
        rows.append(pair)
        print(f'  {body:<12s} {len(a):>5d} x {len(b):>5d} -> {len(pair):>6d} pairs')

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(['TARGET_NAME', 'score']).reset_index(drop=True)

    # Move the most useful columns to the front so the table is easy to
    # scan; keep the remaining columns in their existing order.
    lead = ['TARGET_NAME', 'score', 'set1_opus_id', 'set2_opus_id']
    d_cols = [c for c in out.columns if c.startswith('d_')]
    rest = [c for c in out.columns if c not in lead and c not in d_cols]
    out = out[lead + d_cols + rest]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Find cross-mission image pairs with matched geometry.')
    p.add_argument('--set1-name', default='set1',
                   help='Display name for the first dataset.')
    p.add_argument('--set1-moon', nargs='+', required=True,
                   help='moon_summary.lbl file(s)/glob(s) for set 1.')
    p.add_argument('--set1-index', nargs='+', required=True,
                   help='index.lbl file(s)/glob(s) for set 1.')
    p.add_argument('--set1-camera', choices=CAMERA_CHOICES, default='ANY',
                   help='Restrict set 1 to NAC, WAC, or ANY (default ANY). '
                        'Allows feeding the same index files into both '
                        'sets and comparing e.g. COISS NAC vs COISS WAC.')
    p.add_argument('--set2-name', default='set2',
                   help='Display name for the second dataset.')
    p.add_argument('--set2-moon', nargs='+', required=True,
                   help='moon_summary.lbl file(s)/glob(s) for set 2.')
    p.add_argument('--set2-index', nargs='+', required=True,
                   help='index.lbl file(s)/glob(s) for set 2.')
    p.add_argument('--set2-camera', choices=CAMERA_CHOICES, default='ANY',
                   help='Restrict set 2 to NAC, WAC, or ANY (default ANY).')

    p.add_argument('--min-fov-fraction', type=float, default=0.05,
                   help='Minimum fraction of the image width that the '
                        'body must span (default 0.05).')
    p.add_argument('--max-fov-fraction', type=float, default=1.0,
                   help='Maximum fraction (1.0 = full disk; >1 means '
                        'limb is outside the frame).')
    p.add_argument('--max-phase-diff', type=float, default=5.0,
                   help='Maximum allowed |Delta phase| (deg).')
    p.add_argument('--max-incidence-diff', type=float, default=5.0,
                   help='Maximum allowed |Delta incidence| (deg).')
    p.add_argument('--max-emission-diff', type=float, default=5.0,
                   help='Maximum allowed |Delta emission| (deg).')
    p.add_argument('--max-sub-solar-lat-diff', type=float, default=5.0,
                   help='Maximum allowed |Delta sub-solar latitude| (deg). '
                        'Controls how closely the Sun is at the same '
                        'planetocentric latitude in both images so '
                        'shadow direction matches.')
    p.add_argument('--max-sub-obs-lat-diff', type=float, default=5.0,
                   help='Maximum allowed |Delta sub-observer latitude| '
                        '(deg). Controls how closely the same body '
                        'hemisphere is being viewed.')
    p.add_argument('--max-sub-solar-lon-diff', type=float, default=5.0,
                   help='Maximum allowed angular |Delta sub-solar IAU '
                        'longitude| (deg, with 0/360 wrap). Controls '
                        'which longitude on the body is illuminated.')
    p.add_argument('--max-sub-obs-lon-diff', type=float, default=5.0,
                   help='Maximum allowed angular |Delta sub-observer '
                        'IAU longitude| (deg, with 0/360 wrap). For '
                        'tidally locked moons this is equivalent to '
                        'matching orbital phase / leading-vs-trailing '
                        'face.')

    p.add_argument('--phase-weight', type=float, default=1.0,
                   help='Score weight for phase-angle difference.')
    p.add_argument('--incidence-weight', type=float, default=1.0,
                   help='Score weight for incidence-angle difference.')
    p.add_argument('--emission-weight', type=float, default=1.0,
                   help='Score weight for emission-angle difference.')
    p.add_argument('--sub-solar-lat-weight', type=float, default=1.0,
                   help='Score weight for sub-solar latitude difference.')
    p.add_argument('--sub-obs-lat-weight', type=float, default=1.0,
                   help='Score weight for sub-observer latitude difference.')
    p.add_argument('--sub-solar-lon-weight', type=float, default=1.0,
                   help='Score weight for sub-solar longitude difference.')
    p.add_argument('--sub-obs-lon-weight', type=float, default=1.0,
                   help='Score weight for sub-observer longitude difference.')

    p.add_argument('--targets', nargs='+', default=None,
                   help='Restrict matching to these TARGET_NAME values '
                        '(case-insensitive). Default: all common bodies.')
    p.add_argument('--include-non-clear', action='store_true',
                   help='Skip the clear-filter restriction.')

    p.add_argument('--output', '-o', default=None,
                   help='Write all matches to this CSV path.')
    p.add_argument('--top', type=int, default=20,
                   help='Number of best matches per body to print '
                        '(default 20). Use 0 to suppress per-body output.')
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    set1 = load_set(args.set1_name, args.set1_moon, args.set1_index)
    set2 = load_set(args.set2_name, args.set2_moon, args.set2_index)
    if set1.empty or set2.empty:
        print('At least one set produced no rows; nothing to do.',
              file=sys.stderr)
        return 1

    set1 = add_derived_columns(set1)
    set2 = add_derived_columns(set2)

    targets = None
    if args.targets:
        targets = {t.strip().upper() for t in args.targets}

    set1_camera = args.set1_camera if args.set1_camera != 'ANY' else None
    set2_camera = args.set2_camera if args.set2_camera != 'ANY' else None

    print(f'\nFiltering set "{args.set1_name}":')
    set1f = filter_set(set1,
                       min_fov_fraction=args.min_fov_fraction,
                       max_fov_fraction=args.max_fov_fraction,
                       clear_only=not args.include_non_clear,
                       camera=set1_camera,
                       targets=targets)
    print(f'\nFiltering set "{args.set2_name}":')
    set2f = filter_set(set2,
                       min_fov_fraction=args.min_fov_fraction,
                       max_fov_fraction=args.max_fov_fraction,
                       clear_only=not args.include_non_clear,
                       camera=set2_camera,
                       targets=targets)

    matches = find_matches(
        set1f, set2f,
        max_phase_diff=args.max_phase_diff,
        max_incidence_diff=args.max_incidence_diff,
        max_emission_diff=args.max_emission_diff,
        max_sub_solar_lat_diff=args.max_sub_solar_lat_diff,
        max_sub_obs_lat_diff=args.max_sub_obs_lat_diff,
        max_sub_solar_lon_diff=args.max_sub_solar_lon_diff,
        max_sub_obs_lon_diff=args.max_sub_obs_lon_diff,
        phase_weight=args.phase_weight,
        incidence_weight=args.incidence_weight,
        emission_weight=args.emission_weight,
        sub_solar_lat_weight=args.sub_solar_lat_weight,
        sub_obs_lat_weight=args.sub_obs_lat_weight,
        sub_solar_lon_weight=args.sub_solar_lon_weight,
        sub_obs_lon_weight=args.sub_obs_lon_weight)

    if matches.empty:
        print('\nNo image pairs satisfy the requested constraints.')
        return 0

    print(f'\nFound {len(matches)} candidate pairs across '
          f'{matches["TARGET_NAME"].nunique()} bodies.')

    if args.top > 0:
        for body, group in matches.groupby('TARGET_NAME', sort=True):
            print(f'\n=== {body}: top {min(args.top, len(group))} '
                  f'of {len(group)} pairs ===')
            with pd.option_context('display.max_columns', None,
                                   'display.width', 200,
                                   'display.float_format',
                                   lambda v: f'{v:8.3f}'):
                print(group.head(args.top).to_string(index=False))

    if args.output:
        matches.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f'\nWrote {len(matches)} matches to {args.output}')
        write_unique_filename_lists(matches, args.output)

    return 0


def write_unique_filename_lists(matches: pd.DataFrame,
                                csv_path: str) -> None:
    """Write per-instrument lists of unique base filenames.

    For each set, derives an instrument tag (e.g. "coiss", "vgiss")
    from the volume IDs in the matches and writes one file containing
    the lowercase, alphabetically sorted unique image base names that
    appear in that set's column of the CSV. The output files are named
    "<csv-base>_<tag>.txt" alongside the CSV. If both sets share the
    same instrument tag, the files are suffixed with the set role
    instead so they don't collide.
    """

    csv_root, _ = os.path.splitext(csv_path)

    def _tag(volumes: pd.Series, fallback: str) -> str:
        prefixes = (volumes.astype(str).str.split('_').str[0]
                    .str.strip().str.lower())
        prefixes = prefixes[prefixes != '']
        if prefixes.empty:
            return fallback
        return prefixes.mode().iat[0]

    tag1 = _tag(matches['set1_volume'], 'set1')
    tag2 = _tag(matches['set2_volume'], 'set2')
    if tag1 == tag2:
        tag1, tag2 = 'set1', 'set2'

    for tag, file_col in [(tag1, 'set1_file'), (tag2, 'set2_file')]:
        # Strip any trailing Cassini-style version number (e.g. the
        # "_1" in N1519182285_1) so that different versions of the
        # same image collapse to a single base filename.
        stems = sorted({
            re.sub(r'_\d+$', '', base_stem(s)).lower()
            for s in matches[file_col] if s
        })
        out_path = f'{csv_root}_{tag}.txt'
        with open(out_path, 'w') as f:
            for stem in stems:
                f.write(stem + '\n')
        print(f'Wrote {len(stems):>5d} unique {tag} filenames to {out_path}')


if __name__ == '__main__':
    sys.exit(main())
