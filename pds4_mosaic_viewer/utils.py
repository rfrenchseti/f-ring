"""General utilities for the mosaic viewer."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import numpy.ma as ma

# J2000 epoch in TDB
_J2000 = datetime(2000, 1, 1, 12, 0, 0)


def tdb_to_utc_str(tdb_seconds: float) -> str:
    """Convert TDB seconds past J2000 to a UTC datetime string.

    The TDB-UTC difference (~69 s) is negligible for display.
    """
    dt = _J2000 + timedelta(seconds=float(tdb_seconds))
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def build_full_width_metadata(
    metadata_params: np.ndarray,
    n_long: int,
    long_interval: float,
) -> dict[str, ma.MaskedArray]:
    """Expand sparse per-longitude metadata into full-width masked arrays.

    The metadata_params table has one row per valid longitude and must define
    every field listed in ``float_fields`` and ``int_fields``. Returns a dict
    mapping field name -> masked array of shape (n_long,), with valid values
    only at columns that have data.
    """
    table_corot = metadata_params['rings_corotating_ring_longitude']
    col_idx = np.intp(np.round(table_corot / long_interval))
    col_idx = np.clip(col_idx, 0, n_long - 1)

    float_fields = [
        'rings_corotating_ring_longitude',
        'rings_observed_event_tdb',
        'rings_inertial_ring_longitude',
        'rings_radial_resolution',
        'rings_longitudinal_resolution',
        'rings_incidence_angle',
        'rings_phase_angle',
        'rings_emission_angle',
        'core_radius',
        'longitude_ascending_node',
        'longitude_pericenter',
        'true_anomaly',
        'corotating_longitude_prometheus',
        'radius_prometheus',
        'corotating_longitude_pandora',
        'radius_pandora',
    ]
    # Mosaics include per-column source image index; reprojected-image metadata does not.
    names = metadata_params.dtype.names
    if names is None:
        names = ()
    int_fields = ['image_index'] if 'image_index' in names else []

    result: dict[str, ma.MaskedArray] = {}
    for field in float_fields:
        arr = ma.masked_all(n_long, dtype=np.float64)
        arr[col_idx] = metadata_params[field].astype(np.float64)
        result[field] = arr

    for field in int_fields:
        arr = ma.masked_all(n_long, dtype=np.intp)
        arr[col_idx] = metadata_params[field].astype(np.intp)
        result[field] = arr

    if 'image_index' not in result:
        result['image_index'] = ma.masked_all(n_long, dtype=np.intp)

    return result


def compute_ew(image_ma: ma.MaskedArray, radial_interval: float) -> ma.MaskedArray:
    """Compute per-column equivalent width (EW) in km.

    EW = sum over radii of I/F * radial_interval.
    """
    return ma.sum(image_ma, axis=0) * radial_interval


def compute_ewmu(
    ew: ma.MaskedArray,
    emission_deg: ma.MaskedArray,
) -> ma.MaskedArray:
    """Compute EW * |cos(emission)| (viewing-angle-corrected EW)."""
    mu = np.abs(np.cos(np.radians(emission_deg.filled(0.0))))
    result = ew * mu
    combined_mask = ma.getmaskarray(ew) | ma.getmaskarray(emission_deg)
    return ma.array(result, mask=combined_mask)


def show_radii_to_pixel_ys(
    radii_rel_km: list[float],
    n_radii: int,
    radial_interval: float,
) -> list[int]:
    """Convert radial offsets from the local core (km) to display pixel Y values.

    The radial grid is core-following: offsets are measured from the F ring
    core at each longitude, not from the mean core radius.

    In display coordinates, pixel_y=0 is outer (top) and pixel_y=n_radii-1
    is inner (bottom).  Out-of-range values are omitted.
    """
    if n_radii < 1 or not radii_rel_km:
        return []
    rel = np.asarray(radii_rel_km, dtype=np.float64)
    arr_row = (n_radii - 1) / 2.0 + rel / radial_interval
    pix_y = (n_radii - 1) - np.round(arr_row).astype(np.intp)
    mask = (pix_y >= 0) & (pix_y < n_radii)
    return pix_y[mask].tolist()


def _hsv1_to_rgb(hue_arr: np.ndarray) -> np.ndarray:
    """Vectorised HSV (s=1, v=1) → RGB.  Returns float32 array (n, 3)."""
    n = len(hue_arr)
    h6 = (hue_arr % 1.0) * 6.0
    sector = np.floor(h6).astype(int) % 6
    f = (h6 - np.floor(h6)).astype(np.float32)

    r = np.empty(n, dtype=np.float32)
    g = np.empty(n, dtype=np.float32)
    b = np.empty(n, dtype=np.float32)

    for s, (rv, gv, bv) in enumerate([
        ('1', 'f', '0'),
        ('q', '1', '0'),
        ('0', '1', 'f'),
        ('0', 'q', '1'),
        ('f', '0', '1'),
        ('1', '0', 'q'),
    ]):
        m = sector == s
        for arr, code in [(r, rv), (g, gv), (b, bv)]:
            if code == '1':
                arr[m] = 1.0
            elif code == '0':
                arr[m] = 0.0
            elif code == 'f':
                arr[m] = f[m]
            else:  # 'q' = 1 - f
                arr[m] = 1.0 - f[m]

    return np.stack([r, g, b], axis=1)


def compute_color_column(
    values_ma: ma.MaskedArray,
    minval: float,
    maxval: float,
) -> np.ndarray:
    """Compute per-column RGB tinting array (n, 3) float32 in [0, 1].

    Uses the same HSV colour scale as the original display_mosaic.py:
    low values → blue, high values → red.
    """
    n = len(values_ma)
    color_data = np.zeros((n, 3), dtype=np.float32)
    valid = ~ma.getmaskarray(values_ma)
    if not np.any(valid):
        return color_data

    vf = values_ma.filled(float(minval)).astype(np.float64)
    if minval == maxval:
        norm = np.ones(n, dtype=np.float32)
    else:
        norm = np.clip((vf - minval) / (maxval - minval), 0.0, 1.0).astype(np.float32)

    hue = ((1.0 - norm) * 0.66).astype(np.float32)
    rgb = _hsv1_to_rgb(hue)
    color_data[valid] = rgb[valid]
    return color_data
