"""Check the navigation of reprojected images by looking at the F ring core.

A reprojected image is radius versus co-rotating longitude, with the radial axis
centered on the modeled core radius. If the image is navigated correctly the
bright F ring core sits near a radial offset of zero and runs flat across the
image. Bad navigation shows up as a core that is displaced from the center, or
that slopes or wanders as a function of longitude.

For each image this measures the radial offset of the core in every longitude
column, then reduces those to three numbers:

    offset   the median offset of the core from the center, in km
    rise     how far the core tilts across the whole image, in km
    scatter  the robust scatter of the per-column offsets, in km

An image is flagged when any of the three exceeds its threshold. The defaults
are deliberately loose: the core is genuinely eccentric and wanders tens of km
about the Albers model, so only gross errors should be reported. They were set
against 1,890 images from twelve observations graded 'G' for navigation, whose
offsets run to about 70 km at the 5th/95th percentiles and whose total tilt
reaches 123 km at the 99th; 0.3% of that known-good population is flagged.

The core is located coarsely over the whole radial range first and only then
centroided, so an image whose core lies far outside the search window reports
how far off it really is rather than being clipped to the window edge.

Usage:
    python mosaics/analyze_reproj_navigation.py [OBSID ...] [options]

    # every observation flagged 'O' or 'R' in the observation list
    python mosaics/analyze_reproj_navigation.py --notes OR

    # write a CSV of every image, and diagnostic plots for the flagged ones
    python mosaics/analyze_reproj_navigation.py --notes OR \
        --csv nav.csv --plot-dir nav_plots
"""

import argparse
import csv
import os
import sys
import warnings

import msgpack
import msgpack_numpy
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import f_ring_util.f_ring as f_ring  # noqa: E402


SENTINEL = -999

# The observation list lives with the PDS4 generator; it is the only place the
# 'O' and 'R' classifications are recorded.
OBSERVATION_LIST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'pds4_bundle_gen',
    'observation_list.csv')


def read_observation_notes():
    """Return a dict of obsid -> notes string from the observation list."""
    notes = {}
    with open(OBSERVATION_LIST_PATH, 'r') as fp:
        reader = csv.reader(fp)
        next(reader)
        for row in reader:
            if row:
                notes[row[0]] = row[6]
    return notes


def read_reproj_img(reproj_path):
    """Read a reprojected image file and return its metadata dict."""
    with open(reproj_path, 'rb') as fp:
        return msgpack.unpackb(fp.read(), object_hook=msgpack_numpy.decode)


def img_to_repro_path(arguments, image_path):
    """Convert a calibrated image path to its reprojected image path."""
    components = image_path.split('/')
    vol = components[-4]
    sclk_dir = components[-2]
    image_name = components[-1]
    suffix = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d-REPRO.DAT' % (
        arguments.ring_radius, arguments.radius_inner_delta,
        arguments.radius_outer_delta, arguments.radius_resolution,
        arguments.longitude_resolution, arguments.radial_zoom_amount,
        arguments.longitude_zoom_amount))
    image_name = image_name.replace('_CALIB.IMG', suffix)
    return f_ring.file_clean_join(f_ring.REPRO_DIR, vol, sclk_dir, image_name)


def measure_core_offsets(img, radii, *, search_km, baseline_km, min_snr):
    """Measure the core's radial offset in each longitude column.

    The core is found as the brightest feature within search_km of the center.
    A column is only measured if that feature stands above the noise of the
    baseline region by min_snr.

    Parameters:
        img (np.array): the reprojected image, radius by longitude
        radii (np.array): the radial offset of each row, in km
        search_km (float): only look for the core within this distance of zero
        baseline_km (float): rows beyond this distance define the background
        min_snr (float): the smallest peak-to-noise ratio worth measuring

    Returns:
        tuple: (offsets, snr), each one value per column, NaN where unmeasured
    """
    prof = np.where(img == SENTINEL, np.nan, img).astype(float)

    # The background and its noise come from the parts of the radial range far
    # enough from the core not to contain it.
    outer = np.abs(radii) > baseline_km
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        base = np.nanmedian(prof[outer, :], axis=0)
        noise = np.nanstd(prof[outer, :], axis=0)
    signal = prof - base[None, :]

    # Find the core coarsely first, over the whole radial range, so that a badly
    # navigated image whose core lies outside the search window is measured
    # honestly instead of being clipped to the edge of the window.
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        # A column with no valid data at all is expected; it becomes NaN and is
        # dropped below rather than being worth a warning.
        warnings.simplefilter('ignore', RuntimeWarning)
        full_peak = np.nanmax(signal, axis=0)
        peak_row = np.nanargmax(np.nan_to_num(signal, nan=-np.inf), axis=0)
        snr = np.where(noise > 0, full_peak / noise, np.nan)
    strong = np.isfinite(snr) & (snr >= min_snr)
    coarse = float(np.median(radii[peak_row[strong]])) if strong.any() else 0.

    # Then refine by centroiding within search_km of where the core actually is.
    window = np.abs(radii - coarse) <= search_km
    win_signal = signal[window, :]
    win_radii = radii[window]

    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        peak = np.nanmax(win_signal, axis=0)

    # Centroid the pixels above half the peak. Using the half-peak rather than
    # everything positive keeps the wings of neighboring material out of it.
    half = 0.5 * peak
    weights = np.where(win_signal >= half[None, :], win_signal, 0.)
    weights = np.nan_to_num(weights, nan=0., posinf=0., neginf=0.)
    total = weights.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        offsets = (weights * win_radii[:, None]).sum(axis=0) / total

    valid_rows = np.isfinite(win_signal).sum(axis=0)
    bad = (~np.isfinite(snr) | (snr < min_snr) | (total <= 0) |
           (valid_rows < window.sum() // 2))
    offsets = np.where(bad, np.nan, offsets)
    return offsets, snr, coarse


def unwrap_longitudes(longitudes):
    """Return longitudes made monotonic across the 0/360 boundary."""
    if len(longitudes) == 0:
        return longitudes
    unwrapped = longitudes.copy()
    if unwrapped.max() - unwrapped.min() > 180.:
        unwrapped = np.where(unwrapped < 180., unwrapped + 360., unwrapped)
    return unwrapped


def summarize(offsets, longitudes):
    """Reduce per-column core offsets to offset, slope, scatter and coverage."""
    good = np.isfinite(offsets)
    n_good = int(good.sum())
    result = {
        'n_columns': len(offsets),
        'n_measured': n_good,
        'coverage': n_good / len(offsets) if len(offsets) else 0.,
        'offset_km': np.nan,
        'scatter_km': np.nan,
        'slope_km_per_deg': np.nan,
        'rise_km': np.nan,
        'span_deg': np.nan,
    }
    if n_good == 0:
        return result

    off = offsets[good]
    lon = unwrap_longitudes(longitudes[good])
    median = float(np.median(off))
    result['offset_km'] = median
    # Median absolute deviation, scaled to be comparable to a standard
    # deviation, so one wild column does not dominate.
    result['scatter_km'] = float(1.4826 * np.median(np.abs(off - median)))
    result['span_deg'] = float(lon.max() - lon.min())
    if n_good >= 10 and result['span_deg'] > 0.1:
        slope, _ = np.polyfit(lon - lon.mean(), off, 1)
        result['slope_km_per_deg'] = float(slope)
        # The tilt that matters is how far the core actually rises across the
        # image. A slope in km/deg looks alarming on a narrow high-resolution
        # image that in truth barely tilts at all.
        result['rise_km'] = float(slope * result['span_deg'])
    return result


def analyze_image(arguments, image_path):
    """Measure one reprojected image. Returns a summary dict, or None."""
    reproj_path = img_to_repro_path(arguments, image_path)
    if not os.path.exists(reproj_path):
        return None
    metadata = read_reproj_img(reproj_path)
    img = metadata['img']
    long_antimask = metadata['long_antimask']

    n_rad = img.shape[0]
    radii = (arguments.radius_inner_delta +
             np.arange(n_rad) * arguments.radius_resolution)
    longitudes = (np.arange(len(long_antimask))[long_antimask] *
                  arguments.longitude_resolution)
    if len(longitudes) != img.shape[1]:
        # Should not happen, but do not silently mismatch the two axes.
        return None

    offsets, snr, coarse = measure_core_offsets(
        img, radii,
        search_km=arguments.search_km,
        baseline_km=arguments.baseline_km,
        min_snr=arguments.min_snr)
    summary = summarize(offsets, longitudes)
    summary['coarse_offset_km'] = coarse
    summary['image'] = os.path.basename(image_path).replace('_CALIB.IMG', '')
    summary['reproj_path'] = reproj_path
    summary['median_snr'] = float(np.nanmedian(snr)) if np.isfinite(snr).any() else np.nan
    summary['_offsets'] = offsets
    summary['_longitudes'] = longitudes
    summary['_img'] = img
    summary['_radii'] = radii
    return summary


def flags_for(summary, arguments):
    """Return the list of reasons this image looks badly navigated."""
    reasons = []
    if summary['n_measured'] == 0:
        return ['no column had enough signal to measure']
    if summary['coverage'] < arguments.min_coverage:
        reasons.append(f'only {summary["coverage"]*100:.0f}% of columns measurable')
    if abs(summary['offset_km']) > arguments.max_offset_km:
        reasons.append(f'core offset {summary["offset_km"]:+.0f} km')
    if (np.isfinite(summary['rise_km']) and
            abs(summary['rise_km']) > arguments.max_rise_km):
        reasons.append(f'core tilts {summary["rise_km"]:+.0f} km across the image')
    if summary['scatter_km'] > arguments.max_scatter_km:
        reasons.append(f'core scatter {summary["scatter_km"]:.0f} km')
    return reasons


def plot_image(summary, obsid, out_path):
    """Write a diagnostic plot of one image with the measured core overlaid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img = np.where(summary['_img'] == SENTINEL, np.nan, summary['_img'])
    radii = summary['_radii']
    lon = summary['_longitudes']
    if len(lon) == 0:
        return

    finite = img[np.isfinite(img)]
    if len(finite) == 0:
        return
    lo, hi = np.percentile(finite, [1., 99.5])

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.imshow(img, aspect='auto', cmap='gray', vmin=lo, vmax=hi, origin='lower',
              extent=(0, len(lon), radii[0], radii[-1]))
    ax.plot(np.arange(len(lon)) + 0.5, summary['_offsets'], '.', ms=1.5,
            color='#ff5030', label='measured core')
    ax.axhline(0, color='#30a0ff', lw=0.8, ls='--', label='modeled core')
    ax.set_ylim(radii[0], radii[-1])
    ax.set_xlabel('Longitude column')
    ax.set_ylabel('Core-relative radius (km)')
    ax.set_title(f'{obsid} / {summary["image"]}   '
                 f'offset {summary["offset_km"]:+.0f} km, '
                 f'tilt {summary["rise_km"]:+.0f} km across, '
                 f'scatter {summary["scatter_km"]:.0f} km', fontsize=9)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def obsid_image_paths(arguments, obsid):
    """Return every source image of an observation, used in the mosaic or not."""
    _, metadata_path = f_ring.mosaic_paths(arguments, obsid)
    if not os.path.exists(metadata_path):
        return []
    with open(metadata_path, 'rb') as fp:
        metadata = msgpack.unpackb(fp.read(), object_hook=msgpack_numpy.decode)
    return list(metadata['image_path_list'])


def main():
    parser = argparse.ArgumentParser(
        description='Check reprojected image navigation using the F ring core')
    f_ring.add_parser_arguments(parser)
    parser.add_argument('--notes', default='',
                        help='Only process observations whose observation-list '
                             'notes contain any of these letters, e.g. "OR"')
    parser.add_argument('--search-km', type=float, default=400.,
                        help='Look for the core within this distance of center')
    parser.add_argument('--baseline-km', type=float, default=600.,
                        help='Rows beyond this distance define the background')
    parser.add_argument('--min-snr', type=float, default=5.,
                        help='Smallest peak-to-noise ratio worth measuring')
    parser.add_argument('--max-offset-km', type=float, default=100.,
                        help='Flag an image whose core offset exceeds this')
    parser.add_argument('--max-rise-km', type=float, default=150.,
                        help='Flag an image whose core tilts more than this in '
                             'total across the image')
    parser.add_argument('--max-scatter-km', type=float, default=100.,
                        help='Flag an image whose core scatter exceeds this')
    parser.add_argument('--min-coverage', type=float, default=0.25,
                        help='Flag an image with less measurable coverage')
    parser.add_argument('--csv', default=None,
                        help='Write per-image results to this CSV file')
    parser.add_argument('--plot-dir', default=None,
                        help='Write a diagnostic plot per flagged image here')
    parser.add_argument('--plot-all', action='store_true', default=False,
                        help='Plot every image, not only the flagged ones')
    arguments = parser.parse_args()
    f_ring.init(arguments)

    notes_by_obsid = read_observation_notes()
    wanted = set(arguments.notes.upper())

    if arguments.plot_dir:
        os.makedirs(arguments.plot_dir, exist_ok=True)

    rows = []
    flagged = []
    for obsid in f_ring.enumerate_obsids(arguments):
        notes = notes_by_obsid.get(obsid, '')
        if wanted and not (wanted & set(notes.upper())):
            continue
        image_paths = obsid_image_paths(arguments, obsid)
        if not image_paths:
            print(f'{obsid}: no mosaic metadata')
            continue
        print(f'{obsid} ({notes or "-"}): {len(image_paths)} images')
        for image_path in image_paths:
            summary = analyze_image(arguments, image_path)
            if summary is None:
                print(f'    {os.path.basename(image_path)}: no reprojected file')
                continue
            summary['obsid'] = obsid
            summary['notes'] = notes
            reasons = flags_for(summary, arguments)
            summary['flags'] = '; '.join(reasons)
            rows.append(summary)
            if reasons:
                flagged.append(summary)
                print(f'    {summary["image"]:14s} FLAG  {summary["flags"]}')
            elif arguments.verbose:
                print(f'    {summary["image"]:14s} ok    '
                      f'offset {summary["offset_km"]:+6.1f} km, '
                      f'slope {summary["slope_km_per_deg"]:+6.2f} km/deg, '
                      f'scatter {summary["scatter_km"]:5.1f} km')
            if arguments.plot_dir and (reasons or arguments.plot_all):
                plot_image(summary, obsid,
                           os.path.join(arguments.plot_dir,
                                        f'{obsid}_{summary["image"]}.png'))

    print()
    print(f'{len(rows)} images analyzed, {len(flagged)} flagged')
    if rows:
        offs = np.array([r['offset_km'] for r in rows])
        slopes = np.array([r['slope_km_per_deg'] for r in rows])
        rises = np.array([r['rise_km'] for r in rows])
        scats = np.array([r['scatter_km'] for r in rows])
        for name, vals, unit in (('offset', offs, 'km'),
                                 ('slope', slopes, 'km/deg'),
                                 ('rise', rises, 'km'),
                                 ('scatter', scats, 'km')):
            v = vals[np.isfinite(vals)]
            if len(v) == 0:
                continue
            print(f'  {name:8s} median {np.median(v):+8.2f} {unit:6s} '
                  f'  5th/95th {np.percentile(v, 5):+8.2f} / '
                  f'{np.percentile(v, 95):+8.2f}   max|.| {np.max(np.abs(v)):8.2f}')

    if arguments.csv:
        fields = ['obsid', 'notes', 'image', 'n_columns', 'n_measured',
                  'coverage', 'offset_km', 'scatter_km', 'slope_km_per_deg',
                  'rise_km', 'span_deg', 'coarse_offset_km', 'median_snr',
                  'flags']
        with open(arguments.csv, 'w', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f'Wrote {arguments.csv}')


if __name__ == '__main__':
    main()
