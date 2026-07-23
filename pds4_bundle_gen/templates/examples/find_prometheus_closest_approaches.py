"""Find the ten closest Prometheus approaches using the bundle global index.

Selects reprojected images whose global-index Prometheus longitude lies in each
product's valid corotating-longitude range (including 360 degree wrap), then
plots the ten smallest separations ``mean_core_radius - radius_prometheus``
(km). A red circle marks the Prometheus position on the image.

Requirements:
    matplotlib
    numpy
    pandas
    pds4_tools

This program must be run from the ``document/user_guide`` directory so
mosaic_utils is available.

Usage: python find_prometheus_closest_approaches.py <bundle_root>
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import numpy.ma as ma

from mosaic_utils import (contrast_stretch,
                          get_element,
                          get_mosaic_name_from_reproj_img_label,
                          get_reproj_img_name_from_label,
                          read_index_df,
                          read_reproj_img_ma)


def is_corot_longitude_in_range(long, lo, hi):
    """True where long lies in [lo, hi] on the corotating circle (deg).

    If lo <= hi, the valid arc does not wrap around 360 degrees.
    If lo > hi, check [lo, 360) and [0, hi].

    Arguments:
        long (np.array): The corotating longitudes to check.
        lo (np.array): The minimum corotating longitudes.
        hi (np.array): The maximum corotating longitudes.

    Returns:
        np.array: True where long lies in [lo, hi] on the corotating circle.
    """
    no_wrap = lo <= hi
    return np.where(
        no_wrap,
        (long >= lo) & (long <= hi),
        (long >= lo) | (long <= hi),
    )


def crop_reproj_to_corot_window(full_image_ma, long_interval,
                                min_long, max_long):
    """Crop image to the label min/max corotating longitude window.

    Crop the image from full size to the window defined by min_long and
    max_long. When min_long > max_long, the image is resliced to wrap around
    the 360 deg longitude boundary.

    Arguments:
        full_image_ma (np.ma.masked_array): The full image data.
        long_interval (float): The longitudinal sampling interval.
        min_long (float): The minimum corotating longitude.
        max_long (float): The maximum corotating longitude.

    Returns:
        numpy.ma.MaskedArray: The cropped and resliced image.
    """
    min_idx = int(np.round(min_long / long_interval))
    max_idx = int(np.round(max_long / long_interval))
    if min_idx <= max_idx:
        cropped = full_image_ma[:, min_idx:max_idx + 1]
    else:
        part1 = full_image_ma[:, min_idx:]
        part2 = full_image_ma[:, :max_idx + 1]
        cropped = ma.concatenate([part1, part2], axis=1)
    return cropped


def corot_lon_tick_label(v, _pos):
    """Format corotating longitude tick label as [0, 360) deg."""
    s = f'{v % 360.0:.1f}'
    return s.rstrip('0').rstrip('.')


def format_corot_longitude_ticks(ax):
    """Set x-axis tick labels to corotating longitude in [0, 360) deg."""
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(corot_lon_tick_label))


def add_prometheus_marker(ax, row, min_long, max_long, x0, x1,
                          radial_min_km, radial_max_km):
    """Draw a hollow red circle at Prometheus.

    Arguments:
        ax (matplotlib.axes.Axes): The axes to draw on.
        row (pandas.Series): The row from the global index DataFrame.
        min_long (float): The minimum corotating longitude.
        max_long (float): The maximum corotating longitude.
        x0 (float): The minimum corotating longitude X limit used for plotting.
        x1 (float): The maximum corotating longitude X limit used for plotting.
        radial_min_km (float): The minimum radial value.
        radial_max_km (float): The maximum radial value.
    """

    # First convert the Prometheus longitude in the range [0, 360) to be
    # consistent with the X axis on our plot. If there is wrap around, then
    # a longitude < min_long is actually greater than 360.
    xp = row['corotating_longitude_prometheus']
    if min_long > max_long and xp < min_long:
        xp = xp + 360.0

    # Get the Y position as an offset (in km) from the core radius.
    yp = row['radius_prometheus'] - row['mean_core_radius']

    # Compute the width and height of the plot in pixels.
    p0 = np.asarray(ax.transData.transform((x0, yp)))
    p1 = np.asarray(ax.transData.transform((x1, yp)))
    span_lon_pix = float(np.linalg.norm(p1 - p0))

    q0 = np.asarray(ax.transData.transform((xp, radial_min_km)))
    q1 = np.asarray(ax.transData.transform((xp, radial_max_km)))
    span_rad_pix = float(np.linalg.norm(q1 - q0))

    frac = 0.07  # Fraction of the smaller span to use for the radius.
    r_pix = max(frac * min(span_lon_pix, span_rad_pix), 4.0)
    # Convert to diameter in points
    diameter_pt = max(5.5 * r_pix * 72.0 / ax.figure.dpi, 8.0)

    ax.plot(xp, yp, 'o', ms=diameter_pt, mfc='none', mew=1.5, mec='red',
            zorder=10, clip_on=True)


def main():
    """Read the global index and plot the ten closest Prometheus approaches."""

    if len(sys.argv) != 2:
        print('Usage: python find_prometheus_closest_approaches.py '
              '<bundle_root>')
        sys.exit(1)

    bundle_root = os.path.abspath(sys.argv[1])
    index_lbl = os.path.join(
        bundle_root, 'miscellaneous', 'global_reproj_img_index.lblx')

    df = read_index_df(index_lbl)

    # Restrict the DataFrame to images with Prometheus in the images' corotating
    # longitude range.
    # When reading from the global index file, we use "rings_" for the prefix
    # because the process of reading the file converts ":" to "_".
    prometheus_long = df['corotating_longitude_prometheus']
    min_long = df['rings_minimum_corotating_ring_longitude']
    max_long = df['rings_maximum_corotating_ring_longitude']
    in_range = is_corot_longitude_in_range(prometheus_long, min_long, max_long)
    df_in_range = df.loc[in_range].copy()

    if df_in_range.empty:
        print('No rows left after longitude filter (check index / columns).')
        sys.exit(1)

    # Sort by increasing distance of Prometheus from the core.
    diff_km = (df_in_range['mean_core_radius'].astype(float)
               - df_in_range['radius_prometheus'].astype(float))
    df_in_range_10 = (
        df_in_range.assign(_prometheus_core_sep_km=diff_km)
        .sort_values('_prometheus_core_sep_km', ascending=True)
        .head(10))

    fig, axs = plt.subplots(5, 2, figsize=(11, 9))
    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.97)
    fig.suptitle('Ten closest Prometheus approaches', y=0.995)

    # Iterate over the rows describing the ten closest Prometheus approaches
    # and plot each.
    for ax, (_, row) in zip(axs.ravel(), df_in_range_10.iterrows()):
        filespec = row['file_spec']
        lbl_path = os.path.normpath(os.path.join(bundle_root, filespec))
        if not os.path.isfile(lbl_path):
            print(f'Missing label: {lbl_path}')
            ax.set_title(f'Missing label:\n{filespec}', fontsize=8)
            ax.axis('off')
            continue

        # Read the image data and get the sampling intervals.
        # Note that contrary to the metadata_params table or index file, when
        # reading from the label we use "rings:" instead of "rings_" for the
        # prefix.
        label, full_image_ma, _meta = read_reproj_img_ma(lbl_path)
        long_iv = get_element(
            label, 'rings:reprojection_grid_longitudinal_sampling_interval')
        rad_iv = get_element(
            label, 'rings:reprojection_grid_radial_sampling_interval')
        min_long = get_element(
            label, 'rings:minimum_corotating_ring_longitude')
        max_long = get_element(
            label, 'rings:maximum_corotating_ring_longitude')

        # Crop the image to the corotating longitude range.
        cropped_image_ma = crop_reproj_to_corot_window(
            full_image_ma, long_iv, min_long, max_long)

        # Contrast-stretch the image and compute the radial extent.
        stretched_image_ma = contrast_stretch(cropped_image_ma)
        nr = stretched_image_ma.shape[0]
        radial_min_km = -0.5 * (nr - 1) * rad_iv
        radial_max_km = 0.5 * (nr - 1) * rad_iv

        # Set the title.
        mosaic_name = get_mosaic_name_from_reproj_img_label(label)
        reproj_img_name = get_reproj_img_name_from_label(label)
        dist = row['_prometheus_core_sep_km']
        ax.set_title(
            f'{mosaic_name} / {reproj_img_name} '
            f'(Prometheus to Core {dist:.1f} km)',
            fontsize=8,
        )

        # We handle the case of the X axis wrapping around at 360 by actually
        # letting the max value be greater than 360, and then displaying the
        # proper value using a custom tick value display function.
        x0 = min_long
        x1 = max_long + 360.0 if min_long > max_long else max_long

        ax.imshow(
            stretched_image_ma,
            aspect='auto',
            cmap='gray',
            extent=(x0, x1, radial_min_km, radial_max_km),
            origin='lower',
        )
        add_prometheus_marker(
            ax, row, min_long, max_long, x0, x1,
            radial_min_km, radial_max_km)
        if min_long > max_long:
            # Custom tick value display function.
            format_corot_longitude_ticks(ax)

        ax.set_xlabel('Corotating longitude (deg)')
        ax.set_ylabel('Core Rel (km)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
