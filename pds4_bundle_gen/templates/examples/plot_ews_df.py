"""Read a single mosaic using pandas DataFrames and plot the mosaic and
equivalent widths adjusted for viewing geometry.

Requirements:
    matplotlib
    numpy
    pandas
    pds4_tools

This program must be run from the ``document/user_guide`` directory so
mosaic_utils is available.

Usage: python plot_ews_df.py <mosaic_label_path>
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from mosaic_utils import (contrast_stretch,
                          get_element,
                          get_mosaic_name_from_mosaic_label,
                          read_mosaic_ma_df)


def add_equivalent_width(metadata_df, image_ma_data, long_interval,
                         radial_interval):
    """Add ``equivalent_width`` from radial I/F sums and viewing geometry.

    For each metadata row (one corotating longitude with valid data),
    EW = (radial integral times radial sampling interval) multiplied by
         mu = abs(cos(emission_angle)).

    Parameters:
        metadata_df (pd.DataFrame): Metadata from ``read_mosaic_ma_df``;
            mutated in place.
        image_ma_data (np.ma.MaskedArray): Mosaic I/F (masked).
        long_interval (float): Longitudinal sampling interval (degrees).
        radial_interval (float): Radial sampling interval (km).

    Returns:
        pd.DataFrame: Same ``metadata_df`` with a new column
        ``equivalent_width`` added (for chaining).
    """
    # The following code is necessary because the metadata_params table
    # only contains rows for longitudes with valid data. We want to extract the
    # corotating longitudes from the mosaic that match the longitudes listed
    # in the metadata_params table so that we can have equal-sized data sets
    # to operate on.
    # Get the indices of the good corotating longitudes in the mosaic image.
    # When reading from the metadata_params table, we use "rings_" for the
    # prefix because the process of reading the table converts ":" to "_".
    long = metadata_df['rings_corotating_ring_longitude'].to_numpy()
    # We assume that the corotating longitudes are in the range
    # [0, 360) and are appropriately quantized by long_interval.
    # We use np.round to avoid problems with floating point precision.
    long_idx = np.intp(np.round(long / long_interval))

    # Compute "mu", abs(cos(emission_angle)), to photometrically adjust the
    # equivalent widths for viewing angle.
    emission = metadata_df['rings_emission_angle'].to_numpy()
    mu = np.abs(np.cos(np.radians(emission)))

    # Extract the good longitudes from the mosaic image.
    mosaic_img_good_long = image_ma_data[:, long_idx]
    # Compute the sum of the I/F values for the good longitudes, replacing any
    # masked (sentinel) values with NaN because that's what pandas likes.
    integral = ma.filled(ma.sum(mosaic_img_good_long, axis=0), np.nan)
    # Exclude longitudes with incomplete radial coverage: masked interior
    # pixels shrink the integral, giving spuriously low EWs at coverage edges.
    # Require at least 99% valid radial pixels, mirroring the pipeline's
    # default --maximum-bad-pixels-percentage of 1%.
    valid_frac = (mosaic_img_good_long.count(axis=0) /
                  mosaic_img_good_long.shape[0])
    integral[valid_frac < 0.99] = np.nan
    # Convert the integral to equivalent width by multiplying by the viewing
    # angle correction factor and the radial sampling interval.
    metadata_df['equivalent_width'] = integral * mu * radial_interval

    return metadata_df


def build_plot(label):
    """Build the plots for the mosaic and equivalent widths.

    Make two plots. The top has radius going from the inner to the outer edge
    in km and shows the image_ma_data. The bottom shows the equivalent widths.
    There is only one set of longitudes at the very bottom.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # Raise subplot area (default top≈0.88 leaves a large gap); no per-axes
    # title.
    fig.subplots_adjust(hspace=.05, top=0.92)

    mosaic_name = get_mosaic_name_from_mosaic_label(label)
    fig.suptitle(
        f'Mosaic and Equivalent Widths for {mosaic_name}',
        y=0.97,
    )

    return axs


def main():
    """Plot equivalent widths adjusted for viewing geometry."""

    if len(sys.argv) != 2:
        print('Usage: python plot_ews_df.py <mosaic_label_path>')
        sys.exit(1)

    mosaic_label_path = sys.argv[1]

    label, image_ma_data, metadata_df = read_mosaic_ma_df(mosaic_label_path)

    # Get the sampling intervals for the reprojection grid.
    # Note that contrary to the metadata_params table, when reading from the
    # label we use "rings:" instead of "rings_" for the prefix.
    long_interval = get_element(
        label, 'rings:reprojection_grid_longitudinal_sampling_interval')
    radial_interval = get_element(
        label, 'rings:reprojection_grid_radial_sampling_interval')

    add_equivalent_width(
        metadata_df, image_ma_data, long_interval, radial_interval)

    ###### Everything from here onward is just plotting. ######

    # Symmetric relative-radius grid: line 0 is the inner (-) edge, last line
    # is the outer (+) edge. We assume the same extent in radius on either side
    # of the center.
    nradii = image_ma_data.shape[0]
    radial_min_km = -0.5 * (nradii - 1) * radial_interval
    radial_max_km = 0.5 * (nradii - 1) * radial_interval
    ytick_step = 200
    radius_y_ticks = np.arange(
        radial_min_km, radial_max_km + 0.5 * ytick_step, ytick_step)
    long_x_ticks = np.arange(0, 361, 30)

    # Contrast-stretch the image data so it's easier to see.
    greyscale_image_ma_data = contrast_stretch(image_ma_data)

    axs = build_plot(label)

    # Top plot - contrast-stretched image.
    axs[0].imshow(
        greyscale_image_ma_data,
        aspect='auto',
        cmap='gray',
        extent=(0.0, 360.0 - long_interval, radial_min_km, radial_max_km),
        origin='lower',
    )
    axs[0].set_yticks(radius_y_ticks)
    axs[0].set_ylabel('Radius Relative to Core (km)')
    # sharex links x-axis tick *locations*; set_xticks([]) on axs[0] would clear
    # them for axs[1] too. Only hide ticks/labels on the upper panel.
    axs[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # Bottom plot - equivalent widths.
    # Note this is a sparse plot, with one corotating longitude per DataFrame
    # row. Missing data is simply not plotted.
    axs[1].plot(metadata_df['rings_corotating_ring_longitude'],
                metadata_df['equivalent_width'], '.', ms=1, linestyle='none')
    axs[1].set_xlim(0.0, 360.0)
    axs[1].set_xticks(long_x_ticks)
    axs[1].set_ylabel(r'$EW \times \mu$ (km)')
    axs[1].set_xlabel('Corotating Longitude (deg)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
