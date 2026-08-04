"""Read a single mosaic using a ma.masked_array and plot the mosaic and
equivalent widths adjusted for viewing geometry.

Requirements:
    matplotlib
    numpy
    pds4_tools

This program must be run from the ``document/user_guide`` directory so
mosaic_utils is available.

Usage: python plot_ews_ma.py <mosaic_label_path>
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from mosaic_utils import (contrast_stretch,
                          get_element,
                          get_mosaic_name_from_mosaic_label,
                          read_mosaic_ma)


def compute_ews(image_ma_data, long_interval, radial_interval,
                metadata_params):
    """Compute the equivalent widths for a mosaic adjusted for viewing geometry.

    Returns the equivalent width (EW) for each longitude.
    EW = (radial integral times radial sampling interval) multiplied by
         mu = abs(cos(emission_angle)).

    Parameters:
        image_ma_data (ma.masked_array): The image data.
        long_interval (float): The longitudinal sampling interval (degrees).
        radial_interval (float): The radial sampling interval (km).
        metadata_params (np.ndarray): The metadata parameters as a structured
            numpy array.

    Returns:
        np.ma.MaskedArray: Equivalent widths for the mosaic image adjusted for
        viewing geometry.
    """
    # The following code is necessary because the metadata_params table
    # only contains rows for longitudes with valid data. If you want
    # an array that is the full width of the mosaic image with data
    # stored in the appropriate locations based on the corotating longitudes
    # listed in the table, you need to put them there explicitly.

    # Get the corotating longitude column and convert the corotating
    # longitudes into indexes for the full mosaic image.
    # When reading from the metadata_params table, we use "rings_" for the
    # prefix because the process of reading the table converts ":" to "_".
    table_corot = metadata_params['rings_corotating_ring_longitude']
    # We assume that the corotating longitudes are in the range
    # [0, 360) and are appropriately quantized by long_interval.
    # We use np.round to avoid problems with floating point precision.
    table_corot_idx = np.intp(np.round(table_corot / long_interval))

    # Get the emission angle column and convert it to a full-width array by
    # storing each value in the appropriate location corresponding to the
    # corotating longitude index.
    table_emission = metadata_params['rings_emission_angle']

    # Start with fully masked array of the proper size (#longitudes,).
    mosaic_emission = ma.masked_all(image_ma_data.shape[1])

    # Store the emission angle values in the appropriate locations.
    mosaic_emission[table_corot_idx] = table_emission

    # Compute "mu", abs(cos(emission_angle)), to photometrically adjust the
    # equivalent widths for viewing angle.
    mu = np.abs(np.cos(np.radians(mosaic_emission)))

    # Make an image array adjusted for viewing angle.
    adj_image_ma_data = image_ma_data * mu

    # Compute the equivalent widths for the mosaic, which are the integral
    # of I/F values for each radial slice multiplied by the radial sampling
    # interval.
    ews = np.sum(adj_image_ma_data * radial_interval, axis=0)

    # Exclude longitudes with incomplete radial coverage: masked interior
    # pixels shrink the integral, giving spuriously low EWs at coverage edges.
    # Require at least 99% valid radial pixels, mirroring the pipeline's
    # default --maximum-bad-pixels-percentage of 1%.
    valid_frac = image_ma_data.count(axis=0) / image_ma_data.shape[0]
    ews = ma.masked_where(valid_frac < 0.99, ews)

    return ews


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
        print('Usage: python plot_ews_ma.py <mosaic_label_path>')
        sys.exit(1)

    mosaic_label_path = sys.argv[1]

    # Read the mosaic.
    (label, image_ma_data, metadata_params) = read_mosaic_ma(mosaic_label_path)

    # Get the sampling intervals for the reprojection grid.
    # Note that contrary to the metadata_params table, when reading from the
    # label we use "rings:" instead of "rings_" for the prefix.
    long_interval = get_element(
        label, 'rings:reprojection_grid_longitudinal_sampling_interval')
    radial_interval = get_element(
        label, 'rings:reprojection_grid_radial_sampling_interval')

    # Compute the equivalent widths.
    ews = compute_ews(image_ma_data, long_interval, radial_interval,
                      metadata_params)

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
        extent=(0.0, 360.0-long_interval, radial_min_km, radial_max_km),
        origin='lower',
    )
    axs[0].set_yticks(radius_y_ticks)
    axs[0].set_ylabel('Radius Relative to Core (km)')
    # sharex links x-axis tick *locations*; set_xticks([]) on axs[0] would clear
    # them for axs[1] too. Only hide ticks/labels on the upper panel.
    axs[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # Bottom plot - equivalent widths.
    # Create a set of corotating longitudes for the X axis of the bottom plot
    # ranging from [0, 360) degrees.
    x_axis = np.arange(ews.size) * long_interval
    # Plot EWs. Note this is not a sparse plot. Every corotating longitude is
    # plotted. Masked values are ignored by matplotlib.
    axs[1].plot(x_axis, ews, '.', ms=1, linestyle='none')
    axs[1].set_xlim(0.0, 360.0)
    axs[1].set_xticks(long_x_ticks)
    axs[1].set_ylabel(r'$EW \times \mu$ (km)')
    axs[1].set_xlabel('Corotating Longitude (deg)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
