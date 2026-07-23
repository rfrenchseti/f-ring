"""Read and display single reprojected image using a ma.masked_array.

Requirements:
    matplotlib
    numpy
    pds4_tools

This program must be run from the ``document/user_guide`` directory so
mosaic_utils is available.

Usage: python display_reproj_img.py <reproj_img_label_path>
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from mosaic_utils import (contrast_stretch,
                          get_element,
                          get_mosaic_name_from_reproj_img_label,
                          get_reproj_img_name_from_label,
                          read_reproj_img_ma)


def main():
    """Display one reprojected image."""

    if len(sys.argv) != 2:
        print('Usage: python display_reproj_img.py <reproj_img_label_path>')
        sys.exit(1)

    reproj_img_label_path = sys.argv[1]

    # Read the reprojected image.
    (label, image_ma_data, _metadata_params) = read_reproj_img_ma(
        reproj_img_label_path)

    # Get the sampling intervals for the reprojection grid.
    # Note that contrary to the metadata_params table, when reading from the
    # label we use "rings:" instead of "rings_" for the prefix.
    long_interval = get_element(
        label, 'rings:reprojection_grid_longitudinal_sampling_interval')
    radial_interval = get_element(
        label, 'rings:reprojection_grid_radial_sampling_interval')

    mosaic_name = get_mosaic_name_from_reproj_img_label(label)
    reproj_img_name = get_reproj_img_name_from_label(label)

    # Symmetric relative-radius grid: line 0 is the inner (-) edge, last line
    # is the outer (+) edge. We assume the same extent in radius on either side
    # of the center.
    nradii = image_ma_data.shape[0]
    radial_min_km = -0.5 * (nradii - 1) * radial_interval
    radial_max_km = 0.5 * (nradii - 1) * radial_interval
    ytick_step = 200
    radius_y_ticks = np.arange(
        radial_min_km, radial_max_km + 0.5 * ytick_step, ytick_step)

    # Set the X axis tick locations.
    long_x_ticks = np.arange(0, 361, 30)

    # Contrast-stretch the image data so it's easier to see.
    greyscale_image_ma_data = contrast_stretch(image_ma_data)

    plt.figure(figsize=(10, 3))
    plt.title(f'Reprojected Image for {mosaic_name} / {reproj_img_name}')
    plt.imshow(
        greyscale_image_ma_data,
        aspect='auto',
        cmap='gray',
        # The extent is only [0, 360) so handle the open interval.
        extent=(0.0, 360.0-long_interval, radial_min_km, radial_max_km),
        origin='lower',
    )
    plt.yticks(radius_y_ticks)
    plt.ylabel('Radius Relative to Core (km)')
    plt.xticks(long_x_ticks)
    plt.xlabel('Corotating Longitude (deg)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
