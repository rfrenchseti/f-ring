import argparse
import numpy as np
import numpy.ma as ma
import os
import sys

import msgpack
import msgpack_numpy

from imgdisp import ImageDisp
from PIL import Image
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'external'))

import f_ring_util.f_ring as f_ring

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--display-radius-inner 140020 --display-radius-outer 140600 --radius-polar-base-factor 1.5'
    command_list = command_line_str.split()

parser = argparse.ArgumentParser()

##
## Options for mosaic creation
##
parser.add_argument('--display-radius-inner', type=float, default=None)
parser.add_argument('--display-radius-outer', type=float, default=None)
parser.add_argument('--radius-polar-base-factor', type=float, default=1.0)
parser.add_argument('--image-size', type=int, default=1024)
parser.add_argument('--whitepoint', type=float, default=None)
parser.add_argument('--blackpoint', type=float, default=None)
parser.add_argument('--gamma', type=float, default=None)
parser.add_argument('--display', action='store_true', default=False)

f_ring.add_parser_arguments(parser)

arguments = parser.parse_args(command_list)

#####################################################################################
#
# CREATE ONE POLAR PROJECTION MOSAIC
#
#####################################################################################

def polar_project_mosaic(obsid):
    data_path, metadata_path = f_ring.bkgnd_sub_mosaic_paths(arguments, obsid)
    with np.load(data_path) as npz:
        img = ma.MaskedArray(**npz)
        img.mask = False
        img = ma.masked_equal(img, -999)
        img = ma.filled(img, 0)
    with open(metadata_path, 'rb') as metadata_fp:
        metadata = msgpack.unpackb(metadata_fp.read(),
                                   max_str_len=40*1024*1024,
                                   object_hook=msgpack_numpy.decode)

    image_size = arguments.image_size
    radius_polar_base_factor = arguments.radius_polar_base_factor

    image_center = image_size // 2

    polar_img = np.zeros((image_size, image_size), dtype='float32')

    radius_inner = arguments.ring_radius + arguments.radius_inner_delta
    radius_outer = arguments.ring_radius + arguments.radius_outer_delta
    display_radius_inner = arguments.display_radius_inner
    if display_radius_inner is None:
        display_radius_inner = radius_inner
    display_radius_outer = arguments.display_radius_outer
    if display_radius_outer is None:
        display_radius_outer = radius_outer

    display_radius_min = (display_radius_outer -
                          (display_radius_outer-radius_inner) *
                          radius_polar_base_factor)
    display_radius_total = display_radius_outer - display_radius_min

    num_rad = img.shape[0]
    num_long = img.shape[1]

    # A grid with (y,x) for each final image pixel
    pixel_num = np.arange(image_size)
    polar_pixel_y, polar_pixel_x = np.meshgrid(pixel_num, pixel_num)

    # A grid with (rad,long) for each final image pixel
    offset_polar_pixel_x = polar_pixel_x-image_center
    offset_polar_pixel_y = polar_pixel_y-image_center
    polar_r = np.hypot(offset_polar_pixel_x, offset_polar_pixel_y)
    polar_r = polar_r*display_radius_total/image_center + display_radius_min
    polar_l = np.arctan2(offset_polar_pixel_y, offset_polar_pixel_x) % (2*np.pi)

    # A grid with (source rad pixel, source long pixel) for each final
    # image pixel
    img_x = polar_l*num_long/(2*np.pi)
    img_y = (polar_r-radius_inner)*num_rad/(radius_outer-radius_inner)
    img_x = np.round(img_x).astype(int)
    img_y = np.round(img_y).astype(int)

    mask = np.where((img_x >= 0) & (img_x < num_long) &
                    (img_y >= 0) & (img_y < num_rad))
    polar_pixel_x = polar_pixel_x[mask]
    polar_pixel_y = polar_pixel_y[mask]
    img_x = img_x[mask]
    img_y = img_y[mask]
    polar_img[polar_pixel_y, polar_pixel_x] = img[img_y, img_x]

    if arguments.blackpoint is not None:
        blackpoint = arguments.blackpoint
    else:
        blackpoint = max(np.min(polar_img), 0)
    whitepoint_ignore_frac = 0.995
    if arguments.whitepoint is not None:
        whitepoint = arguments.whitepoint
    else:
        img_sorted = np.sort(polar_img, axis=None)
        whitepoint = img_sorted[np.clip(int(len(img_sorted) * whitepoint_ignore_frac),
                                        0, len(img_sorted)-1)]
    if arguments.gamma is not None:
        gamma = arguments.gamma
    else:
        gamma = 0.5

    # The +0 forces a copy - necessary for PIL
    scaled_polar = np.cast['int8'](ImageDisp.scale_image(polar_img, blackpoint,
                                                         whitepoint, gamma))[::-1,:]+0
    pil_img = Image.frombuffer('L', (scaled_polar.shape[1], scaled_polar.shape[0]),
                               scaled_polar, 'raw', 'L', 0, 1)
    png_path = f_ring.polar_png_path(arguments, obsid, make_dirs=True)
    pil_img.save(png_path, 'PNG')

    if arguments.display:
        plt.imshow(polar_img)
        plt.show()

#     sp.misc.imsave(options.output_path + obs_id + options.format, final_img)


#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

cur_obsid = None

for obsid in f_ring.enumerate_obsids(arguments):
    if cur_obsid != obsid:
        cur_obsid = obsid
        print('Processing', obsid)
        polar_project_mosaic(obsid)
