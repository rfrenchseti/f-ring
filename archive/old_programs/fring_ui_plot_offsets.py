'''
Created on Sep 19, 2011

@author: rfrench
'''

from optparse import OptionParser
import fring_util
import os
import os.path
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cb_util_file import *
import cb_config

POSTER = False

DIR_ROOT = os.path.join(RESULTS_ROOT, 'f-ring', 'offset-plots')

color_background = (1,1,1)
color_foreground = (0,0,0)
markersize = 8

if POSTER:
    markersize = 8
    matplotlib.rc('figure', facecolor=color_background)
    matplotlib.rc('axes', facecolor=color_background, edgecolor=color_foreground, labelcolor=color_foreground)
    matplotlib.rc('xtick', color=color_foreground, labelsize=18)
    matplotlib.rc('xtick.major', size=0)
    matplotlib.rc('xtick.minor', size=0)
    matplotlib.rc('ytick', color=color_foreground, labelsize=18)
    matplotlib.rc('ytick.major', size=0)
    matplotlib.rc('ytick.minor', size=0)
    matplotlib.rc('font', size=18)
    matplotlib.rc('legend', fontsize=24)

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['--verbose',
#                 '-a',
#'ISS_115RF_FMOVIEEQX001_PRIME',
                'ISS_029RF_FMOVIE001_VIMS',
#                 'ISS_044RF_FMOVIE001_VIMS',
#                'ISS_106RF_FMOVIE002_PRIME',
#                'ISS_132RI_FMOVIE001_VIMS',
#                'ISS_029RF_FMOVIE002_VIMS',
                ]

parser = OptionParser()

#
# The default behavior is to check the timestamps
# on the input file and the output file and recompute if the output file is out of date.
# Several options change this behavior:
#   --no-xxx: Don't recompute no matter what; this may leave you without an output file at all
#   --no-update: Don't recompute if the output file exists, but do compute if the output file doesn't exist at all
#   --recompute-xxx: Force recompute even if the output file exists and is current
#


##
## General options
##
parser.add_option('--allow-exception', dest='allow_exception',
                  action='store_true', default=False,
                  help="Allow exceptions to be thrown")

fring_util.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

#####################################################################################
#
#
#
#####################################################################################

g_num_images = 0
g_num_no_attempt = 0
g_num_no_offset = 0
g_num_used_stars = 0
g_num_used_model = 0
g_num_model_overrides = 0

verbose = True

def plot_obsid(obsid, image_path_list):
    offset_u_list = []
    offset_v_list = []
    num_stars_list = []
    num_good_stars_list = []
    used_objects_type_list = []

    global g_num_images
    global g_num_no_attempt
    global g_num_no_offset
    global g_num_used_stars
    global g_num_used_model
    global g_num_model_overrides

    num_images = 0
    num_no_attempt = 0
    num_no_offset = 0
    num_used_stars = 0
    num_used_model = 0
    num_model_overrides = 0

    for image_path in image_path_list:
        metadata = file_read_offset_metadata(image_path)
        num_images += 1
        if metadata is None or 'error' in metadata:
            num_no_attempt += 1
            continue
        auto_offset = metadata['offset']
        object_type = None
        stars_metadata = metadata['stars_metadata']
        if auto_offset is None:
            num_no_offset += 1
            offset_u_list.append(None)
            offset_v_list.append(None)
            num_stars_list.append(-1)
            num_good_stars_list.append(-1)
            used_objects_type_list.append('stars')
        else:
            offset_u_list.append(auto_offset[0])
            offset_v_list.append(auto_offset[1])
1            print(image_path, list(stars_metadata.keys()))
            num_stars_list.append(stars_metadata['num_stars'])
            num_good_stars_list.append(stars_metadata['num_good_stars'])
            object_type = metadata['used_objects_type']
            if metadata['model_overrides_stars']:
                object_type = 'override'
                num_model_overrides += 1
            used_objects_type_list.append(object_type)

    if len(offset_u_list) == 0:
        return

    x_min = -0.5
    x_max = len(offset_u_list)-0.5

    if POSTER:
        fig = plt.figure(figsize=(11.55,5))
    else:
        fig = plt.figure(figsize=(17,11))

    u_color = '#3399ff'
    v_color = '#0000cc'

    ax = fig.add_subplot(211)
    plt.plot(offset_u_list, '-', color=u_color, ms=5)
    plt.plot(offset_v_list, '-', color=v_color, ms=5)
    for i in range(len(offset_u_list)):
        if offset_u_list[i] is not None and offset_v_list[i] is not None:
            if used_objects_type_list[i] == 'stars':
                num_used_stars += 1
                plt.plot(i, offset_u_list[i], '*', mec=u_color, mfc=u_color, ms=markersize*1.25)
                plt.plot(i, offset_v_list[i], '*', mec=v_color, mfc=v_color, ms=markersize*1.25)
            elif used_objects_type_list[i] == 'model':
                num_used_model += 1
                plt.plot(i, offset_u_list[i], 'o', mec=u_color, mfc='none', ms=markersize, mew=1)
                plt.plot(i, offset_v_list[i], 'o', mec=v_color, mfc='none', ms=markersize, mew=1)
            elif used_objects_type_list[i] == 'override':
                num_used_model += 1
                plt.plot(i, offset_u_list[i], 'o', mec=u_color, mfc=u_color, ms=markersize, mew=1)
                plt.plot(i, offset_v_list[i], 'o', mec=v_color, mfc=v_color, ms=markersize, mew=1)
            else:
                plt.plot(i, offset_u_list[i], '^', mec=u_color, mfc='none', ms=markersize*1.5, mew=2)
                plt.plot(i, offset_v_list[i], '^', mec=v_color, mfc='none', ms=markersize*1.5, mew=2)
    plt.xlim(x_min, x_max)
    ax.set_xticklabels('')
    if POSTER:
        ax.get_yaxis().set_ticks([-30,-20,-10,0,10])
    plt.ylabel('Pixel Offset')

    if not POSTER:
        plt.title('X/Y Offset')

    ax.yaxis.set_label_coords(-0.055, 0.5)

    stars_color = '#ff8000'
    good_color = '#336600'

    ax = fig.add_subplot(212)
    plt.plot(num_stars_list, '-o', color=stars_color, mec=stars_color, mfc=stars_color, ms=markersize*.5)
    plt.plot(num_good_stars_list, '-o', color=good_color, mec=good_color, mfc=good_color, ms=markersize*.55)
    plt.xlim(x_min, x_max)
    plt.ylim(-0.5, max(np.max(num_good_stars_list),
                       np.max(num_stars_list))+0.5)
    plt.ylabel('# of Good Stars')
    plt.xlabel('Image Number')
    if POSTER:
        ax.get_xaxis().set_ticks([0,174])
        ax.get_yaxis().set_ticks([0,10,20,30])
        plt.xticks([0,174],['1','175'])
    if not POSTER:
        plt.title('Total Stars vs. Good Stars')

    ax.yaxis.set_label_coords(-0.055, 0.5)

    if not POSTER:
        plt.suptitle(obsid)

    plt.subplots_adjust(left=0.025, right=0.975, top=1, bottom=0.0, hspace=0.18)
    filename = os.path.join(DIR_ROOT, obsid+'.png')
    if POSTER:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    else:
        plt.savefig(filename, bbox_inches='tight')

    plt.close()

    print('%-40s %4d %4d %4d %4d %4d %4d' % (obsid, num_images, num_no_attempt, num_no_offset,
                                             num_used_stars, num_used_model, num_model_overrides))

    g_num_images += num_images
    g_num_no_attempt += num_no_attempt
    g_num_no_offset += num_no_offset
    g_num_used_stars += num_used_stars
    g_num_used_model += num_used_model
    g_num_model_overrides += num_model_overrides


#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

print('%-40s #IMG  ERR NOFF STAR MODL OVER' % ('OBSID'))

cur_obsid = None
image_path_list = []
for obsid, image_name, image_path in fring_util.enumerate_files(options, args):
#    print obsid, image_name
    if cur_obsid is None:
        cur_obsid = obsid
    if cur_obsid != obsid:
        if len(image_path_list) != 0:
            plot_obsid(cur_obsid, image_path_list)
        obsid_list = []
        image_path_list = []
        cur_obsid = obsid
    image_path_list.append(image_path)

# Final mosaic
if len(image_path_list) != 0:
    plot_obsid(cur_obsid, image_path_list)

print()
print('%-40s %4d %4d %4d %4d %4d %4d' % ('TOTAL', g_num_images, g_num_no_attempt, g_num_no_offset,
                         g_num_used_stars, g_num_used_model, g_num_model_overrides))
