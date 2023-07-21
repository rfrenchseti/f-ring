################################################################################
# ring_ui_reproject.py
#
# Reprojection of images into ring mosaics, with GUI and without.
################################################################################

# Issues:
# Use existing OFFSET programs
# Store manual offsets in a different file
# Use modern logging
# 80-char lines
# Handle rings other than F ring

import argparse
import logging
import numpy as np
import numpy.ma as ma
import os
import subprocess
import sys
import time

import cProfile, pstats, io
import traceback

from tkinter import *
from imgdisp import ImageDisp, FloatEntry

import cspyce

import nav.logging
from nav.config import PYTHON_EXE
from nav.file import (img_to_log_path,
                      img_to_offset_path,
                      read_offset_metadata,
                      write_offset_metadata)
from nav.file_oops import read_img_file
from nav.image import shift_image
from nav.offset import master_find_offset
from nav.nav import Navigation
from nav.ring_mosaic import (rings_generate_longitudes,
                             rings_reproject,
                             rings_ring_corotating_to_inertial,
                             rings_ring_inertial_to_corotating,
                             rings_ring_pixels)
from ring.ring_util import (img_to_repro_path,
                            read_repro,
                            ring_add_parser_arguments,
                            ring_basic_cmd_line,
                            ring_enumerate_files,
                            ring_init,
                            write_repro,
                            OffRepData,
                            RING_REPROJECT_PY)


command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--all-obsid --image-log-console-level info --max-subprocesses 3 --verbose'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser()

#
# For each of (offset, reprojection), the default behavior is to check the
# timestamps on the input file and the output file and recompute if the output
# file is out of date.
# Several options change this behavior:
#   --no-xxx: Don't recompute no matter what; this may leave you without an
#             output file at all
#   --no-update: Don't recompute if the output file exists, but do compute if
#                the output file doesn't exist at all
#   --recompute-xxx: Force recompute even if the output file exists and
#                    is current
#


##
## General options
##
parser.add_argument('--allow-exception', action='store_true', default=True,
                    help="Allow exceptions to be thrown")
parser.add_argument('--profile', action='store_true', default=False,
                    help="Do performance profiling")
parser.add_argument('--max-subprocesses', type=int, default=0,
                    help="Fork a subprocess for each file")
parser.add_argument('--image-logfile-level', default='info',
                    help='Logging level for the individual logfiles')
parser.add_argument('--image-log-console-level', default='info',
                    help='Logging level for the console')

##
## Options for finding the pointing offset
##
parser.add_argument('--no-auto-offset', action='store_true', default=False,
                    help="Don't compute the automatic offset even if we don't have one")
parser.add_argument('--no-update-auto-offset', action='store_true',
                    default=False,
                    help="Don't compute the automatic offset unless we don't have one")
parser.add_argument('--recompute-auto-offset', action='store_true',
                    default=False,
                    help='Recompute the automatic offset even if we already have one that is current')
parser.add_argument('--display-offset-reproject', action='store_true',
                    default=False,
                    help='Display the offset and reprojection and allow manual change')
parser.add_argument('--display-invalid-offset', dest='display_invalid_offset',
                    action='store_true', default=False,
                    help='Display the offset and reprojection and allow manual change only for images that have bad automatic offsets')
parser.add_argument('--display-invalid-reproject', action='store_true',
                    default=False,
                    help='Display the offset and reprojection and allow manual change only for images that have a bad reprojection')
parser.add_argument('--no-allow-stars', action='store_true', default=False,
                    help="Don't allow stars during auto offset")
parser.add_argument('--no-allow-moons', action='store_true', default=False,
                    help="Don't allow moons during auto offset")

##
## Options for reprojection
##
parser.add_argument('--no-reproject', action='store_true', default=False,
                    help="Don't compute the reprojection even if we don't have one")
parser.add_argument('--no-update-reproject', action='store_true', default=False,
                    help="Don't compute the reprojection unless if we don't have one")
parser.add_argument('--recompute-reproject', action='store_true', default=False,
                    help='Recompute the reprojection even if we already have one that is current')
parser.add_argument('--omit-saturns-shadow', action='store_true', default=False,
                    help="Omit Saturn's shadow during reprojection")

ring_add_parser_arguments(parser)

arguments = parser.parse_args(command_list)

assert not (arguments.display_offset_reproject and arguments.max_subprocesses)

ring_init(arguments)


class OffRepDispData:
    """Data used to display reproject info."""
    def __init__(self):
        self.obs = None
        self.toplevel = None
        self.imdisp_offset = None
        self.entry_x_offset = None
        self.entry_y_offset = None
        self.off_longitudes = None
        self.off_radii = None
        self.label_off_inertial_longitude = None
        self.label_off_corot_longitude = None
        self.label_off_radius = None
        self.imdisp_repro = None
        self.repro_overlay = None
        self.label_inertial_longitude = None
        self.label_corot_longitude = None
        self.label_radius = None
        self.repro_longitudes = None
        self.repro_phase_angles = None
        self.repro_incidence_angle = None
        self.repro_emission_angles = None
        self.repro_radial_resolutions = None
        self.repro_angular_resolutions = None
        self.button_b1_down = False
        self.button_b3_down = False


################################################################################
#
# RUN IN A SUBPROCESS
#
################################################################################

def collect_cmd_line():
    ret = ring_basic_cmd_line(arguments)
    if arguments.verbose:
        ret += ['--verbose']
    if arguments.no_auto_offset:
        ret += ['--no-auto-offset']
    if arguments.no_update_auto_offset:
        ret += ['--no-update-auto-offset']
    if arguments.recompute_auto_offset:
        ret += ['--recompute-auto-offset']
    if arguments.no_reproject:
        ret += ['--no-reproject']
    if arguments.no_update_reproject:
        ret += ['--no-update-reproject']
    if arguments.recompute_reproject:
        ret += ['--recompute-reproject']
    ret += ['--image-logfile-level', arguments.image_logfile_level]
    ret += ['--image-log-console-level', arguments.image_log_console_level]

    return ret

def run_and_maybe_wait(args):
    said_waiting = False
    while len(subprocess_list) == arguments.max_subprocesses:
        if arguments.verbose and not said_waiting:
            print('WAITING')
            said_waiting = True
        for i in range(len(subprocess_list)):
            if subprocess_list[i].poll() is not None:
                del subprocess_list[i]
                break
        if len(subprocess_list) == arguments.max_subprocesses:
            time.sleep(1)

    if arguments.verbose:
        print('SPAWNING SUBPROCESS')

    pid = subprocess.Popen(args)
    subprocess_list.append(pid)


################################################################################
#
# LOGGING
#
################################################################################

def setup_image_logging(offrepdata):
    if offrepdata.image_log_filehandler is not None: # Already set up
        return

    if image_logfile_level != nav.logging.LOGGING_SUPERCRITICAL:
        image_log_path = img_to_log_path(offrepdata.image_path,
                                         arguments.instrument_host,
                                         'ring_repro')

        if os.path.exists(image_log_path):
            os.remove(image_log_path) # XXX Need option to not do this

        offrepdata.image_log_filehandler = nav.logging.add_file_handler(
                                                image_log_path, image_logfile_level)
    else:
        offrepdata.image_log_filehandler = None


################################################################################
#
# FIND THE POINTING OFFSET
#
################################################################################

#
# The primary entrance for finding pointing offset
#

def offset_one_image(offrepdata, option_no, option_no_update, option_recompute,
                     save_results=True):
    if arguments.verbose:
        print('** Find offset', offrepdata.obsid, '/', end=' ')
        print(offrepdata.image_name, '-', end=' ')

    if option_no: # Just don't do anything - we hope you know what you're doing!
        if arguments.verbose:
            print('Ignored because of --no-auto_offset')
        return

    if os.path.exists(offrepdata.offset_path):
        if option_no_update:
            if arguments.verbose:
                print('Ignored because offset file already exists')
            return # Offset file already exists, don't update
        # Save the manual offset!
        metadata = read_offset_metadata(offrepdata.image_path,
                                        arguments.instrument_host,
                                        'saturn')
        if 'manual_offset' in metadata:
            offrepdata.manual_offset = metadata['manual_offset']
        time_offset = os.stat(offrepdata.offset_path).st_mtime
    else:
        time_offset = 0

    time_image = os.stat(offrepdata.image_path).st_mtime
    if time_offset >= time_image and not option_recompute:
        # The offset file exists and is more recent than the image,
        # and we're not forcing a recompute
        if arguments.verbose:
            print('Ignored because offset file is up to date')
        return

    if arguments.max_subprocesses:
        if arguments.verbose:
            print('QUEUEING SUBPROCESS')
        offrepdata.subprocess_run = True
        return

    setup_image_logging(offrepdata)

    # Recompute the automatic offset
    offrepdata.obs = Navigation(read_img_file(offrepdata.image_path,
                                              arguments.instrument_host),
                                arguments.instrument_host)
    adjust_voyager_calibration(offrepdata.obs, offrepdata.image_path)

    rings_config = nav.config.RINGS_DEFAULT_CONFIG.copy()
    # rings_config['fiducial_feature_threshold'] = 1 # XXX
    # rings_config['fiducial_feature_margin'] = 30 # XXX
    # rings_config['fiducial_ephemeris_width'] = 10 # XXX

    try:
        offrepdata.off_metadata = master_find_offset(offrepdata.obs,
                         allow_stars=not arguments.no_allow_stars,
                         allow_moons=not arguments.no_allow_moons,
                         create_overlay=True,
                         rings_config=rings_config)
    except:
        if arguments.verbose:
            print('COULD NOT FIND VALID OFFSET - PROBABLY SPICE ERROR')
        print('EXCEPTION:')
        print(sys.exc_info())
        err = 'Offset finding failed:\n' + traceback.format_exc()
        offrepdata.off_metadata = {}
        offrepdata.off_metadata['error'] = str(sys.exc_info()[1])
        offrepdata.off_metadata['error_traceback'] = err
        if arguments.allow_exception:
            raise
    if ('offset' in offrepdata.off_metadata and
        offrepdata.off_metadata['offset'] is not None):
        offrepdata.the_offset = offrepdata.off_metadata['offset']
        if arguments.verbose:
            print('FOUND %6.2f, %6.2f' % (offrepdata.the_offset[0],
                                          offrepdata.the_offset[1]))
    else:
        offrepdata.the_offset = None
        if arguments.verbose:
            print('COULD NOT FIND VALID OFFSET - PROBABLY BAD IMAGE')

    if offrepdata.manual_offset:
        offrepdata.off_metadata['manual_offset'] = offrepdata.manual_offset

    if save_results:
        write_offset_metadata(offrepdata.image_path, arguments.instrument_host,
                              offrepdata.off_metadata)


################################################################################
#
# REPROJECT ONE IMAGE
#
################################################################################

def adjust_voyager_calibration(obs, filepath):
    if os.path.split(filepath)[-1].startswith('C3'):
        # Voyager 1 @ Saturn needs to be adjusted
        obs.data *= 3.345

def _update_offrepdata_repro(offrepdata, metadata):
    if metadata is None:
        offrepdata.repro_long_antimask = None
        offrepdata.repro_img = None
        offrepdata.repro_longitudes = None
        offrepdata.repro_radial_resolutions = None
        offrepdata.repro_angular_resolutions = None
        offrepdata.repro_phase_angles = None
        offrepdata.repro_emission_angles = None
        offrepdata.repro_incidence_angle = None
        offrepdata.repro_time = None
    else:
        offrepdata.repro_long_antimask = metadata['long_antimask']
        offrepdata.repro_img = metadata['img']
        offrepdata.repro_radial_resolutions = metadata['mean_radial_resolution']
        offrepdata.repro_angular_resolutions = metadata['mean_angular_resolution']
        offrepdata.repro_phase_angles = metadata['mean_phase']
        offrepdata.repro_emission_angles = metadata['mean_emission']
        offrepdata.repro_incidence_angle = metadata['incidence']
        offrepdata.repro_time = metadata['time']

        # XXX ADD REST OF METADATA HERE AND DURING COMMIT

        full_longitudes = rings_generate_longitudes(
                longitude_resolution=np.radians(arguments.longitude_resolution))
        offrepdata.repro_longitudes = full_longitudes[
                                        offrepdata.repro_long_antimask]

def _write_repro_data(offrepdata):
    metadata = {}
    metadata['bad_pixel_map'] = offrepdata.bad_pixel_map
    metadata['img'] = offrepdata.repro_img
    metadata['long_antimask'] = offrepdata.repro_long_antimask
    metadata['mean_radial_resolution'] = offrepdata.repro_radial_resolutions
    metadata['mean_angular_resolution'] = offrepdata.repro_angular_resolutions
    metadata['mean_phase'] = offrepdata.repro_phase_angles
    metadata['mean_emission'] = offrepdata.repro_emission_angles
    metadata['incidence'] = offrepdata.repro_incidence_angle
    metadata['time'] = offrepdata.repro_time

    # To make the directories if necessary
    offrepdata.repro_path = img_to_repro_path(arguments,
                                              offrepdata.image_path,
                                              arguments.instrument_host,
                                              make_dirs=True)

    write_repro(offrepdata.repro_path, metadata)

def _reproject_one_image(offrepdata):
    if offrepdata.obs is None:
        image_logger.debug(f'Reading image file {offrepdata.image_path}')
        offrepdata.obs = Navigation(read_img_file(offrepdata.image_path,
                                                  arguments.instrument_host),
                                    arguments.instrument_host)
        adjust_voyager_calibration(offrepdata.obs, offrepdata.image_path)

    offset = None

    if offrepdata.manual_offset is not None:
        offset = offrepdata.manual_offset
    elif offrepdata.the_offset is not None:
        offset = offrepdata.the_offset
    else:
        image_logger.error(f'No offset found - aborting')
        print('NO OFFSET - REPROJECTION FAILED')
        _update_offrepdata_repro(offrepdata, None)
        return

    image_logger.info('Starting reprojection')
    try:
        data = offrepdata.obs.data.view(ma.MaskedArray)
        data.mask = offrepdata.bad_pixel_map
        data = ma.masked_equal(data, -999)
        if arguments.instrument_host == 'cassini':
            # For Cassini, 0 always means bad or missing data
            data = ma.masked_equal(data, 0)
        ret = rings_reproject(offrepdata.obs, data=data, offset=offset,
                              longitude_resolution=
                                np.radians(arguments.longitude_resolution),
                              radius_resolution=arguments.radius_resolution,
                              radius_range=(arguments.ring_radius+
                                            arguments.radius_inner_delta,
                                            arguments.ring_radius+
                                            arguments.radius_outer_delta),
                              corot_type=arguments.corot_type,
                              zoom_amt=(arguments.radial_zoom_amount,
                                        arguments.longitude_zoom_amount),
                              omit_saturns_shadow=arguments.omit_saturns_shadow)
    except:
        if arguments.verbose:
            print('REPROJECTION FAILED')
        print('EXCEPTION:')
        print(sys.exc_info())
        if arguments.allow_exception:
            raise
        ret = None

    image_logger.info('Reprojection complete')

    _update_offrepdata_repro(offrepdata, ret)

def reproject_one_image(offrepdata, option_no, option_no_update,
                        option_recompute):
    # Input file: offset_path (<IMAGE>_CALIB.IMG.OFFSET)
    # Output file: repro_path (<IMAGE>_<RES_DATA>_REPRO.IMG)

    if arguments.verbose:
        print('** Reproject', offrepdata.obsid, '/', offrepdata.image_name, '-', end=' ')

    if offrepdata.subprocess_run:
        if arguments.verbose:
            print('LETTING SUBPROCESS HANDLE IT')
        return

    if option_no:  # Just don't do anything
        if arguments.verbose:
            print('Ignored because of --no-reproject')
        return

    if os.path.exists(offrepdata.repro_path):
        if option_no_update:
            if arguments.verbose:
                print('Ignored because repro file already exists')
            return # Repro file already exists, don't update
        time_repro = os.stat(offrepdata.repro_path).st_mtime
    else:
        time_repro = 0

    if not os.path.exists(offrepdata.offset_path):
        if arguments.verbose:
            print('NO OFFSET FILE - ABORTING')
        return

    time_offset = os.stat(offrepdata.offset_path).st_mtime
    if time_repro >= time_offset and not option_recompute:
        # The repro file exists and is more recent than the image, and we're
        # not forcing a recompute
        if arguments.verbose:
            print('Ignored because repro file is up to date')
        return

    offrepdata.off_metadata = read_offset_metadata(offrepdata.image_path,
                                                   arguments.instrument_host, 'saturn')
    status = offrepdata.off_metadata['status']
    if status != 'ok':
        if arguments.verbose:
            print('Ignored because offsetting skipped file')
        return
    if 'offset' not in offrepdata.off_metadata:
        if arguments.verbose:
            print('Skipped because no offset field present')
        return
    if offrepdata.off_metadata['offset'] is None:
        offrepdata.the_offset = None
    else:
        offrepdata.the_offset = offrepdata.off_metadata['offset']
    if not 'manual_offset' in offrepdata.off_metadata:
        offrepdata.manual_offset = None
    else:
        offrepdata.manual_offset = offrepdata.off_metadata['manual_offset']

    if offrepdata.the_offset is None and offrepdata.manual_offset is None:
        if arguments.verbose:
            print('OFFSET IS INVALID - ABORTING')
        return

    if arguments.max_subprocesses:
        if arguments.verbose:
            print('QUEUEING SUBPROCESS')
        offrepdata.subprocess_run = True
        return

    setup_image_logging(offrepdata)

    if arguments.verbose:
        print('Running')

    _reproject_one_image(offrepdata)

    _write_repro_data(offrepdata)

    if arguments.verbose:
        print('   Reproject', offrepdata.obsid, '/', end=' ')
        print(offrepdata.image_name, '- DONE')


################################################################################
#
# DISPLAY ONE IMAGE AND ITS REPROJECTION ALLOWING MANUAL CHANGING OF THE OFFSET
#
################################################################################

def draw_repro_overlay(offrepdata, offrepdispdata):
    if offrepdata.repro_img is None:
        return

    # Mark the F ring core
    repro_overlay = np.zeros(offrepdata.repro_img.shape + (3,))
    y = int(float(arguments.radius_outer_delta)/
            (arguments.radius_outer_delta-arguments.radius_inner_delta)*
            offrepdata.repro_img.shape[0])
    y = repro_overlay.shape[0]-1-y
    if 0 <= y < repro_overlay.shape[0]:
        repro_overlay[y, :, 0] = 255

    # Mark the bad pixels
    repro_overlay[:, :, 1] = (offrepdata.repro_img == -999) * 192
    offrepdispdata.imdisp_repro.set_overlay(0, repro_overlay)

# Draw the offset curves
def draw_offset_overlay(offrepdata, offrepdispdata):
    # Blue - 0,0 offset
    # Red - auto offset
    # Green - manual offset
    try:
        offset_overlay = offrepdata.off_metadata['overlay'].copy()
        if offset_overlay.shape[:2] != offrepdata.obs.data.shape:
            # Correct for the expanded size of extdata
            diff_y = (offset_overlay.shape[0]-offrepdata.obs.data.shape[0])/2
            diff_x = (offset_overlay.shape[1]-offrepdata.obs.data.shape[1])/2
            offset_overlay = offset_overlay[diff_y:diff_y+
                                                offrepdata.obs.data.shape[0],
                                            diff_x:diff_x+
                                                offrepdata.obs.data.shape[1],:]
    except:
        offset_overlay = np.zeros((offrepdata.obs.data.shape + (3,)))
    if offrepdata.manual_offset is not None:
        if offrepdata.the_offset is None:
            x_diff = int(offrepdata.manual_offset[0])
            y_diff = int(offrepdata.manual_offset[1])
        else:
            x_diff = int(offrepdata.manual_offset[0] - offrepdata.the_offset[0])
            y_diff = int(offrepdata.manual_offset[1] - offrepdata.the_offset[1])
        offset_overlay = shift_image(offset_overlay, x_diff, y_diff)

    x_pixels, y_pixels = rings_ring_pixels(arguments.corot_type,
                                           offrepdata.obs) # No offset - blue
    x_pixels = np.clip(x_pixels.astype('int'), 0, offset_overlay.shape[1]-1)
    y_pixels = np.clip(y_pixels.astype('int'), 0, offset_overlay.shape[0]-1)
    offset_overlay[y_pixels, x_pixels, 2] = 255

    if (offrepdata.the_offset is not None and
        tuple(offrepdata.the_offset) != (0,0)):
        # Auto offset - red
        x_pixels, y_pixels = rings_ring_pixels(arguments.corot_type,
                                               offrepdata.obs,
                                               offset=offrepdata.the_offset)
        x_pixels = np.clip(x_pixels.astype('int'), 0, offset_overlay.shape[1]-1)
        y_pixels = np.clip(y_pixels.astype('int'), 0, offset_overlay.shape[0]-1)
        offset_overlay[y_pixels, x_pixels, 0] = 255

    if (offrepdata.manual_offset is not None and
        tuple(offrepdata.manual_offset) != (0,0)):
        # Manual offset - green
        x_pixels, y_pixels = rings_ring_pixels(arguments.corot_type,
                                               offrepdata.obs,
                                               offset=offrepdata.manual_offset)
        x_pixels = np.clip(x_pixels.astype('int'), 0, offset_overlay.shape[1]-1)
        y_pixels = np.clip(y_pixels.astype('int'), 0, offset_overlay.shape[0]-1)
        offset_overlay[y_pixels, x_pixels, 1] = 255

    if offrepdata.bad_pixel_map is not None:
        offset_overlay[offrepdata.bad_pixel_map, 1] = 255 # Green

    offrepdispdata.imdisp_offset.set_overlay(0, offset_overlay)
    offrepdispdata.imdisp_offset.pack(side=LEFT)

    offrepdispdata.offset_overlay = offset_overlay

# The callback for mouse move events on the offset image
def callback_mouse_offset(x, y, offrepdata, offrepdispdata):
    if offrepdispdata.button_b1_down or offrepdispdata.button_b3_down:
        if offrepdata.bad_pixel_map is None:
            offrepdata.bad_pixel_map = np.zeros_like(offrepdata.obs.data,
                                                     dtype='bool')

        xmin = max(0, int(x)-7)
        xmax = min(offrepdata.obs.data.shape[1]-1, int(x)+7)
        ymin = max(0, int(y)-7)
        ymax = min(offrepdata.obs.data.shape[0]-1, int(y)+7)
        val = 0 # Erase
        if offrepdispdata.button_b1_down:
            val = 1 # Mask
        offrepdata.bad_pixel_map[ymin:ymax+1, xmin:xmax+1] = val
        offrepdispdata.offset_overlay[ymin:ymax+1, xmin:xmax+1, 1] = val * 255
        offrepdispdata.imdisp_offset.set_overlay(0, offrepdispdata.offset_overlay)

    if offrepdata.manual_offset is not None:
        x -= offrepdata.manual_offset[0]
        y -= offrepdata.manual_offset[1]
    elif offrepdata.the_offset is not None:
        x -= offrepdata.the_offset[0]
        y -= offrepdata.the_offset[1]
    if (x < 0 or x > offrepdata.obs.data.shape[1]-1 or
        y < 0 or y > offrepdata.obs.data.shape[0]-1):
        return

    x = int(x)
    y = int(y)

    if offrepdispdata.off_longitudes is not None:
        offrepdispdata.label_off_corot_longitude.config(text=
                        ('%7.3f'%(np.degrees(offrepdispdata.off_longitudes[y,x]))))
        offrepdispdata.label_off_inertial_longitude.config(
               text=('%7.3f'%(np.degrees(rings_ring_corotating_to_inertial(
                                 arguments.corot_type,
                                 offrepdispdata.off_longitudes[y,x],
                                 offrepdata.obs.midtime)))))
    if offrepdispdata.off_radii is not None:
        offrepdispdata.label_off_radius.config(text=
                        ('%7.3f'%offrepdispdata.off_radii[y,x]))

# The callbacks for button B1 press and release on the offset image
def callback_b1press_offset(x, y, offrepdata, offrepdispdata):
    print('B1 down')
    offrepdispdata.button_b1_down = True
    # Simulate a mouse move to set the bad pixel
    callback_mouse_offset(x, y, offrepdata, offrepdispdata)

def callback_b1release_offset(x, y, offrepdata, offrepdispdata):
    offrepdispdata.button_b1_down = False

# The callbacks for button B3 press and release on the offset image
def callback_b3press_offset(x, y, offrepdata, offrepdispdata):
    print('B3 down')
    offrepdispdata.button_b3_down = True
    # Simulate a mouse move to unset the bad pixel
    callback_mouse_offset(x, y, offrepdata, offrepdispdata)

def callback_b3release_offset(x, y, offrepdata, offrepdispdata):
    offrepdispdata.button_b3_down = False


# "Manual from auto" button pressed
def command_man_from_auto(offrepdata, offrepdispdata):
    offrepdata.manual_offset = offrepdata.the_offset
    offrepdispdata.entry_x_offset.delete(0, END)
    offrepdispdata.entry_y_offset.delete(0, END)
    if offrepdata.manual_offset is not None:
        offrepdispdata.entry_x_offset.insert(0,
                    '%6.2f'%offrepdata.the_offset[0])
        offrepdispdata.entry_y_offset.insert(0,
                    '%6.2f'%offrepdata.the_offset[1])
    draw_offset_overlay(offrepdata, offrepdispdata)

def command_man_from_cassini(offrepdata, offrepdispdata):
    offrepdata.manual_offset = (0.,0.)
    offrepdispdata.entry_x_offset.delete(0, END)
    offrepdispdata.entry_y_offset.delete(0, END)
    if offrepdata.manual_offset is not None:
        offrepdispdata.entry_x_offset.insert(0,
                    '%6.2f'%offrepdata.manual_offset[0])
        offrepdispdata.entry_y_offset.insert(0,
                    '%6.2f'%offrepdata.manual_offset[1])
    draw_offset_overlay(offrepdata, offrepdispdata)

# <Enter> key pressed in a manual offset text entry box
def command_enter_offset(event, offrepdata, offrepdispdata):
    if (offrepdispdata.entry_x_offset.get() == "" or
        offrepdispdata.entry_y_offset.get() == ""):
        offrepdata.manual_offset = None
    else:
        offrepdata.manual_offset = (float(offrepdispdata.entry_x_offset.get()),
                                    float(offrepdispdata.entry_y_offset.get()))
    draw_offset_overlay(offrepdata, offrepdispdata)

# "Recalculate offset" button pressed
def command_recalc_offset(offrepdata, offrepdispdata):
    offset_one_image(offrepdata, False, False, True, save_results=False)
    offrepdata.manual_offset = None
    offrepdispdata.entry_x_offset.delete(0, END)
    offrepdispdata.entry_y_offset.delete(0, END)
    if offrepdata.the_offset is None:
        auto_x_text = 'Auto X Offset: None'
        auto_y_text = 'Auto Y Offset: None'
    else:
        auto_x_text = 'Auto X Offset: %6.2f'%offrepdata.the_offset[0]
        auto_y_text = 'Auto Y Offset: %6.2f'%offrepdata.the_offset[1]

    offrepdispdata.auto_x_label.config(text=auto_x_text)
    offrepdispdata.auto_y_label.config(text=auto_y_text)
    draw_offset_overlay(offrepdata, offrepdispdata)
#    refresh_repro_img(offrepdata, offrepdispdata)

# "Refresh reprojection" button pressed
def refresh_repro_img(offrepdata, offrepdispdata):
    _reproject_one_image(offrepdata)

    offrepdispdata.repro_longitudes = offrepdata.repro_longitudes
    offrepdispdata.repro_radial_resolutions = offrepdata.repro_radial_resolutions
    offrepdispdata.repro_angular_resolutions = offrepdata.repro_angular_resolutions
    offrepdispdata.repro_phase_angles = offrepdata.repro_phase_angles
    offrepdispdata.repro_emission_angles = offrepdata.repro_emission_angles
    offrepdispdata.repro_incidence_angle = offrepdata.repro_incidence_angle

#    temp_img = None
#    if offrepdata.repro_img is not None:
#        temp_img = offrepdata.repro_img[::1,:] # Flip it upside down for display - Saturn at bottom XXX
    if offrepdata.repro_img is None:
        offrepdispdata.imdisp_repro.update_image_data([np.zeros((1024,1024))],
                                                      [None])
    else:
        # In the reprojected image, -999 means bad or missing pixel
        # It will be masked in draw_repro_overlay, but we don't want it to affect
        # calculation of blackpoint
        clean_img = offrepdata.repro_img.copy()
        clean_img[clean_img == -999] = 0
        offrepdispdata.imdisp_repro.update_image_data([clean_img],
                                                      [None])
    draw_repro_overlay(offrepdata, offrepdispdata)

# "Commit changes" button pressed
def command_commit_changes(offrepdata, offrepdispdata):
    if (offrepdispdata.entry_x_offset.get() == "" or
        offrepdispdata.entry_y_offset.get() == ""):
        offrepdata.manual_offset = None
        offrepdata.off_metadata['manual_offset'] = None
    else:
        offrepdata.manual_offset = (float(offrepdispdata.entry_x_offset.get()),
                                    float(offrepdispdata.entry_y_offset.get()))
        offrepdata.off_metadata['manual_offset'] = offrepdata.manual_offset
    write_offset_metadata(offrepdata.image_path, arguments.instrument_host,
                          offrepdata.off_metadata)
    _write_repro_data(offrepdata)

# Setup the offset/reproject window with no data
def setup_offset_reproject_window(offrepdata, offrepdispdata):

    offrepdispdata.off_radii = (offrepdata.obs.bp.ring_radius('saturn:ring')
                                .vals.astype('float'))
    offrepdispdata.off_longitudes = (offrepdata.obs.bp
                                     .ring_longitude('saturn:ring')
                                     .vals.astype('float'))
    offrepdispdata.off_longitudes = rings_ring_inertial_to_corotating(
                           arguments.corot_type,
                           offrepdispdata.off_longitudes,
                           offrepdata.obs.midtime)

    offrepdispdata.toplevel = Tk()
    offrepdispdata.toplevel.title(
        f'{offrepdata.obsid} / {offrepdata.image_name} ('
        f'RR {arguments.radius_resolution:.2f}, '
        f'LR {arguments.longitude_resolution:.2f}, '
        f'RZ {arguments.radial_zoom_amount:d}, '
        f'LZ {arguments.longitude_zoom_amount:d})')

    frame_toplevel = Frame(offrepdispdata.toplevel)

    # The original image and overlaid ring curves
    offrepdispdata.imdisp_offset = ImageDisp([offrepdata.obs.data],
                                             canvas_size=(1024,1024),
                                             enlarge_limit=5,
                                             auto_update=True,
                                             parent=frame_toplevel)
#    offrepdispdata.imdisp_offset.set_image_params(0., 0.00121, 0.5) # XXX - N1557046172_1


    # The reprojected image
    if offrepdata.repro_img is None:
        offrepdispdata.imdisp_repro = ImageDisp([np.zeros((1024,1024))],
                                    parent=frame_toplevel,
                                    canvas_size=(512,512),
                                    overlay_list=[np.zeros((1024,1024,3))],
                                    enlarge_limit=5,
                                    auto_update=True,
                                    one_zoom=False,
                                    flip_y=True)
    else:
        # In the reprojected image, -999 means bad or missing pixel
        # It will be masked in draw_repro_overlay, but we don't want it to affect
        # calculation of blackpoint
        clean_img = offrepdata.repro_img.copy()
        clean_img[clean_img == -999] = 0
        offrepdispdata.imdisp_repro = ImageDisp([clean_img],
                                    parent=frame_toplevel,
                                    canvas_size=(512,512),
                                    overlay_list=[offrepdispdata.repro_overlay],
                                    enlarge_limit=5,
                                    auto_update=True,
                                    one_zoom=False,
                                    flip_y=True)

    ###############################################
    # The control/data pane of the original image #
    ###############################################

    img_addon_control_frame = offrepdispdata.imdisp_offset.addon_control_frame

    gridrow = 0
    gridcolumn = 0

    if offrepdata.the_offset is None:
        auto_x_text = 'Auto X Offset: None'
        auto_y_text = 'Auto Y Offset: None'
    else:
        auto_x_text = 'Auto X Offset: %6.2f'%offrepdata.the_offset[0]
        auto_y_text = 'Auto Y Offset: %6.2f'%offrepdata.the_offset[1]

    offrepdispdata.auto_x_label = Label(img_addon_control_frame,
                                        text=auto_x_text)
    offrepdispdata.auto_x_label.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1
    offrepdispdata.auto_y_label = Label(img_addon_control_frame,
                                        text=auto_y_text)
    offrepdispdata.auto_y_label.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    # X offset and Y offset entry boxes
    # We should really use variables for the Entry boxes,
    # but for some reason they don't work
    label = Label(img_addon_control_frame, text='X Offset')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)

    offrepdispdata.entry_x_offset = FloatEntry(img_addon_control_frame)
    offrepdispdata.entry_x_offset.delete(0, END)
    if offrepdata.manual_offset is not None:
        offrepdispdata.entry_x_offset.insert(0,
                                '%6.2f'%offrepdata.manual_offset[0])
    offrepdispdata.entry_x_offset.grid(row=gridrow, column=gridcolumn+1,
                                       sticky=W)
    gridrow += 1

    label = Label(img_addon_control_frame, text='Y Offset')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)

    offrepdispdata.entry_y_offset = FloatEntry(img_addon_control_frame)
    offrepdispdata.entry_y_offset.delete(0, END)
    if offrepdata.manual_offset is not None:
        offrepdispdata.entry_y_offset.insert(0,
                                '%6.2f'%offrepdata.manual_offset[1])
    offrepdispdata.entry_y_offset.grid(row=gridrow, column=gridcolumn+1,
                                       sticky=W)
    gridrow += 1

    enter_offset_command = (lambda x, offrepdata=offrepdata,
                                   offrepdispdata=offrepdispdata:
                            command_enter_offset(x, offrepdata, offrepdispdata))
    offrepdispdata.entry_x_offset.bind('<Return>', enter_offset_command)
    offrepdispdata.entry_y_offset.bind('<Return>', enter_offset_command)

    # Set manual to automatic
    button_man_from_auto_command = (lambda offrepdata=offrepdata,
                                           offrepdispdata=offrepdispdata:
                                    command_man_from_auto(offrepdata,
                                                          offrepdispdata))
    button_man_from_auto = Button(img_addon_control_frame,
                                  text='Set Manual from Auto',
                                  command=button_man_from_auto_command)
    button_man_from_auto.grid(row=gridrow, column=gridcolumn+1)
    gridrow += 1

    #Set manual to Cassini
    button_man_from_cassini_command = (lambda offrepdata=offrepdata,
                                              offrepdispdata=offrepdispdata:
                                       command_man_from_cassini(offrepdata,
                                                                offrepdispdata))
    button_man_cassini_auto = Button(img_addon_control_frame,
                                     text='Set Manual from Cassini',
                                     command=button_man_from_cassini_command)
    button_man_cassini_auto.grid(row=gridrow, column=gridcolumn+1)
    gridrow += 1

    # Recalculate auto offset
    button_recalc_offset_command = (lambda offrepdata=offrepdata,
                                           offrepdispdata=offrepdispdata:
                                    command_recalc_offset(offrepdata,
                                                          offrepdispdata))
    button_recalc_offset = Button(img_addon_control_frame,
                                  text='Recalculate Offset',
                                  command=button_recalc_offset_command)
    button_recalc_offset.grid(row=gridrow, column=gridcolumn+1)
    gridrow += 1

    # Refresh reprojection buttons
    button_refresh_command = (lambda offrepdata=offrepdata,
                                     offrepdispdata=offrepdispdata:
                              refresh_repro_img(offrepdata,
                                                offrepdispdata))
    button_refresh = Button(img_addon_control_frame,
                            text='Refresh Reprojection',
                            command=button_refresh_command)
    button_refresh.grid(row=gridrow, column=gridcolumn+1)
    gridrow += 1

    # Commit results button - saves new offset and reprojection
    button_commit_changes_command = (lambda offrepdata=offrepdata,
                                            offrepdispdata=offrepdispdata:
                                     command_commit_changes(offrepdata,
                                                            offrepdispdata))
    button_commit_changes = Button(img_addon_control_frame,
                                   text='Commit Changes',
                                   command=button_commit_changes_command)
    button_commit_changes.grid(row=gridrow, column=gridcolumn+1)
    gridrow += 1

    gridcolumn += 2
    gridrow = 0

    # Display for longitude and radius
    label = Label(img_addon_control_frame, text='Co-Rot Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_off_corot_longitude = Label(img_addon_control_frame,
                                                     text='')
    offrepdispdata.label_off_corot_longitude.grid(row=gridrow,
                                                  column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(img_addon_control_frame, text='Radius:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_off_radius = Label(img_addon_control_frame, text='')
    offrepdispdata.label_off_radius.grid(row=gridrow, column=gridcolumn+1,
                                         sticky=W)
    gridrow += 1

    label = Label(img_addon_control_frame, text='Inertial Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_off_inertial_longitude = Label(img_addon_control_frame,
                                                        text='')
    offrepdispdata.label_off_inertial_longitude.grid(row=gridrow,
                                                     column=gridcolumn+1,
                                                     sticky=W)
    gridrow += 1

    callback_mouse_offset_command = (lambda x, y, offrepdata=offrepdata,
                                           offrepdispdata=offrepdispdata:
                                              callback_mouse_offset(x, y, offrepdata,
                                                              offrepdispdata))
    offrepdispdata.imdisp_offset.bind_mousemove(0, callback_mouse_offset_command)

    callback_b1press_offset_command = (lambda x, y, offrepdata=offrepdata,
                                              offrepdispdata=offrepdispdata:
                                              callback_b1press_offset(x, y, offrepdata,
                                                                      offrepdispdata))
    offrepdispdata.imdisp_offset.bind_b1press(0, callback_b1press_offset_command)

    callback_b1release_offset_command = (lambda x, y, offrepdata=offrepdata,
                                              offrepdispdata=offrepdispdata:
                                                callback_b1release_offset(x, y, offrepdata,
                                                                          offrepdispdata))
    offrepdispdata.imdisp_offset.bind_b1release(0, callback_b1release_offset_command)

    callback_b3press_offset_command = (lambda x, y, offrepdata=offrepdata,
                                              offrepdispdata=offrepdispdata:
                                              callback_b3press_offset(x, y, offrepdata,
                                                                      offrepdispdata))
    offrepdispdata.imdisp_offset.bind_b3press(0, callback_b3press_offset_command)

    callback_b3release_offset_command = (lambda x, y, offrepdata=offrepdata,
                                              offrepdispdata=offrepdispdata:
                                                callback_b3release_offset(x, y, offrepdata,
                                                                          offrepdispdata))
    offrepdispdata.imdisp_offset.bind_b3release(0, callback_b3release_offset_command)


    ##################################################
    # The control/data pane of the reprojected image #
    ##################################################

    repro_addon_control_frame = offrepdispdata.imdisp_repro.addon_control_frame

    gridrow = 0
    gridcolumn = 0

    label = Label(repro_addon_control_frame, text='Date:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_date = Label(repro_addon_control_frame,
                                      text=cspyce.et2utc(offrepdata.obs.midtime,
                                                         'C', 0))
    offrepdispdata.label_date.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(repro_addon_control_frame, text='Co-Rot Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_corot_longitude = Label(repro_addon_control_frame,
                                                 text='')
    offrepdispdata.label_corot_longitude.grid(row=gridrow, column=gridcolumn+1,
                                              sticky=W)
    gridrow += 1

    label = Label(repro_addon_control_frame, text='Radius:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_radius = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_radius.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(repro_addon_control_frame, text='Phase:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_phase = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_phase.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(repro_addon_control_frame, text='Incidence:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_incidence = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_incidence.grid(row=gridrow, column=gridcolumn+1,
                                        sticky=W)
    gridrow += 1

    label = Label(repro_addon_control_frame, text='Emission:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_emission = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_emission.grid(row=gridrow, column=gridcolumn+1,
                                       sticky=W)
    gridrow += 1

    label = Label(repro_addon_control_frame, text='Radial Res:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_radial_resolution = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_radial_resolution.grid(row=gridrow, column=gridcolumn+1,
                                                sticky=W)
    gridrow += 1

    label = Label(repro_addon_control_frame, text='Angular Res:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_angular_resolution = Label(repro_addon_control_frame, text='')
    offrepdispdata.label_angular_resolution.grid(row=gridrow, column=gridcolumn+1,
                                                sticky=W)
    gridrow += 1

    label = Label(repro_addon_control_frame, text='Inertial Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    offrepdispdata.label_inertial_longitude = Label(repro_addon_control_frame,
                                                    text='')
    offrepdispdata.label_inertial_longitude.grid(row=gridrow,
                                                 column=gridcolumn+1,
                                                 sticky=W)
    gridrow += 1

    offrepdispdata.imdisp_repro.pack(side=LEFT)

    callback_repro_command = (lambda x, y, offrepdata=offrepdata,
                                           offrepdispdata=offrepdispdata:
                              callback_repro(x, y, offrepdata, offrepdispdata))
    offrepdispdata.imdisp_repro.bind_mousemove(0, callback_repro_command)

    frame_toplevel.pack()

    draw_offset_overlay(offrepdata, offrepdispdata)
    draw_repro_overlay(offrepdata, offrepdispdata)

# Display the original and reproject images (if any)
def display_offset_reproject(offrepdata, offrepdispdata, option_invalid_offset,
                             option_invalid_reproject, do_mainloop=True):
    if arguments.verbose:
        print('** Display', offrepdata.obsid, '/', offrepdata.image_name)
    if offrepdata.off_metadata is None:
        offrepdata.off_metadata = read_offset_metadata(offrepdata.image_path,
                                                       arguments.instrument_host, 'saturn')
        if offrepdata.off_metadata is None:
            offrepdata.the_offset = None
            offrepdata.manual_offset = None
        else:
            if ('offset' not in offrepdata.off_metadata or
                offrepdata.off_metadata['offset'] is None):
                offrepdata.the_offset = None
            else:
                offrepdata.the_offset = offrepdata.off_metadata['offset']
            if 'manual_offset' not in offrepdata.off_metadata:
                offrepdata.manual_offset = None
            else:
                offrepdata.manual_offset = offrepdata.off_metadata['manual_offset']

    if (option_invalid_offset and
        (offrepdata.the_offset is not None or
         offrepdata.manual_offset is not None)):
        if arguments.verbose:
            print('Skipping because not invalid')
        return

    # The original image

    if offrepdata.obs is None:
        offrepdata.obs = Navigation(read_img_file(offrepdata.image_path,
                                                  arguments.instrument_host),
                                    arguments.instrument_host)
        adjust_voyager_calibration(offrepdata.obs, offrepdata.image_path)

    if offrepdata.repro_img is not None:
        offrepdispdata.repro_overlay = np.zeros(offrepdata.repro_img.shape +
                                                (3,))
    else:
        offrepdispdata.repro_overlay = None

    offrepdispdata.repro_longitudes = offrepdata.repro_longitudes
    offrepdispdata.repro_radial_resolutions = offrepdata.repro_radial_resolutions
    offrepdispdata.repro_angular_resolutions = offrepdata.repro_angular_resolutions
    offrepdispdata.repro_phase_angles = offrepdata.repro_phase_angles
    offrepdispdata.repro_emission_angles = offrepdata.repro_emission_angles
    offrepdispdata.repro_incidence_angle = offrepdata.repro_incidence_angle

    setup_offset_reproject_window(offrepdata, offrepdispdata)
    draw_repro_overlay(offrepdata, offrepdispdata)

    if do_mainloop:
        mainloop()

# The callback for mouse move events on the reprojected image
def callback_repro(x, y, offrepdata, offrepdispdata):
    if offrepdispdata.repro_longitudes is None:
        return

    x = int(x)

    offrepdispdata.label_corot_longitude.config(text=
                        ('%7.3f'%(np.degrees(offrepdispdata.repro_longitudes[x]))))
    offrepdispdata.label_inertial_longitude.config(text=('%7.3f'%(
                      np.degrees(rings_ring_corotating_to_inertial(
                             arguments.corot_type,
                             offrepdispdata.repro_longitudes[x],
                             offrepdata.obs.midtime)))))
    radius = (y*arguments.radius_resolution+
              arguments.ring_radius+
              arguments.radius_inner_delta)
    offrepdispdata.label_radius.config(text='%7.3f'%radius)
    offrepdispdata.label_radial_resolution.config(text=
                ('%7.3f'%offrepdispdata.repro_radial_resolutions[x]))
    offrepdispdata.label_angular_resolution.config(text=
                ('%7.4f'%np.degrees(offrepdispdata.repro_angular_resolutions[x])))
    offrepdispdata.label_phase.config(text=
                ('%7.3f'%(np.degrees(offrepdispdata.repro_phase_angles[x]))))
    offrepdispdata.label_emission.config(text=
                ('%7.3f'%(np.degrees(offrepdispdata.repro_emission_angles[x]))))
    offrepdispdata.label_incidence.config(text=
                ('%7.3f'%(np.degrees(offrepdispdata.repro_incidence_angle))))


#####################################################################################
#
# THE MAIN LOOP
#
#####################################################################################

# Set up per-image logging
_LOGGING_NAME = 'cb.' + __name__
image_logger = logging.getLogger(_LOGGING_NAME)

image_logfile_level = nav.logging.decode_level(arguments.image_logfile_level)
image_log_console_level = nav.logging.decode_level(arguments.image_log_console_level)

nav.logging.set_default_level(nav.logging.min_level(image_logfile_level,
                                                    image_log_console_level))
nav.logging.set_util_flux_level(logging.CRITICAL)

nav.logging.remove_console_handler()
nav.logging.add_console_handler(image_log_console_level)

subprocess_list = []

offrepdispdata = OffRepDispData()

cur_obsid = None
obsid_list = []
image_name_list = []
image_path_list = []
repro_path_list = []
for obsid, image_name, image_path in ring_enumerate_files(arguments):
    offrepdata = OffRepData()
    offrepdata.obsid = obsid
    offrepdata.image_name = image_name
    offrepdata.image_path = image_path

    offrepdata.offset_path = img_to_offset_path(image_path,
                                                arguments.instrument_host)
    offrepdata.repro_path = img_to_repro_path(arguments,
                                              image_path,
                                              arguments.instrument_host,
                                              make_dirs=False)

    offrepdata.subprocess_run = False

    if arguments.profile:
        pr = cProfile.Profile()
        pr.enable()

    offrepdata.image_log_filehander = None

    if arguments.verbose:
        print('Processing', obsid, '/', image_name)

    ret = read_repro(offrepdata.repro_path)
    _update_offrepdata_repro(offrepdata, ret)
    if ret is not None:
        offrepdata.bad_pixel_map = ret['bad_pixel_map']
    if offrepdata.bad_pixel_map is not None:
        offrepdata.bad_pixel_map = offrepdata.bad_pixel_map.copy() # Make writeable

    # Pointing offset
    offset_one_image(offrepdata,
                     arguments.no_auto_offset,
                     arguments.no_update_auto_offset,
                     arguments.recompute_auto_offset)

    # Reprojection
    reproject_one_image(offrepdata,
                        arguments.no_reproject,
                        arguments.no_update_reproject,
                        arguments.recompute_reproject)

    nav.logging.remove_file_handler(offrepdata.image_log_filehandler)

    if arguments.max_subprocesses and offrepdata.subprocess_run:
        run_and_maybe_wait([PYTHON_EXE,
                            RING_REPROJECT_PY] +
                           collect_cmd_line() +
                           [offrepdata.obsid+'/'+offrepdata.image_name])

    # Display offset and reprojection
    if (arguments.display_offset_reproject or
        arguments.display_invalid_offset or
        arguments.display_invalid_reproject):
        display_offset_reproject(offrepdata, offrepdispdata,
                                 arguments.display_invalid_offset,
                                 arguments.display_invalid_reproject,
                                 do_mainloop=not arguments.profile)

    del offrepdata
    offrepdata = None

    if arguments.profile:
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        ps.print_callers()
        print(s.getvalue())
        assert False

while len(subprocess_list):
    for i in range(len(subprocess_list)):
        if subprocess_list[i].poll() is not None:
            del subprocess_list[i]
            break
    if len(subprocess_list):
        time.sleep(1)
