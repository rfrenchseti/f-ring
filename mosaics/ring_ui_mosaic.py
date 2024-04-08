################################################################################
# ring_ui_mosaic.py
#
# Combination of ring reprojections into mosaics, with GUI and without.
################################################################################

import argparse
import colorsys
import numpy as np
import os
import subprocess
import sys

import matplotlib.pyplot as plt

from tkinter import *
from imgdisp import ImageDisp

import julian

from nav.config import (CB_RESULTS_ROOT,
                        PYTHON_EXE)
import nav.logging_setup
from nav.ring_mosaic import (rings_fring_corotating_to_inertial,
                             rings_generate_longitudes,
                             rings_mosaic_add,
                             rings_mosaic_init)
from ring.ring_util import (img_to_repro_path,
                            mosaic_paths,
                            mosaic_png_path,
                            read_mosaic,
                            read_repro,
                            ring_add_parser_arguments,
                            ring_basic_cmd_line,
                            ring_enumerate_files,
                            ring_init,
                            write_mosaic,
                            write_mosaic_pngs,
                            RING_REPROJECT_PY,
                            MosaicData)


command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--verbose --ring-type FMOVIE --all-obsid'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser()

#
# The default behavior is to check the timestamps on the input file and the
# output file and recompute if the output file is out of date.
# Several options change this behavior:
#   --no-xxx: Don't recompute no matter what; this may leave you without an
#             output file at all
#   --no-update: Don't recompute if the output file exists, but do compute
#                if the output file doesn't exist at all
#   --recompute-xxx: Force recompute even if the output file exists and
#                    is current
#


##
## General options
##
parser.add_argument('--allow-exception', dest='allow_exception',
                  action='store_true', default=False,
                  help="Allow exceptions to be thrown")

##
## Options for mosaic creation
##
parser.add_argument('--no-mosaic', action='store_true', default=False,
                    help="Don't compute the mosaic even if we don't have one")
parser.add_argument('--no-update-mosaic', action='store_true', default=False,
                    help="Don't compute the mosaic unless we don't have one")
parser.add_argument('--recompute-mosaic', action='store_true', default=False,
                    help="Recompute the mosaic even if we already have one that is current")
parser.add_argument('--display-mosaic', action='store_true', default=False,
                    help='Display the mosaic')

ring_add_parser_arguments(parser)

arguments = parser.parse_args(command_list)

ring_init(arguments)


class MosaicDispData:
    def __init__(self):
        self.toplevel = None
        self.imdisp = None

################################################################################
#
# MAKE A MOSAIC
#
################################################################################

def _update_mosaicdata(mosaicdata, metadata):
    mosaicdata.radius_resolution = metadata['radius_resolution']
    mosaicdata.longitude_resolution = metadata['longitude_resolution']
    mosaicdata.img = metadata['img']
    mosaicdata.long_antimask = metadata['long_antimask']
    mosaicdata.image_numbers = metadata['image_number']
    mosaicdata.ETs = metadata['time']
    mosaicdata.emission_angles = metadata['mean_emission']
    mosaicdata.incidence_angle = metadata['mean_incidence']
    mosaicdata.phase_angles = metadata['mean_phase']
    mosaicdata.radial_resolutions = metadata['mean_radial_resolution']
    mosaicdata.angular_resolutions = metadata['mean_angular_resolution']
    full_longitudes = rings_generate_longitudes(
                longitude_resolution=np.radians(arguments.longitude_resolution))
    full_longitudes[np.logical_not(mosaicdata.long_antimask)] = -999
    mosaicdata.longitudes = full_longitudes
    mosaicdata.obsid_list = metadata['obsid_list']
    mosaicdata.image_name_list = metadata['image_name_list']
    mosaicdata.image_path_list = metadata['image_path_list']
    mosaicdata.repro_path_list = metadata['repro_path_list']
    mosaicdata.repro_path_list = [
        x.replace('/cdaps-results/fring',
                  CB_RESULTS_ROOT) for x in mosaicdata.repro_path_list]

def check_make_mosaic(mosaicdata, option_no, option_no_update, option_recompute):
    # Input files: image_path_list (includes repro suffix)
    # Output files:
    #  mosaic_data_filename (the basic 2-D array)
    #  mosaic_metadata_filename (obsid_list, image_name_list, image_path_list)
    #  large_png_filename (full size mosaic graphic)
    if arguments.verbose:
        print('Make_mosaic:', mosaicdata.obsid)

    if option_no:  # Just don't do anything
        if arguments.verbose:
            print('Not doing anything because of --no-mosaic')
        return

    if not option_recompute:
        if (os.path.exists(mosaicdata.data_path+'.npy') and
            os.path.exists(mosaicdata.metadata_path) and
            os.path.exists(mosaicdata.full_png_path) and
            os.path.exists(mosaicdata.small_png_path)):
            if option_no_update:
                if arguments.verbose:
                    print('Not doing anything because output files already', end=' ')
                    print('exist and --no-update-mosaic')
                return  # Mosaic file already exists, don't update

            # Find the latest repro time
            max_repro_mtime = 0
            for repro_path in mosaicdata.repro_path_list:
                time_repro = os.stat(repro_path).st_mtime
                max_repro_mtime = max(max_repro_mtime, time_repro)

            if (os.stat(mosaicdata.data_path+'.npy').st_mtime >
                            max_repro_mtime and
                os.stat(mosaicdata.metadata_path).st_mtime >
                            max_repro_mtime and
                os.stat(mosaicdata.full_png_path).st_mtime > max_repro_mtime and
                os.stat(mosaicdata.small_png_path).st_mtime > max_repro_mtime):
                # The mosaic file exists and is more recent than the reprojected
                # images, and we're not forcing a recompute
                if arguments.verbose:
                    print('Not doing anything because output files already', end=' ')
                    print('exist and are current')
                return

    make_mosaic(mosaicdata)

def make_mosaic(mosaicdata):
    print('Making mosaic for', mosaicdata.obsid)

    mosaic_metadata = rings_mosaic_init((arguments.ring_radius+
                                         arguments.radius_inner_delta,
                                         arguments.ring_radius+
                                         arguments.radius_outer_delta),
                                        np.radians(
                                                arguments.longitude_resolution),
                                        arguments.radius_resolution)

    for i, repro_path in enumerate(mosaicdata.repro_path_list):
        repro_metadata = read_repro(repro_path)
        if arguments.instrument_host == 'cassini':
            # For Cassini, zero means invalid or missing pixel in older
            # reprojected files, so convert to -999 which is the current
            # invalid pixel value
            repro_metadata['img'] = repro_metadata['img'].copy()
            repro_metadata['img'][repro_metadata['img'] == 0] = -999
        # repro_good_long_antimask = repro_metadata['long_antimask']
        # repro_num_good_long = np.sum(repro_good_long_antimask)
        # XXX Always use at least 10% of the reprojected image if we use any of it
        resolution_block = 1 # int(repro_num_good_long * 0.1)

        if arguments.verbose:
            print('Adding mosaic data for', repro_path)
        rings_mosaic_add(mosaic_metadata, repro_metadata, i,
                         resolution_block=resolution_block)

    mosaic_metadata['obsid_list'] = mosaicdata.obsid_list
    mosaic_metadata['image_name_list'] = mosaicdata.image_name_list
    mosaic_metadata['image_path_list'] = mosaicdata.image_path_list
    mosaic_metadata['repro_path_list'] = mosaicdata.repro_path_list

    _update_mosaicdata(mosaicdata, mosaic_metadata)

    write_mosaic(mosaicdata.data_path, mosaicdata.img,
                 mosaicdata.metadata_path, mosaic_metadata)

    write_mosaic_pngs(mosaicdata.full_png_path,
                      mosaicdata.small_png_path,
                      mosaicdata.img)

    print('Mosaic saved')

################################################################################
#
# DISPLAY ONE MOSAIC
#
################################################################################

mosaicdispdata = MosaicDispData()

def command_refresh_color(mosaicdata, mosaicdispdata):
    color_sel = mosaicdispdata.var_color_by.get()

    if color_sel == 'none':
        mosaicdispdata.imdisp.set_color_column(0, None)
        return

    minval = None
    maxval = None

    if color_sel == 'relradresolution':
        valsrc = mosaicdata.radial_resolutions
    if color_sel == 'relangresolution':
        valsrc = mosaicdata.angular_resolutions
    elif color_sel == 'relphase':
        valsrc = mosaicdata.phase_angles
    elif color_sel == 'absphase':
        valsrc = mosaicdata.phase_angles
        minval = 0.
        maxval = 180.
    elif color_sel == 'relemission':
        valsrc = mosaicdata.emission_angles
    elif color_sel == 'absemission':
        valsrc = mosaicdata.emission_angles
        minval = 0.
        maxval = 180.
    elif color_sel == 'imageno':
        valsrc = mosaicdata.image_numbers
    elif color_sel == 'imageparity':
        valsrc = mosaicdata.image_numbers & 1
    elif color_sel == 'longitude':
        minval = 0.
        maxval = 360.
        valsrc = np.degrees(
            rings_fring_corotating_to_inertial(mosaicdata.longitudes,
                                               mosaicdata.ETs))

    if minval is None:
        minval = np.min(valsrc[np.where(mosaicdata.longitudes >= 0.)[0]])
        maxval = np.max(valsrc[np.where(mosaicdata.longitudes >= 0.)[0]])

    color_data = np.zeros((mosaicdata.longitudes.shape[0], 3))

    for col in range(len(mosaicdata.longitudes)):
        if mosaicdata.longitudes[col] >= 0.:
            color = colorsys.hsv_to_rgb((1-
                        (float(valsrc[col])-minval)/(maxval-minval))*.66, 1, 1)
            color_data[col,:] = color

    mosaicdispdata.imdisp.set_color_column(0, color_data)

def setup_mosaic_window(mosaicdata, mosaicdispdata):
    mosaicdispdata.toplevel = Tk()
    mosaicdispdata.toplevel.title(
        f'{mosaicdata.obsid} (RR {arguments.radius_resolution:.2f}, '
        f'LR {arguments.longitude_resolution:.2f}, '
        f'RZ {arguments.radial_zoom_amount:d}, '
        f'LZ {arguments.longitude_zoom_amount:d})')

    frame_toplevel = Frame(mosaicdispdata.toplevel)

    # In the mosaic, -999 means bad or missing pixel. It will be masked, but we
    # don't want it to affect calculation of blackpoint.
    clean_img = mosaicdata.img.copy()
    mask = (clean_img == -999)
    clean_img[mask] = 0
    overlay = np.zeros(clean_img.shape + (3,))
    overlay[:, :, 1] = mask * 192
    mosaicdispdata.imdisp = ImageDisp([clean_img], [overlay],
                                      canvas_size=(1024,512),
                                      parent=frame_toplevel, flip_y=True,
                                      shrink_limit=(5,5),
                                      enlarge_limit=(0,2),
                                      one_zoom=False)

    #############################################
    # The control/data pane of the mosaic image #
    #############################################

    gridrow = 0
    gridcolumn = 0

    addon_control_frame = mosaicdispdata.imdisp.addon_control_frame

    label = Label(addon_control_frame, text='Inertial Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    # We make this one fixed-width so that the color-control column stays
    # in one place
    mosaicdispdata.label_inertial_longitude = Label(addon_control_frame,
                                                    text='', anchor='w',
                                                    width=28)
    mosaicdispdata.label_inertial_longitude.grid(row=gridrow,
                                                 column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Co-Rot Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    # We make this one fixed-width so that the color-control column stays
    # in one place
    mosaicdispdata.label_longitude = Label(addon_control_frame, text='',
                                           anchor='w', width=28)
    mosaicdispdata.label_longitude.grid(row=gridrow, column=gridcolumn+1,
                                        sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Radius:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_radius = Label(addon_control_frame, text='')
    mosaicdispdata.label_radius.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Phase:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_phase = Label(addon_control_frame, text='')
    mosaicdispdata.label_phase.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Incidence:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_incidence = Label(addon_control_frame, text='')
    mosaicdispdata.label_incidence.grid(row=gridrow, column=gridcolumn+1,
                                        sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Emission:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_emission = Label(addon_control_frame, text='')
    mosaicdispdata.label_emission.grid(row=gridrow, column=gridcolumn+1,
                                       sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Radial Res:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_radial_resolution = Label(addon_control_frame, text='')
    mosaicdispdata.label_radial_resolution.grid(row=gridrow, column=gridcolumn+1,
                                                sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Angular Res:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_angular_resolution = Label(addon_control_frame, text='')
    mosaicdispdata.label_angular_resolution.grid(row=gridrow, column=gridcolumn+1,
                                                 sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Image:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_image = Label(addon_control_frame, text='')
    mosaicdispdata.label_image.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='OBSID:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_obsid = Label(addon_control_frame, text='')
    mosaicdispdata.label_obsid.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Date:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_date = Label(addon_control_frame, text='')
    mosaicdispdata.label_date.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    gridrow = 0
    gridcolumn = 2

    mosaicdispdata.var_color_by = StringVar()
    refresh_color = lambda: command_refresh_color(mosaicdata, mosaicdispdata)

    label = Label(addon_control_frame, text='Color by:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='None',
                variable=mosaicdispdata.var_color_by,
                value='none', command=refresh_color).grid(row=gridrow,
                                                          column=gridcolumn,
                                                          sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Rel Radial Res',
                variable=mosaicdispdata.var_color_by,
                value='relradresolution',
                command=refresh_color).grid(row=gridrow,
                                            column=gridcolumn,
                                            sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Rel Angular Res',
                variable=mosaicdispdata.var_color_by,
                value='relangresolution',
                command=refresh_color).grid(row=gridrow,
                                            column=gridcolumn,
                                            sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Abs Phase',
                variable=mosaicdispdata.var_color_by,
                value='absphase', command=refresh_color).grid(row=gridrow,
                                                              column=gridcolumn,
                                                              sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Rel Phase',
                variable=mosaicdispdata.var_color_by,
                value='relphase', command=refresh_color).grid(row=gridrow,
                                                              column=gridcolumn,
                                                              sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Abs Emission',
                variable=mosaicdispdata.var_color_by,
                value='absemission',
                command=refresh_color).grid(row=gridrow,
                                            column=gridcolumn,
                                            sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Rel Emission',
                variable=mosaicdispdata.var_color_by,
                value='relemission',
                command=refresh_color).grid(row=gridrow,
                                            column=gridcolumn,
                                            sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Image #',
                variable=mosaicdispdata.var_color_by,
                value='imageno',
                command=refresh_color).grid(row=gridrow,
                                            column=gridcolumn,
                                            sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Image # Parity',
                variable=mosaicdispdata.var_color_by,
                value='imageparity',
                command=refresh_color).grid(row=gridrow,
                                            column=gridcolumn,
                                            sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Abs Inertial Longitude',
                variable=mosaicdispdata.var_color_by,
                value='longitude',
                command=refresh_color).grid(row=gridrow,
                                            column=gridcolumn,
                                            sticky=W)
    mosaicdispdata.var_color_by.set('none')

    gridrow += 1
    button_show_longitudes_command = (lambda mosaicdata=mosaicdata,
                                             mosaicdispdata=mosaicdispdata:
                                             command_show_longitudes(mosaicdata,
                                                                     mosaicdispdata))
    button_show_longitudes = Button(addon_control_frame,
                                    text='Show Repro Longitudes',
                                    command=button_show_longitudes_command)
    button_show_longitudes.grid(row=gridrow, column=gridcolumn)

    gridrow += 1
    button_refresh_mosaic_command = (lambda mosaicdata=mosaicdata,
                                            mosaicdispdata=mosaicdispdata:
                                            command_refresh_mosaic(mosaicdata,
                                                                   mosaicdispdata))
    button_refresh_mosaic = Button(addon_control_frame,
                                   text='Refresh Mosaic',
                                   command=button_refresh_mosaic_command)
    button_refresh_mosaic.grid(row=gridrow, column=gridcolumn)


    callback_mosaic_move_command = (lambda x, y, mosaicdata=mosaicdata:
                                        callback_move_mosaic(x, y, mosaicdata))
    mosaicdispdata.imdisp.bind_mousemove(0, callback_mosaic_move_command)

    callback_mosaic_b1press_command = (lambda x, y, mosaicdata=mosaicdata:
                                    callback_b1press_mosaic(x, y, mosaicdata))
    mosaicdispdata.imdisp.bind_b1press(0, callback_mosaic_b1press_command)

    mosaicdispdata.imdisp.pack(side=LEFT)

    frame_toplevel.pack()


def display_mosaic(mosaicdata, mosaicdispdata):
    if mosaicdata.img is None:
        (mosaicdata.data_path,
         mosaicdata.metadata_path) = mosaic_paths(arguments, mosaicdata.obsid,
                                                  make_dirs=True)

        metadata = read_mosaic(mosaicdata.data_path, mosaicdata.metadata_path)
        _update_mosaicdata(mosaicdata, metadata)

    setup_mosaic_window(mosaicdata, mosaicdispdata)

    mainloop()

# The callback for the Refresh Mosaic button
def command_refresh_mosaic(mosaicdata, mosaicdispdata):
    make_mosaic(mosaicdata)
    # In the mosaic, -999 means bad or missing pixel. It will be masked, but we
    # don't want it to affect calculation of blackpoint.
    clean_img = mosaicdata.img.copy()
    mask = (clean_img == -999)
    clean_img[mask] = 0
    overlay = np.zeros(clean_img.shape + (3,))
    overlay[:, :, 1] = mask * 192
    mosaicdispdata.imdisp.update_image_data([clean_img], [overlay],
                                            recompute_scales=False)

# The callback for mouse move events on the mosaic image
def callback_move_mosaic(x, y, mosaicdata):
    x = int(x)
    if x < 0: return
    if mosaicdata.longitudes[x] < 0:  # Invalid longitude
        mosaicdispdata.label_inertial_longitude.config(text='')
        mosaicdispdata.label_longitude.config(text='')
        mosaicdispdata.label_phase.config(text='')
        mosaicdispdata.label_incidence.config(text='')
        mosaicdispdata.label_emission.config(text='')
        mosaicdispdata.label_radial_resolution.config(text='')
        mosaicdispdata.label_angular_resolution.config(text='')
        mosaicdispdata.label_image.config(text='')
        mosaicdispdata.label_obsid.config(text='')
        mosaicdispdata.label_date.config(text='')
    else:
        mosaicdispdata.label_inertial_longitude.config(text=
                    ('%7.3f'%(np.degrees(rings_fring_corotating_to_inertial(
                                    mosaicdata.longitudes[x],
                                    mosaicdata.ETs[x])))))
        mosaicdispdata.label_longitude.config(text=
                    ('%7.3f'%(np.degrees(mosaicdata.longitudes[x]))))
        mosaicdispdata.label_phase.config(text=
                    ('%7.3f'%(np.degrees(mosaicdata.phase_angles[x]))))
        mosaicdispdata.label_incidence.config(text=
                    ('%7.3f'%(np.degrees(mosaicdata.incidence_angle))))
        mosaicdispdata.label_emission.config(text=
                    ('%7.3f'%(np.degrees(mosaicdata.emission_angles[x]))))
        mosaicdispdata.label_radial_resolution.config(text=
                    ('%7.3f'%mosaicdata.radial_resolutions[x]))
        mosaicdispdata.label_angular_resolution.config(text=
                    ('%7.4f'%np.degrees(mosaicdata.angular_resolutions[x])))
        mosaicdispdata.label_image.config(text=
                    mosaicdata.image_name_list[mosaicdata.image_numbers[x]] +
                    ' ('+str(mosaicdata.image_numbers[x])+')')
        mosaicdispdata.label_obsid.config(text=
                    mosaicdata.obsid_list[mosaicdata.image_numbers[x]])
        mosaicdispdata.label_date.config(text=
            julian.ymdhms_format_from_tai(julian.tai_from_tdb(
                float(mosaicdata.ETs[x])), sep=' '))

    y = int(y)
    if y < 0:
        return

    radius = (y*arguments.radius_resolution+
              arguments.ring_radius+arguments.radius_inner_delta)
    mosaicdispdata.label_radius.config(text = '%7.3f'%radius)

# The command for Mosaic button press - rerun offset/reproject
def callback_b1press_mosaic(x, y, mosaicdata):
    if x < 0: return
    x = int(x)
    if mosaicdata.longitudes[x] < 0:  # Invalid longitude - nothing to do
        return
    image_number = mosaicdata.image_numbers[x]
    subprocess.Popen([PYTHON_EXE, RING_REPROJECT_PY,
                      '--display-offset-reproject',
                      '--no-auto-offset',
                      '--no-reproject',
                      mosaicdata.obsid_list[image_number] + '/' +
                      mosaicdata.image_name_list[image_number]] +
                     ring_basic_cmd_line(arguments))

def command_show_longitudes(mosicdata, mosaicdispdata):
    plt.figure()
    for i, repro_path in enumerate(mosaicdata.repro_path_list):
        repro_metadata = read_repro(repro_path)
        repro_good_long_antimask = repro_metadata['long_antimask']
        scale = int(len(repro_good_long_antimask) / 360)
        for x in range(len(repro_good_long_antimask)):
            if np.any(repro_good_long_antimask[x*scale:(x+1)*scale-1]):
                plt.plot(x, i, '.', color='black', mec='black')
    plt.show()


################################################################################
#
# THE MAIN LOOP
#
################################################################################

nav.logging_setup.set_main_module_name('ring_ui_mosaic')

# Each entry in the list is a tuple of obsid_list, image_name_list,
# image_path_list, repro_path_list
mosaic_list = []

cur_obsid = None
obsid_list = []
image_name_list = []
image_path_list = []
repro_path_list = []
for obsid, image_name, image_path in ring_enumerate_files(arguments):
    repro_path = img_to_repro_path(arguments, image_path,
                                   arguments.instrument_host)

    if cur_obsid is None:
        cur_obsid = obsid
    if cur_obsid != obsid:
        if len(obsid_list) != 0:
            if arguments.verbose:
                print('Queueing obsid', obsid_list[0])
            mosaic_list.append((obsid_list, image_name_list, image_path_list,
                                repro_path_list))
        obsid_list = []
        image_name_list = []
        image_path_list = []
        repro_path_list = []
        cur_obsid = obsid
    if os.path.exists(repro_path):
        obsid_list.append(obsid)
        image_name_list.append(image_name)
        image_path_list.append(image_path)
        repro_path_list.append(repro_path)

# Final mosaic
if len(obsid_list) != 0:
    if arguments.verbose:
        print('Queueing obsid', obsid_list[0])
    mosaic_list.append((obsid_list, image_name_list, image_path_list,
                        repro_path_list))
    obsid_list = []
    image_name_list = []
    image_path_list = []
    repro_path_list = []

for mosaic_info in mosaic_list:
    mosaicdata = MosaicData()
    (mosaicdata.obsid_list, mosaicdata.image_name_list,
     mosaicdata.image_path_list,
     mosaicdata.repro_path_list) = mosaic_info
    mosaicdata.obsid = mosaicdata.obsid_list[0]
    (mosaicdata.data_path,
     mosaicdata.metadata_path) = mosaic_paths(arguments, mosaicdata.obsid,
                                              make_dirs=True)
    mosaicdata.full_png_path = mosaic_png_path(arguments, mosaicdata.obsid,
                                               'full', make_dirs=True)
    mosaicdata.small_png_path = mosaic_png_path(arguments, mosaicdata.obsid,
                                                'small', make_dirs=True)
    check_make_mosaic(mosaicdata, arguments.no_mosaic, arguments.no_update_mosaic,
                      arguments.recompute_mosaic)

if arguments.display_mosaic:
    for mosaic_info in mosaic_list:
        mosaicdata = MosaicData()
        (mosaicdata.obsid_list, mosaicdata.image_name_list,
         mosaicdata.image_path_list,
         mosaicdata.repro_path_list) = mosaic_info
        mosaicdata.obsid = mosaicdata.obsid_list[0]
        (mosaicdata.data_path,
        mosaicdata.metadata_path) = mosaic_paths(arguments, mosaicdata.obsid,
                                                make_dirs=True)
        mosaicdata.full_png_path = mosaic_png_path(arguments, mosaicdata.obsid,
                                                'full', make_dirs=True)
        mosaicdata.small_png_path = mosaic_png_path(arguments, mosaicdata.obsid,
                                                    'small', make_dirs=True)

        display_mosaic(mosaicdata, mosaicdispdata)
