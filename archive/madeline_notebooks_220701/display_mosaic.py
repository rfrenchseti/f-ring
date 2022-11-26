import argparse
import colorsys
import numpy as np
import numpy.ma as ma
import os
import pickle
import subprocess
import sys

import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image
from imgdisp import ImageDisp

import julian

import f_ring_util.f_ring as f_ring

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--verbose'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser()

parser.add_argument(
    '--show-radii', default='',
    help='Comma-separated list of radii to highlight')

f_ring.add_parser_arguments(parser)

arguments = parser.parse_args(command_list)

class MosaicData(object):
    def __init__(self):
        self.img = None

class MosaicDispData:
    def __init__(self):
        self.toplevel = None
        self.imdisp_offset = None
        self.imdisp_repro = None

################################################################################
#
# UPDATE A MOSAIC
#
################################################################################

def update_mosaicdata(mosaicdata, metadata):
    mosaicdata.radius_resolution = metadata['radius_resolution']
    mosaicdata.longitude_resolution = metadata['longitude_resolution']
    mosaicdata.long_mask = metadata['long_mask']
    mosaicdata.image_numbers = metadata['image_numbers']
    mosaicdata.ETs = metadata['ETs']
    mosaicdata.emission_angles = metadata['emission_angles']
    mosaicdata.incidence_angle = metadata['incidence_angle']
    mosaicdata.phase_angles = metadata['phase_angles']
    mosaicdata.resolutions = metadata['resolutions']
    mosaicdata.longitudes = metadata['longitudes']
    mosaicdata.obsid_list = metadata['obsid_list']
    mosaicdata.image_name_list = metadata['image_name_list']
    mosaicdata.image_path_list = metadata['image_path_list']
    mosaicdata.repro_path_list = metadata['repro_path_list']


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

    if color_sel == 'relresolution':
        valsrc = mosaicdata.resolutions
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
    mosaicdispdata.toplevel.title(mosaicdata.obsid)
    frame_toplevel = Frame(mosaicdispdata.toplevel)

    mask_overlay = np.zeros((mosaicdata.img.shape[0], mosaicdata.img.shape[1], 3))
    mask_overlay[ma.getmaskarray(mosaicdata.img), 0] = 1

    if arguments.show_radii:
        for radius_str in arguments.show_radii.split(','):
            radius = float(radius_str)
            radius_pix = int(((radius-arguments.ring_radius+arguments.radius_inner_delta) /
                              arguments.radius_resolution))
            mask_overlay[radius_pix, :, 1] = 1


    mosaicdispdata.imdisp = ImageDisp([mosaicdata.img.data],
                                      overlay_list=[mask_overlay],
                                      canvas_size=(1500,401),
                                      parent=frame_toplevel, flip_y=True,
                                      one_zoom=False, enlarge_limit=(0,2))

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

    label = Label(addon_control_frame, text='Resolution:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_resolution = Label(addon_control_frame, text='')
    mosaicdispdata.label_resolution.grid(row=gridrow, column=gridcolumn+1,
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

    label = Label(addon_control_frame, text='Long EW:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_ew = Label(addon_control_frame, text='')
    mosaicdispdata.label_ew.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Long EW*mu:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_ewmu = Label(addon_control_frame, text='')
    mosaicdispdata.label_ewmu.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    ews = np.sum(mosaicdata.img, axis=0) * mosaicdata.radius_resolution
    ewsmu = ews*np.abs(np.cos(mosaicdata.emission_angles))

    ew_mean = np.mean(ews)
    ew_std = np.std(ews)
    ewmu_mean = np.mean(ewsmu)
    ewmu_std = np.std(ewsmu)
    text_ew_stats = ('%.5f +/- %.5f'%(ew_mean, ew_std))
    text_ewmu_stats = ('%.5f +/- %.5f'%(ewmu_mean, ewmu_std))

    label = Label(addon_control_frame, text='Full EW:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_ew_stats = Label(addon_control_frame, text=text_ew_stats)
    mosaicdispdata.label_ew_stats.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Full EW*mu:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_ewmu_stats = Label(addon_control_frame, text=text_ewmu_stats)
    mosaicdispdata.label_ewmu_stats.grid(row=gridrow, column=gridcolumn+1, sticky=W)
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
    Radiobutton(addon_control_frame, text='Rel Resolution',
                variable=mosaicdispdata.var_color_by,
                value='relresolution',
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
    mosaicdispdata.var_color_by.set('none')

    callback_mosaic_move_command = (lambda x, y, mosaicdata=mosaicdata:
                                        callback_move_mosaic(x, y, mosaicdata))
    mosaicdispdata.imdisp.bind_mousemove(0, callback_mosaic_move_command)

    callback_mosaic_b1press_command = (lambda x, y, mosaicdata=mosaicdata:
                                           callback_b1press_mosaic(x, y, mosaicdata))
    mosaicdispdata.imdisp.bind_b1press(0, callback_mosaic_b1press_command)

    mosaicdispdata.imdisp.pack(side=LEFT)

    frame_toplevel.pack()


ew_limit_phase = 0
ew_range_lower = 0
ew_range_upper = 0

def callback_b1press_mosaic(x, y, mosaicdata):
    global ew_limit_phase, ew_range_lower, ew_range_upper
    if x < 0:
        return
    if ew_limit_phase == 0:
        ew_limit_phase = 1
        ew_range_lower = int(y)
        return
    ew_limit_phase = 0
    ew_range_upper = int(y)
    ew_range_lower, ew_range_upper = (min(ew_range_lower, ew_range_upper),
                                      max(ew_range_lower, ew_range_upper))
    ew_data = (np.sum(mosaicdata.img[ew_range_lower:ew_range_upper+1], axis=0) *
               arguments.radius_resolution)
    radius_lower = int(ew_range_lower*arguments.radius_resolution+
                       arguments.ring_radius+arguments.radius_inner_delta)
    radius_upper = int(ew_range_upper*arguments.radius_resolution+
                       arguments.ring_radius+arguments.radius_inner_delta)
    ew_mean = np.mean(ew_data)
    ew_std = np.std(ew_data)
    plt.plot(np.degrees(mosaicdata.longitudes), ew_data,
             label=f'{radius_lower:d}-{radius_upper:d}={ew_mean:.2f}\u00b1{ew_std:.2f}')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('I/F')
    plt.xlim(0, 360)
    plt.title('Limited range I/F')
    plt.legend()
    plt.tight_layout()
    plt.show()

def display_mosaic(mosaicdata, mosaicdispdata):
    setup_mosaic_window(mosaicdata, mosaicdispdata)
    mainloop()

# The callback for mouse move events on the mosaic image
def callback_move_mosaic(x, y, mosaicdata):
    x = int(x)
    if x < 0:
        return
    ew = np.sum(mosaicdata.img[:, x]) * mosaicdata.radius_resolution
    ewmu = ew*np.abs(np.cos(mosaicdata.emission_angles[x]))
    if ew is ma.masked:  # Invalid longitude
        mosaicdispdata.label_inertial_longitude.config(text='')
        mosaicdispdata.label_longitude.config(text='')
        mosaicdispdata.label_phase.config(text='')
        mosaicdispdata.label_incidence.config(text='')
        mosaicdispdata.label_emission.config(text='')
        mosaicdispdata.label_resolution.config(text='')
        mosaicdispdata.label_image.config(text='')
        mosaicdispdata.label_obsid.config(text='')
        mosaicdispdata.label_date.config(text='')
        mosaicdispdata.label_ew.config(text='')
        mosaicdispdata.label_ewmu.config(text='')
    else:
        mosaicdispdata.label_inertial_longitude.config(text=
                    ('%7.3f'%(np.degrees(f_ring.fring_corotating_to_inertial(
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
        mosaicdispdata.label_resolution.config(text=
                    ('%7.3f'%mosaicdata.resolutions[x]))
        mosaicdispdata.label_image.config(text=
                    mosaicdata.image_name_list[mosaicdata.image_numbers[x]] +
                    ' ('+str(mosaicdata.image_numbers[x])+')')
        mosaicdispdata.label_obsid.config(text=
                    mosaicdata.obsid_list[mosaicdata.image_numbers[x]])
        mosaicdispdata.label_date.config(text=
            julian.ymdhms_format_from_tai(julian.tai_from_tdb(
                float(mosaicdata.ETs[x])), sep=' '))
        mosaicdispdata.label_ew.config(text=('%.5f'%ew))
        mosaicdispdata.label_ewmu.config(text=('%.5f'%ewmu))

    y = int(y)
    if y < 0:
        return

    radius = (y*arguments.radius_resolution+
              arguments.ring_radius+arguments.radius_inner_delta)
    mosaicdispdata.label_radius.config(text = '%7.3f'%radius)


################################################################################
#
# THE MAIN LOOP
#
################################################################################

for obs_id in f_ring.enumerate_obsids(arguments):
    mosaicdata = MosaicData()
    data_path, metadata_path = f_ring.bkgnd_sub_mosaic_paths(arguments, obs_id)
    mosaicdata.obsid = obs_id
    with np.load(data_path) as npz:
        mosaicdata.img = ma.MaskedArray(**npz)
    with open(metadata_path, 'rb') as metadata_fp:
        metadata = pickle.load(metadata_fp, encoding='latin1')
    update_mosaicdata(mosaicdata, metadata)
    display_mosaic(mosaicdata, mosaicdispdata)
