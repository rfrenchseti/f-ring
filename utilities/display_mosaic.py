import argparse
import colorsys
import numpy as np
import numpy.ma as ma
import os
import sys

import msgpack
import msgpack_numpy

import matplotlib.pyplot as plt
from tkinter import *

PROFILE_AVAILABLE = True
try:
    import cProfile
    import io
    import pstats
except ImportError:
    PROFILE_AVAILABLE = False

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'external'))

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
parser.add_argument(
    '--ew-single-radius', action='store_true', default=False,
    help='Single left click displays EW profile of current radius, otherwise '
         'double click to specify a range')
parser.add_argument(
    '--compute-slope', action='store_true', default=False,
    help='Single left click print radius and longitude, second click includes slope')

if PROFILE_AVAILABLE:
    parser.add_argument(
        '--profile', action='store_true', default=False,
        help='Profile performance'
    )

f_ring.add_parser_arguments(parser)

arguments = parser.parse_args(command_list)

f_ring.init(arguments)


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
    mosaicdata.long_antimask = metadata['long_antimask']
    mosaicdata.image_numbers = metadata['image_number']
    mosaicdata.ETs = metadata['time']
    mosaicdata.emission_angles = np.degrees(metadata['mean_emission'])
    mosaicdata.incidence_angle = np.degrees(metadata['mean_incidence'])
    mosaicdata.phase_angles = np.degrees(metadata['mean_phase'])
    mosaicdata.radial_resolutions = metadata['mean_radial_resolution']
    mosaicdata.angular_resolutions = metadata['mean_angular_resolution']
    mosaicdata.longitudes = np.degrees(metadata['longitudes'])
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

    if color_sel == 'relradresolution':
        valsrc = mosaicdata.radial_resolutions
    elif color_sel == 'relangresolution':
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

    if minval is None:
        minval = np.min(valsrc[np.where(mosaicdata.longitudes >= 0.)[0]])
        maxval = np.max(valsrc[np.where(mosaicdata.longitudes >= 0.)[0]])

    color_data = np.zeros((mosaicdata.longitudes.shape[0], 3))

    for col in range(len(mosaicdata.longitudes)):
        if mosaicdata.longitudes[col] >= 0.:
            if minval == maxval:
                color = 1
            else:
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

    mask_overlay = np.zeros((mosaicdata.img.shape[0], mosaicdata.img.shape[1], 3))
    mask_overlay[ma.getmaskarray(mosaicdata.img), 0] = 1
    unmasked_data = mosaicdata.img.copy()
    unmasked_data[ma.getmaskarray(mosaicdata.img)] = 0

    if arguments.show_radii:
        for radius_str in arguments.show_radii.split(','):
            radius = float(radius_str)
            radius_pix = int(((radius-arguments.ring_radius+arguments.radius_inner_delta) /
                              arguments.radius_resolution))
            mask_overlay[radius_pix, :, 1] = 1


    mosaicdispdata.imdisp = ImageDisp([unmasked_data],
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

    label = Label(addon_control_frame, text='Radial Res:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_rad_resolution = Label(addon_control_frame, text='')
    mosaicdispdata.label_rad_resolution.grid(row=gridrow, column=gridcolumn+1,
                                             sticky=W)
    gridrow += 1

    label = Label(addon_control_frame, text='Angular Res:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    mosaicdispdata.label_ang_resolution = Label(addon_control_frame, text='')
    mosaicdispdata.label_ang_resolution.grid(row=gridrow, column=gridcolumn+1,
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
    Radiobutton(addon_control_frame, text='Rel Radius Resolution',
                variable=mosaicdispdata.var_color_by,
                value='relradresolution',
                command=refresh_color).grid(row=gridrow,
                                            column=gridcolumn,
                                            sticky=W)
    gridrow += 1
    Radiobutton(addon_control_frame, text='Rel Angular Resolution',
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
ew_last_radius = 0
ew_last_longitude = 0

def callback_b1press_mosaic(x, y, mosaicdata):
    global ew_limit_phase, ew_range_lower, ew_range_upper
    global ew_last_radius, ew_last_longitude
    if x < 0:
        return
    if arguments.compute_slope:
        longitude = x * arguments.longitude_resolution
        radius = (y * arguments.radius_resolution +
                  arguments.ring_radius + arguments.radius_inner_delta)
        if ew_limit_phase == 0:
            ew_last_longitude = longitude
            ew_last_radius = radius
            ew_limit_phase = 1
        else:
            slope = abs((longitude-ew_last_longitude) / (radius-ew_last_radius))
            print(f'{ew_last_longitude:7.3f}, {ew_last_radius:10.3f}, '
                  f'{longitude:7.3f}, {radius:10.3f}, {slope:.8f}')
            ew_limit_phase = 0
        return

    if not arguments.ew_single_radius and ew_limit_phase == 0:
        ew_limit_phase = 1
        ew_range_lower = int(y)
        return
    ew_limit_phase = 0
    ew_range_upper = int(y)
    if arguments.ew_single_radius:
        ew_range_upper = ew_range_lower
    ew_range_lower2, ew_range_upper2 = (min(ew_range_lower, ew_range_upper),
                                        max(ew_range_lower, ew_range_upper))
    ew_data = (np.sum(mosaicdata.img[ew_range_lower2:ew_range_upper2+1], axis=0) *
               arguments.radius_resolution)
    radius_lower = int(ew_range_lower2*arguments.radius_resolution+
                       arguments.ring_radius+arguments.radius_inner_delta)
    radius_upper = int(ew_range_upper2*arguments.radius_resolution+
                       arguments.ring_radius+arguments.radius_inner_delta)
    ew_mean = np.mean(ew_data)
    ew_std = np.std(ew_data)
    plt.plot(mosaicdata.longitudes, ew_data,
             label=f'{radius_lower:d}-{radius_upper:d}={ew_mean:.2f}\u00b1{ew_std:.2f}')
    plt.xlabel('Longitude (degrees)')
    plt.xlim(0, 360)
    if arguments.ew_single_radius:
        plt.ylabel('I/F')
        plt.title('Single radius I/F')
    else:
        plt.ylabel('EW (km)')
        plt.title('Limited range EW')
    plt.legend()
    plt.tight_layout()
    plt.show()

def display_mosaic(mosaicdata, mosaicdispdata):
    setup_mosaic_window(mosaicdata, mosaicdispdata)
    if PROFILE_AVAILABLE:
        if arguments.profile:
            return
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
        mosaicdispdata.label_rad_resolution.config(text='')
        mosaicdispdata.label_ang_resolution.config(text='')
        mosaicdispdata.label_image.config(text='')
        mosaicdispdata.label_obsid.config(text='')
        mosaicdispdata.label_date.config(text='')
        mosaicdispdata.label_ew.config(text='')
        mosaicdispdata.label_ewmu.config(text='')
    else:
        mosaicdispdata.label_inertial_longitude.config(text=
                    ('%7.3f'%(f_ring.fring_corotating_to_inertial(
                                            mosaicdata.longitudes[x],
                                            mosaicdata.ETs[x]))))
        mosaicdispdata.label_longitude.config(text=
                    ('%7.3f'%(mosaicdata.longitudes[x])))
        mosaicdispdata.label_phase.config(text=
                    ('%7.3f'%(mosaicdata.phase_angles[x])))
        mosaicdispdata.label_incidence.config(text=
                    ('%7.3f'%(mosaicdata.incidence_angle)))
        mosaicdispdata.label_emission.config(text=
                    ('%7.3f'%(mosaicdata.emission_angles[x])))
        mosaicdispdata.label_rad_resolution.config(text=
                    ('%7.3f'%mosaicdata.radial_resolutions[x]))
        mosaicdispdata.label_ang_resolution.config(text=
                    ('%7.3f'%np.degrees(mosaicdata.angular_resolutions[x])))
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

if PROFILE_AVAILABLE:
    if arguments.profile:
        profile = cProfile.Profile()
        profile.enable()

for obs_id in f_ring.enumerate_obsids(arguments):
    mosaicdata = MosaicData()
    data_path, metadata_path = f_ring.bkgnd_sub_mosaic_paths(arguments, obs_id)
    mosaicdata.obsid = obs_id
    with np.load(data_path) as npz:
        mosaicdata.img = ma.MaskedArray(**npz)
        mosaicdata.img.mask = False
        mosaicdata.img = ma.masked_equal(mosaicdata.img, -999)
    with open(metadata_path, 'rb') as metadata_fp:
        metadata = msgpack.unpackb(metadata_fp.read(),
                                   max_str_len=40*1024*1024,
                                   object_hook=msgpack_numpy.decode)
        if 'mean_resolution' in metadata: # Old format
            metadata['mean_radial_resolution'] = res = metadata['mean_resolution']
            del metadata['mean_resolution']
            metadata['mean_angular_resolution'] = np.zeros(res.shape)
        if 'long_mask' in metadata: # Old format
            metadata['long_antimask'] = metadata['long_mask']
            del metadata['long_mask']
    update_mosaicdata(mosaicdata, metadata)
    display_mosaic(mosaicdata, mosaicdispdata)
    if PROFILE_AVAILABLE and arguments.profile:
            break

if PROFILE_AVAILABLE and arguments.profile:
    profile.disable()
    s = io.StringIO()
    ps = pstats.Stats(profile, stream=s).sort_stats('cumulative')
    ps.print_stats()
    ps.print_callers()
    with open('profile.txt', 'w') as fp:
        fp.write(s.getvalue())
