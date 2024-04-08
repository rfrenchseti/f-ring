################################################################################
# ring_ui_bkgnd.py
#
# Background subtraction of mosaics, with GUI and without.
################################################################################


import argparse
import numpy as np
import numpy.ma as ma
import os.path
import sys

from imgdisp import ImageDisp
from tkinter import *

import julian

import ring.ring_model_bkgnd as ring_model_bkgnd
from ring.ring_util import (ring_add_parser_arguments,
                            ring_init,
                            read_bkgnd_metadata,
                            read_mosaic,
                            ring_enumerate_files,
                            write_bkgnd,
                            write_bkgnd_sub_mosaic,
                            write_mosaic_pngs,
                            BkgndData)
import nav.logging_setup
from nav.ring_mosaic import (rings_fring_corotating_to_inertial,
                             rings_generate_longitudes)
# from cb_util_image import *

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = 'ISS_039RF_FMOVIE002_VIMS --display-bkgnd --verbose'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser()

parser.add_argument('--no-bkgnd', action='store_true', default=False,
                    help="Don't compute the background even if we don't have one")
parser.add_argument('--no-update-bkgnd', action='store_true', default=False,
                    help="Don't compute the background unless we don't have one")
parser.add_argument('--recompute-bkgnd', action='store_true', default=False)
parser.add_argument('--display-bkgnd', action='store_true', default=False)
parser.add_argument('--use-default-params', action='store_true', default=False)

ring_add_parser_arguments(parser)

arguments = parser.parse_args(command_list)

ring_init(arguments)

class BkgndDispData:
    """Data used to display background info."""
    def __init__(self):
        self.toplevel = None
        self.mask_overlay = None
        self.imdisp = None
        self.radial_profile_canvas = None
        self.label_longitude = None
        self.label_radius = None
        self.label_phase = None
        self.label_incidence = None
        self.label_emission = None
        self.label_radial_resolution = None
        self.label_angular_resolution = None
        self.label_image = None
        self.label_obsid = None
        self.label_date = None
        self.label_ew = None
        self.label_ewmu = None
        self.label_ew_stats = None
        self.label_ewmu_stats = None
        self.var_row_cutoff = None
        self.scale_row_cutoff = None
        self.var_column_cutoff = None
        self.scale_column_cutoff = None
        self.var_ignore_fraction = None
        self.scale_ignore_fraction = None
        self.var_row_blur = None
        self.scale_row_blur = None
        self.var_ring_lower_limit = None
        self.scale_ring_lower_limit = None
        self.var_ring_upper_limit = None
        self.scale_ring_upper_limit = None
        self.var_inside_background_pixels = None
        self.var_outside_background_pixels = None
        self.scale_inside_background_pixels = None
        self.scale_outside_background_pixels = None
        self.var_degree = None
        self.scale_degree = None
        self.radial_profile_last_x = None
        self.radial_profile_last_y = None


################################################################################
#
# COMPUTE ONE BKGND
#
################################################################################

def _update_bkgnddata(bkgnddata, metadata):
    bkgnddata.radius_resolution = metadata['radius_resolution']
    bkgnddata.longitude_resolution = metadata['longitude_resolution']
    bkgnddata.mosaic_img = metadata['img']
    bkgnddata.long_antimask = metadata['long_antimask']
    bkgnddata.image_numbers = metadata['image_number']
    bkgnddata.ETs = metadata['time']
    bkgnddata.emission_angles = metadata['mean_emission']
    bkgnddata.incidence_angle = metadata['mean_incidence']
    bkgnddata.phase_angles = metadata['mean_phase']
    bkgnddata.radial_resolutions = metadata['mean_radial_resolution']
    bkgnddata.angular_resolutions = metadata['mean_angular_resolution']
    full_longitudes = rings_generate_longitudes(
                longitude_resolution=bkgnddata.longitude_resolution)
    full_longitudes[np.logical_not(bkgnddata.long_antimask)] = -999
    bkgnddata.longitudes = full_longitudes
    bkgnddata.obsid_list = metadata['obsid_list']
    bkgnddata.image_name_list = metadata['image_name_list']
    bkgnddata.image_path_list = metadata['image_path_list']
    bkgnddata.repro_path_list = metadata['repro_path_list']

def _update_metadata(bkgnddata, metadata):
    metadata['radius_resolution'] = bkgnddata.radius_resolution
    metadata['longitude_resolution'] = bkgnddata.longitude_resolution
    metadata['img'] = bkgnddata.mosaic_img
    metadata['long_antimask'] = bkgnddata.long_antimask
    metadata['image_number'] = bkgnddata.image_numbers
    metadata['time'] = bkgnddata.ETs
    metadata['mean_emission'] = bkgnddata.emission_angles
    metadata['mean_incidence'] = bkgnddata.incidence_angle
    metadata['mean_phase'] = bkgnddata.phase_angles
    metadata['mean_radial_resolution'] = bkgnddata.radial_resolutions
    metadata['mean_angular_resolution'] = bkgnddata.angular_resolutions
    full_longitudes = rings_generate_longitudes(
                longitude_resolution=np.radians(arguments.longitude_resolution))
    full_longitudes[np.logical_not(bkgnddata.long_antimask)] = -999
    bkgnddata.longitudes = full_longitudes
    metadata['obsid_list'] = bkgnddata.obsid_list
    metadata['image_name_list'] = bkgnddata.image_name_list
    metadata['image_path_list'] = bkgnddata.image_path_list
    metadata['repro_path_list'] = bkgnddata.repro_path_list

def _update_corrected_mosaic_img(bkgnddata):
    bkgnddata.corrected_mosaic_img = bkgnddata.mosaic_img - bkgnddata.bkgnd_model.data
    bkgnddata.corrected_mosaic_img[bkgnddata.mosaic_img == -999] = -999
    bad_long = np.all(bkgnddata.bkgnd_model_mask, axis=0)
    bkgnddata.corrected_mosaic_img[:, bad_long] = -999
    bkgnddata.long_antimask = bkgnddata.orig_long_antimask & ~bad_long
    full_longitudes = rings_generate_longitudes(
                longitude_resolution=np.radians(arguments.longitude_resolution))
    full_longitudes[np.logical_not(bkgnddata.long_antimask)] = -999
    bkgnddata.longitudes = full_longitudes


#
# Read in mosaic if we don't already have it
#
def read_bkgnd_mosaic(bkgnddata):
    if bkgnddata.mosaic_img is None:
        metadata_filename = bkgnddata.mosaic_metadata_filename
        data_filename = bkgnddata.mosaic_data_filename

        if not os.path.exists(data_filename+'.npy'):
            print(f'{data_filename}.npy not found')
            return False
        if not os.path.exists(metadata_filename):
            print(f'{metadata_filename} not found')
            return False
        print(metadata_filename)
        metadata = read_mosaic(data_filename, metadata_filename)

        _update_bkgnddata(bkgnddata, metadata)
        bkgnddata.orig_long_antimask = bkgnddata.long_antimask

    bkgnddata.bkgnd_model_mask = None # Start with an empty mask
    return True

#
# Check if all the background files already exist
#
def all_bkgnd_files_exist(bkgnddata):
    return (os.path.exists(bkgnddata.bkgnd_model_filename) and
            os.path.exists(bkgnddata.bkgnd_metadata_filename))

#
# Commit the current results
#
def save_bkgnd_results(bkgnddata):
    write_bkgnd(bkgnddata.bkgnd_model_filename,
                bkgnddata.bkgnd_model,
                bkgnddata.bkgnd_metadata_filename,
                bkgnddata)

    write_mosaic_pngs(bkgnddata.full_png_path,
                      bkgnddata.small_png_path,
                      bkgnddata.corrected_mosaic_img)

    if bkgnddispdata.var_ring_lower_limit is not None:
        bkgnddata.ring_lower_limit = bkgnddispdata.var_ring_lower_limit.get()
        bkgnddata.ring_upper_limit = bkgnddispdata.var_ring_upper_limit.get()

    write_bkgnd_sub_mosaic(bkgnddata.bkgnd_sub_mosaic_filename,
                           bkgnddata.corrected_mosaic_img,
                           bkgnddata.bkgnd_sub_mosaic_metadata_filename,
                           bkgnddata)


#
# Compute background for one mosaic, if we don't already have one (based on arguments)
#
def make_bkgnd(bkgnddata, option_no, option_no_update, option_recompute,
               save_results=True, override_defaults=False):
    if arguments.verbose:
        print('** Compute background', bkgnddata.obsid)

    if option_no:  # Just don't do anything - we hope you know what you're doing!
        if arguments.verbose:
            print('Ignored because of --no-bkgnd')
        return

    if not option_recompute:
        if all_bkgnd_files_exist(bkgnddata):
            if option_no_update:
                if arguments.verbose:
                    print('Not doing anything because output files already exist '+
                          'and --no-update-bkgnd')
                return  # Mosaic file already exists, don't update

            max_mosaic_mtime = max(
                os.stat(bkgnddata.mosaic_data_filename+'.npy').st_mtime,
                os.stat(bkgnddata.mosaic_metadata_filename).st_mtime)
            if (os.stat(bkgnddata.bkgnd_model_filename).st_mtime > max_mosaic_mtime and
                os.stat(bkgnddata.bkgnd_metadata_filename).st_mtime > max_mosaic_mtime):
                # The mosaic file exists and is more recent than the reprojected images,
                # and we're not forcing a recompute
                if arguments.verbose:
                    print('Not doing anything because output files already exist '+
                          'and are current')
                return

    if not read_bkgnd_mosaic(bkgnddata):
        return

    if (os.path.exists(bkgnddata.bkgnd_metadata_filename) and
        not arguments.use_default_params and
        not override_defaults):
        read_bkgnd_metadata(bkgnddata.bkgnd_metadata_filename, bkgnddata)

    mosaic_masked = None
    bkgnddata.bkgnd_model = None
    if arguments.verbose:
        print('Computing background model')

    mosaic_masked = ring_model_bkgnd.mask_image(bkgnddata.mosaic_img,
        cutoff_sigmas=bkgnddata.row_cutoff_sigmas,
        ignore_fraction=bkgnddata.row_ignore_fraction,
        row_blur=bkgnddata.row_blur,
        mask_fill_value=-999, debug=True)
    radial_res_scale = 5 / bkgnddata.radius_resolution
    bkgnddata.bkgnd_model = ring_model_bkgnd.model_background(mosaic_masked,
        ring_rows=(int(bkgnddata.ring_lower_limit*radial_res_scale),
                   int(bkgnddata.ring_upper_limit*radial_res_scale)),
        cutoff_sigmas=bkgnddata.column_cutoff_sigmas,
        background_pixels=(int(bkgnddata.column_inside_background_pixels*
                               radial_res_scale),
                           int(bkgnddata.column_outside_background_pixels*
                               radial_res_scale)),
        degree=bkgnddata.degree, debug=True, debug_col=10)
    del mosaic_masked
    mosaic_masked = None

    bkgnddata.bkgnd_model_mask = ma.getmaskarray(bkgnddata.bkgnd_model)
    _update_corrected_mosaic_img(bkgnddata)

    if save_results:
        save_bkgnd_results(bkgnddata)


#####################################################################################
#
# DISPLAY ONE BKGND
#
#####################################################################################

# "Recalculate background" button pressed
def command_recalc_bkgnd(bkgnddata, bkgnddispdata):
    bkgnddata.row_cutoff_sigmas = bkgnddispdata.var_row_cutoff.get()
    bkgnddata.row_ignore_fraction = bkgnddispdata.var_ignore_fraction.get()
    bkgnddata.row_blur = bkgnddispdata.var_row_blur.get()
    bkgnddata.ring_lower_limit = bkgnddispdata.var_ring_lower_limit.get()
    bkgnddata.ring_upper_limit = bkgnddispdata.var_ring_upper_limit.get()
    bkgnddata.column_cutoff_sigmas = bkgnddispdata.var_column_cutoff.get()
    bkgnddata.column_inside_background_pixels = (bkgnddispdata
                                                 .var_inside_background_pixels.get())
    bkgnddata.column_outside_background_pixels = (bkgnddispdata
                                                  .var_outside_background_pixels.get())
    bkgnddata.degree = bkgnddispdata.var_degree.get()
    make_bkgnd(bkgnddata, False, False, True, save_results=False, override_defaults=True)
    _update_corrected_mosaic_img(bkgnddata)

    update_ew_stats(bkgnddata)
    del bkgnddispdata.mask_overlay
    bkgnddispdata.mask_overlay = np.zeros((bkgnddata.mosaic_img.shape[0],
                                           bkgnddata.mosaic_img.shape[1], 3))
    if bkgnddata.bkgnd_model_mask is not None:
        bkgnddispdata.mask_overlay[bkgnddata.bkgnd_model_mask, 0] = 255
        bkgnddispdata.mask_overlay[bkgnddata.bkgnd_model_mask, 1] = 0
        bkgnddispdata.mask_overlay[bkgnddata.bkgnd_model_mask, 2] = 0

    # In the mosaic, -999 means bad or missing pixel. It will be masked, but we
    # don't want it to affect calculation of blackpoint.
    clean_img1 = bkgnddata.mosaic_img.copy()
    mask = (clean_img1 == -999)
    clean_img1[mask] = 0
    bkgnddispdata.mask_overlay[mask, 1] = 192

    clean_img2 = bkgnddata.corrected_mosaic_img.copy()
    mask = (clean_img2 == -999)
    clean_img2[mask] = 0
    overlay = np.zeros((clean_img2.shape[0], clean_img2.shape[1], 3))
    overlay[mask, 1] = 192

    bkgnddispdata.imdisp.update_image_data(
        [clean_img1, clean_img2],
        [bkgnddispdata.mask_overlay, overlay])
    radial_profile_update(bkgnddata, bkgnddispdata, bkgnddata.mosaic_img)

# "Commit changes" button pressed
def command_commit_changes(offrepdata, offrepdispdata):
    save_bkgnd_results(bkgnddata)


#
# The mouse move callback for the mosaic image - plot the radial profile
#
def callback_radial1(x, y, bkgnddata, bkgnddispdata):
    longitude_num = x
    if longitude_num < 0 or longitude_num > bkgnddata.mosaic_img.shape[1]:
        return
    if bkgnddata.bkgnd_model is None:
        return
    bkgnddispdata.radial_profile_last_x = x
    bkgnddispdata.radial_profile_last_y = y
    radial_profile_update(bkgnddata, bkgnddispdata, bkgnddata.mosaic_img)

#
# The mouse move callback for the background-sub image - plot the radial profile
#
def callback_radial2(x, y, bkgnddata, bkgnddispdata):
    longitude_num = x
    if longitude_num < 0 or longitude_num > bkgnddata.mosaic_img.shape[1]:
        return
    if bkgnddata.bkgnd_model is None:
        return
    bkgnddispdata.radial_profile_last_x = x
    bkgnddispdata.radial_profile_last_y = y
    radial_profile_update(bkgnddata, bkgnddispdata, bkgnddata.corrected_mosaic_img)

def radial_profile_update(bkgnddata, bkgnddispdata, mosaic_img):
    if bkgnddispdata.radial_profile_last_x is None:
        return

    longitude_num = int(bkgnddispdata.radial_profile_last_x)

    mosaic_img_min = ma.min(mosaic_img[mosaic_img != -999])
    mosaic_img_max = ma.max(mosaic_img[mosaic_img != -999])
    radial_profile = mosaic_img[:,longitude_num] - mosaic_img_min
    radial_profile_mask = bkgnddata.bkgnd_model_mask[:,longitude_num]
    bkgnd_profile = bkgnddata.bkgnd_model.data[:,longitude_num] - mosaic_img_min
    long_good = bkgnddata.long_antimask[longitude_num]
    bkgnddispdata.radial_profile_canvas.delete('line')
    xsize = float(bkgnddispdata.radial_profile_canvas.cget('width'))
    ysize = float(bkgnddispdata.radial_profile_canvas.cget('height'))
    xscale = float(xsize) / float(radial_profile.shape[0])
    yscale = float(ysize) / float(mosaic_img_max-mosaic_img_min)

    # Plot lower and upper ring limits
    radial_res_scale = 5 / bkgnddata.radius_resolution
    xscale_pix = xscale * radial_res_scale

    bkgnddispdata.radial_profile_canvas.create_line(
        [bkgnddispdata.var_ring_lower_limit.get() * xscale_pix,
         0,
         bkgnddispdata.var_ring_lower_limit.get() * xscale_pix,
         (mosaic_img_max-mosaic_img_min)*yscale],
        fill='green', tags='line')
    bkgnddispdata.radial_profile_canvas.create_line(
        [bkgnddispdata.var_ring_upper_limit.get() * xscale_pix,
         0,
         bkgnddispdata.var_ring_upper_limit.get() * xscale_pix,
         (mosaic_img_max-mosaic_img_min)*yscale],
        fill='green', tags='line')

    # Plot lower and upper background pixel limits
    bkgnddispdata.radial_profile_canvas.create_line(
        [(bkgnddispdata.var_ring_lower_limit.get() -
          bkgnddispdata.var_inside_background_pixels.get()) * xscale_pix,
          0,
         (bkgnddispdata.var_ring_lower_limit.get() -
          bkgnddispdata.var_inside_background_pixels.get()) * xscale_pix,
         (mosaic_img_max-mosaic_img_min)*yscale],
        fill='blue', tags='line')
    bkgnddispdata.radial_profile_canvas.create_line(
        [(bkgnddispdata.var_ring_upper_limit.get() +
          bkgnddispdata.var_outside_background_pixels.get()) * xscale_pix,
          0,
         (bkgnddispdata.var_ring_upper_limit.get() +
          bkgnddispdata.var_outside_background_pixels.get()) * xscale_pix,
         (mosaic_img_max-mosaic_img_min)*yscale],
        fill='blue', tags='line')

    # Plot current mouse Y position
    bkgnddispdata.radial_profile_canvas.create_line(
        [bkgnddispdata.radial_profile_last_y * xscale,
         0,
         bkgnddispdata.radial_profile_last_y * xscale,
         (mosaic_img_max-mosaic_img_min)*yscale],
        fill='cyan', tags='line')
    # Update the radius
    bkgnddispdata.radius = (bkgnddispdata.radial_profile_last_y *
                            bkgnddata.radius_resolution +
                            arguments.ring_radius +
                            arguments.radius_inner_delta)

    if long_good:
        # Plot background model profile
        c_list = []
        for radius in range(bkgnd_profile.shape[0]):
            val = bkgnd_profile[radius]
            cx = float(radius) * xscale
            cy = ysize-1 - val * yscale
            c_list.append(cx)
            c_list.append(cy)
        bkgnddispdata.radial_profile_canvas.create_line(*c_list, fill='red', tags='line')

        # Plot radial profile
        for radius in range(radial_profile.shape[0]-1): # 0 = closest to Saturn
            if radial_profile_mask[radius]:
                color = '#80ff80'
            else:
                color = 'black'
            bkgnddispdata.radial_profile_canvas.create_line(
                [float(radius) * xscale, ysize-1 - radial_profile[radius] * yscale,
                 float(radius+1) * xscale, ysize-1 - radial_profile[radius+1] * yscale],
                fill=color, tags='line', width=2)

    # Update text data
    if bkgnddata.longitudes[longitude_num] < 0:  # Invalid longitude
        bkgnddispdata.label_inertial_longitude.config(text='')
        bkgnddispdata.label_longitude.config(text='')
        bkgnddispdata.label_radius.config(text='')
        bkgnddispdata.label_phase.config(text='')
        bkgnddispdata.label_incidence.config(text='')
        bkgnddispdata.label_emission.config(text='')
        bkgnddispdata.label_radial_resolution.config(text='')
        bkgnddispdata.label_angular_resolution.config(text='')
        bkgnddispdata.label_image.config(text='')
        bkgnddispdata.label_obsid.config(text='')
        bkgnddispdata.label_date.config(text='')
        bkgnddispdata.label_ew.config(text='')
        bkgnddispdata.label_ewmu.config(text='')
    else:
        bkgnddispdata.label_inertial_longitude.config(text=
                    ('%7.3f'%(np.degrees(rings_fring_corotating_to_inertial(
                                              bkgnddata.longitudes[longitude_num],
                                              bkgnddata.ETs[longitude_num])))))
        bkgnddispdata.label_longitude.config(text=
                    ('%7.3f'%(np.degrees(bkgnddata.longitudes[longitude_num]))))
        bkgnddispdata.label_radius.config(text=('%7.3f'%bkgnddispdata.radius))
        bkgnddispdata.label_phase.config(text=('%7.3f'%
                    np.degrees(bkgnddata.phase_angles[longitude_num])))
        bkgnddispdata.label_incidence.config(text=
            ('%7.3f'%np.degrees(bkgnddata.incidence_angle)))
        bkgnddispdata.label_emission.config(text=('%7.3f'%(
                                np.degrees(bkgnddata.emission_angles[longitude_num]))))
        bkgnddispdata.label_radial_resolution.config(text=
            ('%7.3f'%bkgnddata.radial_resolutions[longitude_num]))
        bkgnddispdata.label_angular_resolution.config(text=
            ('%7.4f'%np.degrees(bkgnddata.angular_resolutions[longitude_num])))
        bkgnddispdata.label_image.config(text=
            bkgnddata.image_name_list[bkgnddata.image_numbers[longitude_num]])
        bkgnddispdata.label_obsid.config(text=
            bkgnddata.obsid_list[bkgnddata.image_numbers[longitude_num]])
        bkgnddispdata.label_date.config(text=
            julian.ymdhms_format_from_tai(julian.tai_from_tdb(
                float(bkgnddata.ETs[longitude_num])), sep=' '))
        zero_image = bkgnddata.corrected_mosaic_img.copy()
        zero_image[zero_image == -999] = 0
        ew = np.sum(zero_image[
                        int(bkgnddispdata.var_ring_lower_limit.get() * radial_res_scale):
                        int(bkgnddispdata.var_ring_upper_limit.get() * radial_res_scale)+1,
                        longitude_num]) * bkgnddata.radius_resolution
        ewmu = ew*np.abs(np.cos(bkgnddata.emission_angles[longitude_num]))
        bkgnddispdata.label_ew.config(text=('%.5f'%ew))
        bkgnddispdata.label_ewmu.config(text=('%.5f'%ewmu))

#
# Update EW stats
#
def update_ew_stats(bkgnddata):
    if bkgnddata.bkgnd_model is None:
        bkgnddispdata.label_ew_stats.config(text='N/A')
        bkgnddispdata.label_ewmu_stats.config(text='N/A')
        return
    radial_res_scale = 5 / bkgnddata.radius_resolution
    subimg = bkgnddata.corrected_mosaic_img[
            int(bkgnddispdata.var_ring_lower_limit.get() * radial_res_scale):
            int(bkgnddispdata.var_ring_upper_limit.get() * radial_res_scale)+1,
            bkgnddata.long_antimask]
    subimg[subimg == -999] = 0
    ews = np.sum(subimg, axis=0) * bkgnddata.radius_resolution
    ewsmu = ews*np.abs(np.cos(bkgnddata.emission_angles[bkgnddata.long_antimask]))

    bkgnddata.ew_mean = np.mean(ews)
    bkgnddata.ew_std = np.std(ews)
    bkgnddata.ewmu_mean = np.mean(ewsmu)
    bkgnddata.ewmu_std = np.std(ewsmu)
    bkgnddispdata.label_ew_stats.config(text=
        ('%.5f +/- %.5f'%(bkgnddata.ew_mean, bkgnddata.ew_std)))
    bkgnddispdata.label_ewmu_stats.config(text=
        ('%.5f +/- %.5f'%(bkgnddata.ewmu_mean, bkgnddata.ewmu_std)))

bkgnddispdata = BkgndDispData()

def setup_bkgnd_window(bkgnddata, bkgnddispdata):
    bkgnddispdata.toplevel = Tk()
    bkgnddispdata.toplevel.title(
        f'{bkgnddata.obsid} (RR {arguments.radius_resolution:.2f}, '
        f'LR {arguments.longitude_resolution:.2f}, '
        f'RZ {arguments.radial_zoom_amount:d}, '
        f'LZ {arguments.longitude_zoom_amount:d})')
    frame_toplevel = Frame(bkgnddispdata.toplevel)  # Master frame

    # Background canvas
    bkgnddispdata.mask_overlay = np.zeros((bkgnddata.mosaic_img.shape[0],
                                           bkgnddata.mosaic_img.shape[1], 3))
    if bkgnddata.bkgnd_model_mask is not None:
        bkgnddispdata.mask_overlay[bkgnddata.bkgnd_model_mask, 0] = 1
        bkgnddispdata.mask_overlay[bkgnddata.bkgnd_model_mask, 1] = 0
        bkgnddispdata.mask_overlay[bkgnddata.bkgnd_model_mask, 2] = 0

    # In the mosaic, -999 means bad or missing pixel. It will be masked, but we
    # don't want it to affect calculation of blackpoint.
    clean_img1 = bkgnddata.mosaic_img.copy()
    mask = (clean_img1 == -999)
    clean_img1[mask] = 0
    bkgnddispdata.mask_overlay[mask, 1] = 192

    clean_img2 = bkgnddata.corrected_mosaic_img.copy()
    mask = (clean_img2 == -999)
    clean_img2[mask] = 0
    overlay = np.zeros((clean_img2.shape[0], clean_img2.shape[1], 3))
    overlay[mask, 1] = 192

    bkgnddispdata.imdisp = ImageDisp(
        [clean_img1, clean_img2],
        overlay_list=[bkgnddispdata.mask_overlay, overlay],
        parent=frame_toplevel,
        canvas_size=(1024,512),
        shrink_limit=(None,5),
        one_zoom=False,
        flip_y=True,
        separate_contrast=True)
    bkgnddispdata.imdisp.grid(row=0, column=0)

    # Frame for radial profile and mouse-point info
    frame_rightside = Frame(frame_toplevel)

    bkgnddispdata.radial_profile_canvas = Canvas(frame_rightside, width=512,
                                                 height=512, bg='white')
    bkgnddispdata.radial_profile_canvas.grid(row=0, column=0)

    bkgnddispdata.imdisp.bind_mousemove(0,
        lambda x, y, bkgnddata=bkgnddata, bkgnddispdata=bkgnddispdata:
            callback_radial1(x, y, bkgnddata, bkgnddispdata))
    bkgnddispdata.imdisp.bind_mousemove(1,
        lambda x, y, bkgnddata=bkgnddata, bkgnddispdata=bkgnddispdata:
            callback_radial2(x, y, bkgnddata, bkgnddispdata))

    ######################################################
    # The control/data pane of the radial profile window #
    ######################################################

    gridrow = 0
    gridcolumn = 0

    radial_control_frame = Frame(frame_rightside)

    label = Label(radial_control_frame, text='Inertial Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_inertial_longitude = Label(radial_control_frame, text='')
    bkgnddispdata.label_inertial_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='Co-Rot Long:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_longitude = Label(radial_control_frame, text='')
    bkgnddispdata.label_longitude.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='Radius:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_radius = Label(radial_control_frame, text='')
    bkgnddispdata.label_radius.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='Phase:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_phase = Label(radial_control_frame, text='')
    bkgnddispdata.label_phase.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='Incidence:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_incidence = Label(radial_control_frame, text='')
    bkgnddispdata.label_incidence.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='Emission:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_emission = Label(radial_control_frame, text='')
    bkgnddispdata.label_emission.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='Radial Res:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_radial_resolution = Label(radial_control_frame, text='')
    bkgnddispdata.label_radial_resolution.grid(row=gridrow, column=gridcolumn+1,
                                               sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='Angular Res:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_angular_resolution = Label(radial_control_frame, text='')
    bkgnddispdata.label_angular_resolution.grid(row=gridrow, column=gridcolumn+1,
                                                sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='Image:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_image = Label(radial_control_frame, text='')
    bkgnddispdata.label_image.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='OBSID:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_obsid = Label(radial_control_frame, text='')
    bkgnddispdata.label_obsid.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='Date:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_date = Label(radial_control_frame, text='')
    bkgnddispdata.label_date.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='EW:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_ew = Label(radial_control_frame, text='')
    bkgnddispdata.label_ew.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='EW*mu:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_ewmu = Label(radial_control_frame, text='')
    bkgnddispdata.label_ewmu.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='EW stats:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_ew_stats = Label(radial_control_frame, text='')
    bkgnddispdata.label_ew_stats.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    label = Label(radial_control_frame, text='EW*mu stats:')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.label_ewmu_stats = Label(radial_control_frame, text='')
    bkgnddispdata.label_ewmu_stats.grid(row=gridrow, column=gridcolumn+1, sticky=W)
    gridrow += 1

    radial_control_frame.grid(row=1, column=0, sticky=NW)

    ##################################################
    # The control/data pane of the background window #
    ##################################################

    gridrow = 0
    gridcolumn = 0

    addon_control_frame = bkgnddispdata.imdisp.addon_control_frame

    label = Label(addon_control_frame, text='Row cutoff sigmas')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_row_cutoff = DoubleVar()
    bkgnddispdata.scale_row_cutoff = Scale(addon_control_frame, orient=HORIZONTAL,
                                           resolution=0.1,
                                           from_=0., to=80.,
                                           variable=bkgnddispdata.var_row_cutoff)
    bkgnddispdata.var_row_cutoff.set(bkgnddata.row_cutoff_sigmas)
    bkgnddispdata.scale_row_cutoff.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1

    label = Label(addon_control_frame, text='Column cutoff sigmas')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_column_cutoff = DoubleVar()
    bkgnddispdata.scale_column_cutoff = Scale(addon_control_frame, orient=HORIZONTAL,
                                              resolution=0.1,
                                              from_=0., to=20.,
                                              variable=bkgnddispdata.var_column_cutoff)
    bkgnddispdata.var_column_cutoff.set(bkgnddata.column_cutoff_sigmas)
    bkgnddispdata.scale_column_cutoff.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1

    label = Label(addon_control_frame, text='Row ignore fraction')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_ignore_fraction = DoubleVar()
    bkgnddispdata.scale_ignore_fraction = Scale(addon_control_frame, orient=HORIZONTAL,
                                            resolution=0.001,
                                            from_=0., to_=0.1,
                                            variable=bkgnddispdata.var_ignore_fraction)
    bkgnddispdata.var_ignore_fraction.set(bkgnddata.row_ignore_fraction)
    bkgnddispdata.scale_ignore_fraction.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1

    label = Label(addon_control_frame, text='Row blur')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_row_blur = IntVar()
    bkgnddispdata.scale_row_blur = Scale(addon_control_frame, orient=HORIZONTAL,
                                         resolution=1,
                                         from_=0, to=20,
                                         variable=bkgnddispdata.var_row_blur)
    bkgnddispdata.var_row_blur.set(bkgnddata.row_blur)
    bkgnddispdata.scale_row_blur.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1

    ring_limit_command = (lambda val, bkgnddata=bkgnddata, bkgnddispdata=bkgnddispdata:
                          radial_profile_update(bkgnddata, bkgnddispdata,
                                                bkgnddata.mosaic_img))

    label = Label(addon_control_frame, text='Ring lower limit')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_ring_lower_limit = IntVar()
    bkgnddispdata.scale_ring_lower_limit = Scale(addon_control_frame, orient=HORIZONTAL,
                                             resolution=1,
                                             from_=0, to=600,
                                             variable=bkgnddispdata.var_ring_lower_limit,
                                             command=ring_limit_command)
    bkgnddispdata.var_ring_lower_limit.set(bkgnddata.ring_lower_limit)
    bkgnddispdata.scale_ring_lower_limit.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1

    label = Label(addon_control_frame, text='Ring upper limit')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_ring_upper_limit = IntVar()
    bkgnddispdata.scale_ring_upper_limit = Scale(addon_control_frame, orient=HORIZONTAL,
                                             resolution=1,
                                             from_=0, to=1000,
                                             variable=bkgnddispdata.var_ring_upper_limit,
                                             command=ring_limit_command)
    bkgnddispdata.var_ring_upper_limit.set(bkgnddata.ring_upper_limit)
    bkgnddispdata.scale_ring_upper_limit.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1

    label = Label(addon_control_frame, text='Min inside bkgnd pixels')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_inside_background_pixels = IntVar()
    bkgnddispdata.scale_inside_background_pixels = Scale(
        addon_control_frame, orient=HORIZONTAL, resolution=1,
        from_=0., to=500, variable=bkgnddispdata.var_inside_background_pixels,
        command=ring_limit_command)
    bkgnddispdata.var_inside_background_pixels.set(
        bkgnddata.column_inside_background_pixels)
    bkgnddispdata.scale_inside_background_pixels.grid(row=gridrow,
                                                      column=gridcolumn+1, sticky=N)
    gridrow += 1

    label = Label(addon_control_frame, text='Min outside bkgnd pixels')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_outside_background_pixels = IntVar()
    bkgnddispdata.scale_outside_background_pixels = Scale(
        addon_control_frame, orient=HORIZONTAL, resolution=1,
        from_=0., to=500, variable=bkgnddispdata.var_outside_background_pixels,
        command=ring_limit_command)
    bkgnddispdata.var_outside_background_pixels.set(
        bkgnddata.column_outside_background_pixels)
    bkgnddispdata.scale_outside_background_pixels.grid(row=gridrow,
                                                       column=gridcolumn+1, sticky=N)
    gridrow += 1

    label = Label(addon_control_frame, text='Polynomial degree')
    label.grid(row=gridrow, column=gridcolumn, sticky=W)
    bkgnddispdata.var_degree = IntVar()
    bkgnddispdata.scale_degree = Scale(addon_control_frame, orient=HORIZONTAL,
                                       resolution=1,
                                       from_=1, to=2, variable=bkgnddispdata.var_degree)
    bkgnddispdata.var_degree.set(bkgnddata.degree)
    bkgnddispdata.scale_degree.grid(row=gridrow, column=gridcolumn+1, sticky=N)
    gridrow += 1

    # Recalculate background
    button_recalc_bkgnd_command = (lambda bkgnddata=bkgnddata,
                                    bkgnddispdata=bkgnddispdata:
                                        command_recalc_bkgnd(bkgnddata, bkgnddispdata))
    button_recalc_bkgnd = Button(addon_control_frame, text='Recalc Bkgnd',
                                 command=button_recalc_bkgnd_command)
    button_recalc_bkgnd.grid(row=gridrow, column=gridcolumn+1)
    gridrow += 1

    # Commit changes button
    button_commit_changes_command = (lambda bkgnddata=bkgnddata,
                                      bkgnddispdata=bkgnddispdata:
                                         command_commit_changes(bkgnddata, bkgnddispdata))
    button_commit_changes = Button(addon_control_frame, text='Commit Changes',
                                   command=button_commit_changes_command)
    button_commit_changes.grid(row=gridrow, column=gridcolumn+1)

    frame_rightside.grid(row=0, column=1, sticky=N)
    frame_toplevel.pack()


#
# Display the mosaic and background
#
def display_bkgnd(bkgnddata, bkgnddispdata):
    if arguments.verbose:
        print('** Display background', bkgnddata.obsid)

    if bkgnddata.mosaic_img is None:
        if not read_bkgnd_mosaic(bkgnddata):
            return

    bkgnddata.bkgnd_model = None
    # Background params already set in main loop
    if os.path.exists(bkgnddata.bkgnd_model_filename):
        with np.load(bkgnddata.bkgnd_model_filename) as npz:
            bkgnddata.bkgnd_model = ma.MaskedArray(**npz)
            bkgnddata.bkgnd_model_mask = ma.getmaskarray(bkgnddata.bkgnd_model)
        _update_corrected_mosaic_img(bkgnddata)
    if os.path.exists(bkgnddata.bkgnd_metadata_filename):
        read_bkgnd_metadata(bkgnddata.bkgnd_metadata_filename, bkgnddata)

    setup_bkgnd_window(bkgnddata, bkgnddispdata)
    update_ew_stats(bkgnddata)

    mainloop()

################################################################################
#
# THE MAIN LOOP
#
################################################################################

nav.logging_setup.set_main_module_name('ring_ui_bkgnd')

for obs_id, image_name, full_path in ring_enumerate_files(arguments,
                                                          yield_obsid_only=True):
    bkgnddata = BkgndData(obs_id, arguments)

    bkgnddata.mosaic_img = None
    bkgnddata.row_cutoff_sigmas = 7
    bkgnddata.row_ignore_fraction = 0.01
    bkgnddata.row_blur = 5
    bkgnddata.ring_lower_limit = 50
    bkgnddata.ring_upper_limit = 350
    bkgnddata.column_cutoff_sigmas = 4
    bkgnddata.column_inside_background_pixels = 30
    bkgnddata.column_outside_background_pixels = 30
    bkgnddata.degree = 1

    bkgnddata.obsid = obs_id

    make_bkgnd(bkgnddata, arguments.no_bkgnd, arguments.no_update_bkgnd,
               arguments.recompute_bkgnd)

    if arguments.display_bkgnd:
        display_bkgnd(bkgnddata, bkgnddispdata)
