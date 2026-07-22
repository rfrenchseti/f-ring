################################################################################
# main_clump_find_wavelet.py
#
# Top-level entry point for finding clumps in mosaics.
################################################################################

# Detects Clumps
# Produces Scalograms
# Creates Clump Database
#
# NOTES:
# Update-Clump-Database does not work with Voyager data.

import os
import pickle
import sys

import argparse
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndimage
import scipy.optimize as sciopt
import scipy.interpolate as interp

import julian

import clump_bandpass_filter
import clump_gaussian_fit
import clump_util
import f_ring_util.f_ring as f_ring
import wavelet_util

#===============================================================================
#
# Command line processing
#
#===============================================================================

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = [
                # '--all-obsid',
#                'ISS_041RF_FMOVIE001_VIMS',
                '--scale-min', '3.5', '--scale-max', '80',
                '--clump-size-min', '3.5', '--clump-size-max', '40.0',

# Voyager
#                '--voyager',

# Cassini normal
#                '--ignore-voyager'

# Cassini downsampled
                '--ignore-voyager',
                #'--downsample'

                # '--replace-clump-database',

#                '--plot-scalogram',
#                '--color-contours',
#                '--prefilter',
#                'V1I'
                ]

#    cmd_line = ['--prefilter',
#                '--scale-min', '3', '--scale-max', '20',
#                '--clump-size-min', '2.1', '--clump-size-max', '49.9',
##                '--ignore-bad-obsids',
##                '--plot-scalogram',
#                '--replace-clump-database',
#                '--voyager'
#
#                ]

parser = argparse.ArgumentParser()

parser.add_argument('--voyager', dest='voyager', action='store_true', default=False)
parser.add_argument('--ignore-voyager', dest='ignore_voyager', action='store_true', default=False)
parser.add_argument('--downsample', dest='downsample', action='store_true', default=False)
parser.add_argument('--scale-min', type=float, dest='scale_min', default=1.,
                help='The minimum scale (in full-width degrees) to analyze with wavelets')
parser.add_argument('--scale-max', type=float, dest='scale_max', default=30.,
                help='The maximum scale (in full-width degrees) to analyze with wavelets')
parser.add_argument('--scale-step', type=float, dest='scale_step', default=0.1,
                help='The scale step (in full-width degrees) to analyze with wavelets')
parser.add_argument('--clump-size-min', type=float, dest='clump_size_min', default=5.,
                    help='The minimum clump size (in full-width degrees) to detect')
parser.add_argument('--clump-size-max', type=float, dest='clump_size_max', default=15.,
                    help='The maximum clump size (in full-width degrees) to detect')
parser.add_argument('--prefilter', dest='prefilter', action='store_true', default=False,
                    help='Prefilter the data with a bandpass filter')
parser.add_argument('--plot-scalogram', dest='plot_scalogram',
                    action='store_true', default=False,
                    help='Plot the scalogram and EW data')
parser.add_argument('--update-clump-database', dest='update_clump_database',
        action='store_true', default=False,
        help='Read in the old clump database and update it by replacing selected OBSIDs')
parser.add_argument('--replace-clump-database', dest='replace_clump_database',
                    action='store_true', default=False,
                    help='Start with a fresh clump database')
parser.add_argument('--dont-use-fft', dest='dont_use_fft',
                    action='store_true', default=False,
                    help='Use a non-FFT version of CWT')
parser.add_argument('--color-contours', dest='color_contours',
                    action='store_true', default=False,
                    help='Use colored contours in scalogram')

f_ring.add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)

#===============================================================================
#
# Helper functions
#
#===============================================================================

def smooth_ew(ew_data, long_res_deg, smooth_deg):
    """Perform a moving average taking masked entries into account."""
    smooth_pix = int(smooth_deg // long_res_deg // 2)
    smoothed_ew = ma.zeros(ew_data.shape[0])

    if not ma.any(ew_data.mask):
        for n in range(ew_data.shape[0]):
            smoothed_ew[n] = ma.mean(ew_data[max(n-smooth_pix, 0):
                                             min(n+smooth_pix+1, ew_data.shape[0])])
    else:
        for n in range(ew_data.shape[0]):
            if ew_data[n] is ma.masked:
                smoothed_ew[n] = ma.masked
            else:
                smoothed_ew[n] = ma.mean(ew_data[max(n-smooth_pix, 0):
                                                 min(n+smooth_pix+1, ew_data.shape[0])])

    return smoothed_ew

def downsample_ew(ew_data, mosaic_data):
    assert False  # Not currently used
    src_mask = ma.getmaskarray(ew_data)
    (longitudes, resolutions, mosaic_image_numbers,
     ETs, emission_angles, mosaic_incidence_angles,
     phase_angles) = mosaic_data
    longitudes = longitudes.view(ma.MaskedArray)
    longitudes.mask = src_mask
    resolutions = resolutions.view(ma.MaskedArray)
    resolutions.mask = src_mask
    mosaic_image_numbers = mosaic_image_numbers.view(ma.MaskedArray)
    mosaic_image_numbers.mask = src_mask
    ETs = ETs.view(ma.MaskedArray)
    ETs.mask = src_mask
    emission_angles = emission_angles.view(ma.MaskedArray)
    emission_angles.mask = src_mask
    mosaic_incidence_angles = mosaic_incidence_angles.view(ma.MaskedArray)
    mosaic_incidence_angles.mask = src_mask
    phase_angles = phase_angles.view(ma.MaskedArray)
    phase_angles.mask = src_mask

    bin = int(len(ew_data)/720.)
    new_len = int(len(ew_data)/bin)
    assert new_len == 720
    new_arr = ma.zeros(new_len)  # ma.zeros automatically makes a masked array
    src_mask = ma.getmaskarray(ew_data)
    for i in range(new_len):
        # Point i averages the semi-open interval [i*bin, (i+1)*bin)
        # ma.mean does the right thing - if all the values are masked,
        # the result is also masked
        new_arr[i] = ma.mean(ew_data[i*bin:(i+1)*bin])
        longitudes[i] = ma.min(longitudes[i*bin:(i+1)*bin])
        resolutions[i] = ma.max(resolutions[i*bin:(i+1)*bin])
        mosaic_image_numbers[i] = ma.min(mosaic_image_numbers[i*bin:(i+1)*bin])
        ETs[i] = ma.min(ETs[i*bin:(i+1)*bin])
        emission_angles[i] = ma.min(emission_angles[i*bin:(i+1)*bin])
        mosaic_incidence_angles[i] = ma.min(mosaic_incidence_angles[i*bin:(i+1)*bin])
        phase_angles[i] = ma.min(phase_angles[i*bin:(i+1)*bin])
    mosaic_data = (longitudes, resolutions, mosaic_image_numbers,
     ETs, emission_angles, mosaic_incidence_angles,
     phase_angles)
    return new_arr, mosaic_data

def adjust_ew_for_zero_phase(ew_data, phase_angles, emission_angles, incidence_angles):
    return ew_data # XXX
    new_ew_data = f_ring.compute_corrected_ew(
                ew_data * f_ring.compute_mu(emission_angles),
                emission_angles, np.mean(incidence_angles))
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    plt.plot(ew_data* clump_util.compute_mu(emission_angles), color='black')
#    plt.plot(new_ew_data, color='red')
    for i in range(len(ew_data)):
        ratio = (f_ring.clump_phase_curve(0) /
                 f_ring.clump_phase_curve(phase_angles[i]))
#        print phase_angles[i], ratio
        new_ew_data[i] *= ratio
#    plt.plot(new_ew_data, color='green')
#    plt.show()
    return new_ew_data


def detect_local_maxima(arr):
    """Takes an array and detects the peaks using the local maximum filter."""
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    # Modified to detect maxima instead of minima
    # Define a connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = ndimage.generate_binary_structure(len(arr.shape), 2)
    # Apply the local minimum filter; all locations of minimum value in their
    # neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_max = ndimage.maximum_filter(arr, footprint=neighborhood) == arr
    # local_max is a mask that contains the peaks we are looking for, but also
    # the background. In order to isolate the peaks we must remove the background
    # from the mask.
    #
    # We create the mask of the background
    background = (arr==0)
    #
    # A little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = ndimage.binary_erosion(background, structure=neighborhood,
                                               border_value=1)
    # We obtain the final mask, containing only peaks, by removing the background
    # from the local_min mask
    detected_maxima = local_max & ~eroded_background
    return np.where(detected_maxima)

def find_clumps_internal(ew_data, longitudes, obs_id, metadata, wavelet_type='SDG'):
#     if arguments.downsample:
# #        ew_data = downsample_ew(ew_data)
#         long_res_deg = 360./len(ew_data)
#         longitudes = np.arange(len(ew_data))*long_res_deg
#         tripled_ew_data = np.tile(ew_data, 3)
#         tripled_ew_mask = ma.getmaskarray(tripled_ew_data)
#         tripled_ew_data.mask = tripled_ew_mask
#     else:

    # We have to triple the data or else the edges get screwed up
    tripled_ew_data = np.tile(ew_data, 3)

    orig_ew_start_idx = ew_data.size
    orig_ew_end_idx = ew_data.size*2
    orig_ew_start_long = 360.
    orig_ew_end_long = 720.

    clump_db_entry = clump_util.ClumpDBEntry()
    clump_db_entry.obsid = obs_id
    clump_db_entry.ew_data = ew_data # Smoothed and normalized

    if arguments.voyager:
        clump_db_entry.et = ETs[0]
        clump_db_entry.resolution_min = None
        clump_db_entry.resolution_max = None
        clump_db_entry.emission_angle = None
        clump_db_entry.incidence_angle = None
        clump_db_entry.phase_angle = None
        clump_db_entry.et_min = ETs[0]
        clump_db_entry.et_max = ETs[0]
        clump_db_entry.et_min_longitude = longitudes[0]
        clump_db_entry.et_max_longitude = longitudes[-1]
        clump_db_entry.smoothed_ew = None
    else:
        ew_mask = ma.getmaskarray(ew_data)
        clump_db_entry.et = np.mean(ETs[~ew_mask])
        clump_db_entry.resolution_min = np.min(resolutions[~ew_mask])
        clump_db_entry.resolution_max = np.max(resolutions[~ew_mask])
        clump_db_entry.emission_angle = np.mean(emission_angles[~ew_mask])
        clump_db_entry.incidence_angle = np.mean(incidence_angle)
        clump_db_entry.phase_angle = np.mean(phase_angles[~ew_mask])

        min_et = 1e38
        max_et = 0
        for idx in range(len(longitudes)):
            if longitudes[idx] >= 0:
                if ETs[idx] < min_et:
                    min_et = ETs[idx]
                    min_et_long = longitudes[idx]
                if ETs[idx] > max_et:
                    max_et = ETs[idx]
                    max_et_long = longitudes[idx]
                else:
                    max_et_long = 0.0
                    min_et_long = 0.0
        clump_db_entry.et_min = min_et
        clump_db_entry.et_max = max_et
        clump_db_entry.et_min_longitude = min_et_long
        clump_db_entry.et_max_longitude = max_et_long

    # Create the scales at which we wish to do the analysis
    # The scale is the half-width of the wavelet
    # The user gives us the scale in degrees, but we need to convert it to indices
    scale_min = arguments.scale_min/2.
    scale_max = arguments.scale_max/2.
    scale_step = arguments.scale_step/2.
    wavelet_scales = np.arange(scale_min  / long_res_deg,
                               scale_max  / long_res_deg+0.0001,
                               scale_step / long_res_deg)

    # Initialize the mother wavelet
    print(f'Starting CWT process: {wavelet_type}')
    if wavelet_type == 'SDG':
        mother_wavelet = wavelet_util.SDGWavelet(tripled_ew_data.size, wavelet_scales,
                                                 sampf=long_res_deg)
    if wavelet_type == 'FDG':
        mother_wavelet = wavelet_util.FDGWavelet(tripled_ew_data.size, wavelet_scales,
                                                 sampf=long_res_deg)
    # Perform the continuous wavelet transform
    if arguments.dont_use_fft:
        wavelet = mother_wavelet.cwt_nonfft(tripled_ew_data,
                                       startx=ew_data.size+205/long_res_deg, # XXX Huh?
                                       endx=ew_data.size+215/long_res_deg)
    else:
        wavelet = mother_wavelet.cwt(tripled_ew_data)

    # Find the clumps

    # First find the local maxima
    tripled_xform = wavelet.wt.real
    xform_maxima_scale_idx, xform_maxima_long_idx = detect_local_maxima(tripled_xform)

    def fit_wavelet_func(params, wavelet, data):
        base, wscale = params
        return np.sqrt(np.sum((wavelet * wscale + base - data)**2))

    # For each found clump, fit the wavelet to it and see if it's OK

    clump_list = []

    print('Starting clump fitting / wavelet process')
    for maximum_scale_idx, maximum_long_idx in zip(xform_maxima_scale_idx,
                                                   xform_maxima_long_idx):
        longitude = maximum_long_idx * long_res_deg
        scale = wavelet_scales[maximum_scale_idx] * long_res_deg * 2 # Full-width degrees
        if arguments.clump_size_max >= scale >= arguments.clump_size_min:
            long_start_deg = longitude - scale/2. # Left side of positive part of wavelet
            long_end_deg = longitude + scale/2.   # Right side
            if longitude < orig_ew_start_long or longitude >= orig_ew_end_long:
                # If the clump is more than half out of the center EW range,
                # we don't care about it because it will be picked up on the other side
                continue
            if not arguments.voyager:
                if tripled_ew_data[maximum_long_idx] is ma.masked:
                    # The clump is centered on masked data - don't trust it
                    continue
            long_start_idx = int(round(long_start_deg / long_res_deg))
            long_end_idx = int(round(long_end_deg / long_res_deg))
            if (tripled_ew_data[
                    long_start_idx -
                    int(wavelet_scales[maximum_scale_idx]/8)] is ma.masked or
                tripled_ew_data[
                    long_end_idx +
                    int(wavelet_scales[maximum_scale_idx]/8)] is ma.masked):
                # We don't want clumps on the edges of the masked area
                continue
            # Get the master wavelet shape
            mexhat = mother_wavelet._wavelet_vals[maximum_scale_idx].real
            mh_start_idx = int(mexhat.size/2 - wavelet_scales[maximum_scale_idx])
            mh_end_idx   = int(mexhat.size/2 + wavelet_scales[maximum_scale_idx])
            # Extract just the positive part
            restr_mexhat = mexhat[mh_start_idx:mh_end_idx+1]
            extracted_data = tripled_ew_data[long_start_idx:
                                             long_start_idx+restr_mexhat.size]
            mexhat_data, residual, array, *_ = sciopt.fmin_powell(
                fit_wavelet_func, (1., 1.), args=(restr_mexhat, extracted_data),
                disp=False, full_output=True)
            mexhat_base, mexhat_height = mexhat_data
            if mexhat_height < 0:
                continue
            longitude -= orig_ew_start_long
            clump = clump_util.ClumpData()
            clump.clump_db_entry = clump_db_entry
            clump.longitude = longitude
            clump.scale = scale
            clump.longitude_idx = maximum_long_idx-orig_ew_start_idx
            clump.scale_idx = int(wavelet_scales[maximum_scale_idx]*2.)
            clump.mexhat_base = mexhat_base
            clump.mexhat_height = mexhat_height
            clump.abs_height = clump.mexhat_height*mexhat[mexhat.size//2]
            clump.max_long = maximum_long_idx
            clump.matched = False
            clump.residual = residual * long_res_deg
            clump.wave_type = wavelet_type

            (clump.fit_left_deg, clump.fit_right_deg,
             clump.fit_width_idx, clump.fit_width_deg,
             clump.fit_height, clump.int_fit_height,
             clump.g_center, clump.g_sigma,
             clump.g_base, clump.g_height) = clump_gaussian_fit.refine_fit(
                clump, ew_data, arguments.voyager, arguments.downsample)

            print(f'Clump Long {longitude:6.2f} Width {scale:5.2f} '
                  f'Base {mexhat_base:7.4f} Height {residual:8.4f}   '
                  f'Clump Left {clump.fit_left_deg:6.2f} Right {clump.fit_right_deg:6.2f}'
                  f' Center {clump.g_center:6.2f} Sigma {clump.g_sigma:6.2f}')

            fit_width = (clump.fit_right_deg - clump.fit_left_deg) % 360
            if fit_width < arguments.clump_size_min or fit_width > arguments.clump_size_max:
                print(f'WIDTH ({fit_width:7.4f}) OUT OF RANGE - IGNORING')
                continue

            if (abs(clump.fit_left_deg-clump.g_center) <= .5 or
                abs(clump.fit_right_deg-clump.g_center) <= 0.5):
                ratio = 100
            else:
                ratio = (((clump.fit_right_deg-clump.g_center)%360) /
                         ((clump.g_center-clump.fit_left_deg)%360))
            if ratio < 1:
                ratio = 1/ratio
            if ratio > 4:
                print('EXCESSIVE ASYMMETRY - IGNORING')
                continue

            found_dup = False
            for old_clump in clump_list:
                if (old_clump.fit_left_deg == clump.fit_left_deg and
                    old_clump.fit_right_deg == clump.fit_right_deg):
                    found_dup = True
                    break
            if found_dup:
                print('DUPLICATE - IGNORING')
                continue

            # Calculate clump sigma height
            profile_sigma = ma.std(ew_data)
            clump.clump_sigma = clump.abs_height / profile_sigma
            clump.fit_sigma = clump.fit_height / profile_sigma
            clump_list.append(clump)

    clump_db_entry.clump_list = clump_list

    wavelet_data = (wavelet, mother_wavelet, wavelet_scales)
    print('Clump finding complete')
    return (clump_db_entry, wavelet_data)

def select_clumps(arguments, ew_data, longitudes, obs_id, metadata, clump_database):
    print('Starting SDG')
    sdg_clump_db_entry, sdg_wavelet_data = find_clumps_internal(
        ew_data, longitudes, obs_id, metadata, wavelet_type='SDG')
    print('Starting FDG')
    fdg_clump_db_entry, fdg_wavelet_data = find_clumps_internal(
        ew_data, longitudes, obs_id, metadata, wavelet_type='SDG') # XXX

    sdg_list = sdg_clump_db_entry.clump_list
    fdg_list = fdg_clump_db_entry.clump_list

    print('Choosing the best wavelets')
    new_clump_list = []
    longitude_list = []
    for n, clump_s in enumerate(sdg_list):
        f_match_list = []
        s_match_list = [clump_s]
        for clump_f in fdg_list:
            if ((clump_s.longitude - 2.0 <= clump_f.longitude <=
                 clump_s.longitude + 2.0) and
                (0.5*clump_s.scale <= clump_f.scale <=
                 2.5*clump_s.scale)):
                f_match_list.append(clump_f)
                clump_f.matched = True
                clump_s.matched = True
        if not clump_s.matched:
            continue
        f_res = 999999.
        s_res = 999999.
        for f_clump in f_match_list:
            if f_clump.residual < f_res:
                f_res = f_clump.residual
                best_f = f_clump
        for s_clump in s_match_list:
            if s_clump.residual < s_res:
                s_res = s_clump.residual
                best_s = s_clump

        if best_f.residual < best_s.residual:
            new_clump_list.append(best_f)
            longitude_list.append(best_f.longitude)
        else:
            new_clump_list.append(best_s)
            longitude_list.append(best_s.longitude)


    # Check for any left-over clumps that weren't matched.
    # Need to make sure they're not at the same longitudes as clumps that
    # were already matched. This causes double chains to be made -
    # making for a difficult time in the GUI selection process
    if arguments.voyager or arguments.downsample:
        range = 0.5
    else:
        range = 0.04

    for clump_f in fdg_list:
        if not clump_f.matched:
            for longitude in longitude_list:
                if longitude - range <= clump_f.longitude <= longitude + range:
                    clump_f.matched = True
                    break
            if clump_f.matched:
                continue  # Already matched, go to the next f clump
            new_clump_list.append(clump_f)

    for clump_s in sdg_list:
        if not clump_s.matched:
            for longitude in longitude_list:
                if longitude - range <= clump_s.longitude <= longitude + range:
                    clump_s.matched = True
                    break  # Already matched, go to the next s clump
            if clump_s.matched:
                break
            else:
                new_clump_list.append(clump_s)

    new_clump_list.sort(key=lambda x: x.g_center)

    for clump in new_clump_list:
        print(f'Final clump GS {clump.g_center:6.2f} FLD {clump.fit_left_deg:6.2f} '
              f'FRD {clump.fit_right_deg:6.2f}')

    # It doesn't matter which clump_db_entry we choose since the metadata is the same
    # for both. The only differences are the clumps.
    fdg_clump_db_entry.clump_list = new_clump_list
    for clump in new_clump_list:
        clump.clump_db_entry = fdg_clump_db_entry
    clump_database[obs_id] = fdg_clump_db_entry

    return (sdg_wavelet_data, fdg_wavelet_data)

def plot_scalogram(sdg_wavelet_data, fdg_wavelet_data, clump_db_entry):
    """Plot the scalogram for one clump."""
    ew_data = clump_db_entry.ew_data
    clump_list = clump_db_entry.clump_list
    obs_id = clump_db_entry.obsid

    orig_ew_start_idx = ew_data.size
    orig_ew_end_idx = ew_data.size*2
    orig_ew_start_long = 360.
    orig_ew_end_long = 720.

    longitudes = np.arange(ew_data.size)*long_res_deg

    for wavelet_form in ('SDG', 'FDG'):
        if wavelet_form == 'SDG':
            wavelet, mother_wavelet, wavelet_scales = sdg_wavelet_data
        else:
            wavelet, mother_wavelet, wavelet_scales = fdg_wavelet_data

        # .real would need to be changed if we use a complex wavelet
        xform = wavelet.wt[:,orig_ew_start_idx:orig_ew_end_idx].real

        scales_axis = wavelet_scales * long_res_deg * 2  # Full-width degrees for plotting

        fig = plt.figure()

        # Make the contour plot of the wavelet transform
        ax1 = fig.add_subplot(311)
        if arguments.color_contours:
            ax1.contourf(longitudes, scales_axis, xform, 256)
        else:
            ct = ax1.contour(longitudes, scales_axis, xform, 32)
            ax1.clabel(ct, inline=1, fontsize=8)
        ax1.set_ylim((scales_axis[0], scales_axis[-1]))
        ax1.set_xlim((longitudes[0], longitudes[-1]))
        ax1.set_title( wavelet_form + ' Scalogram '+obs_id)
        ax1.set_ylabel('Full width scale (deg)')

        # Make the data plot with overlaid detected clumps
        ax2 = fig.add_subplot(312, sharex=ax1)
        ax2.plot(longitudes, ew_data)

        tripled_longitudes = np.append(longitudes-360.,
                                       np.append(longitudes, longitudes+360.))

        for clump_data in clump_list:
            if clump_data.wave_type == 'SDG':
                wavelet_color = 'red'
            if clump_data.wave_type == 'FDG':
                wavelet_color = 'green'
            clump_util.plot_single_clump(ax2, ew_data, clump_data, 0., 360.,
                                         color=wavelet_color)

        ax2.set_xlim((longitudes[0], longitudes[-1]))
        ax2.set_ylim(ma.min(ew_data), ma.max(ew_data))
        ax2.grid()
        # align time series fig with scalogram fig
        t = ax2.get_position()
        ax2pos = t.get_points()
        ax2pos[1][0] = ax1.get_position().get_points()[1][0]
        t.set_points(ax2pos)
        ax2.set_position(t)
        ax2.set_xlabel('Longitude (deg)')

        # Make the wavelet spectrogram
        spectrum = wavelet.get_wps()
        ax3 = fig.add_subplot(313)
        ax3.plot(scales_axis, spectrum)

        ax3.set_xlim((scales_axis[0], scales_axis[-1]))
        # ax2.set_ylim(ma.min(ew_data), ma.max(ew_data))
        ax3.set_xlabel('Full width scale (deg)')
        ax3.set_ylabel('Spectral Intensity')

    plt.show()


#===============================================================================
#
# MAIN LOOP
#
#===============================================================================

# f_ring_util.ring_init(arguments)

clump_database = {}

if arguments.voyager:

    for root, dirs, files in os.walk(f_ring.VOYAGER_PATH):

        for file in files:
            if '.STACK' in file:
                filename = file[:-6]
                (ew_data_path, ew_mask_path) = f_ring.ew_paths(arguments, filename)

                ew_data = np.load(ew_data_path +'.npy')
                data_path, metadata_path, large_png_path, small_png_path = f_ring.mosaic_paths(arguments, filename)

                metadata_fp = open(metadata_path, 'rb')
                metadata = pickle.load(metadata_fp)
                metadata_fp.close()

                (longitudes, resolutions, image_numbers,
                 ETs, emission_angles, incidence_angles,
                 phase_angles) = metadata

                print('VOYAGER FILE', filename, '#LONG', len(longitudes))

                if arguments.prefilter:
                    ew_data = bandpass_filter.fft_filter(ew_data, long_res_deg,
                                                         obs_id, plot=False)
                    ew_data = ew_data.real
                    assert False

                ew_data = ew_data.view(ma.MaskedArray)
                ew_data.mask = False
                ew_data = adjust_ew_for_zero_phase(ew_data, phase_angles, emission_angles, incidence_angles)

                if filename[:2] == 'V1':
                    ew_data *= 1.34
                elif filename[:2] == 'V2':
                    ew_data *= 0.97
                else:
                    assert False

                # We want to ignore any features < 3 degrees
                ew_data = smooth_ew(ew_data, long_res_deg, 3.0)

                ew_data = ew_data.view(ma.MaskedArray)
                ew_data.mask = False
                (sdg_wavelet, fdg_wavelet) = select_clumps(arguments, ew_data,
                                                           longitudes, filename,
                                                           metadata, clump_database)

                if arguments.plot_scalogram:
                    plot_scalogram(sdg_wavelet, fdg_wavelet, clump_database[filename] )

else:
    if arguments.update_clump_database:
        # Read the old database first so we can update it
        clump_database_path, clump_chains_path = f_ring.clumpdb_paths(arguments)
        with open(clump_database_path, 'rb') as clump_database_fp:
            clump_database = pickle.load(clump_database_fp)

    for obs_id in f_ring.enumerate_obsids(arguments):
        print(f'Processing {obs_id}')

        (ew_data_filename, ew_metadata_filename) = f_ring.ew_paths(arguments, obs_id)

        if (not os.path.exists(ew_metadata_filename) or
            not os.path.exists(ew_data_filename+'.npy')):
            print('EW or EW metadata missing')
            continue

        with open(ew_metadata_filename, 'rb') as ew_metadata_fp:
            ew_metadata = pickle.load(ew_metadata_fp, encoding='latin1')
        orig_ew_data = np.load(ew_data_filename+'.npy')
        # Find the mask for valid EW entries. Any place where EW == 0 is bad data.
        ew_mask = (orig_ew_data == 0)

        long_res_deg = np.degrees(ew_metadata['longitude_resolution'])
        ETs = ew_metadata['ETs']
        emission_angles = ew_metadata['emission_angles']
        phase_angles = ew_metadata['phase_angles']
        incidence_angle = ew_metadata['incidence_angle']
        longitudes = ew_metadata['longitudes']
        resolutions = ew_metadata['resolutions']

        long_deg = np.arange(orig_ew_data.size)*long_res_deg
        orig_ew_data = (np.sin(np.radians(long_deg*360/30)) +   # Period = 30 degrees
                        # np.sin(np.radians(long_deg*360/10)) +   # Period = 10 degrees
                        # np.sin(np.radians(long_deg*360/3.2)) +  # Period = 3.2 degrees
                        # np.sin(np.radians(long_deg*360/0.7)) +  # Period = 0.7 degrees
                       0)

        ew_data = adjust_ew_for_zero_phase(orig_ew_data, phase_angles,
                                           emission_angles, incidence_angle)

        print(f'Orig # longitudes {len(longitudes)}, longitude resolution {long_res_deg}')

#        ew_data *= f_ring_util.normalized_ew_factor(np.array(phase_angles), np.array(emission_angles), np.array(mosaic_incidence_angles))

        if arguments.prefilter:
            ew_data = clump_bandpass_filter.fft_filter(ew_data, long_res_deg,
                                                       obs_id, plot=False)

        # Now that we're done with the FFT, we can make ew_data an actual MaskedArray
        ew_data = ew_data.view(ma.MaskedArray)
        ew_data.mask = ew_mask

        # If we didn't do the FFT, we can do a normal smoothing process instead.
        # This works on masked data.
        # if not arguments.prefilter:
        #     # We want to ignore any features < 3 degrees
        #     ew_data = smooth_ew(ew_data, long_res_deg, 3.0)

        mean_emission = np.mean(emission_angles[~ew_mask])
        mean_phase = np.mean(phase_angles[~ew_mask])

        if False:
            plt.plot(longitudes, ew_data, lw=1, color='blue')
            plt.title(f'{obs_id} - EW Profile After Prefilter and Smoothing')
            plt.show()

        if np.all(ew_mask):
            print('ALL DATA IS MASKED - ABORTING')
            continue
        else:
            r_min = np.min(resolutions[~ew_mask])
            r_max = np.max(resolutions[~ew_mask])
            e_min = np.min(emission_angles[~ew_mask])
            e_max = np.max(emission_angles[~ew_mask])
            p_min = np.min(phase_angles[~ew_mask])
            p_max = np.max(phase_angles[~ew_mask])
            print(f'R {r_min:.02f}-{r_max:.02f}  E {e_min:.02f}-{e_max:.02f}  '
                  f'P {p_min:.02f}-{p_max:.02f}  I {incidence_angle:.02f}')

        sdg_wavelet, fdg_wavelet = select_clumps(arguments, ew_data,
                                                 longitudes, obs_id,
                                                 ew_metadata, clump_database)

        if arguments.plot_scalogram:
            plot_scalogram(sdg_wavelet, fdg_wavelet, clump_database[obs_id])

if arguments.update_clump_database or arguments.replace_clump_database:
    if arguments.voyager:
        print('Saving to Voyager clump database')
        clump_arguments = clump_util.ClumpFindOptions()
        clump_arguments.type = 'wavelet mexhat'
        clump_arguments.scale_min = arguments.scale_min
        clump_arguments.scale_max = arguments.scale_max
        clump_arguments.scale_step = arguments.scale_step
        clump_arguments.clump_size_min = arguments.clump_size_min
        clump_arguments.clump_size_max = arguments.clump_size_max
        clump_arguments.prefilter = arguments.prefilter

        clump_database_path, clump_chains_path = f_ring.clumpdb_paths(arguments)
        clump_database_fp = open(clump_database_path, 'wb')
        pickle.dump(clump_arguments, clump_database_fp)
        pickle.dump(clump_database, clump_database_fp)
        clump_database_fp.close()

    elif arguments.downsample:
        downsampled_clump_database_fp = os.path.join(
            f_ring.VOYAGER_PATH, 'downsampled_clump_database.pickle')
        print('Saving to downsampled_clump_database')
        clump_arguments = clump_util.ClumpFindOptions()
        clump_arguments.type = 'wavelet mexhat'
        clump_arguments.scale_min = arguments.scale_min
        clump_arguments.scale_max = arguments.scale_max
        clump_arguments.scale_step = arguments.scale_step
        clump_arguments.clump_size_min = arguments.clump_size_min
        clump_arguments.clump_size_max = arguments.clump_size_max
        clump_arguments.prefilter = arguments.prefilter
        clump_database_path, clump_chains_path = f_ring.clumpdb_paths(arguments)
        downsampled_clump_database_fp = open(downsampled_clump_database_fp, 'wb')

        clump_database_path, clump_chains_path = f_ring.clumpdb_paths(arguments)
        clump_database_fp = open(clump_database_path, 'wb')
        pickle.dump(clump_arguments, clump_database_fp)
        pickle.dump(clump_database, clump_database_fp)
        clump_database_fp.close()

        pickle.dump(clump_arguments, downsampled_clump_database_fp)
        pickle.dump(clump_database, downsampled_clump_database_fp)
        downsampled_clump_database_fp.close()
    else:
        print('Saving to regular clump database')
        clump_arguments = clump_util.ClumpFindOptions()
        clump_arguments.type = 'wavelet mexhat'
        clump_arguments.scale_min = arguments.scale_min
        clump_arguments.scale_max = arguments.scale_max
        clump_arguments.scale_step = arguments.scale_step
        clump_arguments.clump_size_min = arguments.clump_size_min
        clump_arguments.clump_size_max = arguments.clump_size_max
        clump_arguments.prefilter = arguments.prefilter
        clump_database_path, clump_chains_path = f_ring.clumpdb_paths(arguments)
        clump_database_fp = open(clump_database_path, 'wb')

        pickle.dump(clump_arguments, clump_database_fp)
        pickle.dump(clump_database, clump_database_fp)
        clump_database_fp.close()
