##########################################################################################
# Create various types of equivalent width for the observations.
#
# By default we create:
#   EW, EW Std, Normal EW, Normal EW Std
#
# If --ew-core-inner-radius and --ew-core-outer-radius are specified, we also create:
#   EWI, EWC, EWO and their associated Std and Normal versions
# If --tau and --phase-curve-params are also specified, we also create:
#   Normal EW3Z, Normal EW3Z Std, Normal EW3ZPN, Normal EW3ZPN Std
#
# If --radial-step is specified, we also create:
#   EW<radius> and their associated Std and Normal versions for each radius
#
# If --compute-widths is specified, we also create:
#   Width1, Width1 Std, Width2, Width2 Std, Width3, Width3 Std
##########################################################################################

import argparse
import csv
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy.ndimage as nd

import f_ring_util.f_ring as f_ring


cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
   cmd_line = []

parser = argparse.ArgumentParser()

parser.add_argument('--ew-inner-radius', type=int, default=140220-750,
                    help='The inner radius of the range')
parser.add_argument('--ew-outer-radius', type=int, default=140220+750,
                    help='The outer radius of the range')
parser.add_argument('--ew-core-inner-radius', type=int, default=None,
                    help='The inner radius of the core, if applicable')
parser.add_argument('--ew-core-outer-radius', type=int, default=None,
                    help='The outer radius of the core, if applicable')
parser.add_argument('--compute-core-center', action='store_true', default=False,
                    help='Compute the core location based on the brightest pixel')
parser.add_argument('--compute-widths', action='store_true', default=False,
                    help='Compute the widths for each slice')
parser.add_argument('--widths-frac-mode', action='store_true', default=False,
                    help='Compute widths in fractional mode')
parser.add_argument('--tau', type=float, default=None, # default=0.042,
                    help='Tau to use computing 3-zone EW and phase-normalization')
parser.add_argument('--phase-curve-params', type=str, default=None,
                    # default='0.634,1.874,-0.026,0.876',
                    help='The parameters for the phase curve for width computation')
parser.add_argument('--width-thresholds-abs', type=str, default=None,
                    # default='0.002,0.001,0.0002',
                    help='The default width threshold values in absolute mode')
parser.add_argument('--width-thresholds-frac', type=str, default=None,
                    # default='.75,.85,.95',
                    help='The default width threshold values in fractional mode')
parser.add_argument('--plot-results', action='store_true', default=False,
                    help='Plot the EW and Width results')
parser.add_argument('--save-plots', action='store_true', default=False,
                    help='Same as --plot-results but save plots to disk instead')
parser.add_argument('--downsample', type=int, default=1,
                    help='Amount to downsample the mosaic in longitude')
parser.add_argument('--minimum-coverage', type=float, default=0, # XXX
                    help='Minimum coverage (in degrees) allowed for a good obsid')
parser.add_argument('--maximum-bad-pixels-percentage', type=float, default=2,
                    help='Maximum percentage of bad pixels for a good radial slice')
parser.add_argument('--slice-size', type=float, default=0,
                    help='Slice size in degrees longitude')
parser.add_argument('--output-csv-filename', type=str,
                    help='Name of output CSV file')
parser.add_argument('--agg-csv-filename', type=str,
                    help='Name of aggregate (mean) output CSV file')
parser.add_argument('--radial-step', type=int, default=None,
                    help='Radial step size for multiple small radial steps')

f_ring.add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)

if arguments.phase_curve_params is None:
    HG_PARAMS = None
else:
    HG_PARAMS = [float(x) for x in arguments.phase_curve_params.split(',')]
if arguments.widths_frac_mode:
    if arguments.width_thresholds_frac is None:
        WIDTH_THRESHOLDS = None
    else:
        WIDTH_THRESHOLDS = [float(x) for x in arguments.width_thresholds_frac.split(',')]
else:
    if arguments.width_thresholds_abs is None:
        WIDTH_THRESHOLDS = None
    else:
        WIDTH_THRESHOLDS = [float(x) for x in arguments.width_thresholds_abs.split(',')]


##########################################################################################

def max_range(a, fracs, mean_brightness=None):
    ret = []
    frac_idx = 0
    if mean_brightness is None:
        mean_brightness = np.sum(a)
    target = mean_brightness*fracs[frac_idx]
    all_ones = np.ones(len(a), dtype=float)
    for width in range(1, len(a)+1):
        conv = np.convolve(a, all_ones[:width], mode='valid')
        conv[conv < target] = np.inf
        min_idx = np.argmin(conv)
        if conv[min_idx] != np.inf:
            ret.append((min_idx, min_idx+width-1))
            frac_idx += 1
            if frac_idx >= len(fracs):
                return ret
            target = mean_brightness*fracs[frac_idx]
    return None

def width_from_wings(a, fracs, mean_brightness=None):
    assert False
    # a = nd.median_filter(a, 9)
    if mean_brightness is None:
        mask = np.ones(len(a), dtype=bool)
        mask[:wing1_width+1] = True
        mask[wing2_width:] = True
        mean_brightness = np.mean(a[mask])
    ret = []
    for frac in fracs:
        target = mean_brightness * frac
        wing1_pos = np.argmax(a >= target)
        wing2_pos = len(a) - np.argmax(a[::-1] >= target) - 1
        # print(wing1_pos, a[wing1_pos], wing2_pos, a[wing2_pos])
        ret.append((wing1_pos, wing2_pos))
    return ret

def width_from_abs(a, abs_vals):
    ret = []
    for abs_val in abs_vals:
        wing1_pos = np.argmax(a >= abs_val)
        wing2_pos = len(a) - np.argmax(a[::-1] >= abs_val) - 1
        # print(wing1_pos, a[wing1_pos], wing2_pos, a[wing2_pos])
        ret.append((wing1_pos, wing2_pos))
    return ret


##########################################################################################
#
# PROCESS THE ARGUMENTS AND INITIALIZE STATE
#
##########################################################################################

assert ((arguments.ew_inner_radius is None) ==
        (arguments.ew_outer_radius is None)), \
            'Both --ew-inner-radius and --ew-outer-radius must be specified'
assert ((arguments.ew_core_inner_radius is None) ==
        (arguments.ew_core_outer_radius is None)), \
            'Both --ew-core-inner-radius and --ew-core-outer-radius must be specified'


# FIGURE OUT THE RING RADIAL LIMITS

# ew_inner_radius/ew_outer_radius of None or 0 means just use the whole width
# of the mosaic, whatever that is
if arguments.ew_inner_radius is not None and arguments.ew_inner_radius != 0:
    ring_lower_limit = int((arguments.ew_inner_radius -
                            arguments.radius_inner_delta -
                            arguments.ring_radius) / arguments.radius_resolution)
else:
    ring_lower_limit = 0
if arguments.ew_outer_radius is not None and arguments.ew_outer_radius != 0:
    ring_upper_limit = int((arguments.ew_outer_radius +
                            arguments.radius_outer_delta -
                            arguments.ring_radius) / arguments.radius_resolution)
else:
    ring_upper_limit = int((arguments.radius_outer_delta - arguments.radius_inner_delta) /
                           arguments.radius_resolution)
print(f'EW  lower limit: {arguments.ew_inner_radius:6d} km, pix {ring_lower_limit:4d}')
print(f'EW  upper limit: {arguments.ew_outer_radius:6d} km, pix {ring_upper_limit:4d}')

# If there is no core specified, then there is just a single radial region (range1)
# Otherwise there are three ranges - inner skirt (range1), core (range2), outer skirt
# (range3)
three_zone = False
if arguments.ew_core_inner_radius is None:
    ring_lower_limit1 = ring_lower_limit
    ring_upper_limit1 = ring_upper_limit
    ring_lower_limit2 = None
    ring_upper_limit2 = None
    ring_lower_limit3 = None
    ring_upper_limit3 = None
else:
    three_zone = True
    # Zone 1: [Ring inner limit, core inner radius)
    ring_lower_limit1 = ring_lower_limit
    ring_upper_limit1 = int((arguments.ew_core_inner_radius -
                             arguments.radius_inner_delta -
                             arguments.ring_radius) / arguments.radius_resolution)-1
    # Zone 2: [Core inner radius, core outer radius]
    ring_lower_limit2 = ring_upper_limit1 + 1
    ring_upper_limit2 = int((arguments.ew_core_outer_radius +
                             arguments.radius_outer_delta -
                             arguments.ring_radius) / arguments.radius_resolution)
    # Zone 3: (Core outer radius, Ring outer limit]
    ring_lower_limit3 = ring_upper_limit2 + 1
    ring_upper_limit3 = ring_upper_limit
    ring_upper_limit1_km = (ring_upper_limit1 * arguments.radius_resolution +
                            arguments.radius_inner_delta + arguments.ring_radius)
    ring_lower_limit3_km = (ring_lower_limit3 * arguments.radius_resolution +
                            arguments.radius_inner_delta + arguments.ring_radius +
                            arguments.radius_resolution - 1)
    print(f'EWI lower limit: {arguments.ew_inner_radius:6.0f} km, '
          f'pix {ring_lower_limit1:4d}')
    print(f'EWI upper limit: {ring_upper_limit1_km:6.0f} km, '
          f'pix {ring_upper_limit1:4d}')
    print(f'EWC lower limit: {arguments.ew_core_inner_radius:6.0f} km, '
          f'pix {ring_lower_limit2:4d}')
    print(f'EWC upper limit: {arguments.ew_core_outer_radius:6.0f} km, '
          f'pix {ring_upper_limit2:4d}')
    print(f'EWO lower limit: {ring_lower_limit3_km:6.0f} km, '
          f'pix {ring_lower_limit3:4d}')
    print(f'EWO upper limit: {arguments.ew_outer_radius:6.0f} km, '
          f'pix {ring_upper_limit3:4d}')


# HANDLE MULTIPLE RADIAL STEPS

if arguments.radial_step is not None:
    num_radial_steps = int((arguments.ew_outer_radius - arguments.ew_inner_radius) /
                           arguments.radial_step)+1
    radial_step_pix = int(arguments.radial_step / arguments.radius_resolution)
    print(f'Num radial steps: {num_radial_steps}')
else:
    num_radial_steps = None


# INITIALIZE THE OUTPUT CSV FILE WITH HEADERS

csv_fp = None
if arguments.output_csv_filename:
    assert (arguments.slice_size == 0 or
            360 / arguments.slice_size == int(360 / arguments.slice_size))
    csv_fp = open(arguments.output_csv_filename, 'w')
    writer = csv.writer(csv_fp)
    hdr = ['Observation', 'Slice#', 'Num Data', 'Date',
           'Min Long', 'Max Long', 'Mean Long',
           'Min Inertial Long', 'Max Inertial Long', 'Mean Inertial Long',
           'Min Long of Pericenter', 'Max Long of Pericenter', 'Mean Long of Pericenter',
           'Min True Anomaly', 'Max True Anomaly', 'Mean True Anomaly',
           'Min Res', 'Max Res', 'Mean Res',
           'Min Phase', 'Max Phase', 'Mean Phase',
           'Min Emission', 'Max Emission', 'Mean Emission',
           'Incidence',
           '% Coverage',
           'EW', 'EW Std', 'Normal EW', 'Normal EW Std']
    if three_zone:
        hdr += ['EWI', 'EWI Std', 'Normal EWI', 'Normal EWI Std']
        hdr += ['EWC', 'EWC Std', 'Normal EWC', 'Normal EWC Std']
        hdr += ['EWO', 'EWO Std', 'Normal EWO', 'Normal EWO Std']
        if arguments.tau is not None:
            hdr += ['Normal EW3Z', 'Normal EW3Z Std',
                    'Normal EW3ZPN', 'Normal EW3ZPN Std']

    if num_radial_steps is not None:
        for radial_step in range(num_radial_steps):
            start_ew = arguments.ew_inner_radius + radial_step * arguments.radial_step
            hdr += [f'EW{start_ew}', f'EW{start_ew} Std',
                    f'Normal EW{start_ew}', f'Normal EW{start_ew} Std']

    if arguments.compute_widths:
        hdr += ['Width1',  'Width1 Std',
                'Width2',  'Width2 Std',
                'Width3',  'Width3 Std',
                'Width1I', 'Width1I Std',
                'Width1O', 'Width1O Std',
                'Width2I', 'Width2I Std',
                'Width2O', 'Width2O Std',
                'Width3I', 'Width3I Std',
                'Width3O', 'Width3O Std']

    if arguments.compute_core_center:
        hdr += ['Core Offset Median',
                'Core Offset Mean',
                'Core Offset Std']

    writer.writerow(hdr)


# XXX

if arguments.agg_csv_filename:
    assert three_zone
    mean_by_radius_list = []


##########################################################################################
#
# PROCESS THE MOSAICS
#
##########################################################################################

for obs_id in f_ring.enumerate_obsids(arguments):
    (bkgnd_sub_mosaic_filename,
     bkgnd_sub_mosaic_metadata_filename) = f_ring.bkgnd_sub_mosaic_paths(
        arguments, obs_id)

    if not os.path.exists(bkgnd_sub_mosaic_filename):
        print('NO FILE', bkgnd_sub_mosaic_filename)
        continue
    if not os.path.exists(bkgnd_sub_mosaic_metadata_filename):
        print('NO FILE', bkgnd_sub_mosaic_metadata_filename)
        continue

    with open(bkgnd_sub_mosaic_metadata_filename, 'rb') as bkgnd_metadata_fp:
        metadata = pickle.load(bkgnd_metadata_fp, encoding='latin1')
    with np.load(bkgnd_sub_mosaic_filename) as npz:
        bsm_img = ma.MaskedArray(**npz)
        bsm_img = bsm_img.filled(0)

    ds = arguments.downsample

    longitudes = metadata['longitudes'][::ds].view(ma.MaskedArray)
    resolutions = metadata['resolutions'][::ds]
    image_numbers = metadata['image_numbers'][::ds]
    ETs = metadata['ETs'][::ds]
    emission_angles = metadata['emission_angles'][::ds]
    incidence_angle = metadata['incidence_angle']
    phase_angles = metadata['phase_angles'][::ds]
    inertial_longitudes = f_ring.fring_corotating_to_inertial(longitudes, ETs)
    longitude_of_pericenters = f_ring.fring_longitude_of_pericenter(ETs)
    true_anomalies = f_ring.fring_true_anomaly(ETs, inertial_longitudes)
    # bsm = background-subtracted-mosaic
    bsm_img = bsm_img[:,::ds]

    # Make the submosaic for the radial range desired
    if three_zone:
        restr_bsm_img = bsm_img[ring_lower_limit1:ring_upper_limit3+1,:]
    else:
        restr_bsm_img = bsm_img[ring_lower_limit:ring_upper_limit1,:]
    bad_long = longitudes < 0
    percentage_long_ok = float(np.sum(~bad_long)) / len(longitudes) * 100
    # Choose bad longitudes based only on the full radial range desired
    bad_long |= (np.sum(ma.getmaskarray(restr_bsm_img), axis=0) >
                 restr_bsm_img.shape[0]*arguments.maximum_bad_pixels_percentage/100)
    bad_long |= np.sum(restr_bsm_img, axis=0) == 0
    percentage_ew_ok = float(np.sum(~bad_long)) / len(longitudes) * 100
    longitudes[bad_long] = ma.masked
    bsm_img[:, bad_long] = ma.masked
    restr_bsm_img[:, bad_long] = ma.masked

    # Make all the sub-mosaics for 3-zone use
    if three_zone:
        restr_bsm_img1 = bsm_img[ring_lower_limit1:ring_upper_limit1+1,:]
        restr_bsm_img2 = bsm_img[ring_lower_limit2:ring_upper_limit2+1,:]
        restr_bsm_img3 = bsm_img[ring_lower_limit3:ring_upper_limit3+1,:]

        if arguments.tau is not None:
            # Make the whole image be Normal and then tau-augment the center region.
            # Note ring_lower_limit<n> are based on the full mosaic size,
            # but here we're working with the radially-restricted version.
            restr_bsm_img_tau = restr_bsm_img * np.abs(np.cos(emission_angles))
            restr_bsm_img_tau[ring_lower_limit2-ring_lower_limit1:
                              ring_upper_limit2-ring_lower_limit1+1, :] *= (
                f_ring.compute_corrected_ew(1,
                                                 np.degrees(emission_angles),
                                                 np.degrees(incidence_angle),
                                                 arguments.tau))
            # Make the phase-normalized version by dividing by the phase curve at
            # alpha and multiplying by the phase curve at 0.
            restr_bsm_img_tau_pn = (restr_bsm_img_tau /
                          f_ring.hg_func(HG_PARAMS, np.degrees(phase_angles)) *
                          f_ring.hg_func(HG_PARAMS, 0))

    if arguments.agg_csv_filename:
        mean_by_radius = ma.mean(restr_bsm_img_tau_pn, axis=1)
        mean_by_radius_list.append(mean_by_radius)

    print(f'{obs_id:30s} {percentage_long_ok:3.0f}% {percentage_ew_ok:3.0f}%', end='')
    if (np.sum(~bad_long)*arguments.longitude_resolution*arguments.downsample <
            arguments.minimum_coverage):
        print(' Skipped due to poor coverage')
        continue

    # Make the full EW profiles here, which will later by sliced up.
    # But print the statistics based on the whole profile.
    ew_profile = np.sum(restr_bsm_img, axis=0) * arguments.radius_resolution
    ew_profile[bad_long] = ma.masked
    ew_mean = ma.mean(ew_profile)
    ew_std = ma.std(ew_profile)
    print(f' EW {ew_mean:6.3f} +/- {ew_std:6.3f}', end='')
    ew_profile_n = ew_profile * np.abs(np.cos(emission_angles))
    ew_mean_n = ma.mean(ew_profile_n)
    ew_std_n = ma.mean(ew_profile_n)
    print(f' EWN {ew_mean_n:6.3f} +/- {ew_std_n:6.3f}', end='')

    if three_zone:
        if arguments.tau is not None:
            brightness_3z = np.sum(restr_bsm_img_tau, axis=0)
            ew_profile_3z = brightness_3z * arguments.radius_resolution
            ew_profile_3z[bad_long] = ma.masked
            ew_mean_3z = ma.mean(ew_profile_3z)
            ew_std_3z = ma.std(ew_profile_3z)
            print(f' EW3Z {ew_mean_3z:6.3f} +/- {ew_std_3z:6.3f}', end='')

            brightness_3zpn = np.sum(restr_bsm_img_tau_pn, axis=0)
            ew_profile_3z_pn = brightness_3zpn * arguments.radius_resolution
            ew_profile_3z_pn[bad_long] = ma.masked
            ew_mean_3z_pn = ma.mean(ew_profile_3z_pn)
            ew_std_3z_pn = ma.std(ew_profile_3z_pn)
            print(f' EW3ZPN {ew_mean_3z_pn:6.3f} +/- {ew_std_3z_pn:6.3f}', end='')

        brightness1 = np.sum(restr_bsm_img1, axis=0)
        ew_profile1 = brightness1 * arguments.radius_resolution
        ew_profile1[bad_long] = ma.masked
        ew_mean1 = ma.mean(ew_profile1)
        ew_std1 = ma.std(ew_profile1)
        print(f' EWI {ew_mean1:6.3f} +/- {ew_std1:6.3f}', end='')

        brightness2 = np.sum(restr_bsm_img2, axis=0)
        ew_profile2 = brightness2 * arguments.radius_resolution
        ew_profile2[bad_long] = ma.masked
        ew_mean2 = ma.mean(ew_profile2)
        ew_std2 = ma.std(ew_profile2)
        print(f' EWC {ew_mean2:6.3f} +/- {ew_std2:6.3f}', end='')

        brightness3 = np.sum(restr_bsm_img3, axis=0)
        ew_profile3 = brightness3 * arguments.radius_resolution
        ew_profile3[bad_long] = ma.masked
        ew_mean3 = ma.mean(ew_profile3)
        ew_std3 = ma.std(ew_profile3)
        print(f' EWO {ew_mean3:6.3f} +/- {ew_std3:6.3f}', end='')

    if num_radial_steps is not None:
        # print()
        ew_profile_steps = []
        for radial_step in range(num_radial_steps):
            start_ew = arguments.ew_inner_radius + radial_step * arguments.radial_step
            start_rad_pix = radial_step * radial_step_pix
            end_rad_pix = (radial_step+1) * radial_step_pix
            step_brightness = np.sum(restr_bsm_img[start_rad_pix:end_rad_pix, :], axis=0)
            step_ew_profile = step_brightness * arguments.radius_resolution
            step_ew_profile[bad_long] = ma.masked
            step_ew_mean = ma.mean(step_ew_profile)
            step_ew_std = ma.std(step_ew_profile)
            # print(f'  EW{start_ew} {step_ew_mean:6.3f} +/- {step_ew_std:6.3f}')
            ew_profile_steps.append(step_ew_profile)

    rd = arguments.ew_inner_radius - arguments.ring_radius
    rr = arguments.radius_resolution

    if arguments.compute_widths:
        if three_zone and arguments.tau is not None:
            filtered_restr_bsm_img = restr_bsm_img_tau_pn
        else:
            filtered_restr_bsm_img = restr_bsm_img
        # filtered_restr_bsm_img = nd.uniform_filter(restr_bsm_img, (9,49), mode='wrap')

        # We compute the width on the phase-normalized (to phase_angle=0)
        # mosaic. We also tau-adjust the core. Unless those parameters are turned off.
        pn_restr_bsm_img = filtered_restr_bsm_img * arguments.radius_resolution

        widths1 = ma.zeros((len(longitudes), 2), dtype=float)
        widths1[:] = ma.masked
        widths2 = ma.zeros((len(longitudes), 2), dtype=float)
        widths2[:] = ma.masked
        widths3 = ma.zeros((len(longitudes), 2), dtype=float)
        widths3[:] = ma.masked

        for idx in range(len(longitudes)):
            if bad_long[idx]:
                continue
            if arguments.widths_frac_mode:
                ret = max_range(pn_restr_bsm_img[:,idx], WIDTH_THRESHOLDS)
            else:
                ret = width_from_abs(pn_restr_bsm_img[:,idx], WIDTH_THRESHOLDS)
            widths1[idx] = (ret[0][0]*rr+rd, ret[0][1]*rr+rd)
            widths2[idx] = (ret[1][0]*rr+rd, ret[1][1]*rr+rd)
            widths3[idx] = (ret[2][0]*rr+rd, ret[2][1]*rr+rd)

        w1 = widths1[:,1]-widths1[:,0]
        w2 = widths2[:,1]-widths2[:,0]
        w3 = widths3[:,1]-widths3[:,0]

        w1_mean = ma.mean(w1)
        w1_std = ma.std(w1)
        w2_mean = ma.mean(w2)
        w2_std = ma.std(w2)
        w3_mean = ma.mean(w3)
        w3_std = ma.std(w3)
        print(f'  w1: {w1_mean:4.0f} +/- {w1_std:3.0f}', end='')
        print(f'  w2: {w2_mean:4.0f} +/- {w2_std:3.0f}', end='')
        print(f'  w3: {w3_mean:4.0f} +/- {w3_std:3.0f}', end='')

    if arguments.compute_core_center:
        core_centers = np.argmax(restr_bsm_img, axis=0).view(ma.MaskedArray) * rr + rd
        core_centers[bad_long] = ma.masked
        core_median = ma.median(core_centers)
        print(f'  Core: {core_median}', end='')

    print()

    if arguments.output_csv_filename:
        slice_size_in_longitudes = int(arguments.slice_size /
                                       arguments.longitude_resolution)
        if arguments.slice_size == 0:
            num_slices = 1
            slice_size_in_longitudes = len(longitudes)
        else:
            num_slices = len(longitudes) // slice_size_in_longitudes
        for slice_num in range(num_slices):
            slice_start = slice_num * slice_size_in_longitudes
            slice_end = (slice_num+1) * slice_size_in_longitudes
            slice_bad_long = bad_long[slice_start:slice_end]
            if np.all(slice_bad_long):
                continue
            slice_good_long = ~slice_bad_long
            slice_ETs = ETs[slice_start:slice_end][slice_good_long]
            slice_emission_angles = emission_angles[slice_start:slice_end][slice_good_long]
            slice_phase_angles = phase_angles[slice_start:slice_end][slice_good_long]
            slice_resolutions = resolutions[slice_start:slice_end][slice_good_long]
            slice_longitudes = longitudes[slice_start:slice_end][slice_good_long]
            slice_inertial_longs = inertial_longitudes[slice_start:slice_end][slice_good_long]
            slice_longitude_of_pericenters = longitude_of_pericenters[slice_start:slice_end][slice_good_long]
            slice_true_anomalies = true_anomalies[slice_start:slice_end][slice_good_long]
            if arguments.compute_widths:
                slice_w1 = w1[slice_start:slice_end][slice_good_long]
                slice_w2 = w2[slice_start:slice_end][slice_good_long]
                slice_w3 = w3[slice_start:slice_end][slice_good_long]
                slice_w1i = widths1[slice_start:slice_end, 0][slice_good_long]
                slice_w1o = widths1[slice_start:slice_end, 1][slice_good_long]
                slice_w2i = widths2[slice_start:slice_end, 0][slice_good_long]
                slice_w2o = widths2[slice_start:slice_end, 1][slice_good_long]
                slice_w3i = widths3[slice_start:slice_end, 0][slice_good_long]
                slice_w3o = widths3[slice_start:slice_end, 1][slice_good_long]
            if arguments.compute_core_center:
                slice_core_centers = core_centers[slice_start:slice_end][slice_good_long]

            slice_min_long = np.degrees(np.min(slice_longitudes))
            slice_max_long = np.degrees(np.max(slice_longitudes))
            slice_mean_long = np.degrees(np.mean(slice_longitudes))

            slice_min_inertial_long = np.degrees(np.min(slice_inertial_longs))
            slice_max_inertial_long = np.degrees(np.max(slice_inertial_longs))
            slice_mean_inertial_long = np.degrees(np.mean(slice_inertial_longs))

            slice_min_long_of_peri = np.degrees(np.min(slice_longitude_of_pericenters))
            slice_max_long_of_peri = np.degrees(np.max(slice_longitude_of_pericenters))
            slice_mean_long_of_peri = np.degrees(np.mean(slice_longitude_of_pericenters))

            slice_min_true_anomaly = np.degrees(np.min(slice_true_anomalies))
            slice_max_true_anomaly = np.degrees(np.max(slice_true_anomalies))
            slice_mean_true_anomaly = np.degrees(np.mean(slice_true_anomalies))

            slice_min_et = np.min(slice_ETs)
            slice_max_et = np.max(slice_ETs)
            slice_mean_et = np.mean(slice_ETs)
            slice_et_date = f_ring.et2utc(slice_min_et)

            slice_min_em = np.min(slice_emission_angles)
            slice_max_em = np.max(slice_emission_angles)
            slice_mean_em = np.mean(slice_emission_angles)

            slice_min_ph = np.min(slice_phase_angles)
            slice_max_ph = np.max(slice_phase_angles)
            slice_mean_ph = np.mean(slice_phase_angles)

            slice_min_res = np.min(slice_resolutions)
            slice_max_res = np.max(slice_resolutions)
            slice_mean_res = np.mean(slice_resolutions)

            slice_ew_profile = ew_profile[slice_start:slice_end][slice_good_long]
            slice_ew_mean = np.mean(slice_ew_profile)
            slice_ew_std = np.std(slice_ew_profile)
            slice_ew_profile_mu = (slice_ew_profile *
                                   np.abs(np.cos(slice_emission_angles)))
            slice_ew_mean_mu = np.mean(slice_ew_profile_mu)
            slice_ew_std_mu = np.std(slice_ew_profile_mu)

            # if slice_ew_mean <= 0:
            #     print(obs_id, slice_num, 'EW Mean < 0')
            #     continue

            row = [obs_id, slice_num, np.sum(~slice_bad_long), slice_et_date,
                   np.round(slice_min_long, 2),
                   np.round(slice_max_long, 2),
                   np.round(slice_mean_long, 2),
                   np.round(slice_min_inertial_long, 3),
                   np.round(slice_max_inertial_long, 3),
                   np.round(slice_mean_inertial_long, 3),
                   np.round(slice_min_long_of_peri, 3),
                   np.round(slice_max_long_of_peri, 3),
                   np.round(slice_mean_long_of_peri, 3),
                   np.round(slice_min_true_anomaly, 3),
                   np.round(slice_max_true_anomaly, 3),
                   np.round(slice_mean_true_anomaly, 3),
                   np.round(slice_min_res, 8),
                   np.round(slice_max_res, 8),
                   np.round(slice_mean_res, 8),
                   np.round(np.degrees(slice_min_ph), 8),
                   np.round(np.degrees(slice_max_ph), 8),
                   np.round(np.degrees(slice_mean_ph), 8),
                   np.round(np.degrees(slice_min_em), 8),
                   np.round(np.degrees(slice_max_em), 8),
                   np.round(np.degrees(slice_mean_em), 8),
                   np.round(np.degrees(incidence_angle), 8),
                   np.round(percentage_ew_ok, 2),
                   np.round(slice_ew_mean, 8),
                   np.round(slice_ew_std, 8),
                   np.round(slice_ew_mean_mu, 8),
                   np.round(slice_ew_std_mu, 8)]

            if three_zone:
                slice_ew_profile1 = ew_profile1[slice_start:slice_end][slice_good_long]
                slice_ew_mean1 = np.mean(slice_ew_profile1)
                slice_ew_std1 = np.std(slice_ew_profile1)
                slice_ew_profile_mu1 = (slice_ew_profile1 *
                                        np.abs(np.cos(slice_emission_angles)))
                slice_ew_mean_mu1 = np.mean(slice_ew_profile_mu1)
                slice_ew_std_mu1 = np.std(slice_ew_profile_mu1)

                row += [np.round(slice_ew_mean1, 8),
                        np.round(slice_ew_std1, 8),
                        np.round(slice_ew_mean_mu1, 8),
                        np.round(slice_ew_std_mu1, 8)]

                slice_ew_profile2 = ew_profile2[slice_start:slice_end][slice_good_long]
                slice_ew_mean2 = np.mean(slice_ew_profile2)
                slice_ew_std2 = np.std(slice_ew_profile2)
                slice_ew_profile_mu2 = (slice_ew_profile2 *
                                        np.abs(np.cos(slice_emission_angles)))
                slice_ew_mean_mu2 = np.mean(slice_ew_profile_mu2)
                slice_ew_std_mu2 = np.std(slice_ew_profile_mu2)

                row += [np.round(slice_ew_mean2, 8),
                        np.round(slice_ew_std2, 8),
                        np.round(slice_ew_mean_mu2, 8),
                        np.round(slice_ew_std_mu2, 8)]

                slice_ew_profile3 = ew_profile3[slice_start:slice_end][slice_good_long]
                slice_ew_mean3 = np.mean(slice_ew_profile3)
                slice_ew_std3 = np.std(slice_ew_profile3)
                slice_ew_profile_mu3 = (slice_ew_profile3 *
                                        np.abs(np.cos(slice_emission_angles)))
                slice_ew_mean_mu3 = np.mean(slice_ew_profile_mu3)
                slice_ew_std_mu3 = np.std(slice_ew_profile_mu3)

                row += [np.round(slice_ew_mean3, 8),
                        np.round(slice_ew_std3, 8),
                        np.round(slice_ew_mean_mu3, 8),
                        np.round(slice_ew_std_mu3, 8)]

                # Sanity check the math
                slice_ew_mean_threezone_total = (slice_ew_mean1 + slice_ew_mean2 +
                                                 slice_ew_mean3)
                assert abs(slice_ew_mean_threezone_total - slice_ew_mean) < 0.00001

                if arguments.tau is not None:
                    slice_ew_profile_3z = ew_profile_3z[slice_start:slice_end][slice_good_long]
                    slice_ew_mean_3z = np.mean(slice_ew_profile_3z)
                    slice_ew_std_3z = np.std(slice_ew_profile_3z)
                    slice_ew_profile_3z_pn = ew_profile_3z_pn[slice_start:slice_end][slice_good_long]
                    slice_ew_mean_3z_pn = np.mean(slice_ew_profile_3z_pn)
                    slice_ew_std_3z_pn = np.std(slice_ew_profile_3z_pn)
                    row += [np.round(slice_ew_mean_3z, 8),
                            np.round(slice_ew_std_3z, 8),
                            np.round(slice_ew_mean_3z_pn, 8),
                            np.round(slice_ew_std_3z_pn, 8)]

            if num_radial_steps is not None:
                total_step_ew_mean = 0
                for radial_step in range(num_radial_steps):
                    slice_step_ew_profile = (ew_profile_steps[radial_step]
                                                             [slice_start:slice_end]
                                                             [slice_good_long])
                    slice_step_ew_mean = np.mean(slice_step_ew_profile)
                    slice_step_ew_std = np.std(slice_step_ew_profile)
                    slice_step_ew_profile_mu = (slice_step_ew_profile *
                                            np.abs(np.cos(slice_emission_angles)))
                    slice_step_ew_mean_mu = np.mean(slice_step_ew_profile_mu)
                    slice_step_ew_std_mu = np.std(slice_step_ew_profile_mu)
                    total_step_ew_mean += slice_step_ew_mean
                    row += [np.round(slice_step_ew_mean, 8),
                            np.round(slice_step_ew_std, 8),
                            np.round(slice_step_ew_mean_mu, 8),
                            np.round(slice_step_ew_std_mu, 8)]

                # Sanity check the math
                assert abs(total_step_ew_mean - slice_ew_mean) < 0.00001

            if arguments.compute_widths:
                row += [np.round(np.mean(slice_w1), 3),
                        np.round(np.std(slice_w1), 3),
                        np.round(np.mean(slice_w2), 3),
                        np.round(np.std(slice_w2), 3),
                        np.round(np.mean(slice_w3), 3),
                        np.round(np.std(slice_w3), 3),
                        np.round(np.mean(slice_w1i), 3),
                        np.round(np.std(slice_w1i), 3),
                        np.round(np.mean(slice_w1o), 3),
                        np.round(np.std(slice_w1o), 3),
                        np.round(np.mean(slice_w2i), 3),
                        np.round(np.std(slice_w2i), 3),
                        np.round(np.mean(slice_w2o), 3),
                        np.round(np.std(slice_w2o), 3),
                        np.round(np.mean(slice_w3i), 3),
                        np.round(np.std(slice_w3i), 3),
                        np.round(np.mean(slice_w3o), 3),
                        np.round(np.std(slice_w3o), 3)]

            if arguments.compute_core_center:
                row += [np.round(np.median(slice_core_centers), 3),
                        np.round(np.mean(slice_core_centers), 3),
                        np.round(np.std(slice_core_centers), 3)]

            writer.writerow(row)

    if arguments.plot_results or arguments.save_plots:
        fig = plt.figure(figsize=(12, 8))
        plt.suptitle(obs_id)
        if arguments.compute_widths:
            ax = fig.add_subplot(311)
        else:
            ax = fig.add_subplot(111)
        plt.plot(np.degrees(longitudes), ew_profile, label='Full')
        if three_zone:
            plt.plot(np.degrees(longitudes), ew_profile1, label='Inner')
            plt.plot(np.degrees(longitudes), ew_profile2, label='Core')
            plt.plot(np.degrees(longitudes), ew_profile3, label='Outer')
            plt.legend()
        plt.xlim(0, 360)
        plt.ylabel('EW (km)')
        plt.xlabel('Longitude (degrees)')

        if arguments.compute_widths:
            gamma = 0.5
            blackpoint = max(np.min(restr_bsm_img[:, ~bad_long]), 0)
            whitepoint_ignore_frac = 0.995
            img_sorted = sorted(list(restr_bsm_img[:, ~bad_long].flatten()))
            whitepoint = img_sorted[np.clip(int(len(img_sorted)*
                                                whitepoint_ignore_frac),
                                            0, len(img_sorted)-1)]
            greyscale_img = np.floor((np.maximum(restr_bsm_img-blackpoint, 0)/
                                      (whitepoint-blackpoint))**gamma*256)
            greyscale_img = np.clip(greyscale_img, 0, 255)
            ax = fig.add_subplot(312)
            plt.imshow(greyscale_img[::-1, :],
                       extent=(0, 360,
                               arguments.ew_inner_radius - arguments.ring_radius,
                               arguments.ew_outer_radius - arguments.ring_radius),
                       aspect='auto',
                       cmap='gray', vmin=0, vmax=255)
            plt.plot(np.degrees(longitudes), widths1[:,0], color='cyan', lw=1, alpha=0.5)
            plt.plot(np.degrees(longitudes), widths1[:,1], color='cyan', lw=1, alpha=0.5)
            plt.plot(np.degrees(longitudes), widths2[:,0], color='orange', lw=1, alpha=0.5)
            plt.plot(np.degrees(longitudes), widths2[:,1], color='orange', lw=1, alpha=0.5)
            plt.plot(np.degrees(longitudes), widths3[:,0], color='red', lw=1, alpha=0.5)
            plt.plot(np.degrees(longitudes), widths3[:,1], color='red', lw=1, alpha=0.5)
            # plt.legend()
            plt.xlim(0, 360)
            plt.ylim(arguments.ew_inner_radius - arguments.ring_radius,
                     arguments.ew_outer_radius - arguments.ring_radius)
            plt.ylabel('Core offset (km)')
            plt.xlabel('Longitude (degrees)')
            ax = fig.add_subplot(313)
            plt.plot(np.degrees(longitudes), w1, color='cyan', label='W1')
            plt.plot(np.degrees(longitudes), w2, color='orange', label='W2')
            plt.plot(np.degrees(longitudes), w3, color='red', label='W3')
            plt.legend()
            plt.xlim(0, 360)
            plt.ylim(0, 1500)
            plt.ylabel('Ring width (km)')
            plt.xlabel('Longitude (degrees)')

        plt.tight_layout()
        if arguments.save_plots:
            if not os.path.exists('plots_ew'):
                os.mkdir('plots_ew')
            plt.savefig('plots_ew/'+obs_id+'.png')
            fig.clear()
            plt.close()
            plt.cla()
            plt.clf()
        if arguments.plot_results:
            plt.show()

if arguments.output_csv_filename:
    csv_fp.close()

if arguments.agg_csv_filename:
    agg_csv_fp = open(arguments.agg_csv_filename, 'w')
    agg_writer = csv.writer(agg_csv_fp)
    hdr = ['Radius',
           'Mean Normal EW3ZPN']
    agg_writer.writerow(hdr)
    mean_by_radius = ma.mean(mean_by_radius_list, axis=0)
    for radius_num, val in enumerate(mean_by_radius):
        radius = (arguments.ew_inner_radius + radius_num * arguments.radius_resolution -
                  arguments.ring_radius)
        row = [np.round(radius, 3),
               np.round(val, 5)]
        agg_writer.writerow(row)
    agg_csv_fp.close()
