##########################################################################################
# Create various types of equivalent width for the observations.
#
# CSV fields:
#
# *  Prefix                 Column is included when --simple-columns is specified.
#
#   *Observation            The Cassini observation name, including potentially the _N
#                           suffix for multiple mosaics for a single observation.
#
#   *Slice#                 The slice number, ranging from 0 to 360/slice_size. If a
#                           slice has no valid data, or doesn't meet other inclusive
#                           criteria, it is omitted.
#
#   *Num Data               The number of radial slices that went into computing the
#                           results for this slice. This can range from 1 to
#                           slice_size / longitude_resolution.
#
#   *Date                   The date/time (ISO 8601 format) of the minimum ET for the
#                           slice.
#
#   *Min/*Max/Mean Long      The minimum/maximum/mean co-rotating longitude (degrees)
#                           of valid data in the slice, relative to the epoch
#                           2007-01-01T00:00:00.
#
#   Min/Max/Mean Inertial Long
#                           The minimum/maximum/mean inertial longitude (degrees)
#                           of valid data in the slice in J2000.
#
#   Min/Max/Mean Long of Pericenter
#                           The minimum/maximum/mean longitude of pericenter (degrees)
#                           of the F ring for the times of valid data in the slice.
#
#   Min/Max/Mean True Anomaly
#                           The minimum/maximum/mean true anomaly (degrees) for the
#                           longitudes with valid data in the slice.
#
#   Min/Max/*Mean Radial Res
#                           The minimum/maximum/mean radial resolution (km/pixel) for
#                           the valid data in the slice.
#
#   Min/Max/*Mean Angular Res
#                           The minimum/maximum/mean angular resolution (deg/pixel) for
#                           the valid data in the slice.
#
#   Min/Max/*Mean Phase     The minimum/maximum/mean phase angles (degrees) for the
#                           valid data in the slice.
#
#   Min/Max/*Mean Emission  The minimum/maximum/mean emission angles (degrees) for the
#                           valid data in the slice.
#
#   *Incidence              The incidence angle for the valid data in the slice.
#                           We assume this does not change over the course of a single
#                           observation.
#
#   % Coverage              The percentage of 360 longitude covered by this entire
#                           observation (not this slice). If there is more than one
#                           slice for the observation, they all list the same coverage.
#
#   EW Median/
#   EW Mean/
#   EW Std                  The median, mean, and standard deviation of EW measurements
#                           for the valid data in this slice. These EWs are the raw
#                           measurements with no adjustment for viewing geometry.
#
#   Normal EW Median/
#   *Normal EW Mean/
#   *Normal EW Std          The median, mean, and standard deviation of normal
#                           (mu-adjusted) EW measurements for the valid data in this
#                           slice.
#
# If --include-quantiles is specified, we also include:
#
#   *EW Mean 15/EW Std 15/*Normal EW Mean 15/*Normal EW Std 15
#   *EW Mean 25/EW Std 25/*Normal EW Mean 25/*Normal EW Std 25
#   *EW Mean 50/EW Std 50/*Normal EW Mean 50/*Normal EW Std 50
#   *EW Mean 75/EW Std 75/*Normal EW Mean 75/*Normal EW Std 75
#   *EW Mean 85/EW Std 85/*Normal EW Mean 85/*Normal EW Std 85
#                           The mean and standard deviation of the EW/Normal EW
#                           for the given quantile for the valid data in this slice.
#
# If --ew-core-inner-radius and --ew-core-outer-radius are specified, we also create:
#
#   *EWI Mean/EWI Std/*Normal EWI Mean/*Normal EWI Std
#   *EWC Mean/EWC Std/*Normal EWC Mean/*Normal EWC Std
#   *EWO Mean/EWO Std/*Normal EWO Mean/*Normal EWO Std
#                           The mean and standard deviation of the EW/Normal EW
#                           for the inner/core/outer regions for the valid data
#                           in this slice.
#
# If --tau and --phase-curve-params are also specified, we also create:
#
#   *Normal EW3Z Mean/*Normal EW3Z Std/
#   *Normal EW3ZPN Mean/*Normal EW3ZPN Std
#                           The mean and standard deviation of the Normal EW
#                           adjusted for tau and phase-normalized for the valid
#                           data in this slice.
#
# If --radial-step-size is specified, we also create:
#
#   EW<radius> Mean/EW<radius> Std/
#   *Normal EW<radius> Mean/Normal EW<radius> Std
#                           The mean and standard deviation of the EW/Normal EW
#                           for the given radial slice for the valid data in this
#                           slice.
#                           If specified, the radial slices run from
#                               --radial-step-inner-radius to
#                               --radial-step-outer-radius
#                           Otherwise, the values from --ew-inner-radius and
#                           --ew-outer-radius are used.
#
# If --compute-widths is specified, we also create:
#   <TBD>
#
# If --compute-core-center is specified, we also create:
#
#   *Core Offset Median/*Core Offset Mean/*Core Offset Std
#                           The median, mean, and standard deviation of the radial
#                           positions of the core (defined as the brightest pixel
#                           along the radial slice) for the valid data in this
#                           slice.
#
# If --compute-moon-info is specified, we also create:
#
#   *Pandora Distance/*Pandora Long
#   *Prometheus Distance/*Prometheus Long
#                           The distance (km) and co-rotating longitude (degrees)
#                           for Pandora/Prometheus at the mean time of valid data
#                           in this slice.
#
##########################################################################################

import argparse
import csv
import math
import os
import sys
import traceback
import warnings

import msgpack
import msgpack_numpy

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'external'))

import f_ring_util.f_ring as f_ring


# Show a traceback for all warnings - makes it easier to find the source of
# numpy warnings.
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback


cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
   cmd_line = []

parser = argparse.ArgumentParser()

parser.add_argument('--minimum-coverage', type=float, default=0,
                    help='Minimum total coverage (in degrees) allowed for a good obsid')
parser.add_argument('--maximum-bad-pixels-percentage', type=float, default=1,
                    help='Maximum percentage of bad pixels for a good radial slice')
parser.add_argument('--slice-size', type=float, default=0,
                    help='Slice size in degrees longitude; 0 means 360')
parser.add_argument('--minimum-slice-coverage', type=float, default=0,
                    help='Minimum coverage (in degrees) allowed for a good slice')
parser.add_argument('--maximum-slice-radial-resolution', type=float, default=1e38,
                    help='Maximum radial resolution allowed for a good slice')

parser.add_argument('--output-csv-filename', type=str,
                    help='Name of output CSV file')
parser.add_argument('--agg-csv-filename', type=str,
                    help='Name of aggregate (mean) output CSV file')
parser.add_argument('--simple-columns', action='store_true', default=False,
                    help='Only output a restricted set of the more useful columns')

parser.add_argument('--downsample', type=int, default=1,
                    help='Amount to downsample the mosaic in longitude')

parser.add_argument('--longitude-start', type=float, default=0.,
                    help='The starting longitude for all operations')
parser.add_argument('--longitude-end', type=float, default=360.,
                    help='The ending longitude for all operations')

parser.add_argument('--ew-inner-radius', type=int, default=140220-750,
                    help='The inner radius of the range for EW computation; '
                         'this generally excludes the background region')
parser.add_argument('--ew-outer-radius', type=int, default=140220+750,
                    help='The outer radius of the range for EW computation; '
                         'this generally excludes the background region')

parser.add_argument('--ew-core-inner-radius', type=int, default=None,
                    help='The inner radius of the core for a 3-zone model')
parser.add_argument('--ew-core-outer-radius', type=int, default=None,
                    help='The outer radius of the core for a 3-zone model')
parser.add_argument('--tau', type=float, default=None,
                    help='Tau to use computing 3-zone EW and phase-normalization')
parser.add_argument('--phase-curve-params', type=str, default=None,
                    # default='0.634,1.874,-0.026,0.876',
                    help='The H-G parameters for the phase curve '
                         '(scale1, g1, scale2, g2)')

parser.add_argument('--include-quantiles', action='store_true', default=False,
                    help='Include (Normal) EW Mean/Std for quantiles at 15/25/50/75/85%')

parser.add_argument('--radial-step-size', type=int, default=None,
                    help='Radial step size for multiple small radial steps')
parser.add_argument('--radial-step-inner-radius', type=int, default=None,
                    help='Radial step inner radius for multiple small radial steps')
parser.add_argument('--radial-step-outer-radius', type=int, default=None,
                    help='Radial step outer radius for multiple small radial steps')

parser.add_argument('--compute-widths', action='store_true', default=False,
                    help='Compute the widths for each slice')
parser.add_argument('--widths-frac-mode', action='store_true', default=False,
                    help='Compute widths in fractional mode')
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

parser.add_argument('--compute-core-center', action='store_true', default=False,
                    help='Compute the core location based on the brightest pixel')
parser.add_argument('--compute-moon-info', action='store_true', default=False,
                    help='Compute the distance and longitude of Prometheus and Pandora')

f_ring.add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)

f_ring.init(arguments)

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

if arguments.compute_moon_info:
    # Requires SPICE kernels so don't import unless needed
    import f_ring_util.moons as moons


##########################################################################################

def max_range(a, fracs, mean_brightness=None):
    assert False
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
        mean_brightness = ma.mean(a[mask])
    ret = []
    for frac in fracs:
        target = mean_brightness * frac
        wing1_pos = np.argmax(a >= target)
        wing2_pos = len(a) - np.argmax(a[::-1] >= target) - 1
        # print(wing1_pos, a[wing1_pos], wing2_pos, a[wing2_pos])
        ret.append((wing1_pos, wing2_pos))
    return ret

def width_from_abs(a, abs_vals):
    assert False
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

if arguments.radial_step_size is not None:
    radial_step_inner_radius = arguments.radial_step_inner_radius
    if radial_step_inner_radius is None:
        radial_step_inner_radius = arguments.ew_inner_radius
    radial_step_outer_radius = arguments.radial_step_outer_radius
    if radial_step_outer_radius is None:
        radial_step_outer_radius = arguments.ew_outer_radius
    num_radial_steps = int((radial_step_outer_radius - radial_step_inner_radius) /
                           arguments.radial_step_size)+1
    radial_step_pix = int(arguments.radial_step_size / arguments.radius_resolution)
    radial_step_inner_radius_pix = int((radial_step_inner_radius -
                                        arguments.radius_inner_delta -
                                        arguments.ring_radius) /
                                       arguments.radius_resolution) - ring_lower_limit
    print(f'Num radial steps: {num_radial_steps} by {radial_step_pix} '
          f'starting at {radial_step_inner_radius_pix}')
else:
    num_radial_steps = None


# INITIALIZE THE OUTPUT CSV FILE WITH HEADERS

csv_fp = None
if arguments.output_csv_filename:
    assert (arguments.slice_size == 0 or
            360 / arguments.slice_size == int(360 / arguments.slice_size)), \
                'Slice size must divide evenly into 360'
    csv_fp = open(arguments.output_csv_filename, 'w', newline='')
    writer = csv.writer(csv_fp)
    hdr = ['Observation', 'Slice#', 'Num Data', 'Date',
           'Min Long', 'Max Long']
    if not arguments.simple_columns:
        hdr += ['Mean Long',
                'Min Inertial Long', 'Max Inertial Long', 'Mean Inertial Long',
                'Min Long of Pericenter', 'Max Long of Pericenter', 'Mean Long of Pericenter',
                'Min True Anomaly', 'Max True Anomaly', 'Mean True Anomaly',
                'Min Radial Res', 'Max Radial Res']
    hdr += ['Mean Radial Res']
    if not arguments.simple_columns:
        hdr += ['Min Angular Res', 'Max Angular Res']
    hdr += ['Mean Angular Res']
    if not arguments.simple_columns:
        hdr += ['Min Phase', 'Max Phase']
    hdr += ['Mean Phase']
    if not arguments.simple_columns:
        hdr += ['Min Emission', 'Max Emission']
    hdr += ['Mean Emission', 'Incidence']
    if not arguments.simple_columns:
        hdr += ['% Coverage',
                'EW Median', 'EW Mean', 'EW Std',
                'Normal EW Median']
    hdr += ['Normal EW Mean', 'Normal EW Std']
    if arguments.include_quantiles:
        hdr += ['EW Mean 15', 'EW Std 15', 'Normal EW Mean 15', 'Normal EW Std 15',
                'EW Mean 25', 'EW Std 25', 'Normal EW Mean 25', 'Normal EW Std 25',
                'EW Mean 50', 'EW Std 50', 'Normal EW Mean 50', 'Normal EW Std 50',
                'EW Mean 75', 'EW Std 75', 'Normal EW Mean 75', 'Normal EW Std 75',
                'EW Mean 85', 'EW Std 85', 'Normal EW Mean 85', 'Normal EW Std 85']
    if three_zone:
        hdr += ['EWI Mean', 'EWI Std', 'Normal EWI Mean', 'Normal EWI Std']
        hdr += ['EWC Mean', 'EWC Std', 'Normal EWC Mean', 'Normal EWC Std']
        hdr += ['EWO Mean', 'EWO Std', 'Normal EWO Mean', 'Normal EWO Std']
        if arguments.tau is not None:
            hdr += ['Normal EW3Z Mean', 'Normal EW3Z Std',
                    'Normal EW3ZPN Mean', 'Normal EW3ZPN Std']

    if num_radial_steps is not None:
        for radial_step in range(num_radial_steps):
            start_ew = radial_step_inner_radius + radial_step * arguments.radial_step_size
            if not arguments.simple_columns:
                hdr += [f'EW{start_ew} Mean', f'EW{start_ew} Std']
            hdr += [f'Normal EW{start_ew} Mean']
            if not arguments.simple_columns:
                hdr += [f'Normal EW{start_ew} Std']

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

    if arguments.compute_moon_info:
        hdr += ['Pandora Closest Distance',
                'Pandora Closest Long',
                'Pandora Distance',
                'Pandora Long',
                'Pandora Earliest Distance',
                'Pandora Earliest Long',
                'Pandora Latest Distance',
                'Pandora Latest Long',
                'Prometheus Closest Distance',
                'Prometheus Closest Long',
                'Prometheus Distance',
                'Prometheus Long',
                'Prometheus Earliest Distance',
                'Prometheus Earliest Long',
                'Prometheus Latest Distance',
                'Prometheus Latest Long']

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
        metadata = msgpack.unpackb(bkgnd_metadata_fp.read(),
                                   max_str_len=40*1024*1024,
                                   object_hook=msgpack_numpy.decode)
        if 'mean_resolution' in metadata: # Old format
            metadata['mean_radial_resolution'] = res = metadata['mean_resolution']
            del metadata['mean_resolution']
            metadata['mean_angular_resolution'] = np.zeros(res.shape)
        if 'long_mask' in metadata: # Old format
            metadata['long_antimask'] = metadata['long_mask']
            del metadata['long_mask']
    with np.load(bkgnd_sub_mosaic_filename) as npz:
        bsm_img = ma.MaskedArray(**npz)
        bsm_img = ma.masked_equal(bsm_img, -999)

    ds = arguments.downsample

    longitudes = np.degrees(metadata['longitudes'][::ds]).view(ma.MaskedArray)
    orig_longitudes = longitudes.copy()
    radial_resolutions = metadata['mean_radial_resolution'][::ds]
    angular_resolutions = metadata['mean_angular_resolution'][::ds]
    image_numbers = metadata['image_number'][::ds]
    ETs = metadata['time'][::ds]
    emission_angles = metadata['mean_emission'][::ds]
    incidence_angle = metadata['mean_incidence']
    phase_angles = metadata['mean_phase'][::ds]
    inertial_longitudes = f_ring.fring_corotating_to_inertial(longitudes, ETs)
    longitude_of_pericenters = f_ring.fring_longitude_of_pericenter(ETs)
    true_anomalies = f_ring.fring_true_anomaly(inertial_longitudes, ETs)
    # bsm = background-subtracted-mosaic
    bsm_img = bsm_img[:,::ds]

    # Make the submosaic for the radial range desired
    if three_zone:
        restr_bsm_img = bsm_img[ring_lower_limit1:ring_upper_limit3+1,:]
    else:
        restr_bsm_img = bsm_img[ring_lower_limit:ring_upper_limit1,:]
    bad_long = longitudes.data < 0
    if arguments.longitude_start is not None and arguments.longitude_start != 0.:
        bad_long[:int(arguments.longitude_start/arguments.longitude_resolution)] = True
    if arguments.longitude_end is not None and arguments.longitude_end != 360.:
        bad_long[int(arguments.longitude_end/arguments.longitude_resolution)+1:] = True
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
            start_ew = radial_step_inner_radius + radial_step * arguments.radial_step_size
            start_rad_pix = radial_step * radial_step_pix + radial_step_inner_radius_pix
            end_rad_pix = (radial_step+1) * radial_step_pix + radial_step_inner_radius_pix
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
                # Entire slice is bad
                continue
            slice_good_long = ~slice_bad_long
            if (np.sum(slice_good_long) * arguments.longitude_resolution <
                arguments.minimum_slice_coverage):
                # Not enough data in the slice
                continue
            slice_ETs = ETs[slice_start:slice_end][slice_good_long]
            slice_emission_angles = emission_angles[slice_start:slice_end][slice_good_long]
            slice_phase_angles = phase_angles[slice_start:slice_end][slice_good_long]
            slice_radial_resolutions = radial_resolutions[slice_start:slice_end][slice_good_long]
            slice_angular_resolutions = angular_resolutions[slice_start:slice_end][slice_good_long]
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
            slice_min_long = ma.min(slice_longitudes)
            slice_max_long = ma.max(slice_longitudes)
            slice_mean_long = ma.mean(slice_longitudes)

            slice_min_inertial_long = ma.min(slice_inertial_longs)
            slice_max_inertial_long = ma.max(slice_inertial_longs)
            slice_mean_inertial_long = ma.mean(slice_inertial_longs)

            slice_min_long_of_peri = ma.min(slice_longitude_of_pericenters)
            slice_max_long_of_peri = ma.max(slice_longitude_of_pericenters)
            slice_mean_long_of_peri = ma.mean(slice_longitude_of_pericenters)

            slice_min_true_anomaly = ma.min(slice_true_anomalies)
            slice_max_true_anomaly = ma.max(slice_true_anomalies)
            slice_mean_true_anomaly = ma.mean(slice_true_anomalies)

            slice_min_et = ma.min(slice_ETs)
            slice_max_et = ma.max(slice_ETs)
            slice_mean_et = ma.mean(slice_ETs)
            slice_et_date = f_ring.et2utc(slice_min_et)

            slice_min_em = ma.min(slice_emission_angles)
            slice_max_em = ma.max(slice_emission_angles)
            slice_mean_em = ma.mean(slice_emission_angles)

            slice_min_ph = ma.min(slice_phase_angles)
            slice_max_ph = ma.max(slice_phase_angles)
            slice_mean_ph = ma.mean(slice_phase_angles)

            slice_min_rad_res = ma.min(slice_radial_resolutions)
            slice_max_rad_res = ma.max(slice_radial_resolutions)
            slice_mean_rad_res = ma.mean(slice_radial_resolutions)

            slice_min_ang_res = ma.min(slice_angular_resolutions)
            slice_max_ang_res = ma.max(slice_angular_resolutions)
            slice_mean_ang_res = ma.mean(slice_angular_resolutions)

            if slice_min_rad_res > arguments.maximum_slice_radial_resolution:
                continue

            slice_ew_profile = ew_profile[slice_start:slice_end][slice_good_long]
            slice_ew_mean = ma.mean(slice_ew_profile)
            slice_ew_std = ma.std(slice_ew_profile)
            slice_ew_median = ma.median(slice_ew_profile)
            slice_ew_profile_mu = (slice_ew_profile *
                                   np.abs(np.cos(slice_emission_angles)))
            slice_ew_mean_mu = ma.mean(slice_ew_profile_mu)
            slice_ew_std_mu = ma.std(slice_ew_profile_mu)
            slice_ew_median_mu = ma.median(slice_ew_profile_mu)

            if arguments.slice_size == 0:
                slice_ew_profile_sorted = sorted(slice_ew_profile)
                slice_ew_profile_sorted_mu = sorted(slice_ew_profile_mu)
                idx_15_per = int(len(slice_ew_profile) * .15)
                idx_25_per = int(len(slice_ew_profile) * .25)
                idx_50_per = int(len(slice_ew_profile) * .50)
                idx_75_per = int(len(slice_ew_profile) * .75)
                idx_85_per = int(len(slice_ew_profile) * .85)
                slice_ew_mean_15 = ma.mean(slice_ew_profile_sorted[:idx_15_per])
                slice_ew_std_15 = ma.std(slice_ew_profile_sorted[:idx_15_per])
                slice_ew_mean_mu_15 = ma.mean(slice_ew_profile_sorted_mu[:idx_15_per])
                slice_ew_std_mu_15 = ma.std(slice_ew_profile_sorted_mu[:idx_15_per])
                slice_ew_mean_25 = ma.mean(slice_ew_profile_sorted[:idx_25_per])
                slice_ew_std_25 = ma.std(slice_ew_profile_sorted[:idx_25_per])
                slice_ew_mean_mu_25 = ma.mean(slice_ew_profile_sorted_mu[:idx_25_per])
                slice_ew_std_mu_25 = ma.std(slice_ew_profile_sorted_mu[:idx_25_per])
                slice_ew_mean_50 = ma.mean(slice_ew_profile_sorted[:idx_50_per])
                slice_ew_std_50 = ma.std(slice_ew_profile_sorted[:idx_50_per])
                slice_ew_mean_mu_50 = ma.mean(slice_ew_profile_sorted_mu[:idx_50_per])
                slice_ew_std_mu_50 = ma.std(slice_ew_profile_sorted_mu[:idx_50_per])
                slice_ew_mean_75 = ma.mean(slice_ew_profile_sorted[:idx_75_per])
                slice_ew_std_75 = ma.std(slice_ew_profile_sorted[:idx_75_per])
                slice_ew_mean_mu_75 = ma.mean(slice_ew_profile_sorted_mu[:idx_75_per])
                slice_ew_std_mu_75 = ma.std(slice_ew_profile_sorted_mu[:idx_75_per])
                slice_ew_mean_85 = ma.mean(slice_ew_profile_sorted[:idx_85_per])
                slice_ew_std_85 = ma.std(slice_ew_profile_sorted[:idx_85_per])
                slice_ew_mean_mu_85 = ma.mean(slice_ew_profile_sorted_mu[:idx_85_per])
                slice_ew_std_mu_85 = ma.std(slice_ew_profile_sorted_mu[:idx_85_per])

            # if slice_ew_mean <= 0:
            #     print(obs_id, slice_num, 'EW Mean < 0')
            #     continue

            row = [obs_id, slice_num, np.sum(~slice_bad_long), slice_et_date,
                   np.round(slice_min_long, 2),
                   np.round(slice_max_long, 2)]
            if not arguments.simple_columns:
                row += [np.round(slice_mean_long, 2),
                        np.round(slice_min_inertial_long, 3),
                        np.round(slice_max_inertial_long, 3),
                        np.round(slice_mean_inertial_long, 3),
                        np.round(slice_min_long_of_peri, 3),
                        np.round(slice_max_long_of_peri, 3),
                        np.round(slice_mean_long_of_peri, 3),
                        np.round(slice_min_true_anomaly, 3),
                        np.round(slice_max_true_anomaly, 3),
                        np.round(slice_mean_true_anomaly, 3),
                        np.round(slice_min_rad_res, 8),
                        np.round(slice_max_rad_res, 8)]
            row += [np.round(slice_mean_rad_res, 8)]
            if not arguments.simple_columns:
                row += [np.round(slice_min_ang_res, 8),
                        np.round(slice_max_ang_res, 8)]
            row += [np.round(slice_mean_ang_res, 8)]
            if not arguments.simple_columns:
                row += [np.round(np.degrees(slice_min_ph), 8),
                        np.round(np.degrees(slice_max_ph), 8)]
            row += [np.round(np.degrees(slice_mean_ph), 8)]
            if not arguments.simple_columns:
                row += [np.round(np.degrees(slice_min_em), 8),
                        np.round(np.degrees(slice_max_em), 8)]
            row += [np.round(np.degrees(slice_mean_em), 8),
                    np.round(np.degrees(incidence_angle), 8)]
            if not arguments.simple_columns:
                row += [np.round(percentage_ew_ok, 2),
                        np.round(slice_ew_median, 8),
                        np.round(slice_ew_mean, 8),
                        np.round(slice_ew_std, 8),
                        np.round(slice_ew_median_mu, 8)]
            row += [np.round(slice_ew_mean_mu, 8),
                    np.round(slice_ew_std_mu, 8)]

            if arguments.include_quantiles:
                row += [np.round(slice_ew_mean_15, 8),
                        np.round(slice_ew_std_15, 8),
                        np.round(slice_ew_mean_mu_15, 8),
                        np.round(slice_ew_std_mu_15, 8),
                        np.round(slice_ew_mean_25, 8),
                        np.round(slice_ew_std_25, 8),
                        np.round(slice_ew_mean_mu_25, 8),
                        np.round(slice_ew_std_mu_25, 8),
                        np.round(slice_ew_mean_50, 8),
                        np.round(slice_ew_std_50, 8),
                        np.round(slice_ew_mean_mu_50, 8),
                        np.round(slice_ew_std_mu_50, 8),
                        np.round(slice_ew_mean_75, 8),
                        np.round(slice_ew_std_75, 8),
                        np.round(slice_ew_mean_mu_75, 8),
                        np.round(slice_ew_std_mu_75, 8),
                        np.round(slice_ew_mean_85, 8),
                        np.round(slice_ew_std_85, 8),
                        np.round(slice_ew_mean_mu_85, 8),
                        np.round(slice_ew_std_mu_85, 8)]

            if three_zone:
                slice_ew_profile1 = ew_profile1[slice_start:slice_end][slice_good_long]
                slice_ew_mean1 = ma.mean(slice_ew_profile1)
                slice_ew_std1 = ma.std(slice_ew_profile1)
                slice_ew_profile_mu1 = (slice_ew_profile1 *
                                        np.abs(np.cos(slice_emission_angles)))
                slice_ew_mean_mu1 = ma.mean(slice_ew_profile_mu1)
                slice_ew_std_mu1 = ma.std(slice_ew_profile_mu1)

                row += [np.round(slice_ew_mean1, 8),
                        np.round(slice_ew_std1, 8),
                        np.round(slice_ew_mean_mu1, 8),
                        np.round(slice_ew_std_mu1, 8)]

                slice_ew_profile2 = ew_profile2[slice_start:slice_end][slice_good_long]
                slice_ew_mean2 = ma.mean(slice_ew_profile2)
                slice_ew_std2 = ma.std(slice_ew_profile2)
                slice_ew_profile_mu2 = (slice_ew_profile2 *
                                        np.abs(np.cos(slice_emission_angles)))
                slice_ew_mean_mu2 = ma.mean(slice_ew_profile_mu2)
                slice_ew_std_mu2 = ma.std(slice_ew_profile_mu2)

                row += [np.round(slice_ew_mean2, 8),
                        np.round(slice_ew_std2, 8),
                        np.round(slice_ew_mean_mu2, 8),
                        np.round(slice_ew_std_mu2, 8)]

                slice_ew_profile3 = ew_profile3[slice_start:slice_end][slice_good_long]
                slice_ew_mean3 = ma.mean(slice_ew_profile3)
                slice_ew_std3 = ma.std(slice_ew_profile3)
                slice_ew_profile_mu3 = (slice_ew_profile3 *
                                        np.abs(np.cos(slice_emission_angles)))
                slice_ew_mean_mu3 = ma.mean(slice_ew_profile_mu3)
                slice_ew_std_mu3 = ma.std(slice_ew_profile_mu3)

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
                    slice_ew_mean_3z = ma.mean(slice_ew_profile_3z)
                    slice_ew_std_3z = ma.std(slice_ew_profile_3z)
                    slice_ew_profile_3z_pn = ew_profile_3z_pn[slice_start:slice_end][slice_good_long]
                    slice_ew_mean_3z_pn = ma.mean(slice_ew_profile_3z_pn)
                    slice_ew_std_3z_pn = ma.std(slice_ew_profile_3z_pn)
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
                    slice_step_ew_mean = ma.mean(slice_step_ew_profile)
                    slice_step_ew_std = ma.std(slice_step_ew_profile)
                    slice_step_ew_profile_mu = (slice_step_ew_profile *
                                            np.abs(np.cos(slice_emission_angles)))
                    slice_step_ew_mean_mu = ma.mean(slice_step_ew_profile_mu)
                    slice_step_ew_std_mu = ma.std(slice_step_ew_profile_mu)
                    total_step_ew_mean += slice_step_ew_mean
                    if not arguments.simple_columns:
                        row += [np.round(slice_step_ew_mean, 8),
                                np.round(slice_step_ew_std, 8)]
                    row += [np.round(slice_step_ew_mean_mu, 8)]
                    if not arguments.simple_columns:
                        row += [np.round(slice_step_ew_std_mu, 8)]

            if arguments.compute_widths:
                row += [np.round(ma.mean(slice_w1), 3),
                        np.round(ma.std(slice_w1), 3),
                        np.round(ma.mean(slice_w2), 3),
                        np.round(ma.std(slice_w2), 3),
                        np.round(ma.mean(slice_w3), 3),
                        np.round(ma.std(slice_w3), 3),
                        np.round(ma.mean(slice_w1i), 3),
                        np.round(ma.std(slice_w1i), 3),
                        np.round(ma.mean(slice_w1o), 3),
                        np.round(ma.std(slice_w1o), 3),
                        np.round(ma.mean(slice_w2i), 3),
                        np.round(ma.std(slice_w2i), 3),
                        np.round(ma.mean(slice_w2o), 3),
                        np.round(ma.std(slice_w2o), 3),
                        np.round(ma.mean(slice_w3i), 3),
                        np.round(ma.std(slice_w3i), 3),
                        np.round(ma.mean(slice_w3o), 3),
                        np.round(ma.std(slice_w3o), 3)]

            if arguments.compute_core_center:
                row += [np.round(ma.median(slice_core_centers), 3),
                        np.round(ma.mean(slice_core_centers), 3),
                        np.round(ma.std(slice_core_centers), 3)]

            if arguments.compute_moon_info:
                def _is_in_close(val, arr):
                    for x in arr:
                        if math.isclose(val, x):
                            return True
                    return False

                # These are the original longitudes without anything being masked
                # out because we don't want to use the radial slice to compute an EW
                slice_orig_longitudes = orig_longitudes[slice_start:slice_end]

                # Closest the moon gets to the ring during the upcoming orbit
                (pandora_closest_dist,
                 pandora_closest_long) = moons.pandora_close_approach(slice_min_et)
                (prometheus_closest_dist,
                 prometheus_closest_long) = moons.prometheus_close_approach(slice_min_et)

                (pandora_earliest_dist,
                 pandora_earliest_long) = moons.saturn_to_pandora_corot(slice_min_et)
                (pandora_latest_dist,
                 pandora_latest_long) = moons.saturn_to_pandora_corot(slice_max_et)
                (prometheus_earliest_dist,
                 prometheus_earliest_long) = moons.saturn_to_prometheus_corot(slice_min_et)
                (prometheus_latest_dist,
                 prometheus_latest_long) = moons.saturn_to_prometheus_corot(slice_max_et)

                # Find the position of the moon in this slice, if it's there at all
                unique_ets = set(slice_ETs)
                for unique_et in unique_ets:
                    pandora_dist, pandora_long = moons.saturn_to_pandora_corot(unique_et)
                    longs_for_et = [slice_orig_longitudes[i]
                                        for i in range(len(slice_longitudes))
                                            if slice_ETs[i] == unique_et]
                    # Round to the nearest longitude increment
                    pandora_long = (int(pandora_long / arguments.longitude_resolution) *
                                    arguments.longitude_resolution)
                    if _is_in_close(pandora_long, longs_for_et):
                        inertial_long = f_ring.fring_corotating_to_inertial(
                            pandora_long, unique_et)
                        pandora_long = np.round(pandora_long, 3)
                        pandora_dist = np.round(pandora_dist, 3)
                        break
                else:
                    pandora_dist = '--'
                    pandora_long = '--'
                for unique_et in sorted(unique_ets):
                    prometheus_dist, prometheus_long = moons.saturn_to_prometheus_corot(unique_et)
                    longs_for_et = [slice_orig_longitudes[i]
                                        for i in range(len(slice_longitudes))
                                            if slice_ETs[i] == unique_et]
                    # Round to the nearest longitude increment
                    prometheus_long = (int(prometheus_long / arguments.longitude_resolution) *
                                       arguments.longitude_resolution)
                    # print('***', unique_et, prometheus_long,
                    #       min(longs_for_et), max(longs_for_et))
                    if _is_in_close(prometheus_long, longs_for_et):
                        inertial_long = f_ring.fring_corotating_to_inertial(
                            prometheus_long, unique_et)
                        prometheus_long = np.round(prometheus_long, 3)
                        prometheus_dist = np.round(prometheus_dist, 3)
                        break
                else:
                    prometheus_dist = '--'
                    prometheus_long = '--'
                row += [np.round(pandora_closest_dist, 3),
                        np.round(pandora_closest_long, 3),
                        pandora_dist,
                        pandora_long,
                        np.round(pandora_earliest_dist, 3),
                        np.round(pandora_earliest_long, 3),
                        np.round(pandora_latest_dist, 3),
                        np.round(pandora_latest_long, 3),
                        np.round(prometheus_closest_dist, 3),
                        np.round(prometheus_closest_long, 3),
                        prometheus_dist,
                        prometheus_long,
                        np.round(prometheus_earliest_dist, 3),
                        np.round(prometheus_earliest_long, 3),
                        np.round(prometheus_latest_dist, 3),
                        np.round(prometheus_latest_long, 3)]

            writer.writerow(row)

    if arguments.plot_results or arguments.save_plots:
        fig = plt.figure(figsize=(12, 8))
        plt.suptitle(obs_id)
        if arguments.compute_widths:
            ax = fig.add_subplot(311)
        else:
            ax = fig.add_subplot(111)
        plt.plot(longitudes, ew_profile, label='Full')
        if three_zone:
            plt.plot(longitudes, ew_profile1, label='Inner')
            plt.plot(longitudes, ew_profile2, label='Core')
            plt.plot(longitudes, ew_profile3, label='Outer')
            plt.legend()
        plt.xlim(0, 360)
        plt.ylabel('EW (km)')
        plt.xlabel('Longitude (degrees)')

        if arguments.compute_widths:
            gamma = 0.5
            blackpoint = max(ma.min(restr_bsm_img[:, ~bad_long]), 0)
            whitepoint_ignore_frac = 0.995
            img_sorted = sorted(list(restr_bsm_img[:, ~bad_long].flatten()))
            whitepoint = img_sorted[np.clip(int(len(img_sorted)*
                                                whitepoint_ignore_frac),
                                            0, len(img_sorted)-1)]
            greyscale_img = np.floor((ma.maximum(restr_bsm_img-blackpoint, 0)/
                                      (whitepoint-blackpoint))**gamma*256)
            greyscale_img = np.clip(greyscale_img, 0, 255)
            ax = fig.add_subplot(312)
            plt.imshow(greyscale_img[::-1, :],
                       extent=(0, 360,
                               arguments.ew_inner_radius - arguments.ring_radius,
                               arguments.ew_outer_radius - arguments.ring_radius),
                       aspect='auto',
                       cmap='gray', vmin=0, vmax=255)
            plt.plot(longitudes, widths1[:,0], color='cyan', lw=1, alpha=0.5)
            plt.plot(longitudes, widths1[:,1], color='cyan', lw=1, alpha=0.5)
            plt.plot(longitudes, widths2[:,0], color='orange', lw=1, alpha=0.5)
            plt.plot(longitudes, widths2[:,1], color='orange', lw=1, alpha=0.5)
            plt.plot(longitudes, widths3[:,0], color='red', lw=1, alpha=0.5)
            plt.plot(longitudes, widths3[:,1], color='red', lw=1, alpha=0.5)
            # plt.legend()
            plt.xlim(0, 360)
            plt.ylim(arguments.ew_inner_radius - arguments.ring_radius,
                     arguments.ew_outer_radius - arguments.ring_radius)
            plt.ylabel('Core offset (km)')
            plt.xlabel('Longitude (degrees)')
            ax = fig.add_subplot(313)
            plt.plot(longitudes, w1, color='cyan', label='W1')
            plt.plot(longitudes, w2, color='orange', label='W2')
            plt.plot(longitudes, w3, color='red', label='W3')
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
