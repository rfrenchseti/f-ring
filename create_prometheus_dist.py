##########################################################################################
# Compute the distance at closest approach between Prometheus and the F ring core.
# This can be done for a range of times or as supplemental data for observations.
#
# Saturn pole referenced to 2007-01-01
# Prometheus orbit from SPICE kernel sat393.bsp
# F ring orbit from Albers 2012
##########################################################################################

import argparse
import csv
import math
import pickle
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import f_ring_util

import cspyce
import julian


cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
   cmd_line = []

parser = argparse.ArgumentParser()

parser.add_argument('--output-csv-filename', type=str,
                    help='Name of output CSV file for observation-specific dates')
parser.add_argument('--historical-csv-filename', type=str,
                    help='Name of historical output CSV file')
parser.add_argument('--historical-start-date', type=str, default='1978-01-01',
                    help='Start date for historical output')
parser.add_argument('--historical-end-date', type=str, default='2018-01-01',
                    help='Start date for historical output')
parser.add_argument('--historical-step', type=float, default=30*6,
                    help='Number of days between historical outputs')
parser.add_argument('--plot-results', action='store_true', default=False,
                    help='Plot the distance results')
parser.add_argument('--save-plots', action='store_true', default=False,
                    help='Same as --plot-results but save plots to disk instead')

f_ring_util.add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)


kdir = '/home/rfrench/DS/Shared/OOPS-Resources/SPICE'
cspyce.furnsh(os.path.join(kdir, 'General/LSK/naif0012.tls'))
cspyce.furnsh(os.path.join(kdir, 'General/SPK/de438.bsp'))
cspyce.furnsh(os.path.join(kdir, 'Saturn/SPK/sat393.bsp'))
cspyce.furnsh(os.path.join(kdir, 'General/PCK/pck00010_edit_v01.tpc'))

SATURN_ID     = cspyce.bodn2c('SATURN')
PANDORA_ID    = cspyce.bodn2c('PANDORA')
PROMETHEUS_ID = cspyce.bodn2c('PROMETHEUS')

REFERENCE_ET = cspyce.utc2et('2007-01-01') # For Saturn pole
j2000_to_iau_saturn = cspyce.pxform('J2000', 'IAU_SATURN', REFERENCE_ET)

saturn_z_axis_in_j2000 = cspyce.mtxv(j2000_to_iau_saturn, (0,0,1))
saturn_x_axis_in_j2000 = cspyce.ucrss((0,0,1), saturn_z_axis_in_j2000)

J2000_TO_SATURN = cspyce.twovec(saturn_z_axis_in_j2000, 3,
                                saturn_x_axis_in_j2000, 1)

def saturn_to_prometheus(et):
    (prometheus_j2000, lt) = cspyce.spkez(PROMETHEUS_ID, et, 'J2000', 'LT+S', SATURN_ID)
    prometheus_sat = np.dot(J2000_TO_SATURN, prometheus_j2000[0:3])
    dist = np.sqrt(prometheus_sat[0]**2.+prometheus_sat[1]**2.+prometheus_sat[2]**2.)
    longitude = math.atan2(prometheus_sat[1], prometheus_sat[0])
    return (dist, longitude)

def prometheus_close_approach(min_et, min_et_long):
    def compute_r(a, e, arg): # Takes argument of pericenter
        return a*(1-e**2.) / (1+e*np.cos(arg))
    def compute_r_fring(arg):
        return compute_r(f_ring_util.FRING_A, f_ring_util.FRING_E, arg)

    # Find time for 0 long
    et_min = min_et - min_et_long / f_ring_util.FRING_MEAN_MOTION * 86400.
    # Find time for 360 long
    et_max = min_et + 2*np.pi / f_ring_util.FRING_MEAN_MOTION * 86400
    # Find the longitude at the point of closest approach
    min_dist = 1e38
    for et in np.arange(et_min, et_max, 10): # Step by minute
        prometheus_dist, prometheus_longitude = saturn_to_prometheus(et)
        long_peri_fring = ((et-f_ring_util.FRING_ORBIT_EPOCH)/86400 *
                           f_ring_util.FRING_DW +
                           f_ring_util.FRING_W0) % (np.pi*2)
        fring_r = compute_r_fring(prometheus_longitude-long_peri_fring)
        if abs(fring_r-prometheus_dist) < min_dist:
            min_dist = abs(fring_r-prometheus_dist)
            min_dist_long = prometheus_longitude
            min_dist_et = et
    min_dist_long = f_ring_util.fring_inertial_to_corotating(min_dist_long, min_dist_et)
    return min_dist, min_dist_long

if False:
    # Debugging code to plot the orbit of Prometheus
    start_et = f_ring_util.utc2et('2007-01-01')
    et_list = []
    dist_list = []
    long_list = []
    for et in np.arange(start_et, start_et+15*60*60, 60):
        dist = prometheus_close_approach(et)
        et_list.append(et)
        dist_list.append(dist)
        long_list.append(np.degrees(long))
    # plt.plot(et_list, dist_list)
    plt.plot(et_list, long_list)
    plt.show()


if arguments.historical_csv_filename:
    csv_fp = open(arguments.historical_csv_filename, 'w')
    writer = csv.writer(csv_fp)
    hdr = ['Date', 'Prometheus Distance']
    writer.writerow(hdr)

    if arguments.plot_results:
        et_list = []
        dist_list = []

    et1 = f_ring_util.utc2et(arguments.historical_start_date)
    et2 = f_ring_util.utc2et(arguments.historical_end_date)
    for et in np.arange(et1, et2, arguments.historical_step*86400):
        min_dist, min_long = prometheus_close_approach(et, 0)
        date = f_ring_util.et2utc(et)
        print(date)
        if arguments.plot_results:
            et_list.append(np.datetime64(date))
            dist_list.append(min_dist)
        row = [f_ring_util.et2utc(et), np.round(min_dist, 3)]
        writer.writerow(row)
    csv_fp.close()

    if arguments.plot_results:
        plt.plot(et_list, dist_list)
        plt.show()

if arguments.output_csv_filename:
    csv_fp = open(arguments.output_csv_filename, 'w')
    writer = csv.writer(csv_fp)
    hdr = ['Observation', 'Date', 'Prometheus Min Dist', 'Prometheus Long']
    writer.writerow(hdr)

for obs_id in f_ring_util.enumerate_obsids(arguments):
    (bkgnd_sub_mosaic_filename,
     bkgnd_sub_mosaic_metadata_filename) = f_ring_util.bkgnd_sub_mosaic_paths(
        arguments, obs_id)

    if (not os.path.exists(bkgnd_sub_mosaic_filename) or
        not os.path.exists(bkgnd_sub_mosaic_metadata_filename)):
        print('NO FILE', bkgnd_sub_mosaic_filename,
              'OR', bkgnd_sub_mosaic_metadata_filename)
        continue

    with open(bkgnd_sub_mosaic_metadata_filename, 'rb') as bkgnd_metadata_fp:
        metadata = pickle.load(bkgnd_metadata_fp, encoding='latin1')

    longitudes = metadata['longitudes']
    good_long = longitudes >= 0
    mean_et = np.mean(metadata['ETs'][good_long])
    min_et = np.min(metadata['ETs'][good_long])

    min_dist, min_long = prometheus_close_approach(mean_et, 0)
    date_str = f_ring_util.et2utc(min_et)
    print(f'{obs_id:30s}: {date_str} {min_dist:.3f}')

    if arguments.output_csv_filename:
        row = [obs_id,
               date_str,
               np.round(min_dist, 3),
               np.round(np.degrees(min_long), 3)]

        writer.writerow(row)

if arguments.output_csv_filename:
    csv_fp.close()
