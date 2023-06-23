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
import os
import sys

import msgpack
import msgpack_numpy

import matplotlib.pyplot as plt
import numpy as np

import f_ring_util.f_ring as f_ring
import f_ring_util.moons as moons


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
parser.add_argument('--historical-end-date', type=str, default='2019-01-01',
                    help='Start date for historical output')
parser.add_argument('--historical-step', type=float, default=1,
                    help='Number of days between historical outputs')
parser.add_argument('--plot-results', action='store_true', default=False,
                    help='Plot the distance results')
parser.add_argument('--save-plots', action='store_true', default=False,
                    help='Same as --plot-results but save plots to disk instead')
parser.add_argument('--pandora', action='store_true', default=False,
                    help='Do Pandora instead of Prometheus')

f_ring.add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)


if False:
    # Debugging code to plot the orbit of Prometheus
    start_et = f_ring.utc2et('2007-01-01')
    et_list = []
    dist_list = []
    long_list = []
    for et in np.arange(start_et, start_et+15*60*60, 60):
        if arguments.pandora:
            dist = moons.pandora_close_approach(et)
        else:
            dist = moons.prometheus_close_approach(et)
        et_list.append(et)
        dist_list.append(dist)
        long_list.append(np.degrees(long))
    # plt.plot(et_list, dist_list)
    plt.plot(et_list, long_list)
    plt.show()


if arguments.historical_csv_filename:
    csv_fp = open(arguments.historical_csv_filename, 'w')
    writer = csv.writer(csv_fp)
    if arguments.pandora:
        hdr = ['Date', 'Pandora Min Distance']
    else:
        hdr = ['Date', 'Prometheus Min Distance']
    writer.writerow(hdr)

    if arguments.plot_results:
        et_list = []
        dist_list = []

    et1 = f_ring.utc2et(arguments.historical_start_date)
    et2 = f_ring.utc2et(arguments.historical_end_date)
    for et in np.arange(et1, et2, arguments.historical_step*86400):
        if arguments.pandora:
            min_dist, min_long = moons.pandora_close_approach(et)
        else:
            min_dist, min_long = moons.prometheus_close_approach(et)
        date = f_ring.et2utc(et)
        print(date)
        if arguments.plot_results:
            et_list.append(np.datetime64(date))
            dist_list.append(min_dist)
        row = [f_ring.et2utc(et), np.round(min_dist, 3)]
        writer.writerow(row)
    csv_fp.close()

    if arguments.plot_results:
        plt.plot(et_list, dist_list)
        plt.show()

if arguments.output_csv_filename:
    csv_fp = open(arguments.output_csv_filename, 'w')
    writer = csv.writer(csv_fp)
    if arguments.pandora:
        hdr = ['Observation', 'Date', 'Pandora Min Dist', 'Pandora Long']
    else:
        hdr = ['Observation', 'Date', 'Prometheus Min Dist', 'Prometheus Long']
    writer.writerow(hdr)

for obs_id in f_ring.enumerate_obsids(arguments):
    (bkgnd_sub_mosaic_filename,
     bkgnd_sub_mosaic_metadata_filename) = f_ring.bkgnd_sub_mosaic_paths(
        arguments, obs_id)

    if (not os.path.exists(bkgnd_sub_mosaic_filename) or
        not os.path.exists(bkgnd_sub_mosaic_metadata_filename)):
        print('NO FILE', bkgnd_sub_mosaic_filename,
              'OR', bkgnd_sub_mosaic_metadata_filename)
        continue

    with open(bkgnd_sub_mosaic_metadata_filename, 'rb') as bkgnd_metadata_fp:
        metadata = msgpack.unpackb(bkgnd_metadata_fp.read(),
                                   max_str_len=40*1024*1024,
                                   object_hook=msgpack_numpy.decode)

    longitudes = metadata['longitudes']
    good_long = metadata['long_mask']
    mean_et = np.mean(metadata['time'][good_long])
    min_et = np.min(metadata['time'][good_long])

    if arguments.pandora:
        min_dist, min_long = moons.pandora_close_approach(mean_et)
    else:
        min_dist, min_long = moons.prometheus_close_approach(mean_et)
    date_str = f_ring.et2utc(min_et)
    print(f'{obs_id:30s}: {date_str} {min_dist:.3f}')

    if arguments.output_csv_filename:
        row = [obs_id,
               date_str,
               np.round(min_dist, 3),
               np.round(np.degrees(min_long), 3)]

        writer.writerow(row)

if arguments.output_csv_filename:
    csv_fp.close()
