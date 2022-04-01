# python dump_mosaic_stats.py 0 data_files/good_qual_full.csv
# python dump_mosaic_stats.py 1 data_files/good_qual_1deg.csv
# python dump_mosaic_stats.py 10 data_files/good_qual_10deg.csv

# Read in all background-subtracted mosaics, collect statistics, and dump them into
# a CSV file.
# - We only pay attention to radial slices that have NO bad pixels
# - The EW profile is split into "slices" of size slice_size
# - Each slice gets its own statistics
#   - The percent longtiude coverage is the same for all slices of a given mosaic

import csv
import julian
import matplotlib.pyplot as plt
import numpy as np
import sys

from mosaic_util import *

if len(sys.argv) != 3:
    print('Usage: dump_mosaic_stats.py <slice_size> <output_csv_file>')
    sys.exit(-1)

slice_size = int(sys.argv[1])
output_csv_filename = sys.argv[2]

assert slice_size == 0 or (360 % slice_size) == 0

EW_DIR_ROOT = os.environ['EW_DIR']

root_list = get_root_list(EW_DIR_ROOT)

csv_fp = open(output_csv_filename, 'w')
writer = csv.writer(csv_fp)
writer.writerow(['Observation', 'Slice#', 'Date',
                 'Min Res', 'Max Res', 'Mean Res',
                 'Min Phase', 'Max Phase', 'Mean Phase',
                 'Min Emission', 'Max Emission', 'Mean Emission',
                 'Incidence',
                 '% Coverage',
                 'EW', 'EW Std', 'Normal EW', 'Normal EW Std'])

for root in root_list:
    if '166RI' in root or '237RI' in root:
        print('Skipping', root)
        continue
    print(root)

    ew_profile = read_ew(root)
    ew_metadata = read_ew_metadata(root)
    longitudes = ew_metadata['longitudes']
    valid_longitudes = get_ew_valid_longitudes(ew_profile, ew_metadata)

    longitude_resolution = np.degrees(ew_metadata['longitude_resolution'])
    radius_resolution = ew_metadata['radius_resolution']

    num_valid_longitudes = np.sum(valid_longitudes)
    percent_coverage = int(num_valid_longitudes * 100 / len(longitudes))

    print(num_valid_longitudes, percent_coverage)

    # Just assume all the images are part of the same obsid
    obsid = ew_metadata['obsid_list'][0]

    incidence_angle = ew_metadata['incidence_angle']

    ETs = ew_metadata['ETs'][valid_longitudes]
    emission_angles = ew_metadata['emission_angles'][valid_longitudes]
    phase_angles = ew_metadata['phase_angles'][valid_longitudes]
    resolutions = ew_metadata['resolutions'][valid_longitudes]

    ew_profile = ew_profile[valid_longitudes]

    slice_size_in_longitudes = int(slice_size / longitude_resolution)
    if slice_size == 0:
        num_slices = 1
        slice_size_in_longitudes = num_valid_longitudes
    else:
        num_slices = num_valid_longitudes // slice_size_in_longitudes
    for slice_num in range(num_slices):
        slice_start = slice_num * slice_size_in_longitudes
        slice_end = (slice_num+1) * slice_size_in_longitudes
        slice_ETs = ETs[slice_start:slice_end]
        slice_emission_angles = emission_angles[slice_start:slice_end]
        slice_phase_angles = phase_angles[slice_start:slice_end]
        slice_resolutions = resolutions[slice_start:slice_end]
        slice_ew_profile = ew_profile[slice_start:slice_end]

        min_et = np.min(slice_ETs)
        max_et = np.max(slice_ETs)
        mean_et = (min_et+max_et) / 2
        et_date = julian.iso_from_tai(julian.tai_from_tdb(min_et))

        min_em = np.min(slice_emission_angles)
        max_em = np.max(slice_emission_angles)
        mean_em = (min_em+max_em) / 2

        min_ph = np.min(slice_phase_angles)
        max_ph = np.max(slice_phase_angles)
        mean_ph = (min_ph+max_ph) / 2

        min_res = np.min(slice_resolutions)
        max_res = np.max(slice_resolutions)
        mean_res = (min_res+max_res) / 2

        ew_mean = np.mean(slice_ew_profile)
        ew_std = np.std(slice_ew_profile)
        slice_ew_profile_mu = (slice_ew_profile *
                               np.abs(np.cos(slice_emission_angles)))
        ew_mean_mu = np.mean(slice_ew_profile_mu)
        ew_std_mu = np.std(slice_ew_profile_mu)

        if ew_mean <= 0:
            print(root, slice_num, 'EW Mean < 0')
            continue
        writer.writerow([obsid, slice_num, et_date,
                         np.round(min_res, 3),
                         np.round(max_res, 3),
                         np.round(mean_res, 3),
                         np.round(np.degrees(min_ph), 3),
                         np.round(np.degrees(max_ph), 3),
                         np.round(np.degrees(mean_ph), 3),
                         np.round(np.degrees(min_em), 3),
                         np.round(np.degrees(max_em), 3),
                         np.round(np.degrees(mean_em), 3),
                         np.round(np.degrees(incidence_angle), 3),
                         percent_coverage,
                         np.round(ew_mean, 5),
                         np.round(ew_std, 5),
                         np.round(ew_mean_mu, 5),
                         np.round(ew_std_mu, 5)])

csv_fp.close()
