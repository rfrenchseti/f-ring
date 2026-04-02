import argparse
import csv
import sys

import numpy as np

import julian

from ring.ring_util import (mosaic_paths,
                            ring_add_parser_arguments,
                            ring_init,
                            read_mosaic,
                            ring_enumerate_files)

cmd_line = sys.argv[1:]

parser = argparse.ArgumentParser()

ring_add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)

rows = []

for ring_type in ('FMOVIE',
                  #'FMOVIE_BAD', 'FMOVIE_LOW_COV', 'FMOVIE_LOW_RES'
                  ):
    arguments.ring_type = ring_type
    ring_init(arguments)
    for obsid, image_name, image_path in ring_enumerate_files(arguments,
                                                              yield_obsid_only=True):
        data_path, metadata_path = mosaic_paths(arguments, obsid)
        try:
            metadata = read_mosaic(data_path, metadata_path)
        except FileNotFoundError:
            print('Not found:', obsid)
            continue

        long_mask = metadata['long_antimask']
        min_et = np.min(metadata['time'][long_mask])
        max_et = np.max(metadata['time'][long_mask])
        coverage = np.sum(long_mask) / len(long_mask) * 100
        min_radial_res = np.min(metadata['mean_radial_resolution'][long_mask])
        max_radial_res = np.max(metadata['mean_radial_resolution'][long_mask])
        min_angular_res = np.degrees(np.min(metadata['mean_angular_resolution'][long_mask]))
        max_angular_res = np.degrees(np.max(metadata['mean_angular_resolution'][long_mask]))
        phase = np.degrees(np.median(metadata['mean_phase'][long_mask]))
        em = np.degrees(np.median(metadata['mean_emission'][long_mask]))
        inc = np.degrees(metadata['mean_incidence'])
        row = [obsid]
        row.append(julian.ymdhms_format_from_tai(julian.tai_from_tdb(min_et), sep=' '))
        row.append(julian.ymdhms_format_from_tai(julian.tai_from_tdb(max_et), sep=' '))
        row.append(f'{coverage:.2f}')
        row.append(f'{min_radial_res:.2f}')
        row.append(f'{max_radial_res:.2f}')
        row.append(f'{min_angular_res:.5f}')
        row.append(f'{max_angular_res:.5f}')
        row.append(f'{phase:.2f}')
        row.append(f'{em:.2f}')
        row.append(f'{inc:.2f}')
        rows.append(row)

rows.sort(key=lambda x: x[1])

with open('obs_list.csv', 'w') as fp:
    writer = csv.writer(fp)
    writer.writerow(['OBSID', 'Start Date', 'End Date', 'Coverage',
                     'Min Radial Res', 'Max Radial Res',
                     'Min Angular Res', 'Max Angular Res',
                     'Median Ph', 'Median Em', 'Incidence',])
    for row in rows:
        writer.writerow(row)
