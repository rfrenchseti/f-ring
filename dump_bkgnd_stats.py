import csv
import julian
import numpy as np
import matplotlib.pyplot as plt

from bkgnd_util import *

BKGND_SUB_DIR = '/cdaps-results/fring/ring_mosaic/bkgnd_sub_mosaic_FMOVIE'

root_list = get_root_list(BKGND_SUB_DIR)

csv_fp = open('mosaic_summary.csv', 'w')
writer = csv.writer(csv_fp)
writer.writerow(['Observation', 'Date',
                 'Min Res', 'Max Res',
                 'Min Phase', 'Max Phase',
                 'Min Emission', 'Max Emission',
                 'Incidence',
                 'Min Longitude', 'Max Longitude',
                 '% Coverage'])

for root in root_list:
    mosaic = read_mosaic(BKGND_SUB_DIR, root)
    metadata = read_metadata(BKGND_SUB_DIR, root)
    longitudes = metadata['longitudes']
    valid_longitudes = get_valid_longitudes(mosaic, metadata)

    longitude_resolution = metadata['longitude_resolution']

    # Starting and ending longitude
    min_long = np.argmin(valid_longitudes) * longitude_resolution
    max_long = ((len(valid_longitudes)-np.argmin(valid_longitudes[::-1])-1) *
                longitude_resolution)

    percent_coverage = int(np.sum(valid_longitudes) * 100 / len(longitudes))


    # Just assume all the images are part of the same obsid
    obsid = metadata['obsid_list'][0]

    min_et = np.min(metadata['ETs'][valid_longitudes])
    et_date = julian.iso_from_tai(julian.tai_from_tdb(min_et))[:10]

    incidence_angle = metadata['incidence_angle']

    emission_angles = metadata['emission_angles'][valid_longitudes]
    min_em = np.min(emission_angles)
    max_em = np.max(emission_angles)

    phase_angles = metadata['phase_angles'][valid_longitudes]
    min_ph = np.min(phase_angles)
    max_ph = np.max(phase_angles)

    resolutions = metadata['resolutions'][valid_longitudes]
    min_res = np.min(resolutions)
    max_res = np.max(resolutions)

    writer.writerow([obsid, et_date,
                     np.round(min_res, 1),
                     np.round(max_res, 1),
                     np.round(np.degrees(min_ph), 1),
                     np.round(np.degrees(max_ph), 1),
                     np.round(np.degrees(min_em), 1),
                     np.round(np.degrees(max_em), 1),
                     np.round(np.degrees(incidence_angle), 1),
                     int(np.degrees(min_long)), int(np.degrees(max_long)),
                     percent_coverage])

    mosaic = valid_mosaic_subset(mosaic, metadata)
    plt.imshow(mosaic)
    plt.show()

csv_fp.close()
