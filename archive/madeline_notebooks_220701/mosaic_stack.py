#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd

import f_ring_util

#define cropping longitudes (deg.)
lon1 = 155
lon2 = 205
#define cropping radii (core at center, 1000km into image) (km.)
rad1 = 800
rad2 = 1200

#H-G function parameters used for normalization
#(currently using params from 3-region fit, same as full-image)
#need params to be g1, scale1, g2, scale2, etc.
params = np.array([0.632, 1.956, -0.015, 1.007])
#params = np.array([0.64389366, 1.78734564, 0.00707584, 0.96626527])

#filepath to save mosaic image to
filepath = '/Users/mlessard/REU_2022/src/plots/median_mosaic_img.png'


cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
   cmd_line = []
    # cmd_line = ['--voyager']

parser = argparse.ArgumentParser()

parser.add_argument('--voyager', action='store_true', default=False,
                    help='Process Voyager profiles instead of Cassini')
parser.add_argument('--ew-inner-radius', type=int, default=None,
                    help="""The inner radius of the range""")
parser.add_argument('--ew-outer-radius', type=int, default=None,
                    help="""The outer radius of the range""")

f_ring_util.add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)

#median-stacking all 150 mosaic images to see if strand double-stranded or not
#read in for each filename (ONE AT A TIME) cropped part of mosaic around core
#(downsample if needed, [::20], e.g. to take every 20 points)
#read in the next one, stack together
#take the median along the stack
#plot the median value stack image to see what they look like all together

#pixel size 18,000 x 400
#(vertical pixels 5km/pixel, horizontal pixels 0.02deg/pixel of longitude)
#try: 50 degrees of longitude (from center), 400-km wide strip
#80 pixels high, 2500 pixels across
#(... still pretty high, but ok to start with?)

#want the actual image data, so look at create_ews / f_ring_util and modify from that to read in mosaics
#loaded in with pickle

#create empty list to store cropped arrays as they are read in
array_list = []

#go through filenames ONE AT A TIME and delete/overwrite once done
#154 total observations used
i = 0
for obs_id in f_ring_util.enumerate_obsids(arguments):
    (bkgnd_sub_mosaic_filename, bkgnd_sub_mosaic_metadata_filename) = f_ring_util.bkgnd_sub_mosaic_paths(arguments, obs_id)

    #(don't load in EW data paths, don't need)

    if (not os.path.exists(bkgnd_sub_mosaic_filename) or
        not os.path.exists(bkgnd_sub_mosaic_metadata_filename)):
        print('NO FILE', bkgnd_sub_mosaic_filename,
              'OR', bkgnd_sub_mosaic_metadata_filename)
        continue

    with open(bkgnd_sub_mosaic_metadata_filename, 'rb') as bkgnd_metadata_fp:
        metadata = pickle.load(bkgnd_metadata_fp, encoding='latin1')
    with np.load(bkgnd_sub_mosaic_filename) as npz:
        bkgnd_sub_mosaic_img = ma.MaskedArray(**npz)

    print(obs_id+' done')
    #take cropped, sampled section of full (401,18000 mosaic)
    #want (80 pixels high, 2500 pixels across) (400 km x 50 deg. longitude)
    cropped_mosaic = bkgnd_sub_mosaic_img[int(rad1/5):int(rad2/5), int(lon1*50):int(lon2*50)]


    #normalize the cropped mosaic before appending to list
    #get out + crop phase angle values + DON'T convert to degrees (automatically in radians)
    cropped_phase_angles = metadata['phase_angles'][int(lon1*50):int(lon2*50)]
    #cropped_phase_angles = np.degrees(metadata['phase_angles'][int(lon1*50):int(lon2*50)])
    #find the H-G func values at each pixel (km)
    cropped_hg_vals = np.array([f_ring_util.hg_func(params, xpts=phase) for phase in cropped_phase_angles])
    #divide each column of mosaic by hg value (1/km)
    for col in range(cropped_mosaic.shape[1]):
        cropped_mosaic[:,col] = cropped_mosaic[:,col]*cropped_hg_vals[col]
    #multiply each value of mosaic by radius resolution to normalize to 1 (unitless)
    cropped_mosaic = cropped_mosaic*metadata['radius_resolution']


    #append normalized mosaic to list
    array_list.append(cropped_mosaic)

    #if i == 0:
    #    break

    i+=1


#stack mosaics and take median along stacked axis
stacked_mosaic = np.ma.dstack(array_list)
median_mosaic = np.ma.median(stacked_mosaic, axis=2)


#save as image
fig, ax = plt.subplots(1, 1, figsize=(10,4))
img = ax.imshow(median_mosaic, cmap='Greys_r', aspect='auto', origin='lower')

x_labels = [r for r in range(lon1,lon2+1,10)]
y_labels = [r for r in range(rad1-1000,rad2-1000+1,50)]
plt.xticks(ticks=np.linspace(0, ((lon2*50)-(lon1*50)), 6), labels=x_labels)
plt.yticks(ticks=np.linspace(0, 80, 9), labels=y_labels)
plt.xlabel('Longitude (deg.)')
plt.ylabel('Offset from core (km)')

plt.axhline(y=40, color='lightskyblue', linestyle='--', lw=1) #uses axes y, not img

plt.savefig(filepath, bbox_inches='tight')
