'''
Created on Oct 9, 2011

@author: rfrench
'''

from optparse import OptionParser
import numpy as np
import numpy.ma as ma
import pickle
import ringutil
import sys
import os.path
import matplotlib.pyplot as plt
import fringobs

fringobs.Init()

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
#    cmd_line = ['ISS_000RI_SATSRCHAP001_PRIME',
#                'ISS_00ARI_SPKMOVPER001_PRIME',
#                'ISS_006RI_LPHRLFMOV001_PRIME',
#                'ISS_007RI_LPHRLFMOV001_PRIME',
#                'ISS_029RF_FMOVIE001_VIMS',
#                'ISS_031RF_FMOVIE001_VIMS',
#                'ISS_032RF_FMOVIE001_VIMS',
#                'ISS_033RF_FMOVIE001_VIMS',
#                'ISS_036RF_FMOVIE001_VIMS',
#                'ISS_036RF_FMOVIE002_VIMS',
#                'ISS_039RF_FMOVIE002_VIMS',
#                'ISS_039RF_FMOVIE001_VIMS',
#                'ISS_041RF_FMOVIE002_VIMS',
#                'ISS_041RF_FMOVIE001_VIMS',
#                'ISS_044RF_FMOVIE001_VIMS',
#                'ISS_051RI_LPMRDFMOV001_PRIME',
#                'ISS_055RF_FMOVIE001_VIMS',
#                'ISS_055RI_LPMRDFMOV001_PRIME',
#                'ISS_057RF_FMOVIE001_VIMS',
#                'ISS_068RF_FMOVIE001_VIMS',
#                'ISS_075RF_FMOVIE002_VIMS',
#                'ISS_083RI_FMOVIE109_VIMS',
#                'ISS_087RF_FMOVIE003_PRIME',
#                'ISS_089RF_FMOVIE003_PRIME',
#                'ISS_100RF_FMOVIE003_PRIME']
#    cmd_line = ['ISS_036RF_FMOVIE001_VIMS']
    cmd_line = ['-a']
    
parser = OptionParser()

ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

def normalized_phase_curve(alpha):
    return np.exp((7.68894494e-07*(alpha**3)-6.05914982e-05*(alpha**2)+6.62353025e-03*alpha-8.33855150e-01))

#
# OLD DATA
#
coll = fringobs.FRingCollection()
coll.AddCassini()

nacmovie_list = coll.GetObslist('CNL', 'C', 'M', None)  # NAC movies limited to "full" - binned
old_phase_angle_list = []
old_mean_ew_list = []
old_std_ew_list = []

for obs in nacmovie_list:
    old_phase_angle_list.append(obs.phase_angle)
    old_mean_ew_list.append(obs.scaled_integral)
    old_std_ew_list.append(obs.stddev_integral)
    
phase_angle_list = []
mean_ew_list = []
std_ew_list = []

for obs_id, image_name, full_path in ringutil.enumerate_files(options, args, obsid_only=True):
    (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
     bkgnd_mask_filename, bkgnd_model_filename,
     bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obs_id)

    (ew_data_filename, ew_mask_filename) = ringutil.ew_paths(options, obs_id)

    if (not os.path.exists(reduced_mosaic_metadata_filename)) or (not os.path.exists(bkgnd_mask_filename+'.npy')):
        continue
    
    reduced_metadata_fp = open(reduced_mosaic_metadata_filename, 'rb')
    mosaic_data = pickle.load(reduced_metadata_fp)
    obsid_list = pickle.load(reduced_metadata_fp)
    image_name_list = pickle.load(reduced_metadata_fp)
    full_filename_list = pickle.load(reduced_metadata_fp)
    reduced_metadata_fp.close()

    (longitudes, resolutions, image_numbers,
     ETs, emission_angles, incidence_angles,
     phase_angles) = mosaic_data

    mosaic_img = np.load(reduced_mosaic_data_filename+'.npy')

    mosaic_img = mosaic_img.view(ma.MaskedArray)
    mosaic_img.mask = np.load(bkgnd_mask_filename+'.npy')
    bkgnd_model = np.load(bkgnd_model_filename+'.npy')
    bkgnd_model = bkgnd_model.view(ma.MaskedArray)
    bkgnd_model.mask = mosaic_img.mask

    corrected_mosaic_img = mosaic_img - bkgnd_model
    
    percentage_ok = float(len(np.where(longitudes >= 0)[0])) / len(longitudes) * 100
    
    ew_data = np.zeros(len(longitudes))
    ew_data = ew_data.view(ma.MaskedArray)
    movie_phase_angles = []
    for idx in range(len(longitudes)):
        if longitudes[idx] < 0 or np.sum(~corrected_mosaic_img.mask[:,idx]) == 0: # Fully masked?
            ew_data[idx] = ma.masked
        else:
            column = corrected_mosaic_img[:,idx]
            # Sometimes there is a reprojection problem at the edge
            # If the very edge is masked, then find the first non-masked value and mask it, too
            if column.mask[-1]:
                colidx = len(column)-1
                while column.mask[colidx]:
                    colidx -= 1
                column[colidx] = ma.masked
                column[colidx-1] = ma.masked
            if column.mask[0]:
                colidx = 0
                while column.mask[colidx]:
                    colidx += 1
                column[colidx] = ma.masked
                column[colidx+1] = ma.masked
            ew = np.sum(ma.compressed(column)) * options.radius_resolution
            ew *= np.abs(np.cos(emission_angles[idx]*np.pi/180))
            
            if ew / normalized_phase_curve(phase_angles[idx]) < 0.1:
                ew_data[idx] = ma.masked
            else:
                ew_data[idx] = ew
            movie_phase_angles.append(phase_angles[idx])
            emission_angle = emission_angles[idx]
            incidence_angle = incidence_angles[idx]

    np.save(ew_data_filename, ew_data.data)
    np.save(ew_mask_filename, ma.getmask(ew_data))
            
    print '%-30s %3d%% P %7.3f E %7.3f I %7.3f EW %8.5f +/- %8.5f' % (obs_id, percentage_ok,
        np.mean(movie_phase_angles), 180-emission_angle, 180-incidence_angle, ma.mean(ew_data), np.std(ew_data))

    phase_angle_list.append(np.mean(movie_phase_angles))
    mean_ew_list.append(np.mean(ew_data))
    std_ew_list.append(np.std(ew_data))

mean_ew_list = np.array(mean_ew_list)
std_ew_list = np.array(std_ew_list)
std_ew_list = np.where(mean_ew_list < std_ew_list, mean_ew_list*.999, std_ew_list)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.errorbar(old_phase_angle_list, old_mean_ew_list, yerr=old_std_ew_list, fmt='o', mfc='red', mec='red', ecolor='red', ms=8, alpha=0.3)
plt.errorbar(phase_angle_list, mean_ew_list, yerr=std_ew_list, fmt='o', mfc='blue', mec='blue', ecolor='blue', ms=8, alpha=0.3)
ax.set_yscale('log')
plt.xlabel(r'Phase angle ($^\circ$)')
plt.ylabel(r'Equivalent width (km) $\times$ $\mu$')
ax.set_xlim(0,180)
ax.set_ylim(.2, 20)
plt.show()
