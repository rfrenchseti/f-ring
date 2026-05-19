import pickle
import numpy as np
import matplotlib.pyplot as plt
import ringutil
import cspice
from optparse import OptionParser
import sys
import os
import numpy.ma as ma

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = [
                '-a',
                '--ignore-voyager',
                '--ignore-bad-obsids'
                ]

parser = OptionParser()
ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)


def smooth_ew(ew_range, smooth_deg):
    long_res = 360./len(ew_range)
    smooth_pix = smooth_deg // long_res // 2
    #smooth the equivalent width range 
    smoothed_ew = ma.zeros(ew_range.shape[0])
    for n in range(len(ew_range)):
                    if ew_data.mask[n]:
                        smoothed_ew[n] = ma.masked
                    else:
                        smoothed_ew[n] = ma.mean(ew_range[max(n-smooth_pix,0):
                                                         min(n+smooth_pix+1,len(ew_range)-1)])
    return smoothed_ew


mean_EWs_list = []
mean_ETs_list = []
mean_emissions = []
mean_incidences = []
mean_phases = []
i = 0
for obs_id, image_name, full_path in ringutil.enumerate_files(options, args, obsid_only=True):
        
        skip_ids = ['ISS_036RF_FMOVIE001_VIMS',
                    'ISS_036RF_FMOVIE002_VIMS',
                    'ISS_039RF_FMOVIE001_VIMS',
                    'ISS_039RF_FMOVIE002_VIMS',
                    'ISS_041RF_FMOVIE001_VIMS',
                    'ISS_041RF_FMOVIE002_VIMS',
                    'ISS_043RF_FMOVIE001_VIMS',
                    'ISS_044RF_FMOVIE001_VIMS',
                    ]
        if obs_id in skip_ids:
            continue
        
        (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
         bkgnd_mask_filename, bkgnd_model_filename,
         bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obs_id)

        (ew_data_filename, ew_mask_filename) = ringutil.ew_paths(options, obs_id)
     
        if (not os.path.exists(reduced_mosaic_metadata_filename)) or (not os.path.exists(ew_data_filename+'.npy')):
            print 'NO DATA AVAILABLE FOR', obs_id
            continue
    
        reduced_metadata_fp = open(reduced_mosaic_metadata_filename, 'rb')
        mosaic_data = pickle.load(reduced_metadata_fp)
        obsid_list = pickle.load(reduced_metadata_fp)
        image_name_list = pickle.load(reduced_metadata_fp)
        full_filename_list = pickle.load(reduced_metadata_fp)
        reduced_metadata_fp.close()
    
        (mosaic_longitudes, mosaic_resolutions, mosaic_image_numbers,
         mosaic_ETs, mosaic_emission_angles, mosaic_incidence_angles,
         mosaic_phase_angles) = mosaic_data
        print i

    #    cmd_line = ['--write-pdf']
        ew_data = np.load(ew_data_filename+'.npy')
        ew_mask = np.load(ew_mask_filename+'.npy')
        
        mean_emission = np.mean(mosaic_emission_angles[~ew_mask])
        mean_phase = np.mean(mosaic_phase_angles[~ew_mask])
    
#        ew_data *= ringutil.normalized_ew_factor(mean_phase, mean_emission, np.mean(mosaic_incidence_angles[~ew_mask]))
        ew_data *= ringutil.normalized_ew_factor(np.array(mosaic_phase_angles), np.array(mosaic_emission_angles), np.array(mosaic_incidence_angles))
        ew_data = ew_data.view(ma.MaskedArray)
        ew_data.mask = ew_mask
#        smoothed_ew = smooth_ew(ew_data, 3.0)
        mean_ETs_list.append(np.mean(mosaic_ETs[~ew_mask]))
        mean_emissions.append(mean_emission)
        mean_incidences.append(np.mean(mosaic_incidence_angles[~ew_mask]))
        mean_phases.append(mean_phase)
        mean_EWs_list.append(ma.mean(ew_data))
        print 'OBSID %20s Mean EW %4.2f Mean Phase %6.2f Mean Inc. %6.2f Mean Emission %6.2f'%(obs_id, ma.mean(ew_data), mean_phase,np.mean(mosaic_incidence_angles[~ew_mask]), mean_emission)
                                                                                         
        i +=1
                                    
        
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(mean_ETs_list, mean_EWs_list, marker = '.', ls = ' ')
plt.ylabel('Mean EW')
plt.xlabel('ET (s)')
#plt.show()
        
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(mean_emissions, mean_EWs_list, marker = '.', ls = ' ')
plt.ylabel('Mean EW')
plt.xlabel('Emission Angle (deg)')
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(mean_incidences, mean_EWs_list, marker = '.', ls = ' ')
plt.ylabel('Mean EW')
plt.xlabel('Incidence Angle')
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(mean_phases, mean_EWs_list, marker = '.', ls = ' ')
plt.ylabel('Mean EW')
plt.xlabel('Phase Angle')
plt.show()

        