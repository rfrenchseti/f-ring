'''
Created on Nov 28, 2011

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

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['-a']
    
parser = OptionParser()

ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

phase_angle_list = []
mean_ew_list = []
pre_mean_ew_list = []
pre_std_ew_list = []
post_mean_ew_list = []
post_std_ew_list = []

for pre_degree in [45., 90., 135., 180., 225., 270.]:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for obs_id, image_name, full_path in ringutil.enumerate_files(options, args, obsid_only=True):
        (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
        bkgnd_mask_filename, bkgnd_model_filename,
        bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obs_id)
    
        (ew_data_filename, ew_mask_filename) = ringutil.ew_paths(options, obs_id)
    
        if (not os.path.exists(ew_data_filename+'.npy') or not os.path.exists(reduced_mosaic_metadata_filename) or
            not os.path.exists(bkgnd_model_filename+'.npy')):
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
    
        for idx in range(len(obsid_list)):
            if obsid_list[idx] != None:
                break
        model_obsid = obsid_list[idx]
    
        min_et = 1e38
        for idx in range(len(longitudes)):
            if longitudes[idx] >= 0:
                if ETs[idx] < min_et:
                    min_et = ETs[idx]
                    min_et_long = longitudes[idx]
    
        if (model_obsid == 'ISS_007RI_AZSCNLOPH001_PRIME'):
            continue
    
        ew_data = np.load(ew_data_filename+'.npy')
        ew_data = ew_data.view(ma.MaskedArray)
        ew_data.mask = np.load(ew_mask_filename+'.npy')
        phase_angles = phase_angles.view(ma.MaskedArray)
        phase_angles.mask = ew_data.mask
        emission_angles = emission_angles.view(ma.MaskedArray)
        emission_angles.mask = ew_data.mask
        incidence_angles = incidence_angles.view(ma.MaskedArray)
        incidence_angles.mask = ew_data.mask
        ETs = ETs.view(ma.MaskedArray)
        ETs.mask = ew_data.mask
        
        circ_ew_data = np.append(np.append(ew_data.data, ew_data.data), ew_data.data)
        circ_ew_data = circ_ew_data.view(ma.MaskedArray)
        circ_ew_data.mask = np.append(np.append(ma.getmaskarray(ew_data), ma.getmaskarray(ew_data)), ma.getmaskarray(ew_data))
        long_scale = options.longitude_resolution * options.mosaic_reduction_factor
        full360_idx = int(360 / long_scale)
        pre_idx = int(pre_degree / long_scale)
        post_idx = int((360.-pre_degree) / long_scale)
        max_dist, max_dist_long = ringutil.prometheus_close_approach(min_et, min_et_long)    
        prom_idx = full360_idx + int(max_dist_long/long_scale)
        
    #    print len(ew_data), len(circ_ew_data), long_scale, full360_idx, max_dist, max_dist_long, prom_idx
        x = np.append(circ_ew_data[prom_idx-full360_idx/2:prom_idx], circ_ew_data[prom_idx:prom_idx+full360_idx/2])
        x = x.view(ma.MaskedArray)
        x.mask = np.append(circ_ew_data.mask[prom_idx-full360_idx/2:prom_idx], circ_ew_data.mask[prom_idx:prom_idx+full360_idx/2])
        print 'Len EW', len(ew_data), 'Len X', len(x), 'Mask EW', np.sum(ew_data.mask), 'Mask X',
        print np.sum(circ_ew_data.mask[prom_idx-full360_idx/2:prom_idx])+np.sum(circ_ew_data.mask[prom_idx:prom_idx+full360_idx/2])
        print 'Mean EW', ma.mean(ew_data), 
        print 'Mean Pre', ma.mean(circ_ew_data[prom_idx-full360_idx/2:prom_idx]),
        print 'Mean Post',  ma.mean(circ_ew_data[prom_idx:prom_idx+full360_idx/2])
    
    #    print np.sum(ew_data.data), np.sum(circ_ew_data.data)/3
            
        phase_angle_list.append(np.mean(phase_angles))
        pre_mean_ew_list.append(ma.mean(circ_ew_data[prom_idx-pre_idx:prom_idx]))
        pre_std_ew_list.append(ma.std(circ_ew_data[prom_idx-pre_idx:prom_idx]))
        post_mean_ew_list.append(ma.mean(circ_ew_data[prom_idx:prom_idx+post_idx]))
        post_std_ew_list.append(ma.std(circ_ew_data[prom_idx:prom_idx+post_idx]))
        mean_ew_list.append(ma.mean(ew_data))
        
    plt.plot(phase_angle_list, pre_mean_ew_list, 'o', mfc='red', mec='red', ms=8, alpha=0.3, label='Pre')
    plt.plot(phase_angle_list, post_mean_ew_list, 'o', mfc='green', mec='green', ms=8, alpha=0.3, label='Post')
    #plt.plot(phase_angle_list, mean_ew_list, 'o', mfc='blue', mec='blue', ms=8, alpha=0.8)
    
    ax.set_yscale('log')
    plt.xlabel(r'Phase angle ($^\circ$)')
    plt.ylabel(r'Equivalent width (km) $\times$ $\mu$')
    plt.title('Equivalent width pre- (%d deg)/post- (%d deg) Prometheus' % (pre_degree, 360-pre_degree))
    ax.set_xlim(0,180)
    ax.set_ylim(.2, 20)
    plt.legend(numpoints=1, loc='upper left')

plt.show()
