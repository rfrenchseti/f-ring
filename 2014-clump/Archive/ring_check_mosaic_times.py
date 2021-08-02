import ringutil
import sys
from optparse import OptionParser
import numpy as np
import numpy.ma as ma
import pickle
import sys
import os.path
import ringutil
import cspice
import matplotlib.pyplot as plt



cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = [ '-a',
                '--ignore-voyager'
                ]


parser = OptionParser()

ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)


for obs_id, image_name, full_path in ringutil.enumerate_files(options, args, obsid_only=True):
  
    (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
    bkgnd_mask_filename, bkgnd_model_filename,
    bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obs_id)
    
    #      print reduced_mosaic_data_filename
    (ew_data_filename, ew_mask_filename) = ringutil.ew_paths(options, obs_id)
    
    if (not os.path.exists(reduced_mosaic_metadata_filename)) or (not os.path.exists(ew_data_filename+'.npy')):
#        print 'NO DATA AVAILABLE FOR', obs_id
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
    
    ew_mask = np.load(ew_mask_filename+'.npy')
    print '  %30s  |  %20s  |  %20s  |  %20s  |'%( obs_id, cspice.et2utc(ma.min(mosaic_ETs[~ew_mask]), 'C', 0) , cspice.et2utc(ma.mean(mosaic_ETs[~ew_mask]), 'C', 0), cspice.et2utc(ma.max(mosaic_ETs[~ew_mask]), 'C', 0))
    
    
    if obs_id == 'ISS_084RI_FMONITOR002_PRIME':
        for i, et in enumerate(mosaic_ETs):
            if ew_mask[i] == False:
                print i* 0.04, cspice.et2utc(mosaic_ETs[i], 'C', 0), mosaic_longitudes[i]
#        print np.where( mosaic_ETs == ma.min(mosaic_ETs[~ew_mask]))[0]*0.04
    