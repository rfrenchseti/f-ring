import pickle
import numpy as np
import matplotlib.pyplot as plt
import ringutil
import cspice
from optparse import OptionParser
import sys
import os
import numpy.ma as ma
import clumputil


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

#angle_table = '/home/shannon/Paper/paper_id_params_table.txt'
#angle_file = open(angle_table, 'w')
#
#id_data_file = open('/home/shannon/Paper/id_data_table.txt', 'w')
#
#
#angle_file.write('ID #,Date,Cassini Observation ID,Radial Resolution (km/pixel),Phase Angle ( $\mathbf{^o}$ ),Incidence Angle ( $\mathbf{^o}$ ),Emission Angle ( $\mathbf{^o}$ ) \n')
#id_data_file.write('ID #,Date,Cassini Observation ID,First Image,Last Image,Number of Images,Coverage Percentage,Number of Clumps \n')

c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_clumps_list.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
clump_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()


sorted_id_list = clumputil.get_sorted_obsid_list(clump_db)

c_db = {}
for chain in c_approved_list:
    for clump in chain.clump_list:
#            if clump.fit_width_deg < 35.:
        c_clump_db_entry = clumputil.ClumpDBEntry()
        obsid = clump.clump_db_entry.obsid
        if obsid not in c_db.keys():
            
            c_clump_db_entry.clump_list = []
            c_clump_db_entry.clump_list.append(clump)
            c_db[obsid] = c_clump_db_entry
            
        elif obsid in c_db.keys():
            c_db[obsid].clump_list.append(clump)

print c_db
#for obs_id, image_name, full_path in ringutil.enumerate_files(options, args, obsid_only=True):
for a, obs_id in enumerate(sorted_id_list):
        
        (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
         bkgnd_mask_filename, bkgnd_model_filename,
         bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obs_id)

        (ew_data_filename, ew_mask_filename) = ringutil.ew_paths(options, obs_id)
     
        reduced_metadata_fp = open(reduced_mosaic_metadata_filename, 'rb')
        mosaic_data = pickle.load(reduced_metadata_fp)
        obsid_list = pickle.load(reduced_metadata_fp)
        image_name_list = pickle.load(reduced_metadata_fp)
        full_filename_list = pickle.load(reduced_metadata_fp)
        reduced_metadata_fp.close()
    
#        print image_name_list
        (mosaic_longitudes, mosaic_resolutions, mosaic_image_numbers,
         mosaic_ETs, mosaic_emission_angles, mosaic_incidence_angles,
         mosaic_phase_angles) = mosaic_data
    
    #    cmd_line = ['--write-pdf']
        ew_data = np.load(ew_data_filename+'.npy')
        ew_mask = np.load(ew_mask_filename+'.npy')
        
        mean_emission = np.mean(mosaic_emission_angles[~ew_mask])
        mean_phase = clump_db[obs_id].phase_angle
        min_et = clump_db[obs_id].et_min
        mean_incidence = clump_db[obs_id].incidence_angle
        min_resolution = clump_db[obs_id].resolution_min
        max_resolution = clump_db[obs_id].resolution_max
        mean_EW = ma.mean(ew_data)
        
        ew_data = ew_data.view(ma.MaskedArray)
        ew_data.mask = ew_mask
        num_not_masked = ma.count(ew_data)
        percent_not_masked = float(((num_not_masked))/float(len(ew_data)))*100.
#        print percent_not_masked
        
        #find total number of images
        name_list = []
        for image_name in image_name_list:
            if image_name not in name_list:
                name_list.append(image_name)
        num_images = len(name_list)

        if obs_id in c_db.keys():
            num_clumps = len(c_db[obs_id].clump_list)
        elif obs_id not in c_db.keys():
            num_clumps = 0        
            
        if obs_id in c_db.keys():
            print obs_id, ' | ', cspice.et2utc(min_et, 'D', 5), ' | ', cspice.et2utc(clump_db[obs_id].et_max, 'D', 5), ' | ', num_clumps
        
        start_date = cspice.et2utc(min_et, 'C', 0)
        start_date = start_date[0:11]
#        angle_file.write('%i,%s,%20s,%i - %i,%.2f,%.2f,%.2f \n'%(a+1, start_date, obs_id, min_resolution, max_resolution, mean_phase,
#                                                                             mean_incidence, mean_emission))
#        id_data_file.write('%i,%s,%20s,%s,%s,%i,%.2f,%i \n'%(a+1, start_date, obs_id, image_name_list[0], image_name_list[-1],
#                                                               num_images, percent_not_masked, num_clumps))
#
#id_data_file.close()
#angle_file.close()

        
        
        