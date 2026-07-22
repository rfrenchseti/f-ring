'''
A program to compare the differences in inertial longitude to the longitude of pericentre.
'''

from optparse import OptionParser
import numpy as np
import numpy.ma as ma
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import scipy.optimize as sciopt
import scipy.stats as st
import pickle
import sys
import os.path
import ringutil
import cspice
import matplotlib.pyplot as plt
import cwt
import bandpass_filter
import clumputil
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
import matplotlib.transforms as mtransforms
from scipy.stats import norm
import matplotlib.mlab as mlab


cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = []
#    cmd_line = ['--side-analysis', '--plot-fit']
#    cmd_line = ['--write-pdf']

parser = OptionParser()
ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

w0 = 24.2                        #deg
dw = 2.70025       

def list_to_db(c_clump_db,c_approved_list):
    
    c_db = {}
    for chain in c_approved_list:
        for clump in chain.clump_list:
            c_clump_db_entry = clumputil.ClumpDBEntry()
            obsid = clump.clump_db_entry.obsid
            if obsid not in c_db.keys():
                c_clump_db_entry.et_min = c_clump_db[obsid].et_min
                c_clump_db_entry.et_max = c_clump_db[obsid].et_max
                c_clump_db_entry.et_min_longitude = c_clump_db[obsid].et_min_longitude
                c_clump_db_entry.et_max_longitude = c_clump_db[obsid].et_max_longitude
                c_clump_db_entry.smoothed_ew = c_clump_db[obsid].smoothed_ew
                c_clump_db_entry.et = c_clump_db[obsid].et
                c_clump_db_entry.resolution_min = c_clump_db[obsid].resolution_min
                c_clump_db_entry.resolution_max = c_clump_db[obsid].resolution_max
                c_clump_db_entry.emission_angle = c_clump_db[obsid].emission_angle
                c_clump_db_entry.incidence_angle = c_clump_db[obsid].incidence_angle
                c_clump_db_entry.phase_angle = c_clump_db[obsid].phase_angle
                c_clump_db_entry.obsid = obsid
                c_clump_db_entry.clump_list = []
                c_clump_db_entry.ew_data = c_clump_db[obsid].ew_data # Filtered and normalized
                
                c_clump_db_entry.clump_list.append(clump)
                c_db[obsid] = c_clump_db_entry
                
            elif obsid in c_db.keys():
                c_db[obsid].clump_list.append(clump)
                
    return (c_db)

def compare_inert_to_peri(options, obs_id, clump_idx, clump_longitude):
    
    #load metadata - need the ET corresponding to the clump's longitude.
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

    clump_et = mosaic_ETs[clump_idx]
    
    clump_inertial_long = ringutil.CorotatingToInertial(clump_longitude, clump_et)
    dt = clump_et/86400.
    pericentre_long = w0 + dw*dt
    
    return (clump_inertial_long, pericentre_long)

def command_compare_clump_inert_peri(clump_disp_data):


    for i, clump in enumerate(chain.clump_list):
        clump1 = clump
        clump2 = chain.clump_list[i+1]
        
        (clump1_inert, clump1_per) = compare_inert_to_peri(options, clump1.clump_db_entry.obs_id, clump1.longitude_idx, clump1.longitude)
        (clump2_inert, clump2_per) = compare_inert_to_peri(options, clump2.clump_db_entry.obs_id, clump2.longitude_idx, clump2.longitude)
        
        clump1_diff = clump1_per - clump1_inert
        clump2_diff = clump2_per - clump2_inert
        
        print 'CLUMP 1 PERICENTRE:', clump1_per, 'ClUMP 1 INERTIAL LONG:', clump1_inert, 'CLUMP 1 DIFF:', np.abs(clump1_diff)
        print 'CLUMP 2 PERICENTRE:', clump2_per, 'ClUMP 2 INERTIAL LONG:', clump2_inert, 'CLUMP 2 DIFF:', np.abs(clump2_diff)
        
        if i == len(chain.clump_list) -1: 
            break
        
        
c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_clumps_list.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
clump_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()