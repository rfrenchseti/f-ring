'''
A program to add in fake wavelets for clumps that have bad fits (or no fit at all)

@author:Shannon Hicks

'''
import clump_gaussian_fit
import numpy as np
import ringutil
import clumputil
import numpy.ma as ma
import os
import sys
import pickle
from optparse import OptionParser

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['-a']

parser = OptionParser()
ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)


def gauss_fit(tri_ews, clump_left_idx, clump_right_idx, clump_longitude_idx):
    
    
    left_deg, left_idx, right_deg, right_idx, clump_int_height, gauss_center, gauss_base, gauss_height, gauss_sigma= clump_gaussian_fit.fit_gaussian(tri_ews, clump_left_idx, clump_right_idx, clump_longitude_idx)
    
    long_res = 360./(len(tri_ews)/3)
    fit_width_idx = abs((right_idx) - left_idx)
    fit_width_deg = fit_width_idx*long_res
    fit_base = (tri_ews[left_idx] + tri_ews[right_idx])/2.
        
    #        print fit_base, tri_ew[left_idx], tri_ew[right_idx]
    fit_height = ma.max(tri_ews[left_idx:right_idx+1]) - fit_base
    #    norm_fit_height = ma.max(norm_tri_ew[left_idx:right_idx + 1]) - norm_fit_base
    #        
    #    left_deg = (left_idx - len(ew_data))*long_res
    #    right_deg = (right_idx - len(ew_data))*long_res
    #    print fit_base, tri_ew[left_idx], tri_ew[right_idx]
    if right_deg > 360.:
        right_deg = right_deg - 360.
    if left_deg < 0.:
        left_deg += 360.
    
    
    return (left_deg, right_deg, fit_width_idx, fit_width_deg, fit_height, clump_int_height, gauss_center, gauss_sigma, gauss_base, gauss_height )


clump_db_path, clump_chains_path = ringutil.clumpdb_paths(options)
clump_db_fp = open(clump_db_path, 'rb')
clump_find_options = pickle.load(clump_db_fp)
clump_db = pickle.load(clump_db_fp)
clump_db_fp.close()

#obsid, longitude guess,left_guess, right_guess
#clump1 = ('ISS_032RF_FMOVIE001_VIMS', 116., 110.50, 129.50)
#clump2 = ('ISS_098RI_TMAPN30LP001_CIRS', 170., 158., 174.3)
#clump3 = ('ISS_100RF_FMOVIE003_PRIME', 169.8, 158., 177.3)

#set2
#clump1 = ('ISS_079RI_FMONITOR002_PRIME', 219.5, 214.28, 224.2)
#clump2 = ('ISS_082RI_FMONITOR003_PRIME', 212.0, 206.35,217.7)
#clump3 = ('ISS_085RF_FMOVIE003_PRIME_1', 205.35, 202., 216.5 )
#clump4 = ('ISS_085RF_FMOVIE003_PRIME_2', 205.35 , 199.9, 217.5)

#clump1 = ('ISS_032RF_FMOVIE001_VIMS', 137.7, 125.171,150.8)

#
#clump1 = ('ISS_041RF_FMOVIE001_VIMS', 110.0,100.0, 121.0)
#clump2 = ('ISS_081RI_FMOVIE106_VIMS', 225.0, 220., 235.0)
#clump3 = ('ISS_081RI_FMOVIE106_VIMS', 213.7, 210., 218.0)
#clump4 = ('ISS_083RI_FMOVIE109_VIMS', 212.0, 211., 216.0 )

#clump1 = ('ISS_082RI_FMONITOR003_PRIME', 208.0, 200.5, 210.8 )
#clump2 = ('ISS_083RI_FMOVIE109_VIMS', 208.0, 200.6, 210.8 )
#clump3 = ('ISS_085RF_FMOVIE003_PRIME_1', 204.2, 200.0, 207.3 )
#clump4 = ('ISS_085RF_FMOVIE003_PRIME_2', 204.2, 200.0, 207.3 )

#clump1 = ('ISS_006RI_LPHRLFMOV001_PRIME', 318.0, 315., 328. )
clump1 = ('ISS_039RF_FMOVIE001_VIMS', 103.5, 95.0, 110.)
clump_list = [clump1]

#create and add these clumps to the clump database
for new_clump in clump_list:
    obsid, center_guess, left_guess, right_guess = new_clump
    long_res = 360./len(clump_db[obsid].ew_data)
    tri_ews = np.tile(clump_db[obsid].ew_data, 3)
    
    clump = clumputil.ClumpData()
    #calculate gaussian fit
    left_guess_idx = np.floor(left_guess/long_res) + len(clump_db[obsid].ew_data)
    right_guess_idx = np.ceil(right_guess/long_res) + len(clump_db[obsid].ew_data)
    center_guess_idx = np.round(center_guess/long_res) + len(clump_db[obsid].ew_data)
   
    clump.fit_left_deg, clump.fit_right_deg, clump.fit_width_idx, clump.fit_width_deg, clump.fit_height, clump.int_fit_height, clump.g_center, clump.g_sigma, clump.g_base, clump.g_height = gauss_fit(tri_ews, left_guess_idx, right_guess_idx, center_guess_idx)
    
    clump.longitude = clump.g_center
    clump.scale = clump.fit_width_deg
    clump.longitude_idx = clump.g_center/long_res
    clump.scale_idx = clump.fit_width_idx
    clump.mexhat_base = clump.g_base
    clump.mexhat_height = clump.g_height
    clump.abs_height = clump.g_height
    clump.max_long = None
    clump.matched = False
    clump.residual = None
    clump.wave_type = None
    #calculate clump sigma height
    profile_sigma = ma.std(clump_db[obsid].ew_data)
#            height = clump.mexhat_height*mexhat[len(mexhat)//2]
    clump.clump_sigma = clump.abs_height/profile_sigma
    clump.fit_sigma = clump.fit_height/profile_sigma
    
#            print vars(clump)
    clump_db[obsid].clump_list.append(clump)
    print 'CLUMP LONG', clump.g_center, 'CLUMP WIDTH', clump.fit_width_deg
    
clump_database_path, clump_chains_path = ringutil.clumpdb_paths(options)
clump_database_fp = open(clump_database_path, 'wb')

pickle.dump(clump_find_options, clump_database_fp)
#        print vars(clump_database['ISS_041RF_FMOVIE001_VIMS'])
print clump_database_fp
pickle.dump(clump_db, clump_database_fp)
clump_database_fp.close()
