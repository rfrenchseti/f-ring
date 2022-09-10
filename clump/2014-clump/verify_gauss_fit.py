from optparse import OptionParser
import numpy as np
import numpy.ma as ma
import pickle
import sys
import os.path
import ringutil
import cspice
import matplotlib.pyplot as plt
import clumputil
from scipy.stats import norm
import matplotlib.mlab as mlab
import scipy.interpolate as interp
import scipy.optimize as sciopt


cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = []

parser = OptionParser()
ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)
version = '3'
title = 'Fitted Data, Base = 1*min_ew'

def split_chains(master_list):
    
    clump_list = []
    for chain in master_list:
        for clump in chain.clump_list:
            clump_list.append(clump)
    
    return clump_list

def interpolate(arr):
    
    x = np.arange(0, len(arr))
    step = (len(arr)-1)/1000.
    x_int = np.arange(0, len(arr)-1., step)
    if len(x_int) > 1000.:
        x_int = x_int[0:1000]
    f_int = interp.interp1d(x, arr)
    
    return f_int(x_int)         #ew_range with 1000 points

def plot_fitted_clump_on_ew(ax, ew_data, clump):
    long_res = 360./len(ew_data)
    longitudes =np.tile(np.arange(0,360., long_res),3)
    tri_ew = np.tile(ew_data, 3)
    left_idx = clump.fit_left_deg/long_res + len(ew_data)
    right_idx = clump.fit_right_deg/long_res + len(ew_data)
    print longitudes[left_idx], longitudes[right_idx]
    if left_idx > right_idx:
        left_idx -= len(ew_data)

    idx_range = longitudes[left_idx:right_idx]
    print clump.fit_left_deg, clump.fit_right_deg
    print idx_range
    if left_idx < len(ew_data):
        ax.plot(longitudes[left_idx:len(ew_data)-1] - 360., tri_ew[left_idx:len(ew_data)-1], label = 'Fitted Clump',color = 'red', alpha = 0.5, lw = 2)
        ax.plot(longitudes[len(ew_data):right_idx], tri_ew[len(ew_data):right_idx], color = 'red', alpha = 0.5, lw = 2)
    else:
        ax.plot(idx_range, tri_ew[left_idx:right_idx], color = 'red',label = 'Fitted Clump', alpha = 0.5, lw = 2)
   #plotting data-----------------------------------------------------------------------------------------------------------------------------------
def plot_gaussian(clump):
    
#    plt.plot(np.arange(0,360.*3, 0.5), np.tile(clump.clump_db_entry.ew_data,3))
#    plt.show()
    
    tri_ews = np.tile(clump.clump_db_entry.ew_data, 3)
    len_ew_data = len(clump.clump_db_entry.ew_data)
    long_res = 360./(len_ew_data)
    longitudes = np.tile(np.arange(0,360., long_res),3)
    
#    print clump.fit_left_deg, clump.fit_right_deg, clump.g_center
    left_long = clump.fit_left_deg
    right_long = clump.fit_right_deg
    
#    print left_long, right_long
#    if right_long < left_long:
#        left_long -= 360.
# 
    fig = plt.figure()
    ax = fig.add_subplot(111)
# 

    
#    left_clump_idx = left_long/long_res + len_ew_data
#    right_clump_idx = right_long/long_res + len_ew_data
#    new_range = tri_ews[left_clump_idx:right_clump_idx]
#    long_fit_range = longitudes[left_clump_idx:right_clump_idx]
#    print new_range
#    print new_range.mask
#    if right_long > 360.:
#        right_long -= 360.
#    x3 = np.arange(left_long, right_long, long_res) 

#    new_range = interpolate(new_range)
#    x3 = interpolate(x3)

    #for the sake of plotting - interpolate x2 and ew_range
    wav_center_idx = clump.longitude_idx + len_ew_data
    print clump.longitude, clump.scale
    print clump.scale_idx
#    print wav_center_idx
    ew_range = tri_ews[wav_center_idx - clump.scale_idx:wav_center_idx + clump.scale_idx]
    
#    clumputil.plot_fitted_clump_on_ew(ax, clump.clump_db_entry.ew_data, clump)
#    plt.show()
    left_wav_idx = wav_center_idx - clump.scale_idx
    right_wav_idx = wav_center_idx + clump.scale_idx
    
#    print (left_wav_idx - len_ew_data)*long_res, (right_wav_idx - len_ew_data)*long_res
#    old_longs = longitudes[wav_center_idx - clump.scale_idx:wav_center_idx + clump.scale_idx]
#    old_longs = interpolate(old_longs)
#    long_fit_range = interpolate(long_fit_range)
    
    print longitudes[left_wav_idx:len_ew_data -1]
    print longitudes[len_ew_data:right_wav_idx]
#    print clump.g_center, clump.longitude
    
   
    clump_center = np.round(clump.g_center/long_res)*long_res
    clump_sigma = np.round(clump.g_sigma/long_res)*long_res
    
    left_sigma_idx = np.round((clump_center - 2*clump_sigma)/long_res) + len_ew_data
    right_sigma_idx = np.round((clump_center + 2*clump_sigma)/long_res) + len_ew_data
    print left_sigma_idx, right_sigma_idx
    x2 = longitudes[left_sigma_idx:right_sigma_idx]
    print x2
    print clump_center, clump_sigma
#     x2 = interpolate(x2)
#    ew_range = interpolate(ew_range)
    xsd2 = -((x2-clump_center)*(x2-clump_center))/ (clump_sigma**2)
    gauss = np.exp(xsd2/2.)   # Gaussian
    gauss = gauss*clump.g_height + clump.g_base
    
#    gauss = interpolate(gauss)
    if left_wav_idx < len_ew_data:
        plt.plot(longitudes[left_wav_idx:len_ew_data-1]-360., tri_ews[left_wav_idx:len_ew_data-1], label='Data Range', color='Blue', lw=5, alpha=0.5)
        plt.plot(longitudes[len_ew_data:right_wav_idx],tri_ews[len_ew_data: right_wav_idx], color = 'orange', lw = 5, alpha = 0.5)
    else:
        plt.plot(longitudes[left_wav_idx:right_wav_idx], tri_ews[left_wav_idx:right_wav_idx], label = 'Data Range', color = 'Blue', lw = 5, alpha = 0.5)
#    plt.plot(long_fit_range, new_range, label = 'New Clump', color = 'red', lw = 3, alpha = 0.5)
#    plt.plot(x, new_clump_ews, label = 'Old Clump', color = 'pink', alpha = 1, lw = 2)
    plt.plot(x2, gauss)
    plot_fitted_clump_on_ew(ax, clump.clump_db_entry.ew_data, clump)
#    plt.title(title)
    
    y = np.arange(np.min(ew_range), np.max(ew_range), .01)
    x_sig = np.zeros(len(y)) + clump_sigma
    x_2sig = np.zeros(len(y)) + 2*clump_sigma
    x_3sig = np.zeros(len(y)) + 3*clump_sigma
    
    plt.plot(clump_center + x_sig, y, color = 'blue')
    plt.plot(clump_center + x_2sig, y, color = 'blue')
    plt.plot(clump_center + x_3sig, y, color = 'blue')
    
    plt.plot(clump_center - x_sig, y, color = 'blue')
    plt.plot(clump_center - x_2sig, y, color = 'blue')
    plt.plot(clump_center - x_3sig, y, color = 'blue')
    
    plt.title(clump.clump_db_entry.obsid + 'Center: ' + str(clump.g_center) + ', ' + 'Width: ' +str(clump.fit_width_deg))
    plt.legend()     
    plt.show()
#    del x
#    del x2
#    del gauss
#    return (left_long, left_idx, right_long, right_idx, clump_int_height, center, base, height, sigma)
#    plt.savefig('/home/shannon/Documents/gaussian_fits2/test_fit_'+ str(i) + '_'+ version + '.png')


master_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_clumps_list.pickle')
master_list_fp = open(master_list_fp, 'rb')
clump_db, master_list = pickle.load(master_list_fp)
master_list_fp.close()

voyager_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'voyager_approved_clumps_list.pickle')
voyager_list_fp = open(voyager_list_fp, 'rb')
v_clump_db, v_list = pickle.load(voyager_list_fp)
voyager_list_fp.close()

#test only the master list
ds_list = split_chains(master_list)
v_list = split_chains(v_list)

#keep_list = [0, 4,5]
#clump_list = clump_list[keep_list]
for i, clump in enumerate(ds_list):
#    tri_ews = np.tile(clump.clump_db_entry.ew_data, 3)
#    long_res = 360./len(clump.clump_db_entry.ew_data)
#    right_idx = clump.fit_right_deg/long_res + len(clump.clump_db_entry.ew_data)
#    left_idx = clump.fit_left_deg/long_res + len(clump.clump_db_entry.ew_data)
#    if right_idx < left_idx:
#        right_idx += len(clump.clump_db_entry.ew_data)
    plot_gaussian(clump)
