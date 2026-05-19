'''
Create and save the master shape of a clump.
This will define the width and height of a clump for the purposes of the upcoming paper. 

'''
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

def split_chains(master_list):
    
    clump_list = []
    for chain in master_list:
        for clump in chain.clump_list:
            clump_list.append(clump)
    
    return clump_list

def convert_angle(b_angle, s_angle, h_angle, co_angle, clump_width):
    
    base = 20*(np.sin(b_angle))
    scale = clump_width*(np.sin(s_angle)+1)
    height = 10*(np.sin(h_angle)+1)
    center_offset = 0.5*clump_width*np.sin(co_angle)
   
    return (base, scale, height, center_offset)

def interpolate(arr):
    
    x = np.arange(0, len(arr))
    step = (len(arr)-1)/1000.
    x_int = np.arange(0, len(arr)-1., step)
    if len(x_int) > 1000.:
        x_int = x_int[0:1000]
    f_int = interp.interp1d(x, arr)
    
    return f_int(x_int)         #ew_range with 1000 points
    

def fit_seed_clump(clump_list):
    
    def fitting_func(params, seed, ew_data, center_long_idx, clump_half_width_idx):
        co_angle, h_angle, s_angle, b_angle = params
        
        base, scale, height, center_offset_idx = convert_angle(b_angle, s_angle, h_angle, co_angle, clump_half_width_idx)
        center_idx = center_long_idx + center_offset_idx
        #extract data from the center to a scale's width to either side
        ew_range = ew_data[center_idx-scale:center_idx+scale+1]

#        print center-(scale//2)/long_res, center+((scale//2)/long_res)
#        print len(ew_range)
        if len(ew_range) <= 1:
            return 2*1e20
#        plt.plot(ew_data)
#        plt.show()
        #interpolate the ew_range to 1000 points
        ew_range = interpolate(ew_range)
        
        residual = np.sum((ew_range*height+base-seed)**2)
#        print base, scale, height, center_offset, residual
#        print residual, scale, center
#        print 'OFFSETIDX %12.8f HALFWIDTHIDX %12.8f BASE %12.8f HEIGHT %12.8f RESID %15.8f' % (center_offset_idx, scale, base, height, residual)
        return residual
    
    seed_idx = 0
    seed_clump = clump_list[seed_idx]
    clump_list = list(clump_list)
#    del clump_list[seed_idx]
                                                                #just a test for now, pick the first clump in the list
    full_ew_data = seed_clump.clump_db_entry.ew_data             #full set of normalized data
    long_res = 360./len(full_ew_data)
    
    seed_left_idx = np.round(seed_clump.fit_left_deg/long_res)
    seed_right_idx = np.round(seed_clump.fit_right_deg/long_res)
    seed_center_idx = np.round((seed_right_idx+seed_left_idx)/2)
    seed_half_width_idx = seed_center_idx-seed_left_idx
    print 'SEEDLEFT', seed_left_idx, 'SEEDRIGHT', seed_right_idx,
    print 'SEEDCTR', seed_center_idx, 'SEEDHALFWIDTH', seed_half_width_idx
    
    seed = seed_clump.clump_db_entry.ew_data[seed_left_idx:seed_right_idx+1]
    seed = seed - np.min(seed)
    seed = seed/np.max(seed)
    
#        plt.plot(seed)
#        plt.show()
    
    seed_int = interpolate(seed)
    plt.plot(seed_int, lw = 3.0, color='yellow')
    
    i = 1
    clump_avg_arr = np.zeros(1000)
    for clump in clump_list:
        print 'Clump number', i
        #fit clump to the seed profile
        ew_data = clump.clump_db_entry.ew_data
        clump_left_idx = np.round(clump.fit_left_deg/long_res)
        clump_right_idx = np.round(clump.fit_right_deg/long_res)
        clump_center_idx = np.round((clump_right_idx+clump_left_idx)/2)
        clump_half_width_idx = clump_center_idx-clump_left_idx
        print 'CLUMPLEFT', clump_left_idx, 'CLUMPRIGHT', clump_right_idx,
        print 'CLUMPCTR', clump_center_idx, 'CLUMPHALFWIDTH', clump_half_width_idx
        #fit the seed to the clump to determine the width and center of the clump
        clump_data, residual, array, trash, trash, trash  = sciopt.fmin_powell(fitting_func, (0.,0.,0.,0.),
#                                                                               (np.pi/64., np.pi/64., np.pi/64., np.pi/64.),
                                                                               args=(seed_int, ew_data, clump_center_idx, clump_half_width_idx),
                                                                               ftol = 1e-8, xtol = 1e-8,disp=False, full_output=True)
        
        co_angle, h_angle, s_angle, b_angle = clump_data
#        clump_data2, residual, array, trash, trash, trash  = sciopt.fmin_powell(fitting_func,(co_angle, h_angle, s_angle, b_angle),
#                                                                               args=(seed_int, ew_data, clump.longitude_idx, clump.scale),
#                                                                               ftol = 1e-8, xtol = 1e-8,disp=False, full_output=True)
##        print stuff
#        co_angle, h_angle, s_angle, b_angle = clump_data2
        
        base, scale, height, center_offset_idx = convert_angle(b_angle, s_angle, h_angle, co_angle, clump_half_width_idx)
        
        print '********************************************************'
        print base, scale, height, center_offset_idx, residual
        center = np.round(center_offset_idx + clump_center_idx) 
        new_clump_ews = ew_data[center-scale:center+scale+1]
#        new_clump_ews = new_clump_ews/np.max(new_clump_ews)
        i +=1
#        old_clump_ews = ew_data[clump.fit_left_deg/long_res:clump.fit_right_deg/long_res]
#        plt.plot(old_clump_ews, 'red')
#        plt.plot(new_clump_ews, 'blue')
##        plt.plot(seed_int, 'red')
        
        new_clump_ews = interpolate(new_clump_ews)
        
        new_clump_ews = new_clump_ews* height
        new_clump_ews = new_clump_ews + base 
#        new_clump_ews -= np.min(new_clump_ews)
        
        plt.plot(new_clump_ews)
#        plt.show()
        
        clump_avg_arr = clump_avg_arr + new_clump_ews
    
#    clump_avg_arr = clump_avg_arr + seed_int
#    plt.show()
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Using Seed '+str(seed_idx))

    clump_avg_arr = clump_avg_arr/(len(clump_list)+1)
    clump_avg_arr = clump_avg_arr/np.max(clump_avg_arr)
    clump_avg_arr -= np.min(clump_avg_arr)
    
    plt.plot(clump_avg_arr)
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(1000)
    plt.plot(x, clump_avg_arr, label='Mean Clump', color='green', lw=5, alpha=0.3)

    for func_name in ['Gaussian', 'SDG', 'FDG', 'Triangle']:
        if func_name == 'Gaussian':
            sdg_scale = .4
            xi = np.arange(1000.) / 500. - 1.
            xsd2 = -xi * xi / (sdg_scale**2)
            mw = np.exp(xsd2/2.)   # Gaussian
        elif func_name == 'SDG':
            sdg_scale = .6
            xi = np.arange(1000.) / 500. - 1.
            xsd2 = -xi * xi / (sdg_scale**2)
            mw = (1. + xsd2) * np.exp(xsd2 / 2.)       #Second Derivative Gaussian
        elif func_name == 'FDG':
            sdg_scale = .75
            xi = np.arange(1000.) / 500. - 1.
            xsd2 = -xi * xi / (sdg_scale**2)
            mw = (xsd2*(xsd2 + 6) + 3.)*np.exp(xsd2/2.)   #Fourth Derivative Gaussian
        elif func_name == 'Triangle':
            mw = np.arange(1000.) / 500.
            mw[500:] = mw[-1:499:-1]-1
        else: assert False
    
        clump_data, residual, array, trash, trash, trash  = sciopt.fmin_powell(fitting_func, (0.,0.,0.,0.),
    #                                                                               (np.pi/64., np.pi/64., np.pi/64., np.pi/64.),
                                                                               args=(clump_avg_arr, mw, len(mw)/2, len(mw)/4),
                                                                               ftol = 1e-8, xtol = 1e-8,disp=False, full_output=True)
        
        print func_name, residual
        
        co_angle, h_angle, s_angle, b_angle = clump_data
        base, scale, height, center_offset_idx = convert_angle(b_angle, s_angle, h_angle, co_angle, clump_half_width_idx)
        
        mw = mw*height + base
        
        plt.plot(mw, label=func_name)
    
    plt.legend()     
    plt.show()

master_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'master_clumps_list.pickle')
master_list_fp = open(master_list_fp, 'rb')
clump_db, master_list = pickle.load(master_list_fp)
master_list_fp.close()

#master_clump_fp = os.path.join(ringutil.ROOT, 'clump-data', 'master_clump_data.pickle')
#master_clump_fp = open(master_clump_fp, 'wb')


#areas under the clumps should all be the same - scale the profiles accordingly

clump_list = split_chains(master_list)
clump_list = np.array(clump_list)
#delete bad clumps
keep_list = [1,3,5,9,11]
#keep_list = [1,2]
#keep_list = [0,1,2,3,8,9,10,11,15,16,19, 20, 21]

clump_list = clump_list[keep_list]

#test_list = clump_list[0,2]
#del clump_list[0]
#del clump_list[3:7]
#del clump_list[8]
#del clump_list[13:15]
fit_seed_clump(clump_list)

#plt.show()
