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


version = '3'
title = 'Fitted Data, Base = 1*min_ew'
def split_chains(master_list):
    
    clump_list = []
    for chain in master_list:
        for clump in chain.clump_list:
            clump_list.append(clump)
    
    return clump_list

def convert_angle(b_angle, s_angle, h_angle,m_angle,clump_width,min_ew, max_ew):
    
    min_max = max_ew - min_ew
    
    base = min_max/2.*(np.sin(b_angle)+.5) + min_ew
    scale = 0.5*clump_width*(np.sin(s_angle)+1) + 0.25*clump_width
    height = min_max/2.*(np.sin(h_angle)+1.5)
    center_offset = 0.5*clump_width*np.sin(m_angle)
   
    return (base, scale, height, center_offset)
#def convert_angle(b_angle, s_angle, h_angle,m_angle,clump_width,min_ew, max_ew):
#    
#    base = min_ew*(np.sin(b_angle)+1) + min_ew
#    scale = 0.5*clump_width*(np.sin(s_angle)+1) + 0.25*clump_width
#    height = max_ew*(np.sin(h_angle)+1)
#    center_offset = 0.5*clump_width*np.sin(m_angle)
#   
#    return (base, scale, height, center_offset)

def interpolate(arr):
    
    x = np.arange(0, len(arr))
    step = (len(arr)-1)/1000.
    x_int = np.arange(0, len(arr)-1., step)
    if len(x_int) > 1000.:
        x_int = x_int[0:1000]
    f_int = interp.interp1d(x, arr)
    
    return f_int(x_int)         #ew_range with 1000 points

def find_edges(clump_ews, gauss, longitudes, clump_center, sigma, long_res):
    center = len(clump_ews)/2
    right_idx= ma.argmin(clump_ews[center::]) + center 
    left_idx = ma.argmin(clump_ews[0:center])
    
    left_long = longitudes[left_idx]
        
    right_long = longitudes[right_idx]
    
    clump_data = clump_ews[left_idx:right_idx+1]
    #take the average for the base
    base = (clump_ews[left_idx] + clump_ews[right_idx])/2.
    clump_int_height = np.sum(clump_data - base)*long_res
    
    slope = (clump_ews[right_idx]-clump_ews[left_idx])/(right_idx-left_idx)
    intercept = clump_ews[left_idx]
    base_sub_clump = clump_data - (np.arange(0, len(clump_data))*slope+intercept)
    clump_fit_height = np.max(base_sub_clump)
    
    return (left_long, right_long, clump_int_height, clump_fit_height)
    
def fit_gaussian(tri_ews, clump_left_idx, clump_right_idx, clump_longitude_idx, plot = False):
    
    #All indexes are assumed to be in accordance to a tripled EW profile to compensate for overlapping
    def fitting_func(params, xi, ew_range,clump_half_width,clump_center, min_ew, max_ew):
        m_angle, s_angle, h_angle, b_angle = params

        base, sigma, height, center_offset = convert_angle(b_angle, s_angle, h_angle,m_angle,clump_half_width,min_ew, max_ew)
        center = clump_center + center_offset
        xsd2 = -((xi-center)*(xi-center))/ (sigma**2)
        gauss = np.exp(xsd2/2.)   # Gaussian

        if len(ew_range) <= 1:
            return 2*1e20
        
        residual = np.sum((gauss*height+base-ew_range)**2)

        return residual

    len_ew_data = len(tri_ews)//3
    long_res = 360./(len_ew_data)
    
    clump_longitude = (clump_longitude_idx - len_ew_data)*long_res 
    
    clump_center_idx = np.round((clump_right_idx+clump_left_idx)/2)

    clump_half_width_idx = clump_center_idx-clump_left_idx
#    print 'CLUMPLEFT', clump_left_idx, 'CLUMPRIGHT', clump_right_idx
#    print 'CLUMPCTR', clump_center_idx, 'CLUMPHALFWIDTH', clump_half_width_idx
    #fit the seed to the clump to determine the width and center of the clump
#    print clump_left_idx, clump_right_idx
    old_clump_ews = tri_ews[clump_left_idx:clump_right_idx+1]
    if len(old_clump_ews) < 3:                  #sometimes the downsampled versions don't end up with enough data to fit a gaussian to the arrays
        return(0)
#    print len(new_clump_ews)
#    new_clump_ews = interpolate(new_clump_ews)
#    x = np.arange(fit_left_deg, fit_right_deg-long_res, long_res)
    x = np.arange(clump_left_idx-len_ew_data, clump_right_idx- len_ew_data +1)*long_res
#    print 'GAUSS FIT LEFT IDX', clump_left_idx, 'RIGHT IDX', clump_right_idx
#    print "OLD CLUMP DATA"
#    print x
#    print old_clump_ews
#    x = interpolate(x)
#      
#    clump_mean =  ma.mean(old_clump_ews)              #sigma
#    clump_std = ma.std(old_clump_ews)
    min_ew = np.min(old_clump_ews)
    max_ew = np.max(old_clump_ews)
    
    
    clump_data, residual, array, trash, trash, trash  = sciopt.fmin_powell(fitting_func, (0.,0.,0., 0.),
#                                                                               (np.pi/64., np.pi/64., np.pi/64., np.pi/64.),
                                                                       args=(x, old_clump_ews, clump_half_width_idx*long_res, clump_longitude, min_ew, max_ew),
                                                                       ftol = 1e-8, xtol = 1e-8,disp=False, full_output=True)

    m_angle, s_angle, h_angle, b_angle = clump_data
    base, sigma, height, center_offset = convert_angle(b_angle, s_angle, h_angle, m_angle, clump_half_width_idx*long_res, min_ew, max_ew)
    
    center = clump_longitude + center_offset                    #make this a multiple of our longitude resolution
#    center = np.round(center/long_res)*long_res
#    sigma = np.round(sigma/long_res)*long_res
#    print base, sigma, height, center
    left_sigma = center - 2*sigma
    right_sigma = center + 2*sigma
#    x2 = np.arange(center - 2*sigma, center + 2*sigma, long_res)
    x2 = np.arange(left_sigma,right_sigma +0.01, .01)
#    print 'GAUSSIAN/BACKGROUND LONGS'
#    print x2
#    
#    print left_sigma, right_sigma
    left_sigma_idx = np.floor(left_sigma/long_res) + len_ew_data
    right_sigma_idx = np.ceil(right_sigma/long_res) + len_ew_data
#    print left_sigma_idx, right_sigma_idx 
#        x2 = interpolate(x2)
#        x2 = x2 + center_offset
    xsd2 = -((x2-center)*(x2-center))/ (sigma**2)
    gauss = np.exp(xsd2/2.)   # Gaussian
    gauss = gauss*height + base
    
    while True:
        ew_range = tri_ews[left_sigma_idx:right_sigma_idx+1]
        if ew_range[0] is ma.masked:
            left_sigma_idx += 1
            continue
        if ew_range[-1] is ma.masked:
            right_sigma_idx -= 1
            continue
        break
    
    assert left_sigma_idx < right_sigma_idx
    
    ew_longs = np.arange(left_sigma_idx-len_ew_data, right_sigma_idx-len_ew_data +1)*long_res
#    print len(ew_range), len(ew_longs)
    if len(ew_range) < 3:
        return (0)
    left_long, right_long, clump_int_height, clump_fit_height = find_edges(ew_range, gauss, ew_longs, center, sigma, long_res)
#    print left_long, right_long
    left_idx = left_long/long_res + len_ew_data
    right_idx = right_long/long_res + len_ew_data
#    print left_idx, right_idx
    
#    print 'CENTER', center
#    return (left_long, left_idx, right_ong, right_idx, clump_int_height)
    #plotting data-----------------------------------------------------------------------------------------------------------------------------------
    if plot == True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
     
        new_range = tri_ews[left_idx:right_idx+1]
        #    print new_range
    #    print new_range.mask
    #    x3 = np.arange(left_long, right_long +long_res, long_res)
        x3 = np.arange(left_idx-len_ew_data, right_idx-len_ew_data+1)*long_res
#        print 'NEW CLUMP LONGS'
#        print x3
#        print new_range
    #    new_range = interpolate(new_range)
    #    x3 = interpolate(x3)
    
    #    for the sake of plotting - interpolate x2 and ew_range
    #    x2 = interpolate(x2)
    #    ew_range = interpolate(ew_range)
    #    gauss = interpolate(gauss)
        plt.plot(ew_longs, ew_range, label='Extended Clump', color='Blue', lw=5, alpha=0.5)
        plt.plot(x3, new_range, label = 'New Clump', color = 'red', lw = 3, alpha = 0.5)
        plt.plot(x, old_clump_ews, label = 'Old Clump', color = 'pink', alpha = 1, lw = 2)
        plt.plot(x2, gauss, label = str(residual))
        plt.title(title)
        
        y = np.arange(np.min(old_clump_ews), np.max(old_clump_ews), .01)
        x_sig = np.zeros(len(y)) + sigma
        x_2sig = np.zeros(len(y)) + 2*sigma
        x_3sig = np.zeros(len(y)) + 3*sigma
        
        plt.plot(center + x_sig, y, color = 'blue')
        plt.plot(center + x_2sig, y, color = 'blue')
        plt.plot(center + x_3sig, y, color = 'blue')
        
        plt.plot(center - x_sig, y, color = 'blue')
        plt.plot(center - x_2sig, y, color = 'blue')
        plt.plot(center - x_3sig, y, color = 'blue')
            
        plt.legend()     
        plt.show()
#    del x
#    del x2
#    del gauss
    return (left_long, left_idx, right_long, right_idx, clump_int_height, clump_fit_height, center, base, height, sigma)
#    plt.savefig('/home/shannon/Documents/gaussian_fits2/test_fit_'+ str(i) + '_'+ version + '.png')
#    fig.clf()
#    plt.close()

def refine_fit(clump, ew_data, voyager, downsample):

    long_res = 360./len(ew_data)
    longitudes = np.arange(0, 360., long_res)
    tri_long = np.tile(longitudes, 3)
    #for every clump that passes through - refine the fit
    
#    tri_smooth = np.tile(ew_data, 3)
    tri_ew = np.tile(ew_data, 3)
#    norm_tri_ew = np.tile((ew_data/np.mean(ew_data)),3)
    wav_center_idx = clump.longitude_idx + len(ew_data)
    wav_scale = clump.scale_idx
    
    left_ew_range = tri_ew[wav_center_idx-wav_scale:wav_center_idx]
    right_ew_range = tri_ew[wav_center_idx:wav_center_idx + wav_scale]
    #the range should include the full clumps, and half of the scale size to either side of the clump
    
    left_idx = int(wav_center_idx - (wav_scale - np.argmin(left_ew_range)))
#    print left_idx, int(left_idx)
#    left_min = tri_ew[left_idx]
    
#    print 'LEFT MIN', left_min
#    print l_min_idx
    
    right_idx = wav_center_idx + np.argmin(right_ew_range)
#    right_min = tri_ew[right_idx]
    
#    print 'RIGHT MIN', right_min
    
#    print 'SCALE', clump.scale,'LONG', clump.longitude,'LEFT', left_idx*.04, 'RIGHT', right_idx*.04
#    print 'SCALE', clump.scale,'LONG', clump.longitude,'LEFT', left_idx, 'RIGHT', right_idx
    
    #use RIGHT + 1
     
     #fit a gaussian to the data range in between the left and right idx
    
    if downsample or voyager:
        clump_params = fit_gaussian(tri_ew, left_idx, right_idx, wav_center_idx)
#        print clump_params
        if clump_params != 0:
            left_deg, left_idx, right_deg, right_idx, clump_int_height, clump_fit_height, gauss_center, gauss_base, gauss_height, gauss_sigma = clump_params
        
        else:
            left_deg = (left_idx - len(ew_data))*long_res
            right_deg = (right_idx - len(ew_data))*long_res
            gauss_center = clump.longitude
            gauss_base = clump.mexhat_base
            gauss_height = clump.mexhat_height
            gauss_sigma = clump.scale/2.
            
            fit_base = (tri_ew[left_idx] + tri_ew[right_idx])/2.
            clump_int_height = np.sum(tri_ew[left_idx:right_idx+1] - fit_base)*long_res
            clump_data = tri_ew[left_idx:right_idx+1]
            slope = (tri_ew[right_idx]-tri_ew[left_idx])/(right_idx-left_idx)
            intercept = tri_ew[left_idx]
            base_sub_clump = clump_data - (np.arange(0, len(clump_data))*slope+intercept)
            clump_fit_height = np.max(base_sub_clump)
            
            print 'Gaussian fit failed'
            print clump.longitude
            print clump.mexhat_base
            print clump.mexhat_height
            print clump.scale
            print tri_ew[left_idx]
            print tri_ew[right_idx]
            print tri_ew[left_idx:right_idx+1]
            print '------------'
            
    else:
        left_deg, left_idx, right_deg, right_idx, clump_int_height, clump_fit_height, gauss_center, gauss_base, gauss_height, gauss_sigma = fit_gaussian(tri_ew, left_idx, right_idx, wav_center_idx)

    fit_width_idx = right_idx - left_idx
    fit_width_deg = fit_width_idx*long_res

    if right_deg > 360.:
        right_deg = right_deg - 360.
    if left_deg < 0.:
        left_deg += 360.

#    print left_idx, right_idx
#    plt.plot(tri_ew[left_idx:right_idx])
#    plt.show()
    
#    print left_deg, right_deg, gauss_center, gauss_sigma, gauss_base, gauss_height
#    print left_deg, right_deg, fit_width_idx, fit_width_deg, fit_height, clump_int_height, gauss_center, gauss_sigma, gauss_base, gauss_height
    return (left_deg, right_deg, fit_width_idx, fit_width_deg, clump_fit_height, clump_int_height, gauss_center, gauss_sigma, gauss_base, gauss_height )


#master_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'test_gauss_clumps_list.pickle')
#master_list_fp = open(master_list_fp, 'rb')
#clump_db, master_list = pickle.load(master_list_fp)
#master_list_fp.close()
#
##test only the master list
#clump_list = split_chains(master_list)
#clump_list = np.array(clump_list)

#keep_list = [0, 4,5]
#clump_list = clump_list[keep_list]
#for i, clump in enumerate(clump_list):
#    tri_ews = np.tile(clump.clump_db_entry.ew_data, 3)
#    long_res = 360./len(clump.clump_db_entry.ew_data)
#    right_idx = clump.fit_right_deg/long_res + len(clump.clump_db_entry.ew_data)
#    left_idx = clump.fit_left_deg/long_res + len(clump.clump_db_entry.ew_data)
#    if right_idx < left_idx:
#        right_idx += len(clump.clump_db_entry.ew_data)
#    fit_gaussian(tri_ews, left_idx, right_idx, clump.longitude_idx + len(clump.clump_db_entry.ew_data))

#now test ALL CLUMPS IN THE DATABASE
#i = 0
#for obsid in clump_db.keys():
#    for clump in clump_db[obsid].clump_list:
#        
#        
#        fit_gaussian(i, clump)
#        i += 1
#plt.show()
