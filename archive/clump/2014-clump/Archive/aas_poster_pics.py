'''
DPS POSTER PICTURE GENERATOR

Dependencies:
Ringutil.py

The current databases will produce the exact same pictures as the poster.
If you want new ones - copy whatever updated databases you have into the directory.
To run:
Uncomment/Comment things you do or don't want to run. 
Function executables are down at the bottom of the program.

Author: S Hicks
'''

from optparse import OptionParser
import ringutil
import pickle
import numpy as np
import sys
import os
from imgdisp import ImageDisp
import clumputil
import numpy.ma as ma
import scipy as sp
import matplotlib.pyplot as plt
import Image
import cwt
import cspice
from scipy import signal
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import scipy.optimize as sciopt
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
import matplotlib.transforms as mtransforms

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = [
#                '--prefilter',
                '--scale-min', '5', '--scale-max', '80',
                '--clump-size-min', '2.1', '--clump-size-max', '49.9',
                '--ignore-bad-obsids',
                '--save-clump-database',
                '--plot-scalogram',
                '--color-contours',
#                '--mosaic-reduction-factor', '1',
#                '-a'
                ]

parser = OptionParser()

parser.add_option('--scale-min',
                  type='float', dest='scale_min', default=1.,
                  help='The minimum scale (in full-width degrees) to analyze with wavelets')
parser.add_option('--scale-max',
                  type='float', dest='scale_max', default=30.,
                  help='The maximum scale (in full-width degrees) to analyze with wavelets')
parser.add_option('--scale-step',
                  type='float', dest='scale_step', default=0.1,
                  help='The scale step (in full-width degrees) to analyze with wavelets')
parser.add_option('--clump-size-min',
                  type='float', dest='clump_size_min', default=5.,
                  help='The minimum clump size (in full-width degrees) to detect')
parser.add_option('--clump-size-max',
                  type='float', dest='clump_size_max', default=15.,
                  help='The maximum clump size (in full-width degrees) to detect')
parser.add_option('--prefilter', dest='prefilter',
                  action='store_true', default=False,
                  help='Prefilter the data with a bandpass filter')
parser.add_option('--plot-scalogram', dest='plot_scalogram',
                  action='store_true', default=False,
                  help='Plot the scalogram and EW data')
parser.add_option('--update-clump-database', dest='update_clump_database',
                  action='store_true', default=False,
                  help='Read in the old clump database and update it by replacing selected OBSIDs')
parser.add_option('--replace-clump-database', dest='replace_clump_database',
                  action='store_true', default=False,
                  help='Start with a fresh clump database')
parser.add_option('--save-clump-database', dest='save_clump_database',
                  action='store_true', default=False,
                  help='Start with a fresh clump database')
parser.add_option('--dont-use-fft', dest='dont_use_fft',
                  action='store_true', default=False,
                  help='Use a non-FFT version of CWT')
parser.add_option('--color-contours', dest='color_contours',
                  action='store_true', default=False,
                  help='Use colored contours in scalogram')



ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)


ROOT = '/home/shannon/Documents/AAS_poster_pics/'
OBSID = 'ISS_059RF_FMOVIE002_VIMS'

root_clump_db = {}

axes_color = (0,0,0)
clump_color = (0.69, 0.13, 0.07 )
image_clump_color = (0.8, 0 , 0)
profile_color = 'black'
fdg_color = '#365B9E'

def downsample_ew(ew_range):
    bin = int(len(ew_range)/720.)
    new_len = int(len(ew_range)/bin)   # Make sure it's an integer size - this had better be 720!
    new_arr = ma.zeros(new_len)  # ma.zeros automatically makes a masked array
    src_mask = ma.getmaskarray(ew_range)
    for i in range(new_len):
        # Point i averages the semi-open interval [i*bin, (i+1)*bin)
        # ma.mean does the right thing - if all the values are masked, the result is also masked
        new_arr[i] = ma.mean(ew_range[i*bin:(i+1)*bin])
    
    return new_arr

def plot_single_clump(ax, ew_data, clump, long_min, long_max, label=False, color=clump_color):
    ncolor = color
    long_res = 360. / len(ew_data)
    longitudes = np.arange(len(ew_data)*3) * long_res - 360.
    mother_wavelet = cwt.SDG(len_signal=len(ew_data)*3, scales=np.array([int(clump.scale_idx/2)]))
    mexhat = mother_wavelet.coefs[0].real # Get the master wavelet shape
    mh_start_idx = round(len(mexhat)/2.-clump.scale_idx/2.)
    mh_end_idx = round(len(mexhat)/2.+clump.scale_idx/2.)
    mexhat = mexhat[mh_start_idx:mh_end_idx+1] # Extract just the positive part
    mexhat = mexhat*clump.mexhat_height+clump.mexhat_base
    longitude_idx = clump.longitude_idx
    if longitude_idx+clump.scale_idx/2 >= len(ew_data): # Runs off right side - make it run off left side instead
        longitude_idx -= len(ew_data)
    idx_range = longitudes[longitude_idx-clump.scale_idx/2+len(ew_data):
                           longitude_idx-clump.scale_idx/2+len(mexhat)+len(ew_data)] # Longitude range in data
    legend = None
    if label:
        legend = 'L=%7.2f W=%7.2f H=%6.3f' % (clump.longitude, clump.scale, clump.mexhat_height)
    ax.plot(idx_range, mexhat, '-', color=ncolor, lw=4, alpha=0.9, label=legend)
    if longitude_idx-clump.scale_idx/2 < 0: # Runs off left side - plot it twice
        ax.plot(idx_range+360, mexhat, '-', color=ncolor, lw=4, alpha=0.9)
    ax.set_xlim(long_min, long_max)
    
def plot_single_ew_profile(ax, ew_data, clump_db_entry, long_min, long_max, label=False, color= profile_color):
    
    long_res = 360. / len(ew_data)
    longitudes = np.arange(len(ew_data)) * long_res
    min_idx = int(long_min / long_res)
    max_idx = int(long_max / long_res)
    long_range = longitudes[min_idx:max_idx]
    ew_range = ew_data[min_idx:max_idx]
    legend = None
    if label:
        legend = clump_db_entry.obsid + ' (' + cspice.et2utc(clump_db_entry.et, 'C', 0) + ')'
    ax.plot(long_range, ew_range, '-', label=legend, color= profile_color, lw = 2.0)
    
def plot_fitted_clump_on_ew(ax, ew_data, clump):
    long_res = 360./len(ew_data)
    longitudes =np.tile(np.arange(0,360., long_res),3)
    tri_ew = np.tile(ew_data, 3)
    left_idx = clump.fit_left_deg/long_res + len(ew_data)
    right_idx = clump.fit_right_deg/long_res + len(ew_data)
    
    if left_idx > right_idx:
        left_idx -= len(ew_data)
    print left_idx, right_idx
    idx_range = longitudes[left_idx:right_idx]
    print idx_range
    if left_idx < len(ew_data):
        ax.plot(longitudes[left_idx:len(ew_data)-1], tri_ew[left_idx:len(ew_data)-1], color = 'blue', alpha = 0.5, lw = 2)
        ax.plot(longitudes[len(ew_data):right_idx], tri_ew[len(ew_data):right_idx], color = 'blue', alpha = 0.5, lw = 2)
    else:
        ax.plot(idx_range, tri_ew[left_idx:right_idx], color = 'blue', alpha = 0.5, lw = 2)
        
def fft_filter(options, mosaicdata, ew_data, plot = False):
        long_res = 360./len(ew_data)
        longitudes = np.arange(0,360, long_res) #18,000 element array
        
        ft = np.fft.fft(ew_data)
        power_ft = np.fft.fft(ew_data -np.mean(ew_data))
        
        ift = np.fft.ifft(ft)
        power = power_ft*power_ft.conjugate()
        
        freq = np.fft.fftfreq(int(longitudes.shape[0]), long_res)*360  # Weird int() cast is because of a problem with 64-bit Python on Windows
        
        power = ft*ft.conjugate()
        
        #PLOT SET 1: Original data and IFT, FFT, POWER
        if plot:
            fit = np.max(power)/freq
            ew_plot = plt.subplot(311)
            ew_plot.plot(longitudes, ew_data, 'red')
            ew_plot.plot(longitudes, ift, 'blue')
     
            ft_plot = plt.subplot(312)
            plt.plot(freq, ft.real, 'green')
            
            p_plot = plt.subplot(313)
            p_plot.plot(freq, power - fit, 'red')
            p_plot.plot(freq, fit, 'blue')
        
            plt.show()
        
        #HIGH PASS FILTER EVERYTHING (remove lower frequencies)
        ft_sub = np.fft.fft(ew_data)
        lose = np.where((freq > -50) & (freq < 50))[0]

        ft_sub[lose] = 0
        
        #LOW PASS FILTER - USE A BOXCAR WINDOWING FUNCTION (removes higher frequencies)
        ift_sub = np.fft.ifft(ft_sub)
        #repeat the signal so that the windowing function doesn't create edge effects
        ift_sub = np.tile(ift_sub, 3)
        box = signal.boxcar(5)
        
        smooth = np.convolve(box/box.sum(), ift_sub, mode = 'valid')

        smooth = smooth[(len(longitudes)-((len(box)/2))):(-(len(longitudes)-(len(box)/2)))]

        freq_ratio = len(freq)/len(smooth)
        
        #fft of the smooth array should show the 3.2 signal and harmonics
        result = np.fft.fft(smooth)
        new_freq = np.fft.fftfreq(len(smooth), long_res)*360
        pow = result*result.conjugate()
        
        if plot:
            low_filt = plt.subplot(411)
            low_filt.plot(freq, ft_sub)
            
            smooth_ft = plt.subplot(412)
            smooth_ft.plot(smooth)
            
            smooth_ew = plt.subplot(413)
            smooth_ew.plot(longitudes,(ew_data - np.mean(ew_data)) - smooth, 'red')
            smooth_ew.plot(smooth, 'blue')
            
            res_p = plt.subplot(414)
            res_p.plot(new_freq, result)
            plt.show()
        
        #now take the ft of the smoothed data and subtract it from the original ft
        #should result in the subtraction of the whole periodicity from the data      
        final_ft = ft - result
        final = np.fft.ifft(final_ft)
        
        if plot:
            final_plot = plt.subplot(211)
            final_plot.plot(longitudes,final, 'blue')
            
            ew_fin = plt.subplot(212)
            ew_fin.plot(longitudes, ew_data - np.mean(ew_data), 'red')
            plt.show()
            
        return (final)
    
def smooth_ew(ew_range, smooth_deg):
    long_res = 360./len(ew_range)
    smooth_pix = smooth_deg // long_res // 2
    #smooth the equivalent width range 
    smoothed_ew = ma.zeros(ew_range.shape[0])
    for n in range(len(ew_range)):
                    if ew_range.mask[n]:
                        smoothed_ew[n] = ma.masked
                    else:
                        smoothed_ew[n] = ma.mean(ew_range[max(n-smooth_pix,0):
                                                         min(n+smooth_pix+1,len(ew_range)-1)])
    return smoothed_ew
    
def detect_local_maxima(arr):
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    # Modified to detect maxima instead of minima
    """
    Takes an array and detects the peaks using the local minimum filter.
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_max = (filters.maximum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_maxima = local_max - eroded_background
    return np.where(detected_maxima)       
    
def refine_fit(clump, smoothed_ew, ew_data, longitudes):

    long_res = 360./len(ew_data)
    tri_long = np.tile(longitudes, 3)
    #for every clump that passes through - refine the fit
    
    tri_smooth = np.tile(smoothed_ew, 3)
    tri_ew = np.tile(ew_data, 3)
    
    wav_center_idx = clump.longitude_idx + len(smoothed_ew)
    wav_scale = clump.scale_idx
    
    left_ew_range = tri_smooth[wav_center_idx-wav_scale:wav_center_idx]
    right_ew_range = tri_smooth[wav_center_idx:wav_center_idx + wav_scale]

#the range should include the full clumps, and half of the scale size to either side of the clump
    
    left_idx = wav_center_idx - (wav_scale - np.argmin(left_ew_range))
    left_min = tri_ew[left_idx]
    
    right_idx = wav_center_idx + np.argmin(right_ew_range)
    right_min = tri_ew[right_idx]
   
    if len(tri_ew[left_idx:right_idx]) > 0:    
        fit_width_idx = abs(right_idx - left_idx)
        fit_width_deg = fit_width_idx*long_res
        fit_base = (tri_ew[left_idx] + tri_ew[right_idx])/2.
        fit_height = ma.max(tri_ew[left_idx:right_idx]) - fit_base
        print fit_width_idx, fit_width_deg, fit_base, fit_height

    else:
        fit_width_idx = abs(right_idx - left_idx)
        fit_width_deg = fit_width_idx*long_res
        fit_base = clump.mexhat_base
        fit_height = clump.mexhat_height
        
    left_deg = (left_idx - len(ew_data))*long_res
    right_deg = (right_idx - len(ew_data))*long_res
    if right_deg > 360.:
        right_deg = right_deg - 360.
    print right_deg, left_deg 
    return (left_deg, right_deg, fit_width_idx, fit_width_deg, fit_base, fit_height)  


def find_clumps_internal(options, ew_data, ew_mask, longitudes, obs_id, mosaic_data, root, wavelet_type = 'SDG'):
    
    (mosaic_longitudes, mosaic_resolutions, mosaic_image_numbers,
 mosaic_ETs, mosaic_emission_angles, mosaic_incidence_angles,
 mosaic_phase_angles) = mosaic_data
    
    print ew_data.mask
    smoothed_ew = smooth_ew(ew_data, 3.5)
    
#    ew_data *= ringutil.normalized_ew_factor(mean_phase, mean_emission)
    long_res = 360./len(ew_data)
    tripled_ew_data = np.tile(ew_data, 3)
    orig_ew_start_idx = len(ew_data)
    orig_ew_end_idx = len(ew_data)*2
    orig_ew_start_long = 360.
    orig_ew_end_long = 720.
    
    # Create the scales at which we wish to do the analysis
    # The scale is the half-width of the wavelet
    # The user gives us the scale in degrees, but we need to convert it to indices
    scale_min = options.scale_min/2.
    scale_max = options.scale_max/2.
    scale_step = options.scale_step/2.
    print long_res
    wavelet_scales = np.arange(scale_min/long_res, scale_max/long_res+0.0001, scale_step/long_res)
    
    if wavelet_type == 'SDG':
        mother_wavelet = cwt.SDG(len_signal=len(tripled_ew_data), scales=wavelet_scales)
    if wavelet_type == 'FDG': 
        mother_wavelet = cwt.FDG(len_signal=len(tripled_ew_data), scales=wavelet_scales)
        
    wavelet = cwt.cwt(tripled_ew_data, mother_wavelet)

    tripled_xform = wavelet.coefs.real
    xform_maxima_scale_idx, xform_maxima_long_idx = detect_local_maxima(tripled_xform) # Includes tripled info
    
    def fit_wavelet_func(params, wavelet, data):
                base, wscale = params
                return np.sum((wavelet*wscale+base-data)**2)
            
            # For each found clump, fit the wavelet to it and see if it's OK
            
    clump_list = []
        
    for maximum_scale_idx, maximum_long_idx in zip(xform_maxima_scale_idx, xform_maxima_long_idx):
        longitude = maximum_long_idx * long_res
        scale = wavelet_scales[maximum_scale_idx] * long_res * 2 # Convert to full-width degrees
        if options.clump_size_max >= scale >= options.clump_size_min:
            long_start_deg = longitude-scale/2. # Left side of positive part of wavelet
            long_end_deg = longitude+scale/2. # Right side
            if longitude < orig_ew_start_long or longitude >= orig_ew_end_long:
                # If the clump is more than half out of the center EW range, we don't care about it
                # because it will be picked up on the other side
                continue
            if tripled_ew_data.mask[maximum_long_idx]:
                # The clump is centered on masked data - don't trust it
                continue
            long_start_idx = round(long_start_deg/long_res)
            long_end_idx = round(long_end_deg/long_res)
            mexhat = mother_wavelet.coefs[maximum_scale_idx].real # Get the master wavelet shape
            mh_start_idx = int(len(mexhat)/2.-wavelet_scales[maximum_scale_idx])
            mh_end_idx = int(len(mexhat)/2.+wavelet_scales[maximum_scale_idx])
            restr_mexhat = mexhat[mh_start_idx:mh_end_idx+1] # Extract just the positive part
            extracted_data = tripled_ew_data[long_start_idx:long_start_idx+len(restr_mexhat)]
            mexhat_data, residual, array, trash, trash, trash = sciopt.fmin_powell(fit_wavelet_func,(1., 1.), args=(restr_mexhat, extracted_data),disp=False, full_output=True)
#            print stuff, len(stuff)
            mexhat_base, mexhat_height = mexhat_data
            
            if mexhat_height < 0:
                continue
            longitude -= orig_ew_start_long
            print 'CLUMP LONG %6.2f WIDTH %5.2f BASE %7.4f HEIGHT %7.4f' % (longitude, scale,
                                                                           mexhat_base, mexhat_height)           
            clump = clumputil.ClumpData()
            clump.longitude = longitude
            clump.scale = scale
            clump.longitude_idx = maximum_long_idx-orig_ew_start_idx
            clump.scale_idx = wavelet_scales[maximum_scale_idx]*2.
            clump.mexhat_base = mexhat_base
            clump.mexhat_height = mexhat_height
            clump.abs_height = clump.mexhat_height*mexhat[len(mexhat)//2]
            clump.max_long = maximum_long_idx
            clump.matched = False
            clump.residual = residual
            clump.wave_type = wavelet_type
            
            #calculate the refined fit parameters
            clump.fit_left_deg, clump.fit_right_deg, clump.fit_width_idx, clump.fit_width_deg, clump.fit_base, clump.fit_height = refine_fit(clump, ew_data, smoothed_ew, longitudes)
            #calculate clump sigma height
            profile_sigma = ma.std(ew_data)
            height = clump.mexhat_height*mexhat[len(mexhat)//2]
            clump.clump_sigma = clump.abs_height/profile_sigma
            clump.fit_sigma = clump.fit_height/profile_sigma
            
            clump_list.append(clump)

    clump_db_entry = clumputil.ClumpDBEntry()
    clump_db_entry.obsid = obs_id
    clump_db_entry.et = np.mean(mosaic_ETs[~ew_mask])
    clump_db_entry.resolution_min = np.min(mosaic_resolutions[~ew_mask])
    clump_db_entry.resolution_max = np.max(mosaic_resolutions[~ew_mask])
    clump_db_entry.emission_angle = np.mean(mosaic_emission_angles[~ew_mask])
    clump_db_entry.incidence_angle = np.mean(mosaic_incidence_angles[~ew_mask])
    clump_db_entry.phase_angle = np.mean(mosaic_phase_angles[~ew_mask])
    clump_db_entry.ew_data = ew_data # Filtered and normalized
#    clump_db_entry.smoothed_ew = smoothed_ew
    clump_db_entry.clump_list = clump_list
    
#    root_clump_db[obs_id] = clump_db_entry
    wavelet_data = (wavelet, mother_wavelet, wavelet_scales)
    return (clump_db_entry, wavelet_data)

def plot_scalogram(sdg_wavelet_data, fdg_wavelet_data, clump_db_entry):
    
    ew_data = clump_db_entry.ew_data
    clump_list = clump_db_entry.clump_list
    obs_id = clump_db_entry.obsid
    
    orig_ew_start_idx = len(ew_data)
    orig_ew_end_idx = len(ew_data)*2
    orig_ew_start_long = 360.
    orig_ew_end_long = 720.
    
    long_res = 360./len(ew_data)
    longitudes = np.arange(len(ew_data))*long_res
    
    wavelet_forms = ['SDG', 'FDG']
    for wavelet_form in wavelet_forms:
        if wavelet_form == 'SDG':
            wavelet, mother_wavelet, wavelet_scales = sdg_wavelet_data
        if wavelet_form == 'FDG':
            wavelet, mother_wavelet, wavelet_scales = fdg_wavelet_data

        xform = wavelet.coefs[:,orig_ew_start_idx:orig_ew_end_idx].real # .real would need to be changed if we use a complex wavelet
        scales_axis = wavelet_scales * long_res * 2 # Full-width degrees for plotting

        color_background = (1,1,1)
        color_foreground = (0,0,0)
        color_dark_grey = (0.5, 0.5, 0.5)
        color_grey = (0.375, 0.375, 0.375)
        color_bright_grey = (0.25, 0.25, 0.25)
        figure_size = (12.0,3.0)
        font_size = 18.0
        fig = plt.figure(figsize = figure_size, facecolor = color_background, edgecolor = color_background, linewidth = 5., dpi = 600)
       
        # Make the contour plot of the wavelet transform
        ax1 = fig.add_subplot(111)
        plt.subplots_adjust(left = 0.06, right = 0.98)
        ax1.tick_params(length = 5., width = 2., labelsize = 14. )
        ax1.yaxis.tick_left()
        ax1.xaxis.tick_bottom()
        ax1.get_yaxis().set_ticks([20.0, 40., 60., 80.])
        
        if options.color_contours:
            ax1.contourf(longitudes, scales_axis, xform, 256)
        else:
            ct = ax1.contour(longitudes, scales_axis, xform, 32)
            ax1.clabel(ct, inline=1, fontsize=8)
    
        ax1.set_ylim((scales_axis[0], scales_axis[-1]))
        ax1.set_xlim((longitudes[0], longitudes[-1]))
        ax1.set_ylabel(r'Full Width Scale ( $\mathbf{^o}$)', fontsize = font_size)
        ax1.get_xaxis().set_ticks([0,90,180,270,360])
        
        # Make the data plot with overlaid detected clumps
        plt.savefig(os.path.join(ROOT, OBSID + '_' + wavelet_form + '_scalogram_contours.png'), dpi = 600, transparent = True)
       
        figure_size = (12.0, 3.0)
        font_size = 18.0
    
        fig = plt.figure(figsize = figure_size, facecolor = color_background, edgecolor = color_background, linewidth = 5.)
        ax2 = fig.add_subplot(111)
        plt.subplots_adjust(top = 0.9, bottom = 0.175, left = 0.060, right = 0.980)
        ax2.plot(longitudes, ew_data, color = 'black')
        ax2.get_yaxis().set_ticks([0.6, 0.8, 1.0, 1.2, 1.4])

        tripled_longitudes = np.append(longitudes-360., np.append(longitudes, longitudes+360.))
                
        for clump_data in clump_list:
            if clump_data.fit_sigma > 0.7:
#                print clump_data.wave_type
                if clump_data.wave_type == 'SDG':
                    wavelet_color = clump_color
                if clump_data.wave_type == 'FDG':
                    wavelet_color = fdg_color
                plot_single_clump(ax2, ew_data, clump_data, 0., 360., color = wavelet_color)
        
        ax2.set_xlim((longitudes[0], longitudes[-1]))
        ax2.set_ylim(ma.min(ew_data), ma.max(ew_data))
        ax2.tick_params(length = 5., width = 2., labelsize = 14. )
        ax2.yaxis.tick_left()
        ax2.xaxis.tick_bottom()
        ax2.get_xaxis().set_ticks([0,90,180,270,360])

        # align time series fig with scalogram fig
        t = ax2.get_position()
        ax2pos = t.get_points()
        ax2pos[1][0] = ax1.get_position().get_points()[1][0]
        t.set_points(ax2pos)
        ax2.set_position(t)
        ax2.set_xlabel(r'Co-Rotating Longitude ( $\mathbf{^o}$)', fontsize = font_size)
        ax2.set_ylabel('Equivalent Width (km)', fontsize = font_size)
    
        plt.savefig(os.path.join(ROOT, OBSID + '_scalogram_profile.png'), dpi = 600, transparent = True)
        
#        plt.show()

def select_clumps(options, ew_data, ew_mask, longitudes, obs_id, metadata, root):
    
    sdg_clump_db_entry, sdg_wavelet_data = find_clumps_internal(options, ew_data, ew_mask, longitudes, obs_id, metadata, root, wavelet_type = 'SDG')
    fdg_clump_db_entry, fdg_wavelet_data = find_clumps_internal(options, ew_data, ew_mask, longitudes, obs_id, metadata, root, wavelet_type = 'FDG')
    
    sdg_list = sdg_clump_db_entry.clump_list
    fdg_list = fdg_clump_db_entry.clump_list
    
    new_clump_list = []
    longitude_list = []
    for clump_s in sdg_list:
        print 'new clump_s', clump_s.max_long
        for clump_f in fdg_list:
#            print 'new_clump_f', clump_f.max_long
            if (clump_s.max_long -50 <= clump_f.max_long <= clump_s.max_long + 50) and (0.8*clump_s.scale <= clump_f.scale <= 1.2*clump_s.scale):
                #there is a match!
                clump_f.matched = True
                clump_s.matched = True
                print 'match'
                if clump_f.residual < clump_s.residual:
                    new_clump_list.append(clump_f)
                    longitude_list.append(clump_f.longitude)
                else:
                    new_clump_list.append(clump_s)
                    longitude_list.append(clump_s.longitude)
                break
    
    #check for any left-over fdg clumps that weren't matched. 
    #need to make sure they're not at the same longitudes as clumps that were already matched.
    #this causes double chains to be made - making for a difficult time in the gui selection process
    if options.voyager or options.downsample:
        range = 0.5
    else:
        range = 0.04
    
    for clump_f in fdg_list:
        if clump_f.matched == False:
            for longitude in longitude_list:
                if longitude - range <= clump_f.longitude <= longitude + range:
                    clump_f.matched = True
                    break
            if clump_f.matched:
                continue        #already matched, go to the next f clump
            else:
                new_clump_list.append(clump_f)
    
    for clump_s in sdg_list:
        if clump_s.matched == False:
            for longitude in longitude_list:
                if longitude - range <= clump_s.longitude <= longitude + range:
                    clump_s.matched = True
                    break                   #label it as already matched, go to the next s clump
            if clump_s.matched:
                break
            else:
                new_clump_list.append(clump_s)
       
    fdg_clump_db_entry.clump_list = new_clump_list      
    if options.plot_scalogram:
        plot_scalogram(sdg_wavelet_data, fdg_wavelet_data, fdg_clump_db_entry)
        
def find_clump_chain(options, clump_db):
    
    obs_list = ['ISS_030RF_FMOVIE001_VIMS', 'ISS_031RF_FMOVIE001_VIMS',
                 'ISS_032RF_FMOVIE001_VIMS', 'ISS_033RF_FMOVIE001_VIMS']
    clump_longs = ['224.68', '226.32', '227.16', '227.16']
    
    chain_list = []
    for i, obs in enumerate(obs_list):
        print obs, i
        for clump in clump_db[obs].clump_list:
            print clump.longitude, clump_longs[i]
            if str(clump.longitude) == clump_longs[i]:
                print 'MATCH!'
                chain_list.append((clump, obs))
    return chain_list

def calc_polar_params(im, width, center, longitudes):
    
    theta_step = (2*np.pi)/im.shape[1]
    theta_array = np.arange(0, 2*np.pi, theta_step)
    
    longitudes = np.where(longitudes >= 0)[0]
    rad = np.arange(0,im.shape[0])
    long_rep = np.repeat(longitudes,im.shape[0])
    radii = np.tile(rad, longitudes.shape[0])
    
    dx= radii*np.cos(long_rep*theta_step)+center
    dy= radii*np.sin(long_rep*theta_step)+center
    dx = np.round(dx).astype(int)
    dy = np.round(dy).astype(int)
    
    return (dx, dy, radii, long_rep, theta_step, theta_array, center)

def draw_clumps(im, clump, options, obsid, color, radius_start):
#    color = (255, 0, 0)
    print color
    radii = np.arange(im.shape[0])*options.radius_resolution + radius_start
    radius_center = np.where(radii == 140220)[0][0]
#    print radius_center
    width = (clump.fit_width_deg//2)/(360./im.shape[1]) #pixels
    height = 30 #pixels
    center = clump.longitude/(360./im.shape[1])
    l_thick = 4
    w_thick = 24
    for i in range(im.shape[2]):
                if (center + width > im.shape[1]):
                    rem = (center + width) - im.shape[1]
                    im[radius_center + height:radius_center + height + l_thick, center-width:im.shape[1], i] = color[i] #lines to the end
                    im[radius_center -height -l_thick:radius_center - height+l_thick,center - width:im.shape[1], i] = color[i]
                    im[radius_center -height -l_thick: radius_center + height +l_thick ,center-width:center-width+w_thick, i] = color[i] # left boundary
                    im[radius_center +height:radius_center + height + l_thick, 0:rem + l_thick, i] = color[i] #extend top line past 0
                    im[radius_center - height -l_thick:radius_center - height + l_thick, 0:rem +l_thick, i] = color[i] # extend bottom line
                    im[radius_center - height -l_thick:radius_center + height +l_thick, rem:rem + w_thick, i] = color[i] #right boundary
                    print im[radius_center - height -l_thick:radius_center + height +l_thick, rem:rem + w_thick, i]
                if (center - width < 0):
                    rem =  abs(center - width)
                    im[radius_center + height:radius_center + height+l_thick, 0:center + width +l_thick, i] = color[i]
                    im[radius_center - height - l_thick:radius_center - height +l_thick, 0:center + width +l_thick, i] = color[i]
                    im[radius_center - height -l_thick:radius_center - height, im.shape[1] - rem -(l_thick +1):im.shape[1], i] = color[i]
                    im[radius_center + height: radius_center + height + l_thick, im.shape[1] - rem -(l_thick +1): im.shape[1], i] = color[i]
                    im[radius_center - height -l_thick:radius_center + height +l_thick, center + width: center + width + w_thick, i] = color[i]
                    im[radius_center -height -l_thick: radius_center + height +l_thick, im.shape[1] - rem -w_thick:im.shape[1] - rem, i] = color[i]
                else:
                    im[radius_center + height:radius_center + height + l_thick, center-width -l_thick:center + width +l_thick, i] = color[i]
                    im[radius_center -height - l_thick:radius_center - height,center - width -l_thick:center + width +l_thick, i] = color[i]
                    im[radius_center -height -l_thick: radius_center + height + l_thick,center-width:center-width+w_thick, i] = color[i]
                    im[radius_center - height -l_thick: radius_center + height + l_thick, center + width:center + width + w_thick, i] = color[i]
#                    print im[radius_center - height -l_thick: radius_center + height + l_thick, center + width:center + width + w_thick, i]
    return (center, radius_center)

def make_rad_mosaic(im, obs_id, longitudes, clump, options):
    
    im = im[250:650, :]
    radius_start = options.radius_start + 250*options.radius_resolution
    
    combined_img = np.zeros((im.shape[0], im.shape[1], 3))
    combined_img[:, :, 0] = im
    combined_img[:, :, 1] = im
    combined_img[:, :, 2] = im
    mode = 'RGB'
    
#    print clump_color
    clump_center = draw_clumps(combined_img, clump, options, obs_id, image_clump_color, radius_start)
    
    width = 2*im.shape[0]
    center = int(width/2)

    final_im = np.zeros((width, width, 3))  
    #new image should be a square a little larger than the 2*height of the original image.
    dx, dy, radii, long_rep, theta_step, theta_arr, center = calc_polar_params(im, width, center, longitudes)
    final_im[dy, dx, :] = np.maximum(combined_img[radii, long_rep, :], 0)**.5
    
    blackpoint = 0
    whitepoint = options.global_whitepoint
    gamma = 1.
    
    final_im = ImageDisp.ScaleImage(final_im, blackpoint, whitepoint, gamma)[::-1,:]+0
    final_im = np.cast['int8'](final_im)
    final_img = Image.frombuffer(mode, (final_im.shape[1], final_im.shape[0]),
                           final_im, 'raw', mode, 0, 1)
  
    sp.misc.imsave(ROOT + obs_id + '.png', final_img)    


def plot_only_profile(longitudes, ew_data):
    
    #DEFINE MATPLOTLIB.RC PARAMS HERE
    color_background = (1,1,1)
    color_foreground = (0,0,0)
    color_dark_grey = (0.5, 0.5, 0.5)
    color_grey = (0.375, 0.375, 0.375)
    color_bright_grey = (0.25, 0.25, 0.25)
    figure_size = (12.0, 3.0)
    font_size = 18.0
    
    fig = plt.figure(figsize = figure_size, facecolor = color_background, edgecolor = color_background, linewidth = 5.)
    ax = fig.add_subplot(111)
    plt.subplots_adjust(top = 0.9, bottom = 0.175, left = 0.060, right = 0.980)
    ax.tick_params(length = 5., width = 2., labelsize = 14. )
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    
#    ax.set_title(OBSID, fontsize = 24.0)
    ax.set_ylabel('Equivalent Width (km)', fontsize = font_size)
    ax.set_xlabel(r'Co-Rotating Longitude ( $\mathbf{^o}$)', fontsize = font_size)
    ax.set_xlim(0,360.)
    ax.get_xaxis().set_ticks([0,90,180,270,360])
    ax.get_yaxis().set_ticks([0.6, 0.8, 1.0, 1.2, 1.4])

    plot_single_ew_profile(ax, ew_data, root_clump_db[OBSID], 0., 360., label = True)
    plt.savefig(os.path.join(ROOT, OBSID + '_profile.png'), dpi = 600,transparent = True)

def plot_clumps(ew_data):
    
    #DEFINE MATPLOTLIB.RC PARAMS HERE
    color_background = (1,1,1)
    color_foreground = (0,0,0)
    color_dark_grey = (0.5, 0.5, 0.5)
    color_grey = (0.375, 0.375, 0.375)
    color_bright_grey = (0.25, 0.25, 0.25)
    figure_size = (12.0, 3.0)
    font_size = 18.0
    
    fig = plt.figure(figsize = figure_size, facecolor = color_background, edgecolor = color_background, linewidth = 5.)
    ax = fig.add_subplot(111)
    plt.subplots_adjust(top = 0.9, bottom = 0.175, left = 0.060, right = 0.980)
    ax.tick_params(length = 5., width = 2., labelsize = 14. )
    ax.get_xaxis().set_ticks([0,90,180,270,360])
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.get_yaxis().set_ticks([0.6, 0.8, 1.0, 1.2, 1.4])

    ax.set_ylabel('Equivalent Width (km)', fontsize = font_size)
    ax.set_xlabel(r'Co-Rotating Longitude ( $\mathbf{^o}$)', fontsize = font_size)
    ax.set_xlim(0., 360.)
    plot_single_ew_profile(ax, ew_data, root_clump_db[OBSID], 0., 360., label = True)
    for clump in root_clump_db[OBSID].clump_list:
        plot_single_clump(ax, ew_data, clump, 0., 360.)
    
    plt.savefig(os.path.join(ROOT, OBSID + '_clump_profile.png'), dpi = 600, transparent = True)
    
def plot_chain_on_ews(chain, full_clump_db):
    
    color_background = (1,1,1)
    color_foreground = (0,0,0)
    color_dark_grey = (0.5, 0.5, 0.5)
    color_grey = (0.375, 0.375, 0.375)
    color_bright_grey = (0.25, 0.25, 0.25)
    figure_size = (5.5,5.5)
    font_size = 18.0
    
    
    for clump_data in chain:
        
        clump, obs = clump_data
        (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
         bkgnd_mask_filename, bkgnd_model_filename,
         bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obs)
        
        mosaic_metadata_fp = open(reduced_mosaic_metadata_filename, 'rb')
        mosaic_data = pickle.load(mosaic_metadata_fp)
        
        (mosaic_longitudes, mosaic_resolutions, mosaic_image_numbers,
         mosaic_ETs, mosaic_emission_angles, mosaic_incidence_angles,
         mosaic_phase_angles) = mosaic_data
        
        (ew_data_filename, ew_mask_filename) = ringutil.ew_paths(options, obs)
        ew_data = np.load(ew_data_filename + '.npy')
        ew_mask = np.load(ew_mask_filename + '.npy')

        mean_emission = np.mean(mosaic_emission_angles[~ew_mask])
        mean_phase = np.mean(mosaic_phase_angles[~ew_mask])

        ew_data *= ringutil.normalized_ew_factor(mean_phase, mean_emission)
        ew_data = ew_data.view(ma.MaskedArray)
        ew_data.mask = ew_mask
        ew_data = smooth_ew(ew_data, 3.0)
        
        if options.prefilter:
            ew_data = fft_filter(options, None, ew_data, plot = False)
            ew_data = ew_data.real
        
        # We have to wait to add the mask because masked arrays don't survive the FFT filter
        ew_data = ew_data.view(ma.MaskedArray)
        ew_data.mask = ew_mask
        
        fig = plt.figure(figsize = figure_size, facecolor = color_background, edgecolor = color_background, linewidth = 5.0)
        ax = fig.add_subplot(111)
        plt.subplots_adjust(right = 0.95)
        ax.tick_params(length = 5., width = 2., labelsize = 14. )
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()
        ax.get_yaxis().set_ticks([2.0, 2.4, 2.8, 3.2, 3.6, 4.0])
        ax.get_xaxis().set_ticks([210., 220., 230., 240.])
        ax.set_ylabel('Equivalent Width (km)', fontsize = font_size)
        ax.set_xlabel(r'Co-Rotating Longitude ( $\mathbf{^o}$)', fontsize = font_size)
        
        plot_single_ew_profile(ax, ew_data, full_clump_db[obs], 210., 240., label = True)
        plot_single_clump(ax, ew_data, clump, 210., 240.)
        plt.savefig(os.path.join(ROOT, obs + '_chain_profile.png'), dpi = 600, transparent = True)
            
def plot_mexhat(options):
    
    wavelet_scales = np.arange(0,26.0)
    x = np.arange (-100.0, 100.1, 1.0)
    mother_wavelet = cwt.SDG(len_signal=len(x), scales=wavelet_scales)
    
    color_background = (1,1,1)
    color_foreground = (0,0,0)
    color_dark_grey = (0.5, 0.5, 0.5)
    color_grey = (0.375, 0.375, 0.375)
    color_bright_grey = (0.25, 0.25, 0.25)
    figure_size = (6.75,3.)
    font_size = 20.0
    
    fig = plt.figure(figsize = figure_size, facecolor = color_background, edgecolor = color_background, linewidth = 5., dpi = 600)
    ax = fig.add_subplot(111, frame_on = False)
    fig.subplots_adjust(top = .98, bottom = 0.02, right = .98, left = 0.02)
    ax.tick_params(length = 5., width = 2., labelsize = 14. )

    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_ylim(-.5,1.0)    
    ax.plot(x, mother_wavelet.coefs[25].real, '-', linewidth = 10.0, color = clump_color)
#    plt.show()
    plt.savefig(os.path.join(ROOT, 'mexhat_wavelet.png'), dpi = 300, transparent = True)
    
def plot_FDG(options):
    
    wavelet_scales = np.arange(0,26.0)
    x = np.arange (-100.0, 100.1, 1.0)
    mother_wavelet = cwt.FDG(len_signal=len(x), scales=wavelet_scales)
    
    color_background = (1,1,1)
    color_foreground = (0,0,0)
    color_dark_grey = (0.5, 0.5, 0.5)
    color_grey = (0.375, 0.375, 0.375)
    color_bright_grey = (0.25, 0.25, 0.25)
    figure_size = (6.75,3.)
    font_size = 20.0
    
    fig = plt.figure(figsize = figure_size, facecolor = color_background, edgecolor = color_background, linewidth = 5., dpi = 600)
    ax = fig.add_subplot(111, frame_on = False)
    fig.subplots_adjust(top = .98, bottom = 0.02, right = .98, left = 0.02)
    ax.tick_params(length = 5., width = 2., labelsize = 14. )

    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_ylim(-2.0,3.0)    
    ax.plot(x, mother_wavelet.coefs[25].real, '-', linewidth = 10.0, color = '#365B9E')
#    plt.show()
    plt.savefig(os.path.join(ROOT, 'FDG_wavelet.png'), dpi = 300, transparent = True)
    
def edit_mosaic(mosaic):
    
    mosaic = mosaic[300:800, :]
    mode = 'L'
    
    blackpoint = 0
    whitepoint = np.max(mosaic)*.25
    gamma = 1.0
    
    final_im = ImageDisp.ScaleImage(mosaic, blackpoint, whitepoint, gamma)[::-1, :]+0
    final_im = np.cast['int8'](final_im)
    final_img = Image.frombuffer(mode, (final_im.shape[1], final_im.shape[0]),
                           final_im, 'raw', mode, 0, 1)
    
    sp.misc.imsave(ROOT + OBSID + '_mosaic_edit.png', final_img)    

FRING_MEAN_MOTION = 581.964

def RelativeRateToSemimajorAxis(rate):  # from ringutil
    return ((FRING_MEAN_MOTION / (FRING_MEAN_MOTION+rate*86400.))**(2./3.) * 140220.)

def make_vel_hist():

    approved_list_fp = os.path.join(ROOT, 'approved_clumps_list.pickle')
    approved_list_fp = open(approved_list_fp, 'rb')
    approved_db, approved_list = pickle.load(approved_list_fp)
    approved_list_fp.close()

    velocities = []
    
    for chain in approved_list:
        velocities.append(chain.rate)
    
    velocities = np.array(velocities)*86400.            #change to deg/day

    
#    velocities = np.array([-0.0122, 0.0686, 0.07402, 0.2338, 0.1907, 0.4218, 0.04746, -0.08209,
#                   -.04511, 0.09507, -0.0739, 0.074, 0.02276, -0.0923, 0.01268, 0.05272,
#                   -0.17065, 0.06402, 0.01921, -0.04132])
    
    figure_size = (10., 4.0)
    fig = plt.figure(figsize = figure_size)
    ax_mm = SubplotHost(fig, 1,1,1)
    
    a0 = RelativeRateToSemimajorAxis(-1./86400) # -1 deg/day
    a1 = RelativeRateToSemimajorAxis(1./86400)  # +1 deg/day
    slope = (a1-a0) / 2.

    aux_trans = mtransforms.Affine2D().translate(-220,0)
    aux_trans.scale(1/slope)
    ax_sma = ax_mm.twin(aux_trans)
    ax_sma.set_viewlim_mode("transform")

    fig.add_subplot(ax_mm)

    plt.subplots_adjust(top = .85, bottom = 0.15, left = 0.08, right = 0.98)
    
    ax_mm.set_xlabel('Relative Mean Motion ( $\mathbf{^o}$/Day )', fontsize = 18.0)
    ax_sma.set_xlabel('Semimajor Axis (km above 140,000)', fontsize = 18.0)
    ax_mm.yaxis.tick_left()

    ax_sma.get_yaxis().set_ticks([])
    ax_mm.get_xaxis().set_ticks(np.arange(-.4,.5,.05))
    ax_sma.get_xaxis().set_ticks(np.arange(120,300,10))

    ax_mm.set_ylabel('Fraction of Counts', fontsize = 18.0)
    
    graph_min = np.min(velocities)
    graph_max = np.max(velocities)
    step = 0.05
    ax_mm.set_xlim(-0.5, 0.5)
    bins = np.arange(-.4,0.4,step)
    
    counts, bins, patches = ax_mm.hist(velocities, bins, weights = np.zeros_like(velocities) + 1./velocities.size,
                                        color = 'black', alpha = 0.9, histtype = 'stepfilled', lw = 0.0)
    
    plt.savefig(os.path.join(ROOT, 'Clump_Velocity_Dist_Dual_Axis.png'), dpi = 600, transparent = True)
    
def make_voyager_vel_hist():
        
    velocities = np.array([-0.44, -0.306, -0.430, -0.232, -0.256, -0.183, -0.304, -0.344,
                       -.410, -0.174, -0.209, -0.289, -0.284, -0.257, -0.063, -0.310,
                       -0.277, -0.329, -0.290, -0.412, -0.198, 0.121, -0.258, -0.015, 
                       -0.299, -0.370, -0.195, -0.247, -0.419, -0.303, -0.213, -0.172, -0.010]) + 0.306
    
    figure_size = (10., 4.0)
    fig = plt.figure(figsize = figure_size)
    ax_mm = SubplotHost(fig, 1,1,1)
    
    a0 = RelativeRateToSemimajorAxis(-1./86400) # -1 deg/day
    a1 = RelativeRateToSemimajorAxis(1./86400)  # +1 deg/day
    slope = (a1-a0) / 2.

    aux_trans = mtransforms.Affine2D().translate(-220,0)
    aux_trans.scale(1/slope)
    ax_sma = ax_mm.twin(aux_trans)
    ax_sma.set_viewlim_mode("transform")

    fig.add_subplot(ax_mm)

    plt.subplots_adjust(top = .85, bottom = 0.15, left = 0.08, right = 0.98)
    
    ax_mm.set_xlabel('Relative Mean Motion ( $\mathbf{^o}$/Day )', fontsize = 18.0)
    ax_sma.set_xlabel('Semimajor Axis (km above 140,000)', fontsize = 18.0)
    ax_mm.yaxis.tick_left()

    ax_sma.get_yaxis().set_ticks([])
    ax_mm.get_xaxis().set_ticks(np.arange(-.4,.5,.05))
    ax_sma.get_xaxis().set_ticks(np.arange(120,300,10))

    ax_mm.set_ylabel('Fraction of Counts', fontsize = 18.0)
    
    graph_min = np.min(velocities)
    graph_max = np.max(velocities)
    step = 0.05
    ax_mm.set_xlim(-0.4, 0.5)
    bins = np.arange(-.4,0.4,step)
    
    counts, bins, patches = ax_mm.hist(velocities, bins, weights = np.zeros_like(velocities) + 1./velocities.size,
                                        color = '#AC7CC9', alpha = 0.9, histtype = 'stepfilled', lw = 0.0)
    
    plt.savefig(os.path.join(ROOT, 'Voyager_Clump_Velocity_Dist_Dual_Axis.png'), dpi = 600, transparent = True)
    
def combined_vel_hist():
    
    approved_list_fp = os.path.join(ROOT, 'approved_clumps_list.pickle')
    approved_list_fp = open(approved_list_fp, 'rb')
    approved_db, approved_list = pickle.load(approved_list_fp)
    approved_list_fp.close()

    c_velocities = []
    
    for chain in approved_list:
        c_velocities.append(chain.rate)
    
    c_velocities = np.array(c_velocities)*86400.            #change to deg/day
    
    v_velocities = np.array([-0.44, -0.306, -0.430, -0.232, -0.256, -0.183, -0.304, -0.344,
                       -.410, -0.174, -0.209, -0.289, -0.284, -0.257, -0.063, -0.310,
                       -0.277, -0.329, -0.290, -0.412, -0.198, 0.121, -0.258, -0.015, 
                       -0.299, -0.370, -0.195, -0.247, -0.419, -0.303, -0.213, -0.172, -0.010]) + 0.306
    
    figure_size = (10., 4.0)
    fig = plt.figure(figsize = figure_size)
    ax_mm = SubplotHost(fig, 1,1,1)
    
    a0 = RelativeRateToSemimajorAxis(-1./86400) # -1 deg/day
    a1 = RelativeRateToSemimajorAxis(1./86400)  # +1 deg/day
    slope = (a1-a0) / 2.

    aux_trans = mtransforms.Affine2D().translate(-220,0)
    aux_trans.scale(1/slope)
    ax_sma = ax_mm.twin(aux_trans)
    ax_sma.set_viewlim_mode("transform")

    fig.add_subplot(ax_mm)

    plt.subplots_adjust(top = .85, bottom = 0.15, left = 0.08, right = 0.98)
    
    ax_mm.set_xlabel('Relative Mean Motion ( $\mathbf{^o}$/Day )', fontsize = 18.0)
    ax_sma.set_xlabel('Semimajor Axis (km above 140,000)', fontsize = 18.0)
    ax_mm.yaxis.tick_left()

    ax_sma.get_yaxis().set_ticks([])
    ax_mm.get_xaxis().set_ticks(np.arange(-.4,.4,.05))
    ax_sma.get_xaxis().set_ticks(np.arange(120,300,10))

    ax_mm.set_ylabel('Fraction of Counts', fontsize = 18.0)
    
    graph_min = np.min(c_velocities)
    graph_max = np.max(c_velocities)
    step = 0.05
    ax_mm.set_xlim(-0.4, 0.4)
    bins = np.arange(-.4,0.4,step)
    
    counts, bins, patches = plt.hist([v_velocities, c_velocities], bins,
                                    weights = [np.zeros_like(v_velocities) + 1./v_velocities.size, np.zeros_like(c_velocities) + 1./c_velocities.size],
                                    label = ['Voyager', 'Cassini'], color = ['#AC7CC9', 'black'], lw = 0.0)
    
    leg =plt.legend()
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
    
    plt.savefig(os.path.join(ROOT, 'Combined_Clump_Velocity_Dist_Dual_Axis.png'), dpi = 600, transparent = True)
#    plt.show()
def list_to_db(v_approved_list, c_approved_list, c_clump_db, v_clump_db):
    
    #make double databases that have clump lists with only approved clumps.
#    v_clump_db_entry = clumputil.ClumpDBEntry()
       
    v_db = {}
    for chain in v_approved_list:
        for clump in chain.clump_list:
            v_clump_db_entry = clumputil.ClumpDBEntry()
            obsid = clump.clump_db_entry.obsid
            if obsid not in v_db.keys():
                v_clump_db_entry.obsid = obsid
                v_clump_db_entry.clump_list = []
                v_clump_db_entry.ew_data = v_clump_db[obsid].ew_data # Filtered and normalized
                v_clump_db_entry.et = v_clump_db[obsid].et
                v_clump_db_entry.resolution_min = None 
                v_clump_db_entry.resolution_max = None
                v_clump_db_entry.emission_angle = None
                v_clump_db_entry.incidence_angle = None
                v_clump_db_entry.phase_angle = None
                v_clump_db_entry.et_min = v_clump_db[obsid].et_min
                v_clump_db_entry.et_max = v_clump_db[obsid].et_max
                v_clump_db_entry.et_min_longitude = v_clump_db[obsid].et_min_longitude
                v_clump_db_entry.et_max_longitude = v_clump_db[obsid].et_max_longitude
                v_clump_db_entry.smoothed_ew = None
 
                v_clump_db_entry.clump_list.append(clump)
                v_db[obsid] = v_clump_db_entry
                
            elif obsid in v_db.keys():
                v_db[obsid].clump_list.append(clump) 
    
#    c_clump_db_entry = clumputil.ClumpDBEntry()
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
                
    return (v_db, c_db)
                
def compare_clumps_per_obsid(v_approved_list, c_approved_list, clump_db, v_clump_db):
    
    #take the approved lists and turn them into clump databases with only the valid clumps that we've tracked.
    #note - we do NOT use the full clump db 
    v_clump_db, clump_db = list_to_db(v_approved_list, c_approved_list, clump_db, v_clump_db)
    
    v_clump_numbers = []
    v_clump_scales = []
    v_clump_heights = []
    v_stddevs = []
    for v_obs in v_clump_db.keys():
        clumps = []
        for clump in v_clump_db[v_obs].clump_list:    
            clumps.append(clump)
            v_clump_scales.append(clump.fit_width_deg)
            v_clump_heights.append(clump.norm_fit_height + 1.) #add 1 because  it's the fitted height above the Mean.
        clump_num = len(clumps)
        v_clump_numbers.append(clump_num)
        v_norm_ew_data = v_clump_db[v_obs].ew_data/np.mean(v_clump_db[v_obs].ew_data)
        v_stddevs.append(np.std(v_norm_ew_data))
        
    v_clump_num_stats = (sum(v_clump_numbers)/len(v_clump_numbers), np.std(v_clump_numbers), np.min(v_clump_numbers), np.max(v_clump_numbers))
    v_scale_stats = (sum(v_clump_scales)/len(v_clump_scales), np.std(v_clump_scales), np.min(v_clump_scales), np.max(v_clump_scales))
    v_clump_height_stats = (sum(v_clump_heights)/len(v_clump_heights), np.std(v_clump_heights), np.min(v_clump_heights), np.max(v_clump_heights))
    v_stddev_stats = (sum(v_stddevs)/len(v_stddevs), np.std(v_stddevs), np.min(v_stddevs), np.max(v_stddevs))
    
    c_clump_numbers = []
    c_clump_scales = []
    c_clump_heights = []
    c_stddevs = []
    max_ets = []
    db_by_time = {}
    
    print clump_db.keys()
    for obs in clump_db.keys():
#        
        max_et = clump_db[obs].et_max
#        print max_et
        db_by_time[max_et] = obs
    
#    print db_by_time.keys()
    for et in sorted(db_by_time.keys()):
        obs = db_by_time[et]
        clumps = []
#        print ma.MaskedArray.count(clump_db[obs].ew_data)
#        if (ma.MaskedArray.count(clump_db[obs].ew_data) == len(clump_db[obs].ew_data)):
#        print obs
        for clump in clump_db[obs].clump_list: 
#                print clump.fit_sigma                  
            clumps.append(clump)
            c_clump_scales.append(clump.fit_width_deg)
            c_clump_heights.append(clump.norm_fit_height + 1.)
            
        clump_num = len(clumps)
        c_clump_numbers.append(clump_num)
        c_norm_ew_data = clump_db[obs].ew_data/np.mean(clump_db[obs].ew_data)
        c_stddevs.append(np.std(c_norm_ew_data))
        max_ets.append(clump_db[obs].et_max)
#            plt.plot(clump_db[obs].ew_data)
#            plt.title(obs)
#            plt.show()
    #statistics arrays of 1) mean, 2) stddev, 3) minimum value, 4) max value
    print c_clump_numbers
    c_clump_num_stats = (sum(c_clump_numbers)/len(c_clump_numbers), np.std(c_clump_numbers), np.min(c_clump_numbers), np.max(c_clump_numbers))
    c_scale_stats = (sum(c_clump_scales)/len(c_clump_scales), np.std(c_clump_scales), np.min(c_clump_scales), np.max(c_clump_scales))
    c_clump_height_stats = (sum(c_clump_heights)/len(c_clump_heights), np.std(c_clump_heights), np.min(c_clump_heights), np.max(c_clump_heights))
    c_stddev_stats = (sum(c_stddevs)/len(c_stddevs), np.std(c_stddevs), np.min(c_stddevs), np.max(c_stddevs))
    
    return (v_clump_scales, v_clump_heights, c_clump_scales, c_clump_heights)

def compare_profiles(v_clump_db, c_clump_db):
    
    obsid_list = ['ISS_000RI_SATSRCHAP001_PRIME','ISS_059RF_FMOVIE002_VIMS','ISS_075RF_FMOVIE002_VIMS','ISS_134RI_SPKMVDFHP002_PRIME']
    
#    OBSID = 'ISS_059RF_FMOVIE002_VIMS'
    color_background = (1,1,1)
    figure_size = (14.5, 2.0)
    font_size = 18.0
        
    fig = plt.figure(figsize = figure_size, facecolor = color_background, edgecolor = color_background, linewidth = 5.)
    ax = fig.add_subplot(111)
    plt.subplots_adjust(top = 1.5, bottom = 0.0)
    ax.tick_params(length = 5., width = 2., labelsize = 14. )
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()

    ax.set_ylabel('Relative Equivalent Width', fontsize = font_size)
    ax.set_xlabel('Co-Rotating Longitude ( $\mathbf{^o}$)', fontsize = font_size)
    ax.set_xlim(0,360.)
    ax.get_xaxis().set_ticks([0,90,180,270,360])
    
  
    for obs_id in obsid_list:
        
        c_ew_data = c_clump_db[obs_id].ew_data
        c_ew_data = c_ew_data/np.mean(c_ew_data)
        longitudes = np.arange(0.,360., 360./len(c_ew_data))
        cassini = plt.plot(longitudes, c_ew_data, color ='black' , lw = 2.0, alpha = 0.9)
        
    
    for v_obs in v_clump_db.keys():
        
        v_ew_data = v_clump_db[v_obs].ew_data 
        v_ew_data = v_ew_data/np.mean(v_ew_data)
        v_ew_data = v_ew_data.view(ma.MaskedArray)
        v_mask = ma.getmaskarray(v_ew_data)
        empty = np.where(v_ew_data == 0.)[0]
        if empty != ():
            v_mask[empty[0]-5:empty[-1]+5] = True
        v_ew_data.mask = v_mask
        longitudes = np.arange(0.,360., 360./len(v_ew_data))
        voyager = plt.plot(longitudes, v_ew_data, color = '#AC7CC9', lw = 2.0, alpha = 0.9)
            
    
    leg = ax.legend([voyager[0], cassini[0]], ['Voyager', 'Cassini'], loc = 1)
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
        
    plt.savefig(os.path.join(ROOT,'comparison_profile.png'), dpi = 600, bbox_inches = 'tight', transparent = True)
#        plt.show()
def make_distribution_plots(v_clump_scales, v_clump_heights, c_clump_scales, c_clump_heights, options):
  
    graph_min = np.min(v_clump_scales)
    graph_max = np.max(v_clump_scales)
    step = 2.5
    bins = np.arange(0,graph_max + step,step)
    
    print graph_max
#      
    figure_size = (10., 4.0)
    fig = plt.figure(figsize = figure_size)
    ax = fig.add_subplot(111)
    plt.subplots_adjust(top = .95, bottom = 0.125, left = 0.08, right = 0.98)
    
    ax.set_xlabel('Clump Width ( $\mathbf{^o}$)', fontsize = 18.0)
    ax.set_ylabel('Fraction of Counts', fontsize = 18.0)
    c_scale_weights = np.zeros_like(c_clump_scales) + 1./len(c_clump_scales)
    v_scale_weights = np.zeros_like(v_clump_scales) + 1./len(v_clump_scales)
    
    counts, bins, patches = plt.hist([v_clump_scales, c_clump_scales], bins,
                                      weights = [v_scale_weights, c_scale_weights], label = ['Voyager', 'Cassini'], color = ['#AC7CC9', 'black'], lw = 0.0)
    
    leg =plt.legend()
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
    
    print v_clump_scales
    print counts[0]
    print counts[1]
#    plt.show()
    plt.savefig(os.path.join(ROOT, 'Clump_width_dist.png'), dpi = 600, transparent = True)   
    
    fig2 = plt.figure(figsize = figure_size)
    ax = fig2.add_subplot(111)
    plt.subplots_adjust(top = .95, bottom = 0.125, left = 0.08, right = 0.98)
    ax.set_xlabel('Clump Relative Brightness', fontsize = 18.0)
    ax.set_ylabel('Fraction of Counts', fontsize = 18.0)
    
    graph_min = np.min(v_clump_heights)
    graph_max = np.max(v_clump_heights)
    step = 0.1
    bins = np.arange(0.0,graph_max + step,step)
    ax.set_xlim(1.0, 5.5)
    
    print graph_max

    v_clump_heights = np.array(v_clump_heights)
    c_clump_heights = np.array(c_clump_heights)
    c_height_weights = np.zeros_like(c_clump_heights) + 1./len(c_clump_heights)
    v_height_weights = np.zeros_like(v_clump_heights) + 1./len(v_clump_heights)
    
    counts, bins, patches = plt.hist([v_clump_heights, c_clump_heights], bins,
                                      weights = [v_height_weights, c_height_weights], label = ['Voyager', 'Cassini'], color = ['#AC7CC9', 'black'], lw = 0.0)

#    print counts
    leg =plt.legend()
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
    plt.savefig(os.path.join(ROOT, 'Clump_height_dist.png'), dpi = 600, transparent = True)

'---------------------------------------------------------------------'
'-----------------------------MAIN -----------------------------------'
'---------------------------------------------------------------------'

#load the entire clump database
clump_db_path = os.path.join(ROOT, 'full_clump_database.pickle')
clump_db_fp = open(clump_db_path, 'rb')
clump_find_options = pickle.load(clump_db_fp)
full_clump_db = pickle.load(clump_db_fp)
clump_db_fp.close()

#load the voyager clump database
voyager_clump_db_path = os.path.join(ROOT, 'voyager_clump_database.pickle')
v_clump_db_fp = open(voyager_clump_db_path, 'rb')
clump_find_options = pickle.load(v_clump_db_fp)
v_clump_db = pickle.load(v_clump_db_fp)
v_clump_db_fp.close()

#load downsampled clump database - need for voyager analysis
ds_clump_db_path = os.path.join(ROOT, 'downsampled_clump_database.pickle')
ds_clump_db_fp = open(ds_clump_db_path, 'rb')
clump_find_options = pickle.load(ds_clump_db_fp)
ds_clump_db = pickle.load(ds_clump_db_fp)
ds_clump_db_fp.close()

cp_ds_clump_db_path = os.path.join(ROOT, 'complete_profiles_downsampled_clump_database.pickle')
cp_ds_clump_db_fp = open(cp_ds_clump_db_path, 'rb')
clump_find_options = pickle.load(cp_ds_clump_db_fp)
cp_ds_clump_db = pickle.load(cp_ds_clump_db_fp)
cp_ds_clump_db_fp.close()

c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_clumps_list.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()

ds_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'downsample_approved_clumps_list.pickle')
ds_approved_list_fp = open(ds_approved_list_fp, 'rb')
ds_approved_db, ds_approved_list = pickle.load(ds_approved_list_fp)
ds_approved_list_fp.close()

v_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'voyager_approved_clumps_list.pickle')
v_approved_list_fp = open(v_approved_list_fp, 'rb')
v_approved_db, v_approved_list = pickle.load(v_approved_list_fp)
v_approved_list_fp.close()

mosaic_fp = os.path.join(ROOT, OBSID + '_mosaic_data.npy')
mosaic_png_fp = os.path.join(ROOT, OBSID + '_mosaic.png')
ew_data_fp = os.path.join(ROOT, OBSID + '_ew_data.npy')
ew_mask_fp = os.path.join(ROOT, OBSID + '_ew_mask.npy')
root_clump_db_fp = os.path.join(ROOT, 'clump_database.pickle')

mosaic_metadata_fp = open(os.path.join(ROOT, OBSID + '_mosaic_metadata.pickle'), 'rb')
mosaic_data = pickle.load(mosaic_metadata_fp)
mosaic = np.load(mosaic_fp)
#Have to run the clump finding program first to generate the clump database.

ew_data = np.load(ew_data_fp)
ew_mask = np.load(ew_mask_fp)

long_res = 360./len(ew_data)
longitudes = np.arange(0,360., long_res)

(mosaic_longitudes, mosaic_resolutions, mosaic_image_numbers,
 mosaic_ETs, mosaic_emission_angles, mosaic_incidence_angles,
 mosaic_phase_angles) = mosaic_data
    
mean_emission = np.mean(mosaic_emission_angles[~ew_mask])
mean_phase = np.mean(mosaic_phase_angles[~ew_mask])

ew_data *= ringutil.normalized_ew_factor(mean_phase, mean_emission)
ew_data = ew_data.view(ma.MaskedArray)
ew_data.mask = ew_mask
ew_data = smooth_ew(ew_data, 3.0)


if options.prefilter:
    ew_data = fft_filter(options, None, ew_data, plot = False)
    ew_data = ew_data.real

# We have to wait to add the mask because masked arrays don't survive the FFT filter
ew_data = ew_data.view(ma.MaskedArray)
ew_mask = ma.getmaskarray(ew_data)
ew_data.mask = ew_mask

#SUBROUTINES -------------------------------------------------------------------------------------------------------------------

#edit_mosaic(mosaic)
#select_clumps(options, ew_data, ew_mask, longitudes, OBSID, mosaic_data, ROOT)
#plot_only_profile(longitudes, ew_data)
#plot_clumps(ew_data)
#plot_mexhat(options)
#plot_FDG(options)
#chain_data = find_clump_chain(options, full_clump_db)
#plot_chain_on_ews(chain_data, full_clump_db)

#VOYAGER PLOTS
v_clump_scales, v_clump_heights, c_clump_scales, c_clump_heights = compare_clumps_per_obsid(v_approved_list, ds_approved_list, ds_clump_db, v_clump_db)
make_distribution_plots(v_clump_scales, v_clump_heights, c_clump_scales, c_clump_heights, options)
#compare_profiles(v_clump_db, cp_ds_clump_db)
#make_vel_hist()
#make_voyager_vel_hist()
#combined_vel_hist()
##
#for chain_tuple in chain_data:
#    clump, obsid = chain_tuple
#    (mosaic_data_path, mosaic_metadata_path,
#     mosaic_large_png_path, mosaic_small_png_path) = ringutil.mosaic_paths(options, obsid)
#            
#    mosaic_img = np.load(mosaic_data_path + '.npy')
#    mosaic_data_fp = open(mosaic_metadata_path, 'rb')
#    mosaic_data = pickle.load(mosaic_data_fp)
#    
#    (longitudes, resolutions,
#    image_numbers, ETs, 
#    emission_angles, incidence_angles,
#    phase_angles) = mosaic_data
#
#    im = mosaic_img
#                
#    make_rad_mosaic(im, obsid, longitudes, clump, options)

#END SUBROUTINES -------------------------------------------------------------------------------------------------------------------

if options.save_clump_database:
    clump_options = clumputil.ClumpFindOptions()
    clump_options.type = 'wavelet mexhat'
    clump_options.scale_min = options.scale_min
    clump_options.scale_max = options.scale_max
    clump_options.scale_step = options.scale_step
    clump_options.clump_size_min = options.clump_size_min
    clump_options.clump_size_max = options.clump_size_max
    clump_options.prefilter = options.prefilter
    
    root_clump_db_fp = open(root_clump_db_fp, 'wb')
    pickle.dump(clump_options, root_clump_db_fp)
    pickle.dump(root_clump_db, root_clump_db_fp)
    root_clump_db_fp.close()


