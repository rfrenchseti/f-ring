'''
Created on Jul 11, 2012
Revised to work with all data sets (Regular Cassini, Downsampled, and Voyager) Oct 02, 2012

Detects Clumps
Produces Scalograms
Creates Clump Database

NOTES:
Remember to use a mosaic reduction factor of 1 when Downsampling!
Update-Clump-Database does not work with Voyager data.

Complete IDS:
'ISS_000RI_SATSRCHAP001_PRIME',
'ISS_00ARI_SPKMOVPER001_PRIME',
'ISS_036RF_FMOVIE001_VIMS',
'ISS_039RF_FMOVIE001_VIMS',
'ISS_041RF_FMOVIE002_VIMS',
'ISS_043RF_FMOVIE001_VIMS',
'ISS_044RF_FMOVIE001_VIMS',
'ISS_055RI_LPMRDFMOV001_PRIME',
'ISS_059RF_FMOVIE002_VIMS',
'ISS_075RF_FMOVIE002_VIMS',
'ISS_134RI_SPKMVDFHP002_PRIME',

@author: rfrench
'''

from optparse import OptionParser
import numpy as np
import numpy.ma as ma
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import scipy.optimize as sciopt
import scipy.interpolate as interp
import pickle
import sys
import os.path
import ringutil
import cspice
import matplotlib.pyplot as plt
import cwt
import bandpass_filter
import clumputil
import clump_gaussian_fit

#===============================================================================
#
# Command line processing
# 
#===============================================================================

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = [
                '-a',
#                'ISS_041RF_FMOVIE001_VIMS',
                '--scale-min', '3.5', '--scale-max', '80',
                '--clump-size-min', '3.5', '--clump-size-max', '40.0',

# Voyager
#                '--voyager',

# Cassini normal
#                '--ignore-voyager', '--mosaic-reduction-factor', '2',

# Cassini downsampled
                '--ignore-voyager', '--downsample', '--mosaic-reduction-factor', '1',

                '--replace-clump-database',

#                '--plot-scalogram',
#                '--color-contours',
#                '--prefilter',
#                'ISS_007RI_HPMRDFMOV001_PRIME',
#                'ISS_036RF_FMOVIE001_VIMS',
#                'ISS_036RF_FMOVIE002_VIMS',
#                'ISS_039RF_FMOVIE001_VIMS',
#                'ISS_039RF_FMOVIE002_VIMS',
#                'ISS_031RF_FMOVIE001_VIMS',
#                'ISS_041RF_FMOVIE001_VIMS',
#                'ISS_043RF_FMOVIE001_VIMS',
#                'ISS_044RF_FMOVIE001_VIMS',
#                'ISS_000RI_SATSRCHAP001_PRIME',
#                'ISS_00ARI_SPKMOVPER001_PRIME',
#                'ISS_036RF_FMOVIE001_VIMS',
#                'ISS_039RF_FMOVIE001_VIMS',
#                'ISS_041RF_FMOVIE002_VIMS',
#                'ISS_043RF_FMOVIE001_VIMS',
#                'ISS_044RF_FMOVIE001_VIMS',
#                'ISS_117RI_EQXSHADOW003_PRIME'
#                'ISS_055RI_LPMRDFMOV001_PRIME',
#                'ISS_059RF_FMOVIE002_VIMS',
#                'ISS_089RF_FMOVIE003_PRIME',
#                'ISS_134RI_SPKMVDFHP002_PRIME',
#                'ISS_013RI_AZSCNHIPH003_PRIME',
#                'ISS_085RF_FMOVIE003_PRIME_1'
#                'ISS_007RI_AZSCNLOPH001_PRIME',
#                'ISS_057RF_FMOVIE001_VIMS'    
#                'test_059RF'
#                'V1I'
                ]

#    cmd_line = ['--prefilter',
#                '--scale-min', '3', '--scale-max', '20',
#                '--clump-size-min', '2.1', '--clump-size-max', '49.9',
##                '--ignore-bad-obsids',
##                '--plot-scalogram',
#                '--replace-clump-database',
#                '--voyager'
#                
#                ]

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
parser.add_option('--dont-use-fft', dest='dont_use_fft',
                  action='store_true', default=False,
                  help='Use a non-FFT version of CWT')
parser.add_option('--color-contours', dest='color_contours',
                  action='store_true', default=False,
                  help='Use colored contours in scalogram')
#parser.add_option('--downsample', dest = 'downsample',
#                  action = 'store_true', default = False)

ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

#===============================================================================
#
# Helper functions
# 
#===============================================================================
clump_database = {}

def smooth_ew(ew_range, smooth_deg):
    long_res = 360./len(ew_range)
    smooth_pix = smooth_deg // long_res // 2
    #smooth the equivalent width range 
    smoothed_ew = ma.zeros(ew_range.shape[0])
    
    if ma.any(ew_range.mask) == False:
        #there are no masked elements
        for n in range(len(ew_range)):
            smoothed_ew[n] = ma.mean(ew_range[max(n-smooth_pix,0):
                                                             min(n+smooth_pix+1,len(ew_range)-1)])
    else:
        for n in range(len(ew_range)):
                        if ew_range.mask[n]:
                            smoothed_ew[n] = ma.masked
                        else:
                            smoothed_ew[n] = ma.mean(ew_range[max(n-smooth_pix,0):
                                                             min(n+smooth_pix+1,len(ew_range)-1)])
        
    return smoothed_ew

def downsample_ew(ew_range, mosaic_data):
    src_mask = ma.getmaskarray(ew_range)
    (mosaic_longitudes, mosaic_resolutions, mosaic_image_numbers,
     mosaic_ETs, mosaic_emission_angles, mosaic_incidence_angles,
     mosaic_phase_angles) = mosaic_data
    mosaic_longitudes = mosaic_longitudes.view(ma.MaskedArray)
    mosaic_longitudes.mask = src_mask
    mosaic_resolutions = mosaic_resolutions.view(ma.MaskedArray)
    mosaic_resolutions.mask = src_mask
    mosaic_image_numbers = mosaic_image_numbers.view(ma.MaskedArray)
    mosaic_image_numbers.mask = src_mask
    mosaic_ETs = mosaic_ETs.view(ma.MaskedArray)
    mosaic_ETs.mask = src_mask
    mosaic_emission_angles = mosaic_emission_angles.view(ma.MaskedArray)
    mosaic_emission_angles.mask = src_mask
    mosaic_incidence_angles = mosaic_incidence_angles.view(ma.MaskedArray)
    mosaic_incidence_angles.mask = src_mask
    mosaic_phase_angles = mosaic_phase_angles.view(ma.MaskedArray)
    mosaic_phase_angles.mask = src_mask

    bin = int(len(ew_range)/720.)
    new_len = int(len(ew_range)/bin)   # Make sure it's an integer size - this had better be 720!
    assert new_len == 720
    new_arr = ma.zeros(new_len)  # ma.zeros automatically makes a masked array
    src_mask = ma.getmaskarray(ew_range)
    for i in range(new_len):
        # Point i averages the semi-open interval [i*bin, (i+1)*bin)
        # ma.mean does the right thing - if all the values are masked, the result is also masked
        new_arr[i] = ma.mean(ew_range[i*bin:(i+1)*bin])
        mosaic_longitudes[i] = ma.min(mosaic_longitudes[i*bin:(i+1)*bin])
        mosaic_resolutions[i] = ma.max(mosaic_resolutions[i*bin:(i+1)*bin])
        mosaic_image_numbers[i] = ma.min(mosaic_image_numbers[i*bin:(i+1)*bin])
        mosaic_ETs[i] = ma.min(mosaic_ETs[i*bin:(i+1)*bin])
        mosaic_emission_angles[i] = ma.min(mosaic_emission_angles[i*bin:(i+1)*bin])
        mosaic_incidence_angles[i] = ma.min(mosaic_incidence_angles[i*bin:(i+1)*bin])
        mosaic_phase_angles[i] = ma.min(mosaic_phase_angles[i*bin:(i+1)*bin])
    mosaic_data = (mosaic_longitudes, mosaic_resolutions, mosaic_image_numbers,
     mosaic_ETs, mosaic_emission_angles, mosaic_incidence_angles,
     mosaic_phase_angles)
    return new_arr, mosaic_data

def interpolate(arr):
    
    x = np.arange(0, len(arr))
    step = (len(arr)-1)/18000.
    x_int = np.arange(0, len(arr)-1., step)
    if len(x_int) > 18000.:
        x_int = x_int[0:18000]
    f_int = interp.interp1d(x, arr)
    
    return f_int(x_int)         #ew_range with 1000 points

#def divide_ews_by_baseline(ew_data):
#    ew_act_data = []
#    for i in range(len(ew_data)):
#        if ma.any(ew_data.mask) == False:
#            ew_act_data.append(ew_data[i])
#        else:
#            if ew_data.mask[i]:
#                continue
#            else:
#                ew_act_data.append(ew_data[i])
#                
#    sorted_ew_data = ma.sort(ew_act_data)
#    percent_cutoff = 0.3
#    cutoff_index = percent_cutoff*len(ew_act_data)
#    
#    lower_ew_data = sorted_ew_data[0:cutoff_index +1]
#    median_baseline = ma.median(lower_ew_data)
#    new_ew_data = ew_data/median_baseline
#    
#    return(new_ew_data)

def adjust_ew_for_zero_phase(ew_data, phase_angles, emission_angles, incidence_angles):
    new_ew_data = ringutil.compute_corrected_ew(ew_data * ringutil.compute_mu(emission_angles),
                                                 emission_angles, np.mean(incidence_angles))
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    plt.plot(ew_data* clumputil.compute_mu(emission_angles), color='black')
#    plt.plot(new_ew_data, color='red')
    for i in range(len(ew_data)):
        ratio = ringutil.clump_phase_curve(0) / ringutil.clump_phase_curve(phase_angles[i])
#        print phase_angles[i], ratio
        new_ew_data[i] *= ratio
#    plt.plot(new_ew_data, color='green')
#    plt.show()
    return new_ew_data
    

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


def find_clumps_internal(options, ew_data, longitudes, obs_id, metadata, wavelet_type = 'SDG'):
    
    ew_mask = ma.getmaskarray(ew_data)
    
    (mosaic_longitudes, mosaic_resolutions, mosaic_image_numbers,
         mosaic_ETs, mosaic_emission_angles, mosaic_incidence_angles,
         mosaic_phase_angles) = metadata


    if options.downsample:
#        ew_data = downsample_ew(ew_data)
        long_res = 360./len(ew_data)
        longitudes = np.arange(len(ew_data))*long_res
        tripled_ew_data = np.tile(ew_data, 3)
        tripled_ew_mask = ma.getmaskarray(tripled_ew_data)
        tripled_ew_data.mask = tripled_ew_mask
    else: 
        tripled_ew_data = np.tile(ew_data, 3)   
        long_res = 360./len(ew_data)
   
    # We have to triple the data or else the edges get screwed up
        
    orig_ew_start_idx = len(ew_data)
    orig_ew_end_idx = len(ew_data)*2
    orig_ew_start_long = 360.
    orig_ew_end_long = 720.
    
    clump_db_entry = clumputil.ClumpDBEntry()
    clump_db_entry.obsid = obs_id
    
    clump_db_entry.ew_data = ew_data # Smoothed and normalized
   
    if options.voyager:
        clump_db_entry.et = mosaic_ETs[0]
        clump_db_entry.resolution_min = None 
        clump_db_entry.resolution_max = None
        clump_db_entry.emission_angle = None
        clump_db_entry.incidence_angle = None
        clump_db_entry.phase_angle = None
        clump_db_entry.et_min = mosaic_ETs[0]
        clump_db_entry.et_max = mosaic_ETs[0]
        clump_db_entry.et_min_longitude = longitudes[0]
        clump_db_entry.et_max_longitude = longitudes[-1]
        clump_db_entry.smoothed_ew = None
    else:

        clump_db_entry.et = np.mean(mosaic_ETs[~ew_mask])
        clump_db_entry.resolution_min = np.min(mosaic_resolutions[~ew_mask])
        clump_db_entry.resolution_max = np.max(mosaic_resolutions[~ew_mask])
        clump_db_entry.emission_angle = np.mean(mosaic_emission_angles[~ew_mask])
        clump_db_entry.incidence_angle = np.mean(mosaic_incidence_angles[~ew_mask])
        clump_db_entry.phase_angle = np.mean(mosaic_phase_angles[~ew_mask])
        print 'INCIDENCE', clump_db_entry.incidence_angle
        
        min_et = 1e38
        max_et = 0
        for idx in range(len(mosaic_longitudes)):
            if mosaic_longitudes[idx] >= 0:
                if mosaic_ETs[idx] < min_et:
                    min_et = mosaic_ETs[idx]
                    min_et_long = mosaic_longitudes[idx]
                if mosaic_ETs[idx] > max_et:
                    max_et = mosaic_ETs[idx]
                    max_et_long = mosaic_longitudes[idx]
                else:
                    max_et_long = 0.0
                    min_et_long = 0.0
                    
        clump_db_entry.et_min = min_et
        clump_db_entry.et_max = max_et
        clump_db_entry.et_min_longitude = min_et_long
        clump_db_entry.et_max_longitude = max_et_long

    # Create the scales at which we wish to do the analysis
    # The scale is the half-width of the wavelet
    # The user gives us the scale in degrees, but we need to convert it to indices
    scale_min = options.scale_min/2.
    scale_max = options.scale_max/2.
    scale_step = options.scale_step/2.
    wavelet_scales = np.arange(scale_min/long_res, scale_max/long_res+0.0001, scale_step/long_res)
#    print wavelet_scales
    # Initialize the mother wavelet
    print 'STARTING CWT PROCESS:' + wavelet_type
    if wavelet_type == 'SDG':
        mother_wavelet = cwt.SDG(len_signal=len(tripled_ew_data), scales=wavelet_scales)
    if wavelet_type == 'FDG': 
        mother_wavelet = cwt.FDG(len_signal=len(tripled_ew_data), scales=wavelet_scales)
    # Perform the continuous wavelet transform
    if options.dont_use_fft:
        wavelet = cwt.cwt_nonfft(tripled_ew_data, mother_wavelet, # XXX
                                 startx=len(ew_data)+205/long_res,
                                 endx=len(ew_data)+215/long_res)
    else:
        wavelet = cwt.cwt(tripled_ew_data, mother_wavelet)

    # Find the clumps
    # First find the local maxima
    tripled_xform = wavelet.coefs.real
#    print tripled_xform
#    print tripled_xform
    xform_maxima_scale_idx, xform_maxima_long_idx = detect_local_maxima(tripled_xform) # Includes tripled info
    
#    print xform_maxima_scale_idx, xform_maxima_long_idx 
    def fit_wavelet_func(params, wavelet, data):
        base, wscale = params
        return np.sqrt(np.sum((wavelet*wscale+base-data)**2))
    
    # For each found clump, fit the wavelet to it and see if it's OK
    
    clump_list = []
    
    print 'STARTING CLUMP FITTING/WAVELET PROCESS'
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
            if options.voyager == False:
                if tripled_ew_data.mask[maximum_long_idx]:
                    # The clump is centered on masked data - don't trust it
                    continue
            long_start_idx = round(long_start_deg/long_res)
            long_end_idx = round(long_end_deg/long_res)
            if (tripled_ew_data.mask[long_start_idx - (wavelet_scales[maximum_scale_idx]/8.)] == True) or (tripled_ew_data.mask[long_end_idx + (wavelet_scales[maximum_scale_idx]/8.)] == True):
                #we don't want clumps on the edges of the masked area
                continue 
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
#            print 'CLUMP LONG %6.2f WIDTH %5.2f BASE %7.4f HEIGHT %7.4f' % (longitude, scale,
#                                                                           mexhat_base, residual)
#                
            clump = clumputil.ClumpData()
            clump.clump_db_entry = clump_db_entry
            clump.longitude = longitude
            clump.scale = scale
            clump.longitude_idx = maximum_long_idx-orig_ew_start_idx
            clump.scale_idx = wavelet_scales[maximum_scale_idx]*2.
            clump.mexhat_base = mexhat_base
            clump.mexhat_height = mexhat_height
            clump.abs_height = clump.mexhat_height*mexhat[len(mexhat)//2]
            clump.max_long = maximum_long_idx
            clump.matched = False
            clump.residual = residual * long_res
            clump.wave_type = wavelet_type
#
           #calculate refined fit
            clump.fit_left_deg, clump.fit_right_deg, clump.fit_width_idx, clump.fit_width_deg, clump.fit_height, clump.int_fit_height, clump.g_center, clump.g_sigma, clump.g_base, clump.g_height = clump_gaussian_fit.refine_fit(clump, ew_data, options.voyager, options.downsample)

            fit_width = (clump.fit_right_deg - clump.fit_left_deg) % 360
            if fit_width < options.clump_size_min or fit_width > options.clump_size_max:
                print 'WIDTH OUT OF RANGE - IGNORING', fit_width
                continue

            print 'CLUMP LEFT', clump.fit_left_deg, 'RIGHT', clump.fit_right_deg, 'CENTER', clump.g_center,
            print 'SIGMA', clump.g_sigma
            if abs(clump.fit_left_deg-clump.g_center) <= .5 or abs(clump.fit_right_deg-clump.g_center) <= 0.5:
                ratio = 100
            else:
                ratio = ((clump.fit_right_deg-clump.g_center)%360) / ((clump.g_center-clump.fit_left_deg)%360)
            if ratio < 1: ratio = 1/ratio
            if ratio > 4: 
                print 'EXCESSIVE ASYMMETRY - IGNORING'
                continue
            
            found_dup = False
            for old_clump in clump_list:
                if (old_clump.fit_left_deg == clump.fit_left_deg and
                    old_clump.fit_right_deg == clump.fit_right_deg):
                    found_dup = True
                    break
            if found_dup:
                print 'DUPLICATE - IGNORING'
                continue
            
            #calculate clump sigma height
            profile_sigma = ma.std(ew_data)
#            height = clump.mexhat_height*mexhat[len(mexhat)//2]
            clump.clump_sigma = clump.abs_height/profile_sigma
            clump.fit_sigma = clump.fit_height/profile_sigma
            
#            clump.print_all()
            
#            print vars(clump)
            clump_list.append(clump)

    clump_db_entry.clump_list = clump_list
            
    wavelet_data = (wavelet, mother_wavelet, wavelet_scales)
    print 'FINISHED SAVING DATA'
    return (clump_db_entry, wavelet_data)

def select_clumps(options, ew_data, longitudes, obs_id, metadata):
    print 'STARTING SDG'
    sdg_clump_db_entry, sdg_wavelet_data = find_clumps_internal(options, ew_data, longitudes, obs_id, metadata, wavelet_type = 'SDG')
    print 'STARTING FDG'
    fdg_clump_db_entry, fdg_wavelet_data = find_clumps_internal(options, ew_data, longitudes, obs_id, metadata, wavelet_type = 'FDG')
    
    sdg_list = sdg_clump_db_entry.clump_list
    fdg_list = fdg_clump_db_entry.clump_list
    
    print 'CHOOSE THE BEST WAVELETS'
    new_clump_list = []
    longitude_list = []
    for n, clump_s in enumerate(sdg_list):
        f_match_list = []
        s_match_list = [clump_s]
#        print 'new clump_s', clump_s.longitude, clump_s.scale
        for clump_f in fdg_list:
#            print 'new_clump_f', clump_f.longitude, clump_f.scale
            if (clump_s.longitude - 2.0 <= clump_f.longitude <= clump_s.longitude + 2.0) and (0.5*clump_s.scale <= clump_f.scale <= 2.5*clump_s.scale):
                #there is a match!
                f_match_list.append(clump_f)
                clump_f.matched = True
                clump_s.matched = True
#                print 'match'
        
#        for clump_s2 in sdg_list[n+1::]:
#            if (clump_s.longitude - 2.0 <= clump_s2.longitude <= clump_s.longitude + 2.0) and (0.5*clump_s.scale <= clump_s2.scale <= 2.0*clump_s.scale):
#                clump_s2.matched = True
#                s_match_list.append(clump_s2)
        if clump_s.matched == False:
            continue      
        f_res = 999999.
        s_res = 999999.
        for f_clump in f_match_list:
            if f_clump.residual < f_res:
                f_res = f_clump.residual
                best_f = f_clump
        for s_clump in s_match_list:
            if s_clump.residual < s_res:
                s_res = s_clump.residual
                best_s = s_clump
            
        if best_f.residual < best_s.residual:
            new_clump_list.append(best_f)
            longitude_list.append(best_f.longitude)
        else:
            new_clump_list.append(best_s)
            longitude_list.append(best_s.longitude)

            
                
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
#            if clump_f.residual >= 1.0:
#            print 'F RESIDUAL TOO BIG', clump_f.longitude, clump_f.scale, clump_f.residual
#                continue
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

    new_clump_list.sort(key=lambda x:x.g_center)
    
    for clump in new_clump_list:
        print 'FINAL CLUMP GS %6.2f FLD %6.2f FRD %6.2f' % (clump.g_center, clump.fit_left_deg, clump.fit_right_deg)
    
    #it doesn't matter which clump_db_entry we choose since the metadata is the same for both. The only differences are the clumps.
    fdg_clump_db_entry.clump_list = new_clump_list
    for clump in new_clump_list:
        clump.clump_db_entry = fdg_clump_db_entry
    clump_database[obs_id] = fdg_clump_db_entry
    
#    print fdg_clump_db_entry.et                                      # The mean ephemeris time (ET) (sec)
#    print fdg_clump_db_entry.et_min                                  # The minimum ET in the mosaic (sec)
#    print fdg_clump_db_entry.et_max                                  # The maximum ET in the mosaic (sec)
#    print fdg_clump_db_entry.et_min_longitude                        # The longitude where the minimum ET occurs
#    print fdg_clump_db_entry.et_max_longitude                        # The longitude where the maximum ET occurs
#    print fdg_clump_db_entry.resolution_min                          # The minimum radial resolution in the mosaic (km/pix)
#    print fdg_clump_db_entry.resolution_max                          # The maximum radial resolution in the mosaic (km/pix)
#    print fdg_clump_db_entry.emission_angle                          # The mean emission angle in the mosaic (deg)
#    print fdg_clump_db_entry.incidence_angle                         # The mean incidence angle in the mosaic (deg)
#    print fdg_clump_db_entry.phase_angle                             # The mean phase angle in the mosaic (deg)
#    print fdg_clump_db_entry.ew_data.mask                                 # Masked array of EWs normalized and possibly filtered
#    print fdg_clump_db_entry.smoothed_ew # The list of clumps in this OBSID

    return (sdg_wavelet_data, fdg_wavelet_data)
    
#    print vars(clump_db_entry)
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
        
#        print orig_ew_start_idx, orig_ew_end_idx
        scales_axis = wavelet_scales * long_res * 2 # Full-width degrees for plotting

        fig = plt.figure()
    
        # Make the contour plot of the wavelet transform
        ax1 = fig.add_subplot(211)
        if options.color_contours:
            ax1.contourf(longitudes, scales_axis, xform, 256)
        else:
#            print len(longitudes), len(scales_axis), xform
            ct = ax1.contour(longitudes, scales_axis, xform, 32)
            ax1.clabel(ct, inline=1, fontsize=8)
        ax1.set_ylim((scales_axis[0], scales_axis[-1]))
        ax1.set_xlim((longitudes[0], longitudes[-1]))
        ax1.set_title( wavelet_form + ' Scalogram '+obs_id)
        ax1.set_ylabel('Full width scale (deg)')
    
        # Make the data plot with overlaid detected clumps
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.plot(longitudes, ew_data)

        tripled_longitudes = np.append(longitudes-360., np.append(longitudes, longitudes+360.))
                
        for clump_data in clump_list:
#            print clump_data.wave_type, clump_data.longitude
            if clump_data.wave_type == 'SDG':
                wavelet_color = 'red'
            if clump_data.wave_type == 'FDG':
#                print 'plotting'
                wavelet_color = 'green'
            clumputil.plot_single_clump(ax2, ew_data, clump_data, 0., 360., color = wavelet_color)
#            plot_scale_idx = clump_data.scale_idx*5
#            maximum_scale_idx = np.where(wavelet_scales*2. == clump_data.scale_idx)[0][0]
#            mexhat = mother_wavelet.coefs[maximum_scale_idx].real # Get the master wavelet shape
#            mh_start_idx = round(len(mexhat)/2.-plot_scale_idx/2.)
#            mh_end_idx = round(len(mexhat)/2.+plot_scale_idx/2.)
#            mexhat = mexhat[mh_start_idx:mh_end_idx+1] # Extract just the positive part
#            longitude_idx = clump_data.longitude_idx
#            if longitude_idx+plot_scale_idx/2 >= len(longitudes): # Runs off right side - make it run off left side instead
#                longitude_idx -= len(longitudes)
#            idx_range = tripled_longitudes[longitude_idx-plot_scale_idx/2+len(longitudes):
#                                           longitude_idx-plot_scale_idx/2+len(longitudes)+len(mexhat)] # Longitude range in data
#            if len(idx_range) != len(mexhat*clump_data.mexhat_height+clump_data.mexhat_base):
#                    idx_range = idx_range[:-1]
#            ax2.plot(idx_range, mexhat*clump_data.mexhat_height+clump_data.mexhat_base, '-',
#                     color='red', lw=3, alpha=0.8)
#            if longitude_idx-plot_scale_idx/2 < 0: # Runs off left side - plot it twice
#                ax2.plot(idx_range+360, mexhat*clump_data.mexhat_height+clump_data.mexhat_base, '-',
#                         color='red', lw=3, alpha=0.8)
                
        ax2.set_xlim((longitudes[0], longitudes[-1]))
        ax2.set_ylim(ma.min(ew_data), ma.max(ew_data))
        ax2.grid()
        # align time series fig with scalogram fig
        t = ax2.get_position()
        ax2pos = t.get_points()
        ax2pos[1][0] = ax1.get_position().get_points()[1][0]
        t.set_points(ax2pos)
        ax2.set_position(t)
        ax2.set_xlabel('Longitude (deg)')
    
    plt.show()          #show both at once

#===============================================================================
#
# MAIN LOOP 
#
#===============================================================================

# Compute longitude step of EW data

if options.voyager:
    
    for root, dirs, files in os.walk(ringutil.VOYAGER_PATH):
    
        for file in files:
            if '.STACK' in file:
                filename = file[:-6]
                (ew_data_path, ew_mask_path) = ringutil.ew_paths(options, filename)
                
                ew_data = np.load(ew_data_path +'.npy')
                data_path, metadata_path, large_png_path, small_png_path = ringutil.mosaic_paths(options, filename)
                
                metadata_fp = open(metadata_path, 'rb')
                metadata = pickle.load(metadata_fp)
                metadata_fp.close()
#           
                (longitudes, resolutions, image_numbers,
                 ETs, emission_angles, incidence_angles,
                 phase_angles) = metadata
#               
                print 'VOYAGER FILE', filename, '#LONG', len(longitudes)

                if options.prefilter:
                    ew_data = bandpass_filter.fft_filter(options, ew_data, plot = True)
                    ew_data = ew_data.real
                    assert False
                
                ew_data = ew_data.view(ma.MaskedArray)
                ew_data.mask = False
                ew_data = adjust_ew_for_zero_phase(ew_data, phase_angles, emission_angles, incidence_angles)
                
                if filename[:2] == 'V1':
                    ew_data *= 1.34
                elif filename[:2] == 'V2':
                    ew_data *= 0.97
                else:
                    assert False
                    
                ew_data = smooth_ew(ew_data, 3.0)
#                 ew_data = ew_data/np.mean(ew_data)
                ew_data = ew_data.view(ma.MaskedArray)
                ew_data.mask = False
                (sdg_wavelet, fdg_wavelet) = select_clumps(options, ew_data, longitudes, filename, metadata)
                
                if options.plot_scalogram:
                    plot_scalogram(sdg_wavelet, fdg_wavelet, clump_database[filename] )
    
else:
    if options.update_clump_database:
        clump_database_path, clump_chains_path = ringutil.clumpdb_paths(options)
        clump_database_fp = open(clump_database_path, 'rb')
        clump_database = pickle.load(clump_database_fp)
        clump_database_fp.close()
    
    for obs_id, image_name, full_path in ringutil.enumerate_files(options, args, obsid_only=True):
    
        long_res = options.longitude_resolution*options.mosaic_reduction_factor
    
        (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
         bkgnd_mask_filename, bkgnd_model_filename,
         bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obs_id)
         
        print reduced_mosaic_data_filename
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
    
        print 'Processing', obs_id, cspice.et2utc(np.max(mosaic_ETs), 'C', 0)
    #    cmd_line = ['--write-pdf']
        ew_data = np.load(ew_data_filename+'.npy')
        ew_mask = np.load(ew_mask_filename+'.npy')
        temp_ew_data = ew_data.view(ma.MaskedArray)
        temp_ew_data.mask = ew_mask
        ew_mask = ma.getmaskarray(temp_ew_data)
                
        mean_emission = np.mean(mosaic_emission_angles[~ew_mask])
        mean_phase = np.mean(mosaic_phase_angles[~ew_mask])

        ew_data = adjust_ew_for_zero_phase(ew_data, mosaic_phase_angles, mosaic_emission_angles, mosaic_incidence_angles)
    
        print 'ORIG #LONG', len(mosaic_longitudes), 'LONGRES', long_res
        
#        ew_data *= ringutil.normalized_ew_factor(np.array(mosaic_phase_angles), np.array(mosaic_emission_angles), np.array(mosaic_incidence_angles))
        
#         if obs_id in monster_list:
#             ew_data_red = ew_data[0:200/long_res]
#             if ma.any(ew_mask):
#                 ew_mask_red = ew_mask[0:200./long_res]
#             else: 
#                 ew_mask_red = False
#             ew_mean = np.mean(ew_data_red[~ew_mask_red])
#             
#         else:
#             ew_mean = np.mean(ew_data[~ew_mask])
#         
#         ew_data /= ew_mean
        if options.prefilter:
            ew_data = bandpass_filter.fft_filter(options,ew_data, plot = True)
            ew_data = ew_data.real
        
        # We have to wait to add the mask because masked arrays don't survive the FFT filter
        #Interpolate the data so that it has a longitude resolution of 0.02
        if options.downsample:
            if len(ew_data) != 18000:
                print 'Warning: Data not at 0.02 resolution - interpolating!'
                full_ew_data = interpolate(ew_data)
                #expand mask
                big_ew_mask = ma.getmaskarray(full_ew_data)
#                
                #check the original mask to see if it's blank. If so, then just make a blank mask array.
                #If not, expand it accordingly.
                if ma.any(ew_mask) == False:
                    full_ew_data = full_ew_data.view(ma.MaskedArray)
                    full_ew_data.mask = big_ew_mask
#                    print full_ew_data.mask
                else:
                    for i, val in enumerate(ew_mask):
                        k = i*2
                        big_ew_mask[k] = val
                        big_ew_mask[k+1] = val
                        
                    full_ew_data = full_ew_data.view(ma.MaskedArray)
                    full_ew_data.mask = big_ew_mask
            else:
                full_ew_data = ew_data.view(ma.MaskedArray)
                full_ew_data.mask = ew_mask
                
            ew_data, mosaic_data = downsample_ew(full_ew_data, mosaic_data)
            ew_mask = ma.getmaskarray(ew_data)

        else:
            ew_data = ew_data.view(ma.MaskedArray)
            ew_data.mask = ew_mask
        
#        if options.downsample:
#            ew_data = downsample_ew(ew_data)

        smoothed_ew = smooth_ew(ew_data, 3.0)
        smoothed_ew = smoothed_ew.view(ma.MaskedArray)
        smoothed_ew.mask = ma.getmaskarray(smoothed_ew)
        
        long_res = 360./len(ew_data)
        longitudes = np.arange(len(ew_data)) * long_res  
        print 'NEW  #LONG', len(longitudes), 'LONGRES', long_res

##        
#        plt.plot(longitudes, smoothed_ew)  
#        plt.show()
#
#        print smoothed_ew.mask

        
        if len(mosaic_resolutions[~ew_mask]) == 0:
            print 'MASK IS EMPTY'
        else:
            print 'Min res', np.min(mosaic_resolutions[~ew_mask]),
            print 'Max res', np.max(mosaic_resolutions[~ew_mask])
            print 'Min E', np.min(mosaic_emission_angles[~ew_mask]),
            print 'Max E', np.max(mosaic_emission_angles[~ew_mask])
            print 'Min I', np.min(mosaic_incidence_angles[~ew_mask]),
            print 'Max I', np.max(mosaic_incidence_angles[~ew_mask])
            print 'Min P', np.min(mosaic_phase_angles[~ew_mask]),
            print 'Max P', np.max(mosaic_phase_angles[~ew_mask])
#        
#        #if the data set is complete and has nothing in the original mask array, it ends up being stripped during the smoothing process.
#        #this puts a false mask back onto the data
#        if not options.downsample:
#            smoothed_ew = smoothed_ew.view(ma.MaskedArray)
#            smoothed_ew.mask = ew_mask
#        
#        plt.plot(longitudes, smoothed_ew)  
#        plt.show()
#        print smoothed_ew.mask
        sdg_wavelet, fdg_wavelet = select_clumps(options, smoothed_ew, longitudes, obs_id, mosaic_data)
        
        if options.plot_scalogram:
            plot_scalogram(sdg_wavelet, fdg_wavelet, clump_database[obs_id])
#        find_clumps(options, ew_data, smoothed_ew, longitudes, obs_id)

if options.update_clump_database or options.replace_clump_database: 
    if  options.voyager:
        print 'save to voyager clump database'
        clump_options = clumputil.ClumpFindOptions()
        clump_options.type = 'wavelet mexhat'
        clump_options.scale_min = options.scale_min
        clump_options.scale_max = options.scale_max
        clump_options.scale_step = options.scale_step
        clump_options.clump_size_min = options.clump_size_min
        clump_options.clump_size_max = options.clump_size_max
        clump_options.prefilter = options.prefilter
        
        clump_database_path, clump_chains_path = ringutil.clumpdb_paths(options)
        clump_database_fp = open(clump_database_path, 'wb')
        pickle.dump(clump_options, clump_database_fp)
        pickle.dump(clump_database, clump_database_fp)
        clump_database_fp.close()
    
    elif options.downsample:
        downsampled_clump_database_fp = os.path.join(ringutil.VOYAGER_PATH, 'downsampled_clump_database.pickle')
        print 'saving to downsampled_clump_database'
#        for obsid in sorted(clump_database):
#            clump_database[obsid].print_all()
#            print '-'*80
        clump_options = clumputil.ClumpFindOptions()
        clump_options.type = 'wavelet mexhat'
        clump_options.scale_min = options.scale_min
        clump_options.scale_max = options.scale_max
        clump_options.scale_step = options.scale_step
        clump_options.clump_size_min = options.clump_size_min
        clump_options.clump_size_max = options.clump_size_max
        clump_options.prefilter = options.prefilter
        clump_database_path, clump_chains_path = ringutil.clumpdb_paths(options)
        downsampled_clump_database_fp = open(downsampled_clump_database_fp, 'wb')
        
        clump_database_path, clump_chains_path = ringutil.clumpdb_paths(options)
        clump_database_fp = open(clump_database_path, 'wb')
        pickle.dump(clump_options, clump_database_fp)
        pickle.dump(clump_database, clump_database_fp)
        clump_database_fp.close()
        
        pickle.dump(clump_options, downsampled_clump_database_fp)
        pickle.dump(clump_database, downsampled_clump_database_fp)
        downsampled_clump_database_fp.close()    
    else:
        print 'saving to regular clump database'
        clump_options = clumputil.ClumpFindOptions()
        clump_options.type = 'wavelet mexhat'
        clump_options.scale_min = options.scale_min
        clump_options.scale_max = options.scale_max
        clump_options.scale_step = options.scale_step
        clump_options.clump_size_min = options.clump_size_min
        clump_options.clump_size_max = options.clump_size_max
        clump_options.prefilter = options.prefilter
        clump_database_path, clump_chains_path = ringutil.clumpdb_paths(options)
        clump_database_fp = open(clump_database_path, 'wb')
        
        pickle.dump(clump_options, clump_database_fp)
#        print vars(clump_database['ISS_041RF_FMOVIE001_VIMS'])
        print clump_database_fp
        pickle.dump(clump_database, clump_database_fp)
        clump_database_fp.close()
