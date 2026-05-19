'''
Created on Nov 22, 2011

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
import colorsys
import cspice
import ringimage

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['ISS_111RF_FMOVIE002_PRIME', '--display-one-at-a-time', '--show-prometheus-position']
#    cmd_line = ['ISS_029RF_FMOVIE001_VIMS', '--display-one-at-a-time', '--show-prometheus-position', '--smooth-gaussian']
#    cmd_line = ['ISS_029RF_FMOVIE001_VIMS', '--display-one-at-a-time', '--show-prometheus-position', '--smooth-gaussian', '--subtract-smoothed',
#                '--show-prometheus-waves']
#    cmd_line = ['ISS_029RF_FMOVIE001_VIMS', '--display-one-at-a-time', '--show-prometheus-position', '--smooth-gaussian', '--subtract-smoothed',
#                '--show-prometheus-waves']
#    cmd_line = ['-a', #'--display-one-at-a-time', '--show-prometheus-position',
#                '--subtract-smoothed', '--subtract-smoothed-degree', '15',
##                '--show-prometheus-waves', '--show-stddev',
#                '--smooth-median', '--smooth-median-degree', '2.0',
##                '--smooth-gaussian', '--smooth-gaussian-degree', '1.75',
##                '--find-clumps', '--find-clumps-stddev', '5.0'
#                ]
#    cmd_line = ['-a', '--subtract-smoothed', '--subtract-smoothed-degree', '15', 
#                '--smooth-gaussian', 'smooth-gaussian-degree', '1.75',
#                '--find-clumps', '--find-clumps-stddev', '2',
#                '--scale-by-time']
#    cmd_line = ['ISS_134RI_SPKMVDFHP002_PRIME', 'ISS_036RF_FMOVIE002_VIMS',
#                '--display-one-at-a-time']
#    cmd_line = ['-a', '--clump-smooth',
#                '--scale-by-time',
#                '--ignore-voyager']
parser = OptionParser()

parser.add_option('--display-one-at-a-time', dest='display_one_at_a_time',
                  action='store_true', default=False,
                  help="Display each az profile one at a time instead of stacked in a single graph")
parser.add_option('--show-prometheus-position', dest='show_prometheus_position',
                  action='store_true', default=False,
                  help="Show the point of closest approach for Prometheus")
parser.add_option('--show-prometheus-waves', dest='show_prometheus_waves',
                  action='store_true', default=False,
                  help="Show the N degree wave positions for Prometheus")
parser.add_option('--prometheus-wavelength',
                  type='float', dest='prometheus_wavelength', default=3.25,
                  help='The wavelength in degrees of Prometheus perturbations')
parser.add_option('--scale-by-time', dest='scale_by_time',
                  action='store_true', default=False,
                  help="Stack the az profiles spaced by time")
parser.add_option('--smooth-gaussian', dest='smooth_gaussian',
                  action='store_true', default=False,
                  help="Smooth the az profile using a Gaussian window")
parser.add_option('--smooth-gaussian-degree',
                  type='float', dest='smooth_gaussian_degree', default=3.25,
                  help='The degree in degrees of the Gaussian window (1/2 window size)')
parser.add_option('--smooth-median', dest='smooth_median',
                  action='store_true', default=False,
                  help="Smooth the az profile using a median filter")
parser.add_option('--smooth-median-degree',
                  type='float', dest='smooth_median_degree', default=3.25,
                  help='The degree in degrees of the median window (1/2 window size)')
parser.add_option('--subtract-smoothed', dest='subtract_smoothed',
                  action='store_true', default=False,
                  help="Subtract a smoothed profile from the original profile to normalize to zero")
parser.add_option('--subtract-smoothed-degree',
                  type='float', dest='subtract_smoothed_degree', default=7.5,
                  help='The degree in degrees of the smoothing window to be subtracted (1/2 window size)')
parser.add_option('--show-stddev', dest='show_stddev',
                  action='store_true', default=False,
                  help="Show vertical bars of standard deviation")
parser.add_option('--find-clumps', dest='find_clumps',
                  action='store_true', default=False,
                  help="Find clumps")
parser.add_option('--find-clumps-stddev',
                  type='float', dest='find_clumps_stddev', default=1.5,
                  help='The number of stddev above the mean a clump has to be')
parser.add_option('--clump-smooth',
                  dest = 'clump_smooth', action = 'store_true', default = False)



ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

def normalized_phase_curve(alpha):
    return np.exp((7.68894494e-07*(alpha**3)-6.05914982e-05*(alpha**2)+6.62353025e-03*alpha-8.33855150e-01))

def smooth_list_gaussian(data, degree=5):
    datamask = ma.getmaskarray(data)  
    origlen = len(data)
    window = degree*2-1
    # np.append doesn't work property on masked arrays  
    newdata = np.append(data.data, data.data[:window]) # Make it circular
    datamask = ma.getmaskarray(data)
    newmask = np.append(datamask, datamask[:window]) # Make it circular
    newdata = ma.MaskedArray(newdata, mask=newmask)
    weight = []  
    for i in range(window):  
        i = i-degree+1  
        frac = i/float(window)  
        gauss = 1/(np.exp((4*(frac))**2))  
        weight.append(gauss)  
    weight = np.array(weight)
    sumweight = np.sum(weight)  
    smoothed = ma.zeros(len(newdata))
    print window, len(weight), 0-degree+1+window-(0-degree+1)
    for i in range(degree-1, len(smoothed)-degree):
        smoothed[i] = ma.sum(newdata[i-degree+1:i-degree+1+window]*weight)/sumweight
        if np.all(newmask[i-degree+1:i-degree+1+window]):
            smoothed[i] = ma.masked
    smoothed[:degree-1] = smoothed[origlen:origlen+degree-1]
    smoothed = smoothed[:origlen]
    smoothed.mask = data.mask
    return smoothed
 
def smooth_list_median(data, degree=5):
    datamask = ma.getmaskarray(data)  
    origlen = len(data)
    window = degree*2-1
    # np.append doesn't work property on masked arrays  
    newdata = np.append(data.data, data.data[:window]) # Make it circular
    datamask = ma.getmaskarray(data)
    newmask = np.append(datamask, datamask[:window]) # Make it circular
    newdata = ma.MaskedArray(newdata, mask=newmask)
    smoothed = ma.zeros(len(newdata))
    for i in range(degree-1, len(smoothed)-degree):  
        smoothed[i] = ma.median(newdata[i-degree+1:i-degree+1+window])
    smoothed[:degree-1] = smoothed[origlen:origlen+degree-1]
    smoothed = smoothed[:origlen]
    smoothed.mask = data.mask
    return smoothed
 
def smooth_list_mean(data, degree=5):
    datamask = ma.getmaskarray(data)  
    origlen = len(data)
    window = degree*2-1
    # np.append doesn't work property on masked arrays  
    newdata = np.append(data.data, data.data[:window]) # Make it circular
    datamask = ma.getmaskarray(data)
    newmask = np.append(datamask, datamask[:window]) # Make it circular
    newdata = ma.MaskedArray(newdata, mask=newmask)
    smoothed = ma.zeros(len(newdata))
    for i in range(degree-1, len(smoothed)-degree):  
        smoothed[i] = ma.mean(newdata[i-degree+1:i-degree+1+window])
    smoothed[:degree-1] = smoothed[origlen:origlen+degree-1]
    smoothed = smoothed[:origlen]
    smoothed.mask = data.mask
    return smoothed

def smooth_ew(ew_range, smooth_deg):
    long_res = 360./len(ew_range)
    smooth_pix = smooth_deg // long_res // 2
    #smooth the equivalent width range 
    smoothed_ew = ma.zeros(ew_range.shape[0])
    for n in range(len(ew_range)):
                    if ew_data.mask[n]:
                        smoothed_ew[n] = ma.masked
                    else:
                        smoothed_ew[n] = ma.mean(ew_range[max(n-smooth_pix,0):
                                                         min(n+smooth_pix+1,len(ew_range)-1)])
    return smoothed_ew

 
def find_clumps(longitudes, data, stddev):
    longitude_list = []
    contiguous = False
    contiguous_idx = None
    
    for idx in range(len(data)):
        if data[idx] > stddev:
            if not contiguous:
                contiguous_idx = idx
            contiguous = True
        else:
            if contiguous:
                longitude_list.append(longitudes[(contiguous_idx+idx)/2])                
            contiguous = False
    
    if contiguous:
        longitude_list.append(longitudes[(contiguous_idx+idx)/2])                

    return longitude_list

db_by_time = {}

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
    
    mean_phase_angle = np.mean(phase_angles)
    ew_data /= normalized_phase_curve(mean_phase_angle)
#    ew_data = np.log10(ew_data)
    
    mean_et = np.mean(ETs)
    
    db_by_time[mean_et] = [longitudes, ew_data, model_obsid, min_et, min_et_long]

if not options.display_one_at_a_time:
    fig = plt.figure()
    ax = fig.add_subplot(111)

y_pos = 1

if options.subtract_smoothed:
    y_scale = 4
else:
    y_scale = 1

# Find the min/max ET for all obs - for plot scaling
global_et_min = None
global_et_max = None
for key in sorted(db_by_time.keys()):
    longitudes, ew_data, obsid, min_et, min_et_long = db_by_time[key]
    if global_et_min == None:
        global_et_min = min_et
    global_et_max = min_et

if options.scale_by_time:
    y_scale = (global_et_max-global_et_min)/75.
    
for key in sorted(db_by_time.keys()):
    if options.display_one_at_a_time:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    longitudes, ew_data, obsid, min_et, min_et_long = db_by_time[key]
    if options.scale_by_time:
        y_pos = min_et - global_et_min
    if options.display_one_at_a_time:
        y_pos = 0

    ew_data_smoothed = ew_data

    plt.plot(longitudes, ew_data_smoothed + y_pos, '-', color='orange')
    
    if options.smooth_gaussian:
        ew_data_smoothed = smooth_list_gaussian(ew_data_smoothed, degree=int(options.smooth_gaussian_degree/
                                                    (options.longitude_resolution*options.mosaic_reduction_factor)))
    if options.smooth_median:
        ew_data_smoothed = smooth_list_median(ew_data_smoothed, degree=int(options.smooth_median_degree/
                                                    (options.longitude_resolution*options.mosaic_reduction_factor)))
    if options.subtract_smoothed:
        ew_data_smoothed = ew_data_smoothed - smooth_list_mean(ew_data_smoothed, degree=int(options.subtract_smoothed_degree/
                                                                                     (options.longitude_resolution*options.mosaic_reduction_factor)))
    if options.clump_smooth:
        ew_data_smoothed = smooth_ew(ew_data_smoothed, 3.2)
    stddev = ma.std(ew_data_smoothed)

    # Scale for plotting
    ew_data_smoothed = ew_data_smoothed * y_scale
    
    y_data_min = ma.min(ew_data_smoothed)
    y_data_max = ma.max(ew_data_smoothed)

    if options.show_stddev:
        for stddev_num in range(1,4):
            plt.plot([0., 360.], [stddev*stddev_num, stddev*stddev_num], '-', lw=1, color='cyan')

    plt.plot(longitudes, ew_data_smoothed + y_pos, '-', color='black')
    if options.show_prometheus_position:
        # Find the longitude at the point of closest approach
        max_dist, max_dist_long = ringutil.prometheus_close_approach(min_et, min_et_long)
        print 'Prometheus dist', max_dist, 'longitude', max_dist_long
        plt.plot([max_dist_long, max_dist_long], [y_data_min+y_pos, y_data_max+y_pos], '-', lw=1, color='red')
        if options.show_prometheus_waves:
            for long_32 in np.arange(max_dist_long-options.prometheus_wavelength,
                                     max_dist_long-360.,
                                     -options.prometheus_wavelength):
                if long_32 < 0:
                    long_32 += 360
                plt.plot([long_32, long_32], [y_data_min+y_pos, y_data_max+y_pos], '-', color='green')

    if options.find_clumps:
        clump_longitudes = find_clumps(longitudes, ew_data_smoothed, stddev*options.find_clumps_stddev)
        for longitude in clump_longitudes:
            plt.plot([longitude, longitude], [y_data_min+y_pos, y_data_max+y_pos], '-', color='magenta', lw=2)
    
    if not options.display_one_at_a_time:
        y_pos += 1
    else:
        ax.set_xlim(0,360)
        ax.set_ylim(-1, np.max(ew_data_smoothed)+1)
        plt.title(obsid)
        plt.show()

if not options.display_one_at_a_time:    
    ax.set_xlim(0,360)
    ax.set_ylim(0, y_pos)
    plt.show()
