# This Python file uses the following encoding: utf-8

'''
One file that generates all the figures for the upcoming paper.
Author: Shannon Hicks

'''
from optparse import OptionParser
import numpy as np
import numpy.ma as ma
import numpy.random as rand
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
import clumputil
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedFormatter
import pylab
import scipy.interpolate as interp
from imgdisp import ImageDisp
import Image
import string
import ringimage
import scipy.stats.distributions as scipydist
from scipy.misc import factorial

debug = True
choose_smaller = False
choose_larger = True

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = [
                '--scale-min', '5', '--scale-max', '50',
                '--clump-size-min', '5', '--clump-size-max', '40.0',
                '--plot-scalogram',
                '--mosaic-reduction-factor', '1',
                '-a'
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

paper_root = os.path.join(ringutil.PAPER_ROOT, 'Figures')
root_clump_db = {}

profile_color = 'black'

FRING_MEAN_MOTION = 581.964
OBSID_SAMPLE_PROFILE = 'ISS_059RF_FMOVIE002_VIMS'


color_background = (1,1,1)
color_foreground = (0,0,0)
color_dark_grey = (0.5, 0.5, 0.5)
color_grey = (0.625, 0.625, 0.625)
color_bright_grey = (0.75, 0.75, 0.75)
markersize = 8.5
markersize_voyager = 3.5

blackpoint = 0.
whitepoint = 0.522407851536
gamma = 0.5
radius_res = options.radius_resolution
radius_start = options.radius_start
matplotlib.rc('figure', facecolor=color_background)
matplotlib.rc('axes', facecolor=color_background, edgecolor=color_foreground, labelcolor=color_foreground)
matplotlib.rc('xtick', color=color_foreground, labelsize=7)
matplotlib.rc('xtick.major', size=6)
matplotlib.rc('xtick.minor', size=4)
matplotlib.rc('ytick', color=color_foreground, labelsize=7)
matplotlib.rc('ytick.major', size=6)
matplotlib.rc('ytick.minor', size=4)
matplotlib.rc('font', size=7)
matplotlib.rc('legend', fontsize=7)

def fix_graph_colors(fig, ax, ax2, legend):
    for line in ax.xaxis.get_ticklines() + ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines() + ax.yaxis.get_ticklines(minor=True):
        line.set_color(color_foreground)
    if legend != None:
        legend.get_frame().set_facecolor(color_background)
        legend.get_frame().set_edgecolor(color_background)
        for text in legend.get_texts():
            text.set_color(color_foreground) 

def save_fig(fig, ax, fn,ax2 = None, legend=None):
    fix_graph_colors(fig, ax, ax2, legend)
    fn = os.path.join(paper_root,fn)
    print 'Saving', fn
    plt.savefig(fn, bbox_inches='tight', facecolor=color_background, dpi=1000)   
    plt.close()

#--------------------------------------HELPER FUNCTIONS----------------------------------------------------------------------


def RelativeRateToSemimajorAxis(rate):  # from ringutil
    return ((FRING_MEAN_MOTION / (FRING_MEAN_MOTION+rate*86400.))**(2./3.) * 140220.)

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

def plot_one_clump(ax, ew_data, clump, long_min, long_max, label=False, clump_color='blue'):
    ncolor = clump_color
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
    ax.plot(idx_range, mexhat, '-', color=ncolor, lw=2, alpha=0.9, label=legend)
    if longitude_idx-clump.scale_idx/2 < 0: # Runs off left side - plot it twice
        ax.plot(idx_range+360, mexhat, '-', color=ncolor, lw=2, alpha=0.9)
    ax.set_xlim(long_min, long_max)
    
def plot_single_ew_profile(ax, ew_data, clump_db_entry, long_min, long_max, label=False, color = profile_color):
    
    long_res = 360. / len(ew_data)
    longitudes = np.arange(len(ew_data)) * long_res
    min_idx = int(long_min / long_res)
    max_idx = int(long_max / long_res)
    long_range = longitudes[min_idx:max_idx]
    ew_range = ew_data[min_idx:max_idx]
    legend = None
    if label:
        legend = clump_db_entry.obsid + ' (' + cspice.et2utc(clump_db_entry.et, 'C', 0) + ')'
    ax.plot(long_range, ew_range, '-', label=legend, color= color, lw = 1.0)
    
def plot_fitted_clump_on_ew(ax, ew_data, clump, color = 'blue'):
    long_res = 360./len(ew_data)
    longitudes =np.tile(np.arange(0,360., long_res),3)
    tri_ew = np.tile(ew_data, 3)
    left_idx = clump.fit_left_deg/long_res + len(ew_data)
    right_idx = clump.fit_right_deg/long_res + len(ew_data)
    
    if left_idx > right_idx:
        left_idx -= len(ew_data)
#    print left_idx, right_idx
    idx_range = longitudes[left_idx:right_idx]
#    print idx_range
    if left_idx < len(ew_data):
        ax.plot(longitudes[left_idx:len(ew_data)-1], tri_ew[left_idx:len(ew_data)-1], color = color, lw = 2)
        ax.plot(longitudes[len(ew_data):right_idx], tri_ew[len(ew_data):right_idx], color = color, lw = 2)
    else:
        ax.plot(idx_range, tri_ew[left_idx:right_idx], color = color, lw = 2)

def convert_angle(b_angle, s_angle, h_angle,m_angle,clump_width,min_ew, max_ew):
    
    min_max = max_ew - min_ew
    
    base = min_max/2.*(np.sin(b_angle)+.5) + min_ew
    scale = 0.5*clump_width*(np.sin(s_angle)+1) + 0.25*clump_width
    height = min_max/2.*(np.sin(h_angle)+1.5)
    center_offset = 0.5*clump_width*np.sin(m_angle)
   
    return (base, scale, height, center_offset)
        
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
    
    return (left_long, right_long, clump_int_height)
    
def fit_gaussian(tri_ews, clump_left_idx, clump_right_idx, clump_longitude_idx):
    
    #All indexes are assumed to be in accordance to a tripled EW profile to compensate for overlapping
    def fitting_func(params, xi, ew_range,clump_half_width,clump_center, min_ew, max_ew):
        m_angle, s_angle, h_angle, b_angle = params

        base, sigma, height, center_offset = convert_angle(b_angle, s_angle, h_angle,m_angle,clump_half_width,min_ew, max_ew)
        
        center = clump_center + center_offset
        xsd2 = -((xi-center)*(xi-center))/ (sigma**2)
        gauss = np.exp(xsd2/2.)   # Gaussian

#        print 'B %8.5f H %8.5f S %8.5f C %9.5f' % (base, height, sigma, center)

        if len(ew_range) <= 1:
            return 2*1e20
        
        residual = np.sum((gauss*height+base-ew_range)**2)

        return residual


    len_ew_data = len(tri_ews)//3
    long_res = 360./(len_ew_data)
    
    clump_longitude = (clump_longitude_idx - len_ew_data)*long_res 
    
    clump_center_idx = np.round((clump_right_idx+clump_left_idx)/2)

#    print 'WAVELET LONGITUDE', clump_longitude, 'MIN WAVELET', (clump_left_idx - len_ew_data)*long_res,
#    print 'MAX WAVELET', (clump_right_idx - len_ew_data)*long_res,
#    print 'CENTER', (clump_center_idx - len_ew_data)*long_res    

    clump_half_width_idx = clump_center_idx-clump_left_idx
    old_clump_ews = tri_ews[clump_left_idx:clump_right_idx+1]
    if len(old_clump_ews) < 3:                  #sometimes the downsampled versions don't end up with enough data to fit a gaussian to the arrays
        return(0)

    x = np.arange(clump_left_idx-len_ew_data, clump_right_idx- len_ew_data +1)*long_res

    min_ew = np.min(old_clump_ews)
    max_ew = np.max(old_clump_ews)

#    print 'MIN EW', min_ew, 'MAX EW', max_ew
#    print 'MAX EW LOCATION', (np.argmax(old_clump_ews) + clump_left_idx - len_ew_data)*long_res
    
    clump_data, residual, array, trash, trash, trash  = sciopt.fmin_powell(fitting_func, (0.,0.,0., 0.),
#                                                                               (np.pi/64., np.pi/64., np.pi/64., np.pi/64.),
                                                                       args=(x, old_clump_ews, clump_half_width_idx*long_res, clump_longitude, min_ew, max_ew),
                                                                       ftol = 1e-8, xtol = 1e-8,disp=False, full_output=True)

    m_angle, s_angle, h_angle, b_angle = clump_data
    base, sigma, height, center_offset = convert_angle(b_angle, s_angle, h_angle, m_angle, clump_half_width_idx*long_res, min_ew, max_ew)
    
    center = clump_longitude + center_offset                    #make this a multiple of our longitude resolution

#    print 'NEW CENTER', center # XXX
    
    left_sigma = center - 2*sigma
    right_sigma = center + 2*sigma

    x2 = np.arange(left_sigma,right_sigma +0.01, .01)

    left_sigma_idx = np.floor(left_sigma/long_res) + len_ew_data
    right_sigma_idx = np.ceil(right_sigma/long_res) + len_ew_data

    xsd2 = -((x2-center)*(x2-center))/ (sigma**2)
    gauss = np.exp(xsd2/2.)   # Gaussian
    gauss = gauss*height + base
    
    ew_range = tri_ews[left_sigma_idx:right_sigma_idx+1]
    ew_longs = np.arange(left_sigma_idx-len_ew_data, right_sigma_idx-len_ew_data +1)*long_res
    
#    print ew_range
    if len(ew_range) < 3:
        return (0)
    left_long, right_long, clump_int_height = find_edges(ew_range, gauss, ew_longs, center, sigma, long_res)
#    print left_long, right_long
    left_idx = left_long/long_res + len_ew_data
    right_idx = right_long/long_res + len_ew_data
    
    return (left_long, left_idx, right_long, right_idx, clump_int_height, center, base, height, sigma)


def baseline_value(ew_data):
    sorted_ew_data = np.sort(ma.compressed(ew_data))
    num_valid = len(sorted_ew_data)
    perc_idx = int(num_valid * 0.15)
    return sorted_ew_data[perc_idx]
    
def baseline_normalize(ew_data):
    return ew_data / baseline_value(ew_data)
    
#---------------------------------------------------------------------------------------------------------------------------------


#===============================================================================
#
# PLOT FDG WAVELET
# 
#===============================================================================

def plot_FDG():
    wavelet_scales = np.arange(0,26.0)
    x = np.arange (-100.0, 100., 1.0)
    mother_wavelet = cwt.FDG(len_signal=len(x), scales=wavelet_scales)
    
    color_background = (1,1,1)
    color_foreground = (0,0,0)
    color_dark_grey = (0.5, 0.5, 0.5)
    color_grey = (0.375, 0.375, 0.375)
    color_bright_grey = (0.25, 0.25, 0.25)
    figure_size = (3.3,2.)
    
    fig = plt.figure(figsize = figure_size, facecolor = color_grey, edgecolor = color_background)
    ax = fig.add_subplot(111, frame_on = True)
    fig.subplots_adjust(top = .98, bottom = 0.02, right = .98, left = 0.02)
    ax.tick_params(length = 5., width = 2., labelsize = 14. )

    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.get_xaxis().set_ticks([0])
    ax.get_yaxis().set_ticks([0])
    ax.set_ylim(-3.0,3.0)    
    ax.grid(True, color = color_grey)
    ax.set_axisbelow(True)
    ax.plot(x, mother_wavelet.coefs[25].real, '-', linewidth = 2.5, color = 'black')
    
    pylab.text(0., -0.5, 'Scale', {'color' : 'k', 'fontsize' : 8})
    ax.arrow(0.,0.0,17., 0.0, head_width = 0.15, head_length = 2.5, fc = 'k', ec = 'k', length_includes_head = True,
             shape='full')
    ax.arrow(17.,0.0,-17., 0.0, head_width = 0.15, head_length = 2.5, fc = 'k', ec = 'k', length_includes_head = True,
             shape='full')
#    ax.arrow(0,0.0,-17., 0.0, head_width = 0.15, head_length = 2.5, fc = 'k', ec = 'k', length_includes_head = True) #draw the other side
#    ax.arrow(0.,-1.025,23., 0.0, head_width = 0.15, head_length = 2.5, fc = 'k', ec = 'k', length_includes_head = True)
#    ax.arrow(0,-1.025,-23., 0.0, head_width = 0.15, head_length = 2.5, fc = 'k', ec = 'k', length_includes_head = True) #draw the other side
    save_fig(fig, ax, 'wavelet_fdg.png')
    plt.close()

    
#===============================================================================
#
# PLOT MEXHAT WAVELET
# 
#===============================================================================

def plot_mexhat():
    wavelet_scales = np.arange(0,26.0)
    x = np.arange (-100.0, 100., 1.0)
    mother_wavelet = cwt.SDG(len_signal=len(x), scales=wavelet_scales)
    
    color_background = (1,1,1)
    color_foreground = (0,0,0)
    color_dark_grey = (0.5, 0.5, 0.5)
    color_grey = (0.375, 0.375, 0.375)
    color_bright_grey = (0.25, 0.25, 0.25)
    figure_size = (3.3,2.)
    
    fig = plt.figure(figsize = figure_size, facecolor = color_background, edgecolor = color_background)
    ax = fig.add_subplot(111, frame_on = True)
    fig.subplots_adjust(top = .98, bottom = 0.02, right = .98, left = 0.02)
    ax.tick_params(length = 5., width = 2., labelsize = 14. )

    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.get_xaxis().set_ticks([0])
    ax.get_yaxis().set_ticks([0])
    ax.set_ylim(-1.0,1.0)    
    ax.grid(True, color = color_grey)
    ax.plot(x, mother_wavelet.coefs[25].real, '-', linewidth = 2.5, color = 'black')
    ax.set_axisbelow(True)
    
    pylab.text(3.5, -0.16, 'Scale', {'color' : 'k', 'fontsize' : 8})
    ax.arrow(0.,0.0,23., 0.0, head_width = 0.05, head_length = 2.5, fc = 'k', ec = 'k', length_includes_head = True,
             shape='full')
    ax.arrow(23.,0.0,-23., 0.0, head_width = 0.05, head_length = 2.5, fc = 'k', ec = 'k', length_includes_head = True,
             shape='full')
#    ax.arrow(0,0.0,-23., 0.0, head_width = 0.05, head_length = 2.5, fc = 'k', ec = 'k', length_includes_head = True) #draw the other side
    save_fig(fig, ax, 'wavelet_mexhat.png')
    plt.close()


#============================================================================
#
# PLOT LONGITUDE COVERAGE OVER TIME
# 
#============================================================================

def plot_long_coverage_over_time():
    fig = plt.figure(figsize = (7.0,3.0))
    ax = fig.add_subplot(111)
    ax.set_ylim(0.,360.)
    ax.get_yaxis().set_ticks(np.arange(0,360 + 45, 90.))
    sorted_ids = clumputil.get_sorted_obsid_list(c_all_clump_db)
    for obsid in sorted_ids:
        ew_data = c_all_clump_db[obsid].ew_data
        ew_mask = ew_data.mask
        long_res = 360./len(ew_data)
        longitudes = np.arange(0,360.,long_res)
        longitudes = longitudes.view(ma.MaskedArray)
        longitudes.mask = ew_mask
        time = np.zeros(len(longitudes)) + c_all_clump_db[obsid].et_max
        plt.plot(time, longitudes, color = 'black')
    
    labels = ['2004 JAN 01 00:00:00', '2005 JAN 01 00:00:000', '2006 JAN 01 00:00:000', '2007 JAN 01 00:00:000',
              '2008 JAN 01 00:00:000', '2009 JAN 01 00:00:000', '2010 JAN 01 00:00:000', '2011 JAN 01 00:00:000']
    
    et_ticks = [cspice.utc2et(label) for label in labels]
    sec_multiple = 3.15569e7               # number of seconds in 12 months 
    tick_min = np.min(et_ticks)
    tick_min = np.ceil((tick_min/sec_multiple))*sec_multiple
    tick_max = np.max(et_ticks)
    tick_max = np.ceil((tick_max/sec_multiple))*sec_multiple
    x_ticks = np.arange(tick_min, tick_max + sec_multiple, sec_multiple)
    ax.get_xaxis().set_ticks(et_ticks)
    ax.set_xlim(tick_min - 20*86400, tick_max + 20*86400)
    et_labels = []
    for k, et in enumerate(labels):
        et_labels.append(et[:4])
    
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.set_xticklabels(et_labels)
    ax.set_xlabel('Observation Date')
    ax.set_ylabel('Co-Rotating Longitude ( $\mathbf{^o}$)')
    fig.tight_layout()
    fig.autofmt_xdate()
    
    save_fig(fig, ax, 'longitude_coverage_over_time.png')


#===============================================================================
#
# PLOT MEAN CLUMP
# 
#===============================================================================

def plot_mean_clump():
    assert False
    
    """THIS SHOULD GO INTO SINGLE_CLUMP_FIGURES"""
    
    def convert_angle(b_angle, s_angle, h_angle, co_angle, clump_width):
        # XXX    
        base = 2*(np.sin(b_angle)+1)-0.1
        scale = clump_width*(np.sin(s_angle)+1)+20
        height = 1*(np.sin(h_angle)+1)
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
    
    def fitting_func(params, seed, ew_data, center_long_idx, clump_half_width_idx):
        co_angle, h_angle, s_angle, b_angle = params
        
        base, scale, height, center_offset_idx = convert_angle(b_angle, s_angle, h_angle, co_angle, clump_half_width_idx)
        center_idx = center_long_idx + center_offset_idx
        #extract data from the center to a scale's width to either side
        scale = clump_half_width_idx # XXX
        ew_range = ew_data[center_idx-scale:center_idx+scale+1]

        if len(ew_range) <= 1:
            return 2*1e20
        #interpolate the ew_range to 1000 points
        ew_range = interpolate(ew_range)
        
        residual = np.sum((ew_range*height+base-seed)**2)

        return residual

    if debug:
        print
        print '-' * 80
        print 'PLOT MEAN CLUMP'
        print '-' * 80
        
#    master_list_filename = os.path.join(ringutil.ROOT, 'clump-data', 'master_clumps_list.pickle')
#    master_list_fp = open(master_list_filename, 'rb')
#    master_clump_db, master_chain_list = pickle.load(master_list_fp)
#    master_list_fp.close()
#
#    clump_list = []
#    for chain in master_chain_list:
#        for clump in chain.clump_list:
#            clump_list.append(clump)
#    
#    # OLD VERSION
#    clump_list = np.array(clump_list)
#    keep_list = [1,3,4,5,9,10,11,12,13,16,18,19,21]
#    clump_list = clump_list[keep_list]
    
    for cass_voy_num in range(2):
        if cass_voy_num == 0:
            db = c_all_clump_db
        else:
            db = v_all_clump_db
        clump_list = []
        for obsid in db.keys():
            for clump in db[obsid].clump_list:
                clump_list.append(clump)
        
        if len(clump_list) > 100:
            clump_list = clump_list[:100]
            
        print 'MEAN CLUMP: Num clumps', len(clump_list)
        
        seed_idx = 10
        seed_clump = clump_list[seed_idx]
        clump_list = list(clump_list)
    
        full_ew_data = seed_clump.clump_db_entry.ew_data             #full set of normalized data
        long_res = 360./len(full_ew_data)
        
        seed_left_idx = np.round(seed_clump.fit_left_deg/long_res)
        seed_right_idx = np.round(seed_clump.fit_right_deg/long_res)
        seed_center_idx = np.round((seed_right_idx+seed_left_idx)/2)
        seed_half_width_idx = seed_center_idx-seed_left_idx
        
        if debug:
            print 'SEEDLEFT', seed_left_idx, 'SEEDRIGHT', seed_right_idx,
            print 'SEEDCTR', seed_center_idx, 'SEEDHALFWIDTH', seed_half_width_idx
        
        seed = seed_clump.clump_db_entry.ew_data[seed_left_idx:seed_right_idx+1]
        seed = seed - np.min(seed)
        seed = seed/np.max(seed)
        seed_int = interpolate(seed)
        
        i = 1
        clump_avg_arr = np.zeros(1000)
        for clump in clump_list:
            if debug:
                print 'Clump number', i
            #fit clump to the seed profile
            ew_data = clump.clump_db_entry.ew_data
            clump_left_idx = np.round(clump.fit_left_deg/long_res)
            clump_right_idx = np.round(clump.fit_right_deg/long_res)
            clump_center_idx = np.round((clump_right_idx+clump_left_idx)/2)
            clump_half_width_idx = clump_center_idx-clump_left_idx
            if debug:
                print 'CLUMPLEFT', clump_left_idx, 'CLUMPRIGHT', clump_right_idx,
                print 'CLUMPCTR', clump_center_idx, 'CLUMPHALFWIDTH', clump_half_width_idx
            #fit the seed to the clump to determine the width and center of the clump
            clump_data, residual, array, trash, trash, trash  = sciopt.fmin_powell(fitting_func, (0.,0.,0.,0.),
                                                                                   args=(seed_int, ew_data, clump_center_idx, clump_half_width_idx),
                                                                                   ftol = 1e-8, xtol = 1e-8,disp=False, full_output=True)
                   
            co_angle, h_angle, s_angle, b_angle = clump_data
            base, scale, height, center_offset_idx = convert_angle(b_angle, s_angle, h_angle, co_angle, clump_half_width_idx)
    
            if debug:     
                print '********************************************************'
                print base, scale, height, center_offset_idx, residual
            center = np.round(center_offset_idx + clump_center_idx)
            if center-scale < 0 or center+scale+1 > len(ew_data):
                continue
            new_clump_ews = ew_data[center-scale:center+scale+1]
            if len(new_clump_ews) < 5:
                continue
            print new_clump_ews
            i += 1
            
            new_clump_ews = interpolate(new_clump_ews)        
            new_clump_ews = new_clump_ews * height
            new_clump_ews = new_clump_ews + base 
    #        print i
    #        plt.plot(new_clump_ews)
    #        plt.show()
            clump_avg_arr = clump_avg_arr + new_clump_ews
    
        clump_avg_arr = clump_avg_arr/(len(clump_list)+1)
        clump_avg_arr = clump_avg_arr/np.max(clump_avg_arr)
        clump_avg_arr -= np.min(clump_avg_arr)
        
        max_idx = np.argmax(clump_avg_arr)
        if max_idx > len(clump_avg_arr)/2:
            clump_avg_arr = clump_avg_arr[max_idx-(len(clump_avg_arr)-max_idx):]
        elif max_idx < len(clump_avg_arr)/2:
            clump_avg_arr = clump_avg_arr[:max_idx*2]
            
    #    if clump_avg_arr[0] < clump_avg_arr[-1]:
    #        for min_idx in range(len(clump_avg_arr)):
    #            if clump_avg_arr[min_idx] >= clump_avg_arr[-1]:
    #                break
    #        clump_avg_arr = clump_avg_arr[min_idx:]
    #    elif clump_avg_arr[0] > clump_avg_arr[-1]:
    #        for max_idx in range(len(clump_avg_arr)-1,0,-1):
    #            if clump_avg_arr[max_idx] >= clump_avg_arr[0]:
    #                break
    #        clump_avg_arr = clump_avg_arr[:max_idx]
        
        clump_avg_arr = interpolate(clump_avg_arr)
        clump_avg_arr -= np.min(clump_avg_arr)
        
        fig = plt.figure(figsize = (3.5,2))
        ax = fig.add_subplot(111)
        plt.plot(clump_avg_arr, label='Mean Clump', color=color_dark_grey, lw=5, alpha=0.5)
    
        for func_name in ['Gaussian', 'SDG', 'FDG']:
            if func_name == 'Gaussian':
                sdg_scale = .4
                xi = np.arange(1000.) / 500. - 1.
                xsd2 = -xi * xi / (sdg_scale**2)
                mw = np.exp(xsd2/2.)   # Gaussian
                color = 'black'
                ls = '-'
            elif func_name == 'SDG':
                sdg_scale = .6
                xi = np.arange(1000.) / 500. - 1.
                xsd2 = -xi * xi / (sdg_scale**2)
                mw = (1. + xsd2) * np.exp(xsd2 / 2.)       #Second Derivative Gaussian
                color = 'black'
                ls = '--'
            elif func_name == 'FDG':
                sdg_scale = .75
                xi = np.arange(1000.) / 500. - 1.
                xsd2 = -xi * xi / (sdg_scale**2)
                mw = (xsd2*(xsd2 + 6) + 3.)*np.exp(xsd2/2.)   #Fourth Derivative Gaussian
                ls = ':'
                color = 'black'
        
            clump_data, residual, array, trash, trash, trash  = sciopt.fmin_powell(fitting_func, (0.,0.,0.,0.),
                                                                                   args=(clump_avg_arr, mw, len(mw)/2, len(mw)/4),
                                                                                   ftol = 1e-8, xtol = 1e-8,disp=False, full_output=True)
    
            if debug:         
                print func_name, residual
            
            co_angle, h_angle, s_angle, b_angle = clump_data
            base, scale, height, center_offset_idx = convert_angle(b_angle, s_angle, h_angle, co_angle, clump_half_width_idx)
            
            mw = mw*height + base
            
            plt.plot(mw, label=func_name, color = color, ls = ls)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        
        leg = plt.legend(handlelength=3, loc='lower center')
        leg.get_frame().set_alpha(0.0)
        leg.get_frame().set_visible(False)
        save_fig(fig, ax, 'mean_clump'+str(cass_voy_num)+'.png', leg)


#===============================================================================
#
# PLOT MATRIX OF EXAMPLE CLUMPS
#
#===============================================================================

def plot_clump_matrix(matrix_clumps):
    m_list = []
    
    for clump_num in matrix_clumps:
        for chain in c_approved_list:
            if chain.clump_num == clump_num:
                m_list.append(chain.clump_list[0])
                break
             
#    m_db = {}
#    m_list = []
#    time_list = [chain.clump_list[0].clump_db_entry.et_max for chain in matrix_list]
#    clump_list = [chain.clump_list[0] for chain in matrix_list]
#    for clump in clump_list:
##        print clump.clump_db_entry.obsid, clump.g_center
#        time = clump.clump_db_entry.et_max
#        if time not in m_db.keys():
#            m_db[time] = []
#            m_db[time].append(clump)
#        elif time in m_db.keys():
#            longs = np.array([aclump.g_center for aclump in m_db[time]])
#            idx = np.where(longs > clump.g_center)[0]
##            print longs, idx
#            if len(idx) == 0:
#                m_db[time].append(clump)
#            if len(idx) > 0:
#                m_db[time].insert(idx[0], clump)
##            print [bclump.g_center for bclump in m_db[time]]
#    for time in sorted(m_db.keys()):
#        for clump in m_db[time]:
#            m_list.append(clump)
            
#    for clump in m_list:
#        print clump.clump_db_entry.obsid, clump.g_center

    y_ax = 3
    x_ax = 7
    fig = plt.figure(1, (7.0, 5.0))
    grid = gridspec.GridSpec(4,3)
    grid.update(wspace=0.05, hspace=0.1)
    for i,clump in enumerate(m_list):
        ax = plt.subplot(grid[i])
        ew_data = clump.clump_db_entry.ew_data
        tri_ews = np.tile(ew_data,3)
        
        long_res = 360./len(ew_data)
        tri_longs = np.tile(np.arange(0.,360.,long_res),3)        
        left_idx_max = ((clump.fit_left_deg - 20.)%360.)/long_res + len(ew_data)
        right_idx_max = ((clump.fit_right_deg + 20.)%360.)/long_res + len(ew_data)
        left_idx = ((clump.fit_left_deg)/long_res) + len(ew_data)
        right_idx = clump.fit_right_deg/long_res + len(ew_data)
        
        g_left = (clump.g_center - 3.*clump.g_sigma)/long_res
        g_right = (clump.g_center + 3*clump.g_sigma)/long_res
        g_long_range = tri_longs[g_left:g_right]
        xsd2 = -((g_long_range-clump.g_center)*(g_long_range-clump.g_center))/ (clump.g_sigma**2)
        gauss = np.exp(xsd2/2.)   # Gaussian
        gauss = gauss*clump.g_height + clump.g_base
        
        x_left = tri_longs[left_idx_max]
        x_right = tri_longs[right_idx_max]
        new_x_left = (x_left+x_right)/2 - 15
        new_x_right = (x_left+x_right)/2 + 15
        x_left = new_x_left
        x_right = new_x_right
        xstep = np.round((x_right-x_left)/5.)
        ax.set_xlim(x_left, x_right)
        xticks = np.arange(x_left + xstep, x_right + xstep, xstep)
        
#        print x_left, x_right, xticks
        xticks.astype(int)
        if len(xticks) > 5:
            xticks = xticks[:5]
#        ax.get_xaxis().set_ticks(xticks)
        
        ax.xaxis.get_major_ticks()[-1].label1.set_visible(False)
        
#        print clump.fit_left_deg, clump.fit_right_deg, long_res, len(ew_data), left_idx, right_idx, left_idx_max, right_idx_max
        left_idx_disp = x_left / long_res
        right_idx_disp = x_right / long_res
        y_max = np.max(tri_ews[left_idx_disp:right_idx_disp+1])
        y_min = np.min(tri_ews[left_idx_disp:right_idx_disp+1])
#        print y_min, y_max
        ax.set_ylim(y_min, y_max)
#        y_step = (y_max - y_min)/3.
#        ax.get_yaxis().set_ticks([y_min + y_step, y_min + 2*y_step])
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        
#        xFormatter = FormatStrFormatter('%d')
#        yFormatter = FormatStrFormatter('%.2f')
#        ax.yaxis.set_major_formatter(yFormatter)
#        ax.xaxis.set_major_formatter(xFormatter)
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()
        
        if right_idx < left_idx:
            left_idx -= len(ew_data)
        if right_idx_max < left_idx_max:
            left_idx_max -= len(ew_data)
            left_xlim = tri_longs[left_idx_max]-360.
        
        if left_idx_max < len(ew_data):
            ax.plot(tri_longs[left_idx_max:len(ew_data)]-360., tri_ews[left_idx_max:len(ew_data)], color = color_dark_grey, lw = 1.0)
            ax.plot(tri_longs[len(ew_data):right_idx_max+1], tri_ews[len(ew_data):right_idx_max+1], color = color_dark_grey, lw = 1.0)
        else:
            ax.plot(tri_longs[left_idx_max:right_idx_max + 1],tri_ews[left_idx_max:right_idx_max + 1], color = color_dark_grey, lw = 1.0)
            
        if left_idx < len(ew_data):
            ax.plot(tri_longs[left_idx:len(ew_data)]-360., tri_ews[left_idx:len(ew_data)], color = 'black', lw = 1.5)
            ax.plot(tri_longs[len(ew_data):right_idx+1], tri_ews[len(ew_data):right_idx+1], color = 'black', lw = 1.5)
        else:
            ax.plot(tri_longs[left_idx:right_idx + 1],tri_ews[left_idx:right_idx +1], color = 'black', lw = 1.5)
            l, = ax.plot(g_long_range, gauss, color = color_grey, ls = ':', lw = 1.5)
            l.set_dashes([3,2])
            
#        ax.set_ylim(np.min(tri_ews[left_idx_max:right_idx_max]), np.max(tri_ews[left_idx_max:right_idx_max]))
        if i == y_ax:
            ax.set_ylabel(r'Equivalent Width')
        if i == x_ax:
            ax.set_xlabel(r'Co-Rotating Longitude ( $\mathbf{^o}$)')
    
    plt.savefig(os.path.join(paper_root, 'example_clump_matrix.png'), bbox_inches='tight', facecolor=color_background, dpi=1000)


#===========================================================================
#
# PLOT THE BEST CHAIN (ISS_029RF) OVER TIME
# 
#===========================================================================

def plot_029rf_over_time():
    fig = plt.figure(figsize = (7.0,7.0))
    #look for the right chain
    for chain in c_approved_list:
        if (len(chain.clump_list) == 6) and (chain.clump_list[0].clump_db_entry.obsid == 'ISS_029RF_FMOVIE001_VIMS'):
            for i,clump in enumerate(chain.clump_list):
                ax = fig.add_subplot(len(chain.clump_list),1,i+1)
                plot_single_ew_profile(ax, clump.clump_db_entry.ew_data, clump.clump_db_entry, 0., 360., color = color_dark_grey)
                plot_fitted_clump_on_ew(ax,clump.clump_db_entry.ew_data, clump, color = 'black')
                ax.set_xlim(0.,360.)
              
                line_height = np.arange(np.min(clump.clump_db_entry.ew_data),np.max(clump.clump_db_entry.ew_data),0.01)
                ax.plot(np.zeros(len(line_height))+clump.g_center,line_height, color = 'black', lw = 1.5)
                
                if i == 2:
                    ax.set_ylabel('X', fontsize = 10, alpha=0)
                
                ax.get_xaxis().set_ticks([0,360])
                ax.get_xaxis().set_ticks(np.arange(0, 360.+45., 45.),minor=True)
                y_min = np.min(clump.clump_db_entry.ew_data)
                y_max = np.max(clump.clump_db_entry.ew_data)
                ystep = (y_max-y_min)/2.
                ax.get_yaxis().set_ticks(np.arange(y_min, y_max + ystep, ystep )) 
                if i < len(chain.clump_list)-1:
                    ax.set_xticklabels('')
                
                fix_graph_colors(fig, ax, None, None)
            
                yFormatter = FormatStrFormatter('%.2f')
                ax.yaxis.set_major_formatter(yFormatter)    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    ax.get_xaxis().set_ticks([0,360])
    ax.get_xaxis().set_ticks(np.arange(0, 360.+45., 45.), minor=True) 
    ax.set_xlabel('Co-Rotating Longitude ( $\mathbf{^o}$)', fontsize = 10)
    plt.figtext(0.01, .70, r'Phase-Normalized Equivalent Width (km)', rotation = 'vertical', fontsize = 10)
    plt.savefig(os.path.join(paper_root,'clump_progression_ISS_029RF.png'), bbox_inches='tight', facecolor=color_background, dpi=1000)


#===============================================================================
#
# PLOT APPEARING AND DISAPPEARING AND PLAIN OLD-FASHIONED CLUMPS OVER TIME
# 
#===============================================================================

def plot_appearing_and_disappearing_clumps(appearing_clump_nums, disappearing_clump_nums, 
                                           appearing_disappearing_clump_nums, basic_clump_nums):
    def draw_clumps(im, clump, long_res, color, clip_left_idx, clip_bottom_idx, rad_center = 140180.):
        radius_center = int((rad_center-radius_start)/radius_res)-clip_bottom_idx
        
        left_idx = int(clump.fit_left_deg/long_res)-clip_left_idx #pixels
        right_idx = int(clump.fit_right_deg/long_res)-clip_left_idx
        ctr_idx = int(clump.g_center/long_res)-clip_left_idx
        height = 80 #pixels
        center_height = 15
        l_thick = 4
        w_thick = 4
        
#        print left_idx, right_idx, radius_center, im.shape
        
        for i in range(len(color)):
            im[radius_center+height:        radius_center+height+l_thick, left_idx: right_idx+l_thick, i] = color[i]
            im[radius_center-height-l_thick:radius_center-height,         left_idx: right_idx+l_thick, i] = color[i]
            im[radius_center-height-l_thick:radius_center+height+l_thick, left_idx: left_idx+w_thick, i] = color[i]
            im[radius_center-height-l_thick:radius_center+height+l_thick, right_idx:right_idx + w_thick, i] = color[i]
            im[radius_center+height-center_height:radius_center+height+center_height+1,
               ctr_idx:ctr_idx+l_thick, i] = color[i]
            im[radius_center-height-center_height:radius_center-height+center_height+1,
               ctr_idx:ctr_idx+l_thick, i] = color[i]
    
    def draw_entire_plot(clump_list):
        fig = plt.figure(figsize = (7.0, .8*len(clump_list)+0.2))
        num_axes = len(clump_list) + 1
        y_abs_max = []
        mosaic_max = []
        mosaic_clips = []

        xlefts = [clump.fit_left_deg for clump in clump_list]
        xrights = [clump.fit_right_deg for clump in clump_list]
        
        x_min_left = np.min(xlefts) - 5.
        x_max_right = np.max(xrights) + 5.
        x_min_left = np.floor(x_min_left/5.)*5.
        x_max_right = np.ceil(x_max_right/5.)*5.
        
        ypos_time = 0.80
        ypos_ylabel = .85
        ypos_xlabel = 0.15
        text_ylabel = 'Phase-Norm E.W. (km)'

        if len(clump_list) == 3:
            ypos_time = 0.80
            ypos_ylabel = .85
            ypos_xlabel = 0.15
            text_ylabel = 'Phase-Norm E.W. (km)'
        elif len(clump_list) == 5:
            ypos_time = 0.77
            ypos_ylabel = .83
            ypos_xlabel = 0.11
            text_ylabel = 'Phase-Norm Equivalent Width (km)'
        elif len(clump_list) == 6:
            ypos_time = 0.77
            ypos_ylabel = .78
            ypos_xlabel = 0.1
            text_ylabel = 'Phase-Normalized Equivalent Width (km)'
            
        for clump_num, clump in enumerate(clump_list):
#            if clump_num == 1:
#                clump.fit_left_deg, clump.fit_right_deg, clump.fit_width_idx, clump.fit_width_deg, clump.fit_height, clump.int_fit_height, clump.g_center, clump.g_sigma, clump.g_base, clump.g_height = refine_fit(clump, clump.clump_db_entry.ew_data, options)

            print 'Width', clump_num, clump.fit_width_deg
            
            ax = fig.add_subplot(num_axes, 2, clump_num*2+1)
            clump_ew_data = clump.clump_db_entry.ew_data
            long_res = 360./len(clump_ew_data)
            
            # Clump plot ticks
            ax.set_xlim(x_min_left, x_max_right)
            ax.get_xaxis().set_ticks(np.arange(x_min_left, x_max_right + 5., 5))
            clump_extract = clump_ew_data[x_min_left/long_res: x_max_right/long_res]
            if ma.any(clump_extract):
                ymin = np.min(clump_extract)-0.04
                ymax = np.max(clump_extract)+0.04
                y_abs_max.append(ymax)
                ystep = ymax/4.
                ytick1 = .8*(ymax-ymin)+ymin
                ytick2 = .2*(ymax-ymin)+ymin
                ytick1 = np.round(ytick1*100)/100
                ytick2 = np.round(ytick2*100)/100
                ax.set_ylim(ymin, ymax)
                ax.get_yaxis().set_ticks([ytick1, ytick2])
            ax.yaxis.tick_left()
            ax.xaxis.tick_bottom()
            xFormatter = FormatStrFormatter('%d')
            yFormatter = FormatStrFormatter('%.2f')
            ax.yaxis.set_major_formatter(yFormatter)
            ax.xaxis.set_major_formatter(xFormatter)
            plt.setp(ax.get_xticklabels(), visible=False)
            
            # Time label
            time = cspice.et2utc(clump.clump_db_entry.et_min, 'C', 0)[:11]
            ax.text(0.78, ypos_time, time, transform = ax.transAxes)
            
            # Plot the clump
            plot_single_ew_profile(ax, clump_ew_data, clump.clump_db_entry, 0.,360., color = color_dark_grey)
            if not clump.ignore_for_chain:
                plot_fitted_clump_on_ew(ax, clump.clump_db_entry.ew_data, clump, color = 'black')

                tri_longs = np.tile(np.arange(0.,360.,long_res),3)        
                
                g_left = (clump.g_center - 3.*clump.g_sigma)/long_res
                g_right = (clump.g_center + 3*clump.g_sigma)/long_res
                g_long_range = tri_longs[g_left:g_right]
                xsd2 = -((g_long_range-clump.g_center)*(g_long_range-clump.g_center))/ (clump.g_sigma**2)
                gauss = np.exp(xsd2/2.)   # Gaussian
                gauss = gauss*clump.g_height + clump.g_base
                l, = ax.plot(g_long_range, gauss, color = color_grey, ls = ':', lw = 1.5)
                l.set_dashes([3,2])

            # Create the mosaic
            obsid = clump.clump_db_entry.obsid
            (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
             bkgnd_mask_filename, bkgnd_model_filename, bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obsid)
                        
            mosaic_img = np.load(reduced_mosaic_data_filename + '.npy')
            mosaic_data_fp = open(reduced_mosaic_metadata_filename, 'rb')
            mosaic_data = pickle.load(mosaic_data_fp)
            
            (longitudes, resolutions,
            image_numbers, ETs, 
            emission_angles, incidence_angles,
            phase_angles) = mosaic_data
            
            mosaic_long_res = 360./len(longitudes)

            # Mosaic ticks
            ax2 = fig.add_subplot(num_axes, 2, clump_num*2+2)
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.setp(ax2.get_yticklabels(), visible=False)
            xticks = np.arange(x_min_left/mosaic_long_res, (x_max_right+5.)/mosaic_long_res, 5./mosaic_long_res)
            if len(xticks) > 0:
                ax2.set_xticks(xticks - xticks[0])
                xtick_labels = xticks*mosaic_long_res
                ax2.set_xticklabels([str(int(tick)) for tick in xtick_labels])
                ax2.tick_params(axis = 'x', direction = 'out', length = 2.0)
                ax2.xaxis.tick_bottom()
            ax2.set_yticks([])
            
            clip_left_idx = int(x_min_left/mosaic_long_res)
            clip_right_idx = int(x_max_right/mosaic_long_res)
            mosaic_clip = mosaic_img[400:650, clip_left_idx:clip_right_idx+1]
            for i in range(mosaic_clip.shape[1]):
                mosaic_clip[:,i] = ringutil.compute_corrected_ew(mosaic_clip[:,i] * ringutil.compute_mu(emission_angles[i+clip_left_idx]),
                                                                 emission_angles[i+clip_left_idx], np.mean(incidence_angles))
                ratio = ringutil.clump_phase_curve(0) / ringutil.clump_phase_curve(phase_angles[i+clip_left_idx])
                mosaic_img[:,i] *= ratio
            if ma.any(mosaic_clip):
                mosaic_max.append(ma.max(mosaic_clip))
            mosaic_clips.append((mosaic_clip, clip_left_idx, 400))

        ax.set_xlabel('X', fontsize = 10, alpha=0)
        ax.set_ylabel('X', fontsize = 10, alpha=0)
        
        total_axes = len(fig.axes)
            
        # Rescale all of the mosaics
        if not ma.any(mosaic_max):
            print 'BAD MAX'
            mosaic_max = 1.
        else:
            mosaic_max = ma.max(mosaic_max)
            even_axes = range(total_axes)[1::2]
            for l, ax_num in enumerate(even_axes):
                ax2 = fig.axes[ax_num]
                clump = clump_list[l]
                mosaic_clip, clip_left_idx, clip_bottom_idx = mosaic_clips[l]
                color_mosaic = np.zeros((mosaic_clip.shape[0], mosaic_clip.shape[1], 3))
                color_mosaic[:,:,0] = mosaic_clip
                color_mosaic[:,:,1] = mosaic_clip
                color_mosaic[:,:,2] = mosaic_clip
                final_im = ImageDisp.ScaleImage(color_mosaic, blackpoint, mosaic_max*0.5, gamma)+0
                if not clump.ignore_for_chain:
                    color = (250, 250, 250)
                    draw_clumps(final_im, clump, mosaic_long_res, color, clip_left_idx, clip_bottom_idx)
                final_im = np.cast['int8'](final_im)
                final_img = Image.frombuffer('RGB', (final_im.shape[1], final_im.shape[0]),
                                             final_im, 'raw', 'RGB', 0, 1)
                
                ax2.imshow(final_img, aspect = 'auto')
            
        fig.tight_layout()
        fig.subplots_adjust(hspace = 0.1, wspace = 0.08)
        plt.setp(ax.get_xticklabels(), visible=True)
        plt.setp(ax2.get_xticklabels(), visible=True)
        plt.setp(ax.get_yticklabels(), visible = True)

        ax2.tick_params(axis = 'x', direction = 'in', length = 2.0)
        plt.figtext(0.4, ypos_xlabel, 'Co-Rotating Longitude ( $\mathbf{^o}$)', fontsize = 10)
        plt.figtext(0.01, ypos_ylabel, text_ylabel, rotation = 'vertical', fontsize = 10)

        
    appearing_list = []
    
    for clump_num in appearing_clump_nums:
        for chain in c_approved_list:
            if chain.clump_num == clump_num:
                appearing_list.append(chain)
                break
        
    disappearing_list = []
    
    for clump_num in disappearing_clump_nums:
        for chain in c_approved_list:
            if chain.clump_num == clump_num:
                disappearing_list.append(chain)
                break

    appearing_disappearing_list = []
    
    for clump_num in appearing_disappearing_clump_nums:
        for chain in c_approved_list:
            if chain.clump_num == clump_num:
                appearing_disappearing_list.append(chain)
                break

#    appearing_list = c_approved_list
#    disappearing_list = c_approved_list
#    appearing_disappearing_list = c_approved_list
    
    basic_list = []
    
    for clump_num in basic_clump_nums:
        for chain in c_approved_list:
            if chain.clump_num == clump_num:
                basic_list.append(chain)
                break

    sorted_id_list = clumputil.get_sorted_obsid_list(c_approved_db)
    sorted_id_list = np.array(sorted_id_list)

    for app_chain in appearing_list:
        print 'Appearing', app_chain.clump_num
        start_id = app_chain.clump_list[0].clump_db_entry.obsid
        before_id_idx = np.where(sorted_id_list == start_id)[0][0]-1    
        before_id = sorted_id_list[before_id_idx]
    
        fake_clump = clumputil.ClumpData()
        fake_clump.clump_db_entry = c_approved_db[before_id]
        fake_clump.fit_left_deg = app_chain.clump_list[0].fit_left_deg
        fake_clump.fit_right_deg = app_chain.clump_list[0].fit_right_deg
        fake_clump.ignore_for_chain = True

        clump_list = [fake_clump] + app_chain.clump_list

        draw_entire_plot(clump_list)

        plt.savefig(os.path.join(paper_root,'clump_appear_' + app_chain.clump_num +'.png'), bbox_inches='tight', facecolor=color_background, dpi=1000)     
        
    for disapp_chain in disappearing_list:
        print 'Disappearing', disapp_chain.clump_num
        start_id = disapp_chain.clump_list[-1].clump_db_entry.obsid
        after_id_idx = np.where(sorted_id_list == start_id)[0][0]+1
        after_id = sorted_id_list[after_id_idx]
        
        fake_clump = clumputil.ClumpData()
        fake_clump.clump_db_entry = c_approved_db[after_id]
        fake_clump.fit_left_deg = disapp_chain.clump_list[0].fit_left_deg
        fake_clump.fit_right_deg = disapp_chain.clump_list[0].fit_right_deg
        fake_clump.ignore_for_chain = True

        clump_list = disapp_chain.clump_list + [fake_clump]

        draw_entire_plot(clump_list)

        plt.savefig(os.path.join(paper_root,'clump_disappear_' + disapp_chain.clump_num +'.png'), bbox_inches='tight', facecolor=color_background, dpi=1000)

    for app_chain in appearing_disappearing_list:
        print 'Appearing and disappearing', app_chain.clump_num
        start_id = app_chain.clump_list[0].clump_db_entry.obsid
        before_id_idx = np.where(sorted_id_list == start_id)[0][0]-1    
        before_id = sorted_id_list[before_id_idx]
    
        fake_clump = clumputil.ClumpData()
        fake_clump.clump_db_entry = c_approved_db[before_id]
        fake_clump.fit_left_deg = app_chain.clump_list[0].fit_left_deg
        fake_clump.fit_right_deg = app_chain.clump_list[0].fit_right_deg
        fake_clump.ignore_for_chain = True

        clump_list = [fake_clump] + app_chain.clump_list

        start_id = app_chain.clump_list[-1].clump_db_entry.obsid
        after_id_idx = np.where(sorted_id_list == start_id)[0][0]+1
        after_id = sorted_id_list[after_id_idx]

        fake_clump = clumputil.ClumpData()
        fake_clump.clump_db_entry = c_approved_db[after_id]
        fake_clump.fit_left_deg = app_chain.clump_list[0].fit_left_deg
        fake_clump.fit_right_deg = app_chain.clump_list[0].fit_right_deg
        fake_clump.ignore_for_chain = True

        clump_list = clump_list + [fake_clump]

        draw_entire_plot(clump_list)

        plt.savefig(os.path.join(paper_root,'clump_appear_disappear_' + app_chain.clump_num +'.png'), bbox_inches='tight', facecolor=color_background, dpi=1000)     
        
    for basic_chain in basic_list:
        print 'Basic', basic_chain.clump_num

        draw_entire_plot(basic_chain.clump_list)

        plt.savefig(os.path.join(paper_root,'clump_' + basic_chain.clump_num +'.png'), bbox_inches='tight', facecolor=color_background, dpi=1000)


#===============================================================================
#
# PLOT COMPARISON OF VOYAGER (ALL) AND CASSINI (SELECTED) PROFILES
# 
#===============================================================================

def plot_voyager_cassini_comparison_profiles():
    
#    obsid_list = ['ISS_000RI_SATSRCHAP001_PRIME','ISS_059RF_FMOVIE002_VIMS','ISS_075RF_FMOVIE002_VIMS', 
#                  'ISS_055RI_LPMRDFMOV001_PRIME', 'ISS_00ARI_SPKMOVPER001_PRIME']
    
    for use_baseline in [False, True]:
        for use_color in [False, True]:
            obsid_list = c_all_clump_db.keys()
            
            color_background = (1,1,1)
            figure_size = (8.5, 2.0)
            font_size = 18.0
                
            fig = plt.figure(figsize = figure_size)
            ax = fig.add_subplot(111)
            plt.subplots_adjust(top = 1.5, bottom = 0.0)
            ax.yaxis.tick_left()
            ax.xaxis.tick_bottom()
    
            if use_baseline:
                ax.set_ylabel('Baseline-Normalized Equivalent Width')
                ax.set_ylim(0,9)
                majorFormatter = FormatStrFormatter('%.1f')
                ax.yaxis.set_major_formatter(majorFormatter)
                ax.get_yaxis().set_ticks([0.0,9.0])
                ax.get_yaxis().set_ticks([1,2,3,4,5,6,7,8], minor=True)
            else:   
                ax.set_ylabel(r'Phase-Normalized Equivalent Width (km)')
                ax.set_ylim(0,3.5)
                ax.get_yaxis().set_ticks([0,3.5])
                ax.get_yaxis().set_ticks([0.5,1,1.5,2,2.5,3], minor=True)
            ax.set_xlabel(r'Co-Rotating Longitude ( $\mathbf{^o}$)')
            ax.set_xlim(0,360.)
            ax.get_xaxis().set_ticks([0,360])
            ax.get_xaxis().set_ticks([0,90,180,270,360], minor=True)
            
            cassini_list = []
            for plot_phase in range(3):
                for obs_id in obsid_list:
                    if not use_baseline and c_all_clump_db[obs_id].incidence_angle > 87:
                        print 'V/C COMPARISON PROFILES SKIPPING', obs_id
                        continue
                    c_ew_data = c_all_clump_db[obs_id].ew_data
                    if use_baseline:
                        c_ew_data = baseline_normalize(c_ew_data)
                    longitudes = np.arange(0.,360., 360./len(c_ew_data))
                    if use_color:
                        marker = '-'
                        color = 'black'
                        lw = 1.0 
                        alpha = 0.6
                        dashes = (1000000,1)
                    else:
                        marker = '-'
                        color = 'black'
                        lw = 1.0 
                        alpha = 0.6
                        dashes = (1000000,1)
                    if (obs_id == 'ISS_036RF_FMOVIE001_VIMS' or
                        obs_id == 'ISS_036RF_FMOVIE002_VIMS' or
                        obs_id == 'ISS_039RF_FMOVIE002_VIMS' or
                        obs_id == 'ISS_039RF_FMOVIE001_VIMS' or
                        obs_id == 'ISS_041RF_FMOVIE002_VIMS' or
                        obs_id == 'ISS_041RF_FMOVIE001_VIMS' or
                        obs_id == 'ISS_043RF_FMOVIE001_VIMS' or
                        obs_id == 'ISS_044RF_FMOVIE001_VIMS'):
                        if plot_phase != 1:
                            continue
                        if use_color:
                            marker = '-'
                            color = '#0080ff'
                            lw = 1.0
                            alpha = 0.7
                        else:
                            marker = ':'
                            color = 'black'
                            lw = 0.7
                            alpha = 1.0
                            dashes = (1,1)
                    elif (obs_id == 'ISS_105RF_FMOVIE002_VIMS' or
                          obs_id == 'ISS_106RF_FMOVIE002_PRIME' or
                          obs_id == 'ISS_107RF_FMOVIE002_PRIME' or
                          obs_id == 'ISS_108RI_SPKMVLFLP001_PRIME' or
                          obs_id == 'ISS_108RF_FMOVIE001_PRIME' or
                          obs_id == 'ISS_109RI_TDIFS20HLP001_CIRS' or
                          obs_id == 'ISS_111RF_FMOVIE002_PRIME' or
                          obs_id == 'ISS_112RF_FMOVIE002_PRIME'):
                        if plot_phase != 2:
                            continue
                        if use_color:
                            marker = '-'
                            color = '#30a030'
                            lw = 1.0
                            alpha = 0.7
                        else:
                            marker = ':'
                            color = 'black'
                            lw = 0.7
                            alpha = 1.0
                            dashes = (1.3,2)

                    elif plot_phase != 0:
                        continue
                    cassini = plt.plot(longitudes, c_ew_data, marker, color = color,
                                       lw = lw, alpha = alpha, dashes = dashes)
                    if len(cassini_list) == plot_phase:
                        cassini_list.append(cassini[0])
                
            for v_obs in v_all_clump_db.keys():
                v_ew_data = v_all_clump_db[v_obs].ew_data 
                v_ew_data = v_ew_data.view(ma.MaskedArray)
                v_mask = ma.getmaskarray(v_ew_data)
                empty = np.where(v_ew_data == 0.)[0]
                if empty != ():
                    v_mask[empty[0]-5:empty[-1]+5] = True
                v_ew_data.mask = v_mask
                if use_baseline:
                    v_ew_data = baseline_normalize(v_ew_data)
                longitudes = np.arange(0.,360., 360./len(v_ew_data))
                dashes = (1000000,1)
                if use_color:
                    if v_obs[:2] == 'V1':
                        marker = '-'
                        color = '#ff0000'
                        lw = 1.5
                        alpha = 0.9
                    else:
                        marker = '-'
                        color = '#980000'
                        lw = 1.5
                        alpha = 0.9
                else:
                    if v_obs[:2] == 'V1':
                        marker = ':'
                        color = 'black'
                        lw = 1.2
                        alpha = 1
                        dashes = (4,2.5)
                    else:
                        marker = '--'
                        color = 'black'
                        lw = 1.2
                        alpha = 1
                        dashes = (7,2.5)
                voyager = plt.plot(longitudes, v_ew_data, marker, color = color,
                                   lw = lw, alpha = 0.9, dashes=dashes)
                if v_obs == 'V1I':
                    voyager1 = voyager
                if v_obs == 'V2I':
                    voyager2 = voyager
                
            if use_baseline:
                leg = ax.legend([voyager1[0], voyager2[0], cassini_list[1], cassini_list[2], cassini_list[0]],
                                ['Voyager 1', 'Voyager 2', 'Cassini 2006', 'Cassini 2009', 'Cassini other'], loc = 1,
                                handlelength=3)
            else:
                leg = ax.legend([voyager1[0], voyager2[0], cassini_list[1], cassini_list[0]],
                                ['Voyager 1', 'Voyager 2', 'Cassini 2006', 'Cassini other'], loc = 1,
                                handlelength=3)
            leg.get_frame().set_alpha(0.0)
            leg.get_frame().set_visible(False)
            
            if use_color:
                color_str = '_color'
            else:
                color_str = ''
            if use_baseline:
                save_fig(fig, ax, 'voyager_cassini_profile_comparison_baseline'+color_str+'.png', leg)
            else:
                save_fig(fig, ax, 'voyager_cassini_profile_comparison'+color_str+'.png', leg)

#===============================================================================
#
# PLOT CASSINI/VOYAGER VELOCITY HISTOGRAM
# 
#===============================================================================

def plot_combined_vel_hist():
    c_velocities = []
    
    for chain in c_approved_list:
        c_velocities.append(chain.rate)
    
    c_velocities = np.array(c_velocities)*86400.            #change to deg/day
    
    v_velocities = np.array([-0.44, -0.306, -0.430, -0.232, -0.256, -0.183, -0.304, # V1
           -.410, -0.174, -0.063, -0.310, -0.277, -0.329, # V1
           -0.290, -0.412, -0.198, -0.258, -0.015, # V2
           -0.299, -.370, -0.195, -0.247, -0.303, -0.213, -0.172, -0.010]) + 0.306 # V2
    
    figure_size = (7.4, 3.0)
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
    
    ax_mm.set_xlabel('Relative Mean Motion ( $\mathbf{^o}$/Day )')
    ax_sma.set_xlabel('Semimajor Axis (km above 140,000)')
    ax_mm.yaxis.tick_left()

    ax_sma.get_yaxis().set_ticks([])
    ax_mm.get_xaxis().set_ticks(np.arange(-.8,.85,.1))
    ax_sma.get_xaxis().set_ticks(np.arange(100,350,20))

    ax_mm.set_ylabel('Fractional Number of Clumps')
    
    graph_min = np.floor(np.min(c_velocities) * 10) / 10. - 0.00001
    graph_max = np.ceil(np.max(c_velocities) * 10) / 10. + 0.00001
    step = 0.05
    ax_mm.set_xlim(graph_min, graph_max)
    bins = np.arange(graph_min,graph_max+step,step)
    
    counts, bins, patches = plt.hist([v_velocities, c_velocities], bins,
                                     weights = [np.zeros_like(v_velocities) + 1./v_velocities.size, np.zeros_like(c_velocities) + 1./c_velocities.size],
                                     label = ['Voyager', 'Cassini'], color = [color_grey, 'black'], lw = 0.0)
    
    pylab.text(0.415, 0.048, 'C54/', {'color' : 'k', 'fontsize' : 8})
    pylab.text(0.415, 0.025, '2009', {'color' : 'k', 'fontsize' : 8})
    pylab.text(0.559, 0.048, 'C19/', {'color' : 'k', 'fontsize' : 8})
    pylab.text(0.559, 0.025, '2006', {'color' : 'k', 'fontsize' : 8})

    leg = plt.legend()
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
    fig.tight_layout()
    save_fig(fig, ax_mm, 'voyager_cassini_clump_velocity_hist.png', ax_sma, leg)
    
    v_counts, bin_edges = np.histogram(v_velocities, bins, normed = 1)
    c_counts, bin_edges = np.histogram(c_velocities, bins, normed = 1)

    print '----------VELOCITY STATS-------------------'
    print
    print '--------|  Voyager | Cassini |---------'
    print ' MIN    |  %5.3f   |  %5.3f  |'%(np.min(v_velocities),np.min(c_velocities))
    print ' MAX    |  %5.3f   |  %5.3f  |'%(np.max(v_velocities),np.max(c_velocities))
    print ' MEAN   |  %5.3f   |  %5.3f  |'%(np.mean(v_velocities),np.mean(c_velocities))
    print ' STD    |  %5.3f   |  %5.3f  |'%(np.std(v_velocities), np.std(c_velocities))


#===============================================================================
# 
# PLOT PHASE CURVES FOR CASSINI AND VOYAGER
#
#===============================================================================

def plot_phase_curves():
    polyfit_order = 3

    def poly_scale_func(params, coeffs, phase_list, log_ew_list):
        scale = params[0]
        return np.sqrt(np.sum((log_ew_list - (np.polyval(coeffs, phase_list) + scale))**2))
        
    def plot_phase_curve(phase_list, ew_list, std_list, coeffs, fmt, ms, ls, mec, mfc, label):
        log_ew_list = np.log10(ew_list)
    
        if coeffs is None:
            coeffs = np.polyfit(phase_list, log_ew_list, polyfit_order)
            scale = 0.
        else:
            scale = sciopt.fmin_powell(poly_scale_func, (1.,), args=(coeffs, phase_list, log_ew_list),
                                       ftol = 1e-8, xtol = 1e-8, disp=False, full_output=False)
    
        std_list = None
        plt.errorbar(phase_list, ew_list, yerr=std_list, ecolor=mec, fmt=fmt, ms=ms, mec=mec, mfc=mfc, mew=0.5, label=label)
        plt.plot(np.arange(0,180), 10**scale*10**np.polyval(coeffs, np.arange(0,180)), ls, lw=1, color=mec)
    
        return coeffs, 10.**scale

    default_coeffs = [6.09918565e-07, -8.81293896e-05, 5.51688159e-03, -3.29583781e-01]

    class ObsData(object):
        pass
    
    obsdata_db = {}
    
    spacecraft_list = ['C', 'V1I', 'V1O', 'V2I', 'V2O']
    
    for key in spacecraft_list:
        obsdata_db[key] = []
        
    total_cassini_obs = 0
    used_cassini_obs = 0
    
    for obs_id, image_name, full_path in ringutil.enumerate_files(options, args, obsid_only=True):
        (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
        bkgnd_mask_filename, bkgnd_model_filename,
        bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obs_id)
    
        (ew_data_filename, ew_mask_filename) = ringutil.ew_paths(options, obs_id)
    
        if (not os.path.exists(ew_data_filename+'.npy')) or (not os.path.exists(reduced_mosaic_metadata_filename)):
            print 'NO DATA FOR', ew_data_filename
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
    
        orig_ew_data = ew_data
        orig_longitudes = longitudes
        orig_emission_angles = emission_angles
        
        ew_data = ma.compressed(ew_data)
        phase_angles = ma.compressed(phase_angles)
        emission_angles = ma.compressed(emission_angles)
        incidence_angles = ma.compressed(incidence_angles)
        ETs = ma.compressed(ETs)
        
        spacecraft = 'C'
        if obs_id[0] == 'V':
            spacecraft = obs_id
    
        if spacecraft == 'C':
            total_cassini_obs += 1
        
        if np.mean(incidence_angles) > 87:
            print 'SKIPPING EQUINOX', obs_id
            continue
    
        if spacecraft == 'C':
            used_cassini_obs += 1
        
        mu_ratio = ringutil.compute_mu(np.min(emission_angles)) / ringutil.compute_mu(np.max(emission_angles))
        if mu_ratio < 1:
            mu_ratio = 1/mu_ratio
        phase_ratio = 10**np.polyval(default_coeffs, np.max(phase_angles)) / 10**np.polyval(default_coeffs, np.min(phase_angles))
        if mu_ratio > phase_ratio:
            # Prefer emission
            zipped = zip(emission_angles, phase_angles, incidence_angles, ETs, ew_data)
            zipped.sort()
            emission_angles = [a for a,b,c,d,e in zipped]
            phase_angles = [b for a,b,c,d,e in zipped]
            incidence_angles = [c for a,b,c,d,e in zipped]
            ETs = [d for a,b,c,d,e in zipped]
            ew_data = [e for a,b,c,d,e in zipped]
        else:
            # Prefer phase
            zipped = zip(phase_angles, emission_angles, incidence_angles, ETs, ew_data)
            zipped.sort()
            phase_angles = [a for a,b,c,d,e in zipped]
            emission_angles = [b for a,b,c,d,e in zipped]
            incidence_angles = [c for a,b,c,d,e in zipped]
            ETs = [d for a,b,c,d,e in zipped]
            ew_data = [e for a,b,c,d,e in zipped]
        
        phase_angles = np.array(phase_angles)
        emission_angles = np.array(emission_angles)
        incidence_angles = np.array(incidence_angles)
        ETs = np.array(ETs)
        ew_data = np.array(ew_data)
        
        for num_splits in range(1,31):
            split_size = len(ew_data) // num_splits
            is_bad = False
            for split in range(num_splits):
                s_ea = emission_angles[split_size*split:split_size*(split+1)]
                s_pa = phase_angles[split_size*split:split_size*(split+1)]
                mu_ratio = ringutil.compute_mu(np.min(s_ea)) / ringutil.compute_mu(np.max(s_ea))
                if mu_ratio < 1:
                    mu_ratio = 1/mu_ratio
                phase_ratio = 10**np.polyval(default_coeffs, np.max(s_pa)) / 10**np.polyval(default_coeffs, np.min(s_pa))
    #            print obs_id, num_splits, np.min(s_ea), np.max(s_ea), mu_ratio, np.min(s_pa), np.max(s_pa), phase_ratio
                if mu_ratio > 1.2 or phase_ratio > 1.2:
                    is_bad = True
                if is_bad:
                    break
            if not is_bad:
                break
    
        split_size = len(ew_data) // num_splits
        
        for split in range(num_splits):
            s_ew_data = ew_data[split_size*split:split_size*(split+1)]
            s_phase_angles = phase_angles[split_size*split:split_size*(split+1)]
            s_emission_angles = emission_angles[split_size*split:split_size*(split+1)]
            s_incidence_angles = incidence_angles[split_size*split:split_size*(split+1)]
            s_ETs = ETs[split_size*split:split_size*(split+1)]
            
            mean_phase = ma.mean(s_phase_angles)
            mean_emission = ma.mean(s_emission_angles)
            mean_et = ma.mean(s_ETs)    
            mean_incidence = ma.mean(s_incidence_angles)
                
            s_ew_data *= ringutil.compute_mu(s_emission_angles)
        
            sorted_ew_data = np.sort(s_ew_data)
            num_valid = len(s_ew_data)
            perc_idx = int(num_valid * 0.15)
            baseline = sorted_ew_data[perc_idx]
            perc_idx = int(num_valid * 0.95)
            peak = sorted_ew_data[perc_idx]
            
            mean_ew = ma.mean(s_ew_data)
        
            obsdata = ObsData()
            obsdata.obs_id = obs_id
            obsdata.mean_phase_angle = mean_phase
            obsdata.phase_angles = s_phase_angles
            obsdata.mean_emission_angle = mean_emission
            obsdata.emission_angles = s_emission_angles
            obsdata.mean_incidence_angle = mean_incidence
            obsdata.mean_ew = mean_ew
            obsdata.ews = s_ew_data
            obsdata.baseline_ew = baseline
            obsdata.peak_ew = peak
            obsdata.mean_et = mean_et
            obsdata_db[spacecraft].append(obsdata)
        
            percentage_ok = float(len(np.where(longitudes >= 0)[0])) / len(longitudes) * 100
        
            print '%-30s/%d %3d%% P %7.3f %7.3f-%7.3f E %7.3f %7.3f-%7.3f I %7.3f %-15s EW %8.5f +/- %8.5f' % (obs_id, split, percentage_ok,
                mean_phase, np.min(s_phase_angles), np.max(s_phase_angles), 
                mean_emission, np.min(s_emission_angles), np.max(s_emission_angles), 
                mean_incidence, cspice.et2utc(mean_et, 'C', 0)[:12], mean_ew, np.std(s_ew_data))
        
    print 'TOTAL CASSINI OBS', total_cassini_obs
    print 'USED CASSINI OBS', used_cassini_obs
    
    v_tau_base = 0.035
    c_tau_base = 0.035
    
    print 'ASSUMED C TAU BASE', c_tau_base
    print 'ASSUMED V TAU BASE', v_tau_base

    c_mean_phase_list = [x.mean_phase_angle for x in obsdata_db['C']]

    c_base_ew_list = ringutil.compute_corrected_ew([x.baseline_ew for x in obsdata_db['C']],
                                          [x.mean_emission_angle for x in obsdata_db['C']],
                                          [x.mean_incidence_angle for x in obsdata_db['C']],
                                          c_tau_base)
    fig = plt.figure(figsize=(3.5,2.2))
    ax = fig.add_subplot(111)
    
    v1io_phase = []
    v1io_base_ews = []
    v2io_phase = []
    v2io_base_ews = []
    
    base_str = ''
    peak_str = ''
    peak_base_str = ''
    
    for v_prof_key in ['V1I', 'V1O', 'V2I', 'V2O']:
        phase_angle = [x.mean_phase_angle for x in obsdata_db[v_prof_key]]
        print phase_angle
        base_ew_list = ringutil.compute_corrected_ew([x.baseline_ew for x in obsdata_db[v_prof_key]],
                                            [x.mean_emission_angle for x in obsdata_db[v_prof_key]],
                                            [x.mean_incidence_angle for x in obsdata_db[v_prof_key]],
                                            v_tau_base)
        if v_prof_key[:2] == 'V1':
            color = 'red'
            v1io_phase.append(phase_angle[0])
            v1io_base_ews.append(base_ew_list[0])
        else:
            color = 'green'
            v2io_phase.append(phase_angle[0])
            v2io_base_ews.append(base_ew_list[0])

    coeffs = default_coeffs
    print coeffs
    _, _ = plot_phase_curve(c_mean_phase_list, c_base_ew_list, None, None, 'o', 3.5, '-', 'black', 'none', 'Cassini')
    _, v1iob_scale = plot_phase_curve(v1io_phase, v1io_base_ews, None, coeffs, '^', 3.5, '-', 'black', 'black', 'Voyager 1')
    _, v2iob_scale = plot_phase_curve(v2io_phase, v2io_base_ews, None, coeffs, '^', 3.5, '-', 'black', 'none', 'Voyager 2')
    print 'Cass Base / V1IO Base  ', 1/v1iob_scale
    print 'Cass Base / V2IO Base  ', 1/v2iob_scale
    print 'V1IO Base / V2IO Base  ', v1iob_scale / v2iob_scale
    
    ax.set_yscale('log')
    plt.xlabel(r'Phase angle ($^\circ$)')
    plt.ylabel(r'$W_\tau$ (km)')
    plt.title('')
    ax.set_xlim(0,180)
    ax.set_ylim(.08, 30)
    ax.get_xaxis().set_ticks([0., 180.])
    ax.get_xaxis().set_ticks([90.], minor=True)
    ax.get_yaxis().set_major_formatter(FixedFormatter(['', '0.1', '1', '10']))    

    leg = plt.legend(loc='upper left', ncol=1, numpoints=1)

    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
    fig.tight_layout()
    save_fig(fig, ax, 'phase_curves.png', legend=leg)

#===============================================================================
# 
#===============================================================================
#===============================================================================
# 
#===============================================================================
#===============================================================================
# 
#===============================================================================
#===============================================================================
# 
#===============================================================================
#===============================================================================
# 
#===============================================================================

def dump_obs_table(c_approved_list, c_approved_db):
    angle_table = os.path.join(paper_root, 'obs_table1.txt')
    angle_file = open(angle_table, 'w')
    
    id_data_file = open(os.path.join(paper_root, 'obs_table2.txt'), 'w')
    
    angle_file.write('ID #,Date,Cassini Observation ID,Radial Resolution (km/pixel),Phase Angle (Deg),Emission Angle (Deg),Incidence Angle (Deg)\n')
    id_data_file.write('ID #,Date,Cassini Observation ID,First Image,Last Image,Number of Images,Coverage Percentage,Number of ECs,Number of MDCs\n')

    sorted_id_list = clumputil.get_sorted_obsid_list(c_approved_db)
    
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
            
            if not c_all_clump_db.has_key(obs_id):
                continue
            
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
        
            ew_data = np.load(ew_data_filename+'.npy')
            ew_mask = np.load(ew_mask_filename+'.npy')
            temp_ew_data = ew_data.view(ma.MaskedArray)
            temp_ew_data.mask = ew_mask
            ew_mask = ma.getmaskarray(temp_ew_data)
            
            min_emission = np.min(mosaic_emission_angles[~ew_mask])
            max_emission = np.max(mosaic_emission_angles[~ew_mask])
            min_phase = np.min(mosaic_phase_angles[~ew_mask])
            max_phase = np.max(mosaic_phase_angles[~ew_mask])
            min_et = c_approved_db[obs_id].et_min
            mean_incidence = c_approved_db[obs_id].incidence_angle
            min_resolution = c_approved_db[obs_id].resolution_min
            max_resolution = np.max(mosaic_resolutions[np.where(mosaic_resolutions < 1000)])
            
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
    
            if c_db.has_key(obs_id):
                num_clumps = len(c_db[obs_id].clump_list)
            else:
                num_clumps = 0
            num_ecs = len(c_all_clump_db[obs_id].clump_list)
            if obs_id in c_db.keys():
                print obs_id, ' | ', cspice.et2utc(min_et, 'D', 5), ' | ', cspice.et2utc(c_approved_db[obs_id].et_max, 'D', 5), ' | ', num_clumps
            
            start_date = cspice.et2utc(min_et, 'C', 0)
            start_date = start_date[0:11]
            angle_file.write('%i,%s,%20s,%i - %i,%.1f - %.1f,%.1f - %.1f,%.1f\n'%(a+1, start_date, obs_id, min_resolution, max_resolution, min_phase, max_phase,
                                                                                 min_emission, max_emission, mean_incidence))
            id_data_file.write('%i,%s,%20s,%s,%s,%i,%.1f,%i,%i\n'%(a+1, start_date, obs_id, image_name_list[0], image_name_list[-1],
                                                                   num_images, percent_not_masked, num_ecs, num_clumps))
    
    id_data_file.close()
    angle_file.close()
    
def dump_clump_table(c_approved_list, c_approved_db):
    file = 'clump_data_table.txt'
    clump_table_fn = os.path.join(paper_root, file)
    clump_table = open(clump_table_fn, 'w')
    
    clump_table.write('Clump Number,First ID #,Last ID #,Number of Observations,Longitude at Epoch (deg),Minimum - Maximum Lifetime (Days),\
Relative Mean Motion (deg/day),Semimajor Axis (km),Median Width (deg),dw/dt,Median BN Int Brt (deg),Median PN Int Brt (km deg),Median BN Peak Brt (deg),Median PN Peak Brt (km deg)\n')

    chain_time_db = {}
    for chain in c_approved_list:
        chain.skip = False
        if not 'clump_num' in chain.__class__.userattr:
            chain.__class__.userattr += ['clump_num'] # This is a horrible hack due to loading in classes through Pickle
        chain.clump_num = None
        start_date = chain.clump_list[0].clump_db_entry.et_max
        if start_date not in chain_time_db.keys():
            chain_time_db[start_date] = []
            chain_time_db[start_date].append(chain)
        elif start_date in chain_time_db.keys():
            chain_time_db[start_date].append(chain)
    
    for obsid in c_approved_db:
        max_time = c_approved_db[obsid].et_max
        if max_time not in chain_time_db.keys():
            chain_time_db[max_time] = []

    for chain_time in chain_time_db:
        chain_list = chain_time_db[chain_time]
        chain_list.sort(key=lambda x: x.clump_list[0].g_center * 1000 + x.clump_list[1].g_center)

    sorted_id_list = clumputil.get_sorted_obsid_list(c_approved_db)
    num_db = {}
    for i, obsid in enumerate(sorted_id_list):
#        print obsid, i +1
        num_db[obsid] = i +1
            
    dw_dt_list = []
    ddeltah_dt_list = []
    ddeltaw_dt_list = []
    widths_list = []
    num = 1
    for time in sorted(chain_time_db.keys()):
        for a,chain in enumerate(chain_time_db[time]):
            parent_clump_start_long = '%6.2f'%(chain.clump_list[0].g_center)
            parent_clump_end_long = '%6.2f'%(chain.clump_list[-1].g_center)
            parent_clump_end_time = chain.clump_list[-1].clump_db_entry.et_max
            num_id = 'C'+str(num)
            chain.clump_num = num_id

            clump_widths = [x.fit_width_deg for x in chain.clump_list]
            
            clump_times = [x.clump_db_entry.et_min/86400 for x in chain.clump_list]
            coeffs = np.polyfit(clump_times, clump_widths, 1)

            clump1 = chain.clump_list[0]
            clump2 = chain.clump_list[-1]
            
            time1 = clump1.clump_db_entry.et/86400.
            time2 = clump2.clump_db_entry.et/86400.
            
            dt = time2-time1
            dw_dt = coeffs[0]

            clump_fracdeltaheight = (clump2.int_fit_height-clump1.int_fit_height)/clump1.int_fit_height/dt
            clump_fracdeltawidth = (clump2.fit_width_deg-clump1.fit_width_deg)/clump1.fit_width_deg/dt
            
            if dt >= 14:
                dw_dt_list.append(dw_dt)
                ddeltah_dt_list.append(clump_fracdeltaheight)
                ddeltaw_dt_list.append(clump_fracdeltawidth)
            else:
                dw_dt = -100
                clump_fracdeltaheight = -100
                clump_fracdeltawidth = -100

            equinox = False            
            for clump in chain.clump_list:
                if clump.clump_db_entry.incidence_angle > 87:
                    equinox = True
                if clump.fit_width_deg <= 40:
                    widths_list.append(clump.fit_width_deg)
                
            max_life = chain.lifetime + chain.lifetime_upper_limit + chain.lifetime_lower_limit
            first_id_num = num_db[chain.clump_list[0].clump_db_entry.obsid]
            last_id_num = num_db[chain.clump_list[-1].clump_db_entry.obsid]
            med_width = np.median(np.array([clump.fit_width_deg for clump in chain.clump_list]))
            med_pn_bright = np.median(np.array([clump.int_fit_height for clump in chain.clump_list]))
            med_bn_bright = np.median(np.array([clump.int_fit_height/baseline_value(clump.clump_db_entry.ew_data) for clump in chain.clump_list]))
            med_peak_pn_bright = np.median(np.array([clump.fit_height for clump in chain.clump_list]))
            med_peak_bn_bright = np.median(np.array([clump.fit_height/baseline_value(clump.clump_db_entry.ew_data) for clump in chain.clump_list]))
            
            start_long = chain.clump_list[0].g_center
            num_obs = len(chain.clump_list)
            str_format = '%s,%i,%i,%1i,%.1f,%i - %i,%.3f x %.3f,%.1f x %.1f,%.1f,%.2f,%.3f,'%(num_id, first_id_num, last_id_num, num_obs,
                                                                                                  start_long, np.round(chain.lifetime), np.round(max_life), chain.rate*86400.,
                                                                                                  chain.rate_err*86400., chain.a, chain.a_err, med_width, 
                                                                                                  dw_dt,med_bn_bright)
            if equinox:
                str_format += 'N/A,'
            else:
                str_format += '%.3f,' % med_pn_bright
                
            str_format += '%.3f,' % med_peak_bn_bright
            
            if equinox:
                str_format += 'N/A'
            else:
                str_format += '%.3f' % med_peak_pn_bright 
                 
            chain.clump_num = num_id
            print str_format
            clump_table.write(str_format+'\n')
        
            num +=1

    clump_table.close()

    print '-----------DW/DT STATS----------'
    print 'NUM CLUMPS', len(dw_dt_list)
    print 'MIN %.2f' % np.min(dw_dt_list)
    print 'MAX %.2f' % np.max(dw_dt_list)
    print 'MEAN %.2f' % np.mean(dw_dt_list)
    print 'STD %.2f' % np.std(dw_dt_list)
    print 'MEDIAN %.2f' % np.median(dw_dt_list)
    print
    
    print '-----------DH%/DT STATS----------'
    print 'NUM CLUMPS', len(ddeltah_dt_list)
    print 'MIN %.2f' % np.min(ddeltah_dt_list)
    print 'MAX %.2f' % np.max(ddeltah_dt_list)
    print 'MEAN %.2f' % np.mean(ddeltah_dt_list)
    print 'STD %.2f' % np.std(ddeltah_dt_list)
    print 'MEDIAN %.2f' % np.median(ddeltah_dt_list)
    
    step = 0.05
    fig = plt.figure(figsize = (3.5,2))
    ax = fig.add_subplot(111)
    b_min = (np.min(dw_dt_list) // step) * step
    b_max = (np.max(dw_dt_list) // step + 1) * step
    bins = np.arange(b_min, b_max+step, step)
    dw_dt_weights = np.zeros_like(dw_dt_list) + 1./len(dw_dt_list)
    plt.hist(dw_dt_list, bins, weights = dw_dt_weights, color=color_foreground)
    plt.xlim(b_min,b_max)
    ax.set_xlabel(r'$\Delta w/\Delta t$ ($^\circ$/day)')
    ax.set_ylabel('Fractional Number of Clumps')

    save_fig(fig, ax, 'change_width_over_time.png')

    step = 0.01
    fig = plt.figure(figsize = (3.5,2))
    ax = fig.add_subplot(111)
    b_min = (np.min(ddeltah_dt_list) // step) * step
    b_max = (np.max(ddeltah_dt_list) // step + 1) * step
    bins = np.arange(b_min, b_max+step, step)
    ddeltah_dt_weights = np.zeros_like(ddeltah_dt_list) + 1./len(ddeltah_dt_list)
    plt.hist(ddeltah_dt_list, bins, weights = ddeltah_dt_weights, color=color_foreground)
    plt.xlim(b_min,b_max)
    ax.set_xlabel(r'$Fractional \Delta b/\Delta t$ ($^\circ$/day)')
    ax.set_ylabel('Fractional Number of Clumps')

    save_fig(fig, ax, 'change_brightness_over_time.png')

    fig = plt.figure(figsize = (3.5,2))
    ax = fig.add_subplot(111)
    plt.plot(ddeltaw_dt_list, ddeltah_dt_list, 'o', color='black')
    coeffs = np.polyfit(ddeltaw_dt_list, ddeltah_dt_list, 1)
    print 'DW/DH coeffs', coeffs
    plt.plot([np.min(ddeltaw_dt_list),np.max(ddeltaw_dt_list)],
             [np.polyval(coeffs, np.min(ddeltaw_dt_list)), np.polyval(coeffs, np.max(ddeltaw_dt_list))], '-', color='red')
    ax.set_xlabel(r'Fractional Change in Width')
    ax.set_ylabel('Fractional Change in Brightness')

    save_fig(fig, ax, 'dwidth_vs_dbrightness.png')


#===============================================================================
# 
# PLOT THE SCALOGRAM FOR THE SAMPLE PROFILE
#
#===============================================================================

def plot_sample_scalogram():
    obs_id = OBSID_SAMPLE_PROFILE
    # This is only used for creating the sample profiles
#    mosaic_fn = os.path.join(paper_root, OBSID_SAMPLE_PROFILE + '_mosaic_data.npy')
#    mosaic_png_fn = os.path.join(paper_root, OBSID_SAMPLE_PROFILE + '_mosaic.png')
#    ew_data_fn = os.path.join(paper_root, OBSID_SAMPLE_PROFILE + '_ew_data.npy')
#    ew_mask_fn = os.path.join(paper_root, OBSID_SAMPLE_PROFILE + '_ew_mask.npy')
#    ew_data_fn, ew_mask_fn = ew_paths(options, obs_id)

#    root_clump_db_fn = os.path.join(paper_root, 'clump_database.pickle')
    
    _, metadata_fn, _, _ = ringutil.mosaic_paths(options, obs_id)
    mosaic_metadata_fp = open(metadata_fn, 'rb')
    metadata = pickle.load(mosaic_metadata_fp)
    mosaic_metadata_fp.close()

#    mosaic = np.load(mosaic_fn)
    
    #Have to run the clump finding program first to generate the clump database.
    
#    ew_data = np.load(ew_data_fn)
#    ew_mask = np.load(ew_mask_fn)
#    temp_ew_data = ew_data.view(ma.MaskedArray)
#    temp_ew_data.mask = ew_mask
#    ew_mask = ma.getmaskarray(temp_ew_data)
    
    ew_data = c_all_clump_db[OBSID_SAMPLE_PROFILE].ew_data
    ew_mask = ma.getmaskarray(ew_data)
    
    long_res = 360./len(ew_data)
    longitudes = np.arange(0,360., long_res)
    
    print 'STARTING SDG'
    sdg_clump_db_entry, sdg_wavelet_data = find_clumps_internal(options, ew_data, ew_mask, longitudes, obs_id, metadata, wavelet_type = 'SDG')
    print 'STARTING FDG'
    fdg_clump_db_entry, fdg_wavelet_data = find_clumps_internal(options, ew_data, ew_mask, longitudes, obs_id, metadata, wavelet_type = 'FDG')
    
    sdg_list = sdg_clump_db_entry.clump_list
    fdg_list = fdg_clump_db_entry.clump_list
    
    print 'CHOOSE THE BEST WAVELETS'
    new_clump_list = []
    longitude_list = []
    for n, clump_s in enumerate(sdg_list):
        f_match_list = []
        s_match_list = [clump_s]
        for clump_f in fdg_list:
            if (clump_s.longitude - 2.0 <= clump_f.longitude <= clump_s.longitude + 2.0) and (0.5*clump_s.scale <= clump_f.scale <= 2.5*clump_s.scale):
                #there is a match!
                f_match_list.append(clump_f)
                clump_f.matched = True
                clump_s.matched = True
        
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
    #it doesn't matter which clump_db_entry we choose since the metadata is the same for both. The only differences are the clumps.
    fdg_clump_db_entry.clump_list = new_clump_list
    root_clump_db[OBSID_SAMPLE_PROFILE] = fdg_clump_db_entry     
    plot_scalogram(sdg_wavelet_data, fdg_wavelet_data, fdg_clump_db_entry)
    
    return (sdg_wavelet_data, fdg_wavelet_data)

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

def find_clumps_internal(options, ew_data, ew_mask, longitudes, obs_id, metadata, wavelet_type = 'SDG'):
    
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
        return np.sum((wavelet*wscale+base-data)**2)
    
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
#
           #calculate refined fit
            clump.fit_left_deg, clump.fit_right_deg, clump.fit_width_idx, clump.fit_width_deg, clump.fit_height, clump.int_fit_height, clump.g_center, clump.g_sigma, clump.g_base, clump.g_height = refine_fit(clump, ew_data, options)
            
            #calculate clump sigma height
            profile_sigma = ma.std(ew_data)
#            height = clump.mexhat_height*mexhat[len(mexhat)//2]
            clump.clump_sigma = clump.abs_height/profile_sigma
            clump.fit_sigma = clump.fit_height/profile_sigma
            
#            print vars(clump)
            clump_list.append(clump)
    
    clump_db_entry = clumputil.ClumpDBEntry()
    clump_db_entry.obsid = obs_id
    
    clump_db_entry.clump_list = clump_list
    clump_db_entry.ew_data = ew_data # Smoothed and normalized
   
    clump_db_entry.et = np.mean(mosaic_ETs[~ew_mask])
    clump_db_entry.resolution_min = np.min(mosaic_resolutions[~ew_mask])
    clump_db_entry.resolution_max = np.max(mosaic_resolutions[~ew_mask])
    clump_db_entry.emission_angle = np.mean(mosaic_emission_angles[~ew_mask])
    clump_db_entry.incidence_angle = np.mean(mosaic_incidence_angles[~ew_mask])
    clump_db_entry.phase_angle = np.mean(mosaic_phase_angles[~ew_mask])

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
        
    wavelet_data = (wavelet, mother_wavelet, wavelet_scales)
    return (clump_db_entry, wavelet_data)

def refine_fit(clump, ew_data, options):

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

    
    right_idx = wav_center_idx + np.argmin(right_ew_range)

    
    if (options.downsample == True) or (options.voyager == True):
        clump_params = fit_gaussian(tri_ew, left_idx, right_idx, wav_center_idx)
#        print clump_params
        if clump_params != 0:
            left_deg, left_idx, right_deg, right_idx, clump_int_height, gauss_center, gauss_base, gauss_height, gauss_sigma = clump_params
            fit_width_idx = abs(right_idx - left_idx)
            fit_width_deg = fit_width_idx*long_res
            fit_base = (tri_ew[left_idx] + tri_ew[right_idx])/2.
            
#            print fit_base, tri_ew[left_idx], tri_ew[right_idx]
            fit_height = ma.max(tri_ew[left_idx:right_idx+1]) - fit_base
        
        elif clump_params == 0:
            left_deg = (left_idx - len(ew_data))*long_res
            right_deg = (right_idx - len(ew_data))*long_res
            gauss_center = clump.longitude
            gauss_base = clump.mexhat_base
            gauss_height = clump.mexhat_height
            gauss_sigma = clump.scale/2.
            
            fit_width_idx = abs((right_idx) - left_idx)
            fit_width_deg = fit_width_idx*long_res
    
            fit_base = (tri_ew[left_idx] + tri_ew[right_idx])/2.
            clump_int_height = np.sum(tri_ew[left_idx:right_idx+1] - fit_base)*long_res
            fit_height = ma.max(tri_ew[left_idx:right_idx+1]) - fit_base
            
    elif (options.downsample == False) or (options.voyager == False):
        left_deg, left_idx, right_deg, right_idx, clump_int_height, gauss_center, gauss_base, gauss_height, gauss_sigma= fit_gaussian(tri_ew, left_idx, right_idx, wav_center_idx)
        fit_width_idx = abs((right_idx) - left_idx)
        fit_width_deg = fit_width_idx*long_res
        fit_base = (tri_ew[left_idx] + tri_ew[right_idx])/2.
            
#        print fit_base, tri_ew[left_idx], tri_ew[right_idx]
        fit_height = ma.max(tri_ew[left_idx:right_idx+1]) - fit_base

    if right_deg > 360.:
        right_deg = right_deg - 360.
    if left_deg < 0.:
        left_deg += 360.


    return (left_deg, right_deg, fit_width_idx, fit_width_deg, fit_height, clump_int_height, gauss_center, gauss_sigma, gauss_base, gauss_height )

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
    
    wavelet_forms = ['SDG'] #['SDG', 'FDG']
    for wavelet_form in wavelet_forms:
        if wavelet_form == 'SDG':
            wavelet, mother_wavelet, wavelet_scales = sdg_wavelet_data
        if wavelet_form == 'FDG':
            wavelet, mother_wavelet, wavelet_scales = fdg_wavelet_data
        
        xform = wavelet.coefs[:,orig_ew_start_idx:orig_ew_end_idx].real # .real would need to be changed if we use a complex wavelet
        
#        print orig_ew_start_idx, orig_ew_end_idx
        scales_axis = wavelet_scales * long_res * 2 # Full-width degrees for plotting

        color_background = (1,1,1)
        color_foreground = (0,0,0)
        color_dark_grey = (0.5, 0.5, 0.5)
        color_grey = (0.375, 0.375, 0.375)
        color_bright_grey = (0.25, 0.25, 0.25)
        figure_size = (7.0,4.5)
        font_size = 18.0
        fig = plt.figure(figsize = figure_size)
    
        # Make the contour plot of the wavelet transform
        ax1 = fig.add_subplot(211)
#        ax1 = fig.add_subplot(111)
        plt.subplots_adjust(left = 0.06, right = 0.98)
#        ax1.tick_params(length = 5., width = 2., labelsize = 14. )
        ax1.yaxis.tick_left()
        ax1.xaxis.tick_bottom()
        ax1.get_yaxis().set_ticks([20.0, 40., 60., 80.])
        
        if options.color_contours:
            ax1.contourf(longitudes, scales_axis, xform, 256)
        else:
#            print len(longitudes), len(scales_axis), xform
            ct = ax1.contour(longitudes, scales_axis, xform, 32, colors='k')
#            ax1.clabel(ct, inline=1, fontsize=8)
        
        ax1.set_ylim((scales_axis[0], scales_axis[-1]))
        ax1.set_xlim((longitudes[0], longitudes[-1]))
        ax1.set_ylabel(r'Full Width Scale ( $\mathbf{^o}$)')
#        ax1.get_xaxis().set_ticks([0,360])
#        ax1.get_xaxis().set_ticks([90,180,270], minor=True)
        ax1.get_xaxis().set_ticks([])
        fig.tight_layout()
        
#        save_fig(fig,ax1, 'scalogram_contours_' + OBSID_SAMPLE_PROFILE + '.png')
        # Make the data plot with overlaid detected clumps
#        figure_size = (7.0, 2.0)
#        font_size = 18.0
    
#        fig = plt.figure(figsize = figure_size)
#        ax2 = fig.add_subplot(111)
        ax2 = fig.add_subplot(212)
        fig.tight_layout()
        plt.subplots_adjust(top = 0.9, bottom = 0.175, left = 0.060, right = 0.980)
#        ax2.plot(longitudes, ew_data, color = 'black', lw = 1.0)
#        ax2.get_yaxis().set_ticks([np.array(np.min(ew_data), np.max(ew_data) + 0.4, 0.2)])
        ax2.set_yticks([0.6, 0.8, 1.0])

        ax2.plot(longitudes, ew_data, color = '#808080', lw = 1.0)

        tripled_longitudes = np.append(longitudes-360., np.append(longitudes, longitudes+360.))
                
        for clump_data in clump_list:
#            print clump_data.wave_type, clump_data.longitude
            if clump_data.wave_type == 'SDG':
#                wavelet_color = '#841F27'  #brick red
                wavelet_color = '#808080'
            if clump_data.wave_type == 'FDG':
#                print 'plotting'
#                wavelet_color = '#0AAAC2'  #dark blue
                wavelet_color = '#404040'
            plot_one_clump(ax2, ew_data, clump_data, 0., 360., clump_color = wavelet_color)
                
        ax2.set_xlim((longitudes[0], longitudes[-1]))
        ax2.set_ylim(ma.min(ew_data), ma.max(ew_data))
#        ax2.tick_params(length = 5., width = 2., labelsize = 14. )
        ax2.yaxis.tick_left()
        ax2.xaxis.tick_bottom()
        ax2.get_xaxis().set_ticks([0,360])
        ax2.get_xaxis().set_ticks([90,180,270], minor=True)

        # align time series fig with scalogram fig
        t = ax2.get_position()
        ax2pos = t.get_points()
        ax2pos[1][0] = ax1.get_position().get_points()[1][0]
        t.set_points(ax2pos)
        ax2.set_position(t)
        ax2.set_xlabel(r'Co-Rotating Longitude ( $\mathbf{^o}$)')
        ax2.set_ylabel('Phase-Normalized E.W. (km)')
    
        save_fig(fig, ax2, 'profile_with_wavelets_' + OBSID_SAMPLE_PROFILE + '.png')
    
def plot_sample_profile():

    obs_id = OBSID_SAMPLE_PROFILE
    
    (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
     bkgnd_mask_filename, bkgnd_model_filename,
     bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obs_id)

    (ew_data_filename, ew_mask_filename) = ringutil.ew_paths(options, obs_id)

    if (not os.path.exists(reduced_mosaic_metadata_filename)) or (not os.path.exists(bkgnd_mask_filename+'.npy')):
        print 'NO FILE', reduced_mosaic_metadata_filename, 'OR', bkgnd_mask_filename+'.npy'
        assert False
    
    bkgnd_metadata_fp = open(bkgnd_metadata_filename, 'rb')
    bkgnd_data = pickle.load(bkgnd_metadata_fp)
    (row_cutoff_sigmas, row_ignore_fraction, row_blur,
     ring_lower_limit, ring_upper_limit, column_cutoff_sigmas,
     column_inside_background_pixels, column_outside_background_pixels, degree) = bkgnd_data
    bkgnd_metadata_fp.close()
    
    reduced_metadata_fp = open(reduced_mosaic_metadata_filename, 'rb')
    mosaic_data = pickle.load(reduced_metadata_fp)
    obsid_list = pickle.load(reduced_metadata_fp)
    image_name_list = pickle.load(reduced_metadata_fp)
    full_filename_list = pickle.load(reduced_metadata_fp)
    reduced_metadata_fp.close()

    (longitudes, resolutions, image_numbers,
     ETs, emission_angles, incidence_angles,
     phase_angles) = mosaic_data

    mosaic_img = np.load(reduced_mosaic_data_filename+'.npy')

    mosaic_img = mosaic_img.view(ma.MaskedArray)
    mosaic_img.mask = np.load(bkgnd_mask_filename+'.npy')
    bkgnd_model = np.load(bkgnd_model_filename+'.npy')
    bkgnd_model = bkgnd_model.view(ma.MaskedArray)
    bkgnd_model.mask = mosaic_img.mask

    corrected_mosaic_img = mosaic_img - bkgnd_model
    
    percentage_ok = float(len(np.where(longitudes >= 0)[0])) / len(longitudes) * 100
    
    mean_incidence = ma.mean(incidence_angles)
    
    ew_data = np.zeros(len(longitudes))
    ew_data = ew_data.view(ma.MaskedArray)
    movie_phase_angles = []
    for idx in range(len(longitudes)):
        if longitudes[idx] < 0 or np.sum(~corrected_mosaic_img.mask[:,idx]) == 0: # Fully masked?
            ew_data[idx] = ma.masked
        else:
            column = corrected_mosaic_img[:,idx][ring_lower_limit:ring_upper_limit+1]
            # Sometimes there is a reprojection problem at the edge
            # If the very edge is masked, then find the first non-masked value and mask it, too
            if column.mask[-1]:
                colidx = len(column)-1
                while column.mask[colidx]:
                    colidx -= 1
                column[colidx] = ma.masked
                column[colidx-1] = ma.masked
            if column.mask[0]:
                colidx = 0
                while column.mask[colidx]:
                    colidx += 1
                column[colidx] = ma.masked
                column[colidx+1] = ma.masked
            ew = np.sum(ma.compressed(column)) * options.radius_resolution
            mu_ew = ew*np.abs(np.cos(emission_angles[idx]*np.pi/180))

            corr_ew = ringutil.compute_corrected_ew(mu_ew, emission_angles[idx], mean_incidence)
            ratio = ringutil.clump_phase_curve(0) / ringutil.clump_phase_curve(phase_angles[idx])
            phase_ew = corr_ew * ratio
            
            if phase_ew < 0.2 or phase_ew > 3.5:
                ew_data[idx] = ma.masked
            else:
                ew_data[idx] = ew
            movie_phase_angles.append(phase_angles[idx])
            emission_angle = emission_angles[idx]
            incidence_angle = incidence_angles[idx]

    # When we find a masked column, mask one column on either side because this data tends to be bad too
    old_mask = ma.getmaskarray(ew_data)
    new_mask = old_mask.copy()
    for idx in range(len(new_mask)):
        if old_mask[idx]:
            if idx > 0:
                new_mask[idx-1] = True
            if idx > 1:
                new_mask[idx-2] = True
            if idx < len(new_mask)-1:
                new_mask[idx+1] = True
            if idx < len(new_mask)-2:
                new_mask[idx+2] = True
    ew_data.mask = new_mask
    
    figure_size = (7.0, 2.0)
    font_size = 18.0
    
    fig = plt.figure(figsize = figure_size)
    ax = fig.add_subplot(111)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    
    ax.set_ylabel('Equivalent Width (km)')
    ax.set_xlabel(r'Co-Rotating Longitude ( $\mathbf{^o}$)')
    ax.set_xlim(0,360.)
    ax.get_xaxis().set_ticks([0,360])
    ax.get_xaxis().set_ticks([90,180,270], minor=True)
#    print np.min(ew_data), np.max(ew_data)
#    print np.arange(np.min(ew_data), np.max(ew_data) + 0.4, 0.2)
    ax.set_ylim(1,5)
    ax.get_yaxis().set_ticks([1,5])
    ax.get_yaxis().set_ticks([2,3,4], minor=True)

    plt.plot(longitudes, ew_data, lw = 1.0, color = 'black')
    fig.tight_layout()
    save_fig(fig, ax, 'profile_' + OBSID_SAMPLE_PROFILE + '.png')


#===============================================================================
# 
#===============================================================================

def plot_sample_mosaic():
    (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
     bkgnd_mask_filename, bkgnd_model_filename, bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, OBSID_SAMPLE_PROFILE)

    mosaic_img = np.load(reduced_mosaic_data_filename + '.npy')
    mosaic_data_fp = open(reduced_mosaic_metadata_filename, 'rb')
    mosaic_data = pickle.load(mosaic_data_fp)
    
    (longitudes, resolutions,
    image_numbers, ETs, 
    emission_angles, incidence_angles,
    phase_angles) = mosaic_data
    
    rad_min = 139800
    rad_max = 140500
    rad_min_idx = int((rad_min-options.radius_start)/options.radius_resolution)
    rad_max_idx = int((rad_max-options.radius_start)/options.radius_resolution)
    
    mosaic_img = mosaic_img[rad_min_idx:rad_max_idx,:]
    
    blackpoint = max(np.min(mosaic_img), 0)
    whitepoint = np.max(mosaic_img) * 0.5
    gamma = 0.5
    # The +0 forces a copy - necessary for PIL
    scaled_mosaic = np.cast['int8'](ImageDisp.ScaleImage(mosaic_img, blackpoint,
                                                         whitepoint, gamma))[::-1,:]+0
    img = Image.frombuffer('L', (scaled_mosaic.shape[1], scaled_mosaic.shape[0]),
                           scaled_mosaic, 'raw', 'L', 0, 1)
    img.save(os.path.join(paper_root, 'mosaic_ISS_059RF_FMOVIE002_VIMS.png'), 'PNG')
    
#===============================================================================
# 
#===============================================================================

def plot_poisson():    
    def poisson(k,mean):
        return mean**k * np.exp(-mean) / factorial(k)

    fig = plt.figure(figsize = (3.5,2))
    ax = fig.add_subplot(111)

    mean = np.arange(0., 3., 0.001)
    
    ########################
    # Voyager
    ########################
    
    # Probability of observing >= 2 clumps twice
    
    # Probabiliy of (not) observing exactly 0 clumps or exactly 1 clumps
    prob_ge2 = 1. - poisson(0,mean) - poisson(1,mean)
    
    # Probability of observing two or more independent clumps
    prob_vgr = prob_ge2**2

    print 'Voyager probability', prob_vgr
        
    ########################
    # Cassini
    ########################
    
    prob_eq0 = poisson(0,mean)
    prob_eq1 = poisson(1,mean)
    prob_eq2 = poisson(2,mean)
    
    for n in range(5,11):
    
        # Probability of never observing a bright clump
        # 0 clumps in each of N slots
        prob_never_of_n = prob_eq0**n
    
        # Probability of observing exactly one bright clump
        # 1 clump in the first slot, then 0 clumps...or...
        # 1 clump in the second slot, otherwise 0...or...
        # N times
        prob_once_of_n = n * prob_eq0**(n-1) * prob_eq1
    
        # Probability of observing exactly two bright clumps
        # 1 clump in first slot, then 1 clump in second slot, then 0s...or...
        # 1 clump in first slot, then 1 clump in third slot, then 0s...or...
        # First clump can be in one of N locations
        # Second clump can be in one of N-1 locations but divide by 2 because
        #   the two clumps are interchangeable
        # Zeros in remaining locations
        # We can also have two clumps at the same time in any of N slots
        prob_twice_of_n = n*(n-1)/2. * prob_eq0**(n-2) * prob_eq1**2 + \
                          n * prob_eq0**(n-1) * prob_eq2
    
        prob_cas = prob_never_of_n + prob_once_of_n + prob_twice_of_n
    
        if n == 5:
            plt.plot(mean, prob_vgr * prob_cas, '-', label='$N='+str(n)+'$', color='black')
        else:
            plt.plot(mean, prob_vgr * prob_cas, ':', label='$N='+str(n)+'$', color='black',
                     dashes=((n-4),(n-4)))
            
    
    plt.xlabel('Expected Number of Simultaneous Bright ECs ($\lambda$)')
    plt.ylabel('Probability')
    ax.set_yticks([0,.002,.004,.006,.008,.01])
    leg = plt.legend(handlelength=3.45)

    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
    save_fig(fig, ax, 'lambda_significance.png', leg)    


#===============================================================================
################################################################################
#
# STUFF RELATED TO SINGLE CLUMPS
#
################################################################################
#===============================================================================

#def compute_clump_attributes(clump):
#    ew_data = clump.clump_db_entry.ew_data
#    long_res = 360. / len(ew_data)
#    dbl_ew_data = np.tile(ew_data, 2)
#    left_idx = int(clump.fit_left_deg / long_res)
#    right_idx = int(clump.fit_right_deg / long_res)
#    if left_idx > right_idx:
#        right_idx += len(ew_data) 
#    left_val = dbl_ew_data[left_idx]
#    right_val = dbl_ew_data[right_idx]
#    ctr_val = np.max(dbl_ew_data[left_idx:right_idx+1])
#    left_height = ctr_val - left_val
#    right_height = ctr_val - right_val
#    if left_val == ctr_val or right_val == ctr_val:
#        height = 0.
#    else:
#        height = ctr_val-(left_val+right_val)/2
#    if right_height > left_height:
#        asym_ratio = right_height / left_height
#    else:
#        asym_ratio = left_height / right_height
#    width = right_idx*long_res - left_idx*long_res
#    return height, width, asym_ratio
#
#def limit_single_clumps(clump_list):
#    new_clump_list = []
#    for clump in clump_list:
#        if clump.clump_db_entry is None:
#            print 'NO DB'
#            clump.print_all()
#            assert False
#        if clump.int_fit_height < 0:
#            if debug:
#                print 'NEG INT HEIGHT'
#            continue 
#        ew_data = clump.clump_db_entry.ew_data
#        long_res = 360. / len(ew_data)
#        left_idx = int(clump.fit_left_deg / long_res)
#        right_idx = int(clump.fit_right_deg / long_res)
#        if ((left_idx >= 0 and left_idx < len(ew_data) and ma.count_masked(ew_data[left_idx]) == 1) or
#            (right_idx >= 0 and right_idx < len(ew_data) and ma.count_masked(ew_data[right_idx]) == 1)):
#            if debug:
#                print 'Edge is masked'
#            continue
#        height, width, asym_ratio = compute_clump_attributes(clump)
#        if debug:
#            print 'Height', height
##        if height < 0.1:# or height > 2: # XXX
##            continue
#        if width < 3.5 or width > 40:
#            continue
#        if debug:
#            print 'Asym', asym_ratio
#        if asym_ratio > 5:
#            continue
#    
#        new_clump_list.append(clump)
#        
#    return new_clump_list
#
#def choose_correct_single_clumps(clump_db):
#    for obsid in sorted(clump_db):
#        if debug:
#            print obsid
#        clump_db_entry = clump_db[obsid]
#        new_list = []
#        restr_clump_list = limit_single_clumps(clump_db_entry.clump_list)
#        restr_clump_list.sort(key=lambda x: x.fit_left_deg)
#        for clump_num, clump in enumerate(restr_clump_list):
#            clump_left_deg = clump.fit_left_deg
#            clump_right_deg = clump.fit_right_deg
#            if clump_right_deg < clump_left_deg:    
#                clump_right_deg += 360
#            found_match = False
#            for sec_clump_num, sec_clump in enumerate(restr_clump_list):
#                if clump_num == sec_clump_num:
#                    if debug:
#                        print 'SAME'
#                    continue
#                sec_clump_left_deg = sec_clump.fit_left_deg
#                sec_clump_right_deg = sec_clump.fit_right_deg
#                sec_clump_left_deg2 = sec_clump_left_deg
#                sec_clump_right_deg2 = sec_clump_right_deg
#                
#                if sec_clump_right_deg < sec_clump_left_deg:    
#                    sec_clump_right_deg += 360
#                    sec_clump_left_deg2 -= 360
#                if debug:
#                    print '%7.2f %7.2f %7.2f %7.2f W %7.2f %7.2f' % (clump_left_deg, clump_right_deg,
#                                                       sec_clump_left_deg, sec_clump_right_deg,
#                                                       clump_right_deg-clump_left_deg,
#                                                       sec_clump_right_deg-sec_clump_left_deg) 
#                if sec_clump_left_deg == clump_left_deg and sec_clump_right_deg == clump_right_deg:
#                    if clump_num > sec_clump_num:
#                        if debug:
#                            print 'IDENTICAL'
#                        found_match = True
#                        break
#                if (abs(sec_clump_left_deg-clump_left_deg) <= 2 and
#                    abs(sec_clump_right_deg-clump_right_deg) <= 2):
#                    # Identical within 2 deg - let the larger one go through
#                    if sec_clump_right_deg - sec_clump_left_deg > clump_right_deg - clump_left_deg:
#                        if debug:
#                            print 'CLOSE AND SMALLER'
#                        found_match = True
#                        break
#                if (choose_smaller and 
#                    ((sec_clump_left_deg < clump_right_deg and sec_clump_right_deg > clump_left_deg) or
#                     (sec_clump_left_deg2 < clump_right_deg and sec_clump_right_deg2 > clump_left_deg))):
#                    if sec_clump_right_deg - sec_clump_left_deg < clump_right_deg - clump_left_deg:
#                        if debug:
#                            print 'ENCLOSED CHOOSING SMALLER'
#                        found_match = True
#                        break
#                if (choose_larger and 
#                    ((sec_clump_left_deg < clump_right_deg and sec_clump_right_deg > clump_left_deg) or
#                     (sec_clump_left_deg2 < clump_right_deg and sec_clump_right_deg2 > clump_left_deg))):
#                    if sec_clump_right_deg - sec_clump_left_deg > clump_right_deg - clump_left_deg:
#                        if debug:
#                            print 'ENCLOSED CHOOSING LARGER'
#                        found_match = True
#                        break
#            if not found_match:
#                if debug:
#                    if clump_right_deg-clump_left_deg == 36.5:
#                        print 'KEEPING CLUMP'
#                new_list.append(clump)
#                
#        clump_db_entry.clump_list = new_list
##        if obsid == 'ISS_031RF_FMOVIE001_VIMS' or obsid == 'ISS_055RI_LPMRDFMOV001_PRIME' or obsid == 'ISS_051RI_LPMRDFMOV001_PRIME' or obsid == 'ISS_134RI_SPKMVDFHP002_PRIME' or obsid == 'ISS_033RF_FMOVIE001_VIMS':
##        plot_single_clumps(clump_db_entry, new_list, obsid)

#------------------------------------------------------------------------------------------------

voyager_clump_db_path = os.path.join(paper_root, 'voyager_clumpdb_137500_142500_05.000_0.020_10_02_137500_142500.pickle')
print 'VOYAGER ALL CLUMP DB:', voyager_clump_db_path
v_clump_db_fp = open(voyager_clump_db_path, 'rb')
clump_find_options = pickle.load(v_clump_db_fp)
v_all_clump_db = pickle.load(v_clump_db_fp)
v_clump_db_fp.close()
for v_obs in v_all_clump_db.keys(): # Fix masking
    v_all_clump_db[v_obs].ew_data[np.where(v_all_clump_db[v_obs].ew_data == 0.)] = ma.masked

cassini_clump_db_path = os.path.join(paper_root, 'downsampled_clumpdb_137500_142500_05.000_0.020_10_01_137500_142500.pickle')
print 'CASSINI DOWNSAMPLED ALL CLUMP DB:', cassini_clump_db_path
c_clump_db_fp = open(cassini_clump_db_path, 'rb')
clump_find_options = pickle.load(c_clump_db_fp)
c_all_clump_db = pickle.load(c_clump_db_fp)
c_clump_db_fp.close()

#XXX choose_correct_single_clumps(v_all_clump_db)
#XXX choose_correct_single_clumps(c_all_clump_db)

c_approved_list_fp = os.path.join(paper_root, 'approved_list_w_errors.pickle')
print 'CASSINI APPROVED CLUMP DB:', c_approved_list_fp
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()

#===============================================================================
# Things related to approved clumps 
#===============================================================================

# We MUST do this or else the clump_num field won't be initialized
#dump_clump_table(c_approved_list, c_approved_db)
#dump_obs_table(c_approved_list, c_approved_db)

### Clump profiles ###
#XXX plot_appearing_and_disappearing_clumps(['C38'], ['C16'], [], ['C21', 'C26'])
#XXX plot_appearing_and_disappearing_clumps([], [], [], ['C63'])

#plot_appearing_and_disappearing_clumps(['C33'], ['C12'], [], ['C17', 'C22'])
#plot_clump_matrix(['C2', 'C6', 'C9', 'C13', 'C15', 'C30', 'C31', 'C35', 'C39'])
#plot_029rf_over_time()
plot_combined_vel_hist()

#===============================================================================
# Things not related to approved or single clumps at all 
#===============================================================================

### These don't change - don't need to regenerate very often ###
#plot_FDG()
#plot_mexhat()
#plot_long_coverage_over_time()
#plot_sample_scalogram()
#plot_sample_profile()
#plot_voyager_cassini_comparison_profiles()
#plot_phase_curves()
#plot_sample_mosaic()

#plot_poisson()
