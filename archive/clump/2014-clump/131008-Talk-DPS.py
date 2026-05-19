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
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pylab
import scipy.interpolate as interp
from imgdisp import ImageDisp
import Image
import string
import ringimage
import scipy.stats.distributions as scipydist

debug = False
choose_smaller = False
choose_larger = True

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = [
#                '--prefilter',
                '--scale-min', '5', '--scale-max', '80',
                '--clump-size-min', '5', '--clump-size-max', '49.9',
                '--ignore-bad-obsids',
                '--save-clump-database',
                '--plot-scalogram',
                '--color-contours',
                '--mosaic-reduction-factor', '1',
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

paper_root = os.path.join(ringutil.PAPER_ROOT, 'Talk-DPS2013')
root_clump_db = {}

profile_color = 'black'

FRING_MEAN_MOTION = 581.964
OBSID_SAMPLE_PROFILE = 'ISS_059RF_FMOVIE002_VIMS'

color_voyager = '#7BCF70'
color_cassini = '#D1A264'
color_wavelet = '#F74F4F'
color_wavelets = ('#F74F4F', '#e08e24')
color_clump = (0xf7/256., 0x4f/256., 0x4f/256.)
color_profile = '#ffffff'
color_axis = '#d0d0d0'
color_text = '#d0d0d0'
color_poisson = '#00a0a0'
lw_poisson = 4
lw_profile = 2
lw_clump = 3
lw_wavelet = 3
ms_scatter = 20
fullsize_figure = (8,5)

color_background = (0,0,0)
color_foreground = (1,1,1)
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
matplotlib.rc('axes', facecolor=color_background, edgecolor=color_axis, labelcolor=color_text)
matplotlib.rc('xtick', color=color_axis, labelsize=20)
matplotlib.rc('xtick.major', size=8)
matplotlib.rc('xtick.minor', size=6)
matplotlib.rc('ytick', color=color_axis, labelsize=20)
matplotlib.rc('ytick.major', size=8)
matplotlib.rc('ytick.minor', size=6)
matplotlib.rc('font', size=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('text', color=color_text)

def fix_graph_colors(fig, ax, ax2, legend):
    for line in ax.xaxis.get_ticklines() + ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines() + ax.yaxis.get_ticklines(minor=True):
        line.set_color(color_foreground)
    if legend != None:
        legend.get_frame().set_facecolor(color_background)
        legend.get_frame().set_edgecolor(color_background)
        for text in legend.get_texts():
            text.set_color(color_text) 

def save_fig(fig, ax, fn,ax2 = None, legend=None):
    fix_graph_colors(fig, ax, ax2, legend)
    fn = os.path.join(paper_root,fn)
    print 'Saving', fn
    plt.savefig(fn, bbox_inches='tight', facecolor=color_background)   
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
    ax.plot(idx_range, mexhat, '-', color=ncolor, lw=lw_wavelet, alpha=0.9, label=legend)
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
    ax.plot(long_range, ew_range, '-', label=legend, color= color, lw = lw_profile)
    
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
        ax.plot(longitudes[left_idx:len(ew_data)-1], tri_ew[left_idx:len(ew_data)-1], color = color, lw = lw_wavelet)
        ax.plot(longitudes[len(ew_data):right_idx], tri_ew[len(ew_data):right_idx], color = color, lw = lw_wavelet)
    else:
        ax.plot(idx_range, tri_ew[left_idx:right_idx], color = color, lw = lw_wavelet)

def convert_angle(b_angle, s_angle, h_angle,m_angle,clump_width,min_ew, max_ew):
    
    base = min_ew*(np.sin(b_angle)+1) + min_ew
    scale = 0.5*clump_width*(np.sin(s_angle)+1) + 0.25*clump_width
    height = max_ew*(np.sin(h_angle)+1)
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
        h_angle, s_angle, b_angle, m_angle = params

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
    old_clump_ews = tri_ews[clump_left_idx:clump_right_idx+1]
    if len(old_clump_ews) < 3:                  #sometimes the downsampled versions don't end up with enough data to fit a gaussian to the arrays
        return(0)

    x = np.arange(clump_left_idx-len_ew_data, clump_right_idx- len_ew_data +1)*long_res

    min_ew = np.min(old_clump_ews)
    max_ew = np.max(old_clump_ews)
    
    
    clump_data, residual, array, trash, trash, trash  = sciopt.fmin_powell(fitting_func, (0.,0.,0., 0.),
#                                                                               (np.pi/64., np.pi/64., np.pi/64., np.pi/64.),
                                                                       args=(x, old_clump_ews, clump_half_width_idx*long_res, clump_longitude, min_ew, max_ew),
                                                                       ftol = 1e-8, xtol = 1e-8,disp=False, full_output=True)

    h_angle, s_angle, b_angle, m_angle = clump_data
    base, sigma, height, center_offset = convert_angle(b_angle, s_angle, h_angle, m_angle, clump_half_width_idx*long_res, min_ew, max_ew)
    
    center = clump_longitude + center_offset                    #make this a multiple of our longitude resolution

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
    figure_size = (3.5,2.)
    
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
    
    pylab.text(-10.5, -0.8, 'Scale', {'color' : 'k', 'fontsize' : 10})
    ax.arrow(0.,-1.025,23., 0.0, head_width = 0.15, head_length = 2.5, fc = 'k', ec = 'k', length_includes_head = True)
    ax.arrow(0,-1.025,-23., 0.0, head_width = 0.15, head_length = 2.5, fc = 'k', ec = 'k', length_includes_head = True) #draw the other side
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
    figure_size = (3.5,2.)
    
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
    
    pylab.text(-10.5, 0.1, 'Scale', {'color' : 'k', 'fontsize' : 10})
    ax.arrow(0.,0.0,23., 0.0, head_width = 0.05, head_length = 2.5, fc = 'k', ec = 'k', length_includes_head = True)
    ax.arrow(0,0.0,-23., 0.0, head_width = 0.05, head_length = 2.5, fc = 'k', ec = 'k', length_includes_head = True) #draw the other side
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
    sorted_ids = clumputil.get_sorted_obsid_list(c_approved_db)
    for obsid in sorted_ids:
        ew_data = c_approved_db[obsid].ew_data
        ew_mask = ew_data.mask
        long_res = 360./len(ew_data)
        longitudes = np.arange(0,360.,long_res)
        longitudes = longitudes.view(ma.MaskedArray)
        longitudes.mask = ew_mask
        time = np.zeros(len(longitudes)) + c_approved_db[obsid].et_max
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
    
    def fitting_func(params, seed, ew_data, center_long_idx, clump_half_width_idx):
        co_angle, h_angle, s_angle, b_angle = params
        
        base, scale, height, center_offset_idx = convert_angle(b_angle, s_angle, h_angle, co_angle, clump_half_width_idx)
        center_idx = center_long_idx + center_offset_idx
        #extract data from the center to a scale's width to either side
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
        
    master_list_filename = os.path.join(ringutil.ROOT, 'clump-data', 'master_clumps_list.pickle')
    master_list_fp = open(master_list_filename, 'rb')
    master_clump_db, master_chain_list = pickle.load(master_list_fp)
    master_list_fp.close()

    clump_list = []
    for chain in master_chain_list:
        for clump in chain.clump_list:
            clump_list.append(clump)
    
    clump_list = np.array(clump_list)
    keep_list = [1,3,4,5,9,10,11,12,13,16,18,19,21]
    clump_list = clump_list[keep_list]
    print 'MEAN CLUMP: Num clumps', len(clump_list)
    
    seed_idx = 0
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
        new_clump_ews = ew_data[center-scale:center+scale+1]
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
    
    fig = plt.figure(figsize = (3.5,3.5))
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
    save_fig(fig, ax, 'mean_clump.png', legend=leg)


#===============================================================================
#
# PLOT MATRIX OF EXAMPLE CLUMPS
#
#===============================================================================

def plot_clump_matrix(matrix_clumps):
    matrix_list = []
    
    for clump_num in matrix_clumps:
        for chain in c_approved_list:
            if chain.clump_num == clump_num:
                matrix_list.append(chain)
                break
             
    m_db = {}
    m_list = []
    time_list = [chain.clump_list[0].clump_db_entry.et_max for chain in matrix_list]
    clump_list = [chain.clump_list[0] for chain in matrix_list]
    for clump in clump_list:
        print clump.clump_db_entry.obsid, clump.g_center
        time = clump.clump_db_entry.et_max
        if time not in m_db.keys():
            m_db[time] = []
            m_db[time].append(clump)
        elif time in m_db.keys():
            longs = np.array([aclump.g_center for aclump in m_db[time]])
            idx = np.where(longs > clump.g_center)[0]
            print longs, idx
            if len(idx) == 0:
                m_db[time].append(clump)
            if len(idx) > 0:
                m_db[time].insert(idx[0], clump)
            print [bclump.g_center for bclump in m_db[time]]
    for time in sorted(m_db.keys()):
        for clump in m_db[time]:
            m_list.append(clump)
            
    for clump in m_list:
        print clump.clump_db_entry.obsid, clump.g_center

    y_ax = 3
    x_ax = 7
    fig = plt.figure(1, (7.0, 5.0))
    grid = gridspec.GridSpec(4,3)
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
        xstep = np.round((x_right-x_left)/5.)
        ax.set_xlim(x_left, x_right)
        xticks = np.arange(x_left + xstep, x_right + xstep, xstep)
        
        print x_left, x_right, xticks
        xticks.astype(int)
        if len(xticks) > 5.:
            xticks = xticks[:-1]
        ax.get_xaxis().set_ticks(xticks)
        
        ax.xaxis.get_major_ticks()[-1].label1.set_visible(False)
        
        y_max = np.max(tri_ews[left_idx_max:right_idx_max+1])
        y_min = np.min(tri_ews[left_idx_max:right_idx_max+1])
        ax.set_ylim(y_min, y_max)
        y_step = (y_max - y_min)/3.
        ax.get_yaxis().set_ticks([y_min, y_min + 1*y_step, y_min + 2*y_step, y_min + 3*y_step])
        
        xFormatter = FormatStrFormatter('%d')
        yFormatter = FormatStrFormatter('%.1f')
        ax.yaxis.set_major_formatter(yFormatter)
        ax.xaxis.set_major_formatter(xFormatter)
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
            
        ax.set_ylim(np.min(tri_ews[left_idx_max:right_idx_max]), np.max(tri_ews[left_idx_max:right_idx_max]))
        if i == y_ax:
            ax.set_ylabel('Normalized Equivalent Width (km)')
        if i == x_ax:
            ax.set_xlabel('Co-Rotating Longitude ( $\mathbf{^o}$)')
    
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
                
                ax.get_xaxis().set_ticks(np.arange(0, 360.+45., 45.))
                y_min = np.min(clump.clump_db_entry.ew_data)
                y_max = np.max(clump.clump_db_entry.ew_data)
                ystep = (y_max-y_min)/4.
                ax.get_yaxis().set_ticks(np.arange(y_min, y_max + ystep, ystep )) 
                if i < len(chain.clump_list)-1:
                    ax.set_xticklabels('')
                
                fix_graph_colors(fig, ax, None, None)
            
                yFormatter = FormatStrFormatter('%.1f')
                ax.yaxis.set_major_formatter(yFormatter)    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    ax.get_xaxis().set_ticks(np.arange(0, 360.+45., 45.)) 
    ax.set_xlabel('Co-Rotating Longitude ( $\mathbf{^o}$)', fontsize = 10)
    plt.figtext(0.01, .66, 'Normalized Equivalent Width', rotation = 'vertical', fontsize = 10)
    plt.savefig(os.path.join(paper_root,'clump_progression_ISS_029RF.png'), bbox_inches='tight', facecolor=color_background, dpi=1000)


#===============================================================================
#
# PLOT APPEARING AND DISAPPEARING AND PLAIN OLD-FASHIONED CLUMPS OVER TIME
# 
#===============================================================================

def plot_appearing_and_disappearing_clumps(appearing_clump_nums, disappearing_clump_nums, basic_clump_nums):
    def draw_clumps(im, left_idx, right_idx, color, radius_center):
        height = 40 #pixels
        l_thick = 10
        w_thick = 10
        
        for i in range(len(color)):
            im[radius_center + height:radius_center + height + l_thick, left_idx:right_idx +l_thick, i] = color[i]*256
            im[radius_center - height - l_thick:radius_center - height, left_idx:right_idx +l_thick, i] = color[i]*256
            im[radius_center - height - l_thick: radius_center + height + l_thick,left_idx:left_idx+w_thick, i] = color[i]*256
            im[radius_center - height - l_thick: radius_center + height + l_thick, right_idx:right_idx + w_thick, i] = color[i]*256
    
    def draw_entire_plot(clump_list):
        fig = plt.figure(figsize = (10.0, 6)) #1.2*len(clump_list)+0.2))
        num_axes = len(clump_list) + 1
        y_abs_max = []
        mosaic_max = []
        mosaic_clips = []
        mosaic_clumps = []
        
        xlefts = [clump.fit_left_deg for clump in clump_list]
        xrights = [clump.fit_right_deg for clump in clump_list]
        
        x_min_left = np.min(xlefts) - 5.
        x_max_right = np.max(xrights) + 5.
        x_min_left = np.floor(x_min_left/5.)*5.
        x_max_right = np.ceil(x_max_right/5.)*5.
        
        if len(clump_list) == 3:
            ypos_time = 0.80
            ypos_ylabel = .85
            ypos_xlabel = 0.15
            text_ylabel = 'Normalized E.W. (km)'
        elif len(clump_list) == 5:
            ypos_time = 0.77
            ypos_ylabel = .83
            ypos_xlabel = 0.11
            text_ylabel = 'Normalized Equivalent Width (km)'
        elif len(clump_list) == 6:
            ypos_time = 0.77
            ypos_ylabel = .78
            ypos_xlabel = 0.1
            text_ylabel = 'Normalized Equivalent Width (km)'
            
        i = 1
        for clump in clump_list:
            ax = fig.add_subplot(num_axes, 2, i)
            clump_ew_data = clump.clump_db_entry.ew_data
            long_res = 360./len(clump_ew_data)
            i += 1
            ax.set_xlim(x_min_left, x_max_right)

            clump_extract = clump_ew_data[x_min_left/long_res: x_max_right/long_res]
            if ma.any(clump_extract):
                ymin = np.min(clump_extract)
                ymax = np.max(clump_extract) - ymin
            else:
                ymin = 0
                ymax = 1
            ymin = np.min(clump_extract)-0.04
            ymax = np.max(clump_extract)+0.04
            y_abs_max.append(ymax)
            ystep = (ymax)/4.
            ytick1 = .8*(ymax-ymin)+ymin
            ytick2 = .2*(ymax-ymin)+ymin
            ytick1 = np.round(ytick1*100)/100
            ytick2 = np.round(ytick2*100)/100
            ax.set_ylim(ymin, ymax)
            ax.get_yaxis().set_ticks([])
            ax.yaxis.tick_left()
            ax.xaxis.tick_bottom()
            xFormatter = FormatStrFormatter('%d')
            yFormatter = FormatStrFormatter('%.2f')
            ax.yaxis.set_major_formatter(yFormatter)
            ax.xaxis.set_major_formatter(xFormatter)
            plt.setp(ax.get_xticklabels(), visible=False)
            time = cspice.et2utc(clump.clump_db_entry.et_min, 'C', 0)[:11]
            ax.text(0.015, ypos_time, time, transform = ax.transAxes, size=14)
            xticks = np.arange(x_min_left, x_max_right + 5., 5)
            ax.get_xaxis().set_ticks(xticks)
            ax.set_xticklabels(['']+[str(int(xticks[1]))]+['']*(len(xticks)-4)+[str(int(xticks[-1]))]+[''])
            
            plot_single_ew_profile(ax, clump_ew_data, clump.clump_db_entry, 0.,360., color = color_profile)
            if not clump.ignore_for_chain:
                plot_fitted_clump_on_ew(ax, clump.clump_db_entry.ew_data, clump, color = color_clump)
            
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
            
            mu = ringutil.mu(c_approved_db[obsid].emission_angle)
            mosaic_img = mosaic_img*mu
            m_long_res = 360./mosaic_img.shape[1]

            #make mosaic caption
            ax2 = fig.add_subplot(num_axes, 2, i)
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.setp(ax2.get_yticklabels(), visible=False)
            xticks = np.arange((x_min_left)/m_long_res, (x_max_right)/m_long_res + 5./m_long_res, 5./m_long_res)
            if len(xticks) > 0:
                ax2.set_xticks(xticks - xticks[0])
                ax2.set_xticklabels(['']+[str(int(xticks[1]*m_long_res))]+['']*(len(xticks)-4)+[str(int(xticks[-1]*m_long_res))]+[''])
                ax2.tick_params(axis = 'x', direction = 'out', length = 2.0)
                ax2.xaxis.tick_bottom()
            ax2.set_yticks([])            
            
            mosaic_extract = mosaic_img[400:650, (x_min_left -10.)/m_long_res: (x_max_right +10.)/m_long_res + 1]
            if ma.any(mosaic_extract):
                mosaic_max.append(ma.max(mosaic_extract))
            color_mosaic = np.zeros((mosaic_img.shape[0], mosaic_img.shape[1], 3))
            color_mosaic[:,:,0] = mosaic_img
            color_mosaic[:,:,1] = mosaic_img
            color_mosaic[:,:,2] = mosaic_img
            mosaic_clip = color_mosaic[400:650, x_min_left/m_long_res: x_max_right/m_long_res + 1, :]
            
            mosaic_clips.append(mosaic_clip)
            radius_center = (140220-radius_start)/radius_res - 400
            mosaic_clumps.append((clump,
                                  clump.fit_left_deg/m_long_res-x_min_left/m_long_res,
                                  clump.fit_right_deg/m_long_res-x_min_left/m_long_res,
                                  radius_center))
            
            i+=1

#        ax.set_xlabel('X', fontsize = 10, alpha=0)
#        ax.set_ylabel('X', fontsize = 10, alpha=0)
        
        total_axes = len(fig.axes)
            
        #rescale all of the mosaics
        if not ma.any(mosaic_max):
            print 'BAD MAX'
            mosaic_max = 1.
        else:
            mosaic_max = ma.max(mosaic_max)
            even_axes = range(total_axes)[1::2]
            for l, ax_num in enumerate(even_axes):
                ax2 = fig.axes[ax_num]
                mosaic_clip = mosaic_clips[l]
                mode = 'RGB'
                final_im = ImageDisp.ScaleImage(mosaic_clip, blackpoint, mosaic_max*0.25, gamma)+0
                final_im = np.cast['int8'](final_im)
                clump, left_idx, right_idx, radius_center = mosaic_clumps[l]
                if not clump.ignore_for_chain:
                    color = color_clump
                    draw_clumps(final_im, left_idx, right_idx, color, radius_center)
                    
                final_img = Image.frombuffer(mode, (final_im.shape[1], final_im.shape[0]),
                                       final_im, 'raw', mode, 0, 1)
                ax2.imshow(final_img, aspect = 'auto')
            
        fig.tight_layout()
        fig.subplots_adjust(hspace = 0.1, wspace = 0.08)
        plt.setp(ax.get_xticklabels(), visible=True)
        plt.setp(ax2.get_xticklabels(), visible=True)
        plt.setp(ax.get_yticklabels(), visible = True)

        ax2.tick_params(axis = 'x', direction = 'in', length = 2.0)
#        plt.figtext(0.4, ypos_xlabel, 'Co-Rotating Longitude ( $\mathbf{^o}$)', fontsize = 10)
#        plt.figtext(0.01, ypos_ylabel, text_ylabel, rotation = 'vertical', fontsize = 10)

        
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
    
    obsid_list = ['ISS_000RI_SATSRCHAP001_PRIME','ISS_059RF_FMOVIE002_VIMS','ISS_075RF_FMOVIE002_VIMS','ISS_134RI_SPKMVDFHP002_PRIME', 
                  'ISS_055RI_LPMRDFMOV001_PRIME', 'ISS_00ARI_SPKMOVPER001_PRIME']
    
    color_background = (1,1,1)
    font_size = 18.0
        
    fig = plt.figure(figsize = (11,6))
    ax = fig.add_subplot(111)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()

    ax.set_ylabel('Normalized Equivalent Width')
    ax.set_xlabel('Co-Rotating Longitude ( $\mathbf{^o}$)')
    ax.set_xlim(0,360.)
    ax.get_xaxis().set_ticks([0,360])
    ax.get_yaxis().set_ticks([])
    
    for obs_id in obsid_list:
        c_ew_data = c_approved_db[obs_id].ew_data
        longitudes = np.arange(0.,360., 360./len(c_ew_data))
        cassini = plt.plot(longitudes, c_ew_data, color =color_cassini , lw = lw_profile, alpha = 1)
        
    for v_obs in v_all_clump_db.keys():
        v_ew_data = v_approved_db[v_obs].ew_data 
        v_ew_data = v_ew_data.view(ma.MaskedArray)
        v_mask = ma.getmaskarray(v_ew_data)
        empty = np.where(v_ew_data == 0.)[0]
        if empty != ():
            v_mask[empty[0]-5:empty[-1]+5] = True
        v_ew_data.mask = v_mask
        longitudes = np.arange(0.,360., 360./len(v_ew_data))
        voyager = plt.plot(longitudes, v_ew_data, color = color_voyager, lw = lw_profile, alpha = 0.9)
            
    leg = ax.legend([voyager[0], cassini[0]], ['Voyager', 'Cassini'], loc = 1)
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
      
    save_fig(fig, ax, 'voyager_cassini_profile_comparison.png', legend=leg)

    obsid_list = ['ISS_000RI_SATSRCHAP001_PRIME','ISS_059RF_FMOVIE002_VIMS','ISS_075RF_FMOVIE002_VIMS','ISS_134RI_SPKMVDFHP002_PRIME', 
                  'ISS_055RI_LPMRDFMOV001_PRIME', 'ISS_00ARI_SPKMOVPER001_PRIME',
                  'ISS_036RF_FMOVIE001_VIMS', #'ISS_036RF_FMOVIE002_VIMS',
                  'ISS_039RF_FMOVIE001_VIMS', 'ISS_039RF_FMOVIE002_VIMS']
    
    color_background = (1,1,1)
    font_size = 18.0
        
    fig = plt.figure(figsize = (11,6))
    ax = fig.add_subplot(111)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()

    ax.set_ylabel('Normalized Equivalent Width')
    ax.set_xlabel('Co-Rotating Longitude ( $\mathbf{^o}$)')
    ax.set_xlim(0,360.)
    ax.get_xaxis().set_ticks([0,360])
    ax.get_yaxis().set_ticks([])
    
    for obs_id in obsid_list:
        c_ew_data = c_approved_db[obs_id].ew_data
        longitudes = np.arange(0.,360., 360./len(c_ew_data))
        cassini = plt.plot(longitudes, c_ew_data, color =color_cassini , lw = lw_profile, alpha = 1)
        
    for v_obs in v_all_clump_db.keys():
        v_ew_data = v_approved_db[v_obs].ew_data 
        v_ew_data = v_ew_data.view(ma.MaskedArray)
        v_mask = ma.getmaskarray(v_ew_data)
        empty = np.where(v_ew_data == 0.)[0]
        if empty != ():
            v_mask[empty[0]-5:empty[-1]+5] = True
        v_ew_data.mask = v_mask
        longitudes = np.arange(0.,360., 360./len(v_ew_data))
        voyager = plt.plot(longitudes, v_ew_data, color = color_voyager, lw = lw_profile, alpha = 0.9)
            
    leg = ax.legend([voyager[0], cassini[0]], ['Voyager', 'Cassini'], loc = 1)
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
      
    save_fig(fig, ax, 'voyager_cassini_profile_comparison_monster.png', legend=leg)

    obsid_list = ['ISS_000RI_SATSRCHAP001_PRIME','ISS_059RF_FMOVIE002_VIMS','ISS_075RF_FMOVIE002_VIMS','ISS_134RI_SPKMVDFHP002_PRIME', 
                  'ISS_055RI_LPMRDFMOV001_PRIME', 'ISS_00ARI_SPKMOVPER001_PRIME',
                  'ISS_106RF_FMOVIE002_PRIME', 'ISS_107RF_FMOVIE002_PRIME', 'ISS_108RI_SPKMVLFLP001_PRIME']
    
    color_background = (1,1,1)
    font_size = 18.0
        
    fig = plt.figure(figsize = (11,6))
    ax = fig.add_subplot(111)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()

    ax.set_ylabel('Normalized Equivalent Width')
    ax.set_xlabel('Co-Rotating Longitude ( $\mathbf{^o}$)')
    ax.set_xlim(0,360.)
    ax.get_xaxis().set_ticks([0,360])
    ax.get_yaxis().set_ticks([])
    
    for obs_id in obsid_list:
        c_ew_data = c_approved_db[obs_id].ew_data
        longitudes = np.arange(0.,360., 360./len(c_ew_data))
        cassini = plt.plot(longitudes, c_ew_data, color =color_cassini , lw = lw_profile, alpha = 1)
        
    for v_obs in v_all_clump_db.keys():
        v_ew_data = v_approved_db[v_obs].ew_data 
        v_ew_data = v_ew_data.view(ma.MaskedArray)
        v_mask = ma.getmaskarray(v_ew_data)
        empty = np.where(v_ew_data == 0.)[0]
        if empty != ():
            v_mask[empty[0]-5:empty[-1]+5] = True
        v_ew_data.mask = v_mask
        longitudes = np.arange(0.,360., 360./len(v_ew_data))
        voyager = plt.plot(longitudes, v_ew_data, color = color_voyager, lw = lw_profile, alpha = 0.9)
            
    leg = ax.legend([voyager[0], cassini[0]], ['Voyager', 'Cassini'], loc = 1)
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
      
    save_fig(fig, ax, 'voyager_cassini_profile_comparison_2009.png', legend=leg)


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
    
    fig = plt.figure(figsize = (10,4))
    ax_mm = SubplotHost(fig, 1,1,1)
    
    a0 = RelativeRateToSemimajorAxis(-1./86400) # -1 deg/day
    a1 = RelativeRateToSemimajorAxis(1./86400)  # +1 deg/day
    slope = (a1-a0) / 2.

    aux_trans = mtransforms.Affine2D().translate(-220,0)
    aux_trans.scale(1/slope)
    ax_sma = ax_mm.twin(aux_trans)
    ax_sma.set_viewlim_mode("transform")

    fig.add_subplot(ax_mm)

#    plt.subplots_adjust(top = .85, bottom = 0.15, left = 0.08, right = 0.98)
    
    ax_mm.set_xlabel('Relative Mean Motion ( $\mathbf{^o}$/Day )')
    ax_sma.set_xlabel('Semimajor Axis (km above 140,000)')
    ax_mm.yaxis.tick_left()

    ax_sma.get_yaxis().set_ticks([])
    ax_mm.get_xaxis().set_ticks(np.arange(-.8,.85,.2))
    ax_sma.get_xaxis().set_ticks(np.arange(100,350,60))

    ax_mm.set_ylabel('% of Clumps')
    ax_mm.set_yticks([])            

    graph_min = np.floor(np.min(c_velocities) * 10) / 10. - 0.00001
    graph_max = np.ceil(np.max(c_velocities) * 10) / 10. + 0.00001
    step = 0.05
#    ax_mm.set_xlim(graph_min, graph_max)
    ax_mm.set_xlim(-0.82, 0.82)
    bins = np.arange(graph_min,graph_max+step,step)
    
    counts, bins, patches = plt.hist([v_velocities, c_velocities], bins,
                                     weights = [np.zeros_like(v_velocities) + 1./v_velocities.size, np.zeros_like(c_velocities) + 1./c_velocities.size],
                                     label = ['Voyager', 'Cassini'], color = [color_voyager, color_cassini], lw = 0.0)
    
    leg = plt.legend()
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
#    fig.tight_layout()
    save_fig(fig, ax_mm, 'voyager_cassini_clump_velocity_hist.png', ax_sma, leg)
    
    v_counts, bin_edges = np.histogram(v_velocities, bins, normed = 1)
    c_counts, bin_edges = np.histogram(c_velocities, bins, normed = 1)

    print '----------VELOCITY STATS-------------------'
    print '# Cassini', len(c_velocities)
    print '# Voyager', len(v_velocities)
    print
    print '--------|  Voyager | Cassini |---------'
    print ' MIN    |  %5.3f   |  %5.3f  |'%(np.min(v_velocities),np.min(c_velocities))
    print ' MAX    |  %5.3f   |  %5.3f  |'%(np.max(v_velocities),np.max(c_velocities))
    print ' MEAN   |  %5.3f   |  %5.3f  |'%(np.mean(v_velocities),np.mean(c_velocities))
    print ' STD    |  %5.3f   |  %5.3f  |'%(np.std(v_velocities), np.std(c_velocities))



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
    
    angle_file.write('ID #,Date,Cassini Observation ID,Radial Resolution (km/pixel),Phase Angle (Deg),Incidence Angle (Deg,Emission Angle (Deg)\n')
    id_data_file.write('ID #,Date,Cassini Observation ID,First Image,Last Image,Number of Images,Coverage Percentage,Number of Clumps\n')

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
            temp_ew_data = ew_data.view(ma.MaskedArray)
            temp_ew_data.mask = ew_mask
            ew_mask = ma.getmaskarray(temp_ew_data)
            
            mean_emission = np.mean(mosaic_emission_angles[~ew_mask])
            mean_phase = c_approved_db[obs_id].phase_angle
            min_et = c_approved_db[obs_id].et_min
            mean_incidence = c_approved_db[obs_id].incidence_angle
            min_resolution = c_approved_db[obs_id].resolution_min
#            max_resolution = c_approved_db[obs_id].resolution_max
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
    
            if obs_id in c_db.keys():
                num_clumps = len(c_db[obs_id].clump_list)
            elif obs_id not in c_db.keys():
                num_clumps = 0        
                
            if obs_id in c_db.keys():
                print obs_id, ' | ', cspice.et2utc(min_et, 'D', 5), ' | ', cspice.et2utc(c_approved_db[obs_id].et_max, 'D', 5), ' | ', num_clumps
            
            start_date = cspice.et2utc(min_et, 'C', 0)
            start_date = start_date[0:11]
            angle_file.write('%i,%s,%20s,%i - %i,%.2f,%.2f,%.2f \n'%(a+1, start_date, obs_id, min_resolution, max_resolution, mean_phase,
                                                                                 mean_incidence, mean_emission))
            id_data_file.write('%i,%s,%20s,%s,%s,%i,%.2f,%i \n'%(a+1, start_date, obs_id, image_name_list[0], image_name_list[-1],
                                                                   num_images, percent_not_masked, num_clumps))
    
    id_data_file.close()
    angle_file.close()
    
def dump_clump_table(c_approved_list, c_approved_db):
    file = 'clump_data_table.txt'
    clump_table_fn = os.path.join(paper_root, file)
    clump_table = open(clump_table_fn, 'w')
    
    clump_table.write('Clump Number,First ID #,Last ID #,Number of Observations,Longitude at Epoch (deg),Minimum - Maximum Lifetime (Days),\
Relative Mean Motion (deg/day),Semimajor Axis (km),Median Width (deg),Median Brightness (km^2 x 10^4)\n')

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
            
    def print_chain_data(chain, num_id, split = False, parent = False):
        km_per_deg = 881027.02/360.
        max_life = chain.lifetime + chain.lifetime_upper_limit + chain.lifetime_lower_limit
        #    start_date = cspice.et2utc(chain.clump_list[0].clump_db_entry.et, 'C', 0)
        first_id_num = num_db[chain.clump_list[0].clump_db_entry.obsid]
    #    print first_id_num, chain.clump_list[0].clump_db_entry.obsid
        last_id_num = num_db[chain.clump_list[-1].clump_db_entry.obsid]
        med_width = np.median(np.array([clump.fit_width_deg for clump in chain.clump_list]))
        abs_width_change = ((chain.clump_list[-1].fit_width_deg - chain.clump_list[0].fit_width_deg) /
                            ((chain.clump_list[-1].clump_db_entry.et - chain.clump_list[0].clump_db_entry.et) / 86400))
#        rel_width_change = abs_width_change / chain.clump_list[0].fit_width_deg
        med_bright = np.median(np.array([clump.int_fit_height*km_per_deg for clump in chain.clump_list]))*(1./1e4)
        
    #    print np.array([clump.int_fit_height*km_per_deg for clump in chain.clump_list])*(1./1e4)
#        print [clump.g_center for clump in chain.clump_list]
    #    REFERENCE_DATE = "1 JANUARY 2007"       
    #    REFERENCE_ET = cspice.utc2et(REFERENCE_DATE)
    #    dt = chain.clump_list[0].clump_db_entry.et - REFERENCE_ET
    #    epoch_long = (chain.clump_list[0].g_center - chain.rate*dt)%360.
    
        start_long = chain.clump_list[0].g_center
        num_obs = len(chain.clump_list)
        if split:
            num_obs -= 1
            first_id_num = num_db[chain.clump_list[1].clump_db_entry.obsid]
            start_long = chain.clump_list[1].g_center
            str_format = '%s,%i,%i,%1i,%.2f,%.2f - %.2f,%.3f %6.3f,%.1f  %6.1f,%.2f,%.2f'%(num_id, first_id_num, last_id_num, num_obs,
                                                                                                start_long, chain.lifetime, max_life, chain.rate*86400.,
                                                                                                chain.rate_err*86400., chain.a, chain.a_err, med_width,
                                                                                                med_bright)
        if parent:
            num_obs = 1
            first_id_num = num_db[chain.clump_list[0].clump_db_entry.obsid]
            last_id_num = num_db[chain.clump_list[0].clump_db_entry.obsid]
            start_long = chain.clump_list[0].g_center
            med_width = chain.clump_list[0].fit_width_deg
            med_bright = chain.clump_list[0].int_fit_height*km_per_deg/1e4
            
            str_format = '%s,%i,%i,%1i,%.2f,%s,%s,%s,%.2f,%.2f'%(num_id, first_id_num, last_id_num, num_obs,
                                                                                                    start_long, 'N/A', 'N/A',
                                                                                                    'N/A', med_width,
                                                                                                    med_bright)
        if (parent == False) and (split == False):
    #        print 'C'
            str_format = '%s,%i,%i,%1i,%.2f,%.2f - %.2f,%.3f %6.3f,%.1f  %6.1f,%.2f,%.2f'%(num_id, first_id_num, last_id_num, num_obs,
                                                                                                start_long, chain.lifetime, max_life, chain.rate*86400.,
                                                                                                chain.rate_err*86400., chain.a, chain.a_err, med_width, 
                                                                                                med_bright)
        return str_format
    
    num = 1
    for time in sorted(chain_time_db.keys()):
        for a,chain in enumerate(chain_time_db[time]):
#            print a, chain.clump_list[0].g_center
            if chain.skip == False:
                parent_clump_start_long = '%6.2f'%(chain.clump_list[0].g_center)
                parent_clump_end_long = '%6.2f'%(chain.clump_list[-1].g_center)
                parent_clump_end_time = chain.clump_list[-1].clump_db_entry.et_max
                num_id = 'C'+str(num)
                chain.clump_num = num_id
                
                is_parent = False
                
                #check to see if this clump is the beginning of a split
                for b, new_chain in enumerate(chain_time_db[time][a+1::]):
                    
                    new_parent_start_long = '%6.2f'%(new_chain.clump_list[0].g_center)
    #                print parent_clump_start_long, chain.clump_list[0].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid
                    if new_parent_start_long == parent_clump_start_long:
                        print 'Found a splitting clump', parent_clump_start_long, chain.clump_list[0].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid

                        if not is_parent:
                            is_parent = True
                            parent_str_format = print_chain_data(chain, num_id, parent = True)
                            print parent_str_format
                            clump_table.write(parent_str_format+'\n')
                            num_id = num_id + "'"
                            chain.clump_num = num_id 
                            str_format = print_chain_data(chain, num_id, split = True)
                            print str_format
                            clump_table.write(str_format+'\n')
                            
                        num_id = num_id + "'"
                        chain.clump_num = num_id
                        str_format = print_chain_data(new_chain, num_id, split = True)
                        print str_format
                        clump_table.write(str_format+'\n')
                                
                        #skip this clump so that it isn't put in the table a second time
                        new_chain.skip = True
                        
                if not is_parent:
                    chain.clump_num = num_id
                    str_format = print_chain_data(chain, num_id)        
                    print str_format
                    clump_table.write(str_format+'\n')
                
        
                #check to see if parent chain split at the end
                c = 0
                for new_chain in chain_time_db[parent_clump_end_time]:
                    new_parent_start_long = '%6.2f'%(new_chain.clump_list[0].g_center)
#                    print parent_clump_end_long, new_parent_start_long

                    if new_parent_start_long == parent_clump_end_long:
                        print 'Parent clump has split', parent_clump_end_long, chain.clump_list[-1].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid
                       
                        new_num_id = num_id + "'"*(c+1)
    #                    print new_num_id
                        chain.clump_num = new_num_id
                        str_format = print_chain_data(new_chain, new_num_id, split = True)
                        print str_format
                        clump_table.write(str_format+'\n')
    #                  
                        #delete the chain so that it isn't put in the table a second time
                        new_chain.skip = True
                        c +=1
                num +=1
                
    clump_table.close()


#===============================================================================
# 
# PLOT THE SCALOGRAM FOR THE SAMPLE PROFILE
#
#===============================================================================

def plot_sample_scalogram(orig_clump_db_entry, restr_clump_db_entry):
    for passno in range(2):
        if passno == 0:
            ew_data = orig_clump_db_entry.ew_data
            clump_list = orig_clump_db_entry.clump_list
            obs_id = orig_clump_db_entry.obsid
        else:
            ew_data = restr_clump_db_entry.ew_data
            clump_list = restr_clump_db_entry.clump_list
            obs_id = restr_clump_db_entry.obsid

        long_res = 360./len(ew_data)
        longitudes = np.arange(len(ew_data))*long_res
    
        fig = plt.figure(figsize = (11,1))
        ax = fig.add_subplot(111)
        ax.set_yticks([])
    
        ax.plot(longitudes, ew_data, color = color_profile, lw = lw_profile)
    
        clump_list.sort(key=lambda x: x.g_center)
        for i, clump in enumerate(clump_list):
            if passno == 0:
                plot_one_clump(ax, ew_data, clump, 0., 360., clump_color = color_wavelet)
            else:
                plot_fitted_clump_on_ew(ax, ew_data, clump, color = color_wavelets[i%2])
                
        ax.set_xlim((longitudes[0], longitudes[-1]))
        ax.set_ylim(ma.min(ew_data), ma.max(ew_data))
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()
        ax.get_xaxis().set_ticks([])
    
#        ax.set_xlabel(r'Co-Rotating Longitude ( $\mathbf{^o}$)')
#        ax.set_ylabel('')
    
        save_fig(fig, ax, 'profile_with_wavelets_'+str(passno)+'.png')
    
def plot_sample_profile(longitudes, ew_data):
    
    figure_size = (8,5)
    font_size = 18.0
    
    fig = plt.figure(figsize = figure_size)
    ax = fig.add_subplot(111)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    
    ax.set_ylabel('Normalized Equivalent Width (km)')
    ax.set_xlabel(r'Co-Rotating Longitude ( $\mathbf{^o}$)')
    ax.set_xlim(0,360.)
    ax.get_xaxis().set_ticks([0,90,180,270,360])
    print np.min(ew_data), np.max(ew_data)
    print np.arange(np.min(ew_data), np.max(ew_data) + 0.4, 0.2)
    ax.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8,2.0])

    plt.plot(longitudes, ew_data, lw = 2.0, color = color_foreground)
#    fig.tight_layout()
    save_fig(fig, ax, 'profile_' + OBSID_SAMPLE_PROFILE + '.png')


#===============================================================================
# 
#===============================================================================
    
def plot_splitting_clumps():
    
    def draw_clumps(im, clump, color, mosaic_dimensions,rad_center = 140220.):
        top_row, bot_row, left_bound, right_bound = mosaic_dimensions
        long_res = 360./len(clump.clump_db_entry.ew_data)
        radii = np.arange(len(clump.clump_db_entry.ew_data))*radius_res + radius_start
        radii = radii[top_row:bot_row]
        radius_center = np.where(radii == rad_center)[0][0]
#        print radius_res, im.shape[1], radius_center
#        sys.exit(2)
        
        left_idx = (clump.fit_left_deg)/(long_res)-left_bound #pixels
        right_idx = (clump.fit_right_deg)/(long_res) - left_bound
        height = 30 #pixels
#        center = clump.g_center/(360./im.shape[1])
        l_thick = 4
        w_thick = 4
        
        for i in range(len(color)):
            im[radius_center + height:radius_center + height + l_thick, left_idx:right_idx +l_thick, i] = color[i]
            im[radius_center - height - l_thick:radius_center - height, left_idx:right_idx +l_thick, i] = color[i]
            im[radius_center - height - l_thick: radius_center + height + l_thick,left_idx:left_idx+w_thick, i] = color[i]
            im[radius_center - height - l_thick: radius_center + height + l_thick, right_idx:right_idx + w_thick, i] = color[i]
        
    chain_time_db = {}
    for chain in c_approved_list:
        chain.skip = False
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

    num = 1        
    for time in sorted(chain_time_db.keys()):
        for a,chain in enumerate(chain_time_db[time]):
            if chain.skip == False:
                parent_clump_start_long = '%6.2f'%(chain.clump_list[0].g_center)
                parent_clump_end_long = '%6.2f'%(chain.clump_list[-1].g_center)
                parent_clump_end_time = chain.clump_list[-1].clump_db_entry.et_max
                num_id = 'C'+str(num)
                print 'PROCESSING', num_id
                
                #check to see if this clump is the beginning of a split
                for b, new_chain in enumerate(chain_time_db[time][a+1::]):
                    new_parent_start_long = '%6.2f'%(new_chain.clump_list[0].g_center)
#                    print parent_clump_start_long, chain.clump_list[0].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid
                    if new_parent_start_long == parent_clump_start_long:
                        print 'Found a splitting clump', parent_clump_start_long, chain.clump_list[0].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid
                        new_num_id = num_id+"'"
                        
                        #skip this clump so that it isn't put in the table a second time
                        new_chain.skip = True
                        
                        split_db = {}
                        x_min_left = 9999
                        x_max_right = -9999
                        for clump in chain.clump_list:
                            if clump.fit_left_deg < x_min_left:
                                x_min_left = clump.fit_left_deg
                            if clump.fit_right_deg > x_max_right:
                                x_max_right = clump.fit_right_deg
                            t = clump.clump_db_entry.et_max
                            
                            if t not in split_db.keys():
                                split_db[t] = []
                                split_db[t].append((num_id,clump))
                            else:
                                split_db[t].append((num_id,clump))
                                
                        for clump in new_chain.clump_list:
                            if clump.fit_left_deg < x_min_left:
                                x_min_left = clump.fit_left_deg
                            if clump.fit_right_deg > x_max_right:
                                x_max_right = clump.fit_right_deg
                            t = clump.clump_db_entry.et_max
                            
                            if t not in split_db.keys():
                                split_db[t] = []
                                split_db[t].append((new_num_id,clump))
                            else:
                                split_db[t].append((new_num_id, clump)) 
                        
                        #round xmax and xmin to nearest multiple of five for graphing prettiness
                        x_min_left = np.floor((x_min_left/5.))*5.
                        x_max_right = np.ceil((x_max_right/5.))*5.
                        fig = plt.figure(figsize = (7.0,3.0))
                        num_subplots = len(split_db.keys())
                        
                        
                        ax_num = 1
                        mosaic_max = []
                        mosaic_clips = []
                        clump_data_db = {}
                        for key in split_db.keys():
                            clump_data_db[key] = []
                                
                        for d, key in enumerate(sorted(split_db.keys())):
                            
                            ax = fig.add_subplot(num_subplots, 2, ax_num)
                            ew_data = split_db[key][0][1].clump_db_entry.ew_data
                           
                            long_res = 360./len(ew_data)
                            mosaic_dimensions = (400, 650, (x_min_left-10.)/long_res, (x_max_right +10.)/long_res)
                            
                            
                            ax.get_xaxis().set_ticks(np.arange(x_min_left-10, x_max_right +10.+ 10, 10))
                            ax.set_xlim(x_min_left-10., x_max_right +10.)
                            y_min = np.min(ew_data[(x_min_left-10.)/long_res: (x_max_right + 10.)/long_res])
                            y_max = np.max(ew_data[(x_min_left-10.)/long_res: (x_max_right + 10.)/long_res]) - y_min +0.2
                            ystep = (y_max)/4.
                            ax.get_yaxis().set_ticks(np.arange(0.0, y_max + ystep, ystep ))
                            ax.set_ylim(0., y_max) 
                            yFormatter = FormatStrFormatter('%.1f')
                            xFormatter = FormatStrFormatter('%d')
                            ax.yaxis.set_major_formatter(yFormatter)
                            ax.xaxis.set_major_formatter(xFormatter)
                            ax.yaxis.tick_left()
                            ax.xaxis.tick_bottom()
                            plt.setp(ax.get_xticklabels(), visible=False)
                            
                            ax.text(0.75, 0.85, cspice.et2utc(split_db[key][0][1].clump_db_entry.et_min, 'C',0)[:11], transform = ax.transAxes)
                            
                            plot_single_ew_profile(ax, ew_data-y_min, split_db[key][0][1].clump_db_entry, 0., 360., color = color_dark_grey)
                            
                        #plot image of splitting clump
                            ax2 = fig.add_subplot(num_subplots, 2, ax_num +1)
                            obsid = split_db[key][0][1].clump_db_entry.obsid
                            ax2.xaxis.tick_bottom()
                            plt.setp(ax2.get_xticklabels(), visible=False)
                            plt.setp(ax2.get_yticklabels(), visible=False)
                            xticks = np.arange((x_min_left-10),x_max_right+10 + 10., 10.)*(1/long_res)
                            ax2.set_xticks(xticks - xticks[0])
                            xtick_labels = xticks*long_res
                            plt.setp(ax2, 'xticklabels', [str(int(tick)) for tick in xtick_labels] )  #for some reason ax2.set_xticklabels doesn't work - so we do it this way
                            ax2.tick_params(axis = 'x', direction = 'out', length = 2.0)
                            ax2.xaxis.tick_bottom()
                            ax2.set_yticks([])
                            
                            (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
                                bkgnd_mask_filename, bkgnd_model_filename, bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obsid)
                                        
                            mosaic_img = np.load(reduced_mosaic_data_filename + '.npy')
                            mosaic_data_fp = open(reduced_mosaic_metadata_filename, 'rb')
                            mosaic_data = pickle.load(mosaic_data_fp)
                            
                            (longitudes, resolutions,
                            image_numbers, ETs, 
                            emission_angles, incidence_angles,
                            phase_angles) = mosaic_data
                            
                            mu = ringutil.mu(c_approved_db[obsid].emission_angle)
                            mosaic_img = mosaic_img*mu
                            mosaic_max.append(ma.max(mosaic_img[mosaic_dimensions[0]:mosaic_dimensions[1], mosaic_dimensions[2]: mosaic_dimensions[3]]))
                            color_mosaic = np.zeros((mosaic_img.shape[0], mosaic_img.shape[1], 3))
                            color_mosaic[:,:,0] = mosaic_img
                            color_mosaic[:,:,1] = mosaic_img
                            color_mosaic[:,:,2] = mosaic_img
                #            color = (250, 0, 0)
                #            draw_clumps(color_mosaic, clump, color)
                                
                            for i,clump_data in enumerate(split_db[key]):
                                
                                clump_id, clump = clump_data
                              
                                colors = ['#A60C00', '#0AAAC2']
                                
#                                if i == 0:
                                if (d==0) and (i==0):
                                    ax.text(clump.g_center-0.5, y_max-0.15, clump_id)
                                    plot_fitted_clump_on_ew(ax,clump.clump_db_entry.ew_data-y_min, clump, color = 'black')
                                    clump_data_db[key].append((clump, d, i, 1))
                                if (d !=0):
                                    clump_id = clump_id + "'"
                                    print clump_id
                                    if clump_id[-2::] == "''":
                                        color_clump = colors[-1]
                                        clump_data_db[key].append((clump, d, i, -1))
                                    else:
                                        color_clump = colors[0]
                                        clump_data_db[key].append((clump, d, i, 0))
                                        
                                    plot_fitted_clump_on_ew(ax,clump.clump_db_entry.ew_data-y_min, clump, color = color_clump)
#                                    draw_clumps(color_mosaic, clump, im_clump_color)
                                    ax.text(clump.g_center-1.0, y_max-0.15, clump_id)
                            
                            mosaic_clip = color_mosaic[mosaic_dimensions[0]:mosaic_dimensions[1],mosaic_dimensions[2]: mosaic_dimensions[3], :]
                            mosaic_clips.append(mosaic_clip)
                            ax_num += 2
                            
                        print clump_data_db
                        mosaic_max = ma.max(mosaic_max)
                        even_axes = range(len(fig.axes))[1::2]
                        et_keys = sorted(split_db.keys())
                        for l, ax_num in enumerate(even_axes):
                            rgb_colors = [(166,12,0),(10,170,194)] #same as colors, but in rgb form
                            ax2 = fig.axes[ax_num]
                            mosaic_clip = mosaic_clips[l]
                            mode = 'RGB'
                            final_im = ImageDisp.ScaleImage(mosaic_clip, blackpoint, mosaic_max*.7, gamma)+0
                            
                            
                            for clump_data in clump_data_db[et_keys[l]]:
                                clump, row, clump_num, color_key = clump_data
                                if (row == 0) and (clump_num == 0):
                                    im_clump_color = (255, 255,255)
                                    draw_clumps(final_im, clump, im_clump_color, mosaic_dimensions)
                                if (row != 0):
                                    draw_clumps(final_im, clump, rgb_colors[color_key], mosaic_dimensions)
                                    
                            final_im = np.cast['int8'](final_im)
                            final_img = Image.frombuffer(mode, (final_im.shape[1], final_im.shape[0]),
                                                   final_im, 'raw', mode, 0, 1)
                            ax2.imshow(final_img, aspect = 'auto')
                        
                        fig.tight_layout()
                        fig.subplots_adjust(hspace = 0.1, wspace = 0.08)
#                        plt.setp(ax.get_xticklabels(), visible=True)
                        plt.setp(ax2.get_xticklabels(), visible=True)
                        ax2.tick_params(axis = 'x', direction = 'in', length = 2.0)
#                        plt.setp(ax.get_yticklabels(), visible = True)
                        
                        ax.set_xlabel('Co-Rotating Longitude ( $\mathbf{^o}$)', fontsize = 7)
                        plt.setp(ax.get_xticklabels(), visible=True)
                        plt.figtext(-0.01, .72, 'Normalized Equivalent Width (km)', rotation = 'vertical', fontsize = 7)
                        plt.savefig(os.path.join(paper_root,'Clump_split' + num_id +'.png'), bbox_inches='tight', facecolor=color_background, dpi=1000)  
#                        plt.show()    

                #check to see if parent chain split at the end
                split_chains = []
                c = 0
                for new_chain in chain_time_db[parent_clump_end_time]:
                    new_parent_start_long = '%6.2f'%(new_chain.clump_list[0].g_center)
                    if new_parent_start_long == parent_clump_end_long:
                        print 'Parent clump has split', parent_clump_end_long, chain.clump_list[-1].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid
                        new_num_id = num_id + "'"*(c+1)
                        print new_num_id
                        print '%6.2f'%(new_chain.clump_list[1].g_center) 
                        print len(new_chain.clump_list)
                        if num_id == 'C61':
                            if len(new_chain.clump_list) == 2:
                                new_num_id = num_id + "'"
                                print new_num_id
                            if len(new_chain.clump_list) > 2:
                                if ('%6.2f'%(new_chain.clump_list[1].g_center) == '214.26') and ('%6.2f'%(new_chain.clump_list[2].g_center) == '209.34'):
                                    new_num_id = num_id + "'''"
                                    print new_num_id
                                if ('%6.2f'%(new_chain.clump_list[1].g_center) == '222.42'): 
                                    new_num_id = num_id + "''"
                                    print new_num_id
                                if ('%6.2f'%(new_chain.clump_list[1].g_center) == '214.26') and ('%6.2f'%(new_chain.clump_list[2].g_center) == '212.80'):
                                    new_num_id = num_id + "''''"
                                    print new_num_id
                        if num_id == 'C35':
                            print '%6.2f'%(new_chain.clump_list[1].g_center)
#                            if ('%6.2f'%(new_chain.clump_list[1].g_center) == ' 90.79'):
#                                new_num_id = num_id + "'"
                            if ('%6.2f'%(new_chain.clump_list[1].g_center) == ' 97.84'):
                                new_num_id = num_id + "'"
                            if ('%6.2f'%(new_chain.clump_list[1].g_center) == '104.65'):
                                new_num_id = num_id + "''"
                        split_chains.append((new_num_id,new_chain))
                        #skip the chain so that it isn't put in the table a second time
                        new_chain.skip = True
                        c +=1
                        
                    if len(split_chains) == 2 or (len(split_chains) ==3) or (len(split_chains) == 4):
                        split_db = {}
                        x_max_right = -9999
                        x_min_left = 9999
                        for clump_id, split_chain in split_chains:
                            for k,clump in enumerate(split_chain.clump_list):
                                if clump.fit_left_deg < x_min_left:
                                    x_min_left = clump.fit_left_deg
                                if clump.fit_right_deg > x_max_right:
                                    x_max_right = clump.fit_right_deg
                                t = clump.clump_db_entry.et_max
                                if t not in split_db.keys():
                                    split_db[t] = []
                                    if k ==0:
                                        split_db[t].append((num_id, clump))
                                    else:
                                        split_db[t].append((clump_id, clump))
                                else:
                                    if k ==0:
                                        split_db[t].append((num_id, clump))
                                    else:
                                        split_db[t].append((clump_id, clump))
                        
                        fig = plt.figure(figsize = (7.0,7))
                        x_min_left = np.floor((x_min_left/5.))*5.
                        x_max_right = np.ceil((x_max_right/5.))*5.
                        parent_axes = len(chain.clump_list[:-1])
                        ax_num = 1
                        mosaic_max = []
                        mosaic_clips = []
                        clump_data_db = {}
                        for idx in range(len(chain.clump_list[:-1])):
                            clump_data_db[idx] = []
                        for i,clump in enumerate(chain.clump_list[:-1]):
                            
                            clump_data_db[i].append((clump, 10))
                            ax = fig.add_subplot(parent_axes, 2, ax_num)
                            ew_data = clump.clump_db_entry.ew_data
                            
                            long_res = 360./len(ew_data)
                            mosaic_dimensions = (400, 650, (x_min_left-10.)/long_res, (x_max_right +10.)/long_res)
                            ax.get_xaxis().set_ticks(np.arange(x_min_left-10., x_max_right +10. + 10, 10))
                            ax.set_xlim(x_min_left-10., x_max_right+10.)
                            y_min = np.min(ew_data[(x_min_left-10.)/long_res: (x_max_right + 10.)/long_res])
                            y_max = np.max(ew_data[(x_min_left-10.)/long_res: (x_max_right + 10.)/long_res]) - y_min
                            if num_id == 'C35':
                                y_max += 0.2
                            if num_id == 'C61':
                                y_max += 0.5
                            ystep = (y_max)/4.
                            ax.get_yaxis().set_ticks(np.arange(0.0, y_max + ystep, ystep ))
                            ax.set_ylim(0.0, y_max) 
                            xFormatter = FormatStrFormatter('%d')
                            yFormatter = FormatStrFormatter('%.1f')
                            ax.yaxis.set_major_formatter(yFormatter)
                            ax.xaxis.set_major_formatter(xFormatter)
                            ax.yaxis.tick_left()
                            ax.xaxis.tick_bottom()
                            plt.setp(ax.get_xticklabels(), visible=False)
                            
                            y_clump_max = np.max(ew_data[clump.fit_left_deg/long_res:clump.fit_right_deg/long_res])- y_min
                            ax.text(clump.g_center -1.0, y_clump_max + 0.08, num_id)
                            
                            plot_single_ew_profile(ax, ew_data-y_min, clump.clump_db_entry, 0., 360., color = color_dark_grey)
                            plot_fitted_clump_on_ew(ax, ew_data-y_min, clump, color = 'black')
                            
                            ax.text(0.77, 0.87, cspice.et2utc(clump.clump_db_entry.et_min, 'C',0)[:11], transform = ax.transAxes)

                            #plot image------------------------------------
                            ax2 = fig.add_subplot(parent_axes, 2, ax_num +1)
                            obsid = clump.clump_db_entry.obsid
                            ax2.xaxis.tick_bottom()
                            plt.setp(ax2.get_xticklabels(), visible=False)
                            plt.setp(ax2.get_yticklabels(), visible=False)
                            xticks = np.arange((x_min_left-10),x_max_right+10 + 10., 10.)*(1/long_res)
                            ax2.set_xticks(xticks - xticks[0])
                            xtick_labels = xticks*long_res
                            plt.setp(ax2, 'xticklabels', [str(int(tick)) for tick in xtick_labels] )  #for some reason ax2.set_xticklabels doesn't work - so we do it this way
                            ax2.tick_params(axis = 'x', direction = 'out', length = 2.0)
                            ax2.xaxis.tick_bottom()
                            ax2.set_yticks([])
                            
                            (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
                                bkgnd_mask_filename, bkgnd_model_filename, bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obsid)
                                        
                            mosaic_img = np.load(reduced_mosaic_data_filename + '.npy')
                            mosaic_data_fp = open(reduced_mosaic_metadata_filename, 'rb')
                            mosaic_data = pickle.load(mosaic_data_fp)
                                                        
                            mu = ringutil.mu(c_approved_db[obsid].emission_angle)
                            mosaic_img = mosaic_img*mu
                            mosaic_max.append(ma.max(mosaic_img[400:650, (x_min_left -10.)/long_res: (x_max_right +10.)/long_res + 1]))
                            color_mosaic = np.zeros((mosaic_img.shape[0], mosaic_img.shape[1], 3))
                            color_mosaic[:,:,0] = mosaic_img
                            color_mosaic[:,:,1] = mosaic_img
                            color_mosaic[:,:,2] = mosaic_img
                            
#                            color = (250, 0, 0)
#                            draw_clumps(color_mosaic, clump, color)
                            mosaic_clip = color_mosaic[mosaic_dimensions[0]:mosaic_dimensions[1],mosaic_dimensions[2]:mosaic_dimensions[3], :]
                            mosaic_clips.append(mosaic_clip)
                            ax_num += 2
                            
                        num_subplots = len(split_db.keys())
                        #change the axes geometry to add the split chains to the figure
                        total_subs = num_subplots + parent_axes
                        for m in range(len(fig.axes)):
                            fig.axes[m].change_geometry(total_subs, 2, m+1)
                        start_ax = len(fig.axes)
                        ax_num = start_ax +1
                        
                        print clump_data_db
                        offset = len(clump_data_db.keys())
                        for key in range(len(split_db.keys())):
                            clump_data_db[key + offset] = []
                        print clump_data_db   
                        for d, key in enumerate(sorted(split_db.keys())):
                            sub_num = parent_axes + d+1 
                            ew_data = split_db[key][0][1].clump_db_entry.ew_data
                            long_res = 360./len(ew_data)
                            mosaic_dimensions = (400, 650, (x_min_left-10.)/long_res, (x_max_right +10.)/long_res)
                            
                            ax = fig.add_subplot(total_subs, 2, ax_num)
                            ax.get_xaxis().set_ticks(np.arange(x_min_left-10.,x_max_right +10.+10, 10))
                            ax.set_xlim(x_min_left - 10., x_max_right + 10.)
                            y_min = np.min(ew_data[(x_min_left-10.)/long_res:(x_max_right-10.)/long_res])
                            y_max = np.max(ew_data[(x_min_left-10.)/long_res:(x_max_right-10.)/long_res]) - y_min
                            if (d==0) and (num_id == 'C35'):
                                y_max += 0.2
                            if (d==0) and (num_id == 'C61'):
                                y_max += 0.5
                            elif d !=0:
                                y_max += 0.2
                            ystep = y_max/4.
                            ax.get_yaxis().set_ticks(np.arange(0.0, y_max + ystep, ystep ))
                            ax.set_ylim(0.0, y_max)
                            ax.yaxis.tick_left()
                            ax.xaxis.tick_bottom()
                            plt.setp(ax.get_xticklabels(), visible=False)
                            
                            ax.text(0.77, 0.87, cspice.et2utc(split_db[key][0][1].clump_db_entry.et_min, 'C',0)[:11], transform = ax.transAxes)
                            plot_single_ew_profile(ax, ew_data- y_min, split_db[key][0][1].clump_db_entry, 0., 360., color = color_dark_grey)
                            
                            ax2 = fig.add_subplot(total_subs, 2, ax_num +1)
                            obsid = split_db[key][0][1].clump_db_entry.obsid
                            ax2.xaxis.tick_bottom()
                            plt.setp(ax2.get_xticklabels(), visible=False)
                            plt.setp(ax2.get_yticklabels(), visible=False)
                            xticks = np.arange((x_min_left-10),x_max_right+10 + 10., 10.)*(1/long_res)
                            ax2.set_xticks(xticks - xticks[0])
                            xtick_labels = xticks*long_res
                            plt.setp(ax2, 'xticklabels', [str(int(tick)) for tick in xtick_labels] )  #for some reason ax2.set_xticklabels doesn't work - so we do it this way
                            ax2.tick_params(axis = 'x', direction = 'out', length = 2.0)
                            ax2.xaxis.tick_bottom()
                            ax2.set_yticks([])
                            
                            (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
                                bkgnd_mask_filename, bkgnd_model_filename, bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obsid)
                                        
                            mosaic_img = np.load(reduced_mosaic_data_filename + '.npy')
                            mosaic_data_fp = open(reduced_mosaic_metadata_filename, 'rb')
                            mosaic_data = pickle.load(mosaic_data_fp)
                                                        
                            mu = ringutil.mu(c_approved_db[obsid].emission_angle)
                            mosaic_img = mosaic_img*mu
                            mosaic_max.append(ma.max(mosaic_img[400:650, (x_min_left -10.)/long_res: (x_max_right +10.)/long_res + 1]))
                            color_mosaic = np.zeros((mosaic_img.shape[0], mosaic_img.shape[1], 3))
                            color_mosaic[:,:,0] = mosaic_img
                            color_mosaic[:,:,1] = mosaic_img
                            color_mosaic[:,:,2] = mosaic_img
                            mosaic_clip = color_mosaic[mosaic_dimensions[0]:mosaic_dimensions[1],mosaic_dimensions[2]:mosaic_dimensions[3], :]
                            mosaic_clips.append(mosaic_clip)
                            
                            
                            for i,clump_data in enumerate(split_db[key]):
                                clump_id, clump = clump_data
#                                print i + offset, clump_id
                                if (d==1) and ((clump_id == "C61'''") or (clump_id == "C61''''")):
                                    continue
#                                print clump_id
                                colors = ['#A60C00', '#0AAAC2', '#14A300', '#9462BF']
                                rgb_colors = [(166, 12, 0), (10, 170, 194), (20, 163, 0), (197, 131,255 )]
                                if (clump_id[-4::] == "''''"):
                                    clump_color = colors[3]
                                    clump_data_db[d + offset].append((clump, 3))
#                                    im_clump_color = rgb_colors[3]
                                if (clump_id[-3::] == "'''") and (clump_id[-4::] != "''''"):
                                    clump_color = colors[2]
                                    clump_data_db[d +offset].append((clump, 2))
#                                    im_clump_color = rgb_colors[2]
                                if (clump_id[-2::] == "''") and (clump_id[-3::] != "'''") and (clump_id[-4::] != "''''"):
                                    clump_color = colors[1]
                                    clump_data_db[d +offset].append((clump, 1))
#                                    im_clump_color = rgb_colors[1]
                                if (d==0):
                                    clump_color = 'black'
                                    clump_data_db[d +offset].append((clump, 10))
                                elif (d != 0) and (clump_id[-1::] == "'") and (clump_id[-2::] != "''") and (clump_id[-3::] != "'''") and (clump_id[-4::] != "''''"):
                                    clump_color = colors[0]
#                                    im_clump_color = rgb_colors[0]
                                    clump_data_db[d +offset].append((clump, 0))
#                                draw_clumps(color_mosaic, clump, im_clump_color)
                                plot_fitted_clump_on_ew(ax,clump.clump_db_entry.ew_data - y_min, clump, color = clump_color)
                                y_clump_max = np.max(ew_data[clump.fit_left_deg/long_res:clump.fit_right_deg/long_res]) -y_min
                                if clump_id == 'C61':
                                    y_clump_max += 0.08
                                if clump_id == 'C35':
                                    y_clump_max += 0.05
                                elif (clump_id != 'C35') and (clump_id != 'C61'):
                                    y_clump_max += 0.05
                                ax.text(clump.g_center-1.5, y_clump_max, clump_id)
#                                print clump_data_db
                           
#                                line_height = np.arange(y_min,y_max,0.01)
                        
                            xFormatter = FormatStrFormatter('%d')
                            yFormatter = FormatStrFormatter('%.1f')
                            ax.yaxis.set_major_formatter(yFormatter)    
                            ax.xaxis.set_major_formatter(xFormatter) 
                            ax_num += 2
                           
                        mosaic_max = ma.max(mosaic_max)
                        even_axes = range(len(fig.axes))[1::2]
#                        print clump_data_db
                        for l, ax_num in enumerate(even_axes):
                            rgb_colors = [(166, 12, 0), (10, 170, 194), (20, 163, 0), (197, 131,255 )] #same as clump colors for profiles
                            ax2 = fig.axes[ax_num]
                            mosaic_clip = mosaic_clips[l]
                            mode = 'RGB'
                            final_im = ImageDisp.ScaleImage(mosaic_clip, blackpoint, mosaic_max*0.8, gamma)+0
                            
                            for clump_data in clump_data_db[l]:
                                clump, color_key = clump_data
                                if color_key == 10:
                                    im_clump_color = (255, 255, 255)
                                    draw_clumps(final_im, clump, im_clump_color, mosaic_dimensions)
                                else:
                                    draw_clumps(final_im, clump, rgb_colors[color_key], mosaic_dimensions)
                                
                            final_im = np.cast['int8'](final_im)
                            final_img = Image.frombuffer(mode, (final_im.shape[1], final_im.shape[0]),
                                                   final_im, 'raw', mode, 0, 1)
                            
                            ax2.imshow(final_img, aspect = 'auto')   
                            
                        fig.tight_layout()
                        fig.subplots_adjust(hspace = 0.125, wspace = 0.08)
#                        plt.setp(ax.get_xticklabels(), visible=True)
                        plt.setp(ax2.get_xticklabels(), visible=True)
                        ax2.tick_params(axis = 'x', direction = 'in', length = 2.0)
#                        fig.subplots_adjust(hspace = 0.125)
                        plt.setp(ax.get_xticklabels(), visible=True)
                        ax.set_xlabel('Co-Rotating Longitude ( $\mathbf{^o}$)', fontsize = 10)
                        plt.figtext(-0.025, .62, 'Normalized Equivalent Width (km)', rotation = 'vertical', fontsize = 10)

                        plt.savefig(os.path.join(paper_root,'clump_split' + num_id +'.png'), bbox_inches='tight', facecolor=color_background, dpi=1000)        
                        
#                        plt.show()
                num +=1


#===============================================================================
################################################################################
#
# STUFF RELATED TO SINGLE CLUMPS
#
################################################################################
#===============================================================================

def plot_single_clump_distributions():
    v_clump_numbers = []
    v_clump_raw_numbers = []
    v_clump_scales = []
    v_clump_heights = []
    v_stddevs = []
    for v_obs in v_all_clump_db.keys():
        for clump in v_all_clump_db[v_obs].clump_list:
            v_clump_scales.append(clump.fit_width_deg)
            v_clump_heights.append(clump.int_fit_height)
        coverage = ma.MaskedArray.count(v_all_clump_db[v_obs].ew_data)/float(len(v_all_clump_db[v_obs].ew_data))
        if coverage >= 0.8:
            v_clump_numbers.append(len(v_all_clump_db[v_obs].clump_list) / coverage)
        v_clump_raw_numbers.append(len(v_all_clump_db[v_obs].clump_list))
        v_stddevs.append(np.std(v_all_clump_db[v_obs].ew_data))

    v_clump_num_stats = (np.mean(v_clump_numbers),
                         np.std(v_clump_numbers),
                         np.min(v_clump_numbers),
                         np.max(v_clump_numbers))
    v_scale_stats = (np.mean(v_clump_scales),
                     np.std(v_clump_scales),
                     np.min(v_clump_scales),
                     np.max(v_clump_scales))
    v_clump_height_stats = (np.mean(v_clump_heights),
                            np.std(v_clump_heights),
                            np.min(v_clump_heights),
                            np.max(v_clump_heights))
    v_stddev_stats = (np.mean(v_stddevs),
                      np.std(v_stddevs),
                      np.min(v_stddevs),
                      np.max(v_stddevs))

    c_clump_numbers = []
    c_clump_raw_numbers = []
    c_clump_scales = []
    c_clump_heights = []
    c_stddevs = []
    c_ets = []
    
    for c_obs in c_all_clump_db.keys():
        for clump in c_all_clump_db[c_obs].clump_list:
            c_clump_scales.append(clump.fit_width_deg)
            c_clump_heights.append(clump.int_fit_height)
        coverage = ma.MaskedArray.count(c_all_clump_db[c_obs].ew_data)/float(len(c_all_clump_db[c_obs].ew_data))
        if coverage >= 0.8:
            c_clump_numbers.append(len(c_all_clump_db[c_obs].clump_list) / coverage)
            c_ets.append(c_all_clump_db[c_obs].et_min)
#            if c_clump_numbers[-1] >= 10 and c_clump_numbers[-1] <= 12:
#                print '10-12', c_obs, c_clump_numbers[-1], len(c_all_clump_db[c_obs].clump_list), coverage
#                for clump in c_all_clump_db[c_obs].clump_list:
#                    clump.print_all()
        c_clump_raw_numbers.append(len(c_all_clump_db[c_obs].clump_list))
        c_stddevs.append(np.std(c_all_clump_db[c_obs].ew_data))

    c_clump_num_stats = (np.mean(c_clump_numbers),
                         np.std(c_clump_numbers),
                         np.min(c_clump_numbers),
                         np.max(c_clump_numbers))
    c_scale_stats = (np.mean(c_clump_scales),
                     np.std(c_clump_scales),
                     np.min(c_clump_scales),
                     np.max(c_clump_scales))
    c_clump_height_stats = (np.mean(c_clump_heights),
                            np.std(c_clump_heights),
                            np.min(c_clump_heights),
                            np.max(c_clump_heights))
    c_stddev_stats = (np.mean(c_stddevs),
                      np.std(c_stddevs),
                      np.min(c_stddevs),
                      np.max(c_stddevs))
    
    print '-----------------------STATS TABLE--------------------------'
    if choose_smaller:
        print 'SMALLER CLUMPS PREFERRED'
    if choose_larger:
        print 'LARGER CLUMPS PREFERRED'
    print
    print '----------------------| Voyager | Cassini ------------------'
    print
    print 'AVG # Clumps/Observ   |  %5.1f   |  %5.1f  '%(v_clump_num_stats[0], c_clump_num_stats[0])
    print 'STD # Clumps/Observ   |  %5.1f   |  %5.1f  '%(v_clump_num_stats[1], c_clump_num_stats[1])
    print 'MIN # Clumps/Observ   |  %5.1f   |  %5.1f  '%(v_clump_num_stats[2], c_clump_num_stats[2])
    print 'MAX # Clumps/Observ   |  %5.1f   |  %5.1f  '%(v_clump_num_stats[3], c_clump_num_stats[3])
    print
    print 'AVG Clump Fit Width   |  %5.2f   |  %5.2f  '%(v_scale_stats[0],c_scale_stats[0])
    print 'STD Clump Fit Width   |  %5.2f   |  %5.2f  '%(v_scale_stats[1],c_scale_stats[1])
    print 'MIN Clump Fit Width   |  %5.2f   |  %5.2f  '%(v_scale_stats[2],c_scale_stats[2])
    print 'MAX Clump Fit Width   |  %5.2f   |  %5.2f  '%(v_scale_stats[3],c_scale_stats[3])
    print
    print 'AVG Clump Int Height  |  %7.4f | %7.4f  '%(v_clump_height_stats[0], c_clump_height_stats[0])
    print 'STD Clump Int Height  |  %7.4f | %7.4f  '%(v_clump_height_stats[1], c_clump_height_stats[1])
    print 'MIN Clump Int Height  |  %7.4f | %7.4f  '%(v_clump_height_stats[2], c_clump_height_stats[2])
    print 'MAX Clump Int Height  |  %7.4f | %7.4f  '%(v_clump_height_stats[3], c_clump_height_stats[3])
    print
    print 'AVG OBSID STDDEV      |  %5.2f   |  %5.2f  '%(v_stddev_stats[0], c_stddev_stats[0])
    print 'STD OBSID STDDEV      |  %5.2f   |  %5.2f  '%(v_stddev_stats[1], c_stddev_stats[1])
    print 'MIN OBSID STDDEV      |  %5.2f   |  %5.2f  '%(v_stddev_stats[2], c_stddev_stats[2])
    print 'MAX OBSID STDDEV      |  %5.2f   |  %5.2f  '%(v_stddev_stats[3], c_stddev_stats[3])
    print '------------------------------------------------------------'
    print
    print 'TOTAL NUMBER OF CASSINI CLUMPS (WEIGHTED):', np.sum(c_clump_numbers)
    print 'TOTAL NUMBER OF VOYAGER CLUMPS (WEIGHTED):', np.sum(v_clump_numbers)
    print
    print 'TOTAL NUMBER OF CASSINI CLUMPS (RAW):', np.sum(c_clump_raw_numbers)
    print 'TOTAL NUMBER OF VOYAGER CLUMPS (RAW):', np.sum(v_clump_raw_numbers)
    print
    
    # Width comparison
    
    step = 2.5
    graph_min = 0.0
    graph_max = (int(max(np.max(c_clump_scales), np.max(v_clump_scales)) / step) + 0) * step
    bins = np.arange(0, graph_max+step,step)

    fig = plt.figure(figsize = (11,2.5))
    ax = fig.add_subplot(111)
    
    ax.set_xlabel('Clump Width ( $\mathbf{^o}$)')
    ax.set_ylabel('% of Clumps')
    c_scale_weights = np.zeros_like(c_clump_scales) + 1./len(c_clump_scales)
    v_scale_weights = np.zeros_like(v_clump_scales) + 1./len(v_clump_scales)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    scale_counts, bins, patches = plt.hist([v_clump_scales, c_clump_scales], bins,
                                      weights = [v_scale_weights, c_scale_weights], label = ['Voyager', 'Cassini'],
                                      color = [color_voyager, color_cassini], lw = 0.0)
    
#    ax.get_yaxis().set_ticks(np.arange(0.0, 0.5 + 0.1, 0.1))
    ax.get_yaxis().set_ticks([])
    ax.set_xlim(graph_min, graph_max)
    leg = plt.legend()
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
    save_fig(fig, ax, 'clump_width_dist.png', legend=leg)   
    
    # Brightness comparison
    
    fig2 = plt.figure(figsize = (11,2.5))
    ax = fig2.add_subplot(111)
    ax.set_xlabel(r'Clump Brightness ( $\mathbf{^o}$)')
    ax.set_ylabel('% of Clumps')
    
    v_min = np.min(v_clump_heights)
    v_max = np.max(v_clump_heights)
    c_max = np.max(c_clump_heights)
    c_min = np.min(c_clump_heights)
    
    graph_max = max(c_max, v_max)
    graph_min = min(c_min, v_min)
    
    step = 0.2
    bins = np.arange(graph_min,graph_max + step,step)
    ax.get_xaxis().set_ticks(np.arange(0.,graph_max+1., 1.0))
    ax.set_xlim(0,3)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    
    c_height_weights = np.zeros_like(c_clump_heights) + 1./len(c_clump_heights)
    v_height_weights = np.zeros_like(v_clump_heights) + 1./len(v_clump_heights)
    
    plt.hist([v_clump_heights, c_clump_heights], bins,
             weights = [v_height_weights, c_height_weights], label = ['Voyager', 'Cassini'], color = [color_voyager, color_cassini], lw = 0.0)
    
    leg = plt.legend()
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
    ax.set_yticks([])            
    save_fig(fig2, ax, 'clump_brightness_dist_zoom.png', legend=leg)

    fig2 = plt.figure(figsize = (11,2.5))
    ax = fig2.add_subplot(111)
    ax.set_xlabel(r'Clump Brightness ( $\mathbf{^o}$)')
    ax.set_ylabel('% of Clumps')

    step = 5
    bins = np.arange(graph_min,graph_max + step,step)
    ax.get_xaxis().set_ticks(np.arange(0.,graph_max+10., 10.0))
    ax.set_xlim(0,((graph_max//10+1)*10))
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    
    c_height_weights = np.zeros_like(c_clump_heights) + 1./len(c_clump_heights)
    v_height_weights = np.zeros_like(v_clump_heights) + 1./len(v_clump_heights)
    
    height_counts, bins, patches = plt.hist([v_clump_heights, c_clump_heights], bins,
                                      weights = [v_height_weights, c_height_weights], label = ['Voyager', 'Cassini'], color = [color_voyager, color_cassini], lw = 0.0)
    
    leg = plt.legend()
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
    ax.set_yticks([])            
    save_fig(fig2, ax, 'clump_brightness_dist.png', legend=leg)

    # Number of clumps over time
    
    fig = plt.figure(figsize = (10,2.5))
    ax = fig.add_subplot(111)
    
    labels = ['2004 JAN 01 00:00:000', '2005 JAN 01 00:00:000', '2006 JAN 01 00:00:000', '2007 JAN 01 00:00:000',
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
    fig.autofmt_xdate()
#    ax.get_yaxis().set_ticks(np.arange(20, 55+3, 3))
    ax.set_ylim(8,32)
    ax.set_yticks([10,20,30])
    plt.plot(c_ets, c_clump_numbers, '.', color = color_cassini, ms=ms_scatter)

    coeffs = np.polyfit(c_ets, c_clump_numbers, 1)
    x_range = np.arange(np.min(c_ets), np.max(c_ets), 86400*10)
    plt.plot(x_range, np.polyval(coeffs, x_range), '-', color=color_poisson, lw=lw_poisson)

    ax.set_ylabel('# of Clumps')
#    ax.set_xlabel('Date')
#    fig.tight_layout()
    save_fig(fig, ax, 'num_clumps_vs_time.png')

    # Number of clumps per ID

    for spacecraft in ['Cassini', 'Voyager']:
        fig = plt.figure(figsize = (10,6))
        ax = fig.add_subplot(111)
        ax.set_xlabel('Weighted # of Clumps per Profile')
        ax.set_ylabel('% of Profiles')
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        
        step = 2
        bin_min = (np.min(c_clump_numbers) // step - 1) * step 
        bin_max = (np.max(c_clump_numbers) // step + 2) * step
        bins = np.arange(bin_min, bin_max+step, step)  # Bins are semi-open interval [x,y)
        c_clump_numbers = np.array(c_clump_numbers)
        num_weights = np.zeros_like(c_clump_numbers) + 1./len(c_clump_numbers)
        
        counts, xbins, patches = plt.hist(c_clump_numbers, bins, weights = num_weights, color = color_cassini, lw = 0.0)

        if spacecraft == 'Voyager':
            v_clump_numbers = np.array(v_clump_numbers)
            num_weights = np.zeros_like(v_clump_numbers) + 1./len(v_clump_numbers)
            
            plt.hist(v_clump_numbers, bins, weights = num_weights, color = color_voyager, lw = 0.0, alpha=0.8)
        
        best_poisson_mean = None
        best_resid = 1e38
        for poisson_mean in np.arange(bin_min, bin_max, 0.01):
            dist = 0.
            for offset in range(step):
                dist += scipydist.poisson.pmf(bins[:-1]+offset, poisson_mean)
            resid = np.sqrt(np.sum((counts-dist)**2))
            if resid < best_resid:
                best_poisson_mean = poisson_mean
                best_resid = resid
        print 'Best', spacecraft, 'Poisson Mean', best_poisson_mean
        if spacecraft == 'Cassini':
            c_best_poisson_mean = best_poisson_mean
        else:
            v_best_poisson_mean = best_poisson_mean
        pbins = np.arange(bin_min, bin_max-(step-1)/2., 1)
        dist = 0.
        for offset in range(step):
            dist += scipydist.poisson.pmf(pbins+offset, best_poisson_mean)

        plt.plot(pbins+(step-1)/2., dist, '-', color=color_poisson, lw=lw_poisson)
        ax.set_xlim(bin_min, bin_max)
        ax.set_ylim(0, np.max(counts))
        ax.set_yticks([])            
        save_fig(fig, ax, 'clump_numbers_dist_'+spacecraft+'.png', legend=leg)
        
    v_scale_dist = scale_counts[0]
    c_scale_dist = scale_counts[1]
    v_height_dist = height_counts[0]
    c_height_dist = height_counts[1]    
    
    graph_min = np.min(c_clump_numbers)
    graph_max = np.max(c_clump_numbers)
    step = 1.0
    bins = np.arange(0,graph_max+step,step)

    fig = plt.figure(figsize = (11,6))
    ax = fig.add_subplot(111)
    
    c_num_weights = np.zeros_like(c_clump_numbers) + 1./len(c_clump_numbers)
    v_num_weights = np.zeros_like(v_clump_numbers) + 1./len(v_clump_numbers)
    
    num_counts, bins, patches = plt.hist([v_clump_numbers, c_clump_numbers], bins,
                                      weights = [v_num_weights, c_num_weights], label = ['Voyager', 'Cassini'], color = ['#AC7CC9', 'black'], lw = 0.0)

    v_num_dist = num_counts[0]
    c_num_dist = num_counts[1]

    clump_num_d, clump_num_pks = st.ks_2samp(c_num_dist, v_num_dist)
    clump_scale_d, clump_scale_pks = st.ks_2samp(v_scale_dist, c_scale_dist)
    clump_height_d, clump_height_pks = st.ks_2samp(v_height_dist, c_height_dist)

    c_poisson_dist = scipydist.poisson.pmf(bins, c_best_poisson_mean)
    v_poisson_dist = scipydist.poisson.pmf(bins, v_best_poisson_mean)

    c_clump_num_poisson_d, c_clump_num_poisson_p = st.ks_2samp(c_num_dist, c_poisson_dist)
    v_clump_num_poisson_d, v_clump_num_poisson_p = st.ks_2samp(v_num_dist, v_poisson_dist)
    
    print
    print '----------------K-S STATS--------------'
    print '(Low p means distributions are DIFFERENT)'
    print '-------------     -|   D   |   p    --------'
    print '---------------------------------------'
    print 'Clumps per ID      | %5.3f | %5.5f '%(clump_num_d, clump_num_pks)
    print 'C # Clumps Poisson | %5.3f | %5.5f '%(c_clump_num_poisson_d, c_clump_num_poisson_p)
    print 'V # Clumps Poisson | %5.3f | %5.5f '%(v_clump_num_poisson_d, v_clump_num_poisson_p)
    print 'Clump Widths       | %5.3f | %5.5f '%(clump_scale_d, clump_scale_pks)
    print 'Clump Height       | %5.3f | %5.5f '%(clump_height_d, clump_height_pks)
    print

def plot_clumps_vs_prometheus():    
    long_differences = []
    
    for obs in sorted(c_all_clump_db.keys()):
        coverage = ma.MaskedArray.count(c_all_clump_db[obs].ew_data)/float(len(c_all_clump_db[obs].ew_data))
        if coverage < 0.8:
            continue
        p_rad, p_long = ringimage.saturn_to_prometheus(c_all_clump_db[obs].et_min)
        for clump in c_all_clump_db[obs].clump_list:
            long_diff = clump.g_center - p_long
            if long_diff < -180: long_diff += 360
            if long_diff > 180: long_diff -= 360
            long_differences.append(long_diff)
    
    print 'Num Prometheus/Clumps =', len(long_differences)
    
    fig = plt.figure(figsize = (10,2.5))
    ax = fig.add_subplot(111)
    bins = np.arange(-180., 181., 30.)
    plt.hist(long_differences, bins, weights = np.zeros_like(long_differences) + 1./len(long_differences),
             color=color_cassini)
    ax.set_xlabel(r'Clump Longitude $-$ Prometheus Longitude')
    ax.set_ylabel('% of Clumps')
    plt.xlim(-180,180)
    plt.ylim(0,0.12)
    ax.get_xaxis().set_ticks(np.arange(-180,181,60))
    ax.set_yticks([])
    
#    fig.tight_layout()
    save_fig(fig, ax, 'clump_prometheus_long_diff.png')

    #===========================================================================
    # 
    #===========================================================================
    
def plot_single_clumps(clump_db_entry, clump_list, obsid):
    colors = ['red', 'green', 'blue', 'magenta', 'cyan']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    long_res = 360./len(clump_db_entry.ew_data)
    longitudes = np.arange(len(clump_db_entry.ew_data)) * long_res
    plt.plot(longitudes, clump_db_entry.ew_data, '-', color='black', lw=1)
    
    clump_list.sort(key=lambda x: x.g_center)
    
    for n, clump in enumerate(clump_list):
        left_idx = int(clump.fit_left_deg/long_res)
        right_idx = int(clump.fit_right_deg/long_res)
        if right_idx < left_idx:
            plt.plot(longitudes[left_idx:],
                     clump_db_entry.ew_data[left_idx:]+(n%5)*.1, '-',
                     color=colors[n%len(colors)], lw=1)
            plt.plot(longitudes[:right_idx+1],
                     clump_db_entry.ew_data[:right_idx+1]+(n%5)*.1, '-',
                     color=colors[n%len(colors)], lw=1)
        else:
            plt.plot(longitudes[left_idx:right_idx+1],
                     clump_db_entry.ew_data[left_idx:right_idx+1]+(n%5)*.1, '-',
                     color=colors[n%len(colors)], lw=1)
        
    ax.set_xlim(0,360)
    plt.title(obsid)
    plt.show()

def compute_clump_attributes(clump):
    ew_data = clump.clump_db_entry.ew_data
    long_res = 360. / len(ew_data)
    dbl_ew_data = np.tile(ew_data, 2)
    left_idx = int(clump.fit_left_deg / long_res)
    right_idx = int(clump.fit_right_deg / long_res)
    if left_idx > right_idx:
        right_idx += len(ew_data) 
    left_val = dbl_ew_data[left_idx]
    right_val = dbl_ew_data[right_idx]
    ctr_val = np.max(dbl_ew_data[left_idx:right_idx+1])
    left_height = ctr_val - left_val
    right_height = ctr_val - right_val
    if left_val == ctr_val or right_val == ctr_val:
        height = 0.
    else:
        height = ctr_val-(left_val+right_val)/2
    if right_height > left_height:
        asym_ratio = right_height / left_height
    else:
        asym_ratio = left_height / right_height
    width = right_idx*long_res - left_idx*long_res
    return height, width, asym_ratio

def limit_single_clumps(clump_list):
    new_clump_list = []
    for clump in clump_list:
        if clump.clump_db_entry is None:
            print 'NO DB'
            clump.print_all()
            assert False
        if clump.int_fit_height < 0:
            if debug:
                print 'NEG INT HEIGHT'
            continue 
        ew_data = clump.clump_db_entry.ew_data
        long_res = 360. / len(ew_data)
        left_idx = int(clump.fit_left_deg / long_res)
        right_idx = int(clump.fit_right_deg / long_res)
        if ((left_idx >= 0 and left_idx < len(ew_data) and ma.count_masked(ew_data[left_idx]) == 1) or
            (right_idx >= 0 and right_idx < len(ew_data) and ma.count_masked(ew_data[right_idx]) == 1)):
            if debug:
                print 'Edge is masked'
            continue
        height, width, asym_ratio = compute_clump_attributes(clump)
        if debug:
            print 'Height', height
        if height < 0.1:# or height > 2: # XXX
            continue
        if width < 3.5 or width > 40:
            continue
        if debug:
            print 'Asym', asym_ratio
        if asym_ratio > 5:
            continue
    
        new_clump_list.append(clump)
        
    return new_clump_list

def choose_correct_single_clumps(clump_db):
    for obsid in sorted(clump_db):
        if debug:
            print obsid
        clump_db_entry = clump_db[obsid]
        new_list = []
        restr_clump_list = limit_single_clumps(clump_db_entry.clump_list)
        restr_clump_list.sort(key=lambda x: x.fit_left_deg)
        for clump_num, clump in enumerate(restr_clump_list):
            clump_left_deg = clump.fit_left_deg
            clump_right_deg = clump.fit_right_deg
            if clump_right_deg < clump_left_deg:    
                clump_right_deg += 360
            found_match = False
            for sec_clump_num, sec_clump in enumerate(restr_clump_list):
                if clump_num == sec_clump_num:
                    if debug:
                        print 'SAME'
                    continue
                sec_clump_left_deg = sec_clump.fit_left_deg
                sec_clump_right_deg = sec_clump.fit_right_deg
                sec_clump_left_deg2 = sec_clump_left_deg
                sec_clump_right_deg2 = sec_clump_right_deg
                
                if sec_clump_right_deg < sec_clump_left_deg:    
                    sec_clump_right_deg += 360
                    sec_clump_left_deg2 -= 360
                if debug:
                    print '%7.2f %7.2f %7.2f %7.2f W %7.2f %7.2f' % (clump_left_deg, clump_right_deg,
                                                       sec_clump_left_deg, sec_clump_right_deg,
                                                       clump_right_deg-clump_left_deg,
                                                       sec_clump_right_deg-sec_clump_left_deg) 
                if sec_clump_left_deg == clump_left_deg and sec_clump_right_deg == clump_right_deg:
                    if clump_num > sec_clump_num:
                        if debug:
                            print 'IDENTICAL'
                        found_match = True
                        break
                if (abs(sec_clump_left_deg-clump_left_deg) <= 2 and
                    abs(sec_clump_right_deg-clump_right_deg) <= 2):
                    # Identical within 2 deg - let the larger one go through
                    if sec_clump_right_deg - sec_clump_left_deg > clump_right_deg - clump_left_deg:
                        if debug:
                            print 'CLOSE AND SMALLER'
                        found_match = True
                        break
                if (choose_smaller and 
                    ((sec_clump_left_deg < clump_right_deg and sec_clump_right_deg > clump_left_deg) or
                     (sec_clump_left_deg2 < clump_right_deg and sec_clump_right_deg2 > clump_left_deg))):
                    if sec_clump_right_deg - sec_clump_left_deg < clump_right_deg - clump_left_deg:
                        if debug:
                            print 'ENCLOSED CHOOSING SMALLER'
                        found_match = True
                        break
                if (choose_larger and 
                    ((sec_clump_left_deg < clump_right_deg and sec_clump_right_deg > clump_left_deg) or
                     (sec_clump_left_deg2 < clump_right_deg and sec_clump_right_deg2 > clump_left_deg))):
                    if sec_clump_right_deg - sec_clump_left_deg > clump_right_deg - clump_left_deg:
                        if debug:
                            print 'ENCLOSED CHOOSING LARGER'
                        found_match = True
                        break
            if not found_match:
                if debug:
                    if clump_right_deg-clump_left_deg == 36.5:
                        print 'KEEPING CLUMP'
                new_list.append(clump)
                
        clump_db_entry.clump_list = new_list
#        if obsid == 'ISS_031RF_FMOVIE001_VIMS' or obsid == 'ISS_055RI_LPMRDFMOV001_PRIME' or obsid == 'ISS_051RI_LPMRDFMOV001_PRIME' or obsid == 'ISS_134RI_SPKMVDFHP002_PRIME' or obsid == 'ISS_033RF_FMOVIE001_VIMS':
#        plot_single_clumps(clump_db_entry, new_list, obsid)

def analyze_approved_clumps():
    v_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'voyager_approved_clumps_list.pickle')
    v_approved_list_fp = open(v_approved_list_fp, 'rb')
    v_approved_db, v_approved_list = pickle.load(v_approved_list_fp)
    v_approved_list_fp.close()
    
    c_approved_list_fp = os.path.join(ringutil.ROOT, 'Paper', 'Figures', 'approved_list_w_errors.pickle')
    c_approved_list_fp = open(c_approved_list_fp, 'rb')
    c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
    c_approved_list_fp.close()
    
    for approved_list in [v_approved_list, c_approved_list]:
        print 'V/C'
        height_list = []
        ratio_list = []
        for chain in approved_list:
            h_list = []
            a_list = []
            for clump in chain.clump_list:
                height, width, asym_ratio = compute_clump_attributes(clump)
                if height == 0:
                    continue
                h_list.append(height)
                a_list.append(asym_ratio)
            print chain.clump_list[0].clump_db_entry.obsid, chain.clump_list[0].g_center, np.median(h_list), np.median(a_list)
            height_list.append(np.median(h_list))
            ratio_list.append(np.median(a_list))
        for thelist in [height_list, ratio_list]:
            print 'HT/AR'
            array = np.array(thelist)
            print 'MIN', np.min(array)
            print 'MAX', np.max(array)
            print 'MEAN', np.mean(array)
            print 'STD', np.std(array)
            plt.figure()
            plt.hist(array, 100)
    plt.show()
            
#analyze_approved_clumps()

#------------------------------------------------------------------------------------------------

voyager_clump_db_path = os.path.join(paper_root, 'voyager_clump_database.pickle')
v_clump_db_fp = open(voyager_clump_db_path, 'rb')
clump_find_options = pickle.load(v_clump_db_fp)
v_all_clump_db = pickle.load(v_clump_db_fp)
v_clump_db_fp.close()
for v_obs in v_all_clump_db.keys(): # Fix masking
    v_all_clump_db[v_obs].ew_data[np.where(v_all_clump_db[v_obs].ew_data == 0.)] = ma.masked

cassini_clump_db_path = os.path.join(ringutil.ROOT, 'clump-data', 'downsampled_clumpdb_137500_142500_05.000_0.020_10_01_137500_142500.pickle')
c_clump_db_fp = open(cassini_clump_db_path, 'rb')
clump_find_options = pickle.load(c_clump_db_fp)
c_all_clump_db = pickle.load(c_clump_db_fp)
c_clump_db_fp.close()

cassini_clump_db_path = os.path.join(ringutil.ROOT, 'clump-data', 'downsampled_clumpdb_137500_142500_05.000_0.020_10_01_137500_142500.pickle')
c_clump_db_fp = open(cassini_clump_db_path, 'rb')
clump_find_options = pickle.load(c_clump_db_fp)
c_orig_clump_db = pickle.load(c_clump_db_fp)
c_clump_db_fp.close()

choose_correct_single_clumps(v_all_clump_db)
choose_correct_single_clumps(c_all_clump_db)

v_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'voyager_approved_clumps_list.pickle')
v_approved_list_fp = open(v_approved_list_fp, 'rb')
v_approved_db, v_approved_list = pickle.load(v_approved_list_fp)
v_approved_list_fp.close()

c_approved_list_fp = os.path.join(paper_root, 'approved_list_w_errors.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()

#===============================================================================
# Things related to approved clumps 
#===============================================================================

# We MUST do this or else the clump_num field won't be initialized
# XXX 
dump_clump_table(c_approved_list, c_approved_db)
#dump_obs_table(c_approved_list, c_approved_db)

### These don't change - don't need to regenerate very often ###
#plot_FDG()
#plot_mexhat()
#plot_mean_clump()
#plot_long_coverage_over_time()
plot_sample_scalogram(c_orig_clump_db[OBSID_SAMPLE_PROFILE], c_all_clump_db[OBSID_SAMPLE_PROFILE])

### Clump profiles ###
#plot_clump_matrix(['C8', 'C18', 'C19', 'C21', 'C37', 'C44', 'C55', 'C72', 'C78'])
#plot_029rf_over_time()
#plot_splitting_clumps()

### Cassini vs. Voyager ###
plot_combined_vel_hist()
plot_voyager_cassini_comparison_profiles()

#===============================================================================
# Things related to single clumps 
#===============================================================================

plot_single_clump_distributions()
plot_clumps_vs_prometheus()

plot_appearing_and_disappearing_clumps([], [], ['C21'])
