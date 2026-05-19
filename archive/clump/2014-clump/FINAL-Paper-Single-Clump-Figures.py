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
import clump_gaussian_fit

debug = False
choose_smaller = False
choose_larger = True


paper_root = os.path.join(ringutil.PAPER_ROOT, 'Figures')
root_clump_db = {}

axes_color = (0,0,0)
#clump_color = (0.69, 0.13, 0.07 )
image_clump_color = (0.8, 0 , 0)
profile_color = 'black'
fdg_color = '#365B9E'

color_background = (1,1,1)
color_foreground = (0,0,0)
color_dark_grey = (0.5, 0.5, 0.5)
color_grey = (0.625, 0.625, 0.625)
color_bright_grey = (0.75, 0.75, 0.75)
markersize = 8.5
markersize_voyager = 3.5

blackpoint = 0.
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

def baseline_value(ew_data):
    sorted_ew_data = np.sort(ma.compressed(ew_data))
    num_valid = len(sorted_ew_data)
    perc_idx = int(num_valid * 0.15)
    return sorted_ew_data[perc_idx]
    
def baseline_normalize(ew_data):
    return ew_data / baseline_value(ew_data)
    

def plot_single_clump_distributions():
    for fake_name in ['nonfake']:#'all', 'nonfake', 'fake2006', 'fake2009']:
        plot_single_clump_distributions2(fake_name)

def plot_single_clump_distributions2(fake_name):
#    clump_heights_absnum_graph_max = 9.9
#    clump_bheights_absnum_graph_max = 49
#    heights_bins = np.arange(0., clump_heights_absnum_graph_max+1, 1)
#    bheights_bins = np.arange(0., clump_bheights_absnum_graph_max+5, 5)

    v_clump_frac = []
    v_clump_numbers = []
    v_clump_raw_numbers = []
    v_clump_scales = []
    v_clump_heights = []
    v_clump_bheights = []
    v_clump_pheights = []
    v_clump_pbheights = []
    v_stddevs = []
#    v_clump_heights_absnum = None
#    v_clump_bheights_absnum = None
    for v_obs in sorted(v_all_clump_db.keys()):
        long_res = 360./len(v_all_clump_db[v_obs].ew_data)
        num_not_clump = ma.MaskedArray.count(v_all_clump_db[v_obs].ew_data)
        v_ew_data = v_all_clump_db[v_obs].ew_data.copy() 
        v_ew_data = v_ew_data.view(ma.MaskedArray)
        v_mask = ma.getmaskarray(v_ew_data)
        empty = np.where(v_ew_data == 0.)[0]
        if empty != (): # This only handles the case of a single missing range of data - OK for Voyager
            v_mask[empty[0]-5:empty[-1]+5] = True # We only do this for the purpose of computing the baseline value
        v_ew_data.mask = v_mask
        bv = baseline_value(v_ew_data)
#        print v_obs, bv

        coverage = num_not_clump/float(len(v_all_clump_db[v_obs].ew_data))

#        heights = [clump.int_fit_height for clump in v_all_clump_db[v_obs].clump_list]
#        heights_bin = np.histogram(heights, heights_bins)[0] / coverage
#        if v_clump_heights_absnum is None:
#            v_clump_heights_absnum = heights_bin
#        else:
#            v_clump_heights_absnum += heights_bin
        
#        bheights = [clump.int_fit_height/bv for clump in v_all_clump_db[v_obs].clump_list]
#        bheights_bin = np.histogram(bheights, bheights_bins)[0] / coverage
#        if v_clump_bheights_absnum is None:
#            v_clump_bheights_absnum = bheights_bin
#        else:
#            v_clump_bheights_absnum += bheights_bin
             
        for clump in v_all_clump_db[v_obs].clump_list:
            v_clump_scales.append(clump.fit_width_deg)
            v_clump_heights.append(clump.int_fit_height)
            v_clump_bheights.append(clump.int_fit_height / bv)
            v_clump_pheights.append(clump.fit_height)
            v_clump_pbheights.append(clump.fit_height / bv)
            clump_width = (clump.fit_right_deg-clump.fit_left_deg)
            if clump_width < 0: clump_width += 360
            num_not_clump -= clump_width / long_res
        v_clump_frac.append(1-(float(num_not_clump) / ma.MaskedArray.count(v_all_clump_db[v_obs].ew_data)))
        if coverage >= 0.8:
            v_clump_numbers.append(len(v_all_clump_db[v_obs].clump_list) / coverage)
        v_clump_raw_numbers.append(len(v_all_clump_db[v_obs].clump_list))
        v_stddevs.append(np.std(v_all_clump_db[v_obs].ew_data))

#    v_clump_heights_absnum /= len(v_all_clump_db.keys())
#    v_clump_bheights_absnum /= len(v_all_clump_db.keys())
    
    v_clump_bheights = np.array(v_clump_bheights)

    v_clump_frac_stats = (np.mean(v_clump_frac),
                          np.std(v_clump_frac),
                          np.min(v_clump_frac),
                          np.max(v_clump_frac))
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
    v_clump_bheight_stats = (np.mean(v_clump_bheights),
                            np.std(v_clump_bheights),
                            np.min(v_clump_bheights),
                            np.max(v_clump_bheights))
    v_clump_pheight_stats = (np.mean(v_clump_pheights),
                            np.std(v_clump_pheights),
                            np.min(v_clump_pheights),
                            np.max(v_clump_pheights))
    v_clump_pbheight_stats = (np.mean(v_clump_pbheights),
                            np.std(v_clump_pbheights),
                            np.min(v_clump_pbheights),
                            np.max(v_clump_pbheights))
    v_stddev_stats = (np.mean(v_stddevs),
                      np.std(v_stddevs),
                      np.min(v_stddevs),
                      np.max(v_stddevs))

    c_clump_frac = []
    c_clump_numbers = []
    c_clump_raw_numbers = []
    c_clump_scales = []
    c_clump_heights = []
    c_clump_bheights = []
    c_clump_pheights = []
    c_clump_pbheights = []
    c_stddevs = []
    c_ets = []
#    c_clump_heights_absnum = None
#    c_clump_bheights_absnum = None
    
    for c_obs in sorted(c_all_clump_db.keys()):
        long_res = 360./len(c_all_clump_db[c_obs].ew_data)
        num_not_clump = ma.MaskedArray.count(c_all_clump_db[c_obs].ew_data)
        bv = baseline_value(c_all_clump_db[c_obs].ew_data)
#        print c_obs, bv
        num_clumps = 0
        new_clump_list = []
        for clump in c_all_clump_db[c_obs].clump_list:
            if fake_name[:4] == 'fake':
                if clump.wave_type != fake_name:
                    continue
            elif fake_name == 'nonfake' and clump.wave_type[:4] == 'fake':
                continue
            num_clumps += 1
            new_clump_list.append(clump)
#            print clump.clump_db_entry.obsid, clump.clump_db_entry.incidence_angle
            if clump.clump_db_entry.incidence_angle <= 87 or fake_name == 'fake2009':
                c_clump_scales.append(clump.fit_width_deg)
                c_clump_heights.append(clump.int_fit_height)
                c_clump_bheights.append(clump.int_fit_height / bv)
                c_clump_pheights.append(clump.fit_height)
                c_clump_pbheights.append(clump.fit_height / bv)
            clump_width = (clump.fit_right_deg-clump.fit_left_deg)
            if clump_width < 0: clump_width += 360
            num_not_clump -= clump_width / long_res

        coverage = ma.MaskedArray.count(c_all_clump_db[c_obs].ew_data)/float(len(c_all_clump_db[c_obs].ew_data))

#        heights = [clump.int_fit_height for clump in new_clump_list]
#        heights_bin = np.histogram(heights, heights_bins)[0] / coverage
#        if c_clump_heights_absnum is None:
#            c_clump_heights_absnum = heights_bin
#        else:
#            c_clump_heights_absnum += heights_bin
#        bheights = [clump.int_fit_height/bv for clump in new_clump_list]
#        bheights_bin = np.histogram(bheights, bheights_bins)[0] / coverage
#        if c_clump_bheights_absnum is None:
#            c_clump_bheights_absnum = bheights_bin
#        else:
#            c_clump_bheights_absnum += bheights_bin
             
        c_clump_frac.append(1-(float(num_not_clump) / ma.MaskedArray.count(c_all_clump_db[c_obs].ew_data)))
#        print c_obs, c_clump_frac[-1]
#        plot_single_clumps(c_all_clump_db[c_obs], c_all_clump_db[c_obs].clump_list, c_obs)

        if coverage >= 0.8:
            c_clump_numbers.append(num_clumps / coverage)
            c_ets.append(c_all_clump_db[c_obs].et_min)
        c_clump_raw_numbers.append(len(c_all_clump_db[c_obs].clump_list))
        c_stddevs.append(np.std(c_all_clump_db[c_obs].ew_data))

#    c_clump_heights_absnum /= len(c_all_clump_db.keys())
#    c_clump_bheights_absnum /= len(c_all_clump_db.keys())

    c_clump_bheights = np.array(c_clump_bheights)
    
    c_clump_frac_stats = (np.mean(c_clump_frac),
                          np.std(c_clump_frac),
                          np.min(c_clump_frac),
                          np.max(c_clump_frac))
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
    c_clump_bheight_stats = (np.mean(c_clump_bheights),
                            np.std(c_clump_bheights),
                            np.min(c_clump_bheights),
                            np.max(c_clump_bheights))
    c_clump_pheight_stats = (np.mean(c_clump_pheights),
                            np.std(c_clump_pheights),
                            np.min(c_clump_pheights),
                            np.max(c_clump_pheights))
    c_clump_pbheight_stats = (np.mean(c_clump_pbheights),
                            np.std(c_clump_pbheights),
                            np.min(c_clump_pbheights),
                            np.max(c_clump_pbheights))
    c_stddev_stats = (np.mean(c_stddevs),
                      np.std(c_stddevs),
                      np.min(c_stddevs),
                      np.max(c_stddevs))
    
    print
    print
    print '-----------------------STATS TABLE', fake_name, '--------------------------'
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
    if fake_name != 'fake2009':
        print '(Using Absolute Photometry)'
        print 'AVG Clump Int Height  |  %7.4f | %7.4f  '%(v_clump_height_stats[0], c_clump_height_stats[0])
        print 'STD Clump Int Height  |  %7.4f | %7.4f  '%(v_clump_height_stats[1], c_clump_height_stats[1])
        print 'MIN Clump Int Height  |  %7.4f | %7.4f  '%(v_clump_height_stats[2], c_clump_height_stats[2])
        print 'MAX Clump Int Height  |  %7.4f | %7.4f  '%(v_clump_height_stats[3], c_clump_height_stats[3])
        print
        print 'AVG Clump Peak Height |  %7.4f | %7.4f  '%(v_clump_pheight_stats[0], c_clump_pheight_stats[0])
        print 'STD Clump Peak Height |  %7.4f | %7.4f  '%(v_clump_pheight_stats[1], c_clump_pheight_stats[1])
        print 'MIN Clump Peak Height |  %7.4f | %7.4f  '%(v_clump_pheight_stats[2], c_clump_pheight_stats[2])
        print 'MAX Clump Peak Height |  %7.4f | %7.4f  '%(v_clump_pheight_stats[3], c_clump_pheight_stats[3])
        print
    print '(Using Baseline Photometry)'
    print 'AVG Clump Int Height  |  %7.4f | %7.4f  '%(v_clump_bheight_stats[0], c_clump_bheight_stats[0])
    print 'STD Clump Int Height  |  %7.4f | %7.4f  '%(v_clump_bheight_stats[1], c_clump_bheight_stats[1])
    print 'MIN Clump Int Height  |  %7.4f | %7.4f  '%(v_clump_bheight_stats[2], c_clump_bheight_stats[2])
    print 'MAX Clump Int Height  |  %7.4f | %7.4f  '%(v_clump_bheight_stats[3], c_clump_bheight_stats[3])
    print
    print 'AVG Clump Peak Height |  %7.4f | %7.4f  '%(v_clump_pbheight_stats[0], c_clump_pbheight_stats[0])
    print 'STD Clump Peak Height |  %7.4f | %7.4f  '%(v_clump_pbheight_stats[1], c_clump_pbheight_stats[1])
    print 'MIN Clump Peak Height |  %7.4f | %7.4f  '%(v_clump_pbheight_stats[2], c_clump_pbheight_stats[2])
    print 'MAX Clump Peak Height |  %7.4f | %7.4f  '%(v_clump_pbheight_stats[3], c_clump_pbheight_stats[3])
    print
    print 'AVG OBSID STDDEV      |  %5.2f   |  %5.2f  '%(v_stddev_stats[0], c_stddev_stats[0])
    print 'STD OBSID STDDEV      |  %5.2f   |  %5.2f  '%(v_stddev_stats[1], c_stddev_stats[1])
    print 'MIN OBSID STDDEV      |  %5.2f   |  %5.2f  '%(v_stddev_stats[2], c_stddev_stats[2])
    print 'MAX OBSID STDDEV      |  %5.2f   |  %5.2f  '%(v_stddev_stats[3], c_stddev_stats[3])
    print
    print 'AVG Clump Coverage    |  %5.3f   |  %5.3f  '%(v_clump_frac_stats[0], c_clump_frac_stats[0])
    print 'STD Clump Coverage    |  %5.3f   |  %5.3f  '%(v_clump_frac_stats[1], c_clump_frac_stats[1])
    print 'MIN Clump Coverage    |  %5.3f   |  %5.3f  '%(v_clump_frac_stats[2], c_clump_frac_stats[2])
    print 'MAX Clump Coverage    |  %5.3f   |  %5.3f  '%(v_clump_frac_stats[3], c_clump_frac_stats[3])
    print
    print '------------------------------------------------------------'
    print
    print 'TOTAL NUMBER OF CASSINI CLUMPS (WEIGHTED):', np.sum(c_clump_numbers)
    print 'TOTAL NUMBER OF VOYAGER CLUMPS (WEIGHTED):', np.sum(v_clump_numbers)
    print
    print 'TOTAL NUMBER OF CASSINI CLUMPS (RAW):', np.sum(c_clump_raw_numbers)
    print 'TOTAL NUMBER OF VOYAGER CLUMPS (RAW):', np.sum(v_clump_raw_numbers)
    print
    print 'TOTAL NUMBER OF CASSINI CLUMPS (NON-EQUINOX):', len(c_clump_scales)
    print 'TOTAL NUMBER OF VOYAGER CLUMPS (NON-EQUINOX):', len(v_clump_scales)
    print

################
    
    # Width comparison
    
    step = 2.5
    graph_min = 0.0
    graph_max = (int(max(np.max(c_clump_scales), np.max(v_clump_scales)) / step) + 1) * step
    bins = np.arange(0, graph_max+step,step)

    figure_size = (3.3, 2.0)
    fig = plt.figure(figsize = figure_size)
    ax = fig.add_subplot(111)
    plt.subplots_adjust(top = .95, bottom = 0.125, left = 0.08, right = 0.98)
    
    ax.set_xlabel('Clump Width ( $\mathbf{^o}$)')
    ax.set_ylabel('Fractional Number of Clumps')
    c_scale_weights = np.zeros_like(c_clump_scales) + 1./len(c_clump_scales)
    v_scale_weights = np.zeros_like(v_clump_scales) + 1./len(v_clump_scales)
    x_max = (np.max(c_clump_scales) // step) * step
    plt.xlim(0, x_max)
    ax.get_xaxis().set_ticks(np.arange(0, x_max+step, step*2))
    
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    scale_counts, bins, patches = plt.hist([v_clump_scales, c_clump_scales], bins,
                                      weights = [v_scale_weights, c_scale_weights], label = ['Voyager', 'Cassini'], color = [color_grey, 'black'], lw = 0.0)
    
    ax.get_yaxis().set_ticks(np.arange(0.0, 0.4 + 0.1, 0.1))
    leg = plt.legend()
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
    save_fig(fig, ax, 'clump_width_dist_'+fake_name+'.png', leg)   

#################
    
    # Brightness comparison - absolute, abs number averaged by profile
    
#    fig2 = plt.figure(figsize = figure_size)
#    ax = fig2.add_subplot(111)
#    plt.subplots_adjust(top = .95, bottom = 0.125, left = 0.08, right = 0.98)
#    ax.set_xlabel(r'Phase-Normalized Brightness ( km $\mathbf{^o}$)')
#    ax.set_ylabel('Number of Clumps')
#    
#    graph_max = clump_heights_absnum_graph_max
#    graph_min = 0
#    
#    fig2 = plt.figure(figsize = figure_size)
#    ax = fig2.add_subplot(111)
#    plt.subplots_adjust(top = .95, bottom = 0.125, left = 0.08, right = 0.98)
#    ax.set_xlabel(r'Phase-Normalized Brightness ( km $\mathbf{^o}$)')
#    ax.set_ylabel('Number of Clumps')
#
#    step = 1
#    ax.get_xaxis().set_ticks(np.arange(0.,graph_max+2., 2.0))
#    ax.set_xlim(0,((graph_max//10+1)*10))
#    ax.xaxis.tick_bottom()
#    ax.yaxis.tick_left()
#    
#    bar_width = 0.45
#    bin_centers = (heights_bins[1:] + heights_bins[:-1])/2
#    
#    print bin_centers
#    print v_clump_heights_absnum
#    
#    plt.bar(bin_centers-bar_width/2, v_clump_heights_absnum, bar_width, 
#                                      label = 'Voyager', color = color_grey, lw = 0.0)
#    plt.bar(bin_centers+bar_width/2, c_clump_heights_absnum, bar_width, 
#                                      label = 'Cassini', color = 'black', lw = 0.0)
#    
#    leg = plt.legend()
#    leg.get_frame().set_alpha(0.0)
#    leg.get_frame().set_visible(False)
#    save_fig(fig2, ax, 'clump_brightnessnum_dist_'+fake_name+'.png', leg)
#
#    # Brightness comparison - baseline, abs number averaged by profile
#    
#    fig2 = plt.figure(figsize = figure_size)
#    ax = fig2.add_subplot(111)
#    plt.subplots_adjust(top = .95, bottom = 0.125, left = 0.08, right = 0.98)
#    ax.set_xlabel(r'Phase-Normalized Brightness ( km $\mathbf{^o}$)')
#    ax.set_ylabel('Number of Clumps')
#    
#    graph_max = clump_bheights_absnum_graph_max
#    graph_min = 0
#    
#    fig2 = plt.figure(figsize = figure_size)
#    ax = fig2.add_subplot(111)
#    plt.subplots_adjust(top = .95, bottom = 0.125, left = 0.08, right = 0.98)
#    ax.set_xlabel(r'Baseline-Normalized Brightness ( km $\mathbf{^o}$)')
#    ax.set_ylabel('Number of Clumps')
#
#    step = 5
#    ax.get_xaxis().set_ticks(np.arange(0.,graph_max+2., 10.0))
#    ax.set_xlim(0,((graph_max//10+1)*10))
#    ax.xaxis.tick_bottom()
#    ax.yaxis.tick_left()
#    
#    bar_width = 0.45*5
#    bin_centers = (bheights_bins[1:] + bheights_bins[:-1])/2
#    
#    print bin_centers
#    print v_clump_bheights_absnum
#    
#    plt.bar(bin_centers-bar_width/2, v_clump_bheights_absnum, bar_width, 
#                                      label = 'Voyager', color = color_grey, lw = 0.0)
#    plt.bar(bin_centers+bar_width/2, c_clump_bheights_absnum, bar_width, 
#                                      label = 'Cassini', color = 'black', lw = 0.0)
#    
#    leg = plt.legend()
#    leg.get_frame().set_alpha(0.0)
#    leg.get_frame().set_visible(False)
#    save_fig(fig2, ax, 'clump_brightnessnum_dist_baseline_'+fake_name+'.png', leg)

#####################

    # Brightness comparison - absolute
    
    for zoom_level, hist_step, plot_step, plot_xmin, plot_xmax, plot_ymax in [('dim',0.2,1.,0,3,0.3001),('bright',0.5,1.,3,9,0.016),('full',0.5,2.,0,10,None)]:
        fig2 = plt.figure(figsize = figure_size)
        ax = fig2.add_subplot(111)
        plt.subplots_adjust(top = .95, bottom = 0.125, left = 0.08, right = 0.98)
        ax.set_xlabel(r'Phase-Normalized Integrated Brightness ( km $\mathbf{^o}$)')
        ax.set_ylabel('Fractional Number of Clumps')
        
        v_min = np.min(v_clump_heights)
        v_max = np.max(v_clump_heights)
        c_max = np.max(c_clump_heights)
        c_min = np.min(c_clump_heights)

        graph_max = max(c_max, v_max)
        graph_min = 0 # min(c_min, v_min)
        
        bins = np.arange(graph_min,graph_max + hist_step, hist_step)
        
        if plot_xmin is None:
            ax.get_xaxis().set_ticks(np.arange(0.,graph_max+plot_step, plot_step))
            ax.set_xlim(0,((graph_max//plot_step+1)*plot_step))
        else:
            ax.get_xaxis().set_ticks(np.arange(plot_xmin,plot_xmax+plot_step, plot_step))
            ax.set_xlim(plot_xmin, plot_xmax)
        if plot_ymax is not None:
            ax.set_ylim(0, plot_ymax)
            if zoom_level == 'bright':
                ax.get_yaxis().set_ticks([0., 0.01])
        yFormatter = FormatStrFormatter('%.2f')
        ax.yaxis.set_major_formatter(yFormatter)    
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        
        c_height_weights = np.zeros_like(c_clump_heights) + 1./len(c_clump_heights)
        v_height_weights = np.zeros_like(v_clump_heights) + 1./len(v_clump_heights)
        
        height_counts, bins, patches = plt.hist([v_clump_heights, c_clump_heights], bins,
                 weights = [v_height_weights, c_height_weights], label = ['Voyager', 'Cassini'], color = [color_grey, 'black'], lw = 0.0)
        
    #    print height_counts
    #    print bins
        
        leg = plt.legend()
        leg.get_frame().set_alpha(0.0)
        leg.get_frame().set_visible(False)
        save_fig(fig2, ax, 'clump_brightness_dist_'+zoom_level+'_'+fake_name+'.png', leg)

    # Brightness comparison - baseline
    
    for zoom_level, hist_step, plot_step, plot_xmin, plot_xmax, plot_ymax in [('dim',0.25,1.,0,5,0.25),('bright',2.5,10.,5,45,0.08),('full',2.5,10.,None,None,None)]:
        fig2 = plt.figure(figsize = figure_size)
        ax = fig2.add_subplot(111)
        plt.subplots_adjust(top = .95, bottom = 0.125, left = 0.08, right = 0.98)
        ax.set_xlabel(r'Baseline-Normalized Integrated Brightness ( $\mathbf{^o})$')
        ax.set_ylabel('Fractional Number of Clumps')
        
        v_min = np.min(v_clump_bheights)
        v_max = np.max(v_clump_bheights)
        c_max = np.max(c_clump_bheights)
        c_min = np.min(c_clump_bheights)
        
        graph_max = max(c_max, v_max)
        graph_min = 0 # min(c_min, v_min)
        
    #    print graph_min, graph_max
        
        bins = np.arange(graph_min,graph_max + hist_step, hist_step)
        if plot_xmin is None:
            ax.get_xaxis().set_ticks(np.arange(graph_min,graph_max+plot_step, plot_step))
            ax.set_xlim(0,((graph_max//plot_step+1)*plot_step))
        else:
            ax.get_xaxis().set_ticks(np.arange(plot_xmin,plot_xmax+plot_step, plot_step))
            ax.set_xlim(plot_xmin, plot_xmax)
        if plot_ymax is not None:
            ax.set_ylim(0, plot_ymax)
        yFormatter = FormatStrFormatter('%.2f')
        ax.yaxis.set_major_formatter(yFormatter)    
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        
        c_height_weights = np.zeros_like(c_clump_bheights) + 1./len(c_clump_bheights)
        v_height_weights = np.zeros_like(v_clump_bheights) + 1./len(v_clump_bheights)
        
        bheight_counts, bins, patches = plt.hist([v_clump_bheights, c_clump_bheights], bins,
                 weights = [v_height_weights, c_height_weights], label = ['Voyager', 'Cassini'], color = [color_grey, 'black'], lw = 0.0)
        
        leg = plt.legend()
        leg.get_frame().set_alpha(0.0)
        leg.get_frame().set_visible(False)
        save_fig(fig2, ax, 'clump_brightness_baseline_dist_'+zoom_level+'_'+fake_name+'.png', leg)

    # Brightness comparison - absolute peak
    
    for zoom_level, hist_step, plot_step, plot_xmin, plot_xmax, plot_ymax in [('dim',0.025,0.1,0,0.301,0.25),('bright',0.05,0.2,0.3,1.1,0.04),('full',0.1,0.25,0,1.0,None)]:
        fig2 = plt.figure(figsize = figure_size)
        ax = fig2.add_subplot(111)
        plt.subplots_adjust(top = .95, bottom = 0.125, left = 0.08, right = 0.98)
        ax.set_xlabel(r'Phase-Normalized Peak Brightness (km)')
        ax.set_ylabel('Fractional Number of Clumps')
        
        v_min = np.min(v_clump_pheights)
        v_max = np.max(v_clump_pheights)
        c_max = np.max(c_clump_pheights)
        c_min = np.min(c_clump_pheights)
        
        graph_max = max(c_max, v_max)
        graph_min = 0 # min(c_min, v_min)
        
    #    print graph_min, graph_max
        
        bins = np.arange(graph_min,graph_max + hist_step, hist_step)
        if plot_xmin is None:
            ax.get_xaxis().set_ticks(np.arange(graph_min,graph_max+plot_step, plot_step))
            ax.set_xlim(0,((graph_max//plot_step+1)*plot_step))
        else:
            ax.get_xaxis().set_ticks(np.arange(plot_xmin,plot_xmax+plot_step, plot_step))
            ax.set_xlim(plot_xmin, plot_xmax)
        if plot_ymax is not None:
            ax.set_ylim(0, plot_ymax)
            if zoom_level == 'bright':
                ax.get_yaxis().set_ticks([0., 0.01, 0.02, 0.03, 0.04])            
        yFormatter = FormatStrFormatter('%.2f')
        ax.yaxis.set_major_formatter(yFormatter)    
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        
        c_height_weights = np.zeros_like(c_clump_pheights) + 1./len(c_clump_pheights)
        v_height_weights = np.zeros_like(v_clump_pheights) + 1./len(v_clump_pheights)
        
        pheight_counts, bins, patches = plt.hist([v_clump_pheights, c_clump_pheights], bins,
                 weights = [v_height_weights, c_height_weights], label = ['Voyager', 'Cassini'], color = [color_grey, 'black'], lw = 0.0)
        
        leg = plt.legend()
        leg.get_frame().set_alpha(0.0)
        leg.get_frame().set_visible(False)
        save_fig(fig2, ax, 'clump_peak_dist_'+zoom_level+'_'+fake_name+'.png', leg)

    # Brightness comparison - baseline peak
    
    for zoom_level, hist_step, plot_step, plot_xmin, plot_xmax, plot_ymax in [('dim',0.1,0.2,0,1,0.4),('bright',0.5,1,1,6,0.05),('full',0.55,2,0,6,1)]:
        fig2 = plt.figure(figsize = figure_size)
        ax = fig2.add_subplot(111)
        plt.subplots_adjust(top = .95, bottom = 0.125, left = 0.08, right = 0.98)
        ax.set_xlabel(r'Baseline-Normalized Peak Brightness')
        ax.set_ylabel('Fractional Number of Clumps')
        
        v_min = np.min(v_clump_pbheights)
        v_max = np.max(v_clump_pbheights)
        c_max = np.max(c_clump_pbheights)
        c_min = np.min(c_clump_pbheights)
        
        graph_max = max(c_max, v_max)
        graph_min = 0 # min(c_min, v_min)
        
    #    print graph_min, graph_max
        
        bins = np.arange(graph_min,graph_max + hist_step, hist_step)
        if plot_xmin is None:
            ax.get_xaxis().set_ticks(np.arange(graph_min,graph_max+plot_step, plot_step))
            ax.set_xlim(0,((graph_max//plot_step+1)*plot_step))
        else:
            ax.get_xaxis().set_ticks(np.arange(plot_xmin,plot_xmax+plot_step, plot_step))
            ax.set_xlim(plot_xmin, plot_xmax)
        if plot_ymax is not None:
            ax.set_ylim(0, plot_ymax)
            if zoom_level == 'bright':
                ax.get_yaxis().set_ticks([0., 0.01, 0.02, 0.03, 0.04, 0.05])
        yFormatter = FormatStrFormatter('%.2f')
        ax.yaxis.set_major_formatter(yFormatter)    
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        
        c_height_weights = np.zeros_like(c_clump_pbheights) + 1./len(c_clump_pbheights)
        v_height_weights = np.zeros_like(v_clump_pbheights) + 1./len(v_clump_pbheights)
        
        pheight_counts, bins, patches = plt.hist([v_clump_pbheights, c_clump_pbheights], bins,
                 weights = [v_height_weights, c_height_weights], label = ['Voyager', 'Cassini'], color = [color_grey, 'black'], lw = 0.0)
        
        leg = plt.legend()
        leg.get_frame().set_alpha(0.0)
        leg.get_frame().set_visible(False)
        save_fig(fig2, ax, 'clump_peak_baseline_dist_'+zoom_level+'_'+fake_name+'.png', leg)

    # Width vs. Absolute Brightness
    
    def compute_sigma_b(x, y, coeffs):
        x = np.array(x)
        y = np.array(y)
        delta = len(x) * np.sum(x**2) - np.sum(x)**2
        sigma_y = np.sqrt(np.sum((y-coeffs[1]-coeffs[0]*x)**2)/(len(c_clump_scales)-2))
        sigma_b = sigma_y * np.sqrt(len(c_clump_scales) / delta)
        return sigma_b

    fig = plt.figure(figsize = (8.6,4.5))
    fig.subplots_adjust(hspace=0.3)
    ax = fig.add_subplot(221)
    plt.plot(c_clump_scales, c_clump_heights, 'o', mec='black', mfc='none', ms=5,
             label=' ')
    coeffs = np.polyfit(c_clump_scales, c_clump_heights, 1)
    print 'C W/H', fake_name, 'coeffs', coeffs
    print 'SIGMA_B', compute_sigma_b(c_clump_scales, c_clump_heights, coeffs)

    plt.plot(v_clump_scales, v_clump_heights, '+', color='black', ms=5,
             label=' ')

    plt.plot([np.min(c_clump_scales),np.max(c_clump_scales)],
             [np.polyval(coeffs, np.min(c_clump_scales)), np.polyval(coeffs, np.max(c_clump_scales))], '-', color='black',
             label='Cassini')

    coeffs = np.polyfit(v_clump_scales, v_clump_heights, 1)
    print 'V W/H coeffs', coeffs
    print 'SIGMA_B', compute_sigma_b(v_clump_scales, v_clump_heights, coeffs)
    plt.plot([np.min(v_clump_scales),np.max(v_clump_scales)],
             [np.polyval(coeffs, np.min(v_clump_scales)), np.polyval(coeffs, np.max(v_clump_scales))], '-', color=color_grey,
             label='Voyager')
    ax.set_xlabel(r'Width ( $\mathbf{^o}$)')
    ax.set_ylabel(r'Phase-Norm Int Brightness ( km $\mathbf{^o}$)')
#    ax.set_ylim(0, (max(np.max(c_clump_heights), np.max(v_clump_heights))//2+1)*2)
    ax.set_ylim(0, 14)
    
    legend = plt.legend(numpoints=1, ncol=2, columnspacing=-1)
    legend.get_frame().set_alpha(0.0)
    legend.get_frame().set_visible(False)

    legend.get_frame().set_facecolor(color_background)
    legend.get_frame().set_edgecolor(color_background)
    for text in legend.get_texts():
        text.set_color(color_foreground) 

#    save_fig(fig, ax, 'width_vs_brightness_'+fake_name+'.png', leg)


    # Width vs. Baseline Brightness
    
    ax = fig.add_subplot(222)
    plt.plot(c_clump_scales, c_clump_bheights, 'o', mec='black', mfc='none', ms=5,
             label=' ')
    coeffs = np.polyfit(c_clump_scales, c_clump_bheights, 1)
    print 'C W/H', fake_name, 'coeffs', coeffs,
    print 'RESID', np.sqrt(np.sum((c_clump_bheights-np.polyval(coeffs, c_clump_scales))**2))
    print 'SIGMA_B', compute_sigma_b(c_clump_scales, c_clump_bheights, coeffs)
    
    plt.plot(v_clump_scales, v_clump_bheights, '+', color='black', ms=5,
             label=' ')

    plt.plot([np.min(c_clump_scales),np.max(c_clump_scales)],
             [np.polyval(coeffs, np.min(c_clump_scales)), np.polyval(coeffs, np.max(c_clump_scales))], '-', color='black',
             label='Cassini')

    coeffs = np.polyfit(v_clump_scales, v_clump_bheights, 1)
    print 'V W/H coeffs', coeffs,
    print 'RESID', np.sqrt(np.sum((v_clump_bheights-np.polyval(coeffs, v_clump_scales))**2))
    print 'SIGMA_B', compute_sigma_b(v_clump_scales, v_clump_bheights, coeffs)
    plt.plot([np.min(v_clump_scales),np.max(v_clump_scales)],
             [np.polyval(coeffs, np.min(v_clump_scales)), np.polyval(coeffs, np.max(v_clump_scales))], '-', color=color_grey,
             label='Voyager')
    ax.set_xlabel(r'Width ( $\mathbf{^o}$)')
    ax.set_ylabel(r'Baseline-Norm Int Brightness ( $\mathbf{^o}$)')
    ax.set_ylim(0, (max(np.max(c_clump_bheights), np.max(v_clump_bheights))//5+1)*5)

    legend = plt.legend(numpoints=1, ncol=2, columnspacing=-1)
    legend.get_frame().set_alpha(0.0)
    legend.get_frame().set_visible(False)

    legend.get_frame().set_facecolor(color_background)
    legend.get_frame().set_edgecolor(color_background)
    for text in legend.get_texts():
        text.set_color(color_foreground) 

#    save_fig(fig, ax, 'width_vs_brightness_baseline_'+fake_name+'.png', leg)


    # Width vs. Peak Absolute Brightness
    
    ax = fig.add_subplot(223)
    plt.plot(c_clump_scales, c_clump_pheights, 'o', mec='black', mfc='none', ms=5,
             label=' ')
    coeffs = np.polyfit(c_clump_scales, c_clump_pheights, 1)
    print 'C W/H', fake_name, 'coeffs', coeffs
    print 'SIGMA_B', compute_sigma_b(c_clump_scales, c_clump_pheights, coeffs)

    plt.plot(v_clump_scales, v_clump_pheights, '+', color='black', ms=5,
             label=' ')
    
    plt.plot([np.min(c_clump_scales),np.max(c_clump_scales)],
             [np.polyval(coeffs, np.min(c_clump_scales)), np.polyval(coeffs, np.max(c_clump_scales))], '-', color='black',
             label='Cassini')

    coeffs = np.polyfit(v_clump_scales, v_clump_pheights, 1)
    print 'V W/H coeffs', coeffs
    print 'SIGMA_B', compute_sigma_b(v_clump_scales, v_clump_pheights, coeffs)
    plt.plot([np.min(v_clump_scales),np.max(v_clump_scales)],
             [np.polyval(coeffs, np.min(v_clump_scales)), np.polyval(coeffs, np.max(v_clump_scales))], '-', color=color_grey,
             label='Voyager')
    ax.set_xlabel(r'Width ( $\mathbf{^o}$)')
    ax.set_ylabel(r'Phase-Norm Peak Brightness (km)')
#    ax.set_ylim(0, (max(np.max(c_clump_pheights), np.max(v_clump_pheights))//2+1)*2)

    legend = plt.legend(numpoints=1, ncol=2, columnspacing=-1)
    legend.get_frame().set_alpha(0.0)
    legend.get_frame().set_visible(False)

    legend.get_frame().set_facecolor(color_background)
    legend.get_frame().set_edgecolor(color_background)
    for text in legend.get_texts():
        text.set_color(color_foreground) 

#    save_fig(fig, ax, 'width_vs_peak_brightness_'+fake_name+'.png', leg)


    # Width vs. Peak Baseline Brightness
    
    ax = fig.add_subplot(224)
    plt.plot(c_clump_scales, c_clump_pbheights, 'o', mec='black', mfc='none', ms=5,
             label=' ')
    coeffs = np.polyfit(c_clump_scales, c_clump_pbheights, 1)
    print 'C W/H', fake_name, 'coeffs', coeffs,
    print 'RESID', np.sqrt(np.sum((c_clump_pbheights-np.polyval(coeffs, c_clump_scales))**2))
    print 'SIGMA_B', compute_sigma_b(c_clump_scales, c_clump_pbheights, coeffs)

    plt.plot(v_clump_scales, v_clump_pbheights, '+', color='black', ms=5,
             label=' ')

    plt.plot([np.min(c_clump_scales),np.max(c_clump_scales)],
             [np.polyval(coeffs, np.min(c_clump_scales)), np.polyval(coeffs, np.max(c_clump_scales))], '-', color='black',
             label='Cassini')

    coeffs = np.polyfit(v_clump_scales, v_clump_pbheights, 1)
    print 'V W/H coeffs', coeffs,
    print 'RESID', np.sqrt(np.sum((v_clump_pbheights-np.polyval(coeffs, v_clump_scales))**2))
    print 'SIGMA_B', compute_sigma_b(v_clump_scales, v_clump_pbheights, coeffs)
    plt.plot([np.min(v_clump_scales),np.max(v_clump_scales)],
             [np.polyval(coeffs, np.min(v_clump_scales)), np.polyval(coeffs, np.max(v_clump_scales))], '-', color=color_grey,
             label='Voyager')
    ax.set_xlabel(r'Width ( $\mathbf{^o}$)')
    ax.set_ylabel(r'Baseline-Norm Peak Brightness')
#    ax.set_ylim(0, (max(np.max(c_clump_pbheights), np.max(v_clump_pbheights))//2+1)*2)

    legend = plt.legend(numpoints=1, ncol=2, columnspacing=-1)
    legend.get_frame().set_alpha(0.0)
    legend.get_frame().set_visible(False)

    legend.get_frame().set_facecolor(color_background)
    legend.get_frame().set_edgecolor(color_background)
    for text in legend.get_texts():
        text.set_color(color_foreground) 

    save_fig(fig, ax, 'width_vs_brightness_matrix_'+fake_name+'.png')


    # Number of clumps over time
    
    fig = plt.figure(figsize = (3.5,3.0))
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
#    ax.set_ylim(20., 55.)
    plt.plot(c_ets, c_clump_numbers, '.', color = 'black')

    coeffs = np.polyfit(c_ets, c_clump_numbers, 1)
    x_range = np.arange(np.min(c_ets), np.max(c_ets), 86400*10)
    plt.plot(x_range, np.polyval(coeffs, x_range), '-', color='red')

    ax.set_ylabel('Weighted Clumps per Observation')
    ax.set_xlabel('Date')
    fig.tight_layout()
    save_fig(fig, ax, 'num_clumps_vs_time_'+fake_name+'.png')

    # Number of clumps per ID

    for spacecraft, clump_numbers in [('Voyager', v_clump_numbers), ('Cassini', c_clump_numbers)]:
        fig = plt.figure(figsize = (3.5,2))
        ax = fig.add_subplot(111)
        ax.set_xlabel('Weighted Number of Clumps per Observation')
        ax.set_ylabel('Fractional Number of Observations')
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        
        step = 2
        bin_min = (np.min(clump_numbers) // step - 1) * step 
        bin_max = (np.max(clump_numbers) // step + 1) * step
        bins = np.arange(bin_min, bin_max, step)  # Bins are semi-open interval [x,y)
        clump_numbers = np.array(clump_numbers)
        num_weights = np.zeros_like(clump_numbers) + 1./len(clump_numbers)
        plt.xlim(bin_min, bin_max)
        ax.get_xaxis().set_ticks(np.arange(bin_min, bin_max+step, step*2))
        
        counts, xbins, patches = plt.hist(clump_numbers, bins, weights = num_weights, color = color_grey, lw = 0.0)
    
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

        plt.plot(pbins+(step-1)/2., dist, '-', color='black', lw=2)
        save_fig(fig, ax, 'clump_numbers_dist_'+spacecraft+'_'+fake_name+'.png', leg)
        
    v_scale_dist = scale_counts[0]
    c_scale_dist = scale_counts[1]
    
    graph_min = np.min(c_clump_numbers)
    graph_max = np.max(c_clump_numbers)
    step = 1.0
    bins = np.arange(0,graph_max+step,step)

    figure_size = (10., 4.0)
    fig = plt.figure(figsize = figure_size)
    ax = fig.add_subplot(111)
    
    c_num_weights = np.zeros_like(c_clump_numbers) + 1./len(c_clump_numbers)
    v_num_weights = np.zeros_like(v_clump_numbers) + 1./len(v_clump_numbers)
    
    num_counts, bins, patches = plt.hist([v_clump_numbers, c_clump_numbers], bins,
                                      weights = [v_num_weights, c_num_weights], label = ['Voyager', 'Cassini'], color = ['#AC7CC9', 'black'], lw = 0.0)

    v_num_dist = num_counts[0]
    c_num_dist = num_counts[1]

    clump_num_d, clump_num_pks = st.ks_2samp(c_clump_numbers, v_clump_numbers)
    clump_scale_d, clump_scale_pks = st.ks_2samp(v_clump_scales, c_clump_scales)
    clump_height_d, clump_height_pks = st.ks_2samp(v_clump_heights, c_clump_heights)
    clump_bheight_d, clump_bheight_pks = st.ks_2samp(v_clump_bheights, c_clump_bheights)
    clump_pheight_d, clump_pheight_pks = st.ks_2samp(v_clump_pheights, c_clump_pheights)
    clump_pbheight_d, clump_pbheight_pks = st.ks_2samp(v_clump_pbheights, c_clump_pbheights)

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
    print 'Clump Height Abs   | %5.3f | %5.5f '%(clump_height_d, clump_height_pks)
    print 'Clump Height Base  | %5.3f | %5.5f '%(clump_bheight_d, clump_bheight_pks)
    print 'Clump Peak Abs     | %5.3f | %5.5f '%(clump_pheight_d, clump_pheight_pks)
    print 'Clump Peaks Base   | %5.3f | %5.5f '%(clump_pbheight_d, clump_pbheight_pks)
    print

def plot_clumps_vs_prometheus():    
    long_differences = []
    
    for obs in sorted(c_all_clump_db.keys()):
#        coverage = ma.MaskedArray.count(c_all_clump_db[obs].ew_data)/float(len(c_all_clump_db[obs].ew_data))
#        if coverage < 0.8:
#            continue
        p_rad, p_long = ringimage.saturn_to_prometheus(c_all_clump_db[obs].et_min)
        for clump in c_all_clump_db[obs].clump_list:
            long_diff = clump.g_center - p_long
            if long_diff < -180: long_diff += 360
            if long_diff > 180: long_diff -= 360
            long_differences.append(long_diff)
    
    print 'Num Prometheus/Clumps =', len(long_differences)
    
    fig = plt.figure(figsize = (3.5,2))
    ax = fig.add_subplot(111)
    bins = np.arange(-180., 181., 30.)
    plt.hist(long_differences, bins, weights = np.zeros_like(long_differences) + 1./len(long_differences),
             color='black')
    ax.set_xlabel('Clump Longitude $-$ Prometheus Longitude ($^\circ$)')
    ax.set_ylabel('Fractional Number of Clumps')
    plt.xlim(-180,180)
    ax.get_xaxis().set_ticks(np.arange(-180,181,60))
    
    fig.tight_layout()
    save_fig(fig, ax, 'clump_prometheus_long_diff.png')

def attree_correlation():
    attree_filename = os.path.join(ringutil.ROOT, 'Attree_analysis', 'attree_minijet_catalogue.csv')
    attree_fp = open(attree_filename, 'r')
    attree_lines = attree_fp.readline().rsplit('\r')
    attree_lines = attree_lines[1:]
    attree_fp.close()
    
    total_num_attree = len(attree_lines)
    num_attree_associated = 0
    num_clump_associated = 0
    association_db = {}
    
    total_clumps = 0

    attree_longitudes = []
    attree_classes = []
    attree_obsids = []
    jets_db = {}
    
    bad_obsids = []
    
    for line in attree_lines:
        designation, date, exposure, phase, tip_rad, tip_long, base_rad, base_long, jet_class = line.split(',')
        tip_long = float(tip_long)
        base_long = float(base_long)
        date = list(date)
        date[8] = '/'
        date = ''.join(date)
        et = cspice.utc2et(date)
        base_long = ringutil.InertialToCorotating(base_long, et)
        for c_obs in sorted(c_nonds_all_clump_db.keys()):
            clump_et_min = c_nonds_all_clump_db[c_obs].et_min
            clump_et_max = c_nonds_all_clump_db[c_obs].et_max
            if et <  clump_et_min or et > clump_et_max:
                continue
            long_res = 360./len(c_nonds_all_clump_db[c_obs].ew_data)
            attree_idx = base_long / long_res
            if ma.getmaskarray(c_nonds_all_clump_db[c_obs].ew_data)[attree_idx]:
                print 'Attree no data', c_obs, designation, date, et, clump_et_min, base_long
                continue # We don't even have data for this location
            # This is an Attree observation we can use
            attree_longitudes.append(base_long)
            attree_classes.append(jet_class)
            attree_obsids.append(c_obs)
            if not jets_db.has_key(jet_class):
                jets_db[jet_class] = 1
            else:
                jets_db[jet_class] = jets_db[jet_class]+1

            break
    
    for c_obs in sorted(c_nonds_all_clump_db.keys()):
        if c_obs not in attree_obsids:
            bad_obsids.append(c_obs)
        
    print 'OBSID with no Attree objects', bad_obsids

    c_clump_frac = []
    clump_associated_obs_list = []
    
    good_obs = 0
    
    for c_obs in sorted(c_nonds_all_clump_db.keys()):
        if c_obs in bad_obsids:
            continue
        good_obs += 1    
        long_res = 360./len(c_nonds_all_clump_db[c_obs].ew_data)
        num_not_clump = ma.MaskedArray.count(c_nonds_all_clump_db[c_obs].ew_data)
#        print c_obs, c_clump_frac[-1]
#        plot_single_clumps(c_nonds_all_clump_db[c_obs], c_nonds_all_clump_db[c_obs].clump_list, c_obs)

        num_clump_associated_obs = 0
        
        for clump in c_nonds_all_clump_db[c_obs].clump_list:
            total_clumps += 1
            
            clump_width = (clump.fit_right_deg-clump.fit_left_deg)
            if clump_width < 0:
                clump_width += 360
            num_not_clump -= clump_width / long_res

            found_clump_match = False
            for i in range(len(attree_longitudes)):
                if c_obs != attree_obsids[i]:
                    continue
                base_long = attree_longitudes[i]
                attree_idx = base_long / long_res
                
                if ((clump.fit_left_deg < clump.fit_right_deg and clump.fit_left_deg <= base_long <= clump.fit_right_deg) or
                    (clump.fit_left_deg > clump.fit_right_deg and (clump.fit_left_deg <= base_long or base_long <= clump.fit_right_deg))):
                    found_clump_match = True
                    num_attree_associated += 1
                    jet_class = attree_classes[i]
                    if not association_db.has_key(jet_class):
                        association_db[jet_class] = 1
                    else:
                        association_db[jet_class] = association_db[jet_class]+1
#                    if jet_class == 'Classic - bright head' or jet_class == 'Classic':
#                        print c_obs, 'BASE LONG', base_long, 'FITL', clump.fit_left_deg, 'FITR', clump.fit_right_deg
                
            if found_clump_match:
                num_clump_associated += 1  # With at least one Attree object
                num_clump_associated_obs += 1  # With at least one Attree object

        c_clump_frac.append(1-(float(num_not_clump) / ma.MaskedArray.count(c_nonds_all_clump_db[c_obs].ew_data)))
        print 'Coverage', c_obs, c_clump_frac[-1]
        if len(c_nonds_all_clump_db[c_obs].clump_list) > 0:
            clump_associated_obs_list.append(float(num_clump_associated_obs)/len(c_nonds_all_clump_db[c_obs].clump_list))
                
    c_clump_frac_stats = (np.mean(c_clump_frac),
                          np.std(c_clump_frac),
                          np.min(c_clump_frac),
                          np.max(c_clump_frac))
                
    print 'OBS with at least one Attree object', good_obs, '/', len(c_nonds_all_clump_db.keys())
    print 'Total Attree objects', total_num_attree
    print 'Attree objects we have mosaic data for', len(attree_longitudes)
    print 'Total clumps', total_clumps
    print 'Clumps associated with an Attree object', num_clump_associated, '=', float(num_clump_associated)/total_clumps
    print 'Clumps associated with an Attree object per obs', np.mean(clump_associated_obs_list), '+/-', np.std(clump_associated_obs_list)
    print 'AVG Clump Coverage %5.3f'%(c_clump_frac_stats[0])
    print 'STD Clump Coverage %5.3f'%(c_clump_frac_stats[1])
    print 'MIN Clump Coverage %5.3f'%(c_clump_frac_stats[2])
    print 'MAX Clump Coverage %5.3f'%(c_clump_frac_stats[3])
    print 'Attree objects associated with a clump', num_attree_associated, '=', float(num_attree_associated)/len(attree_longitudes)
    for jet_type in sorted(jets_db.keys()):
        print jet_type, association_db[jet_type], '/', jets_db[jet_type], '=',
        print float(association_db[jet_type])/jets_db[jet_type]
    
        
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
                     clump_db_entry.ew_data[left_idx:]+(0%5)*.1, '-',
                     color=colors[n%len(colors)], lw=3, alpha=0.5)
            plt.plot(longitudes[:right_idx+1],
                     clump_db_entry.ew_data[:right_idx+1]+(0%5)*.1, '-',
                     color=colors[n%len(colors)], lw=3, alpha=0.5)
        else:
            plt.plot(longitudes[left_idx:right_idx+1],
                     clump_db_entry.ew_data[left_idx:right_idx+1]+(0%5)*.1, '-',# 0=n
                     color=colors[n%len(colors)], lw=3, alpha=0.5)
        
    ax.set_xlim(0,360)
    ax.set_ylim(0,np.max(clump_db_entry.ew_data))
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
        if clump.wave_type[:4] == 'fake':
            new_clump_list.append(clump)
            continue
        if clump.clump_db_entry is None:
            print 'NO DB'
            clump.print_all()
            assert False
        if debug:
            print 'CLUMP GC %6.2f FLD %6.2f FRD %6.2f' % (clump.g_center, clump.fit_left_deg, clump.fit_right_deg)
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
        if height < 0.03:# or height > 2: # XXX
            if debug:
                print 'XXX SKIPPING BECAUSE OF HEIGHT', height
            continue
        if width < 3.5 or width > 40:
            if debug:
                print 'XXX SKIPPING BECAUSE OF WIDTH', width
            continue
        if asym_ratio > 5:
            if debug:
                print 'XXX SKIPPING BECAUSE OF ASYM', asym_ratio
            continue
    
        new_clump_list.append(clump)
        
    return new_clump_list

def choose_correct_single_clumps(clump_db):
    for obsid in sorted(clump_db):
        if debug:
            print obsid
        clump_db_entry = clump_db[obsid]
        clump_db_entry.clump_list = limit_single_clumps(clump_db_entry.clump_list)
    
    print 'TOTAL CLUMPS AFTER GEOM BEFORE CHOOSING LARGER', np.sum([len(clump_db[x].clump_list) for x in clump_db.keys()])
        
    for obsid in sorted(clump_db):
        if debug:
            print obsid
        clump_db_entry = clump_db[obsid]
        new_list = []
        restr_clump_list = clump_db_entry.clump_list
        restr_clump_list.sort(key=lambda x: x.fit_left_deg)
        for clump_num, clump in enumerate(restr_clump_list):
            if clump.wave_type[:4] == 'fake':
                new_list.append(clump)
                continue
            clump_left_deg = clump.fit_left_deg
            clump_right_deg = clump.fit_right_deg
            if clump_right_deg < clump_left_deg:    
                clump_right_deg += 360
            found_match = False
            for sec_clump_num, sec_clump in enumerate(restr_clump_list):
                if clump_num == sec_clump_num:
#                    if debug:
#                        print 'SAME'
                    continue
#                if sec_clump.wave_type[:4] == 'fake':
#                    continue
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
#                if debug:
#                    if clump_right_deg-clump_left_deg == 36.5:
#                        print 'KEEPING CLUMP'
                new_list.append(clump)
                
        clump_db_entry.clump_list = new_list
#        if obsid == 'ISS_041RF_FMOVIE001_VIMS' or obsid == 'ISS_055RI_LPMRDFMOV001_PRIME' or obsid == 'ISS_051RI_LPMRDFMOV001_PRIME' or obsid == 'ISS_134RI_SPKMVDFHP002_PRIME' or obsid == 'ISS_033RF_FMOVIE001_VIMS':
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
voyager_clump_db_path = os.path.join(paper_root, 'voyager_clumpdb_137500_142500_05.000_0.020_10_02_137500_142500.pickle')
v_clump_db_fp = open(voyager_clump_db_path, 'rb')
clump_find_options = pickle.load(v_clump_db_fp)
v_all_clump_db = pickle.load(v_clump_db_fp)
v_clump_db_fp.close()
for v_obs in v_all_clump_db.keys(): # Fix masking
    v_all_clump_db[v_obs].ew_data[np.where(v_all_clump_db[v_obs].ew_data == 0.)] = ma.masked

# Fix up bad clump_db_entry for 6/25/13 pickle file
#for obsid in v_clump_db.keys():
#    clump_db_entry = v_clump_db[obsid]
#    for clump in clump_db_entry.clump_list:
#        clump.print_all()
#        print
#        clump.clump_db_entry = clump_db_entry
#
#os.unlink(voyager_clump_db_path)
#v_clump_db_fp = open(voyager_clump_db_path, 'wb')
#pickle.dump(clump_find_options, v_clump_db_fp)
#pickle.dump(v_clump_db, v_clump_db_fp)
#v_clump_db_fp.close()

cassini_clump_db_path = os.path.join(paper_root, 'downsampled_clumpdb_137500_142500_05.000_0.020_10_01_137500_142500.pickle')
c_clump_db_fp = open(cassini_clump_db_path, 'rb')
clump_find_options = pickle.load(c_clump_db_fp)
c_all_clump_db = pickle.load(c_clump_db_fp)
c_clump_db_fp.close()

cassini_clump_db_path = os.path.join(paper_root, 'clumpdb_137500_142500_05.000_0.020_10_02_137500_142500.pickle')
c_clump_db_fp = open(cassini_clump_db_path, 'rb')
clump_find_options = pickle.load(c_clump_db_fp)
c_nonds_all_clump_db = pickle.load(c_clump_db_fp)
c_clump_db_fp.close()

fake_new_clumps = [
    [('ISS_105RI_TDIFS20HP001_CIRS', 309, 321),
     ('ISS_105RF_FMOVIE002_PRIME', 315, 321),
     ('ISS_106RF_FMOVIE002_PRIME', 309, 339),
     ('ISS_108RI_SPKMVLFLP001_PRIME', 305, 340)],
                   
    [('ISS_036RF_FMOVIE001_VIMS', 190, 237),
     ('ISS_036RF_FMOVIE002_VIMS', 187, 246),
     ('ISS_039RF_FMOVIE002_VIMS', 207, 281),
     ('ISS_039RF_FMOVIE001_VIMS', 206, 300),
     ('ISS_041RF_FMOVIE002_VIMS', 207, 336),
     ('ISS_041RF_FMOVIE001_VIMS', 203, 336),
     ('ISS_043RF_FMOVIE001_VIMS', 201, 350),
     ('ISS_044RF_FMOVIE001_VIMS', 201, 350)]
]

for list_num, new_clump_info_list in enumerate(fake_new_clumps):
    print 'FAKING CLUMP'
    if list_num == 0:
        fake_name = 'fake2009'
    else:
        fake_name = 'fake2006'
    print fake_name
    for clump_obsid, clump_start_long, clump_end_long in new_clump_info_list:
        clump_db_entry = c_all_clump_db[clump_obsid]
        long_res = 360./len(clump_db_entry.ew_data)
        clump = clumputil.ClumpData()
        clump.clump_db_entry = clump_db_entry
        clump.longitude = (clump_start_long+clump_end_long)/2
        clump.scale = clump_end_long-clump_start_long
        clump.longitude_idx = int(clump.longitude/long_res)
        clump.scale_idx = int(clump.scale/long_res)
        clump.matched = False
        clump.wave_type = fake_name
        clump.fit_left_deg, clump.fit_right_deg, clump.fit_width_idx, clump.fit_width_deg, clump.fit_height, clump.int_fit_height, clump.g_center, clump.g_sigma, clump.g_base, clump.g_height = clump_gaussian_fit.refine_fit(clump, clump_db_entry.ew_data, False, True)
        print clump_obsid, clump_start_long, clump_end_long,
        print 'FIT LEFT', clump.fit_left_deg, 'RIGHT', clump.fit_right_deg
        c_all_clump_db[clump_obsid].clump_list.append(clump)

        clump_db_entry = c_nonds_all_clump_db[clump_obsid]
        long_res = 360./len(clump_db_entry.ew_data)
        clump = clumputil.ClumpData()
        clump.clump_db_entry = clump_db_entry
        clump.longitude = (clump_start_long+clump_end_long)/2
        clump.scale = clump_end_long-clump_start_long
        clump.longitude_idx = int(clump.longitude/long_res)
        clump.scale_idx = int(clump.scale/long_res)
        clump.matched = False
        clump.wave_type = fake_name
        clump.fit_left_deg, clump.fit_right_deg, clump.fit_width_idx, clump.fit_width_deg, clump.fit_height, clump.int_fit_height, clump.g_center, clump.g_sigma, clump.g_base, clump.g_height = clump_gaussian_fit.refine_fit(clump, clump_db_entry.ew_data, False, False)
        print clump_obsid, clump_start_long, clump_end_long,
        print 'FIT LEFT', clump.fit_left_deg, 'RIGHT', clump.fit_right_deg
        c_nonds_all_clump_db[clump_obsid].clump_list.append(clump)

    print

print 'ORIGINAL VOYAGER SINGLE CLUMPS:', np.sum([len(v_all_clump_db[x].clump_list) for x in v_all_clump_db.keys()])
print 'ORIGINAL CASSINI DOWNSAMPLED SINGLE CLUMPS:', np.sum([len(c_all_clump_db[x].clump_list) for x in c_all_clump_db.keys()])
print 'ORIGINAL CASSINI NON-DOWNSAMPLED SINGLE CLUMPS:', np.sum([len(c_nonds_all_clump_db[x].clump_list) for x in c_nonds_all_clump_db.keys()])
choose_correct_single_clumps(v_all_clump_db)
choose_correct_single_clumps(c_all_clump_db)
choose_correct_single_clumps(c_nonds_all_clump_db)
print 'RESTRICTED VOYAGER SINGLE CLUMPS:', np.sum([len(v_all_clump_db[x].clump_list) for x in v_all_clump_db.keys()])
print 'RESTRICTED CASSINI DOWNSAMPLED SINGLE CLUMPS:', np.sum([len(c_all_clump_db[x].clump_list) for x in c_all_clump_db.keys()])
print 'RESTRICTED CASSINI NON-DOWNSAMPLED SINGLE CLUMPS:', np.sum([len(c_nonds_all_clump_db[x].clump_list) for x in c_nonds_all_clump_db.keys()])

#attree_correlation()
plot_single_clump_distributions()
#plot_clumps_vs_prometheus()

