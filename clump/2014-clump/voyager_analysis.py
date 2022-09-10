from optparse import OptionParser
import numpy as np
import numpy.ma as ma
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
import bandpass_filter
import clumputil
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
import matplotlib.transforms as mtransforms
from scipy.stats import norm
import matplotlib.mlab as mlab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

print 'THIS CODE IS OUT OF DATE'
assert False

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = []
#    cmd_line = ['--side-analysis', '--plot-fit']
#    cmd_line = ['--write-pdf']

parser = OptionParser()
ringutil.add_parser_options(parser)

parser.add_option('--weighted-mean', dest = 'weighted_mean', action = 'store_true', default = False)

options, args = parser.parse_args(cmd_line)

voyager_root = ringutil.VOYAGER_PATH

def plot_single_clump(ax, ew_data, clump, long_min, long_max, label=False, color='red'):
    long_res = .5
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
    ax.plot(idx_range, mexhat, '-', color='red', lw=2, alpha=0.8, label=legend)
    if longitude_idx-clump.scale_idx/2 < 0: # Runs off left side - plot it twice
        ax.plot(idx_range+360, mexhat, '-', color=color, lw=2, alpha=0.8)
    ax.set_xlim(long_min, long_max)
    
def plot_single_ew_profile(ax, clump_db_entry, long_min, long_max, label=False, color='black'):
    ew_data = clump_db_entry.ew_data
    long_res = .5
    longitudes = np.arange(len(ew_data)) * long_res
    min_idx = int(long_min / long_res)
    max_idx = int(long_max / long_res)
    long_range = longitudes[min_idx:max_idx]
    ew_range = ew_data[min_idx:max_idx]
    legend = None
    if label:
        legend = clump_db_entry.obsid + ' (' + cspice.et2utc(clump_db_entry.et, 'C', 0) + ')'
    ax.plot(long_range, ew_range, '-', label=legend, color=color)
    
def plot_fitted_clump_on_ew(ax, ew_data, clump):
    longitudes =np.tile(np.arange(0,360., 0.5),3)
    tri_ew = np.tile(ew_data, 3)
    left_idx = clump.fit_left_deg/.5 + len(ew_data)
    right_idx = clump.fit_right_deg/.5 + len(ew_data)
    
    if left_idx > right_idx:
        left_idx -= len(ew_data)
    print left_idx, right_idx
    idx_range = longitudes[left_idx:right_idx]
#    print idx_range
    if left_idx < len(ew_data):
        ax.plot(longitudes[left_idx:len(ew_data)-1], tri_ew[left_idx:len(ew_data)-1], color = 'blue', alpha = 0.5, lw = 2)
        ax.plot(longitudes[len(ew_data):right_idx], tri_ew[len(ew_data):right_idx], color = 'blue', alpha = 0.5, lw = 2)
    else:
        ax.plot(idx_range, tri_ew[left_idx:right_idx], color = 'blue', alpha = 0.5, lw = 2)

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
                
def sigma_height_distribution(options, clump_db):
    
    clump_sigma_list = []
    longitudes = np.arange(0,360,.04)
    long_triple = np.tile(longitudes, 3)
    print clump_db
    for obsid in clump_db.keys():
        ew_data = clump_db[obsid].ew_data
        ew_data_triple = np.tile(ew_data, 3)
        
        for clump in clump_db[obsid].clump_list:

            clump_sigma = clump.fit_sigma
            clump_scale = clump.scale
            if clump_sigma > 1.0:
                clump_sigma_list.append(clump_scale)
#            clump_sigma_list.append(clump_sigma)
            
    print len(clump_sigma_list)
    # CREATE FIGURE
    fig = plt.figure()
    ax = fig.add_subplot(111)
    step = .025
    graph_max= ma.max(clump_sigma_list)
    graph_min = ma.min(clump_sigma_list)
    if options.debug:
        print 'MAX, MIN: ', graph_max, graph_min
        
    bins = np.arange(graph_min,graph_max,step)
    
    counts, bins, patches = plt.hist(clump_sigma_list, bins)
    if options.debug:
        print 'COUNTS, BINS: '
        print counts, bins
        
#    if options.plot_fit:
#        y = fit_curve(clump_sigma_list, bins)
#        plt.plot(bins, y, 'w--', linewidth = 1.5, color = 'r')
        
    plt.title('Sigma Height Distribution: ' + 'VOYAGER DATA')
    plt.xlabel('Sigma Clump Heights')
    
    plt.show()
    
def list_to_db(v_approved_list, c_approved_list, c_clump_db, v_clump_db):
    
    #make double databases that have clump lists with only approved clumps.
#    v_clump_db_entry = clumputil.ClumpDBEntry()
    isolated_ids = [
                    'ISS_00ARI_SPKMOVPER001_PRIME',
                    'ISS_000RI_SATSRCHAP001_PRIME',
                    'ISS_013RI_AZSCNHIPH003_PRIME',
                    'ISS_124RD_PINGPONG001_PRIME'
                    ]
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
#            if clump.fit_width_deg < 35.:
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
                
    for obsid in c_clump_db.keys():
        if (obsid not in c_db.keys()) and (obsid not in isolated_ids):
            c_db[obsid] = c_clump_db[obsid]
            c_db[obsid].clump_list = []
                        
    return (v_db, c_db)
                
def compare_clumps_per_obsid(v_approved_list, c_approved_list, clump_db, v_clump_db):
    
    #good coverage ids must have at least 80% coverage and not be the last observation in a series
    #they must also not be stand-alone observations with > 40 days until the next observation
    
## SIMPLE IDS   
#    good_coverage_ids = [
#                         'ISS_029RF_FMOVIE001_VIMS','ISS_031RF_FMOVIE001_VIMS','ISS_032RF_FMOVIE001_VIMS',
#                         'ISS_033RF_FMOVIE001_VIMS','ISS_036RF_FMOVIE001_VIMS','ISS_036RF_FMOVIE002_VIMS',
#                         'ISS_039RF_FMOVIE002_VIMS','ISS_039RF_FMOVIE001_VIMS','ISS_041RF_FMOVIE002_VIMS',
#                         'ISS_041RF_FMOVIE001_VIMS','ISS_043RF_FMOVIE001_VIMS',
#                         'ISS_055RF_FMOVIE001_VIMS','ISS_055RI_LPMRDFMOV001_PRIME','ISS_057RF_FMOVIE001_VIMS',
#                         'ISS_059RF_FMOVIE001_VIMS','ISS_059RF_FMOVIE002_VIMS','ISS_072RI_SPKHRLPDF001_PRIME',
#                         'ISS_075RF_FMOVIE002_VIMS','ISS_083RI_FMOVIE109_VIMS','ISS_091RF_FMOVIE003_VIMS',
#                         'ISS_096RF_FMOVIE004_VIMS','ISS_114RF_FMOVIEEQX001_PRIME','ISS_115RF_FMOVIEEQX001_PRIME',
#                         ]

# UNIONED COVERAGE IDS
    good_coverage_ids = [
                         'ISS_006RI_LPHRLFMOV001_PRIME', 'ISS_007RI_LPHRLFMOV001_PRIME', 'ISS_029RF_FMOVIE001_VIMS', 
                         'ISS_031RF_FMOVIE001_VIMS', 'ISS_032RF_FMOVIE001_VIMS', 'ISS_033RF_FMOVIE001_VIMS', 
                         'ISS_036RF_FMOVIE001_VIMS', 'ISS_036RF_FMOVIE002_VIMS', 'ISS_039RF_FMOVIE002_VIMS', 
                         'ISS_039RF_FMOVIE001_VIMS', 'ISS_041RF_FMOVIE002_VIMS', 'ISS_041RF_FMOVIE001_VIMS', 
                         'ISS_043RF_FMOVIE001_VIMS', 'ISS_055RF_FMOVIE001_VIMS', 'ISS_055RI_LPMRDFMOV001_PRIME', 
                         'ISS_057RF_FMOVIE001_VIMS', 'ISS_059RF_FMOVIE001_VIMS', 'ISS_059RF_FMOVIE002_VIMS', 
                         'ISS_072RI_SPKHRLPDF001_PRIME', 'ISS_075RF_FMOVIE002_VIMS', 'ISS_081RI_FMOVIE106_VIMS', 
                         'ISS_089RF_FMOVIE003_PRIME', 'ISS_091RF_FMOVIE003_PRIME', 'ISS_096RF_FMOVIE004_PRIME', 
                         'ISS_114RF_FMOVIEEQX001_PRIME', 'ISS_132RI_FMOVIE001_VIMS'
                         ]


    

    print ''
    print '                USING IDS FROM THE "UNION-COVERAGE" METHOD               '
    print ''
    print 'NOTES:'
    print '1. All clump number stats have been weighted based on the coverage of the observation'
    print '2. good coverage ids must have at least 80% coverage and not be the last observation in a series'
    print '   -they must also not be stand-alone observations (observations with > 40 days until the next observation)'
    print '3. Clump stats for Width, Brightness and Velocity used only data from obsids containing clumps'
    
    print ''
    print 'USED OBSERVATION IDS FOR -CLUMP NUM/OBSERVATION- STATS'
    print ' --------------------------------------------------'
    for obs in good_coverage_ids:
        print obs
    print ' --------------------------------------------------'
    #take the approved lists and turn them into clump databases with only the valid clumps that we've tracked.
    #note - we do NOT use the full clump db 
    v_clump_db, clump_db = list_to_db(v_approved_list, c_approved_list, clump_db, v_clump_db)
    
    chain_length = []
    lifetimes = []
    for chain in c_approved_list:
        chain_length.append(len(chain.clump_list))
        lifetimes.append(chain.lifetime)
    max_length = np.max(chain_length)
    max_life = np.max(lifetimes)
    min_life = np.min(lifetimes)
    avg_life = np.mean(lifetimes)
    total_clump_num = len(c_approved_list)
    
    v_clump_numbers = []
    v1_clump_scales = []
    v1_clump_heights = []
    v2_clump_scales = []
    v2_clump_heights = []
    
    v_stddevs = []
    for v_obs in v_clump_db.keys():
        clumps = []
#        if v_obs[0:2] == 'V1':
        for clump in v_clump_db[v_obs].clump_list:    
            clumps.append(clump)
#            v_clump_scales.append(clump.fit_width_deg)
#            v_clump_heights.append(clump.int_fit_height + 1.) #add 1 because  it's the fitted height above the Mean.
        clump_num = len(clumps)
        v_clump_numbers.append(clump_num)
        v_norm_ew_data = v_clump_db[v_obs].ew_data
        v_stddevs.append(np.std(v_norm_ew_data))
        
    #use the chains lists to calculate width and height stats. 1 chain = 1 data point
#    for chain in v_approved_list:

    for chain in v_approved_list:
        print chain.clump_list[0].clump_db_entry.obsid[0:2]
        if chain.clump_list[0].clump_db_entry.obsid[0:2] == 'V1':
            print 'a'
            v1_widths = np.array([clump.fit_width_deg for clump in chain.clump_list])
            v1_heights = np.array([clump.int_fit_height for clump in chain.clump_list])
    
            v1_clump_scales.append(np.mean(v1_widths))
            v1_clump_heights.append(np.mean(v1_heights))
            
        if chain.clump_list[0].clump_db_entry.obsid[0:2] == 'V2':
            print 'b'
            v2_widths = np.array([clump.fit_width_deg for clump in chain.clump_list])
            v2_heights = np.array([clump.int_fit_height for clump in chain.clump_list])
    
            v2_clump_scales.append(np.mean(v2_widths))
            v2_clump_heights.append(np.mean(v2_heights))
    print v1_clump_heights
    print v2_clump_heights
#    print v_clump_heights
    v_clump_num_stats = (float(sum(v_clump_numbers))/float(len(v_clump_numbers)), np.std(v_clump_numbers), np.min(v_clump_numbers), np.max(v_clump_numbers))
    v1_scale_stats = (sum(v1_clump_scales)/len(v1_clump_scales), np.std(v1_clump_scales), np.min(v1_clump_scales), np.max(v1_clump_scales))
    v1_clump_height_stats = (sum(v1_clump_heights)/len(v1_clump_heights), np.std(v1_clump_heights), np.min(v1_clump_heights), np.max(v1_clump_heights))
    v2_scale_stats = (sum(v2_clump_scales)/len(v2_clump_scales), np.std(v2_clump_scales), np.min(v2_clump_scales), np.max(v2_clump_scales))
    v2_clump_height_stats = (sum(v2_clump_heights)/len(v2_clump_heights), np.std(v2_clump_heights), np.min(v2_clump_heights), np.max(v2_clump_heights))
    v_stddev_stats = (sum(v_stddevs)/len(v_stddevs), np.std(v_stddevs), np.min(v_stddevs), np.max(v_stddevs))
    
    c_clump_numbers = []
    c_weighted_clump_numbers = []
    c_localized_clump_numbers = []           #number of clumps in only mostly complete profiles
    c_clump_scales = []
    c_clump_heights = []
    c_stddevs = []
    max_ets = []
    db_by_time = {}
    
#    print clump_db.keys()
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
        coverage = ma.MaskedArray.count(clump_db[obs].ew_data)/float(len(clump_db[obs].ew_data))
#        print obs
        for clump in clump_db[obs].clump_list: 
#                print clump.fit_sigma              
                
            clumps.append(clump)
#            c_clump_scales.append(clump.fit_width_deg)
#            c_clump_heights.append(clump.int_fit_height +1.)
            
        clump_num = len(clumps)
        c_clump_numbers.append(clump_num)
        c_weighted_clump_numbers.append(clump_num/coverage)
        if obs in good_coverage_ids:
            c_localized_clump_numbers.append(clump_num/coverage)
        c_norm_ew_data = clump_db[obs].ew_data
        c_stddevs.append(np.std(c_norm_ew_data))
        max_ets.append(clump_db[obs].et_max)
        
    for chain in c_approved_list:
    
        widths = np.array([clump.fit_width_deg for clump in chain.clump_list])
        heights = np.array([clump.int_fit_height for clump in chain.clump_list])
                
        c_clump_scales.append(np.median(widths))
        c_clump_heights.append(np.median(heights))

    #statistics arrays of 1) mean, 2) stddev, 3) minimum value, 4) max value
#    c_clump_num_stats = (float(sum(c_clump_numbers))/float(len(c_clump_numbers)), np.std(c_clump_numbers), np.min(c_clump_numbers), np.max(c_clump_numbers))
    c_localized_clump_num_stats = (np.sum(c_localized_clump_numbers)/float(len(c_localized_clump_numbers)), np.std(c_localized_clump_numbers), np.min(c_localized_clump_numbers), np.max(c_localized_clump_numbers))
    c_scale_stats = (sum(c_clump_scales)/len(c_clump_scales), np.std(c_clump_scales), np.min(c_clump_scales), np.max(c_clump_scales))
    c_clump_height_stats = (sum(c_clump_heights)/len(c_clump_heights), np.std(c_clump_heights), np.min(c_clump_heights), np.max(c_clump_heights))
    c_stddev_stats = (sum(c_stddevs)/len(c_stddevs), np.std(c_stddevs), np.min(c_stddevs), np.max(c_stddevs))
    
    km_per_deg = 881027.02/360.
#    print c_localized_clump_numbers
    print 'VOYAGER CLUMPS/ID:', v_clump_num_stats[0], 'CASSINI ClUMPS/ID:', c_localized_clump_num_stats[0]
    print 'VOYAGER 1 AVG WIDTH:', v1_scale_stats[0], 'VOYAGER 2 AVG WIDTH:', v2_scale_stats[0], 'CASSINI AVG WIDTH:', c_scale_stats[0]
    print 'VOYAGER 1 AVG CLUMP HEIGHT x10^4:', (v1_clump_height_stats[0]*km_per_deg)/1e4, 'VOYAGER 2 AVG CLUMP HEIGHT x10^4:',\
        (v2_clump_height_stats[0]*km_per_deg)/1e4, 'CASSINI AVG CLUMP HEIGHT x10^4:', (c_clump_height_stats[0]*km_per_deg)/1e4
    print ''

    print '-----------------------STATS TABLE--------------------------'
    print ''
    print '----------------------| Voyager | Cassini ------------------'
    print ''
    print 'AVG # Clumps/Observ   |  %5.1f   |  %5.1f  '%(v_clump_num_stats[0], c_localized_clump_num_stats[0])
    print 'STD # Clumps/Observ   |  %5.1f   |  %5.1f  '%(v_clump_num_stats[1], c_localized_clump_num_stats[1])
    print 'MIN # Clumps/Observ   |  %5.1f   |  %5.1f  '%(v_clump_num_stats[2], c_localized_clump_num_stats[2])
    print 'MAX # Clumps/Observ   |  %5.1f   |  %5.1f  '%(v_clump_num_stats[3], c_localized_clump_num_stats[3])
    print ' '
    print 'AVG Clump Fit Width   |  %5.2f   |  %5.2f  '%(v1_scale_stats[0],c_scale_stats[0])
    print 'STD Clump Fit Width   |  %5.2f   |  %5.2f  '%(v1_scale_stats[1],c_scale_stats[1])
    print 'MIN Clump Fit Width   |  %5.2f   |  %5.2f  '%(v1_scale_stats[2],c_scale_stats[2])
    print 'MAX Clump Fit Width   |  %5.2f   |  %5.2f  '%(v1_scale_stats[3],c_scale_stats[3])
    print ' '
    print 'AVG Clump Int Height  |  %5.2f   |  %5.2f  '%(v1_clump_height_stats[0]*km_per_deg/1e4, c_clump_height_stats[0]*km_per_deg/1e4)
    print 'STD Clump Int Height  |  %5.2f   |  %5.2f  '%(v1_clump_height_stats[1]*km_per_deg/1e4, c_clump_height_stats[1]*km_per_deg/1e4)
    print 'MIN Clump Int Height  |  %5.2f   |  %5.2f  '%(v1_clump_height_stats[2]*km_per_deg/1e4, c_clump_height_stats[2]*km_per_deg/1e4)
    print 'MAX Clump Int Height  |  %5.2f   |  %5.2f  '%(v1_clump_height_stats[3]*km_per_deg/1e4, c_clump_height_stats[3]*km_per_deg/1e4)
    print ' '
    print 'AVG OBSID STDDEV      |  %5.2f   |  %5.2f  '%(v_stddev_stats[0], c_stddev_stats[0])
    print 'STD OBSID STDDEV      |  %5.2f   |  %5.2f  '%(v_stddev_stats[1], c_stddev_stats[1])
    print 'MIN OBSID STDDEV      |  %5.2f   |  %5.2f  '%(v_stddev_stats[2], c_stddev_stats[2])
    print 'MAX OBSID STDDEV      |  %5.2f   |  %5.2f  '%(v_stddev_stats[3], c_stddev_stats[3])
    print '------------------------------------------------------------'
    print ''
    print 'MAX CHAIN LENGTH:', max_length
    print 'MIN CLUMP LIFETIME:', min_life
    print 'AVG CLUMP LIFETIME', avg_life
    print 'MAX ClUMP LIFETIME:', max_life
    print ''
    print 'TOTAL NUMBER OF CASSINI CLUMPS:', total_clump_num
    print 'TOTAL NUMBER OF VOYAGER CLUMPS:', len(v_approved_list)
    
    print ' '
    
    return (v1_clump_scales, v1_clump_heights,v2_clump_scales, v2_clump_heights, c_clump_scales, c_clump_heights, c_localized_clump_numbers, v_clump_numbers)

def compare_voyager_fit_clumps(v_clump_db):
    
    for v_obs in v_clump_db.keys():
        for clump in v_clump_db[v_obs].clump_list:
            clump_db_entry = v_clump_db[v_obs]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            plot_single_ew_profile(ax, clump_db_entry, 0., 360.)
            plot_single_clump(ax, clump_db_entry.ew_data, clump, 0., 360.)
            plot_fitted_clump_on_ew(ax, clump_db_entry.ew_data, clump)
            
            print clump.scale, clump.fit_width_deg
            plt.show()

    
def compare_profiles(v_clump_db, c_clump_db):
    
    OBSID = 'ISS_059RF_FMOVIE002_VIMS'
    for v_obs in v_clump_db.keys():
        
        v_ew_data = v_clump_db[v_obs].ew_data 
        v_ew_data = v_ew_data/np.mean(v_ew_data)
        
        c_ew_data = c_clump_db[OBSID].ew_data # Data is already downsampled
        c_ew_data = c_ew_data/np.mean(c_ew_data)
        
        longitudes = np.arange(0, 360., 360./len(v_ew_data))
        
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
        
        voyager = plt.plot(longitudes, v_ew_data, color = '#AC7CC9', lw = 2.0)
        cassini = plt.plot(longitudes, c_ew_data, color ='black' , lw = 2.0, alpha = 1.0)
        leg = ax.legend([voyager[0], cassini[0]], ['Voyager', 'Cassini'], loc = 1)
        leg.get_frame().set_alpha(0.0)
        leg.get_frame().set_visible(False)
        plt.savefig(os.path.join(voyager_root, v_obs + 'comparison_profile.png'), dpi = 600, bbox_inches = 'tight', transparent = True)


def make_distribution_plots(v1_clump_scales, v1_clump_heights, v2_clump_scales, v2_clump_heights, v_clump_nums, c_clump_scales, c_clump_heights, c_clump_nums, options):
  
    km_per_deg = 881027.02/360.
#    graph_min = np.min(v_clump_scales)
#    graph_max = np.max(v_clump_scales)
    graph_min = 0.0
    graph_max = np.max(c_clump_scales)
    step = 2.5
    bins = np.arange(0, graph_max+step,step)
#    bins = np.arange(0, 40., step)
    figure_size = (10., 4.0)
    fig = plt.figure(figsize = figure_size)
    ax = fig.add_subplot(111)
    plt.subplots_adjust(top = .95, bottom = 0.125, left = 0.08, right = 0.98)
    
    ax.set_xlabel('Clump Width ( $\mathbf{^o}$)', fontsize = 18.0)
    ax.set_ylabel('Fraction of Counts', fontsize = 18.0)
    c_scale_weights = np.zeros_like(c_clump_scales) + 1./len(c_clump_scales)
    v1_scale_weights = np.zeros_like(v1_clump_scales) + 1./len(v1_clump_scales)
    v2_scale_weights = np.zeros_like(v2_clump_scales) + 1./len(v2_clump_scales)
    
    scale_counts, bins, patches = plt.hist([ c_clump_scales, v1_clump_scales, v2_clump_scales], bins,
                                      weights = [ c_scale_weights, v1_scale_weights, v2_scale_weights],
                                       label = ['Cassini','Voyager 1', 'Voyager 2'], color = ['black','#AC7CC9', 'red'], lw = 0.0)
    
    leg =plt.legend()
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
    plt.savefig(os.path.join(voyager_root, 'Clump_width_dist.png'), dpi = 600, transparent = True)   
    
    fig2 = plt.figure(figsize = figure_size)
    ax = fig2.add_subplot(111)
    plt.subplots_adjust(top = .95, bottom = 0.125, left = 0.08, right = 0.98)
    ax.set_xlabel('Clump Relative Brightness (km^2)', fontsize = 18.0)
    ax.set_ylabel('Fraction of Counts', fontsize = 18.0)
    
    v1_clump_heights = np.array(v1_clump_heights)*km_per_deg
    v2_clump_heights = np.array(v2_clump_heights)*km_per_deg
    c_clump_heights = np.array(c_clump_heights)*km_per_deg
    
    v1_clump_heights /= 1e4
    v2_clump_heights /= 1e4
    c_clump_heights /= 1e4
    
    v1_min = np.min(v1_clump_heights)
    v1_max = np.max(v1_clump_heights)
    v2_min = np.min(v2_clump_heights)
    v2_max = np.max(v2_clump_heights)
    c_max = np.max(c_clump_heights)
    c_min = np.min(c_clump_heights)
    
    maxes = [v1_max, v2_max, c_max]
    graph_max = np.max(maxes)
    mins = [v1_min, v2_min, c_min]
    graph_min = np.min(mins)
    
    step = 0.2
    bins = np.arange(graph_min,graph_max + step,step)
    ax.get_xaxis().set_ticks(np.arange(0.,graph_max +1., 0.8))
    ax.set_xlim(0.0, graph_max +0.5)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

#    xFormatter = FormatStrFormatter('%.3e')
#    ax.xaxis.set_major_formatter(xFormatter)
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='x') 
    c_height_weights = np.zeros_like(c_clump_heights) + 1./len(c_clump_heights)
    v1_height_weights = np.zeros_like(v1_clump_heights) + 1./len(v1_clump_heights)
    v2_height_weights = np.zeros_like(v2_clump_heights) + 1./len(v2_clump_heights)
    
    height_counts, bins, patches = plt.hist([ c_clump_heights, v1_clump_heights, v2_clump_heights], bins,
                                      weights = [c_height_weights, v1_height_weights, v2_height_weights],
                                       label = [ 'Cassini', 'Voyager 1', 'Voyager 2'], color = ['black','#AC7CC9', 'red'], lw = 0.0)
    
    leg =plt.legend()
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)

#    #INSET GRAPH
#    inset=plt.axes([0.35,0.45,0.4,0.4])
##    inset.get_xaxis().set_ticks(np.arange(1.0,3.0, 0.1))
#    inset.set_xlim(1.0, 2.75)
#    step = 0.1
#    bins = np.arange(1.0,3.0,step)
#        
#    height_counts, bins, patches = plt.hist([v_clump_heights, c_clump_heights], bins,
#                                      weights = [v_height_weights, c_height_weights], color = ['#AC7CC9', 'black'], lw = 0.0)

#    plt.savefig(os.path.join(voyager_root, 'zoomed_Clump_height_dist.png'), dpi = 600, transparent = True)
#    print v_clump_nums
#    print c_clump_nums
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Number of Clumps/ID', fontsize = 18.0)
    ax.set_ylabel('Fractional Number of IDs', fontsize = 18.0)
    
    step = 2
    bins = np.arange(0., 20, step)
    v_clump_nums = np.array(v_clump_nums)
    c_clump_nums = np.array(c_clump_nums)
    v_num_weights = np.zeros_like(v_clump_nums) + 1./len(v_clump_nums)
    c_num_weights = np.zeros_like(c_clump_nums) + 1./len(c_clump_nums)
    
    counts, bins, patches = plt.hist([v_clump_nums, c_clump_nums], bins,
                                      weights = [v_num_weights, c_num_weights], label= ['Voyager', 'Cassini'], color = ['#AC7CC9', 'black'], lw = 0.0)
    leg = plt.legend()
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
    plt.show()
    return(scale_counts, height_counts)



FRING_MEAN_MOTION = 581.964

def RelativeRateToSemimajorAxis(rate):  # from ringutil
    return ((FRING_MEAN_MOTION / (FRING_MEAN_MOTION+rate*86400.))**(2./3.) * 140221.3)

def make_voyager_vel_hist():

    velocities = np.array([-0.44, -0.306, -0.430, -0.232, -0.256, -0.183, -0.304,
                   -.410, -0.174, -0.289, -0.284, -0.257, -0.063, -0.310,
                   -0.277, -0.329, -0.290, -0.412, -0.198, -0.258, -0.015, 
                   -0.299, -0.370, -0.195, -0.247, -0.303, -0.213, -0.172, -0.010]) + 0.306
    
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
    ax_mm.get_xaxis().set_ticks(np.arange(-.5,.5,.05))
    ax_sma.get_xaxis().set_ticks(np.arange(120,300,10))

    ax_mm.set_ylabel('Fraction of Counts', fontsize = 18.0)
    
    step = 0.05
    ax_mm.set_xlim(-0.2, 0.6)
    bins = np.arange(-.5,0.5,step)
    
    counts, bins, patches = ax_mm.hist(velocities, bins, weights = np.zeros_like(velocities) + 1./velocities.size,
                                        color = '#B8372A', alpha = 0.9, histtype = 'stepfilled', lw = 0.0)
    
#    plt.savefig(os.path.join(voyager_root, 'Voyager_Clump_Velocity_Dist_Dual_Axis.png'), dpi = 600, transparent = True)
    plt.show()

def combined_vel_hist(approved_list):
        
    c_velocities = []
    
    for chain in approved_list:
        c_velocities.append(chain.rate)
    
    c_velocities = np.array(c_velocities)*86400.            #change to deg/day
    
    v1_velocities = np.array([-0.44, -0.306, -0.430, -0.232, -0.256, -0.183, -0.304,
               -.410, -0.174, -0.284, -0.257, -0.063, -0.310,
               -0.277, -0.329]) + 0.306
               
    v2_velocities = np.array([-0.290, -0.412, -0.198, -0.258, -0.015, 
               -0.299, -0.370, -0.195, -0.247, -0.303, -0.213, -0.172, -0.010]) + 0.306

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
    ax_mm.get_xaxis().set_ticks(np.arange(-.8,.8,.05))
    ax_sma.get_xaxis().set_ticks(np.arange(120,300,10))

    ax_mm.set_ylabel('Fraction of Counts', fontsize = 18.0)
    
    graph_min = np.min(c_velocities)
    graph_max = np.max(c_velocities)
    step = 0.05
    ax_mm.set_xlim(-0.8, 0.8)
    bins = np.arange(-.8,0.8,step)
    
#    print len(v2_velocities)
    
    counts, bins, patches = plt.hist([c_velocities, v1_velocities, v2_velocities ], bins,
                                    weights = [np.zeros_like(c_velocities) + 1./len(c_velocities), np.zeros_like(v1_velocities) + 1./len(v1_velocities),
                                                np.zeros_like(v2_velocities) + 1./len(v2_velocities)],
                                    label = ['Cassini', 'Voyager 1', 'Voyager 2'], color = ['black', '#AC7CC9', 'red'], lw = 0.0)
    
    
    leg =plt.legend()
    leg.get_frame().set_alpha(0.0)
    leg.get_frame().set_visible(False)
    
#    plt.savefig(os.path.join(voyager_root, 'Combined_Clump_Velocity_Dist_Dual_Axis.png'), dpi = 600, transparent = True)

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
    ax_mm.get_xaxis().set_ticks(np.arange(-.8,.8,.05))
    ax_sma.get_xaxis().set_ticks(np.arange(120,300,10))

    ax_mm.set_ylabel('Fraction of Counts', fontsize = 18.0)
    
    graph_min = np.min(c_velocities)
    graph_max = np.max(c_velocities)
    step = 0.05
    ax_mm.set_xlim(-0.8, 0.8)
    bins = np.arange(-.8,0.8,step)
    
#    counts, bins, patches = plt.hist([v_velocities, c_velocities], bins, normed = 1,
#                            label = ['Voyager', 'Cassini'], color = ['#AC7CC9', 'black'], lw = 0.0)
#    
#    # best fit of CASSINI DATA
#    # add a 'best fit' line
##    np.arange(-0.4,0.4, 0.01)
##    print len(bins)
#    c_y = mlab.normpdf(bins, np.mean(c_velocities), np.std(c_velocities))
#    c_pdf = c_y
#    l = plt.plot(bins, c_pdf, 'r--', color = 'black', linewidth=2)
#    
#    # best fit of VOYAGER DATA
#    # add a 'best fit' line
#    v_y = mlab.normpdf(bins, np.mean(v_velocities), np.std(v_velocities))
#    v_pdf = v_y
#    l = plt.plot(bins, v_pdf, 'r--', color = '#AC7CC9', linewidth=2)
#    print '----------VELOCITY STATS-------------------'
#    print ''
#    print '--------|  Voyager | Cassini |---------'
#    print ' MEAN   |  %5.3f   |  %5.3f  |'%(np.mean(v_velocities),np.mean(c_velocities))
#    print ' STD    |  %5.3f   |  %5.3f  |'%(np.std(v_velocities), np.std(c_velocities))
#    
#    #Do a CHI-Squared test on Counts and v_pdf/c_pdf
#    
#    v_chi, v_p = st.chisquare(counts[0], v_pdf[0:-1])
#    c_chi, c_p = st.chisquare(counts[1], c_pdf[0:-1])
#    
#    print ''
#    print '---------GAUSSIAN FIT STATS ----------------'
#    print ''
#    print '------|  Voyager |  Cassini  |----------------'
#    print 'X^2   |   %5.3f  |  %5.3f  |'%( v_chi, c_chi) 
#    print ' p    |   %5.3f  |  %5.3f  |'%(v_p, c_p)
#    
#    
    plt.show()
     
def ttests(c_mean, v_mean, c_sigma, v_sigma, n_c, n_v):
    
    mean_dif = v_mean - c_mean
    var_err = np.sqrt((c_sigma/n_c) + (v_sigma/n_v))
    
    t = mean_dif/var_err
    
    dof = np.power((c_sigma/n_c) + (v_sigma/n_v), 2)/ \
    ( ((c_sigma/n_c)**2)/(n_c -1) + ((v_sigma/n_v)**2)/(n_v -1) )
    
    
    p_value = st.t.pdf(t, dof)
    
    return (t, dof, p_value)

def kstests(v_scales_counts, c_scales_counts, v_heights_counts, c_heights_counts, c_clump_num, v_clump_num):
    
    graph_min = np.min(c_clump_num)
    graph_max = np.max(c_clump_num)
    step = 1.0
    bins = np.arange(0,graph_max,step)
#      
    figure_size = (10., 4.0)
    fig = plt.figure(figsize = figure_size)
    ax = fig.add_subplot(111)
    plt.subplots_adjust(top = .95, bottom = 0.125, left = 0.08, right = 0.98)
    
    c_num_weights = np.zeros_like(c_clump_num) + 1./len(c_clump_num)
    v_num_weights = np.zeros_like(v_clump_num) + 1./len(v_clump_num)
    
    num_counts, bins, patches = plt.hist([v_clump_num, c_clump_num], bins,
                                      weights = [v_num_weights, c_num_weights], label = ['Voyager', 'Cassini'], color = ['#AC7CC9', 'black'], lw = 0.0)
    
    v_scale_dist = []
    c_scale_dist = []
    v_height_dist = []
    c_height_dist = []
    v_num_dist = []
    c_num_dist = []
    
    def cumulative_distribution(array, output):
        start = 0
        for num in array:
            new = num + start
            output.append(new)
            start = new
#        print start
        return(output)
    
    v_scale_dist = cumulative_distribution(v_scales_counts, v_scale_dist)
    c_scale_dist = cumulative_distribution(c_scales_counts, c_scale_dist)
    v_height_dist = cumulative_distribution(v_heights_counts, v_height_dist)
    c_height_dist = cumulative_distribution(c_heights_counts, c_height_dist)
    v_num_dist = cumulative_distribution(num_counts[0], v_num_dist)
    c_num_dist = cumulative_distribution(num_counts[1], c_num_dist)
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    plt.plot(v_scale_dist, 'blue')
#    plt.plot(c_scale_dist, 'green')

    
    clump_num_d, clump_num_pks = st.ks_2samp(c_num_dist, v_num_dist)
    clump_scale_d, clump_scale_pks = st.ks_2samp(v_scale_dist, c_scale_dist)
    clump_height_d, clump_height_pks = st.ks_2samp(v_height_dist, c_height_dist)
    
    #create the cumulative distribution
    print '----------------K-S STATS--------------'
    print ''
    print '--------------|   D   |   p    --------'
    print '---------------------------------------'
    print 'Clumps per ID | %5.3f | %5.5f '%(clump_num_d, clump_num_pks)
    print 'Clump Widths  | %5.3f | %5.5f '%(clump_scale_d, clump_scale_pks)
    print 'Clump Height  | %5.3f | %5.5f '%(clump_height_d, clump_height_pks)
    print ''
    
#    plt.show()
    
#-------------------------------------
#             Main
#-------------------------------------

voyager_clump_db_path = os.path.join(voyager_root, 'voyager_clump_database.pickle')
v_clump_db_fp = open(voyager_clump_db_path, 'rb')
clump_find_options = pickle.load(v_clump_db_fp)
v_clump_db = pickle.load(v_clump_db_fp)
v_clump_db_fp.close()

#sigma_list = sigma_height_distribution(options, v_clump_db)

ds_clump_db_path = os.path.join(ringutil.VOYAGER_PATH, 'downsampled_clump_database.pickle')
ds_clump_db_fp = open(ds_clump_db_path, 'rb')
clump_find_options = pickle.load(ds_clump_db_fp)
ds_clump_db = pickle.load(ds_clump_db_fp)
ds_clump_db_fp.close()

c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_list_w_errors.pickle')
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

v1_clump_scales, v1_clump_heights,v2_clump_scales, v2_clump_heights, c_clump_scales, c_clump_heights, c_clump_nums, v_clump_nums = compare_clumps_per_obsid(v_approved_list, c_approved_list, c_approved_db, v_approved_db)
scale_counts, height_counts = make_distribution_plots(v1_clump_scales, v1_clump_heights,v2_clump_scales, v2_clump_heights, v_clump_nums, c_clump_scales, c_clump_heights, c_clump_nums, options)
kstests(scale_counts[0], scale_counts[1], height_counts[0], height_counts[1], c_clump_nums, v_clump_nums)
combined_vel_hist(c_approved_list)

#compare_profiles(v_clump_db, ds_clump_db)
#make_voyager_vel_hist()
#compare_voyager_fit_clumps(v_clump_db)


