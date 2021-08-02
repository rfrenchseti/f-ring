'''
A program to test clump attributes to their lifetimes

1. Clump width vs. Lifetime
2. Distance from prometheus vs. lifetime
3. Date vs. lifetime
4. # of clumps/ID vs. lifetime


Author: Shannon Hicks 2/1/2013
'''


import numpy as np
import scipy.optimize as sciopt
import matplotlib.pyplot as plt
import pickle
import ringutil
import clumputil
import os
import sys
from optparse import OptionParser
import numpy.ma as ma
import cspice
import ringimage

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = []
#    cmd_line = ['--side-analysis', '--plot-fit']
#    cmd_line = ['--write-pdf']

parser = OptionParser()
ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)


def compare_time_lifetime(c_approved_db, c_approved_list):
    lifetimes = []
    start_dates = []
    for chain in c_approved_list:
        day1 = chain.clump_list[0].clump_db_entry.et_max
        day2 = chain.clump_list[-1].clump_db_entry.et_max
        lifetime = (day2 - day1)/86400.     #seconds to days
        lifetimes.append(lifetime)
        
        start_dates.append(day1)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(start_dates, lifetimes, marker = '.', ls = '')
    plt.xlabel('Starting ETs (seconds)')
    plt.ylabel('Lifetimes of Clumps (days)')
#    plt.show()
    fig.savefig('Compare_Initial_Time_vs_Lifetime.png')

def compare_size_lifetime(c_approved_list):
    lifetimes = []
    start_size = []
    for chain in c_approved_list:
        day1 = chain.clump_list[0].clump_db_entry.et
        day2 = chain.clump_list[-1].clump_db_entry.et
        lifetime = (day2 - day1)/86400.     #seconds to days
        lifetimes.append(lifetime)
        
        start_size.append(chain.clump_list[0].scale)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(start_size, lifetimes, marker = '.', ls = '')
    plt.xlabel('Starting Widths (deg)')
    plt.ylabel('Lifetimes of Clumps (days)')
#    plt.show()
    plt.savefig('Compare_Initial_Width_vs_Lifetime.png')

def compare_clumpnums_lifetime(c_approved_db, c_approved_list):
    lifetimes = []
    clump_num = []
    
    for chain in c_approved_list:
        day1 = chain.clump_list[0].clump_db_entry.et
        day2 = chain.clump_list[-1].clump_db_entry.et
        lifetime = (day2 - day1)/86400.     #seconds to days
        lifetimes.append(lifetime)
        
        starting_obs = chain.clump_list[0].clump_db_entry.obsid
        clump_num.append(len(c_approved_db[starting_obs].clump_list))
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(clump_num,lifetimes , marker = '.', ls = '')
    plt.xlabel('Number of Clumps per Starting ID')
    plt.ylabel('Lifetimes of Clumps (Days)')
#    plt.show()
    fig.savefig('Compare_Clump_Number_in_Starting_ID_vs_Liftime.png')
    
def compare_prometheus_lifetime(c_approved_db, c_approved_list):
    prom_pos = []
    lifetimes = []
    
    for chain in c_approved_list:
        day1 = chain.clump_list[0].clump_db_entry.et
        day2 = chain.clump_list[-1].clump_db_entry.et
        lifetime = (day2 - day1)/86400.     #seconds to days
        lifetimes.append(lifetime)
    
        min_et = chain.clump_list[0].clump_db_entry.et_min
        min_et_long = chain.clump_list[0].clump_db_entry.et_min_longitude
        max_dist, max_dist_long = ringutil.prometheus_close_approach(min_et, min_et_long)

        clump_start_long = chain.clump_list[0].longitude
        
        dist_from_prom = clump_start_long - max_dist_long

        prom_pos.append(dist_from_prom)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(prom_pos,lifetimes , marker = '.', ls = '')
    plt.xlabel('Distance from Prometheus (Co-Rotating Longitude)')
    plt.ylabel('Lifetimes of Clumps (Days)')
#    plt.show()
    fig.savefig('Compare_Distance_from_Prometheus_vs_Lifetime.png')

def list_to_db(c_clump_db,c_approved_list):
    
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
                
    return (c_db)

def compare_width_over_time(c_approved_list):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    all_widths = []
    all_times = []
    for chain in c_approved_list:
        widths = [clump.fit_width_deg for clump in chain.clump_list ]
#        widths -= widths[0]
        time = [clump.clump_db_entry.et/86400. for clump in chain.clump_list]
#        time -= time[0]
        for i, t in enumerate(time):
            all_times.append(t)
            all_widths.append(widths[i])
        if len(time) == 2:
            color = 'blue'
        else:
            color = 'red'
        time = np.array(time)-time[0]
        time_extend = list(time)
        time_extend.append(time[-1] + chain.lifetime_upper_limit)
        
        widths = np.array(widths)
        widths_extend = list(widths)
        widths_extend.append(widths[-1])
        
        plt.plot(time, widths, marker = 'o', ls = '-', color = color, alpha = 1)
        plt.plot(time_extend[len(time_extend)-2::], widths_extend[len(widths_extend)-2::], marker = 'o', ls = '--', color = color, alpha = 0.5)
        ax.set_xlim(0,100.)
        ax.set_xlabel('Delta T (Days)')
        ax.set_ylabel('Delta W (Deg)')
        ax.set_title("Change in Width over a Clump's Lifetime")

    a, b = np.polyfit(all_times, all_widths, 1)
    x = np.arange(0, np.max(all_times))
    y = a*x + b
    print a, b
#    plt.plot(x, y, lw = 2.0, color = 'green')


#    ax2 = fig.add_subplot(212)
#    ax2.set_xlabel('Starting W (Deg)')
#    ax2.set_ylabel('Delta W (Deg)')
#    width_start = []
#    width_delta = []
#    for chain in c_approved_list:
#        widths = np.array([clump.fit_width_deg for clump in chain.clump_list])
#        width_start.append(widths[0])
#        width_delta.append(widths[-1]- widths[0])
##    bins = np.arange(np.min(width_diff), np.max(width_diff) + 2.0, 2.0)
#    ax2.plot(width_start, width_delta, marker = '.', ls = '')
##    plt.show()
    plt.show()
#    plt.savefig(os.path.join('/home/shannon/Paper/Figures/', 'abs_width_vs_time.png'))
    
def compare_height_over_time(c_approved_list):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    all_heights = []
    all_times = []
    for chain in c_approved_list:
        heights = [clump.int_fit_height for clump in chain.clump_list ]
#        heights -= heights[0]
        time = [clump.clump_db_entry.et/86400. for clump in chain.clump_list]
        
        for i, t in enumerate(time):
            all_times.append(t)
            all_heights.append(heights[i])
            
        if len(time) == 2:
            color = 'blue'
        else:
            color = 'red'
        print time[-1], chain.lifetime_upper_limit
        time = np.array(time)- time[0]
        time_extend = list(time)
        time_extend.append(time_extend[-1] + chain.lifetime_upper_limit)
        
        heights = np.array(heights)
        height_extend = list(heights)
        height_extend.append(heights[-1])
        print time_extend, height_extend
        
        plt.plot(time, heights, marker = 'o', ls = '-', color = color, alpha = 0.5)
        plt.plot(time_extend[len(time_extend)-2::], height_extend[len(height_extend)-2::], marker = 'o', ls = '--', color = color, alpha = 0.5)
        ax.set_xlabel('Delta T (Days)')
        ax.set_ylabel('Delta EW')
        ax.set_xlim(0,100.)
        plt.title("Change in Integrated Height over a Clump's Lifetime")

    a, b = np.polyfit(all_times, all_heights, 1)
    x = np.arange(0, np.max(all_times))
    y = a*x + b
    print a, b
#    ax.plot(x, y, lw = 2.0, color = 'green')
#    plt.show()


    #the distribution of the overall changes in height may be gaussian - let's check
#    ax2 = fig.add_subplot(212)
#    height_diff = []
#    for chain in c_approved_list:
#        heights = np.array([clump.int_fit_height for clump in chain.clump_list])
#        height_diff.append(heights[-1] - heights[0])
#    bins = np.arange(np.min(height_diff), np.max(height_diff) + 0.5, 0.5)
#    ax2.hist(height_diff, bins)
#    ax2.set_xlabel('Delta EW')
#    plt.show()
    plt.savefig(os.path.join('/home/shannon/Paper/Figures/', 'abs_height_vs_time.png'))
    
def compare_height_to_true_long(c_approved_list):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for chain in c_approved_list:
        true_longs = []
        heights = np.array([clump.int_fit_height for clump in chain.clump_list ])
        heights -= heights[0]
        for clump in chain.clump_list:
            long_res = 360./len(clump.clump_db_entry.ew_data)
            print clump.g_center, (clump.g_center%360.)/long_res
            clump_peri, clump_inert_long, clump_true_long = clumputil.compare_inert_to_peri(options, clump.clump_db_entry.obsid, (clump.g_center%360.)/long_res, clump.g_center%360.)
            true_longs.append(clump_true_long)
        
        plt.plot(true_longs, heights, marker = '.', ls = '')
    plt.xlabel('True Longitude (Days)')
    plt.ylabel('Delta EW')
    plt.title("Change in Relative Integrated Height wrt True Longitude")

    plt.savefig(os.path.join('/home/shannon/Paper/Figures/', 'height_true_long.png'))
            
def large_clump_freq(c_approved_list, clump_db):
    #find how often large clumps occur
    
    db= {}
    large_ets = []
    large_widths = []
    for chain in c_approved_list:
        for clump in chain.clump_list:
            if clump.fit_width_deg > 30.:
                large_ets.append(clump.clump_db_entry.et_max)
                large_widths.append(clump.fit_width_deg)
                if clump.clump_db_entry.et_max in db.keys():
                    db[clump.clump_db_entry.et_max].append(clump)
                elif clump.clump_db_entry.obsid not in db.keys():
                    db[clump.clump_db_entry.et_max] = []
                    db[clump.clump_db_entry.et_max].append(clump)
                    
    #fill the database with the rest of the obsids that had no large clumps
    for obsid in clump_db.keys():
        et_max = clump_db[obsid].et_max
        if et_max not in db.keys():
            db[et_max] = []
    
   
    et_list = []
    rates = []
    i=0
    for et in sorted(db.keys()):
        print i
        if len(db[et]) == 0:
            et_list.append(et)

        if len(db[et]) != 0:
            et_list.append(et)
            et_list -= et_list[0]
            print et_list
            num_clumps = len(db[et])
            print num_clumps
            rate = num_clumps/(et_list[-1]/86400.)
            print rate
            rates.append(rate)
            
            del et_list
            et_list = [et]
        i += 1
#    print rates
    print 'AVG NUMBER OF LARGE CLUMPS/DAY:', np.mean(rates), ' +/- ', np.std(rates)
#    print np.mean(rates), np.std(rates)
    
#    plt.plot(large_ets, large_widths, marker = '.', ls = '')
#    plt.show()
#    
def examine_clump_delta_w(c_approved_list):
    
    large_growth_list = []
    large_shrink_list = []
    stagnant_list = []
    
    for chain in c_approved_list:
        
        widths = [clump.fit_width_deg for clump in chain.clump_list]
        widths -= widths[0]
        
        if (0. < widths[-1] < 5.) or (-5. < widths[-1] < 0.):
            stagnant_list.append(chain)
        if widths[-1] > 5.:
            large_growth_list.append(chain)
        if widths[-1] < -5.:
            large_shrink_list.append(chain)
            
    for chain in stagnant_list:
        
        print 'STARTING OBS ', chain.clump_list[0].clump_db_entry.obsid
        print 'STARTING LONG ', chain.clump_list[0].g_center
        print 'LENGTH OF CHAIN ', len(chain.clump_list)
        print ' '
        
def plot_clump_nums_over_time(c_approved_list, c_approved_db):
    
    print ''
    print '                USING IDS FROM THE "UNION-COVERAGE" METHOD     '
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
    
    
    c_db = list_to_db(c_approved_db, c_approved_list)
    
    sorted_ids = clumputil.get_sorted_obsid_list(c_approved_db)
    clump_nums = []
    ets = []
    for obsid in c_db.keys():
        if obsid in good_coverage_ids:
            ew_data = c_db[obsid].ew_data
            fraction_coverage = float(ma.count(ew_data))/float(len(ew_data))
            clump_nums.append(len(c_db[obsid].clump_list)/fraction_coverage)
            ets.append(c_db[obsid].et_min)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    labels = ['2004 JAN 01 00:00:00', '2005 JAN 01 00:00:000', '2006 JAN 01 00:00:000', '2007 JAN 01 00:00:000',
                 '2008 JAN 01 00:00:000', '2009 JAN 01 00:00:000', '2010 JAN 01 00:00:000', '2011 JAN 01 00:00:000']
    
    et_ticks = [cspice.utc2et(label) for label in labels]
    print et_ticks
    sec_multiple = 3.15569e7               # number of seconds in 12 months 
    tick_min = np.min(et_ticks)
    tick_min = np.ceil((tick_min/sec_multiple))*sec_multiple
    tick_max = np.max(et_ticks)
    tick_max = np.ceil((tick_max/sec_multiple))*sec_multiple
#    tick_step = (tick_max -tick_min)/10.
    x_ticks = np.arange(tick_min, tick_max + sec_multiple, sec_multiple)
    ax.get_xaxis().set_ticks(et_ticks)
    ax.set_xlim(tick_min - 20*86400, tick_max + 20*86400)
    et_labels = []
    for k, et in enumerate(labels):
#        date = cspice.et2utc(et,'C',0)
        print et, cspice.et2utc(et_ticks[k], 'C', 0)
        et_labels.append(et[:4])
    
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.set_xticklabels(et_labels)
    fig.autofmt_xdate()
    plt.title('Number of Clumps/Obs over Time')

    
    plt.plot(ets, clump_nums, '.')
    plt.savefig('/home/shannon/Paper/Figures/clump_num_over_time.png')
    
def plot_width_brightness_scatter(c_approved_list):
    
    km_per_deg = 881027.02/360.

    width_0 = []
    width_n = []
    med_width = []
    bright_0 = []
    bright_n = []
    med_bright = []
    
    for chain in c_approved_list:
        for i in range(len(chain.clump_list)-1):
        
            width_0.append(chain.clump_list[i].fit_width_deg)
            width_n.append(chain.clump_list[i+1].fit_width_deg)
            bright_0.append(chain.clump_list[i].int_fit_height*km_per_deg/1e4)
            bright_n.append(chain.clump_list[i+1].int_fit_height*km_per_deg/1e4)
            
            med_width.append(np.median([clump.fit_width_deg for clump in chain.clump_list[i:i+2]]))
            med_bright.append(np.median([clump.int_fit_height*km_per_deg/1e4 for clump in chain.clump_list[i:i+2]]))
    
    width_0 = np.array(width_0)
    width_n = np.array(width_n)
    bright_0 = np.array(bright_0)
    bright_n = np.array(bright_n)
    
    fig_a = plt.figure(figsize = (7.0, 5.0))
    ax_a = fig_a.add_subplot(111)
#    plt.plot(width_0, width_n, '.')
    ax_a.set_xlabel('Width_1')
    ax_a.set_ylabel('Width_2')
    ax_a.set_xlim(0, np.max(width_0))
#    a, b = np.polyfit(width_0, width_n, 1)
    x = np.arange(0, np.max(width_0), 0.01 )
    y = x
#    print a, b
    ax_a.plot(x, y, lw = 2.0, color = 'green')
#    plt.colorbar(med_bright, cmap = 'hsv')
    width_graph = plt.scatter(width_0, width_n, s = 50, c = med_bright, cmap = 'Paired', facecolor = 'none', edgecolor = med_bright)
    bar = plt.colorbar(width_graph, cmap = 'Paired')
    bar.set_label('Median Brightness x 10^4 km^2')
    
    plt.savefig('/home/shannon/Paper/Figures/w0_vs_wn.png', dpi = 1000)
    
    
    fig_b = plt.figure(figsize = (7.0, 5.0))
    ax_b = fig_b.add_subplot(111)
#    plt.plot(bright_0, bright_n, '.')
    ax_b.set_xlabel('Brightness_1 x 10^4 km^2')
    ax_b.set_ylabel('Brightness_2 x 10^4 km^2')
    bright_graph = plt.scatter(bright_0, bright_n, s = 50, c = med_width, cmap = 'Paired', facecolor = 'none', edgecolor = med_bright)
    bar = plt.colorbar(bright_graph, cmap = 'Paired')
    bar.set_label('Median Width (deg)')
    ax_b.set_xlim(0, np.max(bright_0)+1)
    
#    a, b = np.polyfit(bright_0, bright_n, 1)
    x = np.arange(0, np.max(bright_0), 0.01)
    y = x
#    print a, b
    ax_b.plot(x, y, lw = 2.0, color = 'green')
    plt.savefig('/home/shannon/Paper/Figures/b0_vs_bn.png', dpi = 1000)
    
    
#    plt.show()    
    
def plot_clumps_vs_prometheus(c_approved_list, c_approved_db):
    
    clump_db = list_to_db(c_approved_db, c_approved_list)
    
    long_differences = []
    
    for obs in clump_db.keys():
        
        p_rad, p_long = ringutil.prometheus_close_approach(clump_db[obs].et_min, clump_db[obs].et_min_longitude)
        for clump in clump_db[obs].clump_list:
            
            long_diff = clump.g_center - p_long
#            print long_diff
#            if long_diff >= 180.: long_diff = 360. - long_diff
#            if long_diff <= -180.: long_diff = -1*(360 + long_diff)
            if long_diff < 0: long_diff += 360
            long_differences.append(long_diff)
            break
            
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.arange(0, 361.,30.)
    plt.hist(long_differences, bins, weights = np.zeros_like(long_differences) + 1./len(long_differences))
    ax.set_xlabel('Clump Longitude - Prometheus Closest Longitude')
    ax.set_ylabel('Fraction of clumps')
#    plt.savefig('/home/shannon/Paper/Figures/clump_prometheus_long_diff.png')

    long_differences = []
    
    for obs in clump_db.keys():
        
        p_rad, p_long = ringimage.saturn_to_prometheus(clump_db[obs].et_min)
        for clump in clump_db[obs].clump_list:
            
            long_diff = clump.g_center - p_long
#            print long_diff
#            if long_diff >= 180.: long_diff = 360. - long_diff
#            if long_diff <= -180.: long_diff = -1*(360 + long_diff)
            if long_diff < 0: long_diff += 360
            long_differences.append(long_diff)
            break
            
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.arange(0, 361., 30.)
    plt.hist(long_differences, bins, weights = np.zeros_like(long_differences) + 1./len(long_differences))
    ax.set_xlabel('Clump Longitude - Prometheus Longitude')
    ax.set_ylabel('Fraction of clumps')
    
    plt.show()
    
def width_brightness_corr_hists(c_approved_list):
    
    km_per_deg = 881027.02/360.

    
    fract_brightness_dt = []
    dw_dt = []
    fract_w_dt = []
    
    bright_pos = 0
    bright_neg = 0
    
    for chain in c_approved_list:
        for i in range(len(chain.clump_list)-1):
            
            clump1 = chain.clump_list[i]
            clump2 = chain.clump_list[i+1]
            
            width1 = clump1.fit_width_deg
            width2 = clump2.fit_width_deg
            bright1 = clump1.int_fit_height*km_per_deg/1e4
            bright2 = clump2.int_fit_height*km_per_deg/1e4
    
            time1 = clump1.clump_db_entry.et_min/86400.
            time2 = clump2.clump_db_entry.et_min/86400.
            
            dt = time2-time1
            
            if bright2/bright1 < 1:
                bright_neg += 1
            else:
                bright_pos += 1
                
            fract_brightness_dt.append((bright2/bright1)**(1/dt))
            dw_dt.append((width2 - width1)/dt)
            
            fract_width = ((width2 - width1)/np.mean((width1, width2)))/dt
            fract_w_dt.append(fract_width)
        
    print '# POS BRIGHTNESS', bright_pos, 'NEG', bright_neg
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fract_brightness = np.log10(np.array(fract_brightness_dt))
    bins = np.arange(np.min(fract_brightness), np.max(fract_brightness) + 0.01, 0.01)
#    bins = np.logspace(-3, 1, num = 50)
    fract_brightness_weights = np.zeros_like(fract_brightness) + 1./len(fract_brightness)
    plt.hist(fract_brightness, bins, weights = fract_brightness_weights)
    ax.set_xlabel('Fractional Brightness per Day = (B2/B1)**(1/dt)')
    ax.set_ylabel('Fractional Number of Pairwise Clumps')
    
#    plt.savefig('/home/shannon/Paper/Figures/fractional_brightness_per_day.png')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.arange(np.min(dw_dt), np.max(dw_dt)+ 0.25, 0.25)
    dw_dt_weights = np.zeros_like(dw_dt) + 1./len(dw_dt)
    plt.hist(dw_dt, bins, weights = dw_dt_weights)
    ax.set_xlabel('Delta W/Delta t')
    ax.set_ylabel('Fractional Number of Pairwise Clumps')
#    plt.savefig('/home/shannon/Paper/Figures/width_change_per_day.png')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
#    fract_w_dt = np.log10(np.array(fract_w_dt))
    print fract_w_dt
    bins = np.arange(np.min(fract_w_dt), np.max(fract_w_dt)+ 0.01, 0.01)
#    bins = np.logspace(-3,0, 20)
    fract_w_weights = np.zeros_like(fract_w_dt) + 1./len(fract_w_dt)
    plt.hist(fract_w_dt, bins, weights = fract_w_weights)
    ax.set_xlabel('Fractional Width per Day = (delta W/<W>)/dt)')
    ax.set_ylabel('Fractional Number of Pairwise Clumps')
#    plt.savefig('/home/shannon/Paper/Figures/fractional_width_per_day.png')
    plt.show()
    
#use the approved list of clumps for analysis - full resolution set
#gives us the modified clump database with all of the clumps we consider valid

c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_list_w_errors.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
clump_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()


#c_approved_db = list_to_db(clump_db, c_approved_list)
#
#compare_time_lifetime(c_approved_db,c_approved_list)
#compare_size_lifetime(c_approved_list)
#compare_clumpnums_lifetime(c_approved_db, c_approved_list)
#compare_prometheus_lifetime(c_approved_db, c_approved_list)


#examine_clump_delta_w(c_approved_list)
#large_clump_freq(c_approved_list, clump_db)
#compare_width_over_time(c_approved_list)
#compare_height_over_time(c_approved_list)

#plot_clump_nums_over_time(c_approved_list, clump_db)
#compare_height_to_true_long(c_approved_list)
#plot_width_brightness_scatter(c_approved_list)
#plot_clumps_vs_prometheus(c_approved_list, clump_db)
width_brightness_corr_hists(c_approved_list)
