import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import ringutil
import clumputil
import numpy.random as rand
import numpy.ma as ma
import cspice

# All
#start_date = '2004 JAN 01 00:00:00'
#end_date = '2011 JAN 01 00:00:00'

# First - course
#start_date = '2006 SEP 01 00:00:00'
#end_date = '2006 DEC 01 00:00:00'

# Second - course
#start_date = '2006 DEC 01 00:00:00'
#end_date = '2007 JUN 01 00:00:00'

# Third - course
#start_date = '2008 JUN 01 00:00:00'
#end_date = '2009 JAN 01 00:00:00'

# Fourth - course
#start_date = '2009 JAN 01 00:00:00'
#end_date = '2009 SEP 01 00:00:00'

def run_clump_creation_mc(c_approved_db, db_by_time, real_widths, real_times):

    
    tries = 1000000
    
    fraction_seen_dist = []
    
    clump_lifetimes = np.arange(2, 90.+2, 2)
    for lifetime in clump_lifetimes:
        time_range = []
        while len(time_range) == 0:
            time_range = np.arange(cspice.utc2et(start_date), (cspice.utc2et(end_date)- lifetime*86400.) + 86400., 86400.)
        num_seen = 0
        print lifetime
        for k in range(tries):
#            print k
            rand_width = real_widths[rand.randint(0,len(real_widths))]
            rand_long = rand.random()*360.
            start_time = time_range[rand.randint(0, len(time_range))]
            
            sorted_times = np.array(sorted(db_by_time.keys()))
            end_time = start_time + lifetime*86400.
            next_times = []
            for time in sorted_times:
                if start_time <= time <= end_time:
                    next_times.append(time)
            
            if len(next_times) < 2:
                #there's already no way that we would detect a clump
                continue
               
            idx_list = []
            
            for time_num, time in enumerate(next_times):
                obs = db_by_time[time]
                ew_data = c_approved_db[obs].ew_data
                tripled_ew_data = np.tile(ew_data, 3)
                long_res = 360./len(ew_data)
                
                left_bound = (rand_long - rand_width/2)/long_res + len(ew_data)
                right_bound = (rand_long + rand_width/2)/long_res + len(ew_data)
                    
                ew_range = tripled_ew_data[left_bound:right_bound+1]
                if ma.count(ew_range) >= (len(ew_range)*0.90):              #must have 90% of the area of the clump covered with valid data
                    idx_list.append(time_num)
                    num_skipped = 0
          
            # Pairwise idx_list entries must be <= 2 apart to have seen the clump
            ok = False
            for idx_num in range(len(idx_list)-1):
                if idx_list[idx_num]+2 >= idx_list[idx_num+1]:
                    ok = True
            
            if ok:
                num_seen += 1
#            print '--------------------------------------'
    
        fraction_seen = float(num_seen)/float(tries)
#        print fraction_seen

        fraction_seen_dist.append(fraction_seen)

    print start_date
    print end_date
    print 'DAYS', (cspice.utc2et(end_date)-cspice.utc2et(start_date))/86400.
    
    print fraction_seen_dist
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(clump_lifetimes, fraction_seen_dist, '.')
#    plt.savefig('/home/shannon/Paper/Figures/clump_creation_rate_mc.png')
    plt.show()

def analyze_clump_creation(c_approved_db, c_approved_list, db_by_time):
    jan2004_jan2011 = [0.00166, 0.00492, 0.00984, 0.01763, 0.02586, 0.03379, 0.0432, 0.05683, 0.07285, 0.0908, 0.10846, 0.128, 0.14518, 0.16212, 0.17771, 0.19513, 0.20981, 0.22252, 0.2349, 0.2466, 0.25897, 0.26968, 0.27892, 0.28804, 0.30019, 0.30713, 0.3104, 0.31895, 0.32601, 0.33217, 0.33875, 0.34531, 0.3471, 0.35383, 0.35983, 0.36406, 0.37304, 0.37651, 0.38302, 0.38696, 0.39149, 0.39917, 0.40664, 0.40734, 0.41445, 0.42067, 0.42795, 0.4301, 0.43391, 0.44321] # 2557 days
    
#    sep2006_jun2007 = [0.0, 0.00407, 0.00819, 0.01227, 0.01582, 0.0211, 0.03784, 0.07172, 0.11352, 0.18181, 0.25307, 0.32105, 0.38841, 0.45274, 0.52742, 0.59758, 0.65691, 0.70214, 0.74954, 0.79444, 0.83862, 0.87138, 0.90064, 0.92142, 0.93937, 0.9526, 0.96125, 0.96485, 0.96831, 0.97042, 0.97409, 0.97696, 0.9801, 0.98266, 0.98466, 0.98646, 0.9879, 0.98916, 0.99025, 0.99069, 0.99089, 0.9914, 0.99163, 0.99295, 0.9939, 0.99395, 0.99448, 0.9949, 0.99515, 0.99614] # 273 days

    sep2006_dec2006 = [0.0, 0.01286, 0.02528, 0.03871, 0.05092, 0.06583, 0.11253, 0.18165, 0.26248, 0.35883, 0.43144, 0.51266, 0.58597, 0.64371, 0.7002, 0.74558, 0.78475, 0.81286, 0.83997, 0.87049, 0.90113, 0.92524, 0.94737, 0.95068, 0.95625, 0.96167, 0.96596, 0.97296, 0.97924, 0.98229, 0.985, 0.98725, 0.98952, 0.99167, 0.99435, 0.99808, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # 91 days

    dec2006_jun2007 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00455, 0.02514, 0.05766, 0.12279, 0.19546, 0.2631, 0.33229, 0.40681, 0.48321, 0.5619, 0.62136, 0.67237, 0.70997, 0.7496, 0.79227, 0.83296, 0.87283, 0.89917, 0.92533, 0.93841, 0.94766, 0.9522, 0.95565, 0.9605, 0.96435, 0.96892, 0.97353, 0.97749, 0.97979, 0.98159, 0.98296, 0.98422, 0.98382, 0.98502, 0.98635, 0.98684, 0.98731, 0.98855, 0.99009, 0.99087, 0.99164, 0.99297, 0.99366, 0.99544] # 182 days

    jun2008_jan2009 = [0.00221, 0.01304, 0.03793, 0.06192, 0.09345, 0.13193, 0.16888, 0.21575, 0.26943, 0.31862, 0.37384, 0.42767, 0.47782, 0.52609, 0.57215, 0.61627, 0.65459, 0.68922, 0.71715, 0.74643, 0.77066, 0.79448, 0.8167, 0.83906, 0.85703, 0.87386, 0.89259, 0.90546, 0.91624, 0.9279, 0.93677, 0.94539, 0.9536, 0.96081, 0.96605, 0.96872, 0.97233, 0.97639, 0.97985, 0.98351, 0.98629, 0.98811, 0.99041, 0.99257, 0.99474, 0.99623, 0.99799, 0.99891, 0.99931, 0.99988] # 214 days
    
    jan2009_sep2009 = [0.00555, 0.01404, 0.02682, 0.05617, 0.08497, 0.11485, 0.14779, 0.18441, 0.2204, 0.25398, 0.29177, 0.33204, 0.37144, 0.40538, 0.43539, 0.46644, 0.49538, 0.52611, 0.55745, 0.5904, 0.61751, 0.64221, 0.66453, 0.69148, 0.71589, 0.72742, 0.74008, 0.746, 0.75966, 0.76968, 0.78, 0.79104, 0.79765, 0.80659, 0.81619, 0.82376, 0.82847, 0.83436, 0.84269, 0.8481, 0.85516, 0.86044, 0.86601, 0.87381, 0.87994, 0.88626, 0.88949, 0.89549, 0.90279, 0.90821] # 243 days

    mc_data = [('2004 JAN 01 00:00:00', '2011 JAN 01 00:00:00', jan2004_jan2011),
               ('2006 SEP 01 00:00:00', '2006 DEC 01 00:00:00', sep2006_dec2006),
               ('2006 DEC 01 00:00:00', '2007 JUN 01 00:00:00', dec2006_jun2007),
               ('2008 JUN 01 00:00:00', '2009 JAN 01 00:00:00', jun2008_jan2009),
               ('2009 JAN 01 00:00:00', '2009 SEP 01 00:00:00', jan2009_sep2009)]
    
    for mc_datum in mc_data:
        start_date, end_date, frac_list = mc_datum
        start_time = cspice.utc2et(start_date)
        end_time = cspice.utc2et(end_date)
        lifetime_vals = np.arange(2, len(frac_list)*2+2, 2)
        
        num_clumps = 0
        for clump_chain in c_approved_list:
            for clump in clump_chain.clump_list:
                if start_time <= clump.clump_db_entry.et <= end_time:
                    num_clumps += 1
                    break
        
        num_days = float(int((end_time-start_time)/86400.))
        
        print start_date, end_date, num_days, num_clumps
        
        for lifetime_idx, lifetime in enumerate(lifetime_vals):
            frac = frac_list[lifetime_idx]
            
            if frac == 0.:
                frac = 0.01
                
            total_clumps = num_clumps / frac
            clump_days = total_clumps * lifetime
            clumps_per_day = clump_days / num_days
            
            if lifetime == 30 or lifetime == 40 or lifetime == 50:
                print 'LIFE %3d  FRAC %.3f  #CLUMPS %6d  TOTAL %6d  CLUMPDAYS %6d  CLUMPS/DAY %7.2f  PRODRATE %.2f' % (lifetime, frac, num_clumps, total_clumps, clump_days, clumps_per_day, clumps_per_day/lifetime) 

        print
        
c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data',  'approved_list_w_errors.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()


real_widths = []
for key in c_approved_db.keys():
    clump_list = c_approved_db[key].clump_list
    real_widths.append(np.median([clump.fit_width_deg for clump in clump_list]))
    
real_widths.sort()

real_times = []
for key in c_approved_db.keys():
    real_times.append(c_approved_db[key].et_min)
    
real_times.sort()

db_by_time = {}
for key in c_approved_db.keys():
    
    db_by_time[c_approved_db[key].et_min] = key
    
#run_clump_creation_mc(c_approved_db, db_by_time, real_widths, real_times)
    
analyze_clump_creation(c_approved_db, c_approved_list, db_by_time)
