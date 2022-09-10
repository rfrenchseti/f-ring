'''
Run a monte carlo simulation on the longitudinal distribution of clumps in the F ring

Variables to consider:
- How many clumps/ obser vation? - this may need to be fixed for now 
- How many observations? - also may need to be fixed

@author - Shannon Hicks
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import numpy.random as rand
import scipy.stats as st
import clumputil
import ringutil

use_cumulative = True

master_step = 1.
master_bins = np.arange(0., 180. + master_step, master_step)

def monte_carlo_long_dist(real_widths, dist='uniform'):
    debug = False
    
    trials = 1
    observations = 1000
    diff_master_list = []
    master_counts_list = []
    
    for trial_num in range(trials):
        diff_list = []
        for observation_num in range(observations):
#            print observation_num
            profile_ok = False
            while not profile_ok:
                # Try to create a valid clump profile
                num_clumps = rand.randint(2, 20)  # we've only ever seen a max of 18 clumps/ID
                long_list = np.zeros(num_clumps)
                width_list = np.zeros(num_clumps)
                for clump_num in range(num_clumps):
                    profile_ok = False
                    for clump_try_num in range(20): # just try 20 times to fit a clump, then give up and start over
                        width = real_widths[rand.randint(0,len(real_widths)-1)]
#                        width = 0
#                        while width < 3:
#                            dist_part = rand.random()
#                            if dist_part < .1:
#                                width = rand.random()*2+3
#                            elif dist_part < .5:
#                                width = rand.random()*2.5+5
#                            elif dist_part < .7:
#                                width = rand.random()*2.5+7.5
#                            else:
#                                width = rand.random()*2.5+10
#                            width = rand.normal(7.5,2.5)
                        if dist == 'uniform':
                            center = rand.random()*360.
                        elif dist == 'gauss30':
                            center = rand.normal(180, 30, 1)
                        elif dist == 'gauss60':
                            center = rand.normal(180, 60, 1)
                        elif dist == 'gauss60oneside':
                            center = 0
                            while center < 180:
                                center = rand.normal(180, 60, 1)
                        # Look for overlapping clumps
                        if debug:
                            print 'For clumpnum', clump_num, '/', num_clumps, 'try#', clump_try_num, 'long', center, 'width', width
                        for j in range(clump_num):
                            x1 = long_list[j]-width_list[j]/2
                            x2 = long_list[j]+width_list[j]/2
                            y1 = center-width/2
                            y2 = center+width/2
                            if ((x2 >= y1 and x1 <= y2) or
                                (x2+360 >= y1 and x1+360 <= y2) or
                                (x2-360 >= y1 and x1-360 <= y2)):
                                break
                        else: # No overlap - the clump is OK!
                            profile_ok = True
                            if debug:
                                print 'OK!'
                            break
                    else: # We didn't find a good clump
                        if debug:
                            print 'Failed'
                        profile_ok = False
                        break
                    if profile_ok:
                        if debug:
                            print 'Recording'
                        long_list[clump_num] = center
                        width_list[clump_num] = width
            sorted_list = sorted(long_list)
            m = 0
            while m < len(sorted_list)-1:
                diff = sorted_list[m+1]- sorted_list[m]
                if diff >= 180: diff = 360-diff
                diff_list.append(diff)
                m +=1
        
        diff_list = np.array(diff_list)
        
        diff_weights = np.zeros_like(diff_list) + 1./len(diff_list)
        counts, bins = np.histogram(diff_list, master_bins, weights = diff_weights)
        if use_cumulative:
            counts = np.cumsum(counts)
        master_counts_list.append(counts)
        
    master_counts_list = np.array(master_counts_list)
    
    median_counts = np.median(master_counts_list, axis = 0)     #median combine
    
    return median_counts

def ks_test(median_counts, actual_counts):

    def cumulative_distribution(array, output):
            start = 0
            for num in array:
                new = num + start
                output.append(new)
                start = new
    #        print start
            return(output)
    
    simulation_dist = cumulative_distribution(median_counts, [])
    actual_dist = cumulative_distribution(actual_counts, [])
    
#    plt.plot(simulation_dist, 'blue')
#    plt.plot(actual_dist, 'red')
#    plt.show()
    
    print simulation_dist
    print actual_dist
    d, pks = st.ks_2samp(simulation_dist, actual_dist)
    
    #create the cumulative distribution
    print '--------------|   D   |   p    --------'
    print '---------------------------------------'
    print 'Longitude Dist | %5.3f | %5.5f '%(d, pks)
    
def list_to_db(c_approved_list, c_approved_db):
    
    c_db = {}
    for chain in c_approved_list:
        for clump in chain.clump_list:
#            if clump.fit_width_deg < 35.:
            c_clump_db_entry = clumputil.ClumpDBEntry()
            obsid = clump.clump_db_entry.obsid
            if obsid not in c_db.keys():
                c_clump_db_entry = c_approved_db[obsid]
                c_clump_db_entry.clump_list = []
                
                c_clump_db_entry.clump_list.append(clump)
                c_db[obsid] = c_clump_db_entry
                
            elif obsid in c_db.keys():
                c_db[obsid].clump_list.append(clump)
    
    return c_db

def longitude_distribution(approved_list, approved_db):
    
    db = list_to_db(approved_list, approved_db)
    
    diff_list = []
    for key in db.keys():
        clump_list = db[key].clump_list
        sorted_longitudes = sorted([clump.g_center for clump in clump_list])
#        print sorted_longitudes
        
        m = 0
        while m < len(sorted_longitudes)-1:
            diff = sorted_longitudes[m+1]- sorted_longitudes[m]
            if diff >= 180: diff = 360-diff
            if diff >= 2: # We can't be any closer than this - they must be the same clump if so
                diff_list.append(diff)
            m +=1
            #wrap-around
#            diff_last = sorted_longitudes[0] + 360. - sorted_longitudes[-1]
#            diff_list.append(diff_last)
        
    diff_list = np.array(diff_list)

#    fig = plt.figure()
#    plt.title('Actual Longitude Distribution')
#    ax = fig.add_subplot(111)
    diff_weights = np.zeros_like(diff_list) + 1./len(diff_list)
    counts, bins = np.histogram(diff_list, master_bins, weights = diff_weights)
    if use_cumulative:
        counts = np.cumsum(counts)
#    plt.ylabel('Number of Clumps')
#    plt.xlabel('Difference in Longitude of Adjacent Clumps (deg)')
##    plt.savefig(os.path.join('/home/shannon/Paper/Figures/', 'clump_actual_long_dist.png'))        
#    plt.show()
    return (counts)

def get_mini_jet_dist(attree_db):
    
    diff_list = []
    for key in attree_db.keys():
        clump_list = attree_db[key]
        sorted_longitudes = sorted(clump_list)
#        print sorted_longitudes
        
        m = 0
        while m < len(sorted_longitudes)-1:
            diff = sorted_longitudes[m+1]- sorted_longitudes[m]
            if diff >= 180: diff = 360-diff
            if diff >= 2: # We can't be any closer than this - they must be the same clump if so
                diff_list.append(diff)
            m +=1

        
    diff_list = np.array(diff_list)
    diff_weights = np.zeros_like(diff_list) + 1./len(diff_list)
    counts, bins = np.histogram(diff_list, master_bins, weights = diff_weights)

    counts = np.cumsum(counts)
        
#    fig = plt.figure()
#    plt.title('Attree Mini Jets Longitude Distribution')
#    ax = fig.add_subplot(111)
#    diff_weights = np.zeros_like(diff_list) + 1./len(diff_list)
#    counts, bins, patches = plt.hist(diff_list, master_bins, weights = diff_weights)
##    plt.show()
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    bin_centers = (master_bins[1:] + master_bins[:-1])/2
#    plt.plot(bin_centers, np.cumsum(counts))
#    plt.show()
    return counts

c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data',  'approved_list_w_errors.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()

v_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'voyager_approved_clumps_list.pickle')
v_approved_list_fp = open(v_approved_list_fp, 'rb')
v_approved_db, v_approved_list = pickle.load(v_approved_list_fp)
v_approved_list_fp.close()

attree_path = os.path.join(ringutil.ROOT, 'attree_minijet_longitudes.csv')
attree_fp = open(attree_path, 'rb')

attree_db = {}
for line in attree_fp:
#    print line
    line_list = line.split(',')
    obsid = line_list[0]
    long_temp = line_list[1::]
    final_longs = []
    for long in long_temp:
#        print long
        if (long != '') and (long != '\n'):
            final_longs.append(float(long))
#    print obsid, final_longs
    attree_db[obsid] = final_longs
    
attree_counts = get_mini_jet_dist(attree_db)

attree_fp.close()


real_widths = []
for key in c_approved_db.keys():
    clump_list = c_approved_db[key].clump_list
    real_widths.append(np.median([clump.fit_width_deg for clump in clump_list]))
    
real_widths.sort()
    
c_actual_counts = longitude_distribution(c_approved_list, c_approved_db)
v_actual_counts = longitude_distribution(v_approved_list, v_approved_db)
simulation_counts_uniform = monte_carlo_long_dist(real_widths, dist='uniform')
print 'UNIFORM'
print ''
print 'CASSINI COMPARE'
ks_test(simulation_counts_uniform, c_actual_counts)
print ''
print 'VOYAGER COMPARE'
ks_test(simulation_counts_uniform, v_actual_counts)

simulation_counts_gauss30 = monte_carlo_long_dist(real_widths, dist='gauss30')
print 'GAUSS 30'
print ''
print 'CASSINI COMPARE'
ks_test(simulation_counts_gauss30, c_actual_counts)
print ''
print 'VOYAGER COMPARE'
ks_test(simulation_counts_gauss30, v_actual_counts)

simulation_counts_gauss60  = monte_carlo_long_dist(real_widths, dist='gauss60')
print 'GAUSS 60'
print ''
print 'CASSINI COMPARE'
ks_test(simulation_counts_gauss60, c_actual_counts)
print ''
print 'VOYAGER COMPARE'
ks_test(simulation_counts_gauss60, v_actual_counts)

simulation_counts_gauss60oneside  = monte_carlo_long_dist(real_widths, dist='gauss60oneside')
print 'GAUSS 60 ONE SIDE'
print ''
print 'CASSINI COMPARE'
ks_test(simulation_counts_gauss60oneside, c_actual_counts)
print ''
print 'VOYAGER COMPARE'
ks_test(simulation_counts_gauss60oneside, v_actual_counts)


bin_centers = (master_bins[1:] + master_bins[:-1])/2
fig = plt.figure(figsize = (3.5, 3.0))
ax = fig.add_subplot(111)
plt.plot(bin_centers, simulation_counts_uniform, '-', color='black', lw=1.5, label='Uniform')
plt.plot(bin_centers, simulation_counts_gauss30, '--', color='black', lw=1.5, label='Gauss 30', dashes=(4,4))
plt.plot(bin_centers, simulation_counts_gauss60, '-', color='black', lw=1.5, label='Gauss 60', dashes=(8,4))
plt.plot(bin_centers, simulation_counts_gauss60oneside, '-.', color='black', lw=1.5, label='Gauss 60 One side')
plt.plot(bin_centers, c_actual_counts, '-', color='black', lw=2.5, label='Cassini Actual')
plt.plot(bin_centers, v_actual_counts, '-', color = 'black', lw = 2.5, label = 'Voyager Actual', dashes = (12, 4))
plt.plot(bin_centers, attree_counts, '*', color = 'black', lw = 2.5, label = 'Attree Mini Jets')
#ax.set_yscale('log')
plt.xlim(0,90)
ax.set_ylabel('Cumulative Distribution')
ax.set_xlabel('Differences in Co-Rotating Longitude ( $\mathbf{^o}$)')
#plt.legend(loc='lower right')
leg = plt.legend(loc = 'lower right')
leg.get_frame().set_alpha(0.0)
leg.get_frame().set_visible(False)
fig.tight_layout()
#plt.savefig(os.path.join('/home/shannon/Paper/Figures/', 'clump_longitude_dist.png'))
plt.show()
