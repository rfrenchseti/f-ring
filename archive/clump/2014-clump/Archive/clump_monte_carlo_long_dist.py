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
import random
import scipy.stats as st
import clumputil
import ringutil

def monte_carlo_long_dist():
    tries = 1
    observations = 1000000
    diff_master_list = []
    
    master_counts_list = []
    for a in range(tries):
        print a
        diff_list = []
        for i in range(observations):
            num_clumps = random.randrange(2., 20.)  #we've only ever seen a max of 18 clumps/ID
            long_list = np.zeros(num_clumps)
            for k in range(num_clumps):
                long_list[k] = random.random()*360.
        #        print long_list
            sorted_list = sorted(long_list)
        #    print sorted_list
            m = 0
            while m < len(sorted_list)-1:
                diff = sorted_list[m+1]- sorted_list[m]
                if diff >= 180: diff = 360-diff
                diff_list.append(diff)
                m +=1
            #wrap-around
    #        diff_last = sorted_list[0] + 360. - sorted_list[-1]
    #        diff_list.append(diff_last) 
        
        diff_list = np.array(diff_list)
        
        
    #    min_diff = np.min(diff_list)
    #    max_diff = np.max(diff_list)
        bins = np.arange(0., 180. + 2.5, 2.5)
    
        diff_weights = np.zeros_like(diff_list) + 1./len(diff_list)
        counts, bins, patches = plt.hist(diff_list, bins, weights = diff_weights)
        master_counts_list.append(counts)
        
    master_counts_list = np.array(master_counts_list)
    
    median_counts = np.median(master_counts_list, axis = 0)     #median combine
    print len(median_counts), len(bins)
    
#    fig = plt.figure()
#    plt.title('Monte Carlo Longitude Dist')
#    ax = fig.add_subplot(111)
#    plt.plot(bins[:-1], median_counts)
#    plt.ylabel('Median Number of Clumps')
#    plt.xlabel('Difference in Longitude of Adjacent Clumps (deg)')
#    plt.savefig(os.path.join('/home/shannon/Paper/Figures/', 'monte_carlo_long_dist.png'))      

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
    
    plt.plot(simulation_dist, 'blue')
    plt.plot(actual_dist, 'red')
    plt.show()
    
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

def longitude_distribution(c_approved_list, c_approved_db):
    
    c_db = list_to_db(c_approved_list, c_approved_db)
    
    diff_list = []
    for key in c_db.keys():
        clump_list = c_db[key].clump_list
        sorted_longitudes = sorted([clump.g_center for clump in clump_list])
#        print sorted_longitudes
        
        m = 0
        while m < len(sorted_longitudes)-1:
            diff = sorted_longitudes[m+1]- sorted_longitudes[m]
            if diff >= 180: diff = 360-diff
            diff_list.append(diff)
            m +=1
            #wrap-around
#            diff_last = sorted_longitudes[0] + 360. - sorted_longitudes[-1]
#            diff_list.append(diff_last)
        
    diff_list = np.array(diff_list)

    bins = np.arange(0., 180. + 2.5, 2.5)
    
    fig = plt.figure()
    plt.title('Actual Longitude Distribution')
    ax = fig.add_subplot(111)
    diff_weights = np.zeros_like(diff_list) + 1./len(diff_list)
    counts, bins, patches = plt.hist(diff_list, bins, weights = diff_weights)
    plt.ylabel('Number of Clumps')
    plt.xlabel('Difference in Longitude of Adjacent Clumps (deg)')
#    plt.savefig(os.path.join('/home/shannon/Paper/Figures/', 'clump_actual_long_dist.png'))        
    plt.show()
    return (counts)


c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data',  'approved_list_w_errors.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()

simulation_counts = monte_carlo_long_dist()
actual_counts = longitude_distribution(c_approved_list, c_approved_db)

ks_test(simulation_counts, actual_counts)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(simulation_counts, 'red', marker = 'o', ls = '')
plt.plot(actual_counts, 'blue', marker = 'o', ls = '')
plt.show()
    
