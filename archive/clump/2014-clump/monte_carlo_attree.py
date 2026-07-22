
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import numpy.random as rand
import scipy.stats as st
import clumputil
import ringutil
import numpy.ma as ma


master_step = 1.
master_bins = np.arange(0., 180. + master_step, master_step)


def monte_carlo_attree(real_widths):
    trials = 1000000
    
    num_obs = 0
    num_in_jet = 0
    
    for trial_num in range(trials):
        num_jets = -1
        while num_jets < 0:
            num_jets = int(rand.normal(14, 13.1))
#            num_jets = rand.randint(2,40)
        jet_longs = rand.random(num_jets)*360.
        clump_long = rand.random()*360.
        clump_width = real_widths[rand.randint(0,len(real_widths)-1)]
        in_jet = False
        for jet_long in jet_longs:
            if clump_long-clump_width/2 < jet_long and clump_long+clump_width/2 > jet_long:
                in_jet = True
            if (clump_long-clump_width/2 +360. <= jet_long + 360.) and ((clump_long + clump_width/2 + 360.) >= jet_long + 360.):
                in_jet = True
            if ((clump_long-clump_width/2 -360.) >= jet_long -360.) and ((clump_long + clump_width/2 - 360.) <= jet_long -360.):
                in_jet = True
        if in_jet:
            num_in_jet += 1
        num_obs += 1
    
    return float(num_in_jet) / float(num_obs)

def monte_carlo_attree_actual_data(real_widths, attree_db, c_approved_db):
    
    trials = 1000000
    
    num_obs = 0
    num_in_jet = 0
    
    for trial_num in range(trials):
#        print len(attree_db.keys())
        obsid_idx = rand.randint(0, len(attree_db.keys()))
        obsid = attree_db.keys()[obsid_idx]
#        print obsid
        ew_data = c_approved_db[obsid].ew_data
        fraction_coverage = float(ma.count(ew_data))/float(len(ew_data))
#        print fraction_coverage
        num_jets = np.round(len(attree_db[obsid])/fraction_coverage)

        jet_longs = rand.random(num_jets)*360.
        clump_idx = rand.randint(0, len(c_approved_db[obsid].clump_list))
#        clump_long = c_approved_db[obsid].clump_list[clump_idx].g_center
#        clump_width = c_approved_db[obsid].clump_list[clump_idx].fit_width_deg
        clump_long = rand.random()*360.
        clump_width = real_widths[rand.randint(0,len(real_widths)-1)]
        in_jet = False
        for jet_long in jet_longs:
            if clump_long-clump_width/2 < jet_long and clump_long+clump_width/2 > jet_long:
                in_jet = True
            if (clump_long-clump_width/2 +360. >= jet_long) and ((clump_long + clump_width/2 + 360.) <= jet_long):
                in_jet = True
            if ((clump_long-clump_width/2 -360.) >= jet_long) and ((clump_long + clump_width/2 - 360.) <= jet_long):
                in_jet = True
        if in_jet:
            num_in_jet += 1
        num_obs += 1
    
    return float(num_in_jet) / float(num_obs)

def plot_mini_jet_dist(attree_db):
    
    diff_list = []
    num_mj_per_obs = []
    for key in attree_db.keys():
        clump_list = attree_db[key]
        sorted_longitudes = sorted(clump_list)
        num_mj_per_obs.append(len(clump_list))
#        print sorted_longitudes
        
        m = 0
        while m < len(sorted_longitudes)-1:
            diff = sorted_longitudes[m+1]- sorted_longitudes[m]
            if diff >= 180: diff = 360-diff
            if diff >= 2: # We can't be any closer than this - they must be the same clump if so
                diff_list.append(diff)
            m +=1

        
    diff_list = np.array(diff_list)

    fig = plt.figure()
    plt.title('Attree Mini Jets Longitude Distribution')
    ax = fig.add_subplot(111)
    diff_weights = np.zeros_like(diff_list) + 1./len(diff_list)
    counts, bins, patches = plt.hist(diff_list, master_bins, weights = diff_weights)
    plt.savefig('/home/shannon/Paper/Figures/mini_jets_long_dist.png')
    
    fig = plt.figure()
    plt.title('Attree Cumulative Distribution')
    ax = fig.add_subplot(111)
    bin_centers = (master_bins[1:] + master_bins[:-1])/2
    plt.plot(bin_centers, np.cumsum(counts))
    ax.set_xlim(0,90.)
    plt.savefig('/home/shannon/Paper/Figures/mini_jets_cum_dist.png')
    
    fig =plt.figure()
    plt.title('Number of Mini Jets per Profile')
    ax = fig.add_subplot(111)
    bins = np.arange(0, np.max(num_mj_per_obs) + 2, 2)
    plt.hist(num_mj_per_obs, bins, weights = np.zeros_like(num_mj_per_obs) + 1./len(num_mj_per_obs))
    plt.savefig('/home/shannon/Paper/Figures/mini_jets_num_per_profile.png')
#    print 'yikes'
#    plt.show()
c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data',  'approved_list_w_errors.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()

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
    
#plot_mini_jet_dist(attree_db)

attree_fp.close()

real_widths = []
for key in c_approved_db.keys():
    clump_list = c_approved_db[key].clump_list
    for clump in clump_list:
        real_widths.append(clump.fit_width_deg)
    
real_widths.sort()
    
#prob = monte_carlo_attree(real_widths)
#print prob

prob2 = monte_carlo_attree_actual_data(real_widths, attree_db, c_approved_db)
print prob2
