'''
Author: Shannon Hicks
"Upper Lifetime" refers to the number of days after the LAST observation that the clump might have been found
"Lower Lifetime" refers to the number of days before the clump was discovered and that it might have appeared

Maximum total lifetime = the amount of time spanning from the lower limit to the upper limit
'''

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


cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = [
                '--replace_list'
                ]
#    cmd_line = ['--side-analysis', '--plot-fit']
#    cmd_line = ['--write-pdf']

parser = OptionParser()

parser.add_option('--replace_list', dest = 'replace_list', action = 'store_true', default = False)

ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

def calc_lifetime(clump_list):
    
    day1 = clump_list[0].clump_db_entry.et_max
    day2 = clump_list[-1].clump_db_entry.et_max
    lifetime = (day2 - day1)/86400.     #seconds to days
    
    return lifetime

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


def lifetimes_histogram(c_approved_list):
    lifetimes = []
    clump_list_size = []
    for chain in c_approved_list:
        num_clumps = len(chain.clump_list)
        lifetimes.append(chain.lifetime)
        clump_list_size.append(num_clumps)
    print np.mean(lifetimes), np.max(lifetimes), np.min(lifetimes)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.arange(0., 10., 1)
    plt.hist(clump_list_size, bins)
    plt.xlabel('Clump Chain Length')
    plt.ylabel('Number of Chains')
#    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    step = 5.
    bins = np.arange(np.min(lifetimes), np.max(lifetimes)+step, step)
    plt.hist(lifetimes, bins)
    plt.xlabel('Clump Lifetime (Days)')
    plt.ylabel('Number of Clumps')
    plt.show()
    

def upper_limit_lifetimes(c_approved_list, clump_full_db):
    #finds the next obsid with the correct coverage
    #places an upper limit on the lifetime of the clumps
    max_ets = []
    db_by_time = {}
    upper_lifetimes = []
    for obs in clump_full_db.keys():
#        
        max_et = clump_db[obs].et_max
#        print max_et
        db_by_time[max_et] = obs
    sorted_obs = []
    for et in sorted(db_by_time.keys()):
        sorted_obs.append(db_by_time[et])
    sorted_obs = np.array(sorted_obs)

    for chain in c_approved_list:
        first_et = chain.clump_list[0].clump_db_entry.et_max
        last_et = chain.clump_list[-1].clump_db_entry.et_max
        rate = chain.rate
        last_obs = chain.clump_list[-1].clump_db_entry.obsid
        last_longitude = chain.clump_list[-1].g_center
        
        
        start_index = np.where(sorted_obs == last_obs)[0][0] +1
        remaining_obs = sorted_obs[start_index::]
        for obs in remaining_obs:
            new_et = clump_full_db[obs].et_max
            dt = new_et - last_et
            new_long = last_longitude + rate*dt
#            print new_long
            new_long= new_long%360.
            #look for masking at the new longitude and +/- 0.5 degrees to either side
            long_res = options.longitude_resolution*options.mosaic_reduction_factor
            new_long_indices = np.clip(np.arange(int(new_long/long_res - 25), int(new_long/long_res + 25)),
                                       0, (360/long_res)-1) #ignoring running off the sides
            new_long_indices = new_long_indices.astype(int)
#            print new_long_indices
            ew_data = clump_full_db[obs].ew_data
#            print ew_data.mask
            if ma.any(ew_data.mask[new_long_indices]) == True:
                #values are masked - continue on
                continue
            else:
                upper_limit_lifetime = new_et - last_et
                chain.lifetime_upper_limit = upper_limit_lifetime/86400.
                upper_lifetimes.append(upper_limit_lifetime/86400.)
#                print 'START ID %20s, START LONG %6.2f, LIFE %6.2f, FINAL ID %20s, FINAL LONG %6.2f'%(last_obs, last_longitude, upper_limit_lifetime/86400., obs, new_long)
                break
#    print 'MEAN UPPER LIFETIME: ', np.mean(upper_lifetimes)
    return np.array(upper_lifetimes)

def lower_lifetimes(c_approved_list, clump_full_db):
    max_ets = []
    db_by_time = {}
    lower_lifetimes = []
    for obs in clump_full_db.keys():
#        
        max_et = clump_db[obs].et_max
#        print max_et
        db_by_time[max_et] = obs
    sorted_obs = []
    for et in sorted(db_by_time.keys()):
        sorted_obs.append(db_by_time[et])
    sorted_obs = np.array(sorted_obs)

    for chain in c_approved_list:
        first_et = chain.clump_list[0].clump_db_entry.et_max
        first_obs = chain.clump_list[0].clump_db_entry.obsid
        last_et_in_chain = chain.clump_list[-1].clump_db_entry.et_max
        rate = chain.rate
        last_obs_in_chain = chain.clump_list[-1].clump_db_entry.obsid
        first_longitude = chain.clump_list[0].g_center
        
        
        start_index = np.where(sorted_obs == first_obs)[0][0]
        remaining_obs = sorted_obs[0:start_index][::-1]             #go from the previous obsid and backwards
        for obs in remaining_obs:
            new_et = clump_full_db[obs].et_max
            dt = first_et - new_et
            new_long = first_longitude - rate*dt
#            print new_long
            new_long= new_long%360.
            #look for masking at the new longitude and +/- 0.5 degrees to either side
            long_res = options.longitude_resolution*options.mosaic_reduction_factor
            new_long_indices = np.clip(np.arange(int(new_long/long_res - 25), int(new_long/long_res + 25)),
                                       0, (360/long_res)-1) #ignoring running off the sides
            new_long_indices = new_long_indices.astype(int)
#            print new_long_indices
            ew_data = clump_full_db[obs].ew_data
#            print ew_data.mask
            if ma.any(ew_data.mask[new_long_indices]) == True:
                #values are masked - continue on
                continue
            else:
                lower_limit_lifetime = first_et - new_et                #the earliest number of days before the first observation that the clump could have appeared.
                chain.lifetime_lower_limit = lower_limit_lifetime/86400.    #time in days
                lower_lifetimes.append(lower_limit_lifetime/86400.)
#                print 'START ID %20s, START LONG %6.2f, LIFE %6.2f, FINAL ID %20s, FINAL LONG %6.2f'%(first_obs, first_longitude, lower_limit_lifetime/86400., obs, new_long)
                break
#    print 'MEAN LOWER LIFETIME: ', np.mean(lower_lifetimes)
    return np.array(lower_lifetimes)

def lifetimes_hist(lifetimes, upper_lifetimes, lower_lifetimes, max_lifetimes):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.arange(np.min(upper_lifetimes), np.max(upper_lifetimes) + 10., 10.)
    plt.hist(upper_lifetimes, bins)
    plt.xlabel('Upper Lifetimes (Days)')
    plt.ylabel('Number of Clump Chains')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.arange(np.min(lower_lifetimes), np.max(lower_lifetimes) +10., 10.)
    plt.hist(lower_lifetimes, bins)
    plt.xlabel('Lower Lifetimes (Days)')
    plt.ylabel('Number of Clump Chains')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.arange(np.min(max_lifetimes), np.max(max_lifetimes) + 10., 10.)
    plt.hist(max_lifetimes, bins)
    plt.xlabel('Maximum Possible Lifetimes (Days)')
    plt.ylabel('Number of Clump Chains')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.arange(np.min(lifetimes), np.max(lifetimes) + 2., 2.)
    plt.hist(lifetimes, bins)
    plt.xlabel('Original Lifetimes (Days)')
    plt.ylabel('Number of Clump Chains')
    
    plt.show()
    
def clump_birthrate(c_approved_list, clump_db):
    
    #measures how many NEW clumps are found per day
    #use max et as the key so we can more easily calculate the amount of time passed
    birth_db = {}
    for chain in c_approved_list:
        start_clump = chain.clump_list[0]
        start_et = start_clump.clump_db_entry.et_max
        if start_et not in birth_db.keys():
            birth_db[start_et] = []
            birth_db[start_et].append(start_clump)
        else:
            birth_db[start_et].append(start_clump)
    
    clump_nums = []
    #fill the birth db with the other data sets
    for obsid in clump_db.keys():
        et = clump_db[obsid].et_max
        if et not in birth_db.keys():
            birth_db[et] = []
        
    #need a database ordered by time
    et_list_1 = sorted(birth_db.keys())
    clump_nums = [len(birth_db[et]) for et in sorted(birth_db.keys())]
    
    et_list = []
    rates = []
    i=0
    for et in sorted(birth_db.keys()):
        print i
        if len(birth_db[et]) == 0:
            et_list.append(et)

        if len(birth_db[et]) != 0:
            et_list.append(et)
            et_list = np.array(et_list)
            et_list -= et_list[0]
#            print et_list
            num_clumps = len(birth_db[et])
#            print num_clumps
            rate = num_clumps/(et_list[-1]/86400.)
            print rate, num_clumps, et_list[-1]/86400.
            rates.append(rate)
            
            et_list = [et]
        i += 1
    print rates
    print np.mean(rates)
    
    average_clump_rate = np.mean(rates)
    num_ids = len(birth_db.keys())
#    print clump_nums
#    print birth_db.keys()
    print ' AVG NUMBER OF NEW CLUMPS PER DAY: ',average_clump_rate, np.std(rates)
    
    plt.plot(np.array(et_list_1)/86400., clump_nums, marker = '.', ls = '')
    plt.show()
    
def clump_lifetimes_plots(c_approved_list, clump_db):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    lives_list = []
    starting_ets = []
    
    for chain in c_approved_list:
        lives_list.append(chain.lifetime)
        starting_et = chain.clump_list[0].clump_db_entry.et_max
        starting_ets.append(starting_et/86400.)
    
        
    lower_lifetime_list = lower_lifetimes(c_approved_list, clump_db)
    lower_lifetime_list = np.array(lower_lifetime_list)
    upper_lifetimes = upper_limit_lifetimes(c_approved_list, clump_db)
    upper_lifetimes = np.array(upper_lifetimes)
    dl = lower_lifetime_list + upper_lifetimes                          #where dl = maximum possible lifetime

    plt.bar(starting_ets, lives_list, width = 0.5, color = 'red' , edgecolor = 'black')
    plt.bar(starting_ets, dl, bottom = lives_list, width = 0.5, color = 'blue', edgecolor = 'black')
    plt.show()
#use the approved list of clumps for analysis - full resolution set
#gives us the modified clump database with all of the clumps we consider valid

c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_clumps_list.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
clump_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()

#clump_db_path, clump_chains_path = ringutil.clumpdb_paths(options)
#clump_db_fp = open(clump_db_path, 'rb')
#clump_find_options = pickle.load(clump_db_fp)
#clump_full_db = pickle.load(clump_db_fp)
#clump_db_fp.close()

#c_approved_db = list_to_db(clump_db, c_approved_list)

#remove the parent clumps from the split clumps. Now all of the chains will have the correct rates.
c_approved_list = clumputil.remove_parent_clumps(c_approved_list)
for chain in c_approved_list:
    chain.skip = False
    
lives_list = []
for i, chain in enumerate(c_approved_list):
    if chain.skip == False:
        split_chains = clumputil.check_for_split(chain, c_approved_list, i)
        if len(split_chains) > 0:
            print 'a'
            split_chains.append(chain)              #now have a list of all chains that split
            second_clumps = [chain.clump_list[1] for chain in split_chains]
            second_centers = ['%6.2f'%(clump.g_center) for clump in second_clumps]
#                print second_centers
            for s_chain in split_chains:
#                    print len(s_chain.clump_list)
#                    print second_centers.count('%6.2f'%(s_chain.clump_list[1].g_center))
                if (second_centers.count('%6.2f'%(s_chain.clump_list[1].g_center)) > 1):    #only happens in the "triple split" case
                    start = 2
                if len(s_chain.clump_list) == 2:
                    start = 0 
                elif (len(s_chain.clump_list) > 2) and (second_centers.count('%6.2f'%(s_chain.clump_list[1].g_center)) <= 1):
                    start = 1
                
                lifetime = calc_lifetime(s_chain.clump_list[start::])
                lives_list.append(lifetime)
                s_chain.lifetime = lifetime
                print s_chain.clump_list[start].clump_db_entry.obsid, s_chain.clump_list[start].g_center, lifetime

        else: 
            print 'b'
            start = 0
            
            lifetime = calc_lifetime(chain.clump_list[start::])
            lives_list.append(lifetime)
            chain.lifetime = lifetime
            print chain.clump_list[0].clump_db_entry.obsid, chain.clump_list[0].g_center, lifetime

lives_list = np.array(lives_list)
#    
lower_lifetimes = lower_lifetimes(c_approved_list, clump_db)
upper_lifetimes = upper_limit_lifetimes(c_approved_list, clump_db)

#
#for chain in c_approved_list:
#    print chain.lifetime, chain.lifetime_lower_limit, chain.lifetime_upper_limit

#lifetimes_histogram(c_approved_list)
    
if options.replace_list:
    list_w_lives_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_list_w_lives.pickle')
    list_w_lives_fp = open(list_w_lives_fp, 'wb')
    pickle.dump((clump_db, c_approved_list), list_w_lives_fp)
    list_w_lives_fp.close()

#max_lifetimes = lower_lifetimes + lives_list + upper_lifetimes
#lifetimes_hist(lives_list, upper_lifetimes, lower_lifetimes, max_lifetimes)

#clump_lifetimes_plots(c_approved_list, clump_db)
#clump_birthrate(c_approved_list, clump_db)
