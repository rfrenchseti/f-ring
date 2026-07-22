# Take Shannon's original approved chain list, delete the chains we don't actually like,
# add clumps to some of the chains we do like, and redo all the clumps to match the current
# clump database.

import ringutil
import clumputil
import cspice
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pickle
import os
from optparse import OptionParser
import sys
import clump_gaussian_fit

cmd_line = sys.argv[1:]
if len(cmd_line) == 0:
    
    cmd_line = [
                '-a',
                '--replace-list',
                ]
    
parser = OptionParser()

parser.add_option('--replace-list', dest = 'replace_list', action = 'store_true', default = False)
parser.add_option('--replace-databases', dest = 'replace_databases', action = 'store_true', default = False)    

ringutil.add_parser_options(parser)
options, args = parser.parse_args(cmd_line)


voyager_clump_db_path = os.path.join(ringutil.ROOT, 'clump-data', 'voyager_clumpdb_137500_142500_05.000_0.020_10_02_137500_142500.pickle')
v_clump_db_fp = open(voyager_clump_db_path, 'rb')
clump_find_options = pickle.load(v_clump_db_fp)
v_all_clump_db = pickle.load(v_clump_db_fp)
v_clump_db_fp.close()
for v_obs in v_all_clump_db.keys(): # Fix masking
    v_all_clump_db[v_obs].ew_data[np.where(v_all_clump_db[v_obs].ew_data == 0.)] = ma.masked

cassini_clump_db_path = os.path.join(ringutil.ROOT, 'clump-data', 'clumpdb_137500_142500_05.000_0.020_10_02_137500_142500.pickle')
c_clump_db_fp = open(cassini_clump_db_path, 'rb')
clump_find_options = pickle.load(c_clump_db_fp)
c_all_clump_db = pickle.load(c_clump_db_fp)
c_clump_db_fp.close()

c_approved_list_fn = os.path.join(ringutil.ROOT, 'clump-data', 'shannon_approved_list_w_errors.pickle')
c_approved_list_fp = open(c_approved_list_fn, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()

sorted_obsid_list = clumputil.get_sorted_obsid_list(c_all_clump_db)

#for obsid in sorted(c_all_clump_db.keys()):
##    if obsid != 'ISS_036RF_FMOVIE001_VIMS': continue
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    plt.title(obsid)
#    dbentry1 = c_all_clump_db[obsid]
#    dbentry2 = c_approved_db[obsid]
#    long_res1 = 360./len(dbentry1.ew_data)
#    longitudes1 = np.arange(0, 360., long_res1)
#    long_res2 = 360./len(dbentry2.ew_data)
#    longitudes2 = np.arange(0, 360., long_res2)
#
#    plt.plot(longitudes1, dbentry1.ew_data, '-', color='#b0b0b0')
#    plt.plot(longitudes2, dbentry2.ew_data, '-', color='#b0b0b0')
#    for n, clump in enumerate(dbentry1.clump_list):
#        clumputil.plot_fitted_clump_on_ew(ax, dbentry1.ew_data, clump, color=colors[n%len(colors)])
#        ratio = (clump.fit_right_deg-clump.g_center) / (clump.g_center-clump.fit_left_deg)
#        if ratio < 1: ratio = 1/ratio
#        if ratio > 4: 
#            print '%6.2f %6.2f %6.2f' % (clump.g_center, clump.fit_left_deg, clump.fit_right_deg)
#    for n, clump in enumerate(dbentry2.clump_list):
#        clumputil.plot_fitted_clump_on_ew(ax, dbentry2.ew_data, clump, color=colors[n%len(colors)])
#    plt.show()

new_approved_list = []

# Splits are 0/1, 89/90, 80/81, 85/86, 87/88
bad_chain_num_list = [0,1,96,8,89,90,17,83,84,61,80,81,66,70,71,26,33,46,47,85,86,64,87,88,32,44,49,72,7,5,10,51,77,75,68,
                      11,12,18,19,36,58,59,69]
remove_last_clump_list = [39,56,34]
add_prior_clump = [93,56,74]
add_successor_clump = [2,4,6,15,55,91]

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
                   
#    [('ISS_036RF_FMOVIE001_VIMS', 190, 244),
#                ('ISS_036RF_FMOVIE002_VIMS', 190, 248)]

for chain_num, chain in enumerate(c_approved_list):
    print 'CHAIN', chain_num

    if chain_num in bad_chain_num_list:
        print 'SKIPPING AS BAD CHAIN'
        continue
    
    chain_is_ok = True
    
    if chain_num in remove_last_clump_list: # Remove last clump, which is bad
        chain.clump_list = chain.clump_list[:-1]
        print 'SKIPPING LAST CLUMP'
    
    if chain_num in add_prior_clump:
        clump = chain.clump_list[0]
        idx = sorted_obsid_list.index(chain.clump_list[0].clump_db_entry.obsid)
        prior_obsid = sorted_obsid_list[idx-1]
        new_clump_list = c_all_clump_db[prior_obsid].clump_list
        best_dist = 1e38
        best_clump = None
        print 'MATCHING PRIOR %7.2f %7.2f' % (clump.g_center, clump.scale)
        for new_clump in new_clump_list:
            dist = np.sqrt((new_clump.g_center-clump.g_center)**2+
                           (new_clump.scale-clump.scale)**2)
            print '%7.2f %7.2f %7.3f' % (new_clump.g_center, new_clump.scale, dist)
            if dist < best_dist:
                best_dist = dist
                best_clump = new_clump
        if best_dist > 10:
            print 'PRIOR NO MATCHING CLUMP', prior_obsid, clump.g_center, clump.scale
            chain_is_ok = False
            break
        chain.clump_list = [best_clump] + chain.clump_list
        
    if chain_num in add_successor_clump:
        clump = chain.clump_list[-1]
        idx = sorted_obsid_list.index(chain.clump_list[-1].clump_db_entry.obsid)
        successor_obsid = sorted_obsid_list[idx+1]
        new_clump_list = c_all_clump_db[successor_obsid].clump_list
        best_dist = 1e38
        best_clump = None
        print 'MATCHING SUCCESSOR %7.2f %7.2f' % (clump.g_center, clump.scale)
        for new_clump in new_clump_list:
            dist = np.sqrt((new_clump.g_center-clump.g_center)**2+
                           (new_clump.scale-clump.scale)**2)
            print '%7.2f %7.2f %7.3f' % (new_clump.g_center, new_clump.scale, dist)
            if dist < best_dist:
                best_dist = dist
                best_clump = new_clump
        if best_dist > 10:
            print 'PRIOR NO MATCHING CLUMP', successor_obsid, clump.g_center, clump.scale
            chain_is_ok = False
            break
        chain.clump_list = chain.clump_list + [best_clump]
        
    for clump_idx in range(len(chain.clump_list)):
        clump = chain.clump_list[clump_idx]
        obsid = clump.clump_db_entry.obsid
        clump.clump_db_entry = c_all_clump_db[obsid]
        new_clump_list = c_all_clump_db[obsid].clump_list
        best_dist = 1e38
        best_clump = None
        for new_clump in new_clump_list:
            dist = np.sqrt(9*(new_clump.g_center-clump.g_center)**2+
                           (new_clump.scale-clump.scale)**2)
            if dist < best_dist:
                best_dist = dist
                best_clump = new_clump
        if best_dist > 10:
            print 'NO MATCHING CLUMP', obsid
            chain_is_ok = False
            break
#        print 'OK!'
        print '%-30s %6.2f %6.2f / %6.2f %6.2f / %6.2f %6.2f = %7.2f' % (obsid,
                                                                         clump.g_center, best_clump.g_center,
                                                                         clump.fit_left_deg, best_clump.fit_left_deg,
                                                                         clump.fit_right_deg, best_clump.fit_right_deg,
                                                                         best_dist)

        chain.clump_list[clump_idx] = best_clump

    if chain_is_ok:
        new_approved_list.append(chain)
        print 'NEW CHAIN NUM', len(new_approved_list)

    print

for new_clump_info_list in fake_new_clumps:
    print 'FAKING CLUMP'
    clump_list = []
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
        clump.wave_type = 'fake'
        clump.fit_left_deg, clump.fit_right_deg, clump.fit_width_idx, clump.fit_width_deg, clump.fit_height, clump.int_fit_height, clump.g_center, clump.g_sigma, clump.g_base, clump.g_height = clump_gaussian_fit.refine_fit(clump, clump_db_entry.ew_data, False, False)
        print clump_obsid, clump_start_long, clump_end_long,
        print 'FIT LEFT', clump.fit_left_deg, 'RIGHT', clump.fit_right_deg
        clump_list.append(clump)
    chain = clumputil.ClumpChainData()
    chain.clump_list = clump_list
    new_approved_list.append(chain)
    rate, ctr_long, long_err_list = clumputil.fit_rate(chain.clump_list)
    chain.rate = rate
    chain.base_long = ctr_long
    chain.long_err_list = long_err_list
    chain.a = ringutil.RelativeRateToSemimajorAxis(rate)
    print
    
if options.replace_list:
    
    print 'saving Cassini List'
    list_w_errors_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_clumps_list.pickle')
    list_w_errors_fp = open(list_w_errors_fp, 'wb')
    pickle.dump((c_all_clump_db, new_approved_list), list_w_errors_fp)
    list_w_errors_fp.close()

    
#if options.replace_databases:
#    
#    print 'Saving Cassini Database'
#    #cassini ---------------------------------
#    clump_db_path, clump_chains_path = ringutil.clumpdb_paths(options)
#    print clump_db_path
#    clump_db_fp = open(clump_db_path, 'rb')
#    clump_find_options = pickle.load(clump_db_fp)
#    clump_db = pickle.load(clump_db_fp)
#    clump_db_fp.close()
#    
#    clump_database_path, clump_chains_path = ringutil.clumpdb_paths(options)
#    print clump_db_path
#    clump_database_fp = open(clump_database_path, 'wb')
#    
#    pickle.dump(clump_find_options, clump_database_fp)
#    pickle.dump(c_approved_db, clump_database_fp)
#    clump_database_fp.close() 
#    