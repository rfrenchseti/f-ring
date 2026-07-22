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



c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_list_w_errors.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()

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
    chain_list.sort(key=lambda x: x.clump_list[0].g_center)
    
sorted_id_list = clumputil.get_sorted_obsid_list(c_approved_db)

num = 1
for time in sorted(chain_time_db.keys()):
    for chain_num, chain in enumerate(chain_time_db[time]):
        print 'CHAIN', num,
        print 'MM %7.3f +/- %7.3f  A %9.2f +/- %7.3f  LT %6.2f L %6.2f U %6.2f' % (chain.rate*86400, chain.rate_err*86400,
                                                                                   chain.a, chain.a_err, chain.lifetime,
                                                                                   chain.lifetime_lower_limit,
                                                                                   chain.lifetime_upper_limit)
        num += 1
        print '=' * 90
        for clump_num, clump in enumerate(chain.clump_list):
            print 'CLUMP', clump_num,
            print '%-26s %15s' % (clump.clump_db_entry.obsid[4:], cspice.et2utc(clump.clump_db_entry.et_max, 'C', 0))[:11],
            print 'GL %7.3f S %6.3f B %6.4f H %8.6f' % (clump.g_center, clump.g_sigma,
                                                        clump.g_base, clump.g_height),
            print 'FL %7.3f FR %7.3f FW %7.3f' % (clump.fit_left_deg, clump.fit_right_deg, clump.fit_right_deg-clump.fit_left_deg),
            print 'INTH %7.3f' % (clump.int_fit_height)
            
        print
        
            
