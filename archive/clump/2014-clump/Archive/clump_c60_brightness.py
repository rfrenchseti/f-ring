import pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import clumputil
import ringutil
import cspice
import os


c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_list_w_errors.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
clump_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()


long_start = 190
long_end = 250
clump_ids = [
             'ISS_079RI_FMONITOR002_PRIME',
             'ISS_079RF_FRINGMRLF002_PRIME',
             'ISS_081RI_FMOVIE106_VIMS',
             'ISS_082RI_FMONITOR003_PRIME',
             'ISS_083RI_FMOVIE109_VIMS',
             'ISS_085RF_FMOVIE003_PRIME_2'
             ]

for obs in clump_ids:
    
    ew_data = clump_db[obs].ew_data
    long_res = 360./len(ew_data)
    ew_clip = ew_data[long_start/long_res: long_end/long_res + 1]
#    print len(ew_clip)
#    base = (ew_data[long_start/long_res] + ew_data[long_end/long_res])/2.
    base = ma.min(ew_clip)
    clump_int_height = ma.sum(ew_clip - base)*long_res
    
    counts = ma.count(ew_clip)
#    print counts*long_res
    percent_coverage = float(counts)/float(len(ew_clip))
    clump_int_height /= percent_coverage
    print obs, clump_int_height