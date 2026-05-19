from optparse import OptionParser
'''
A program that looks at every profile and checks to see if there is:
 1) less than 40 days before the observation
 2) The union of the 3 observations afterwards have enough coverage
 
 If 1 and 2 are passed then we use the clumps in this observation to count the number of clumps/ID 
'''
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


def get_good_obsids(clump_db):
    
    sorted_id_list = clumputil.get_sorted_obsid_list(clump_db)
    print 'total ids', len(sorted_id_list)
    a=0
    good_observation_ids = []
    while a < len(sorted_id_list)-2:
        current_count = ma.count(clump_db[sorted_id_list[a]].ew_data)
        percent_coverage = (current_count/float(len(clump_db[sorted_id_list[a]].ew_data)))*100
        print sorted_id_list[a], percent_coverage
        if percent_coverage > 80:
            compare_coverage_list = []
            time_between_ids_check = False
            union_coverage_check = False
            current_time = clump_db[sorted_id_list[a]].et_max
            next_id_time1 = clump_db[sorted_id_list[a+1]].et_max
            
            time_diff = (next_id_time1 - current_time)/86400.
            
            if time_diff < 40.:
                time_between_ids_check = True
                compare_coverage_list.append(sorted_id_list[a+1])
                #there is no point in comparing the coverage of the next ids if they are also not close to the previous observation
                
                next_id_time2 = clump_db[sorted_id_list[a+2]].et_max
                time_diff1 = (next_id_time2 - current_time)/86400.
                if time_diff1 < 40.: 
                    compare_coverage_list.append(sorted_id_list[a+2])
                
                    next_id_time3 = clump_db[sorted_id_list[a+3]].et_max
                    time_diff2 = (next_id_time3 - current_time)/86400.
                    if time_diff2 < 40.:
                        compare_coverage_list.append(sorted_id_list[a+3])
                
                #now we must union the masks of the next obsids. Ideally we will have 3 available.
                union_arr = np.zeros([len(clump_db[compare_coverage_list[0]].ew_data)]) + 5
                union_arr = union_arr.view(ma.MaskedArray)
                union_arr.mask = ma.getmaskarray(union_arr)
                union_arr.mask[::] = True
#                print ma.count(union_arr)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(union_arr)
                k = 5            
                for obs in compare_coverage_list:
                    k += 5
                    union_arr.data[::] = k
                    ew_mask = clump_db[obs].ew_data.mask
                    
                    union_arr.mask = union_arr.mask & ew_mask
#                    masked_indices = ma.where(ew_mask == True)[0]
#                    print obs, len(masked_indices)
#                    
#                    if len(masked_indices) != 0.:
#                        for n in masked_indices:
#                            if union_arr.mask[n] == False:
#                                union_arr[n] = ma.masked
                    
                    plt.plot(union_arr)
                ax.set_ylim(0, 25)
#                plt.show()
                num_not_masked = ma.count(union_arr)
                percentage_not_masked = (float(num_not_masked)/float(len(union_arr.mask)))*100
        
#                print percentage_not_masked
                if percentage_not_masked > 80.:
                    union_coverage_check = True
                    
        if (union_coverage_check== True) and (time_between_ids_check == True) and (percent_coverage > 80):
            print 'yes'
            good_observation_ids.append(sorted_id_list[a])
            
                
#        print a
        a += 1


    print good_observation_ids
    print len(good_observation_ids)



c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_list_w_errors.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()

get_good_obsids(c_approved_db)

