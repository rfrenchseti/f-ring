import pickle
import numpy as np
import os
import ringutil
import clumputil


km_per_deg = 881027.02/360.

def find_clump_width_height_slopes(c_approved_list):
    width_slopes = []
    height_slopes = []
    for chain in c_approved_list:
        widths = [clump.fit_width_deg for clump in chain.clump_list]
        times = [clump.clump_db_entry.et/86400. for clump in chain.clump_list]
        heights = [clump.int_fit_height*km_per_deg for clump in chain.clump_list]
        
        w_a, wb = np.polyfit(times, widths, 1)
        width_slopes.append(w_a)

        h_a, hb = np.polyfit(times, heights, 1)
        height_slopes.append(h_a)

    return (np.array(width_slopes), np.array(height_slopes))

def get_attribute_arrays(c_approved_list):
    
    start_widths = np.array([chain.clump_list[0].fit_width_deg for chain in c_approved_list])
    start_height = np.array([chain.clump_list[0].int_fit_height*km_per_deg for chain in c_approved_list])
    
    end_width = np.array([chain.clump_list[-1].fit_width_deg for chain in c_approved_list])
    end_height = np.array([chain.clump_list[-1].int_fit_height*km_per_deg for chain in c_approved_list])
    
    sma = [chain.a for chain in c_approved_list]
    mm = [chain.rate for chain in c_approved_list]
    
    med_width = []
    med_height = []
    for chain in c_approved_list:
        
        w = np.array([clump.fit_width_deg for clump in chain.clump_list])
        h = np.array([clump.int_fit_height*km_per_deg for clump in chain.clump_list])
        
        med_width.append(np.median(w))
        med_height.append(np.median(h)) 
    
    return (start_widths, start_height, end_width, end_height, med_width, med_height, sma, mm)

def correlate_attributes(atts):

    att_names = ('start_w', 'start_h', 'end_w', 'end_h', 'med_w', 'med_h', 'sma', 'mm', 'w_slopes', 'h_slopes')
    
    print '                 | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | '%(att_names)
    
    for i, a in enumerate(atts):
        corrs = []
        for b in atts:
            
            res = np.corrcoef(a, b)[0][1]
            corrs.append(res)
        
#        print corrs
        print ' %15s |   %6.3f  |   %6.3f  |   %6.3f  |   %6.3f  |   %6.3f  |   %6.3f  |   %6.3f  |   %6.3f  |   %6.3f  |   %6.3f  |'%(att_names[i], corrs[0], corrs[1], corrs[2], corrs[3],
                                                                                                         corrs[4], corrs[5], corrs[6], corrs[7], corrs[8], corrs[9] ) 
        corrs = []
     
c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data',  'approved_list_w_errors.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()

start_widths, start_height, end_width, end_height, med_width, med_height, sma, mm = get_attribute_arrays(c_approved_list)
width_slopes, height_slopes = find_clump_width_height_slopes(c_approved_list) 

atts = (start_widths, start_height, end_width, end_height, med_width, med_height, sma, mm, width_slopes, height_slopes)

correlate_attributes(atts)
