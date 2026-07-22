'''
YOU WILL HAVE TO EDIT THE SPLITS BY HAND IN SOME PLACES

'''
import pickle
import os
import sys
import ringutil
import cspice
import clumputil
import numpy as np
from optparse import OptionParser

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
#    cmd_line = []
    cmd_line = ['--write']

parser = OptionParser()
parser.add_option('--write', dest = 'write', action = 'store_true', default = False)
ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_list_w_errors.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()

if options.write:
    file = 'clump_data_table_test.txt'
    clump_table_fp = os.path.join(ringutil.ROOT, file)
    clump_table = open(clump_table_fp, 'w')
    
    clump_table.write('Clump Number,First ID #,Last ID #,Number of Observations,Longitude at Epoch (deg),Minimum - Maximum Lifetime (Days),\
         Relative Mean Motion (deg/day),Semimajor Axis (km),Median Width (deg),Median Brightness (km^2 x 10^4) \n')

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
    chain_list.sort(key=lambda x: x.clump_list[0].g_center * 1000 + x.clump_list[1].g_center)

sorted_id_list = clumputil.get_sorted_obsid_list(c_approved_db)
num_db = {}
for i, obsid in enumerate(sorted_id_list):
    print obsid, i +1
    num_db[obsid] = i +1
        
def print_chain_data(chain, num_id, split = False, parent = False):
    km_per_deg = 881027.02/360.
    max_life = chain.lifetime + chain.lifetime_upper_limit + chain.lifetime_lower_limit
    #    start_date = cspice.et2utc(chain.clump_list[0].clump_db_entry.et, 'C', 0)
    first_id_num = num_db[chain.clump_list[0].clump_db_entry.obsid]
#    print first_id_num, chain.clump_list[0].clump_db_entry.obsid
    last_id_num = num_db[chain.clump_list[-1].clump_db_entry.obsid]
    med_width = np.median(np.array([clump.fit_width_deg for clump in chain.clump_list]))
    abs_width_change = ((chain.clump_list[-1].fit_width_deg - chain.clump_list[0].fit_width_deg) /
                        ((chain.clump_list[-1].clump_db_entry.et - chain.clump_list[0].clump_db_entry.et) / 86400))
    rel_width_change = abs_width_change / chain.clump_list[0].fit_width_deg
    med_bright = np.median(np.array([clump.int_fit_height*km_per_deg for clump in chain.clump_list]))*(1./1e4)
    abs_brt_change = ((chain.clump_list[-1].int_fit_height - chain.clump_list[0].int_fit_height) /
                        ((chain.clump_list[-1].clump_db_entry.et - chain.clump_list[0].clump_db_entry.et) / 86400))
    rel_brt_change = abs_brt_change / chain.clump_list[0].int_fit_height
    
#    print np.array([clump.int_fit_height*km_per_deg for clump in chain.clump_list])*(1./1e4)
    print [clump.g_center for clump in chain.clump_list]
#    REFERENCE_DATE = "1 JANUARY 2007"       
#    REFERENCE_ET = cspice.utc2et(REFERENCE_DATE)
#    dt = chain.clump_list[0].clump_db_entry.et - REFERENCE_ET
#    epoch_long = (chain.clump_list[0].g_center - chain.rate*dt)%360.

    start_long = chain.clump_list[0].g_center
    num_obs = len(chain.clump_list)
    if split:
        num_obs -= 1
        first_id_num = num_db[chain.clump_list[1].clump_db_entry.obsid]
        start_long = chain.clump_list[1].g_center
        str_format = '%s,%i,%i,%1i,%.2f,%.2f - %.2f,%.3f %6.3f,%.1f  %6.1f,%.2f,%.4f,%.2f,%.4f'%(num_id, first_id_num, last_id_num, num_obs,
                                                                                            start_long, chain.lifetime, max_life, chain.rate*86400.,
                                                                                            chain.rate_err*86400., chain.a, chain.a_err, med_width,
                                                                                            rel_width_change, med_bright, rel_brt_change)
    if parent:
        num_obs = 1
        first_id_num = num_db[chain.clump_list[0].clump_db_entry.obsid]
        last_id_num = num_db[chain.clump_list[0].clump_db_entry.obsid]
        start_long = chain.clump_list[0].g_center
        med_width = chain.clump_list[0].fit_width_deg
        med_bright = chain.clump_list[0].int_fit_height*km_per_deg/1e4
        
        str_format = '%s,%i,%i,%1i,%.2f,%s,%s,%s,%.2f,%.2f,%.2f,%.4f'%(num_id, first_id_num, last_id_num, num_obs,
                                                                                                start_long, 'N/A', 'N/A',
                                                                                                'N/A', med_width, 0.,
                                                                                                med_bright, 0.)
    elif (parent == False) and (split == False):
#        print 'C'
        str_format = '%s,%i,%i,%1i,%.2f,%.2f - %.2f,%.3f %6.3f,%.1f  %6.1f,%.2f,%.4f,%.2f,%.4f'%(num_id, first_id_num, last_id_num, num_obs,
                                                                                            start_long, chain.lifetime, max_life, chain.rate*86400.,
                                                                                            chain.rate_err*86400., chain.a, chain.a_err, med_width, rel_width_change,
                                                                                            med_bright, rel_brt_change)
    return str_format

num = 1
for time in sorted(chain_time_db.keys()):
    for a,chain in enumerate(chain_time_db[time]):
        if chain.skip == False:
            parent_clump_start_long = '%6.2f'%(chain.clump_list[0].g_center)
            parent_clump_end_long = '%6.2f'%(chain.clump_list[-1].g_center)
            parent_clump_end_time = chain.clump_list[-1].clump_db_entry.et_max
            num_id = 'C'+str(num)
                
            is_parent = False

            #check to see if this clump is the beginning of a split
            for b, new_chain in enumerate(chain_time_db[time][a+1::]):
                
                new_parent_start_long = '%6.2f'%(new_chain.clump_list[0].g_center)
#                print parent_clump_start_long, chain.clump_list[0].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid
                if new_parent_start_long == parent_clump_start_long:
                    print 'Found a splitting clump', parent_clump_start_long, chain.clump_list[0].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid

                    if not is_parent:
                        is_parent = True
                        parent_str_format = print_chain_data(chain, num_id, parent = True)
                        print parent_str_format
                        if options.write:
                            clump_table.write(parent_str_format+'\n')
                        num_id = num_id + "'"
#                        chain.clump_num = num_id 
                        str_format = print_chain_data(chain, num_id, split = True)
                        print str_format
                        if options.write:
                            clump_table.write(str_format+'\n')
                        
                    num_id = num_id + "'"
#                    chain.clump_num = num_id
                    str_format = print_chain_data(new_chain, num_id, split = True)
                    print str_format
                    if options.write:
                        clump_table.write(str_format+'\n')
                            
                    #skip this clump so that it isn't put in the table a second time
                    new_chain.skip = True
                    
            if not is_parent:
#                chain.clump_num = num_id
                str_format = print_chain_data(chain, num_id)        
                print str_format
                if options.write:
                    clump_table.write(str_format+'\n')
    
            #check to see if parent chain split at the end
            c = 0
            for new_chain in chain_time_db[parent_clump_end_time]:
                new_parent_start_long = '%6.2f'%(new_chain.clump_list[0].g_center)
                
                new_second_start_long = '%6.2f'%(new_chain.clump_list[1].g_center)
                
                if new_parent_start_long == parent_clump_end_long:
                    print 'Parent clump has split', parent_clump_end_long, chain.clump_list[-1].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid
                   
                    new_num_id = num_id + "'"*(c+1)
#                    print new_num_id
                    str_format = print_chain_data(new_chain, new_num_id, split = True)
                    print str_format
                    if options.write:
                        clump_table.write(str_format+'\n')
#                  
                    #delete the chain so that it isn't put in the table a second time
                    new_chain.skip = True
                    c +=1
            num +=1
            
if options.write:        
    clump_table.close()
