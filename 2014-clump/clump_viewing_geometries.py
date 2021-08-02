import pickle
import os
import numpy
import matplotlib.pyplot as plt
import ringutil
import clumputil
import cspice


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
        
sorted_id_list = clumputil.get_sorted_obsid_list(c_approved_db)

def print_chain_data(chain, num_id, split = False, parent = False):
    km_per_deg = 881027.02/360.
    print 'NUM ID: ', num_id
    for clump in chain.clump_list:
        
        print 'EMISSION: %6.2f  |  PHASE: %6.2f  |  INCIDENCE %6.2f  |'%(clump.clump_db_entry.emission_angle, clump.clump_db_entry.phase_angle, clump.clump_db_entry.incidence_angle)
        str_format = ''
    return str_format






num = 1
for time in sorted(chain_time_db.keys()):
    for a,chain in enumerate(chain_time_db[time]):
        if chain.skip == False:
            parent_clump_start_long = '%6.2f'%(chain.clump_list[0].g_center)
            parent_clump_end_long = '%6.2f'%(chain.clump_list[-1].g_center)
            parent_clump_end_time = chain.clump_list[-1].clump_db_entry.et_max
            num_id = 'C'+str(num)
                
            str_format = print_chain_data(chain, num_id)        
            print str_format
            
            #check to see if this clump is the beginning of a split
            for b, new_chain in enumerate(chain_time_db[time][a+1::]):
                
                new_parent_start_long = '%6.2f'%(new_chain.clump_list[0].g_center)
#                print parent_clump_start_long, chain.clump_list[0].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid
                if new_parent_start_long == parent_clump_start_long:
                    print 'Found a splitting clump', parent_clump_start_long, chain.clump_list[0].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid
                    
                    parent_str_format = print_chain_data(new_chain, num_id, parent = True)
                    print parent_str_format
#                    if options.write:
#                        clump_table.write(parent_str_format)
                        
                    num_id = num_id + "'"
                    new_num_id = num_id+"'"
                    str_format = print_chain_data(new_chain, new_num_id, split = True)
                    print str_format
#                    if options.write:
#                        clump_table.write(str_format)
                            
                    #skip this clump so that it isn't put in the table a second time
                    new_chain.skip = True
    
            #check to see if parent chain split at the end
            c = 0
            for new_chain in chain_time_db[parent_clump_end_time]:
                new_parent_start_long = '%6.2f'%(new_chain.clump_list[0].g_center)
                
                if new_parent_start_long == parent_clump_end_long:
                    print 'Parent clump has split', parent_clump_end_long, chain.clump_list[-1].clump_db_entry.obsid, new_parent_start_long, new_chain.clump_list[0].clump_db_entry.obsid
                   
                    new_num_id = num_id + "'"*(c+1)
#                    print new_num_id
                    str_format = print_chain_data(new_chain, new_num_id, split = True)
                    print str_format
#                    if options.write:
#                        clump_table.write(str_format)
#                  
                    #delete the chain so that it isn't put in the table a second time
                    new_chain.skip = True
                    c +=1
            num +=1
















