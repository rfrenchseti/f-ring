import clumputil
import ringutil
import numpy 
import pickle
import os

c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_clumps_list.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()
                        

#remove the last clump from chain 24 - it shouldn't be there
#print c_approved_list[24].clump_list
#del c_approved_list[24].clump_list[-1]
#
#print c_approved_list[24].clump_list

#remove chain number 10 entire
print len(c_approved_list)
i = 0
for chain in c_approved_list:
    print i
    print chain.clump_list[0].clump_db_entry.obsid, chain.clump_list[0].g_center, chain.clump_list[-1].clump_db_entry.obsid, chain.clump_list[-1].g_center
    i +=1
#
del c_approved_list[63]

###
#i = 0
#for chain in c_approved_list:
#    print i
#    print chain.clump_list[0].clump_db_entry.obsid, chain.clump_list[0].g_center
#    i +=1
##    
#del c_approved_list[92:94]
#
#i = 0
#for chain in c_approved_list:
#    print i
#    print chain.clump_list[0].clump_db_entry.obsid, chain.clump_list[0].g_center
#    i +=1
##    
#del c_approved_list[80:82]
#
#
#does the right thing - re-save the list
c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_clumps_list.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'wb')
pickle.dump((c_approved_db, c_approved_list),c_approved_list_fp)
c_approved_list_fp.close()
