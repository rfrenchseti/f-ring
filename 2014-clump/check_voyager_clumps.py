import matplotlib.pyplot as plt
import clumputil
import numpy as np
import numpy.ma as ma
import pickle
import os
import ringutil
import sys
from optparse import OptionParser

cmd_line = sys.argv[1:]
if len(cmd_line) == 0:
    
    cmd_line = ['-a']
    
parser = OptionParser()    
ringutil.add_parser_options(parser)
options, args = parser.parse_args(cmd_line)

voyager_clumps = [
    (13.4, '1A'),
    (332.2, '1B'),
    (136.1, '1C'),
    (153.24, '1D'),
    (20.7, '1E'),
    (26.5, '1F'),
    (3.4, '1G'),
    (341.6, '1H'),
    (335.6, '1G'),
    (61.9, '1J'),
    (262.1, '1K'),
    (329.3, '1L'),
    (331.9, "1L'"),
    (168.5, '1M'),
    (176.9, '1N'),
    (188.1, '1O'),
    (206.4, '1P'),
    (211.7, '1Q'),
    (225.5, '1R'),
    (300.0, '2A'),
    (62.3, '2B'),
    (96.1, '2C'),
    (100.1, "2C'"),
    (121.6, '2D'),
    (223.0, '2E'),
    (199.5, '2F'),
    (205.3, "2F'"),
    (279.5, '2G'),
    (71.0, '2H'),
    (80.3, '2I'),
    (88.9, '2J'),
    (171.6, '2M'),
    (30.7, '2N'),
    (262.4, '2O')
]

v_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'voyager_approved_clumps_list.pickle')
v_approved_list_fp = open(v_approved_list_fp, 'rb')
v_approved_db, v_approved_list = pickle.load(v_approved_list_fp)
v_approved_list_fp.close()


def list_to_db(v_clump_db, v_approved_list):
    
    v_db = {}
    for chain in v_approved_list:
        for clump in chain.clump_list:
            v_clump_db_entry = clumputil.ClumpDBEntry()
            obsid = clump.clump_db_entry.obsid
            if obsid not in v_db.keys():
                v_clump_db_entry.obsid = obsid
                v_clump_db_entry.clump_list = []
                v_clump_db_entry.ew_data = v_clump_db[obsid].ew_data # Filtered and normalized
                v_clump_db_entry.et = v_clump_db[obsid].et
                v_clump_db_entry.resolution_min = None 
                v_clump_db_entry.resolution_max = None
                v_clump_db_entry.emission_angle = None
                v_clump_db_entry.incidence_angle = None
                v_clump_db_entry.phase_angle = None
                v_clump_db_entry.et_min = v_clump_db[obsid].et_min
                v_clump_db_entry.et_max = v_clump_db[obsid].et_max
                v_clump_db_entry.et_min_longitude = v_clump_db[obsid].et_min_longitude
                v_clump_db_entry.et_max_longitude = v_clump_db[obsid].et_max_longitude
                v_clump_db_entry.smoothed_ew = None
 
                v_clump_db_entry.clump_list.append(clump)
                v_db[obsid] = v_clump_db_entry
                
            elif obsid in v_db.keys():
                v_db[obsid].clump_list.append(clump)         

    return v_db

def find_clump_name(clump_long, v_obs):
    max_dist = 1e38
    clump_name = None
    for long, name in voyager_clumps:
        if v_obs[1] != name[0]:
            continue
        if abs(long-clump_long) < max_dist:
            max_dist = abs(long-clump_long)
            clump_name = name

    if max_dist > 7:
        return 'BAD', 0
    return clump_name, max_dist

v_db = list_to_db(v_approved_db, v_approved_list)

for obs in sorted(v_db.keys()):
    print '***', obs, '***'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ew_data_filename, ew_mask_filename = ringutil.ew_paths(options, obs)
    orig_ew_data = np.load(ew_data_filename+'.npy')
    orig_ew_data = orig_ew_data.view(ma.MaskedArray)
    orig_ew_data.mask = np.load(ew_mask_filename+'.npy')
    v_db[obs].ew_data = orig_ew_data
    clumputil.plot_single_ew_profile(ax, v_db[obs], 0.0, 360.)
    clump_name_list = []
    last_right_clump_long = None
    for clump in sorted(v_db[obs].clump_list, key=lambda x: x.g_center):
        clump_name, dist = find_clump_name(clump.g_center, obs)
        print '%7.2f %5.2f = %7.2f-%7.2f' % (clump.g_center, clump.fit_width_deg,
                                             clump.fit_left_deg,
                                             clump.fit_right_deg),
        print clump_name, '(%5.2f)' % dist,
        if clump_name in clump_name_list:
            print 'DUPLICATE',
        if last_right_clump_long > clump.fit_left_deg:
            print 'OVERLAP',
        last_right_clump_long = clump.fit_right_deg
        print
        if clump_name != 'BAD':
            clump_name_list.append(clump_name)
        clumputil.plot_fitted_clump_on_ew(ax, v_db[obs].ew_data, clump)
        clumputil.plot_single_clump(ax, v_db[obs].ew_data, clump, 0.0, 360.)
        long_res = 360./len(v_db[obs].ew_data)
        plt.plot(clump.fit_left_deg, v_db[obs].ew_data[clump.fit_left_deg/long_res], 'o', ms=5, mfc='none', mec='green')
        plt.plot(clump.fit_right_deg, v_db[obs].ew_data[clump.fit_right_deg/long_res], 'o', ms=5, mfc='none', mec='red')
        plt.title(obs)
    
    print 'Not used:',
    for long, name in voyager_clumps:
        if obs[1] == name[0] and name not in clump_name_list:
            print name,
    print
    print
    

for chain in sorted(v_approved_list, key=lambda x: x.clump_list[0].clump_db_entry.obsid+('%7.2f'%x.clump_list[0].g_center)):
    print chain.clump_list[0].clump_db_entry.obsid,
    print '%7.2f' % chain.clump_list[0].g_center,
    print find_clump_name(chain.clump_list[0].g_center, chain.clump_list[0].clump_db_entry.obsid)[0]
    if len(chain.clump_list) > 1:
        print chain.clump_list[1].clump_db_entry.obsid,
        print '%7.2f' % chain.clump_list[1].g_center,
        print find_clump_name(chain.clump_list[1].g_center, chain.clump_list[1].clump_db_entry.obsid)[0]
    print '-------------------'

plt.show()
print


def edit_voyager_clumps(v_approved_db, v_approved_list):
    for i, chain in enumerate(v_approved_list):
        print i
        print chain.clump_list[0].clump_db_entry.obsid, chain.clump_list[0].g_center
    
    del v_approved_list[16]
        #does the right thing - re-save the list
#    v_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'voyager_approved_clumps_list.pickle')
#    v_approved_list_fp = open(v_approved_list_fp, 'wb')
#    pickle.dump((v_approved_db, v_approved_list),v_approved_list_fp)
#    v_approved_list_fp.close()

#edit_voyager_clumps(v_approved_db, v_approved_list)