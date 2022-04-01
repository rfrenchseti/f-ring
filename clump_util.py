'''
A module of all the routines needed to run the Clump Visualization gui.

Walk Clumps: Same as the program in the original track_clumps.py
Plot_EW and Plot Clumps: same functions from visualize_clumps.py
Plot_Profiles_In_Range: Takes a rectangle and plots all of the obs_ids inside the time span
Get_Clumps: Gets all of the clumps in the rectangular area
Track_clumps: Tracks the selected clumps - if it fails to find a chain, it runs a debugger.

Author: Shannon Hicks and Rob French
'''

import numpy as np
import clump_util
import matplotlib.pyplot as plt
import clump_cwt
import pickle
import os

import julian


#===============================================================================
#
#===============================================================================

class ClumpData(object):
    """Data for each clump."""
    validattr = ('clump_db_entry',                          # Back-pointer to ClumpDBEntry
                 'longitude',                               # Longitude 0-360
                 'longitude_idx',                           # Longitude as index into EW array
                 'scale',                                   # Scale of wavelet (degrees)
                 'scale_idx',                               # Scale of wavelet in units of indices into EW array
                 'mexhat_base',                             # The base of the best-fit mexican hat wavelet
                 'mexhat_height',                           # The height of the best-fir mexican hat wavelet
                 'abs_height',                              # The absolute height of the clump (center_height*mexhat_height)
                 'ignore_for_chain',                        # Don't use this clump when finding chains
                 'clump_sigma',                             # Number of sigma above the stdev of the EW Profile
                 'fit_base',
                 'fit_right_deg',
                 'fit_left_deg',
                 'fit_width_idx',
                 'fit_width_deg',
                 'fit_base',
                 'fit_height',
                 'fit_sigma',
                 'int_fit_height',
                 'g_center',
                 'g_sigma',
                 'g_base',
                 'g_height',
                 'matched',
                 'max_long',
                 'residual',
                 'wave_type'
                )
    userattr = []  # More attributes can be added by external users

    def __init__(self):
        for attr in list(ClumpData.validattr)+ClumpData.userattr:
            self.__setattr__(attr, None)

    def __setattr__(self, name, value):
        assert name in ClumpData.validattr or name in ClumpData.userattr
        self.__dict__[name] = value

    def print_all(self):
        for attr in list(ClumpData.validattr)+ClumpData.userattr:
            print(attr, ':', self.__getattribute__(attr))

class ClumpDBEntry(object):
    """Data for each entry in the clump DB - one per OBSID."""
    validattr = ('obsid',                                   # The OBSID
                 'et',                                      # The mean ephemeris time (ET) (sec)
                 'et_min',                                  # The minimum ET in the mosaic (sec)
                 'et_max',                                  # The maximum ET in the mosaic (sec)
                 'et_min_longitude',                        # The longitude where the minimum ET occurs
                 'et_max_longitude',                        # The longitude where the maximum ET occurs
                 'resolution_min',                          # The minimum radial resolution in the mosaic (km/pix)
                 'resolution_max',                          # The maximum radial resolution in the mosaic (km/pix)
                 'emission_angle',                          # The mean emission angle in the mosaic (deg)
                 'incidence_angle',                         # The mean incidence angle in the mosaic (deg)
                 'phase_angle',                             # The mean phase angle in the mosaic (deg)
                 'ew_data',                                 # Masked array of EWs normalized and possibly filtered
                 'clump_list',
                 'smoothed_ew'                             # The list of clumps in this OBSID
                )
    userattr = []  # More attributes can be added by external users

    def __init__(self):
        for attr in list(ClumpDBEntry.validattr)+ClumpDBEntry.userattr:
            self.__setattr__(attr, None)

    def __setattr__(self, name, value):
        assert name in ClumpDBEntry.validattr or name in ClumpDBEntry.userattr
        self.__dict__[name] = value

    def print_all(self):
        for attr in list(ClumpDBEntry.validattr)+ClumpDBEntry.userattr:
            print(attr, ':', self.__getattribute__(attr))

class ClumpChainData(object):
    """Data for each clump chain."""
    validattr = ('rate',                                    # The movement rate relative to the co-rot frame (deg/sec)
                 'base_long',                               # The optimal longitude for the first clump
                 'long_err_list',                           # The longitude errors compared to straight line
                 'clump_list',                              # The list of clumps in this chain
                 'lifetime',
                 'lifetime_upper_limit',
                 'lifetime_lower_limit',
                 'long_err',
                 'rate_err',
                 'a_err',
                 'a',
                 'skip'                                     #need to skip over the chain in the clump table
                )
    userattr = []  # More attributes can be added by external users

    def __init__(self):
        for attr in list(ClumpChainData.validattr)+ClumpChainData.userattr:
            self.__setattr__(attr, None)

    def __setattr__(self, name, value):
        assert name in ClumpChainData.validattr or name in ClumpChainData.userattr
        self.__dict__[name] = value

class ClumpFindOptions(object):
    """Options used to find clumps."""
    validattr = ('type',                                    # The type of clump to find: 'wavelet mexhat'
                 'scale_min',                               # Min scale size
                 'scale_max',                               # Max scale size
                 'scale_step',                              # Scale size increment
                 'clump_size_min',                          # Min clump size to find
                 'clump_size_max',                          # Max clump size to find
                 'prefilter'                                # EWs prefiltered
                )
    userattr = []  # More attributes can be added by external users

    def __init__(self):
        for attr in list(ClumpFindOptions.validattr)+ClumpFindOptions.userattr:
            self.__setattr__(attr, None)

    def __setattr__(self, name, value):
        assert name in ClumpFindOptions.validattr or name in ClumpFindOptions.userattr
        self.__dict__[name] = value

class ClumpChainOptions(object):
    """Options used to find clump chains."""
    validattr = ('clump_size_min',                          # Min clump size used to find chains
                 'clump_size_max',                          # Max clump size used to find chains
                 'longitude_tolerance',                     # Maximum dist longitude can be off the line
                 'max_time',                                # The maximum time a chain might live
                 'max_movement'                             # The maximum movement rate (deg/sec)
                )
    userattr = []  # More attributes can be added by external users

    def __init__(self):
        for attr in list(ClumpChainOptions.validattr)+ClumpChainOptions.userattr:
            self.__setattr__(attr, None)

    def __setattr__(self, name, value):
        assert name in ClumpChainOptions.validattr or name in ClumpChainOptions.userattr
        self.__dict__[name] = value


#===============================================================================
#
#===============================================================================



#===============================================================================
#
#===============================================================================

#
# Given a list of clumps sorted by time, find the slope (deg/sec) and intercept (deg)
# that best fits the longitudes.  Handle the case where the longitudes wrap around the edge.
# Note that we DON'T handle the case where the longitudes span more than 180 degrees.
#
def fit_rate(clump_list):
    long_list = np.array([x.g_center for x in clump_list])
    et_list = np.array([x.clump_db_entry.et for x in clump_list])

    # Normalize the longitudes so the first longitude is 180.  Then we wrap the others around as necessary.
    long_list -= (long_list[0]-180.)
    long_list %= 360.

    # Normalize the times so the first time is 0.
    et_list -= et_list[0]

    rate, ctr_long = np.polyfit(et_list, long_list, 1)
    # Convert the center longitude back to the original coordinate system
    ctr_long += (clump_list[0].g_center-180.)
    ctr_long %= 360.

    long_err_list = []
    for clump in clump_list:
        long_err_list.append(clump.g_center-
                             ((clump.clump_db_entry.et-clump_list[0].clump_db_entry.et)*rate+ctr_long))

    return rate, ctr_long, long_err_list

def walk_clumps(movie_list, clumps_db, max_rate, max_longitude_err, max_time, scale_tolerance,
                cur_clump_chain, clump_chain_list, debug=False, num_skipped=0):
    # movie_list - The sorted list of OBSIDs; movie_list[0] is the current movie
    # clumps_db - All clumps in all movies indexed by OBSID
    # rate - Shift amount in deg/sec
    # longitude_tolerance - How close a clump's longitude has to match (in degrees)
    # cur_clump_chain - The current list of clumps that exist along this rate
    # clump_chain_db - The master place to store all completed clump chains

    found_one = False

    if len(movie_list) > 0:
        # There's at least one movie left
        cur_obsid = movie_list[0]
        clump_list = clumps_db[cur_obsid].clump_list
        cur_ET = clumps_db[cur_obsid].et_max

        delta_t = cur_ET-cur_clump_chain[0].clump_db_entry.et
        if delta_t < max_time:
            # Still within maximum time
            for clump_num, clump in enumerate(clump_list):
                if clump.ignore_for_chain:
                    continue
                if clump.fit_width_deg < (cur_clump_chain[-1].fit_width_deg/scale_tolerance) or clump.fit_width_deg > (cur_clump_chain[-1].fit_width_deg*scale_tolerance):
                    #scale width must be within a REASONABLE range +/- the scale of the previous clump.
                    #this prevents a 5 degree clump getting paired with a 30 degree clump
                    continue
                # Copy the current chain and add on this new clump
                new_clump_chain = cur_clump_chain[:] + [clump]
                # Find the best linear fit to this chain
                rate, base_long, fit_res = fit_rate(new_clump_chain)
                long_err_list = fit_res
                if abs(rate) <= max_rate and np.max(np.abs(long_err_list)) <= max_longitude_err:
                    # So far so good! Remember that we found one
                    found_one = True
                    # Keep walking down to the next movie
                    walk_clumps(movie_list[1:], clumps_db, max_rate, max_longitude_err, max_time,scale_tolerance,
                                new_clump_chain, clump_chain_list, debug)
                    if not num_skipped:
                        # Allow us to skip a single movie
                        walk_clumps(movie_list[2:], clumps_db, max_rate, max_longitude_err, max_time, scale_tolerance,
                                    new_clump_chain, clump_chain_list,  debug, num_skipped=num_skipped+1)

    if not found_one:
        # We didn't find any way to continue down the chain...the current clump_chain
        # is complete.  Record it if we like it.
        if len(cur_clump_chain) >= 2:
            # First check to see if we already have this chain or a superset of this chain
            dup = False
            for chain in clump_chain_list:
                if set(cur_clump_chain).issubset(chain.clump_list):
                    dup = True
                    break
            if not dup:
                chain = ClumpChainData()
                rate, base_long, fit_res = fit_rate(cur_clump_chain)
                long_err_list = fit_res
                chain.rate = rate
                chain.base_long = base_long
                chain.long_err_list = long_err_list
                chain.clump_list = cur_clump_chain
                clump_chain_list.append(chain)
                if debug:
                    print('FOUND CHAIN Rate %11.8f BaseLong %6.2f Longs' % (rate*86400, base_long), end=' ')
                    for i, clump in enumerate(cur_clump_chain):
                        print('%6.2f(%6.2f)' % (clump.g_center, fit_res[i]), end=' ')
                    print()

def get_sorted_obsid_list(clump_db):
    obsid_by_time_db = {}
    for obs_id in list(clump_db.keys()):
        max_et = clump_db[obs_id].et_max
        obsid_by_time_db[max_et] = obs_id
    sorted_obsid_list = []
    for et in sorted(obsid_by_time_db.keys()):
        sorted_obsid_list.append(obsid_by_time_db[et])
    return sorted_obsid_list

# clump.ignore_for_chain must be set before entry
def track_clumps(clump_db, max_movement, longitude_tolerance, max_time, scale_tolerance):
    sorted_obsid_list = get_sorted_obsid_list(clump_db)
    clump_chain_list = []

    # Iterate through each movie in chronological order
    for top_id_num, top_id in enumerate(sorted_obsid_list):
        print('PROCESSING TOPLEVEL', top_id)

        clump_list = clump_db[top_id].clump_list
        remaining_movie_obsids = sorted_obsid_list[top_id_num+1::] #the rest of the ids

        # Now iterate through each top clump
        for top_clump_num, top_clump in enumerate(clump_list):
            if top_clump.ignore_for_chain:

                continue

            walk_clumps(remaining_movie_obsids, clump_db, max_movement,
                        longitude_tolerance, max_time, scale_tolerance,
                        [top_clump], clump_chain_list, debug=True)

    return clump_chain_list

def plot_single_clump(ax, ew_data, clump, long_min, long_max, label=False, color='red'):
    long_res = 360. / len(ew_data)
    longitudes = np.arange(len(ew_data)*3) * long_res - 360.
    mother_wavelet = clump_cwt.SDG(len_signal=ew_data.size*3,
                                   scales=np.array([int(clump.scale_idx/2)]))
    mexhat = mother_wavelet.coefs[0].real # Get the master wavelet shape
    mh_start_idx = int(round(len(mexhat)/2.-clump.scale_idx/2.))
    mh_end_idx =   int(round(len(mexhat)/2.+clump.scale_idx/2.))
    mexhat = mexhat[mh_start_idx:mh_end_idx+1] # Extract just the positive part
    mexhat = mexhat*clump.mexhat_height+clump.mexhat_base
    longitude_idx = clump.longitude_idx
    if longitude_idx+clump.scale_idx/2 >= len(ew_data):
        # Runs off right side - make it run off left side instead
        longitude_idx -= len(ew_data)
    # Longitude range in data
    idx_range = longitudes[longitude_idx-int(clump.scale_idx/2)+len(ew_data):
                           longitude_idx-int(clump.scale_idx/2)+len(mexhat)+len(ew_data)]
    legend = None
    if label:
        legend = (f'L={clump.longitude:7.2f} W={clump.scale:7.2f} '+
                  f'H={clump.mexhat_height:6.3f}')
    ax.plot(idx_range, mexhat, '-', color= color, lw=2, alpha=0.8, label=legend)
    if longitude_idx-clump.scale_idx/2 < 0: # Runs off left side - plot it twice
        ax.plot(idx_range+360, mexhat, '-', color=color, lw=2, alpha=0.8)
    ax.set_xlim(long_min, long_max)

def plot_single_ew_profile(ax, clump_db_entry, long_min, long_max, label=False, color='black'):
    ew_data = clump_db_entry.ew_data
    long_res = 360. / len(ew_data)
    longitudes = np.arange(len(ew_data)) * long_res
    min_idx = int(long_min / long_res)
    max_idx = int(long_max / long_res)
    long_range = longitudes[min_idx:max_idx+1]
    ew_range = ew_data[min_idx:max_idx+1]
    legend = None
    if label:
        legend = clump_db_entry.obsid + ' (' + et2utc(clump_db_entry.et, 'C', 0) + ')'
    ax.plot(long_range, ew_range, '-', label=legend, color=color)

def plot_fitted_clump_on_ew(ax, ew_data, clump, color = 'blue', alpha=0.5):
    long_res = 360./len(ew_data)
    longitudes =np.tile(np.arange(0,360., long_res),3)
    tri_ew = np.tile(ew_data, 3)
    left_idx = clump.fit_left_deg/long_res + len(ew_data)
    right_idx = clump.fit_right_deg/long_res + len(ew_data)

    if left_idx > right_idx:
        left_idx -= len(ew_data)

    idx_range = longitudes[left_idx:right_idx+1]

    if left_idx < len(ew_data):
        ax.plot(longitudes[left_idx:len(ew_data)], tri_ew[left_idx:len(ew_data)], color = color, alpha = alpha, lw = 2)
        ax.plot(longitudes[len(ew_data):right_idx+1], tri_ew[len(ew_data):right_idx+1], color = color, alpha = alpha, lw = 2)
    else:
        ax.plot(idx_range, tri_ew[left_idx:right_idx+1], color = color, alpha = 0.5, lw = 2)

def plot_ew_profiles_in_range(clump_db, long_min, long_max, et_min, et_max, ew_color='black', clump_color='red'):
    sorted_obsid_list = get_sorted_obsid_list(clump_db)

    good_obsid_list = []
    for obsid in sorted_obsid_list:
        if et_min < clump_db[obsid].et < et_max:
            good_obsid_list.append(obsid)

    sub_size = len(good_obsid_list)
    fig, axes = plt.subplots(sub_size, sharex=False)
    if sub_size == 1:
        axes = [axes] # Don't know why they return a single axis in this case

    axes = axes[::-1] # Earliest on the bottom

    for n, obsid in enumerate(good_obsid_list):
        plot_single_ew_profile(axes[n], clump_db[obsid], long_min, long_max, label=True, color=ew_color)
        for clump in clump_db[obsid].clump_list:
            if long_min-clump.scale < clump.longitude < long_max+clump.scale:
                plot_single_clump(axes[n], clump_db[obsid].ew_data, clump, long_min, long_max, color=clump_color)
                plot_fitted_clump_on_ew(axes[n], clump_db[obsid].ew_data, clump)
        axes[n].legend([obsid], loc=1, frameon=False)
        axes[n].set_xlim(long_min, long_max)

    plt.subplots_adjust(top=.96, bottom=.04, right=.96, left=.04)

    plt.show()

def compare_inert_to_peri(options, obs_id, clump_idx, clump_longitude):

    w0 = 24.2                        #deg
    dw = 2.70025

    #load metadata - need the ET corresponding to the clump's longitude.
    (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
         bkgnd_mask_filename, bkgnd_model_filename,
         bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obs_id)

    (ew_data_filename, ew_mask_filename) = ringutil.ew_paths(options, obs_id)

    if (not os.path.exists(reduced_mosaic_metadata_filename)) or (not os.path.exists(ew_data_filename+'.npy')):
        print('NO DATA AVAILABLE FOR', obs_id)
        return
    else:
        reduced_metadata_fp = open(reduced_mosaic_metadata_filename, 'rb')
        mosaic_data = pickle.load(reduced_metadata_fp)
        obsid_list = pickle.load(reduced_metadata_fp)
        image_name_list = pickle.load(reduced_metadata_fp)
        full_filename_list = pickle.load(reduced_metadata_fp)
        reduced_metadata_fp.close()

        (mosaic_longitudes, mosaic_resolutions, mosaic_image_numbers,
         mosaic_ETs, mosaic_emission_angles, mosaic_incidence_angles,
         mosaic_phase_angles) = mosaic_data

        clump_et = mosaic_ETs[clump_idx]

        clump_inertial_long = ringutil.CorotatingToInertial(clump_longitude, clump_et)
        dt = clump_et/86400.
        pericentre_long = (w0 + dw*dt)%360.

        return (clump_inertial_long, pericentre_long, (clump_inertial_long + pericentre_long)%360.)

# This is a poorly named routine. It actually recomputes mean motion for split clumps.
def remove_parent_clumps(c_approved_list):

    for i, chain in enumerate(c_approved_list):
        parent_clump_long = '%6.2f'%(chain.clump_list[0].g_center)
        second_clump = '%6.2f'%(chain.clump_list[1].g_center)
        #check the other chains
        for new_chain in c_approved_list[i+1::]:
            new_parent_long = '%6.2f'%(new_chain.clump_list[0].g_center)
            new_second = '%6.2f'%(new_chain.clump_list[1].g_center)
            if (new_parent_long == parent_clump_long) and (len(new_chain.clump_list) > 2):

                print('Found a splitting clump', parent_clump_long, chain.clump_list[0].clump_db_entry.obsid, new_parent_long, new_chain.clump_list[0].clump_db_entry.obsid)
                #recalculate mean motions for these chains after removing the parent from the calculation.
                if second_clump != new_second:
                    rate, ctr_long, long_err_list = fit_rate(chain.clump_list[1::])
                elif second_clump == new_second:
                    rate, ctr_long, long_err_list = fit_rate(chain.clump_list[2::])
                    print([clump.g_center for clump in chain.clump_list[2::]])

                chain.rate = rate
                chain.base_long = ctr_long
                chain.long_err_list = long_err_list
                chain.a = ringutil.RelativeRateToSemimajorAxis(rate)


                if second_clump != new_second:
                    nrate, nctr_long, nlong_err_list = fit_rate(new_chain.clump_list[1::])
                elif second_clump == new_second:
                    nrate, nctr_long, nlong_err_list = fit_rate(new_chain.clump_list[2::])
                    print([clump.g_center for clump in new_chain.clump_list[2::]])

#                 nrate, nctr_long, nlong_err_list = fit_rate(new_chain.clump_list[1::])
                new_chain.rate = nrate
                new_chain.base_long = nctr_long
                new_chain.long_err_list = nlong_err_list
                new_chain.a = ringutil.RelativeRateToSemimajorAxis(nrate)

    return c_approved_list

def check_for_split(chain, c_approved_list, marker):

    parent_clump_long = '%6.2f'%(chain.clump_list[0].g_center)
    #check the other chains
    split_chains = []
    for new_chain in c_approved_list[marker +1::]:
        new_parent_long = '%6.2f'%(new_chain.clump_list[0].g_center)
        if new_parent_long == parent_clump_long:
            print('Found a splitting clump', parent_clump_long, chain.clump_list[0].clump_db_entry.obsid, new_parent_long, new_chain.clump_list[0].clump_db_entry.obsid)
            split_chains.append(new_chain)
            new_chain.skip = True

    return split_chains

def utc2et(s):
    return julian.tdb_from_tai(julian.tai_from_iso(s))

def et2utc(et, *args):
    # XXX Implement other args
    print(et)
    return 'Fake time'
    return julian.iso_from_tai(julian.tai_from_tdb(et))
