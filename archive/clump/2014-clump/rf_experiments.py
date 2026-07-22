'''
Created on Mar 30, 2013

@author: rfrench
'''

import pickle
import os.path
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from clumputil import *
import cspice
import ringimage

FRING_MEAN_MOTION = 581.964

def RelativeRateToSemimajorAxis(rate):  # from ringutil
    return ((FRING_MEAN_MOTION / (FRING_MEAN_MOTION+rate*86400.))**(2./3.) * 140220.)

def validate_clump(clump):
    if not (clump.longitude/.04-1) <= clump.longitude_idx <= (clump.longitude/.04+1):
        print 'XXX BAD LONGITUDE XXX'
    if not (clump.scale/.04-1) <= clump.scale_idx <= (clump.scale/.04+1):
        print 'XXX BAD SCALE XXX'
    if clump.g_center < 0 or clump.g_center >= 360:
        print 'XXX BAD G_CENTER', clump.g_center, 'XXX'
    if clump.fit_left_deg < 0 or clump.fit_left_deg >= 360:
        print 'XXX BAD FIT_LEFT_DEG', clump.fit_left_deg, 'XXX'
    if clump.fit_right_deg < 0 or clump.fit_right_deg >= 360:
        print 'XXX BAD FIT_RIGHT_DEG', clump.fit_right_deg, 'XXX'
        
def validate_chain(chain):
    rate, ctr_long, long_err_list = fit_rate(chain.clump_list)
    if abs(rate - chain.rate) > 1e-10:
        print 'XXX BAD RATE', rate, chain.rate, 'XXX'
    for i in range(len(chain.clump_list)):
        if abs(long_err_list[i] - chain.long_err_list[i]) > 1e-10:
            print 'XXX BAD LONG ERR', long_err_list[i], chain.long_err_list[i], 'XXX'
    N = len(chain.clump_list)
    if N > 2:
        sigmay = np.sqrt(np.sum(np.array(chain.long_err_list)**2) / (N-2))
    else:
        sigmay = 0.4
    et_list = np.array([x.clump_db_entry.et for x in chain.clump_list])
    D = N * np.sum(et_list**2) - np.sum(et_list)**2
    sigmab = sigmay * np.sqrt(N / D)
    if abs(sigmab - chain.rate_err) > 1e-10:
        print 'XXX BAD RATE ERROR', sigmab, chain.rate_err, 'XXX'
    a = RelativeRateToSemimajorAxis(chain.rate)
    if abs(a-chain.a) > 1e-10:
        print 'XXX BAD A', a, chain.a, 'XXX'
    a_err = max(abs(RelativeRateToSemimajorAxis(chain.rate+chain.rate_err)-chain.a),
                abs(RelativeRateToSemimajorAxis(chain.rate+chain.rate_err)-chain.a)) # XXX
    if abs(a_err-chain.a_err) > 1e-2:
        print 'XXX BAD A_ERR', a_err, chain.a_err, 'XXX'
    
    for clump in chain.clump_list:
        validate_clump(clump)    
    
def print_clump(clump):
    print 'CLUMP %-25s ET %10.2f' % (clump.clump_db_entry.obsid, clump.clump_db_entry.et), 
    print 'WL LONG %6.2f WIDTH %7.2f' % (clump.longitude, clump.scale),
    print 'GAUSS L %7.3f S %6.3f B %6.4f H %8.6f' % (clump.g_center, clump.g_sigma,
                                                     clump.g_base, clump.g_height),
    print 'FL %7.3f FR %7.3f FW %7.3f' % (clump.fit_left_deg, clump.fit_right_deg, clump.fit_right_deg-clump.fit_left_deg),
                                                     
    print

def print_chain(chain):
    print 'CHAIN BASELONG %6.2f RATE %e' % (chain.base_long, chain.rate),
    print 'LONGERR', chain.long_err_list
    
    for clump in chain.clump_list:
        print_clump(clump)

def plot_one_clump(clump, color='black', lw=1, alpha=1):
    left_edge_idx = int(clump.fit_left_deg / .04) % 9000
    right_edge_idx = int(clump.fit_right_deg / .04) % 9000
    center_idx = int(clump.g_center/.04) % 9000
    if clump.fit_left_deg < clump.fit_right_deg:
        plt.plot(long_list[left_edge_idx:right_edge_idx+1], clump.clump_db_entry.ew_data[left_edge_idx:right_edge_idx+1],
                 lw=lw, color=color, alpha=alpha)
    else:
        plt.plot(long_list[left_edge_idx:], clump.clump_db_entry.ew_data[left_edge_idx:],
                 lw=lw, color=color, alpha=alpha)
        plt.plot(long_list[:right_edge_idx+1], clump.clump_db_entry.ew_data[:right_edge_idx+1],
                 lw=lw, color=color, alpha=alpha)
    plt.plot([long_list[center_idx], long_list[center_idx]],
             [np.min(clump.clump_db_entry.ew_data), np.max(clump.clump_db_entry.ew_data)],
             lw=1, color=color, alpha=alpha)

def dump_plot_chains(plot=False, plot_chain=None):
    for chain_num, chain in enumerate(c_approved_list):
        print '*** CHAIN', chain_num
        print_chain(chain)
        validate_chain(chain)
        print

        if plot:       
            if plot_chain is not None and plot_chain != chain_num:
                continue
             
            fig = plt.figure()
            
            for clump_num in range(len(chain.clump_list)):
                clump = chain.clump_list[clump_num]
                ax = fig.add_subplot(len(chain.clump_list),1,clump_num+1)
                plt.plot(long_list, clump.clump_db_entry.ew_data, '-', lw=1, color='#808080')
                plot_one_clump(clump, color='black', lw=1.5)
                ax.set_xlim(0,360)
            plt.show()

def plot_clumps_per_obs():
    for obsid in sorted(clump_db.keys()):
        print 'OBSID', obsid
        clump_db_entry = clump_db[obsid]
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.plot(long_list, clump_db_entry.ew_data, lw=1, color='#808080')
        
        color_num = 0
        colors = ['black', 'red', 'green', 'blue', 'magenta', 'cyan', 'yellow']
        for clump in clump_db_entry.clump_list:
                plt.plot(long_list, clump.clump_db_entry.ew_data, '-', lw=1, color='#808080')
                color = colors[color_num]
                color_num += 1
                if color_num >= len(colors): color_num = 0
                plot_one_clump(clump, color=color, lw=1.5, alpha=0.7)

        ax.set_xlim(0,360)
        plt.show()

def plot_nearby_obs_with_chains():
    clump_db_list = [clump_db[x] for x in clump_db.keys()]
    clump_db_list.sort(key=lambda x: x.et)
    
    start_num = 0
    last_end_num = 100000
    
    while start_num < len(clump_db_list):
        end_num = start_num
        while (end_num < len(clump_db_list)-1 and 
               clump_db_list[end_num+1].et < clump_db_list[end_num].et + 60*86400):
            end_num += 1
    
        if end_num == last_end_num: # No improvement - already saw these
            start_num += 1
            continue
        
        if end_num > start_num + 5: # Too many to display - restrict to 6
            end_num = start_num + 5
            
        last_end_num = end_num
        
        if True or start_num != end_num: # XXX
            # Find all the chains that have clumps in this range
            chain_list = []
            for chain in c_approved_list:
                for clump in chain.clump_list:
                    if clump.clump_db_entry.et >= clump_db_list[start_num].et and clump.clump_db_entry.et <= clump_db_list[end_num].et:
                        chain_list.append(chain)
                        break
            
            print len(chain_list)
            fig = plt.figure()
            for obsid_num in range(end_num-start_num+1):
                ax = fig.add_subplot(end_num-start_num+1,1,obsid_num+1)
                
                clump_db_entry = clump_db_list[start_num+obsid_num]
#                clump_db_entry.ew_data[200/.04:] = ma.masked
                
                print cspice.et2utc(clump_db_entry.et, 'C', 0)[:12], clump_db_entry.obsid
                plt.plot(long_list, clump_db_entry.ew_data, lw=1, color='#808080')
                p_rad, p_long = ringimage.saturn_to_prometheus(clump_db_entry.et_min)
                plt.plot([p_long, p_long], [ma.min(clump_db_entry.ew_data), ma.max(clump_db_entry.ew_data)],
                         '-', color='black', lw=2)

                colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']
                for chain_num, chain in enumerate(chain_list):
                    for clump_num, clump in enumerate(chain.clump_list):
                        if clump.clump_db_entry.et == clump_db_entry.et:
#                            clump.clump_db_entry.ew_data[200/.04:] = ma.masked
                            color = colors[chain_num % len(colors)]
                            plot_one_clump(clump, color=color, lw=2.5, alpha=0.8-0.5*float(clump_num)/(len(chain.clump_list)-1))

                ax.set_xlim(0,360)
                plt.title(clump_db_entry.obsid + ' ' + cspice.et2utc(clump_db_entry.et, 'C', 0)[:12])
            print
            plt.show()
        
        start_num += 1
            

#===============================================================================
# 
#===============================================================================

long_list = np.arange(9000) * .04
        
root = os.path.join(ringutil.ROOT, 'clump-data')

c_approved_list_fn = os.path.join(root, 'approved_list_w_errors.pickle')
c_approved_list_fp = open(c_approved_list_fn, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()

clumpdb_fn = os.path.join(root, 'clumpdb_137500_142500_05.000_0.020_10_02_137500_142500.pickle')
clumpdb_fp = open(clumpdb_fn, 'rb')
clump_find_options = pickle.load(clumpdb_fp)
clump_db = pickle.load(clumpdb_fp)
clumpdb_fp.close()

#plot_clumps_per_obs()

plot_nearby_obs_with_chains()

#dump_plot_chains(plot=False, plot_chain=70)
