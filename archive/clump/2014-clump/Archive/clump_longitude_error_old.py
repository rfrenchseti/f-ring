import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pickle
import sys
import os.path
import ringutil
import cspice
import clumputil
from optparse import OptionParser

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = [
                '-a',
#                '--replace_list'
                ]

parser = OptionParser()

parser.add_option('--replace_list', dest = 'replace_list', action = 'store_true', default = False)
ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)
def calc_lifetime(chain):
    
    day1 = chain.clump_list[0].clump_db_entry.et_max
    day2 = chain.clump_list[-1].clump_db_entry.et_max
    lifetime = (day2 - day1)/86400.     #seconds to days
    
    return lifetime

def calc_mean_motion_error(chain, sigma_y):
    clump_num = len(chain.clump_list)
    et_list = np.array([x.clump_db_entry.et for x in chain.clump_list])
    D = clump_num*np.sum(et_list**2) - (np.sum(et_list))**2
    
    sigma_b = sigma_y*np.sqrt(clump_num/D)
    return sigma_b

def find_error(chain):
    
    rate = chain.rate
    long_list = np.array([x.g_center for x in chain.clump_list])
    et_list = np.array([x.clump_db_entry.et for x in chain.clump_list])
    clump_num = len(chain.clump_list)

    et_list -= et_list[0]
    D = clump_num*np.sum(et_list**2) - (np.sum(et_list))**2

    A = (np.sum(et_list**2)*np.sum(long_list) - np.sum(et_list)*np.sum(long_list*et_list)) / D

    B = (clump_num*np.sum(et_list*long_list) - np.sum(et_list)*np.sum(long_list)) / D

    #B should be the same as the rate, A should be the same as the center longitude

    
    sigma_y = np.sqrt(np.sum(((long_list - A - B*et_list)**2) / (clump_num-2)))

    #To find the uncertainty in A and B:

    sigma_a = sigma_y*np.sqrt(sum(et_list**2) / D)

    sigma_b = sigma_y*np.sqrt(clump_num/D)
#    
#    A += (long_list[0]-180.)
#    A %= 360.
#    
#    print B-rate, A, chain.base_long
    return sigma_y, sigma_b
    
def propogate_mean_motion_error(chain, long_err):
    
    #determine the minimum and maximum slopes
    #min slope
    x2_a = chain.clump_list[-1].g_center -long_err
    x1_a = chain.clump_list[0].g_center + long_err
    y2_a = chain.clump_list[-1].clump_db_entry.et
    y1_a = chain.clump_list[0].clump_db_entry.et
    
    min_slope = (x2_a-x1_a)/(y2_a-y1_a)
    
    #max_slope
    x2_b = chain.clump_list[-1].g_center +long_err
    x1_b = chain.clump_list[0].g_center - long_err
    y2_b = chain.clump_list[-1].clump_db_entry.et
    y1_b = chain.clump_list[0].clump_db_entry.et
    
    max_slope = (x2_b-x1_b)/(y2_b-y1_b)
    
    print min_slope*86400, max_slope*86400
    return((max_slope-min_slope)/2.)

def prop_semimajor_err(rate, rate_error):
    
    a = ringutil.RelativeRateToSemimajorAxis(rate)
    a2 = ringutil.RelativeRateToSemimajorAxis(rate + rate_error)
#    a_min = ringutil.RelativeRateToSemimajorAxis(rate - rate_error)
#    a2 = ringutil.RelativeRateToSemimajorAxis(rate_error)
    da = np.abs(a2 - a)
    
    return (a, da)

def propogate_errors(c_approved_list, avg_long_err):

    for chain in c_approved_list:
        if len(chain.clump_list) >= 3:
            long_err, rate_err = find_error(chain)
            chain.rate_err = rate_err
            chain.long_err = long_err
        elif len(chain.clump_list) < 3:    
            rate_err = calc_mean_motion_error(chain, avg_long_err)
            chain.rate_err = rate_err
            chain.long_err = avg_long_err
            
        a, da = prop_semimajor_err(chain.rate, chain.rate_err)
        chain.a = a
        chain.a_err = da
        
        print chain.rate*86400, chain.rate_err*86400, chain.a, chain.a_err, chain.long_err, len(chain.clump_list)
    return c_approved_list


def find_sigma_y(c_approved_list):
    sigma_y_list = []
    sigma_b_list = []
    lifetimes = []
    for chain in c_approved_list:
        
        if len(chain.clump_list) > 2:
            print '***************************************************'
            sigma_y, sigma_b = find_error(chain)
            sigma_y_list.append(sigma_y)
            sigma_b_list.append(sigma_b)
            
            lifetime = calc_lifetime(chain)
            lifetimes.append(lifetime)
            slope_error = propogate_mean_motion_error(chain, 0.296)
            semimajor_axis, axis_error = prop_semimajor_err(chain.rate, sigma_b)
            print 'STARTING OBS', chain.clump_list[0].clump_db_entry.obsid
            print 'SECOND OBS',  chain.clump_list[1].clump_db_entry.obsid
            print 'STARTING LONG', chain.clump_list[0].g_center
            print 'SECOND LONG', chain.clump_list[1].g_center
            print 'CHAIN LEN ', len(chain.clump_list), 'CHAIN RATE ', chain.rate*86400
            print 'CHAIN LIFETIME: ', lifetime 
            print 'ERROR IN LONGITUDE: ', sigma_y
            print 'ERROR IN MEAN MOTION: ', sigma_b*86400, slope_error*86400
            print 'SEMI MAJOR AXIS/ERROR ', semimajor_axis, axis_error
    
    
    print ' '
    print 'AVERAGE SIGMA Y: ', np.mean(sigma_y_list), np.std(sigma_y_list)
    print 'AVERAGE SIGMA B: ', np.mean(sigma_b_list)*86400, np.std(sigma_b_list)*86400
    
    #distribution of sigma_y errors
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    y_bins = np.arange(np.min(sigma_y_list), np.max(sigma_y_list) + 0.05,0.05 )
    ax1.hist(sigma_y_list, y_bins)
    plt.xlabel('Sigma Y (Scatter Around the Line)')
    plt.title('Average: '+ str(np.mean(sigma_y_list)))
    plt.savefig('/home/shannon/Paper/Figures/sigma_y_dist.png')
#----------------------------------------------------------------------------------------------------------

c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_list_w_lives.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()





find_sigma_y(c_approved_list)
#list_w_errors = propogate_errors(c_approved_list, 0.4)

#sigma_b_list = [chain.rate_err for chain in list_w_errors]
#
#fig = plt.figure()
#ax2 = fig.add_subplot(212)
#b_bins = np.arange(np.min(sigma_b_list)*86400., np.max(sigma_b_list)*86400. + 0.005,0.005 )
#ax2.hist(np.array(sigma_b_list)*86400., b_bins)
#plt.show()


if options.replace_list:
    list_w_errors_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_list_w_errors.pickle')
    list_w_errors_fp = open(list_w_errors_fp, 'wb')
    pickle.dump((c_approved_db, list_w_errors), list_w_errors_fp)
    list_w_errors_fp.close()

#
#plt.plot(lifetimes, sigma_y_list, marker = '.',ls = '')
#plt.show()
