import numpy as np
import numpy.ma as ma
import scipy.optimize as sciopt
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
                '--replace_list'
                ]

parser = OptionParser()

parser.add_option('--replace_list', dest = 'replace_list', action = 'store_true', default = False)
ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)


fring_e_bad = 0.000235
fring_e_good = 0.00235
fring_w0 = 24.2 # deg
fring_dw = 2.70025 # deg/day                

def keplers_equation_resid(p, M, e):
    return np.sqrt((M - (p[0]-e*np.sin(p[0])))**2)

def find_E(M, e):
    result = sciopt.fmin(keplers_equation_resid, (0.,), args=(M, e), disp=False, xtol=1e-20, ftol=1e-20)
    return result[0]

def rf_from_E(a, e, E):
    f_list = []
    for i,angle in enumerate(E):
        f = np.arccos((np.cos(angle)-e)/(1-e*np.cos(angle)))
        if angle > np.pi:
            f = 2*np.pi - f
        f_list.append(f)               #moves the angle back to the correct coordinate
    f_list = np.array(f_list)
    
    return np.array([a*(1-e*np.cos(E)), f_list])   

def E_from_f(e, f):
    tanE2 = np.tan(f/2) / np.sqrt((1+e)/(1-e))
    E = (np.arctan(tanE2)*2) % (2*np.pi)
    return E

def correct_mean_longitude(mean_longitude_corot, et):
    et_days = et/86400.
    w = ((fring_w0 + fring_dw*et_days)*(np.pi/180.)) % (2*np.pi)

    # bad lambda to theta
        
    mean_longitude_bad = ringutil.CorotatingToInertial(mean_longitude_corot, et) * np.pi/180
    M_bad = (mean_longitude_bad - w) % (2*np.pi)
    
    E_bad = find_E(M_bad, fring_e_bad) % (2*np.pi)
    
    r_trash, f = rf_from_E(140220, fring_e_good, [E_bad])
    f = f[0]
    
    theta = (f + w) % (2*np.pi)
    
    # theta to good lambda

    f = (theta - w) % (2*np.pi)
    E_good = E_from_f(fring_e_good, f)
    M_good = E_good - fring_e_good * np.sin(E_good)
    mean_longitude_good = (M_good + w) % (2*np.pi)
    mean_longitude_good_corot = ringutil.InertialToCorotating(mean_longitude_good*180/np.pi, et)
    
    return mean_longitude_good_corot

#===============================================================================
# 
#===============================================================================

# Use for 2-clump chains
def calc_mean_motion_error(clump_list, sigma_y):
    clump_num = len(clump_list)
    assert clump_num == 2
    et_list = np.array([x.clump_db_entry.et for x in clump_list])
    D = clump_num*np.sum(et_list**2) - (np.sum(et_list))**2
    
    sigma_b = sigma_y*np.sqrt(clump_num/D)
    return sigma_b

def find_error(chain, start):
    
#    rate = chain.rate

    long_list = np.array([x.g_center for x in chain.clump_list[start::]])
    et_list = np.array([x.clump_db_entry.et for x in chain.clump_list[start::]])
    clump_num = len(chain.clump_list[start::])

    for i in range(1, len(long_list)):
        if long_list[0] < 40 and long_list[i] >= 320:
            long_list[i] = long_list[i] - 360
        if long_list[0] > 320 and long_list[i] <= 40:
            long_list[i] = long_list[i] + 360

#    print long_list
    
    et_list -= et_list[0]
    D = clump_num*np.sum(et_list**2) - (np.sum(et_list))**2

    A = (np.sum(et_list**2)*np.sum(long_list) - np.sum(et_list)*np.sum(long_list*et_list)) / D

    B = (clump_num*np.sum(et_list*long_list) - np.sum(et_list)*np.sum(long_list)) / D
    #B should be the same as the rate, A should be the same as the center longitude

    
    sigma_y = np.sqrt(np.sum(((long_list - A - B*et_list)**2) / (clump_num-2)))

#    print et_list
#    print A, B, sigma_y
    
    #To find the uncertainty in A and B:

    sigma_a = sigma_y*np.sqrt(sum(et_list**2) / D)

    sigma_b = sigma_y*np.sqrt(clump_num/D)
#    
#    A += (long_list[0]-180.)
#    A %= 360.
#    
#    print B-rate, A, chain.base_long
    return sigma_y, sigma_b
    
def prop_semimajor_err(rate, rate_error):
    
    a = ringutil.RelativeRateToSemimajorAxis(rate)
    a2 = ringutil.RelativeRateToSemimajorAxis(rate + rate_error)
    a_min = ringutil.RelativeRateToSemimajorAxis(rate - rate_error)

    da1 = np.abs(a2 - a)
    da2 = np.abs(a_min-a)
    
    da_avg = (da1 + da2)/2.
    return (a, da_avg)

def propagate_errors(c_approved_list, avg_long_err):

    long_errs = []
    for i, chain in enumerate(c_approved_list):
        print chain.clump_list[0].clump_db_entry.obsid, chain.clump_list[0].g_center, chain.skip
        if chain.skip == False:
            split_chains = clumputil.check_for_split(chain, c_approved_list, i)
            if len(split_chains) > 0:
                split_chains.append(chain)              #now have a list of all chains that split
                second_clumps = [chain.clump_list[1] for chain in split_chains]
                second_centers = ['%6.2f'%(clump.g_center) for clump in second_clumps]
#                print second_centers
                for s_chain in split_chains:
#                    print len(s_chain.clump_list)
#                    print second_centers.count('%6.2f'%(s_chain.clump_list[1].g_center))
                    if (second_centers.count('%6.2f'%(s_chain.clump_list[1].g_center)) > 1):
                        start = 2
                    if len(s_chain.clump_list) == 2:
                        start = 0 
                    elif (len(s_chain.clump_list) > 2) and (second_centers.count('%6.2f'%(s_chain.clump_list[1].g_center)) <= 1):
                        start = 1
                    
                    len_clump_list = len(s_chain.clump_list[start::])
                    print len_clump_list
                    if len_clump_list >= 3:
                        print 'a'
                        long_err, rate_err = find_error(s_chain, start)
                        print rate_err*86400., long_err
                        s_chain.rate_err = rate_err
                        s_chain.long_err = long_err
                        long_errs.append(long_err)
                    elif len_clump_list < 3:
                        print 'b'    
                        rate_err = calc_mean_motion_error(s_chain.clump_list[start::], avg_long_err)
                        print rate_err*86400.
                        s_chain.rate_err = rate_err
                        s_chain.long_err = avg_long_err
#                        long_errs.append(avg_long_err)
                        
                    a, da = prop_semimajor_err(s_chain.rate, s_chain.rate_err)
                    s_chain.a = a
                    s_chain.a_err = da
                    
            #found no splits
            else:
                start = 0
                len_clump_list = len(chain.clump_list)
                
                if len_clump_list >= 3:
                    long_err, rate_err = find_error(chain, start)
                    chain.rate_err = rate_err
                    chain.long_err = long_err
                    long_errs.append(long_err)
                elif len_clump_list < 3:    
                    rate_err = calc_mean_motion_error(chain.clump_list, avg_long_err)
                    chain.rate_err = rate_err
                    chain.long_err = avg_long_err
#                    long_errs.append(avg_long_err)
                a, da = prop_semimajor_err(chain.rate, chain.rate_err)
                chain.a = a
                chain.a_err = da
            
#        print chain.rate*86400, chain.rate_err*86400, chain.a, chain.a_err, chain.long_err, len(chain.clump_list)
#        print chain.a, chain.a_err
    
    print 'LONGERR N MIN MAX', len(long_errs), np.min(long_errs), np.max(long_errs)
    print '66%', np.sort(long_errs)[int(len(long_errs)*.66)]
    
    print long_errs
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    y_bins = np.arange(np.min(long_errs), np.max(long_errs) + 0.05,0.05 )
    ax1.hist(long_errs, y_bins)
    plt.xlabel('Sigma Y (Scatter Around the Line)')
    plt.show()
    return c_approved_list

#def calc_lifetime(clump_list):
#    
#    day1 = clump_list[0].clump_db_entry.et_max
#    day2 = clump_list[-1].clump_db_entry.et_max
#    lifetime = (day2 - day1)/86400.     #seconds to days
#    
#    return lifetime

#THIS IS BROKEN DON'T USE IT
#def find_sigma_y(c_approved_list):
#    sigma_y_list = []
#    sigma_b_list = []
#    lifetimes = []
#    for i,chain in enumerate(c_approved_list):
#        check = clumputil.check_for_split(chain, c_approved_list, i)
#        if check == True:
#            len_clump_list = len(chain.clump_list[1::])
#        else:
#            len_clump_list = len(chain.clump_list)
#        if len_clump_list > 2:
#            print '***************************************************'
#            sigma_y, sigma_b = find_error(chain, 0)
#            sigma_y_list.append(sigma_y)
#            sigma_b_list.append(sigma_b)
#            
#            lifetime = calc_lifetime(chain)
#            lifetimes.append(lifetime)
#            semimajor_axis, axis_error = prop_semimajor_err(chain.rate, sigma_b)
#            print 'STARTING OBS', chain.clump_list[0].clump_db_entry.obsid
#            print 'STARTING LONG', chain.clump_list[0].g_center
#            print 'CHAIN LEN ', len(chain.clump_list), 'CHAIN RATE ', chain.rate*86400
#            print 'CHAIN LIFETIME: ', lifetime 
#            print 'ERROR IN LONGITUDE: ', sigma_y
#            print 'ERROR IN MEAN MOTION: ', sigma_b*86400, slope_error*86400
#            print 'SEMI MAJOR AXIS/ERROR ', semimajor_axis, axis_error
#        
#        #check again to see if there are any other splits. We no longer have a case where there is only one split off the parent.
#        #yes, this is a "hot-fix" And only works in the case where there is a maximum of three splits. If we wanted infinite ones we'd have to make this recursive.
#        
#    
#    print ' '
#    print 'AVERAGE SIGMA Y: ', np.mean(sigma_y_list), np.std(sigma_y_list)
#    print 'AVERAGE SIGMA B: ', np.mean(sigma_b_list)*86400, np.std(sigma_b_list)*86400
#    
#    #distribution of sigma_y errors
#    print np.max(sigma_y_list)
#    fig = plt.figure()
#    ax1 = fig.add_subplot(111)
#    y_bins = np.arange(np.min(sigma_y_list), np.max(sigma_y_list) + 0.05,0.05 )
#    ax1.hist(sigma_y_list, y_bins)
#    plt.xlabel('Sigma Y (Scatter Around the Line)')
#    plt.title('Average: '+ str(np.mean(sigma_y_list)))
#    plt.savefig('/home/shannon/Paper/Figures/sigma_y_dist.png')
    
#----------------------------------------------------------------------------------------------------------

c_approved_list_fp = os.path.join(ringutil.ROOT, 'clump-data', 'approved_list_w_lives.pickle')
c_approved_list_fp = open(c_approved_list_fp, 'rb')
c_approved_db, c_approved_list = pickle.load(c_approved_list_fp)
c_approved_list_fp.close()


# All of this crap is just because the reprojection eccentricity was wrong when finding
# the eccentric anomaly!
for chain in c_approved_list:
    chain.skip = False
    print 'CHAIN'
    for clump in chain.clump_list:
        if clump.g_center >= 0:
            old_long = clump.g_center
            new_long = correct_mean_longitude(old_long, clump.clump_db_entry.et) 
            print '%8.2f %8.2f' % (old_long, new_long)
            if abs(old_long-new_long) > 0.13:
                print '%8.2f %8.2f' % (old_long, new_long)
                print '*****WARNING*****'
            clump.g_center = -new_long # Flag it as new so we don't do it twice

for chain in c_approved_list:
    for clump in chain.clump_list: # Now go flip them back
        if clump.g_center < 0:
            clump.g_center = -clump.g_center

#now all the errors should be correct and only reflect the the split portion of the clumps and not include the parent clump.
c_approved_list = clumputil.remove_parent_clumps(c_approved_list)

for chain in c_approved_list:
    rate, ctr_long, long_err_list = clumputil.fit_rate(chain.clump_list)
    chain.rate = rate
    chain.base_long = ctr_long
    chain.long_err_list = long_err_list
    chain.a = ringutil.RelativeRateToSemimajorAxis(rate)
        
#find_sigma_y(c_approved_list)
list_w_errors = propagate_errors(c_approved_list, 0.3)

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
