'''
Created on Jul 11, 2013

@author: rfrench
'''

import cspice
import ringutil
import ringimage
import numpy as np
import os
import scipy.optimize as sciopt


def keplers_equation_resid(p, M, e):
# p = (E)
# params = (M, e)
    return np.sqrt((M - (p[0]-e*np.sin(p[0])))**2)

# Mean anomaly M = n(t-tau)
# Find E, the eccentric anomaly, the angle from the center of the ellipse and the pericenter to a
# circumscribed circle at the place where the orbit is projected vertically.
def find_E(M, e):
    result = sciopt.fmin(keplers_equation_resid, (0.,), args=(M, e), disp=False, xtol=1e-20, ftol=1e-20)
    return result[0]

eccen = 0.00235
w0 = 24.2                        #deg
dw = 2.70025                

def rf_from_E(a, e, E):
    f_list = []
#    E_angles = (E + 2*np.pi)%(2*np.pi)
#    f_list = np.arctan(2*(np.sqrt(1+e)/np.sqrt(1-e))*np.tan(E_angles/2.))
    for i,angle in enumerate(E):
        f = np.arccos((np.cos(angle)-e)/(1-e*np.cos(angle)))
        if angle > np.pi:
            f = 2*np.pi - f
        f_list.append(f)               #moves the angle back to the correct coordinate
    f_list = np.array(f_list)
    
    return np.array([a*(1-e*np.cos(E)), f_list])   

debug = False

num_longitudes = 900
mean_longitude_list = 2*np.pi/num_longitudes * np.arange(num_longitudes)

clump_width_idx = int(20 / (360./num_longitudes))    
mean_motion = 0.4 # deg/day
mean_motion_idx = int(mean_motion / (360./num_longitudes))

for w in np.arange(0., 2*np.pi, np.pi/4):
    # THE OLD WAY
    
    if debug:
        print 'LAMBDA'
        print mean_longitude_list * 180/np.pi
    
    rotate = int(w/(2*np.pi/len(mean_longitude_list)))
    print 'Rotate amount', rotate
    E_path = os.path.join(ringutil.ROOT, 'eccentric_anom_arr_18000.npy')
    E_list = np.load(E_path)
    E_list = E_list[::18000/num_longitudes]
    E_list = np.roll(E_list, rotate)

#    print E_list * 180/np.pi
    
#    E_list = []
#    for M in mean_longitude_list:
#        E = find_E(M-w, eccen)
#        E_list.append(E)
#    E_list = np.array(E_list)
#    E_list = E_list % (2*np.pi)
    
#    print E_list * 180/np.pi
    
    if debug:
        print 'E'
        print E_list * 180/np.pi
    
    radii_set_list, old_f_list = rf_from_E(140220, eccen, E_list) 
    if debug:
        print 'f'
        print old_f_list * 180/np.pi
    
    old_theta_list = old_f_list + w
    old_theta_list %= 2*np.pi
    
    if debug:
        print 'THETA'
        print old_theta_list * 180/np.pi
    
    # THE NEW WAY
    
    if debug:
        print '-' * 50
    
        print 'LAMBDA'
        print mean_longitude_list * 180/np.pi
    
    M_list = mean_longitude_list - w
    
    if debug:
        print 'M'
        print M_list * 180/np.pi
    
    E_list = []
    for M in M_list:
        E = find_E(M, eccen)
        E_list.append(E)
    E_list = np.array(E_list)
    E_list = E_list % (2*np.pi)
    
    if debug:
        print 'E'
        print E_list * 180/np.pi
        
    radii_set_list, new_f_list = rf_from_E(140220, eccen, E_list) 
    new_f_list = np.array(new_f_list)
    
    if debug:
        print 'f'
        print new_f_list * 180/np.pi
    
    theta_list = new_f_list + w
    theta_list %= 2*np.pi
    
    if debug:
        print 'THETA'
        print theta_list * 180/np.pi
    
        print 'OLD'
        print old_f_list * 180/np.pi
        print 'NEW'
        print new_f_list * 180/np.pi
    
    diff = (old_f_list - new_f_list) * 180/np.pi
    diff = diff % 360
    diff[diff < 0] = diff[diff < 0] + 360
    diff[diff > 180] = 360-diff[diff > 180]
    
    if debug:
        print 'DIFF'
        print diff
    
    print 'W', w * 180/np.pi, 'MINDIFF', np.min(diff), 'MAXDIFF', np.max(diff), 'MEANDIFF', np.mean(diff)

    c_diff_list = []
    for x in range(len(old_f_list)-clump_width_idx):
        c_diff1 = old_f_list[x+clump_width_idx]-old_f_list[x]
        c_diff2 = new_f_list[x+clump_width_idx]-new_f_list[x]
        c_diff = (c_diff1 - c_diff2) * 180/np.pi
        c_diff_list.append(c_diff)
    
    print 'CLUMP DIFF MIN', np.min(c_diff_list), 'MAX', np.max(c_diff_list)
    
    mm1_diff_list = []
    mm30_diff_list = []
    for x in range(len(old_f_list)-mean_motion_idx*30):
        # 1 day
        mm_diff1_1 = (old_f_list[x+mean_motion_idx]-old_f_list[x]) / 1.
        mm_diff2_1 = (new_f_list[x+mean_motion_idx]-new_f_list[x]) / 1.
        # 30 day
        mm_diff1_30 = (old_f_list[x+mean_motion_idx*30]-old_f_list[x]) / 30. 
        mm_diff2_30 = (new_f_list[x+mean_motion_idx*30]-new_f_list[x]) / 30. 
        mm_diff1 = (mm_diff2_1 - mm_diff1_1) * 180/np.pi
        mm_diff30 = (mm_diff2_30 - mm_diff1_30) * 180/np.pi
        if abs(mm_diff1) < 2:
            mm1_diff_list.append(mm_diff1)
        if abs(mm_diff30) < 2:
            mm30_diff_list.append(mm_diff30)
    
    print 'MM 1 DAY MIN', np.min(mm1_diff_list), 'MAX', np.max(mm1_diff_list)
    print 'MM 30 DAY MIN', np.min(mm30_diff_list), 'MAX', np.max(mm30_diff_list)
    