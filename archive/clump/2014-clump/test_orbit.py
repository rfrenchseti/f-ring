'''
Created on Oct 22, 2012

@author: rfrench
'''

import numpy as np
import scipy.optimize as sciopt
import matplotlib.pyplot as plt

fring_a = 140221.3      # km
fring_e = 0.00235
fring_i = 0.00643
fring_wdot = 2.70025    # deg/sec
fring_odot = -2.68778   # deg/sec
saturn_gm = 37931187    # km^3 s^-2

fring_a_peri = fring_a * (1-fring_e)
fring_a_apo = fring_a * (1+fring_e)
FRING_MEAN_MOTION = 581.979

# Orbital period in seconds - this depends only on the semi-major axis
def orbit_period(gm, a):
    return np.sqrt(4 * np.pi**2 * a**3 / gm)

# Orbital mean velocity in km/sec - this depends on the semi-major axis, period, and eccentricity 
def orbit_mean_v(a, T, e):
    return (2 * np.pi * a / T) * (1 - e**2/4)

# Orbital velocity in km/sec at a particular position in the orbit - this depends on GM, the semi-major axis (km),
# and the radial position (km)
def orbit_v(gm, a, r):
    return np.sqrt(gm * (2/r - 1/a))

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

# Find r, the radial distance from the focus, and f, the true anomaly, the angle from the focus and the
# pericenter to the orbit.
def rf_from_E(a, e, E):
    return np.array([a*(1-e*np.cos(E)), np.arccos((np.cos(E)-e)/(1-e*np.cos(E)))])

# Find X,Y (relative to the focus)
def xy_from_E(a, e, E):
    return np.array([a*(np.cos(E)-e), a*np.sqrt(1-e**2)*np.sin(E)])

# Find X,Y (relative to the origin)
def xy_from_E_origin(a, e, E):
    return np.array([a*np.cos(E), a*np.sqrt(1-e**2)*np.sin(E)])

def E_from_xy(x, y, a, e):
#    return np.array([np.arccos((x/a) + e), np.arcsin(y/(a*np.sqrt(1-e**2)))])
    den = a*np.sqrt(1- e**2)*((x/a) + e)
    E_list = np.arctan2(y, den)
    
    E_list[E_list < 0] = E_list[E_list < 0] + 2*np.pi
    
    return E_list

def M_from_E(E_list, e):
    return np.array(E_list - e*np.sin(E_list))
    
def RelativeRateToSemimajorAxis(rate):
    return ((FRING_MEAN_MOTION / (FRING_MEAN_MOTION*rate))**(-2./3.) * fring_a)

    
    
print '*** BASIC ORBITAL PARAMETERS ***'
print 'FRING a =', fring_a, 'km'
print 'FRING e =', fring_e
print 'FRING i =', fring_i, 'deg'
print 'FRING w-dot =', fring_wdot, 'deg/day'
print 'FRING omega-dot =', fring_odot, 'deg/day'
print 'SATURN GM =', saturn_gm, 'km^3/s^2'
print

T = orbit_period(saturn_gm, fring_a)
T_days = T/3600.
deg_per_day = 24./T_days * 360.

print '*** DERIVED ORBITAL PARAMETERS ***'
print 'FRING period T =', T, 'secs =', T_days, 'days'
print 'FRING mean motion =', deg_per_day, 'deg/day'
print 'FRING periapse =', fring_a_peri, 'km'
print 'FRING apoapse =', fring_a_apo, 'km'
print 'FRING apoapse-periapse =', fring_a_apo-fring_a_peri, 'km'

T_360 = T/360.

v_circ = orbit_mean_v(fring_a, T, fring_e)

print 'FRING mean v (order e^2) =', v_circ, 'km/sec'

v_max = orbit_v(saturn_gm, fring_a, fring_a_peri)
v_min = orbit_v(saturn_gm, fring_a, fring_a_apo)

print 'FRING min v =', v_min, 'km/sec'
print 'FRING max v =', v_max, 'km/sec'
print 'FRING maxv/minv =', (v_max-v_min)/v_min*100, '%'

print

# (a,e)
#clump_specs = [(10, .1)]
clump_specs = [(fring_a, fring_e),     # Clump perfectly in the core
               (fring_a, fring_e - 0.000713),           # 
               (fring_a+100, fring_e),
               (fring_a + 100, fring_e + 0.000713)]

#difference in longitude of pericenter from the core's. Implies a different rate of precession.
dw = np.arange(0., 190., 10)*np.pi/180.

for w in dw:
    print 'Relative Change in Long of Pericenter:', w*180/np.pi
    for n, (clump_a, clump_e) in enumerate(clump_specs):
        
        M_list = []
        E_list = []
        for t in np.arange(0., 360.):
            # Mean anomaly goes 0-2pi in equal steps of 2pi/360 (=1 degree)
            M = t / 360. * 2*np.pi  #(mean motion = 1)
            M_list.append(M)
            
            # Given the mean anomaly, find the eccentric anomaly
            E = find_E(M, clump_e)
            E_list.append(E)
            if n == 0:
                
                M_core = np.array(M_list)
                E_core = np.array(E_list)
                
                
        M_list = np.array(M_list)
        E_list = np.array(E_list)
        rotate = int(w/(2*np.pi/len(E_list)))
        E_list = np.roll(E_list, rotate)
        
        # The rest of these can all be vectorized
    
        # With the eccentric anomaly, we can find r, f, x/y
        #where the clump actually is
        x_list, y_list = xy_from_E(clump_a, clump_e, E_list) # Relative to focus - x and y along the ellipse
        print len(x_list), len(y_list)
        #where we reproject the clump to be in the core
        E_clump_core = E_from_xy(x_list, y_list, fring_a, fring_e)
        
        #find Mean Anomalies for the reprojected core position and compare to original M_list for the clump.
        M_clump_core = M_from_E(E_clump_core, fring_e)*180/np.pi

# I'm not sure we have to worry about wrapping - maybe I'm wrong though?        
        print M_list*180./np.pi
        print '   '
        print M_clump_core
        
        M_difference = M_list*(180./np.pi) - M_clump_core
        
        plt.plot(M_list*(180./np.pi), M_difference)
        plt.show()
    
    #    print E_clump_list
    #    x_clump_list, y_clump_list = xy_from_E(clump_a, clump_e, E_clump_list)
        
    #    clump_think_long = np.arctan2(y_list, x_list)*180/np.pi
#    #    clump_act_long = np.arctan2(y_clump_list, x_clump_list)*180/np.pi
#        clump_think_long = E_list*180/np.pi
#        clump_act_long = E_clump_list*180/np.pi
#    
#        
#        # The original reprojection - we would assert that the F ring core, wherever we see it, is actually at
#        # a=140,220, e=0.
#        # In this case the true longitude we are _actually_ we are looking at equals the mean
#        # anomaly, since it just steps around the circle in equal increments.
#        # The longitude we _think_ we are looking at, though, (in terms of the position of a clump in its
#        # orbit) is arctan(y/x).
#    #    #
#    #    circ_act_long = M_list * 180/np.pi
#    #    circ_think_long = np.arctan2(y_list, x_list) * 180/np.pi
#    #    circ_think_long[circ_think_long < 0] = circ_think_long[circ_think_long < 0]+360
#    #    circ_think_long_error = circ_act_long-circ_think_long
#        
#    #    clump_think_long[clump_think_long < 0] = clump_think_long[clump_think_long < 0]+360
#    #    clump_act_long[clump_act_long < 0] = clump_act_long[clump_act_long < 0]+360
#        clump_think_long_error = clump_act_long-clump_think_long
#        
#        plt.plot(clump_think_long, clump_think_long_error)
#        plt.title('Error in longitude we think we are looking at vs. actually looking at')
#        plt.xlabel('Longitude we think we are looking at (deg)')
#        plt.ylabel('Error (degrees)')
#        plt.xlim(0,360)
##        plt.show()
#        
#        print 'Max Long Error:', np.max(clump_think_long_error)
#        # We think the mean motion is 1 at each position
#        clump_think_long_diff = (clump_think_long[1:] - clump_think_long[:-1])
#        
#        # But what is it really? 
#        clump_act_long_diff = (clump_act_long[1:] - clump_act_long[:-1])
#    
#    #    print clump_act_long_diff
#        clump_mean_motion_error = clump_think_long_diff - clump_act_long_diff
#        
#        print 'Max Mean Motion Error:', np.max(clump_mean_motion_error)
#    #    plt.plot(circ_act_long_diff)
#    #    plt.plot(circ_think_long_diff)
#    #    plt.show()
#        
#        plt.plot(clump_think_long[:-1], clump_mean_motion_error)
#        plt.title('Delta Actual longitude')
#        plt.xlabel('Longitude we think we are looking at (deg)')
#        plt.ylabel('Delta Longitude Error (deg/day)')
##        plt.show()
#        
#        
#        #plot clump semi-major axis changes
#        
#    #    clump_think_da = RelativeRateToSemimajorAxis(clump_think_long_diff)
#        clump_act_da = RelativeRateToSemimajorAxis(clump_mean_motion_error + 1.0)
#    #    print clump_think_da
#        clump_da_diff = fring_a - clump_act_da
#        print 'Max Clump Semi-Major Axis Diff', np.max(clump_da_diff)
#        
#        plt.plot(clump_mean_motion_error, clump_da_diff)
#        plt.title('Delta Semi-Major Axis')
#        plt.xlabel('Semi-Major Axis we THINK we have')
#        plt.ylabel('Delta Semi-Major Axis Error')
##        plt.show()
#        
#        print '  '
#        dist_1_2 = np.sqrt((x2-x1)**2+(y2-y1)**2)
#        dist_circ_circ = np.sqrt((x2circ-x1circ)**2+((y2circ-y1circ)**2))
#        dist_1_circ = np.sqrt((x1circ-x1)**2+(y1circ-y1)**2)
#    
#        x1x2_list.append(x2-x1)
#        y1y2_list.append(y2-y1)
#        e1e2_list.append((E2-E1)/dM)
#        
#        dist_1_2_list.append(dist_1_2)
#        dist_1_circ_list.append(dist_1_circ)
#        dist_circ_circ_list.append(dist_circ_circ)
#    
#    circ_dist = dist_circ_circ_list[0]
#    min_1_2 = np.min(dist_1_2_list)
#    max_1_2 = np.max(dist_1_2_list)
#    
#    print '*** CLUMP FOLLOWING ANALYSIS ***'
#    print 'Circle distance =', circ_dist, 'km'
#    print 'Elliptical dist min =', min_1_2, 'km'
#    print 'Elliptical dist max =', max_1_2, 'km'
#    print 'Min ratio elliptical/circle =', min_1_2/circ_dist
#    print 'Max ratio elliptical/circle =', max_1_2/circ_dist
#    print 'Min error in longitude =', 360-(min_1_2/circ_dist)*360, 'degrees'
#    print 'Max error in longitude =', (max_1_2/circ_dist)*360-360, 'degrees'
#    
#    #plt.plot(x1x2_list)
#    #plt.plot(y1y2_list)
#    #plt.title('X2-X1 and Y2-Y1')
#    #plt.show()
#    
#    #plt.plot(e1e2_list)
#    #plt.title('dE/dM')
#    #plt.show()
#    
#    #plt.plot(dist_circ_circ_list)
#    #plt.plot(dist_1_2_list)
#    #plt.show()
#    