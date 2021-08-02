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
    
def RelativeRateToSemimajorAxis(rate):  # from ringutil
    return ((FRING_MEAN_MOTION / (FRING_MEAN_MOTION+rate*86400.))**(2./3.) * fring_a)
    
    
print '*** BASIC ORBITAL PARAMETERS ***'
print 'FRING a =', fring_a, 'km'
print 'FRING e =', fring_e
print 'FRING i =', fring_i, 'deg'
print 'FRING w-dot =', fring_wdot, 'deg/day'
print 'FRING omega-dot =', fring_odot, 'deg/day'
print 'SATURN GM =', saturn_gm, 'km^3/s^2'
print

fring_T = orbit_period(saturn_gm, fring_a)
fring_T_days = fring_T/3600.
deg_per_day = 24./fring_T_days * 360.

print '*** DERIVED ORBITAL PARAMETERS ***'
print 'FRING period T =', fring_T, 'secs =', fring_T_days, 'days'
print 'FRING mean motion =', deg_per_day, 'deg/day'
print 'FRING periapse =', fring_a_peri, 'km'
print 'FRING apoapse =', fring_a_apo, 'km'
print 'FRING apoapse-periapse =', fring_a_apo-fring_a_peri, 'km'

fring_T_360 = fring_T/360.

v_circ = orbit_mean_v(fring_a, fring_T, fring_e)

print 'FRING mean v (order e^2) =', v_circ, 'km/sec'

v_max = orbit_v(saturn_gm, fring_a, fring_a_peri)
v_min = orbit_v(saturn_gm, fring_a, fring_a_apo)

print 'FRING min v =', v_min, 'km/sec'
print 'FRING max v =', v_max, 'km/sec'
print 'FRING maxv/minv =', (v_max-v_min)/v_min*100, '%'

print

# Recalculate fring_T because the value we compute from first principles (GM_Saturn, fring_a) doesn't
# agree with the empirical value of 581.979 deg/day

fring_T = 360. / FRING_MEAN_MOTION * 86400.

fring_a = 140220.
fring_e = 0.00235

num_orbits_diff = 2

# (a,e)
#clump_specs = [(10, .1)]
clump_specs = [#(fring_a, fring_e),     # Clump perfectly in the core
#               (fring_a, 0.00235),
#               (fring_a, fring_e - 0.000713),           # 
               (fring_a+100, fring_e),
               (fring_a, fring_e + 0.000713)]

#difference in longitude of pericenter from the core's. Implies a different rate of precession.
dw = [0.] # np.arange(0., 190., 10)*np.pi/180.

for w in dw:
    print 'Relative Change in Long of Pericenter:', w*180/np.pi
    for n, (clump_a, clump_e) in enumerate(clump_specs):
        print 'Clump spec #', n, 'w, ', w

        # Compute the orbital period of the clump
        clump_T = fring_T * (clump_a/fring_a)**(3./2.)
        
        t0_list = []
        t1_list = []
        M_clump_t0_list = []
        M_clump_t1_list = []
        E_clump_t0_list = []
        E_clump_t1_list = []
        
        for t0_deg in np.arange(0., 360., 10.):  # The location of the clump at t0 in degrees
            t0 = clump_T / 360. * t0_deg

            M_clump_t0 = (t0_deg * np.pi/180) % (2*np.pi)
            E_clump_t0 = find_E(M_clump_t0, clump_e)
                        
            for t1_deg in np.arange(0., 360., 10.):  # The location of the clump at t1 in degrees
                t1 = clump_T / 360. * t1_deg + num_orbits_diff*clump_T
            
                if t1 <= t0:
                    continue
                
                M_clump_t1 = ((t1_deg + num_orbits_diff*360.) * np.pi/180) % (2*np.pi)
                E_clump_t1 = find_E(M_clump_t1, clump_e)
             
                t0_list.append(t0)
                t1_list.append(t1)
                M_clump_t0_list.append(M_clump_t0)
                M_clump_t1_list.append(M_clump_t1)
                E_clump_t0_list.append(E_clump_t0)
                E_clump_t1_list.append(E_clump_t1)

        t0_list = np.array(t0_list)
        t1_list = np.array(t1_list)                        
        M_clump_t0_list = np.array(M_clump_t0_list)
        M_clump_t1_list = np.array(M_clump_t1_list)
        E_clump_t0_list = np.array(E_clump_t0_list)
        E_clump_t1_list = np.array(E_clump_t1_list)

        # where the clump actually is in space relative to focus - x and y along the ellipse
        x_clump_t0_list, y_clump_t0_list = xy_from_E(clump_a, clump_e, E_clump_t0_list) 
        x_clump_t1_list, y_clump_t1_list = xy_from_E(clump_a, clump_e, E_clump_t1_list)
#        plt.plot(x_clump_t0_list, y_clump_t0_list, 'o', color='none', mec='red', ms=15)
#        plt.plot(x_clump_t1_list, y_clump_t1_list, 'o', color='black', mec='black', ms=8)
#        plt.show()
        
        # where we reproject the clump to be in the core on a straight line to the focus
        E_clump_core_t0_list = E_from_xy(x_clump_t0_list, y_clump_t0_list, fring_a, fring_e)
        E_clump_core_t1_list = E_from_xy(x_clump_t1_list, y_clump_t1_list, fring_a, fring_e)
        
        # find Mean Anomalies for the reprojected core position
        M_clump_core_t0_list = M_from_E(E_clump_core_t0_list, fring_e)*180/np.pi
        M_clump_core_t1_list = M_from_E(E_clump_core_t1_list, fring_e)*180/np.pi

#        print 'E_clump_core_t0'
#        print E_clump_core_t0_list[:10]*180/np.pi
#        print 'E_clump_core_t1'
#        print E_clump_core_t1_list[:10]*180/np.pi
#        print 'M_clump_core_t0'
#        print M_clump_core_t0_list[:10]
#        print 'M_clump_core_t1'
#        print M_clump_core_t1_list[:10]

        # Make the mean anomalies relative to the corotating ring
        M_clump_core_t0_list = np.round((M_clump_core_t0_list - (t0_list/86400.*FRING_MEAN_MOTION)%360.),6) % 360.
        M_clump_core_t1_list = np.round((M_clump_core_t1_list - (t1_list/86400.*FRING_MEAN_MOTION)%360.),6) % 360.

        # At t0 we see the clump at M_clump_core_t0; At t1 we see the clump at M_clump_core_t1
        # The relative motion is M_clump_core_t1 - M_clump_core_t0
        # The relative RATE is (M_clump_core_t1 - M_clump_core_t0) / (t1-t0)
        
        m_relative = M_clump_core_t1_list - M_clump_core_t0_list
        m_relative = ((m_relative + 180)%360) - 180  # This deals with wrap-around where 359 deg should be -1 deg
        print 'MOTION OF CLUMP IN CORE REF FRAME'
#        print m_relative
        print 'MIN', np.min(m_relative), 'MAX', np.max(m_relative)
        plt.plot(m_relative, '.', ms=8, color='black')
        plt.title('Clump a=%9.3f; e=%9.7f; orbits=%d'%(clump_a, clump_e, num_orbits_diff))
        plt.ylabel('Motion relative to co-rot (deg)')
        plt.xlabel('Trial')
        plt.show()
#        print ((t1_list-t0_list)/clump_T * 360.) % 360.
        m_relative_rate = m_relative / (t1_list-t0_list)  # In deg/sec
        m_relative_rate_days = m_relative_rate * 86400.
        
        derived_clump_a = RelativeRateToSemimajorAxis(m_relative_rate)
#        print 'DERIVED_CLUMP_A'
#        print derived_clump_a
        clump_a_error = derived_clump_a - clump_a
        plt.plot(clump_a_error, '.', ms=2, color='black')
        plt.title('Clump a=%9.3f; e=%9.7f; orbits=%d'%(clump_a, clump_e, num_orbits_diff))
        plt.ylabel('Error (km)')
        plt.xlabel('Trial')
        plt.show()

