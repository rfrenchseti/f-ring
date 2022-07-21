import math
import os

import numpy as np

import cspyce

import f_ring_util

kdir = '/home/rfrench/DS/Shared/OOPS-Resources/SPICE'
cspyce.furnsh(os.path.join(kdir, 'General/LSK/naif0012.tls'))
cspyce.furnsh(os.path.join(kdir, 'General/SPK/de438.bsp'))
cspyce.furnsh(os.path.join(kdir, 'Saturn/SPK/sat393.bsp'))
cspyce.furnsh(os.path.join(kdir, 'General/PCK/pck00010_edit_v01.tpc'))

SATURN_ID     = cspyce.bodn2c('SATURN')
PANDORA_ID    = cspyce.bodn2c('PANDORA')
PROMETHEUS_ID = cspyce.bodn2c('PROMETHEUS')

REFERENCE_ET = cspyce.utc2et('2007-01-01') # For Saturn pole
j2000_to_iau_saturn = cspyce.pxform('J2000', 'IAU_SATURN', REFERENCE_ET)

saturn_z_axis_in_j2000 = cspyce.mtxv(j2000_to_iau_saturn, (0,0,1))
saturn_x_axis_in_j2000 = cspyce.ucrss((0,0,1), saturn_z_axis_in_j2000)

J2000_TO_SATURN = cspyce.twovec(saturn_z_axis_in_j2000, 3,
                                saturn_x_axis_in_j2000, 1)

def saturn_to_prometheus(et):
    (prometheus_j2000, lt) = cspyce.spkez(PROMETHEUS_ID, et, 'J2000', 'LT+S', SATURN_ID)
    prometheus_sat = np.dot(J2000_TO_SATURN, prometheus_j2000[0:3])
    dist = np.sqrt(prometheus_sat[0]**2.+prometheus_sat[1]**2.+prometheus_sat[2]**2.)
    longitude = math.atan2(prometheus_sat[1], prometheus_sat[0])
    return (dist, longitude)

def prometheus_close_approach(min_et, min_et_long):
    def compute_r(a, e, arg): # Takes argument of pericenter
        return a*(1-e**2.) / (1+e*np.cos(arg))
    def compute_r_fring(arg):
        return compute_r(f_ring_util.FRING_A, f_ring_util.FRING_E, arg)

    # Find time for 0 long
    et_min = min_et - min_et_long / f_ring_util.FRING_MEAN_MOTION * 86400.
    # Find time for 360 long
    et_max = min_et + 2*np.pi / f_ring_util.FRING_MEAN_MOTION * 86400
    # Find the longitude at the point of closest approach
    min_dist = 1e38
    for et in np.arange(et_min, et_max, 10): # Step by minute
        prometheus_dist, prometheus_longitude = saturn_to_prometheus(et)
        long_peri_fring = ((et-f_ring_util.FRING_ORBIT_EPOCH)/86400 *
                           f_ring_util.FRING_DW +
                           f_ring_util.FRING_W0) % (np.pi*2)
        fring_r = compute_r_fring(prometheus_longitude-long_peri_fring)
        if abs(fring_r-prometheus_dist) < min_dist:
            min_dist = abs(fring_r-prometheus_dist)
            min_dist_long = prometheus_longitude
            min_dist_et = et
    min_dist_long = f_ring_util.fring_inertial_to_corotating(min_dist_long, min_dist_et)
    return min_dist, min_dist_long
