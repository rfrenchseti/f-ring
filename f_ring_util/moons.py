import math
import os

import numpy as np

import cspyce

import f_ring_util.f_ring as f_ring

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
    longitude = np.degrees(math.atan2(prometheus_sat[1], prometheus_sat[0]))
    return (dist, longitude)


def saturn_to_pandora(et):
    (pandora_j2000, lt) = cspyce.spkez(PANDORA_ID, et, 'J2000', 'LT+S', SATURN_ID)
    pandora_sat = np.dot(J2000_TO_SATURN, pandora_j2000[0:3])
    dist = np.sqrt(pandora_sat[0]**2.+pandora_sat[1]**2.+pandora_sat[2]**2.)
    longitude = np.degrees(math.atan2(pandora_sat[1], pandora_sat[0]))
    return (dist, longitude)


def _close_approach(min_et, max_et, dist_func):
    if max_et is None:
        max_et = min_et + 360 / f_ring.FRING_MEAN_MOTION * 86400
    # Find the longitude and distance at the point of closest approach
    min_dist = 1e38
    for et in np.arange(min_et, max_et+59, 60): # Step by minute
        saturn_dist, longitude = dist_func(et)
        fring_r = f_ring.fring_radius_at_longitude(longitude, et)
        if abs(fring_r - saturn_dist) < min_dist:
            min_dist = abs(fring_r - saturn_dist)
            min_dist_long = longitude
            min_dist_et = et
    min_dist_long = f_ring.fring_inertial_to_corotating(min_dist_long, min_dist_et)
    return min_dist, min_dist_long


def pandora_close_approach(min_et, max_et=None):
    """Find the distance and longitude of Pandora's closest approach.

    If max_et is not specified, we use one entire orbit starting at min_et.
    """
    return _close_approach(min_et, max_et, saturn_to_pandora)


def prometheus_close_approach(min_et, max_et=None):
    """Find the distance and longitude of Prometheus's closest approach.

    If max_et is not specified, we use one entire orbit starting at min_et.
    """
    return _close_approach(min_et, max_et, saturn_to_prometheus)
