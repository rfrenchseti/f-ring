'''
Created on Sep 19, 2011

@author: rfrench
'''

import numpy as np
import os
import pickle

import julian

def utc2et(s):
    return julian.tdb_from_tai(julian.tai_from_iso(s))

def et2utc(et, *args):
    # XXX Implement other args
    print(et)
    return 'Fake time'
    return julian.iso_from_tai(julian.tai_from_tdb(et))

# Linux
ROOT = '/media/shannon/clumps'
DATA_ROOT = '/media/shannon/clumps/data'
VOYAGER_PATH = '/media/shannon/Voyager/'

def add_parser_options(parser):
    # For file selection
    parser.add_argument('-a', '--all_obs', dest='all_obs', action='store_true', default=False)
    parser.add_argument('--radius_start', dest='radius_start', type=int, default=137500)
    parser.add_argument('--radius_end', dest='radius_end', type=int, default=142500)
    parser.add_argument('-r', '--radius_resolution', type=float, dest='radius_resolution', default=5.0)
    parser.add_argument('-l', '--longitude_resolution', type=float, dest='longitude_resolution',
                      default=0.02)
    parser.add_argument('--reproject-zoom-factor',
                      type=int, dest='reproject_zoom_factor', default=10,
                      metavar='ZOOM',
                      help='How much to inflate the original image before performing the reprojection')
    parser.add_argument('--mosaic-reduction-factor', dest='mosaic_reduction_factor',
                      type=int, default=2,
                      metavar='FACTOR',
                      help='How much to reduce the mosaic before displaying')
    parser.add_argument('--verbose', action='store_true', dest='verbose', default=False)

    parser.add_argument('--core_radius_start', dest='core_radius_start', type=int, default=137500)
    parser.add_argument('--core_radius_end', dest='core_radius_end', type=int, default=142500)
    parser.add_argument('--ignore-bad-obsids', action='store_true', dest='ignore_bad_obsids', default=False)
    parser.add_argument('--global-whitepoint', dest ='global_whitepoint', type=float, default=0.522407851536)
    parser.add_argument('--voyager', dest='voyager', action='store_true', default=False)
    parser.add_argument('--ignore-voyager', dest='ignore_voyager', action='store_true', default=False)
    parser.add_argument('--downsample', dest='downsample', action='store_true', default=False)

def ignored_obsids():
    bad_ids = []
    return bad_ids

def voyager_obsids():

    return ['V1I', 'V1O', 'V2I', 'V2O']

def enumerate_files(options, args, suffix='', obsid_only=False):
    bad_obsids = ignored_obsids()
    voyager_ids = voyager_obsids()
    if options.all_obs:
        dir_list = sorted(os.listdir(DATA_ROOT))
        file_list = []
        for dir in dir_list:
            if os.path.isdir(os.path.join(DATA_ROOT, dir)):
                file_list.append(dir)
    else:
        file_list = args

    for arg in file_list:
        if os.path.exists(arg): # Absolute path
            assert not obsid_only
            path, image_name = os.path.split(arg)
            assert file[0] == 'N' or file[0] == 'W'
            file = file[:11]
            path, obs_id = os.path.split(path)
            if options.ignore_bad_obsids and obs_id in bad_obsids:
                print('IGNORING EXPLICITLY SPECIFIED BUT BAD OBS_ID:', obs_id)
                continue
            if options.ignore_voyager and obs_id in voyager_ids:
                print('IGNORING VOYAGER FILES')
                continue
            yield obs_id, image_name, arg
        else:
            abs_path = os.path.join(DATA_ROOT, arg)
            if os.path.isdir(abs_path): # Observation ID
                if options.ignore_bad_obsids and arg in bad_obsids:
                    continue
                if options.ignore_voyager and arg in voyager_ids:
                    continue
                if obsid_only:
                    yield arg, None, None
                    continue
                filenames = sorted(os.listdir(abs_path))
                for filename in filenames:
                    full_path = os.path.join(DATA_ROOT, arg, filename)
                    if not os.path.isfile(full_path): continue
                    if filename[-len(suffix):] != suffix: continue
                    image_name = filename[:-len(suffix)]
                    yield arg, image_name, full_path

            else: # Single OBSID/IMAGENAME
                obs_id, image_name = os.path.split(arg)
                if options.ignore_bad_obsids and obs_id in bad_obsids:
                    print('IGNORING EXPLICITLY SPECIFIED BUT BAD OBS_ID:', obs_id)
                    continue
                if options.ignore_voyager and obs_id in voyager_ids:
                    print('IGNORING VOYAGER FILES')
                    continue
                abs_path += suffix
                yield obs_id, image_name, abs_path

def mosaic_paths(options, obsid):
    mosaic_res_data = ('_%06d_%06d_%06.3f_%05.3f_%02d' % (options.radius_start, options.radius_end,
                                                          options.radius_resolution,
                                                          options.longitude_resolution,
                                                          options.reproject_zoom_factor))
    data_path = os.path.join(ROOT, 'mosaic-data', obsid+mosaic_res_data+'-data')
    metadata_path = os.path.join(ROOT, 'mosaic-data', obsid+mosaic_res_data+'-metadata.pickle')
    large_png_path = os.path.join(ROOT, 'png', 'full-'+obsid+mosaic_res_data+'.png')
    small_png_path = os.path.join(ROOT, 'png', 'small-'+obsid+mosaic_res_data+'.png')
    return (data_path, metadata_path, large_png_path, small_png_path)

def mosaic_paths_spec(radius_start, radius_end, radius_resolution, longitude_resolution,
                      reproject_zoom_factor, obsid):
    mosaic_res_data = ('_%06d_%06d_%06.3f_%05.3f_%02d' % (radius_start, radius_end,
                                                          radius_resolution,
                                                          longitude_resolution,
                                                          reproject_zoom_factor))
    data_path = os.path.join(ROOT, 'mosaic-data', obsid+mosaic_res_data+'-data')
    metadata_path = os.path.join(ROOT, 'mosaic-data', obsid+mosaic_res_data+'-metadata.pickle')
    large_png_path = os.path.join(ROOT, 'png', 'full-'+obsid+mosaic_res_data+'.png')
    small_png_path = os.path.join(ROOT, 'png', 'small-'+obsid+mosaic_res_data+'.png')
    return (data_path, metadata_path, large_png_path, small_png_path)

def bkgnd_paths(options, obsid):
    bkgnd_res_data = ('_%06d_%06d_%06.3f_%05.3f_%02d_%02d' % (options.radius_start, options.radius_end,
                                                              options.radius_resolution,
                                                              options.longitude_resolution,
                                                              options.reproject_zoom_factor,
                                                              options.mosaic_reduction_factor))
    reduced_mosaic_data_filename = os.path.join(ROOT, 'bkgnd-data',
                                                obsid+bkgnd_res_data+'-data')
    reduced_mosaic_metadata_filename = os.path.join(ROOT, 'bkgnd-data',
                                                    obsid+bkgnd_res_data+'-metadata.pickle')
    bkgnd_mask_filename = os.path.join(ROOT, 'bkgnd-data',
                                       obsid+bkgnd_res_data+'-bkgnd-mask')
    bkgnd_model_filename = os.path.join(ROOT, 'bkgnd-data',
                                        obsid+bkgnd_res_data+'-bkgnd-model')
    bkgnd_metadata_filename = os.path.join(ROOT, 'bkgnd-data',
                                           obsid+bkgnd_res_data+'-bkgnd-metadata.pickle')
    if options.mosaic_reduction_factor == 1:
        data_path, metadata_path, large_png_path,small_png_path = mosaic_paths(options, obsid)
        return(data_path, metadata_path, bkgnd_mask_filename, bkgnd_model_filename, bkgnd_metadata_filename)
    else:
        return (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
                bkgnd_mask_filename, bkgnd_model_filename, bkgnd_metadata_filename)

def bkgnd_paths_spec(radius_start, radius_end, radius_resolution, longitude_resolution,
                     reproject_zoom_factor, mosaic_reduction_factor, obsid):
    bkgnd_res_data = ('_%06d_%06d_%06.3f_%05.3f_%02d_%02d' % (radius_start, radius_end,
                                                              radius_resolution,
                                                              longitude_resolution,
                                                              reproject_zoom_factor,
                                                              mosaic_reduction_factor))
    reduced_mosaic_data_filename = os.path.join(ROOT, 'bkgnd-data',
                                                obsid+bkgnd_res_data+'-data')
    reduced_mosaic_metadata_filename = os.path.join(ROOT, 'bkgnd-data',
                                                    obsid+bkgnd_res_data+'-metadata.pickle')
    bkgnd_mask_filename = os.path.join(ROOT, 'bkgnd-data',
                                       obsid+bkgnd_res_data+'-bkgnd-mask')
    bkgnd_model_filename = os.path.join(ROOT, 'bkgnd-data',
                                        obsid+bkgnd_res_data+'-bkgnd-model')
    bkgnd_metadata_filename = os.path.join(ROOT, 'bkgnd-data',
                                           obsid+bkgnd_res_data+'-bkgnd-metadata.pickle')

    return (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
            bkgnd_mask_filename, bkgnd_model_filename, bkgnd_metadata_filename)

def ew_paths(options, obsid):
    ew_res_data = ('_%06d_%06d_%06.3f_%05.3f_%02d_%02d' % (options.radius_start, options.radius_end,
                                                           options.radius_resolution,
                                                           options.longitude_resolution,
                                                           options.reproject_zoom_factor,
                                                           options.mosaic_reduction_factor))
    ew_data_filename = os.path.join(ROOT, 'ew-data',
                                    obsid+ew_res_data+'-data' +
                                    '_%06d_%06d' % (options.core_radius_start, options.core_radius_end))
    ew_mask_filename = os.path.join(ROOT, 'ew-data',
                                    obsid+ew_res_data+'-mask' +
                                    '_%06d_%06d' % (options.core_radius_start, options.core_radius_end))
    return (ew_data_filename, ew_mask_filename)

def clumpdb_paths(options):
    cl_res_data = ('_%06d_%06d_%06.3f_%05.3f_%02d_%02d_%06d_%06d' %
                   (options.radius_start, options.radius_end,
                    options.radius_resolution,
                    options.longitude_resolution,
                    options.reproject_zoom_factor,
                    options.mosaic_reduction_factor,
                    options.core_radius_start,
                    options.core_radius_end))
    cl_data_filename = os.path.join(ROOT, 'clump-data',
                                    'clumpdb'+cl_res_data+'.pickle')
    cc_data_filename = os.path.join(ROOT, 'clump-data',
                                     'clumpchains'+cl_res_data+'.pickle')
    if options.voyager:
        cl_data_filename = os.path.join(ROOT, 'clump-data',
                                        'voyager_clumpdb'+cl_res_data+'.pickle')
        cc_data_filename = os.path.join(ROOT, 'clump-data',
                                        'voyager_clumpchains'+cl_res_data+'.pickle')
    if options.downsample:
        cl_data_filename = os.path.join(ROOT, 'clump-data',
                                        'downsampled_clumpdb'+cl_res_data+'.pickle')
        cc_data_filename = os.path.join(ROOT, 'clump-data',
                                        'downsampled_clumpchains'+cl_res_data+'.pickle')
    return cl_data_filename, cc_data_filename

ROTATING_ET = utc2et("2007-01-01")
FRING_MEAN_MOTION = 581.964

def ComputeLongitudeShift(img_ET):
    return - (FRING_MEAN_MOTION * ((img_ET - ROTATING_ET) / 86400.)) % 360.

def InertialToCorotating(longitude, ET):
    return (longitude + ComputeLongitudeShift(ET)) % 360.

def CorotatingToInertial(co_long, ET):
    return (co_long - ComputeLongitudeShift(ET)) % 360.

def CorotatingToTrueAnomaly(co_long, ET):
    return (co_long - ComputeLongitudeShift(ET) - 2.7007*(ET/86400.)) % 360.

# Convert a rate of clump motion (in deg/sec) to a semi-major axis (in km)
# Assume the F ring is centered on 140220 km with a rotation rate of FRING_MEAN_MOTION
def RelativeRateToSemimajorAxis(rate):
    return ((FRING_MEAN_MOTION / (FRING_MEAN_MOTION+rate*86400.))**(2./3.) * 140221.3)

def prometheus_close_approach(min_et, min_et_long):
    def compute_r(a, e, arg): # Takes argument of pericenter
        return a*(1-e**2.) / (1+e*np.cos(arg))
    def compute_r_fring(arg):
        return compute_r(bosh2002_fring_a, bosh2002_fring_e, arg)

    bosh2002_epoch_et = utc2et('JD 2451545.0') # J2000
    bosh2002_fring_a = 140223.7
    bosh2002_fring_e = 0.00254
    bosh2002_fring_curly = 24.1 * np.pi/180
    bosh2002_fring_curly_dot = 2.7001 * np.pi/180 / 86400 # rad/sec

    # Find time for 0 long
    et_min = min_et - min_et_long / FRING_MEAN_MOTION * 86400.
    # Find time for 360 long
    et_max = min_et + 360. / FRING_MEAN_MOTION * 86400
    # Find the longitude at the point of closest approach
    min_dist = 1e38
    for et in np.arange(et_min, et_max, 60): # Step by minute
        prometheus_dist, prometheus_longitude = ringimage.saturn_to_prometheus(et)
        prometheus_longitude = CorotatingToInertial(prometheus_longitude, et)
        long_peri_fring = (et-bosh2002_epoch_et) * bosh2002_fring_curly_dot + bosh2002_fring_curly
        fring_r = compute_r_fring(prometheus_longitude-long_peri_fring)
        if abs(fring_r-prometheus_dist) < min_dist:
            min_dist = abs(fring_r-prometheus_dist)
            min_dist_long = prometheus_longitude
            min_dist_et = et
    min_dist_long = InertialToCorotating(min_dist_long, min_dist_et)
    return min_dist, min_dist_long

def compute_mu(e):
    if type(e) == type([]):
        e = np.array(e)
    return np.abs(np.cos(e*np.pi/180.))

def compute_z(mu, mu0, tau, is_transmission):
    transmission_list = tau*(mu-mu0)/(mu*mu0*(np.exp(-tau/mu)-np.exp(-tau/mu0)))
    reflection_list = tau*(mu+mu0)/(mu*mu0*(1-np.exp(-tau*(1/mu+1/mu0))))
    ret = np.where(is_transmission, transmission_list, reflection_list)
    return ret

# This takes EW * mu
def compute_corrected_ew(ew, emission, incidence, tau=0.034):
    if type(emission) == type([]):
        emission = np.array(emission)
    if type(incidence) == type([]):
        incidence = np.array(incidence)
    is_transmission = emission > 90.
    mu = compute_mu(emission)
    mu0 = np.abs(np.cos(incidence*np.pi/180))
    ret = ew * compute_z(mu, mu0, tau, is_transmission)
    return ret

def clump_phase_curve(phase_angles):
    coeffs = np.array([6.09918565e-07, -8.81293896e-05, 5.51688159e-03, -3.29583781e-01])

    return 10**np.polyval(coeffs, phase_angles)

def mu(emission):
    return np.abs(np.cos(emission*np.pi/180.))

def transmission_function(tau, emission, incidence):
    assert False
    #incidence angle is always less than 90, therefore the only case we have to worry about it when Emission Angle changes.
    #E > 90 = Transmission, E < 90 = Reflection
    mu0 = np.abs(np.cos(incidence*np.pi/180.))
    mu = np.abs(np.cos(emission*np.pi/180.))

    if np.mean(emission[np.nonzero(emission)]) > 90:
#        print 'Transmission'
        return mu * mu0 * (np.exp(-tau/mu)-np.exp(-tau/mu0)) / (tau * (mu-mu0))
    elif np.mean(emission[np.nonzero(emission)]) < 90:
#        print 'Reflection'
        return mu * mu0 * (1.-np.exp(-tau*(1/mu+1/mu0))) / (tau * (mu+mu0))


def normalized_ew_factor(alpha, emission, incidence):
    assert False
    tau_eq = 0.033 #French et al. 2012
    return mu(emission)/(transmission_function(tau_eq, emission, incidence)*normalized_phase_curve(alpha))
