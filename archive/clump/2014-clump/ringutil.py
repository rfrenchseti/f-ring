'''
Created on Sep 19, 2011

@author: rfrench
'''

from optparse import OptionParser
import os
import os.path
import ringimage # Just to initialize CSPICE
import cspice
import pickle
import numpy as np
import numpy.ma as ma

if os.getcwd()[1] == ':':
    # Windows
    PYTHON_EXE = 'c:/Users/rfrench/AppData/Local/Enthought/Canopy/User/python.exe'
    ROOT = 'T:/clumps'
    DATA_ROOT = 'T:/clumps/data'
    VOYAGER_PATH = 'T:/clumps/voyager'
    PAPER_ROOT = 'T:/clumps/paper'
else:
    # Linux
    PYTHON_EXE = '/usr/bin/python'
    ROOT = '/media/shannon/clumps'
    DATA_ROOT = '/media/shannon/clumps/data'
    VOYAGER_PATH = '/media/shannon/Voyager/'
    PAPER_ROOT = '/home/shannon/Paper'

PYTHON_RING_REPROJECT = 'ring_reproject.py'
PYTHON_RING_MOSAIC = 'ring_mosaic.py'
PYTHON_RING_BKGND = 'ring_bkgnd.py'
PYTHON_RING_RADIAL_REPROJECT = 'ring_radial_reproject.py'

SUFFIX_CALIB = '_CALIB.IMG'
#VOYAGER_PATH = '/media/shannon/Voyager/'

class OffRepData(object):
    """Offset and Reprojection data."""
    validattr = ('obsid',                                   # The OBSID
                 'image_name',                              # 
                 'image_path',                              # 
                 'offset_path',                             #
                 'repro_path',                              #
                 'zoomarray',                               #
                 'vicar_data',                              #
                 'ringim',                                  #
                 'the_offset',                              #
                 'manual_offset',                           #
                 'fring_number',                            #
                 'repro_img',                               #
                 'img'                                      #
                )
    userattr = []  # More attributes can be added by external users
    
    def __init__(self):
        for attr in list(OffRepData.validattr)+OffRepData.userattr:
            self.__setattr__(attr, None)

    def __setattr__(self, name, value):
        assert name in OffRepData.validattr or name in OffRepData.userattr
        self.__dict__[name] = value

class MosaicData(object):
    """Mosaic metadata."""
    validattr = ('data_path',                               #
                 'metadata_path',                           #
                 'large_png_path',                          #
                 'small_png_path',                          #
                 'obsid',                                   # The OBSID
                 'obsid_list',                              # 
                 'image_name_list',                         # 
                 'image_path_list',                         #
                 'repro_path_list',                         #
                 'img',                                     #
                 'longitudes',                              #
                 'resolutions',                             #
                 'image_numbers',                           #
                 'ETs',                                     #
                 'emission_angles',                         #
                 'incidence_angles',                        #
                 'phase_angles'                             #
                )
    userattr = []  # More attributes can be added by external users
    
    def __init__(self):
        for attr in list(MosaicData.validattr)+MosaicData.userattr:
            self.__setattr__(attr, None)

    def __setattr__(self, name, value):
        assert name in MosaicData.validattr or name in MosaicData.userattr
        self.__dict__[name] = value

class BkgndData(object):
    """Background metadata."""
    validattr = ('obsid',                                   #
                 'mosaic_data_filename',                    #
                 'mosaic_metadata_filename',                #
                 'reduced_mosaic_data_filename',            #
                 'reduced_mosaic_metadata_filename',        #
                 'bkgnd_mask_filename',                     #
                 'bkgnd_model_filename',                    #
                 'bkgnd_metadata_filename',                 #
                 'mosaic_img',                              # 
                 'mosaic_data',                             # 
                 'obsid_list',                              # 
                 'image_name_list',                         #
                 'full_filename_list',                      #
                 'longitudes',                              #
                 'resolutions',                             #
                 'image_numbers',                           #
                 'ETs',                                     #
                 'emission_angles',                         #
                 'incidence_angles',                        #
                 'phase_angles',                             #
                 'bkgnd_model',
                 'bkgnd_model_mask',                        # Either the bkgnd_model's mask, or the getmaskarray mask
                 'correct_mosaic_image',
                 'row_cutoff_sigmas',
                 'row_ignore_fraction',
                 'row_blur',
                 'ring_lower_limit',
                 'ring_upper_limit',
                 'column_cutoff_sigmas',
                 'column_inside_background_pixels',
                 'column_outside_background_pixels',
                 'degree',
                 'ew_mean',
                 'ew_std',
                 'ewmu_mean',
                 'ewmu_std',
                 'corrected_mosaic_img'
                )
    userattr = []  # More attributes can be added by external users
    
    def __init__(self):
        for attr in list(BkgndData.validattr)+BkgndData.userattr:
            self.__setattr__(attr, None)

    def __setattr__(self, name, value):
        assert name in BkgndData.validattr or name in BkgndData.userattr
        self.__dict__[name] = value

def add_parser_options(parser):
    # For file selection
    parser.add_option('-a', '--all_obs', dest='all_obs', action='store_true', default=False)
    parser.add_option('--radius_start', dest='radius_start', type='int', default=137500)
    parser.add_option('--radius_end', dest='radius_end', type='int', default=142500)
    parser.add_option('-r', '--radius_resolution', type='float', dest='radius_resolution', default=5.0)
    parser.add_option('-l', '--longitude_resolution', type='float', dest='longitude_resolution',
                      default=0.02)
    parser.add_option('--reproject-zoom-factor',
                      type='int', dest='reproject_zoom_factor', default=10,
                      metavar='ZOOM',
                      help='How much to inflate the original image before performing the reprojection')
    parser.add_option('--mosaic-reduction-factor', dest='mosaic_reduction_factor',
                      type='int', default=2,
                      metavar='FACTOR',
                      help='How much to reduce the mosaic before displaying')
    parser.add_option('--verbose', action='store_true', dest='verbose', default=False)
    
    parser.add_option('--core_radius_start', dest='core_radius_start', type='int', default=137500)
    parser.add_option('--core_radius_end', dest='core_radius_end', type='int', default=142500)
    parser.add_option('--ignore-bad-obsids', action='store_true', dest='ignore_bad_obsids', default=False)
    parser.add_option('--global-whitepoint', dest = 'global_whitepoint', type = 'float', default = 0.522407851536)
    parser.add_option('--voyager', dest = 'voyager', action = 'store_true', default = False)
    parser.add_option('--ignore-voyager', dest = 'ignore_voyager', action = 'store_true', default = False)
    parser.add_option('--downsample', dest = 'downsample', action = 'store_true', default = False)

def ignored_obsids():
    bad_ids = [              
               'ISS_068RF_FMOVIE001_VIMS',                  #bad coverage
               'ISS_081RI_SPKMVLFLP001_PRIME',              
               'ISS_085RI_SPKMVLFLP001_PRIME',
               'ISS_088RI_SPKMVLFLP001_PRIME',
               'ISS_096RI_TMAPN20LP001_CIRS',
               'ISS_104RI_TMAPS30LP001_CIRS',
               'ISS_118RI_PROPELLR001_PRIME']
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
                print 'IGNORING EXPLICITLY SPECIFIED BUT BAD OBS_ID:', obs_id
                continue
            if options.ignore_voyager and obs_id in voyager_ids:
                print 'IGNORING VOYAGER FILES'
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
                    print 'IGNORING EXPLICITLY SPECIFIED BUT BAD OBS_ID:', obs_id
                    continue
                if options.ignore_voyager and obs_id in voyager_ids:
                    print 'IGNORING VOYAGER FILES'
                    continue
                abs_path += suffix
                yield obs_id, image_name, abs_path

def offset_path(options, image_path, image_name):
    return image_path + '.OFFSET'

def repro_path(options, image_path, image_name):
    repro_res_data = ('_%06d_%06d_%06.3f_%05.3f_%02d' % (options.radius_start, options.radius_end,
                                                         options.radius_resolution,
                                                         options.longitude_resolution,
                                                         options.reproject_zoom_factor))
    return os.path.join(os.path.dirname(image_path), image_name + repro_res_data + '_REPRO.IMG')

def repro_path_spec(radius_start, radius_end, radius_resolution, longitude_resolution,
                    reproject_zoom_factor, image_path, image_name):
    repro_res_data = ('_%06d_%06d_%06.3f_%05.3f_%02d' % (radius_start, radius_end,
                                                         radius_resolution,
                                                         longitude_resolution,
                                                         reproject_zoom_factor))
    return os.path.join(os.path.dirname(image_path), image_name + repro_res_data + '_REPRO.IMG')

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

def draw_line(img, color, x0, y0, x1, y1):
    # Draw a single line in img
    # Implementation of Bresenham's line algorithm
    dx = abs(x1-x0)
    dy = abs(y1-y0) 
    if x0 < x1:
        sx = 1
    else:
        sx = -1
    if y0 < y1:
        sy = 1
    else:
        sy = -1
    err = dx-dy
 
    while True:
        img[y0, x0, :] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2*err
        if e2 > -dy: 
            err = err - dy
            x0 = x0 + sx
        if e2 < dx: 
            err = err + dx
            y0 = y0 + sy 

def draw_lines(img, color, pixels):
    if pixels.shape[0] < 2:
        return
    # Draw multiple lines in img
    py = pixels[0,0]
    px = pixels[0,1]
    for i in range(1, len(pixels)):
        draw_line(img, color, px, py, pixels[i,1], pixels[i,0])
        py = pixels[i,0]
        px = pixels[i,1]

ROTATING_ET = cspice.utc2et("2007-1-1")
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

OFFSET_FILE_VERSION = 2

def read_offset(offset_path):
    if not os.path.exists(offset_path):
        return -1, None, None, -1 
    offset_pickle_fp = open(offset_path, 'rb')
    offset_file_version = pickle.load(offset_pickle_fp)
    
    if type(offset_file_version) == type(()):
        # OLD VERSION BEFORE VERSIONING
        the_offset, old_manual_offset = offset_file_version
        offset_pickle_fp.close()
        fring_version = 0
        if type(the_offset) == type(np.array(None)):
            if the_offset.shape == ():
                the_offset = None
        return 0, the_offset, old_manual_offset, fring_version
    
    if offset_file_version == 1:
        the_offset = pickle.load(offset_pickle_fp)
        fring_version = pickle.load(offset_pickle_fp)
        manual_offset = None
        offset_pickle_fp.close()
    else:
        assert offset_file_version == OFFSET_FILE_VERSION
        the_offset = pickle.load(offset_pickle_fp)
        fring_version = pickle.load(offset_pickle_fp)
        manual_offset = pickle.load(offset_pickle_fp)
        offset_pickle_fp.close()
        
    return offset_file_version, the_offset, manual_offset, fring_version

def write_offset(offset_path, the_offset, manual_offset, fring_version):
    offset_pickle_fp = open(offset_path, 'wb')
    pickle.dump(OFFSET_FILE_VERSION, offset_pickle_fp)
    pickle.dump(the_offset, offset_pickle_fp)
    pickle.dump(fring_version, offset_pickle_fp)
    pickle.dump(manual_offset, offset_pickle_fp)    
    offset_pickle_fp.close()

def prometheus_close_approach(min_et, min_et_long):
    def compute_r(a, e, arg): # Takes argument of pericenter
        return a*(1-e**2.) / (1+e*np.cos(arg))
    def compute_r_fring(arg):
        return compute_r(bosh2002_fring_a, bosh2002_fring_e, arg)

    bosh2002_epoch_et = cspice.utc2et('JD 2451545.0') # J2000
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

#def normalized_phase_curve(alpha):
#    return np.exp((6.56586808e-07*(alpha**3)-9.25559440e-05*(alpha**2)+5.08638514e-03*alpha-2.76364092e-01))

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
