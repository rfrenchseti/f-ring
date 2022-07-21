import numpy as np
import os
import pickle

import scipy.optimize as sciopt

import julian

# These environment variables only need to be set if they are going to be used
BKGND_SUB_MOSAIC_DIR = os.environ.get('BKGND_SUB_MOSAIC_DIR', None)
EW_DIR = os.environ.get('EW_DIR', None)
POLAR_PNG_DIR = os.environ.get('POLAR_PNG_DIR', None)

if BKGND_SUB_MOSAIC_DIR is not None:
    BKGND_SUB_MOSAIC_DIR = BKGND_SUB_MOSAIC_DIR.rstrip('/')
if EW_DIR is not None:
    EW_DIR = EW_DIR.rstrip('/')
if POLAR_PNG_DIR is not None:
    POLAR_PNG_DIR = POLAR_PNG_DIR.rstrip('/')

TWOPI = np.pi*2


# Mosaic metadata is a dictionary with the following keys:
#   ring_lower_limit        An integer indicating the lowest radial index in the
#                           mosaic that is part of the image and not the
#                           background.
#
#   ring_upper_limit        An integer indicating the highest radial index in
#                           the mosaic that is part of the image and not the
#                           background.
#
#   radius_resolution       A floating point number indicating the radial
#                           resolution (km/pixel)
#
#   longitude_resolution    A floating point number indicating the longitude
#                           resolution (radians/pixel)
#
#   incidence_angle         A floating point number indicating the incidence
#                           angle (in radians) at the time of the mosaic.
#
# All following metadata keys contain an array with one entry per co-rotating
# longitude (there are 2PI / longitude_resolution entries).
#
#   long_mask               True if this co-rotating longitude contains any
#                           valid data.
#
#   ETs                     Ephemeris time (in seconds since noon on
#                           January 1, 2000) of the image used to create
#                           each co-rotating longitude.
#
#   emission_angles         Mean emission angle (in radians) of all the
#                           emission angles for the radial positions in the
#                           source image for each co-rotating longitude.
#
#   phase_angles            Mean phase angle (in radians) of all the
#                           phase angles for the radial positions in the
#                           source image for each co-rotating longitude.
#
#   resolutions             Mean radial resolution (in km/pixel) of all the
#                           radial resolutions for the radial positions in the
#                           source image for each co-rotating longitude.
#
#   longitudes              The co-rotating longitude for each longitude
#                           position. <0 means invalid data.
#
#   image_numbers           Integers indicating which source image this
#                           longitude's data came from. These integers are
#                           indexes into the following four arrays.
#     image_name_list       The base names of the images used to create the
#                           mosaic (e.g. N1600258137_1)
#     obsid_list            The Cassini observation name corresponding to each
#                           image.
#     image_path_list       The full path of the image on the computer used
#                           to create the moasic.
#     repro_path_list       The full path of the reprojected image file on the
#                           computer used to create the mosaic.


################################################################################
#
# GENERAL UTILITIES
#
################################################################################

def utc2et(s):
    return julian.tdb_from_tai(julian.tai_from_iso(s))

def et2utc(et):
    return julian.iso_from_tai(julian.tai_from_tdb(et))

def file_clean_join(*args):
    ret = os.path.join(*args)
    return ret.replace('\\', '/')


################################################################################
#
# PARSING ARGUMENTS AND ENUMERATING OBSIDS
#
################################################################################

def add_parser_arguments(parser):
    parser.add_argument(
        'obsid', action='append', nargs='*',
        help='Specific OBSIDs to process')
    parser.add_argument(
        '--ring-type', default='FMOVIE',
        help='The type of ring mosaics; use to retrieve the file lists')
    parser.add_argument(
        '--corot-type', default='',
        help='The type of co-rotation frame to use')
    parser.add_argument(
        '--start-obsid', default='',
        help='The first obsid to process')
    parser.add_argument(
        '--end-obsid', default='',
        help='The last obsid to process')
    parser.add_argument(
        '--ring-radius', type=int, default=140220,
        help='The main ring radius')
    parser.add_argument(
        '--radius-inner-delta', type=int, default=-1000,
        help="""The inner delta from the main ring radius""")
    parser.add_argument(
        '--radius-outer-delta', type=int, default=1000,
        help="""The outer delta from the main ring radius""")
    parser.add_argument(
        '--radius-resolution', type=float, default=5.,
        help='The radial resolution for reprojection')
    parser.add_argument(
        '--longitude-resolution', type=float, default=0.02,
        help='The longitudinal resolution for reprojection')
    parser.add_argument(
        '--radial-zoom-amount', type=int, default=10,
        help='The amount of radial zoom for reprojection')
    parser.add_argument(
        '--longitude-zoom-amount', type=int, default=1,
        help='The amount of longitude zoom for reprojection')
    parser.add_argument('--verbose', action='store_true', default=False)

# Given a directory, this returns a list of all the filenames in that directory
# that end with ".npy", but with that suffix stripped. We call that
# stripped filename a "root" filename.
def get_root_list(dir):
    root_list = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in sorted(filenames):
            if filename.endswith('.npy') and filename.find('WIDTH') == -1:
                root_list.append(os.path.join(dir, filename.replace('.npy', '')))
    root_list.sort()
    return root_list

def enumerate_obsids(arguments):
    data_path, _ = bkgnd_sub_mosaic_paths(arguments, '', make_dirs=False)
    bp = data_path.replace(BKGND_SUB_MOSAIC_DIR+'/', '')
    for dirpath, dirnames, filenames in os.walk(BKGND_SUB_MOSAIC_DIR):
        for filename in sorted(filenames):
            if filename.endswith('.npz'):
                ind = filename.find(bp)
                if ind < 0:
                    continue
                obs_id = filename[:ind]
                if (len(arguments.obsid) > 0 and len(arguments.obsid[0]) > 0 and
                    obs_id not in arguments.obsid[0]):
                    continue
                if arguments.start_obsid and arguments.start_obsid > obs_id:
                    continue
                if arguments.end_obsid and arguments.end_obsid < obs_id:
                    continue
                yield obs_id


################################################################################
#
# COMPUTING PATHS
#
################################################################################

def bkgnd_sub_mosaic_paths_spec(ring_radius, radius_inner, radius_outer,
                                radius_resolution, longitude_resolution,
                                radial_zoom_amount, longitude_zoom_amount,
                                obsid, ring_type, make_dirs=False):
    bkgnd_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d_1' % (
                      ring_radius, radius_inner, radius_outer,
                      radius_resolution, longitude_resolution,
                      radial_zoom_amount, longitude_zoom_amount))
    if make_dirs and not os.path.exists(BKGND_SUB_MOSAIC_DIR):
        os.mkdir(bkgnd_dir)
    data_path = file_clean_join(BKGND_SUB_MOSAIC_DIR,
                                obsid+bkgnd_res_data+'-BKGND-SUB-MOSAIC.npz')
    metadata_path = file_clean_join(
                     BKGND_SUB_MOSAIC_DIR,
                     obsid+bkgnd_res_data+'-BKGND-SUB-METADATA.dat')

    return (data_path, metadata_path)

def bkgnd_sub_mosaic_paths(arguments, obsid, make_dirs=False):
    return bkgnd_sub_mosaic_paths_spec(arguments.ring_radius,
                                       arguments.radius_inner_delta,
                                       arguments.radius_outer_delta,
                                       arguments.radius_resolution,
                                       arguments.longitude_resolution,
                                       arguments.radial_zoom_amount,
                                       arguments.longitude_zoom_amount,
                                       obsid, arguments.ring_type,
                                       make_dirs=make_dirs)

def polar_png_path_spec(ring_radius, radius_inner, radius_outer,
                        radius_resolution, longitude_resolution,
                        radial_zoom_amount, longitude_zoom_amount,
                        obsid, ring_type, make_dirs=False):
    png_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d_1' % (
                      ring_radius, radius_inner, radius_outer,
                      radius_resolution, longitude_resolution,
                      radial_zoom_amount, longitude_zoom_amount))
    if make_dirs and not os.path.exists(POLAR_PNG_DIR):
        os.mkdir(POLAR_PNG_DIR)
    data_path = file_clean_join(POLAR_PNG_DIR,
                                obsid+png_res_data+'-POLAR.png')
    return data_path

def polar_png_path(arguments, obsid, make_dirs=False):
    return polar_png_path_spec(arguments.ring_radius,
                               arguments.radius_inner_delta,
                               arguments.radius_outer_delta,
                               arguments.radius_resolution,
                               arguments.longitude_resolution,
                               arguments.radial_zoom_amount,
                               arguments.longitude_zoom_amount,
                               obsid, arguments.ring_type,
                               make_dirs=make_dirs)

def read_ew(root):
    return np.load(root+'.npy')

def read_ew_metadata(root):
    with open(root+'-METADATA.dat', 'rb') as fp:
        metadata = pickle.load(fp, encoding='latin1')
    return metadata

def get_ew_valid_longitudes(ew, ew_metadata):
    return (ew != 0) & (ew_metadata['longitudes'] >= 0)

def ew_paths_spec(ring_radius, radius_inner, radius_outer,
                  radius_resolution, longitude_resolution,
                  radial_zoom_amount, longitude_zoom_amount,
                  obsid, ring_type, make_dirs=False):
    ew_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d_1' % (
                   ring_radius, radius_inner, radius_outer,
                   radius_resolution, longitude_resolution,
                   radial_zoom_amount, longitude_zoom_amount))
    if make_dirs and not os.path.exists(EW_DIR):
        os.mkdir(EW_DIR)
    data_path = file_clean_join(EW_DIR, obsid+ew_res_data)
    metadata_path = file_clean_join(EW_DIR, obsid+ew_res_data+'-METADATA.dat')

    return data_path, metadata_path

def ew_paths(arguments, obsid, make_dirs=False):
    return ew_paths_spec(arguments.ring_radius,
                         arguments.radius_inner_delta,
                         arguments.radius_outer_delta,
                         arguments.radius_resolution,
                         arguments.longitude_resolution,
                         arguments.radial_zoom_amount,
                         arguments.longitude_zoom_amount,
                         obsid, arguments.ring_type,
                         make_dirs=make_dirs)

def write_ew(ew_filename, ew, ew_metadata_filename, ew_metadata):
    np.save(ew_filename, ew)
    with open(ew_metadata_filename, 'wb') as ew_metadata_fp:
        pickle.dump(ew_metadata, ew_metadata_fp)

def read_width(root):
    return np.load(root+'-WIDTH.npy')

def width_paths_spec(ring_radius, radius_inner, radius_outer,
                     radius_resolution, longitude_resolution,
                     radial_zoom_amount, longitude_zoom_amount,
                     obsid, ring_type, make_dirs=False):
    width_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d_1' % (
                      ring_radius, radius_inner, radius_outer,
                      radius_resolution, longitude_resolution,
                      radial_zoom_amount, longitude_zoom_amount))
    if make_dirs and not os.path.exists(EW_DIR):
        os.mkdir(EW_DIR)
    data_path = file_clean_join(EW_DIR, obsid+width_res_data+'-WIDTH')

    return data_path

def width_paths(arguments, obsid, make_dirs=False):
    return width_paths_spec(arguments.ring_radius,
                            arguments.radius_inner_delta,
                            arguments.radius_outer_delta,
                            arguments.radius_resolution,
                            arguments.longitude_resolution,
                            arguments.radial_zoom_amount,
                            arguments.longitude_zoom_amount,
                            obsid, arguments.ring_type,
                            make_dirs=make_dirs)

def write_width(width_filename, widths):
    np.save(width_filename, widths)

def clumpdb_paths(options):
    cl_res_data = ('_%06d_%06d_%06.3f_%05.3f_%02d_%02d_%06d_%06d' %
                   (options.radius_start, options.radius_end,
                    options.radius_resolution,
                    options.longitude_resolution,
                    options.reproject_zoom_factor,
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


################################################################################
#
# F-RING ORBIT FUNCTIONS
#
################################################################################

# Old data from 2012 paper
# bosh2002_fring_a = 140223.7
# bosh2002_fring_e = 0.00254
# bosh2002_fring_curly = 24.1 * np.pi/180
# bosh2002_fring_curly_dot = 2.7001 * np.pi/180 / 86400 # rad/sec

# F ring orbit from Albers 2012
FRING_ROTATING_ET = utc2et('2007-01-01')
FRING_ORBIT_EPOCH = utc2et('2000-01-01T12:00:00') # J2000
FRING_MEAN_MOTION = np.radians(581.964)
FRING_A = 140221.3
FRING_E = 0.00235
FRING_W0 = np.radians(24.2)
FRING_DW = np.radians(2.70025)

def _compute_fring_longitude_shift(et):
    return - (FRING_MEAN_MOTION *
              ((et - FRING_ROTATING_ET) / 86400.)) % TWOPI

def fring_inertial_to_corotating(longitude, et):
    """Convert inertial longitude to corotating."""
    return (longitude + _compute_fring_longitude_shift(et)) % TWOPI

def fring_corotating_to_inertial(co_long, et):
    """Convert corotating longitude (deg) to inertial."""
    return (co_long - _compute_fring_longitude_shift(et)) % TWOPI

def fring_radius_at_longitude(obs, longitude):
    """Return the radius (km) of the F ring core at a given inertial longitude
    (deg)."""
    curly_w = FRING_W0 + FRING_DW*obs.midtime/86400.

    radius = (FRING_A * (1-FRING_E**2) /
              (1 + FRING_E * np.cos(longitude-curly_w)))

    return radius


################################################################################
#
# SHADOWING AND OBSCURATION ADJUSTMENT
#
################################################################################

def compute_mu(e):
    if isinstance(e, (list, tuple)):
        e = np.array(e)
    return np.abs(np.cos(np.radians(e)))

def compute_mu0(i):
    if isinstance(i, (list, tuple)):
        i = np.array(i)
    return np.abs(np.cos(np.radians(i)))

def compute_z(mu, mu0, tau, is_transmission):
    transmission_list = tau*(mu-mu0)/(mu*mu0*(np.exp(-tau/mu)-np.exp(-tau/mu0)))
    reflection_list = tau*(mu+mu0)/(mu*mu0*(1-np.exp(-tau*(1/mu+1/mu0))))
    ret = np.where(is_transmission, transmission_list, reflection_list)
    return ret

def compute_corrected_ew(normal_ew, emission, incidence, tau):
    if tau == 0:
        return normal_ew
    if isinstance(emission, (tuple,list)):
        emission = np.array(emission)
    if isinstance(incidence, (tuple,list)):
        incidence = np.array(incidence)
    is_transmission = emission > 90.
    mu = compute_mu(emission)
    mu0 = compute_mu0(incidence)
    ret = normal_ew * compute_z(mu, mu0, tau, is_transmission)
    return ret

def compute_corrected_ew_col(obsdata, col_tau=('Normal EW', None),
                             emission_col='Mean Emission',
                             incidence_col='Incidence'):
    total_ew = None
    for i in range(0, len(col_tau), 2):
        col = col_tau[i]
        tau = col_tau[i+1]
        if tau is None:
            ew = obsdata[col]
        else:
            ew = compute_corrected_ew(obsdata[col],
                                      obsdata[emission_col],
                                      obsdata[incidence_col],
                                      tau=tau)
        if total_ew is None:
            total_ew = ew
        else:
            total_ew = total_ew + ew
    return total_ew


################################################################################
#
# PHASE CURVE FITTING
#
################################################################################

def hg(alpha, g):
    # Henyey-Greenstein function
    return (1-g**2) / (1+g**2+2*g*np.cos(np.radians(alpha)))**1.5 / 2

def hg_func(params, xpts):
    ypts = None
    for i in range(len(params)//2):
        g, scale = params[i*2:i*2+2]
        if ypts is None:
            ypts = scale * hg(xpts, g)
        else:
            ypts += scale * hg(xpts, g)
    return ypts

def hg_fit_func(params, xpts, ypts, ystd):
    if ystd is None:
        ystd = 1
    return (ypts - hg_func(params, xpts)) / ystd
    # return np.log(ypts) - np.log(hg_func(params, xpts))

# Fit a phase curve and remove data points more than nstd sigma away
# Use std=None to not remove outliers
# Do the modeling on a copy of the data so we can remove outliers
def fit_hg_phase_function(n_hg, nstd, data, col_tau=('Normal EW', None),
                          phase_col='Mean Phase', std_col=None, verbose=True):
    phasedata = data.copy()
    normal_ew = compute_corrected_ew_col(phasedata, col_tau=col_tau)

    initial_guess = []
    bounds1 = []
    bounds2 = []
    for i in range(n_hg):
        initial_guess.append(-.5)
        initial_guess.append(1.)
        bounds1.append(-1.)
        bounds1.append(0.)
        bounds2.append(1.)
        bounds2.append(1000.)
    while True:
        phase_degrees = phasedata[phase_col]
        std_data = None
        if std_col is not None:
            std_col = phasedata[std_col]
        params = sciopt.least_squares(hg_fit_func, initial_guess,
                                      bounds=(bounds1, bounds2),
                                      args=(phase_degrees, normal_ew, std_col))
        params = params['x']
        phase_model = hg_func(params, phase_degrees)
        ratio = np.log10(normal_ew / phase_model)
        std = np.std(ratio)
        if verbose:
            print('Ratio min', ratio.min(), 'Max', ratio.max(), 'Sigma', std)
        if nstd is None:
            break
        oldlen = len(phasedata)
        keep = ratio.abs() < nstd*std
        phasedata = phasedata[keep]
        normal_ew = normal_ew[keep]
        if len(phasedata) == oldlen:
            break
    return params, phasedata, std

def print_hg_params(params, indent=0):
    total_scale = sum(params[i] for i in range(1,len(params),2))
    res = []
    for i in range(0,len(params)//2):
        g, scale = params[i*2:i*2+2]
        res.append((scale/total_scale, g, scale))
    res.sort(reverse=True)
    for i in range(len(res)):
        print((' ' * indent) +
              ('g%d = %6.3f / scale%d = %6.3f / weight%d = %5.3f' %
               (i+1, res[i][1], i+1, res[i][2], i+1, res[i][0])))


# def transmission_function(tau, emission, incidence):
#     # incidence angle is always less than 90, therefore the only case we have to worry
#     # about is when Emission Angle changes.
#     # E > 90 = Transmission, E < 90 = Reflection
#     mu0 = compute_mu0(incidence)
#     mu = compute_mu(emission)
#
#     if np.mean(emission[np.nonzero(emission)]) > 90:
#         return mu * mu0 * (np.exp(-tau/mu)-np.exp(-tau/mu0)) / (tau * (mu-mu0))
#     return mu * mu0 * (1.-np.exp(-tau*(1/mu+1/mu0))) / (tau * (mu+mu0))
#
