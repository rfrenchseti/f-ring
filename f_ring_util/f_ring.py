import matplotlib.pyplot as plt
import mplcursors
import os
import pickle

import numpy as np
import pandas as pd
import scipy.optimize as sciopt

import julian


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
                root_list.append(file_clean_join(dir, filename.replace('.npy', '')))
    root_list.sort()
    return root_list

def enumerate_obsids(arguments):
    data_path, _ = bkgnd_sub_mosaic_paths(arguments, '', make_dirs=False)
    bp = data_path.replace(BKGND_SUB_MOSAIC_DIR+'/', '')
    for _, _, filenames in os.walk(BKGND_SUB_MOSAIC_DIR):
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
# PATHS
#
################################################################################

# Mosaic and mosaic metata

def mosaic_paths_spec(ring_radius, radius_inner, radius_outer,
                      radius_resolution, longitude_resolution,
                      radial_zoom_amount, longitude_zoom_amount,
                      obsid, ring_type, make_dirs=False):
    mosaic_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d' % (
                       ring_radius, radius_inner, radius_outer,
                       radius_resolution, longitude_resolution,
                       radial_zoom_amount, longitude_zoom_amount))
    if make_dirs and not os.path.exists(MOSAIC_DIR):
        os.mkdir(MOSAIC_DIR)
    data_path = file_clean_join(MOSAIC_DIR,
                                obsid+mosaic_res_data+'-MOSAIC.npy')
    metadata_path = file_clean_join(
                     MOSAIC_DIR,
                     obsid+mosaic_res_data+'-MOSAIC-METADATA.dat')

    return (data_path, metadata_path)

def mosaic_paths(arguments, obsid, make_dirs=False):
    return mosaic_paths_spec(arguments.ring_radius,
                             arguments.radius_inner_delta,
                             arguments.radius_outer_delta,
                             arguments.radius_resolution,
                             arguments.longitude_resolution,
                             arguments.radial_zoom_amount,
                             arguments.longitude_zoom_amount,
                             obsid, arguments.ring_type,
                             make_dirs=make_dirs)

# Background model and background model metadata

def bkgnd_paths_spec(ring_radius, radius_inner, radius_outer,
                     radius_resolution, longitude_resolution,
                     radial_zoom_amount, longitude_zoom_amount,
                     obsid, ring_type, make_dirs=False):
    bkgnd_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d_1' % (
                      ring_radius, radius_inner, radius_outer,
                      radius_resolution, longitude_resolution,
                      radial_zoom_amount, longitude_zoom_amount))
    if make_dirs and not os.path.exists(BKGND_DIR):
        os.mkdir(BKGND_DIR)
    bkgnd_model_path = file_clean_join(
                     BKGND_DIR,
                     obsid+bkgnd_res_data+'-BKGND-MODEL.npz')
    bkgnd_metadata_path = file_clean_join(
                     BKGND_DIR,
                     obsid+bkgnd_res_data+'-BKGND-METADATA.dat')

    return (bkgnd_model_path, bkgnd_metadata_path)

def bkgnd_paths(arguments, obsid, make_dirs=False):
    return bkgnd_paths_spec(arguments.ring_radius,
                            arguments.radius_inner_delta,
                            arguments.radius_outer_delta,
                            arguments.radius_resolution,
                            arguments.longitude_resolution,
                            arguments.radial_zoom_amount,
                            arguments.longitude_zoom_amount,
                            obsid, arguments.ring_type,
                            make_dirs=make_dirs)

# Background-subtracted-mosaic and associated metadata

def bkgnd_sub_mosaic_paths_spec(ring_radius, radius_inner, radius_outer,
                                radius_resolution, longitude_resolution,
                                radial_zoom_amount, longitude_zoom_amount,
                                obsid, ring_type, make_dirs=False):
    bkgnd_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d_1' % (
                      ring_radius, radius_inner, radius_outer,
                      radius_resolution, longitude_resolution,
                      radial_zoom_amount, longitude_zoom_amount))
    if make_dirs and not os.path.exists(BKGND_SUB_MOSAIC_DIR):
        os.mkdir(BKGND_SUB_MOSAIC_DIR)
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
    assert False
    # cl_data_filename = file_clean_join(ROOT, 'clump-data',
    #                                 'clumpdb'+cl_res_data+'.pickle')
    # cc_data_filename = file_clean_join(ROOT, 'clump-data',
    #                                  'clumpchains'+cl_res_data+'.pickle')
    # if options.voyager:
    #     cl_data_filename = file_clean_join(ROOT, 'clump-data',
    #                                     'voyager_clumpdb'+cl_res_data+'.pickle')
    #     cc_data_filename = file_clean_join(ROOT, 'clump-data',
    #                                     'voyager_clumpchains'+cl_res_data+'.pickle')
    # if options.downsample:
    #     cl_data_filename = file_clean_join(ROOT, 'clump-data',
    #                                     'downsampled_clumpdb'+cl_res_data+'.pickle')
    #     cc_data_filename = file_clean_join(ROOT, 'clump-data',
    #                                     'downsampled_clumpchains'+cl_res_data+'.pickle')
    # return cl_data_filename, cc_data_filename


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

def fring_radius_at_longitude(et, longitude):
    """Return the radius (km) of the F ring core at a given inertial longitude
    (deg)."""
    true_anomaly = fring_true_anomaly(et, longitude)

    radius = (FRING_A * (1-FRING_E**2) /
              (1 + FRING_E * np.cos(true_anomaly)))

    return radius

def fring_longitude_of_pericenter(et):
    """Return the longitude of pericenter (radians) at the given time."""
    return (FRING_W0 + FRING_DW*et/86400.) % TWOPI

def fring_true_anomaly(et, longitude):
    """Return the true anomaly (radians) at the given time and inertial longitude."""
    curly_w = FRING_W0 + FRING_DW*et/86400.
    return (longitude - curly_w) % TWOPI


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

def compute_corrected_ew_col(obsdata, col_tau=('Normal EW Mean', None),
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
    return (1-g**2) / (1+g**2-2*g*np.cos(np.radians(alpha)))**1.5 / 2

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

def hg_fit_func_scale(scale, params, xpts, ypts):
    ret = ypts - hg_func(params, xpts)*scale
    return ret

# Fit a phase curve and remove data points more than nstd sigma away
# Use std=None to not remove outliers
# Do the modeling on a copy of the data so we can remove outliers
def fit_hg_phase_function(n_hg, nstd, data, col_tau=('Normal EW Mean', None),
                          phase_col='Mean Phase', std_col=None, verbose=False):
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
        if std_col is not None:
            std_col = phasedata[std_col]
        params = sciopt.least_squares(hg_fit_func, initial_guess,
                                      bounds=(bounds1, bounds2),
                                      args=(phase_degrees, normal_ew, std_col))
        params = params['x']
        phase_model = hg_func(params, phase_degrees)
        ratio = np.log10(normal_ew) - np.log10(phase_model)
        std = np.std(ratio)
        if verbose:
            print('Ratio min', ratio.min(), 'Max', ratio.max(), 'Sigma', std)
        if nstd is None:
            break
        else:
            assert False # Think about ratio and log space XXX
        oldlen = len(phasedata)
        keep = ratio.abs() < nstd*std
        phasedata = phasedata[keep]
        normal_ew = normal_ew[keep]
        if len(phasedata) == oldlen:
            break
    param_list = [(params[i], params[i+1]) for i in range(0, len(params), 2)]
    param_list.sort()
    params = [x for sublist in param_list for x in sublist]
    return params, phasedata, std

def print_hg_params(params, indent=0):
    total_scale = sum(params[i] for i in range(1,len(params),2))
    res = []
    for i in range(0,len(params)//2):
        g, scale = params[i*2:i*2+2]
        res.append((scale/total_scale, g, scale))
    res.sort(reverse=True)
    for i in range(len(res)):
        term = ''
        if i == len(res)-1:
            avg = np.mean([x[2] for x in res])
            term = f' Avg scale {avg:.3f}'
        print((' ' * indent) +
              (f'g{i+1} = {res[i][1]:6.3f} / scale{i+1} = {res[i][2]:6.3f} / weight{i+1} = {res[i][0]:5.3f} {term}'))

# Fit a phase curve and remove data points more than nstd sigma away
# Use std=None to not remove outliers
# Do the modeling on a copy of the data so we can remove outliers
def scale_hg_phase_function(params, data, col_tau=('Normal EW Mean', None),
                            phase_col='Mean Phase'):
    phasedata = data.copy()
    normal_ew = compute_corrected_ew_col(phasedata, col_tau=col_tau)

    initial_guess = [1.]
    bounds1 = [0.]
    bounds2 = [10.]
    phase_degrees = phasedata[phase_col]
    params = sciopt.least_squares(hg_fit_func_scale, initial_guess,
                                  bounds=(bounds1, bounds2),
                                  args=(params, phase_degrees, normal_ew))
    scale = params['x'][0]
    return scale


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


################################################################################
#
# JUPYTER NOTEBOOK AIDS
#
################################################################################

### READ EW STATS AND OBS_LIST RESTRICTIONS

OBS_LIST = None

def read_obs_list():
    global OBS_LIST
    if OBS_LIST is None:
        OBS_LIST = pd.read_csv('../observation_lists/CASSINI_OBSERVATION_LIST.csv',
                               parse_dates=['Date'],
                               index_col='Observation')

def read_cassini_ew_stats(filename, use_obs_list=True, verbose=True):
    obsdata = pd.read_csv(filename, parse_dates=['Date'],
                          index_col='Observation')
    if use_obs_list:
        read_obs_list()
        obsdata = obsdata.join(OBS_LIST, rsuffix='_obslist')
        obsdata = obsdata[obsdata['For Photometry'] == 1]
    if verbose:
        print(f'** SUMMARY STATISTICS - {filename} **')
        print('Unique observation names:', len(obsdata.groupby('Observation')))
        print('Total slices:', len(obsdata))
        print('Starting date:', obsdata['Date'].min())
        print('Ending date:', obsdata['Date'].max())
        print('Time span:', obsdata['Date'].max()-obsdata['Date'].min())
    time0 = np.datetime64('1970-01-01T00:00:00') # Epoch
    obsdata['Date_days'] = (obsdata['Date']-time0).dt.total_seconds()/86400
    obsdata['Mu'] = np.abs(np.cos(np.radians(obsdata['Mean Emission'])))
    obsdata['Mu0'] = np.abs(np.cos(np.radians(obsdata['Incidence'])))
    return obsdata

def read_voyager_ew_stats(filename, verbose=True):
    obsdata = pd.read_csv(filename, parse_dates=['Date'])
    if verbose:
        print(f'** SUMMARY STATISTICS - {filename} **')
        print('Unique observation names:', len(obsdata.groupby('Observation')))
        print('Total slices:', len(obsdata))
        print('Starting date:', obsdata['Date'].min())
        print('Ending date:', obsdata['Date'].max())
        print('Time span:', obsdata['Date'].max()-obsdata['Date'].min())
    time0 = np.datetime64('1970-01-01T00:00:00') # Epoch
    obsdata['Date_days'] = (obsdata['Date']-time0).dt.total_seconds()/86400
    obsdata['Mu'] = np.abs(np.cos(np.radians(obsdata['Mean Emission'])))
    obsdata['Mu0'] = np.abs(np.cos(np.radians(obsdata['Incidence'])))
    return obsdata

def read_showalter_voyager_ew_stats(filename, verbose=True):
    obsdata = pd.read_csv(filename, index_col='FDS', delim_whitespace=True)
    if verbose:
        print(f'** SUMMARY STATISTICS - {filename} **')
        print('Unique FDS:', len(obsdata))
    obsdata = obsdata.rename({'FDS': 'Observation',
                              'inc': 'Incidence',
                              'em': 'Mean Emission',
                              'phase': 'Mean Phase',
                              'EW': 'EW Mean'},
                             axis='columns')
    obsdata['Mu'] = np.abs(np.cos(np.radians(obsdata['Mean Emission'])))
    obsdata['Mu0'] = np.abs(np.cos(np.radians(obsdata['Incidence'])))
    obsdata['Normal EW Mean'] = obsdata['EW Mean'] * obsdata['Mu']
    return obsdata


### OTHER UTILITY FUNCTIONS

def add_hover(obsdata, p1, obsdata2=None, p2=None):
    """Add hover text to scatter points."""
    cursor1 = mplcursors.cursor(p1, hover=True)
    @cursor1.connect('add')
    def on_add(sel):
        row = obsdata.iloc[sel.target.index]
        sel.annotation.set(text=f"{row.name} @ {row['Min Long']:.2f}\n"
                                f"{str(row['Date']).split(' ')[0]} "
                                f"a={row['Mean Phase']:.0f} e={row['Mean Emission']:.0f} "
                                f"i={row['Incidence']:.2f}")
        if obsdata2 is not None:
            for s in cursor2.selections:
                cursor2.remove_selection(s)
    if obsdata2 is not None:
        cursor2 = mplcursors.cursor(p2, hover=True)
        @cursor2.connect('add')
        def on_add(sel):
            row = obsdata2.iloc[sel.target.index]
            sel.annotation.set(text=f"{row.name} @ {row['Min Long']:.2f}\n"
                                    f"{str(row['Date']).split(' ')[0]} "
                                    f"a={row['Mean Phase']:.0f} e={row['Mean Emission']:.0f} "
                                    f"i={row['Incidence']:.2f}")
            for s in cursor1.selections:
                cursor1.remove_selection(s)

# These EWs are raw, not adjusted for emission angle
DATA2012_DICT = {
    'ISS_000RI_SATSRCHAP001_PRIME': [ 2.6, 0.8],
    'ISS_00ARI_SPKMOVPER001_PRIME': [ 3.1, 0.6],
    'ISS_006RI_LPHRLFMOV001_PRIME': [ 4.7, 0.9],
    'ISS_007RI_LPHRLFMOV001_PRIME': [ 1.5, 0.3],
    'ISS_029RF_FMOVIE001_VIMS':     [12.6, 2.7],
    'ISS_031RF_FMOVIE001_VIMS':     [10.3, 1.4],
    'ISS_032RF_FMOVIE001_VIMS':     [ 9.9, 1.8],
    'ISS_033RF_FMOVIE001_VIMS':     [12.9, 1.7],
    'ISS_036RF_FMOVIE001_VIMS':     [13.6, 5.3],
    'ISS_036RF_FMOVIE002_VIMS':     [ 2.9, 2.2],
    'ISS_039RF_FMOVIE002_VIMS':     [ 2.7, 1.7],
    'ISS_039RF_FMOVIE001_VIMS':     [ 1.7, 1.0],
    'ISS_041RF_FMOVIE002_VIMS':     [ 1.8, 1.0],
    'ISS_041RF_FMOVIE001_VIMS':     [ 2.1, 0.9],
    'ISS_044RF_FMOVIE001_VIMS':     [ 2.4, 0.9],
    'ISS_051RI_LPMRDFMOV001_PRIME': [ 8.1, 1.6],
    'ISS_055RF_FMOVIE001_VIMS':     [ 1.3, 0.3],
    'ISS_055RI_LPMRDFMOV001_PRIME': [ 3.2, 0.5],
    'ISS_057RF_FMOVIE001_VIMS':     [ 1.3, 0.3],
    'ISS_068RF_FMOVIE001_VIMS':     [ 0.9, 0.1],
    'ISS_075RF_FMOVIE002_VIMS':     [ 1.2, 0.2],
    'ISS_083RI_FMOVIE109_VIMS':     [ 1.9, 0.6],
    'ISS_087RF_FMOVIE003_PRIME':    [ 0.9, 0.2],
    'ISS_089RF_FMOVIE003_PRIME':    [ 1.0, 0.2],
    'ISS_100RF_FMOVIE003_PRIME':    [ 0.8, 0.1]
}
_data2012_obsname = DATA2012_DICT.keys()
_data2012_ew = [x[0] for x in DATA2012_DICT.values()]
_data2012_std = [x[1] for x in DATA2012_DICT.values()]
DATA2012_DF = pd.DataFrame({'EW Mean': _data2012_ew,
                            'EW Std':  _data2012_std},
                            index=_data2012_obsname)

def find_common_data_2012(obsdata, verbose=True):
    commondata = obsdata.join(DATA2012_DF, on='Observation', how='inner',
                              rsuffix='_2012')

    # Compute the normal EW by adjusting by the mean emission angle in the new data
    commondata['Normal EW Mean_2012'] = commondata['EW Mean_2012'] * np.abs(
            np.cos(np.radians(commondata['Mean Emission'])))

    # Compute the ratios between the new and old data
    # (these should be approximately the same)
    commondata['EW Mean Ratio'] = (commondata['EW Mean'] /
                                   commondata['EW Mean_2012'])
    commondata['Normal EW Mean Ratio'] = (commondata['Normal EW Mean'] /
                                          commondata['Normal EW Mean_2012'])

    if verbose:
        print('Total number of new observation names:', len(obsdata))
        print('Total number of observation names from 2012:', len(DATA2012_DF))
        print('Number of observation names in common:', len(commondata))
        print('Missing observation names:',
              set(DATA2012_DICT.keys())-set(commondata.index))

    return commondata


# These are the images used to compare calibrations. Each is either the first or last
# image used in the 2012 paper for each observation name.
# image_versions = (
#     ('N1466448701_1_CALIB-3.3.IMG', 'N1466448701_1_CALIB-4.0.IMG'), # ISS_000RI_SATSRCHAP001_PRIME
#     ('N1479201492_1_CALIB-3.3.IMG', 'N1479201492_1_CALIB-4.0.IMG'), # ISS_00ARI_SPKMOVPER001_PRIME
#     ('N1492052646_1_CALIB-3.3.IMG', 'N1492052646_1_CALIB-4.0.IMG'), # ISS_006RI_LPHRLFMOV001_PRIME
#     ('N1493613276_1_CALIB-3.3.IMG', 'N1493613276_1_CALIB-4.0.IMG'), # ISS_007RI_LPHRLFMOV001_PRIME
#     ('N1538168640_1_CALIB-3.3.IMG', 'N1538168640_1_CALIB-4.0.IMG'), # ISS_029RF_FMOVIE001_VIMS
#     ('N1541012989_1_CALIB-3.3.IMG', 'N1541012989_1_CALIB-4.0.IMG'), # ISS_031RF_FMOVIE001_VIMS
#     ('N1542047155_1_CALIB-3.3.IMG', 'N1542047155_1_CALIB-4.0.IMG'), # ISS_032RF_FMOVIE001_VIMS
#     ('N1543166702_1_CALIB-3.3.IMG', 'N1543166702_1_CALIB-4.0.IMG'), # ISS_033RF_FMOVIE001_VIMS
#     ('N1545556618_1_CALIB-3.3.IMG', 'N1545556618_1_CALIB-4.0.IMG'), # ISS_036RF_FMOVIE001_VIMS
#     ('N1546748805_1_CALIB-3.3.IMG', 'N1546748805_1_CALIB-4.0.IMG'), # ISS_036RF_FMOVIE002_VIMS
#     ('N1549801218_1_CALIB-3.3.IMG', 'N1549801218_1_CALIB-4.0.IMG'), # ISS_039RF_FMOVIE002_VIMS
#     ('N1551253524_1_CALIB-3.3.IMG', 'N1551253524_1_CALIB-4.0.IMG'), # ISS_039RF_FMOVIE001_VIMS
#     ('N1552790437_1_CALIB-3.3.IMG', 'N1552790437_1_CALIB-4.0.IMG'), # ISS_041RF_FMOVIE002_VIMS
#     ('N1554026927_1_CALIB-3.3.IMG', 'N1554026927_1_CALIB-4.0.IMG'), # ISS_041RF_FMOVIE001_VIMS

#     ('N1557020880_1_CALIB-3.6.IMG', 'N1557020880_1_CALIB-4.0.IMG'), # ISS_044RF_FMOVIE001_VIMS
#     ('N1571435192_1_CALIB-3.6.IMG', 'N1571435192_1_CALIB-4.0.IMG'), # ISS_051RI_LPMRDFMOV001_PRIME
#     ('N1577809417_1_CALIB-3.6.IMG', 'N1577809417_1_CALIB-4.0.IMG'), # ISS_055RF_FMOVIE001_VIMS
#     ('N1578386361_1_CALIB-3.6.IMG', 'N1578386361_1_CALIB-4.0.IMG'), # ISS_055RI_LPMRDFMOV001_PRIME
#     ('N1579790806_1_CALIB-3.6.IMG', 'N1579790806_1_CALIB-4.0.IMG'), # ISS_057RF_FMOVIE001_VIMS
#     ('N1589589182_1_CALIB-3.6.IMG', 'N1589589182_1_CALIB-4.0.IMG'), # ISS_068RF_FMOVIE001_VIMS
#     ('N1593913221_1_CALIB-3.6.IMG', 'N1593913221_1_CALIB-4.0.IMG'), # ISS_075RF_FMOVIE002_VIMS
#     ('N1598806665_1_CALIB-3.6.IMG', 'N1598806665_1_CALIB-4.0.IMG'), # ISS_083RI_FMOVIE109_VIMS
#     ('N1601485634_1_CALIB-3.6.IMG', 'N1601485634_1_CALIB-4.0.IMG'), # ISS_087RF_FMOVIE003_PRIME
#     ('N1602717403_1_CALIB-3.6.IMG', 'N1602717403_1_CALIB-4.0.IMG'), # ISS_089RF_FMOVIE003_PRIME
#     ('N1610364098_1_CALIB-3.6.IMG', 'N1610364098_1_CALIB-4.0.IMG'), # ISS_100RF_FMOVIE003_PRIME
# )

# The ratio of the medians of new / old. The medians are computed using only the pixel values with
# ring radii 139820-140620, to cover the range of +/- 400km from the nominal core semi-major axis
# (a*e = 329 km)
# These are generated using the program utilities/compare_cisscal_versions.py
CISSCAL_RATIO_DICT = { # CISSCAL 4.0 / CISSCAL 3.3-3.6 median ratio for one image
    # New / old ratio
    # CISSCAL 3.3
    'ISS_000RI_SATSRCHAP001_PRIME': 0.874,
    'ISS_00ARI_SPKMOVPER001_PRIME': 0.868,
    'ISS_006RI_LPHRLFMOV001_PRIME': 0.870,
    'ISS_007RI_LPHRLFMOV001_PRIME': 0.851,
    'ISS_029RF_FMOVIE001_VIMS':     0.910,
    'ISS_031RF_FMOVIE001_VIMS':     0.884,
    'ISS_032RF_FMOVIE001_VIMS':     0.883,
    'ISS_033RF_FMOVIE001_VIMS':     1.014,
    'ISS_036RF_FMOVIE001_VIMS':     0.889,
    'ISS_036RF_FMOVIE002_VIMS':     0.888,
    'ISS_039RF_FMOVIE002_VIMS':     0.895,
    'ISS_039RF_FMOVIE001_VIMS':     0.886,
    'ISS_041RF_FMOVIE002_VIMS':     0.896,
    'ISS_041RF_FMOVIE001_VIMS':     0.979,
    # CISSCAL 3.6 from here on
    'ISS_044RF_FMOVIE001_VIMS':     0.949,
    'ISS_051RI_LPMRDFMOV001_PRIME': 1.086,
    'ISS_055RF_FMOVIE001_VIMS':     0.979,
    'ISS_055RI_LPMRDFMOV001_PRIME': 0.929,
    'ISS_057RF_FMOVIE001_VIMS':     0.979,
    'ISS_068RF_FMOVIE001_VIMS':     0.992,
    'ISS_075RF_FMOVIE002_VIMS':     0.979,
    'ISS_083RI_FMOVIE109_VIMS':     0.938,
    'ISS_087RF_FMOVIE003_PRIME':    0.936,
    'ISS_089RF_FMOVIE003_PRIME':    0.936,
    'ISS_100RF_FMOVIE003_PRIME':    0.952
}
_cr_obsname = CISSCAL_RATIO_DICT.keys()
_cr_ratio = CISSCAL_RATIO_DICT.values()
CISSCAL_RATIO_DF = pd.DataFrame({'CISSCAL Ratio': _cr_ratio}, index=_cr_obsname)

def add_cisscal_ratios(obsdata):
    obsdata =  obsdata.join(CISSCAL_RATIO_DF, on='Observation', how='inner',
                            rsuffix='_old')
    obsdata['Adjusted Normal EW Mean_2012'] = (obsdata['Normal EW Mean_2012'] *
                                               obsdata['CISSCAL Ratio'])

    return obsdata


# print('Number of old observations:', len(cr_obsname))
# print('Number of common observations:', len(commondata))

### CALCULATE QUANTILES

def limit_by_quant(obsdata, cutoff1, cutoff2, col='Normal EW Mean'):
    def xform_func(column):
        if quant2 is None:
            return [(None if z > quant1[column.name] else z) for z in column]
        return [(None if z > quant1[column.name] or
                         z < quant2[column.name] else z) for z in column]
    obsdata = obsdata.copy()
    group = obsdata.groupby('Observation')
    quant1 = group.quantile(cutoff1/100, numeric_only=True)[col]
    quant2 = None
    if cutoff2 is not None:
        quant2 = group.quantile(cutoff2/100, numeric_only=True)[col]
    xform = group[col].transform(xform_func)
    obsdata['_control'] = xform
    obsdata.dropna(inplace=True)
    return obsdata

################################################################################

RING_TYPE = 'FMOVIE'
DATA_ROOT = os.path.abspath(os.environ.get('FRING_DATA_ROOT'))
MOSAIC_DIR = file_clean_join(DATA_ROOT, f'mosaic_{RING_TYPE}')
BKGND_DIR = file_clean_join(DATA_ROOT, f'bkgnd_{RING_TYPE}')
BKGND_SUB_MOSAIC_DIR = file_clean_join(DATA_ROOT, f'bkgnd_sub_mosaic_{RING_TYPE}')
REPRO_DIR = file_clean_join(DATA_ROOT, 'ring_repro')


TWOPI = np.pi*2
