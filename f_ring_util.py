import numpy as np
import os
import pickle

import julian

BKGND_SUB_MOSAIC_DIR = os.environ['BKGND_SUB_MOSAIC_DIR']
EW_DIR = os.environ['EW_DIR']

TWOPI = np.pi*2

def utc2et(s):
    return julian.tdb_from_tai(julian.tai_from_iso(s))

def file_clean_join(*args):
    ret = os.path.join(*args)
    return ret.replace('\\', '/')

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
        '--all-obsid', action='store_true', default=False,
        help='Process all OBSIDs of the given type')
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
            if filename.endswith('.npy'):
                root_list.append(os.path.join(dir, filename.replace('.npy', '')))
    root_list.sort()
    return root_list

# Given a root filename, read and return the mosaic data file.
def read_mosaic(root):
    return np.load(root+'-MOSAIC.npy')

# Given a root filename, read and return the metadata. The returned metadata
# is a dictionary with the following keys:
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
def read_mosaic_metadata(root):
    with open(root+'-METADATA.dat', 'rb') as fp:
        metadata = pickle.load(fp, encoding='latin1')
    return metadata

# Return the valid portion of a mosaic.
def valid_mosaic_subset(mosaic, meadata):
    valid_longitudes = get_valid_longitudes(mosaic, metadata)
    lower_limit = metadata['ring_lower_limit']
    upper_limit = metadata['ring_upper_limit']
    return mosaic[lower_limit:upper_limit+1, valid_longitudes]

# Given a mosaic, figure out which longitudes have valid data in ALL radial
# locations that are outside of the radii used to compute the background
# gradient. This returns a 1-D boolean array of longitudes where True means
# that longitude has valid data.
def get_mosaic_valid_longitudes(mosaic, metadata):
    lower_limit = metadata['ring_lower_limit']
    upper_limit = metadata['ring_upper_limit']
    return np.all(mosaic[lower_limit:upper_limit+1, :], axis=0)

def enumerate_obsids(arguments):
    bp = f'_{arguments.ring_radius:06d}'
    for dirpath, dirnames, filenames in os.walk(EW_DIR):
        for filename in sorted(filenames):
            if filename.endswith('.npy'):
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

FRING_ROTATING_ET = None
FRING_MEAN_MOTION = np.radians(581.964)
FRING_A = 140221.3
FRING_E = 0.00235
FRING_W0 = np.radians(24.2)
FRING_DW = np.radians(2.70025)

def _compute_fring_longitude_shift(et):
    global FRING_ROTATING_ET
    if FRING_ROTATING_ET is None:
        FRING_ROTATING_ET = utc2et("2007-01-01")

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



        # # If we need to, reduce the mosaic and save the new version
        # # for future use
        # if arguments.mosaic_reduction_factor != 1 and create_reduced:
        #     bkgnddata.mosaic_img = image_unzoom(bkgnddata.mosaic_img,
        #                                 (1, arguments.mosaic_reduction_factor))
        #     # Don't need mean on longitudes - longitude # is always
        #     # left side of bin
        #     bkgnddata.longitudes = bkgnddata.longitudes[
        #                 ::arguments.mosaic_reduction_factor]
        #     bkgnddata.resolutions = image_unzoom(bkgnddata.resolutions,
        #                                      arguments.mosaic_reduction_factor)
        #     # It doesn't make any sense to average image numbers or time
        #     bkgnddata.image_numbers = bkgnddata.image_numbers[
        #                 ::arguments.mosaic_reduction_factor]
        #     bkgnddata.ETs = bkgnddata.ETs[
        #                 ::arguments.mosaic_reduction_factor]
        #     bkgnddata.emission_angles = image_unzoom(bkgnddata.emission_angles,
        #                                      arguments.mosaic_reduction_factor)
        #     bkgnddata.incidence_angle = bkgnddata.incidence_angle
        #     bkgnddata.phase_angles = image_unzoom(bkgnddata.phase_angles,
        #                                      arguments.mosaic_reduction_factor)
        #     bkgnddata.longitude_resolution *= arguments.mosaic_reduction_factor
        #     reduced_metadata = {}
        #     _update_metadata(bkgnddata, reduced_metadata)
        #     if create_reduced:
        #         ring_util.write_mosaic(bkgnddata.reduced_mosaic_data_filename,
        #                                bkgnddata.mosaic_img,
        #                                bkgnddata.reduced_mosaic_metadata_filename,
        #                                reduced_metadata)
        #
