import argparse
import numpy as np
import numpy.ma as ma
import pickle
import os

import msgpack
import msgpack_numpy

from nav.file import (fixup_byte_to_str,
                      results_path,
                      yield_image_filenames)

import julian

from imgdisp import ImageDisp
from PIL import Image

RING_FILENAMES = None

CB_SOURCE_ROOT = os.environ['CB_SOURCE_ROOT']
CB_RESULTS_ROOT = os.environ['CB_RESULTS_ROOT']

def file_clean_join(*args):
    ret = os.path.join(*args)
    return ret.replace('\\', '/')

RING_RESULTS_ROOT = file_clean_join(CB_RESULTS_ROOT, 'ring_mosaic')

RING_SOURCE_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RING_LISTS_ROOT   = file_clean_join(RING_SOURCE_ROOT, 'ring_mosaic_lists')
RING_REPROJECT_PY = file_clean_join(RING_SOURCE_ROOT, 'ring_ui_reproject.py')
RING_MOSAIC_PY    = file_clean_join(RING_SOURCE_ROOT, 'ring_ui_mosaic.py')
RING_BKGND_PY     = file_clean_join(RING_SOURCE_ROOT, 'ring_ui_bkgnd.py')

################################################################################
#
# Data structures used for each pass
#
################################################################################

class OffRepData(object):
    """Offset and Reprojection data for one image."""
    def __init__(self):
        self.obsid = None                   # Observation ID
        self.image_name = None              # The image basename
        self.image_path = None              # The full image path
        self.obs = None                     # The image data from image_path
        self.bad_pixel_map = None           # Optional mask of user-defined
                                            # bad pixels

        self.offset_path = None             # The path of the offset file
        self.the_offset = None              # The offset being used (x,y)
        self.manual_offset = None           # The manual offset, if any
        self.off_metadata = None            # Full metadata from offset process

        self.repro_path = None              # The path of the repro file
        self.repro_img = None               # The repro data from repro_path
        self.repro_long_mask = None         # The mask of used longitudes
        self.repro_longitudes = None        # The actual used longitudes
        self.repro_resolutions = None       # Resolution by longitude bin #
        self.repro_phase_angles = None      # Phase angle by longitude bin #
        self.repro_emission_angles = None   # Emission angle by longitude bin #
        self.repro_incidence_angle = None   # Incidence angle (scalar)
        self.repro_time = None              # The time the repro was produced

        self.image_log_filehandler = None   # Logger
        self.subprocess_run = None          # True if run in a subprocess

class MosaicData(object):
    """Mosaic metadata for one mosaic."""
    def __init__(self):
        self.img = None

class BkgndData(object):
    """Background metadata for one mosaic."""
    def __init__(self, obsid=None, arguments=None):
        self.obsid = obsid
        if arguments is None:
            self.mosaic_img = None
            self.mosaic_data_filename = None
            self.mosaic_metadata_filename = None
            self.reduced_mosaic_data_filename = None
            self.reduced_mosaic_metadata_filename = None
            self.bkgnd_model_filename = None  # NPZ
            self.bkgnd_metadata_filename = None
            self.full_png_path = None
            self.small_png_path = None
        else:
            (self.mosaic_data_filename,
             self.mosaic_metadata_filename) = mosaic_paths(arguments, obsid)
            (self.reduced_mosaic_data_filename,
             self.reduced_mosaic_metadata_filename,
             self.bkgnd_model_filename,
             self.bkgnd_metadata_filename) = bkgnd_paths(arguments, obsid,
                                                         make_dirs=True)
            (self.bkgnd_sub_mosaic_filename,
             self.bkgnd_sub_mosaic_metadata_filename) = bkgnd_sub_mosaic_paths(
                                                        arguments, obsid,
                                                        make_dirs=True)
            self.full_png_path = bkgnd_png_path(arguments, obsid, 'full',
                                                make_dirs=True)
            self.small_png_path = bkgnd_png_path(arguments, obsid, 'small',
                                                 make_dirs=True)

        self.row_cutoff_sigmas = None
        self.row_ignore_fraction = None
        self.row_blur = None
        self.ring_lower_limit = None
        self.ring_upper_limit = None
        self.column_cutoff_sigmas = None
        self.column_inside_background_pixels = None
        self.column_outside_background_pixels = None
        self.degree = None
        self.corrected_mosaic_img = None

        self.radius_resolution = None
        self.longitude_resolution = None
        self.mosaic_img = None
        self.long_mask = None
        self.image_numbers = None
        self.ETs = None
        self.emission_angles = None
        self.incidence_angle = None
        self.phase_angles = None
        self.resolutions = None
        self.longitudes = None
        self.obsid_list = None
        self.image_name_list = None
        self.image_path_list = None
        self.repro_path_list = None


################################################################################
#
# Argument parsing
#
################################################################################

_VALID_INSTRUMENT_HOSTS = ('cassini', 'voyager')

def _validate_instrument_host(instrument_host):
    """Routine for argparse to validate an instrument host."""
    if not instrument_host in ('cassini', 'voyager'):
        raise argparse.ArgumentTypeError(
             f"{instrument_host} is not a valid instrument host - must be one of "
             f"{_VALID_INSTRUMENT_HOSTS}")
    return instrument_host

def ring_add_parser_arguments(parser):
    parser.add_argument(
        'obsid', action='append', nargs='*',
        help='Specific OBSIDs to process')
    parser.add_argument(
        '--ring-type', type=str, default='FMOVIE',
        help='The type of ring mosaics; use to retrieve the file lists')
    parser.add_argument(
        '--corot-type', type=str, default='',
        help='The type of co-rotation frame to use')
    parser.add_argument(
        '--all-obsid', '-a', action='store_true', default=False,
        help='Process all OBSIDs of the given type')
    parser.add_argument(
        '--start-obsid', default='',
        help='The first obsid to process')
    parser.add_argument(
        '--end-obsid', default='',
        help='The last obsid to process')
    parser.add_argument(
        '--ring-radius', type=int, default=0,
        help='The main ring radius; by default loaded from the ring type')
    parser.add_argument(
        '--radius-inner-delta', type=int, default=0,
        help='''The inner delta from the main ring radius;
                by default loaded from the ring type''')
    parser.add_argument(
        '--radius-outer-delta', type=int, default=0,
        help='''The outer delta from the main ring radius;
                by default loaded from the ring type''')
    parser.add_argument(
        '--radius-resolution', type=float, default=0.,
        help='The radial resolution for reprojection')
    parser.add_argument(
        '--longitude-resolution', type=float, default=0.,
        help='The longitudinal resolution for reprojection')
    parser.add_argument(
        '--radial-zoom-amount', type=int, default=0,
        help='The amount of radial zoom for reprojection')
    parser.add_argument(
        '--longitude-zoom-amount', type=int, default=0,
        help='The amount of longitude zoom for reprojection')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--instrument-host', default='cassini',
        type=_validate_instrument_host,
        help=f'''Which spacecraft to process: {_VALID_INSTRUMENT_HOSTS}
                 (cassini is the default)''')


################################################################################
#
# Initialization of globals
#
################################################################################

def ring_init(arguments):
    global RING_FILENAMES
    RING_FILENAMES = {}
    type_dir = file_clean_join(RING_LISTS_ROOT, 'FILELIST_'+
                               arguments.ring_type.upper())

    default_filename = file_clean_join(type_dir, 'defaults.txt')
    assert os.path.exists(default_filename), default_filename
    default_fp = open(default_filename, 'r')
    default_corot = default_fp.readline().strip()
    default_radius = int(default_fp.readline().strip())
    default_radius_inner = int(default_fp.readline().strip())
    default_radius_outer = int(default_fp.readline().strip())
    default_radius_resolution = float(default_fp.readline().strip())
    default_longitude_resolution = float(default_fp.readline().strip())
    default_radial_zoom_amount = int(default_fp.readline().strip())
    default_longitude_zoom_amount = int(default_fp.readline().strip())
    default_fp.close()
    if arguments.corot_type == '':
        arguments.corot_type = default_corot
    if arguments.ring_radius == 0:
        arguments.ring_radius = default_radius
    if arguments.radius_inner_delta == 0:
        arguments.radius_inner_delta = default_radius_inner
    if arguments.radius_outer_delta == 0:
        arguments.radius_outer_delta = default_radius_outer
    if arguments.radius_resolution == 0:
        arguments.radius_resolution = default_radius_resolution
    if arguments.longitude_resolution == 0:
        arguments.longitude_resolution = default_longitude_resolution
    if arguments.radial_zoom_amount == 0:
        arguments.radial_zoom_amount = default_radial_zoom_amount
    if arguments.longitude_zoom_amount == 0:
        arguments.longitude_zoom_amount = default_longitude_zoom_amount

    for obsid_file in sorted(os.listdir(type_dir)):
        if not obsid_file.endswith('.list'):
            continue
        obsid = obsid_file[:-5]
        fp = open(file_clean_join(type_dir, obsid_file), 'r')
        filenames = fp.readlines()
        filenames = [x.strip() for x in filenames if x[0] != '#']
        fp.close()
        # Accept both Nxxxxxx_x and
        # path/path/path/Nxxxxx_x.IMG
        filenames = [f.split('/')[-1].split('.')[0]
                        for f in filenames
                        if f[0] != '#']
        RING_FILENAMES[obsid] = filenames


################################################################################
#
# Enumerate files based on the command line arguments
#
################################################################################

def ring_enumerate_files(arguments, force_obsid=None, yield_obsid_only=False):
    if force_obsid:
        obsid_db = {force_obsid: RING_FILENAMES[force_obsid]}
    elif (arguments.all_obsid or arguments.start_obsid or arguments.end_obsid or
          len(arguments.obsid[0]) == 0):
        obsid_db = RING_FILENAMES
    else:
        obsid_db = {}
        for arg in arguments.obsid[0]:
            if '/' in arg:
                # OBSID/FILENAME
                obsid, filename = arg.split('/')
                if not obsid in obsid_db:
                    obsid_db[obsid] = []
                obsid_db[obsid].append(filename)
            else:
                # OBSID
                obsid_db[arg] = RING_FILENAMES[arg]

        for obsid in sorted(obsid_db.keys()):
            obsid_db[obsid].sort(key=lambda x: (x[1:8] if x [0] == 'C' else
                                                x[1:13]+x[0]))

    found_start_obsid = True
    if arguments.start_obsid:
        found_start_obsid = False

    for obsid in sorted(obsid_db.keys()):
        if not found_start_obsid:
            if obsid != arguments.start_obsid:
                continue
            found_start_obsid = True
        if yield_obsid_only:
            yield obsid, None, None
        else:
            filename_list = obsid_db[obsid]
            for full_path in yield_image_filenames(
                                    restrict_list=filename_list,
                                    instrument_host=arguments.instrument_host):
                _, filename = os.path.split(full_path)
                if filename[0] == 'C': # Voyager
                    yield obsid, filename[:8], full_path
                else:
                    yield obsid, filename[:13], full_path
        if obsid == arguments.end_obsid:
            break


################################################################################
#
# Create command line arguments for any sub-program
#
################################################################################

def ring_basic_cmd_line(arguments):
    ret  = ['--ring-type', arguments.ring_type]
    ret += ['--corot-type', arguments.corot_type]
    ret += ['--ring-radius', str(arguments.ring_radius)]
    ret += ['--radius-inner', str(arguments.radius_inner_delta)]
    ret += ['--radius-outer', str(arguments.radius_outer_delta)]
    ret += ['--radius-resolution', str(arguments.radius_resolution)]
    ret += ['--longitude-resolution', str(arguments.longitude_resolution)]
    ret += ['--radial-zoom-amount', str(arguments.radial_zoom_amount)]
    ret += ['--longitude-zoom-amount', str(arguments.longitude_zoom_amount)]
    ret += ['--instrument-host', arguments.instrument_host]

    return ret


################################################################################
#
# Create path names for various file types and read and write the files
#
################################################################################

def img_to_repro_path_spec(ring_radius,
                           radius_inner, radius_outer,
                           radius_resolution, longitude_resolution,
                           radial_zoom_amount, longitude_zoom_amount,
                           image_path, instrument_host,
                           make_dirs=False):
    repro_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d-REPRO.dat' % (
                      ring_radius, radius_inner, radius_outer,
                      radius_resolution, longitude_resolution,
                      radial_zoom_amount, longitude_zoom_amount))
    repro_path = results_path(image_path+repro_res_data,
                              instrument_host,
                              'ring_repro',
                              root=RING_RESULTS_ROOT,
                              make_dirs=make_dirs)
    return repro_path

def img_to_repro_path(arguments, image_path, instrument_host, make_dirs=False):
    return img_to_repro_path_spec(arguments.ring_radius,
                                  arguments.radius_inner_delta,
                                  arguments.radius_outer_delta,
                                  arguments.radius_resolution,
                                  arguments.longitude_resolution,
                                  arguments.radial_zoom_amount,
                                  arguments.longitude_zoom_amount,
                                  image_path, instrument_host,
                                  make_dirs=make_dirs)

def read_repro(repro_path):
    if not os.path.exists(repro_path):
        return None
    try:
        with open(repro_path, 'rb') as repro_fp:
            repro_data = msgpack.unpackb(repro_fp.read(),
                                         max_str_len=40*1024*1024,
                                         object_hook=msgpack_numpy.decode)
    except UnicodeDecodeError:
        with open(repro_path, 'rb') as repro_fp:
            repro_data = msgpack.unpackb(repro_fp.read(),
                                         max_str_len=40*1024*1024,
                                         object_hook=msgpack_numpy.decode,
                                         raw=True)
            repro_data = fixup_byte_to_str(repro_data)

    if repro_data is None:
        repro_data = {}
    if 'bad_pixel_map' not in repro_data:
        repro_data['bad_pixel_map'] = None
    if 'mean_resolution' in repro_data: # Old format
        repro_data['mean_radial_resolution'] = res = repro_data['mean_resolution']
        del repro_data['mean_resolution']
        repro_data['mean_angular_resolution'] = np.zeros(res.shape)
    if 'long_mask' in repro_data: # Old format
        repro_data['long_antimask'] = repro_data['long_mask']
        del repro_data['long_mask']

    return repro_data

def write_repro(repro_path, repro_data):
    with open(repro_path, 'wb') as repro_fp:
        repro_fp.write(msgpack.packb(repro_data,
                                     default=msgpack_numpy.encode))


def mosaic_paths_spec(ring_radius, radius_inner, radius_outer,
                      radius_resolution, longitude_resolution,
                      radial_zoom_amount, longitude_zoom_amount,
                      obsid, ring_type, make_dirs=False):
    mosaic_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d' % (
                       ring_radius, radius_inner, radius_outer,
                       radius_resolution, longitude_resolution,
                       radial_zoom_amount, longitude_zoom_amount))
    mosaic_dir = file_clean_join(RING_RESULTS_ROOT, 'mosaic_'+ring_type)
    if make_dirs and not os.path.exists(mosaic_dir):
        os.mkdir(mosaic_dir)
    data_path = file_clean_join(mosaic_dir, obsid+mosaic_res_data+'-MOSAIC')
    metadata_path = file_clean_join(mosaic_dir,
                                    obsid+mosaic_res_data+
                                    '-MOSAIC-METADATA.dat')
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

def read_mosaic(data_path, metadata_path):
    img = np.load(data_path+'.npy')

    try:
        with open(metadata_path, 'rb') as mosaic_metadata_fp:
            metadata = msgpack.unpackb(mosaic_metadata_fp.read(),
                                       object_hook=msgpack_numpy.decode)
    except UnicodeDecodeError:
        with open(metadata_path, 'rb') as mosaic_metadata_fp:
            metadata = msgpack.unpackb(mosaic_metadata_fp.read(),
                                       object_hook=msgpack_numpy.decode,
                                       raw=True)
            metadata = fixup_byte_to_str(metadata)

    metadata['img'] = img

    if 'mean_resolution' in metadata: # Old format
        metadata['mean_radial_resolution'] = res = metadata['mean_resolution']
        del metadata['mean_resolution']
        metadata['mean_angular_resolution'] = np.zeros(res.shape)
    if 'long_mask' in metadata: # Old format
        metadata['long_antimask'] = metadata['long_mask']
        del metadata['long_mask']

    return metadata

def write_mosaic(data_path, img, metadata_path, metadata):
    # Save mosaic image array in binary
    np.save(data_path, img)

    # Save metadata
    metadata = metadata.copy() # Everything except img
    del metadata['img']
    mosaic_metadata_fp = open(metadata_path, 'wb')
    mosaic_metadata_fp.write(msgpack.packb(
                                metadata, default=msgpack_numpy.encode))
    mosaic_metadata_fp.close()


def mosaic_png_path_spec(ring_radius, radius_inner, radius_outer,
                         radius_resolution, longitude_resolution,
                         radial_zoom_amount, longitude_zoom_amount,
                         obsid, ring_type, png_type, make_dirs=False):
    mosaic_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d' % (
                       ring_radius, radius_inner, radius_outer,
                       radius_resolution, longitude_resolution,
                       radial_zoom_amount, longitude_zoom_amount))
    png_dir = file_clean_join(RING_RESULTS_ROOT,
                              'png_mosaic_'+png_type+'_'+ring_type)
    if make_dirs and not os.path.exists(png_dir):
        os.mkdir(png_dir)
    png_path = file_clean_join(png_dir,
                               obsid+mosaic_res_data+'-'+png_type+'.png')
    return png_path

def mosaic_png_path(arguments, obsid, png_type, make_dirs=False):
    return mosaic_png_path_spec(arguments.ring_radius,
                                arguments.radius_inner_delta,
                                arguments.radius_outer_delta,
                                arguments.radius_resolution,
                                arguments.longitude_resolution,
                                arguments.radial_zoom_amount,
                                arguments.longitude_zoom_amount,
                                obsid, arguments.ring_type,
                                png_type,
                                make_dirs=make_dirs)

def write_mosaic_pngs(full_png_path, small_png_path, img, mask_fill_value=-999):
    if mask_fill_value is not None:
        img = img.copy()
        img[img == mask_fill_value] = 0
    valid_cols = np.sum(img, axis=0) != 0
    subimg = img[:, valid_cols]
    blackpoint = max(np.min(subimg), 0)
    whitepoint_ignore_frac = 0.995
    img_sorted = sorted(list(subimg.flatten()))
    whitepoint = img_sorted[np.clip(int(len(img_sorted)*
                                        whitepoint_ignore_frac),
                                    0, len(img_sorted)-1)]
    gamma = 0.5

    print('***', blackpoint, whitepoint, gamma)
    # The +0 forces a copy - necessary for PIL
    scaled_mosaic = np.cast['int8'](ImageDisp.scale_image(img, blackpoint,
                                                          whitepoint,
                                                          gamma))[::-1,:]+0
    pil_img = Image.frombuffer('L', (scaled_mosaic.shape[1],
                                     scaled_mosaic.shape[0]),
                               scaled_mosaic, 'raw', 'L', 0, 1)

    pil_img.save(full_png_path, 'PNG')

    # Reduced mosaic for easier viewing
    scale = max(img.shape[1] // 1920, 1)
    scaled_mosaic = np.cast['int8'](ImageDisp.scale_image(img[:,::scale],
                                                          blackpoint,
                                                          whitepoint,
                                                          gamma))[::-1,:]+0
    pil_img = Image.frombuffer('L', (scaled_mosaic.shape[1],
                                     scaled_mosaic.shape[0]),
                           scaled_mosaic, 'raw', 'L', 0, 1)

    pil_img.save(small_png_path, 'PNG')


def bkgnd_paths_spec(ring_radius, radius_inner, radius_outer,
                     radius_resolution, longitude_resolution,
                     radial_zoom_amount, longitude_zoom_amount,
                     obsid, ring_type, make_dirs=False):
    bkgnd_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d_1' % (
                      ring_radius, radius_inner, radius_outer,
                      radius_resolution, longitude_resolution,
                      radial_zoom_amount, longitude_zoom_amount))
    bkgnd_dir = file_clean_join(RING_RESULTS_ROOT, 'bkgnd_'+ring_type)
    if make_dirs and not os.path.exists(bkgnd_dir):
        os.mkdir(bkgnd_dir)
    data_path = file_clean_join(bkgnd_dir, obsid+bkgnd_res_data+'-MOSAIC')
    reduced_mosaic_data_path = file_clean_join(
                     bkgnd_dir,
                     obsid+bkgnd_res_data+'-REDUCED-MOSAIC')
    reduced_mosaic_metadata_path = file_clean_join(
                     bkgnd_dir,
                     obsid+bkgnd_res_data+'-REDUCED-MOSAIC-METADATA.dat')
    bkgnd_model_path = file_clean_join(
                     bkgnd_dir,
                     obsid+bkgnd_res_data+'-BKGND-MODEL.npz')
    bkgnd_metadata_path = file_clean_join(
                     bkgnd_dir,
                     obsid+bkgnd_res_data+'-BKGND-METADATA.dat')

    return (reduced_mosaic_data_path, reduced_mosaic_metadata_path,
            bkgnd_model_path, bkgnd_metadata_path)

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

def bkgnd_sub_mosaic_paths_spec(ring_radius, radius_inner, radius_outer,
                                radius_resolution, longitude_resolution,
                                radial_zoom_amount, longitude_zoom_amount,
                                obsid, ring_type, make_dirs=False):
    bkgnd_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d_1' % (
                      ring_radius, radius_inner, radius_outer,
                      radius_resolution, longitude_resolution,
                      radial_zoom_amount, longitude_zoom_amount))
    bkgnd_dir = file_clean_join(RING_RESULTS_ROOT,
                                'bkgnd_sub_mosaic_'+ring_type)
    if make_dirs and not os.path.exists(bkgnd_dir):
        os.mkdir(bkgnd_dir)
    data_path = file_clean_join(bkgnd_dir,
                                obsid+bkgnd_res_data+'-BKGND-SUB-MOSAIC.npz')
    metadata_path = file_clean_join(
                     bkgnd_dir,
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

def write_bkgnd(model_filename, model, metadata_filename, bkgnddata):
    np.savez_compressed(model_filename, data=model.data, mask=ma.getmaskarray(model))

    bkgnd_metadata_fp = open(metadata_filename, 'wb')
    bkgnd_data = (bkgnddata.row_cutoff_sigmas, bkgnddata.row_ignore_fraction,
                  bkgnddata.row_blur,
                  bkgnddata.ring_lower_limit, bkgnddata.ring_upper_limit,
                  bkgnddata.column_cutoff_sigmas,
                  bkgnddata.column_inside_background_pixels,
                  bkgnddata.column_outside_background_pixels,
                  bkgnddata.degree)
    bkgnd_data = pickle.dump(bkgnd_data, bkgnd_metadata_fp)
    bkgnd_metadata_fp.close()

def write_bkgnd_sub_mosaic(mosaic_filename, img, metadata_filename, bkgnddata):
    np.savez_compressed(mosaic_filename, data=img, mask=bkgnddata.bkgnd_model_mask)

    metadata = {}
    metadata['ring_lower_limit'] = bkgnddata.ring_lower_limit
    metadata['ring_upper_limit'] = bkgnddata.ring_upper_limit
    metadata['radius_resolution'] = bkgnddata.radius_resolution
    metadata['longitude_resolution'] = bkgnddata.longitude_resolution
    metadata['long_mask'] = bkgnddata.long_mask
    metadata['image_number'] = bkgnddata.image_numbers
    metadata['time'] = bkgnddata.ETs
    metadata['mean_emission'] = bkgnddata.emission_angles
    metadata['mean_incidence'] = bkgnddata.incidence_angle
    metadata['mean_phase'] = bkgnddata.phase_angles
    metadata['mean_radial_resolution'] = bkgnddata.radial_resolutions
    metadata['mean_angular_resolution'] = bkgnddata.angular_resolutions
    metadata['longitudes'] = bkgnddata.longitudes
    metadata['obsid_list'] = bkgnddata.obsid_list
    metadata['image_name_list'] = bkgnddata.image_name_list
    metadata['image_path_list'] = bkgnddata.image_path_list
    metadata['repro_path_list'] = bkgnddata.repro_path_list

    mosaic_metadata_fp = open(metadata_filename, 'wb')
    mosaic_metadata_fp.write(msgpack.packb(
                                metadata, default=msgpack_numpy.encode))
    mosaic_metadata_fp.close()

def read_bkgnd_metadata(metadata_filename, bkgnddata):
    print(metadata_filename)
    bkgnd_metadata_fp = open(metadata_filename, 'rb')
    bkgnd_data = pickle.load(bkgnd_metadata_fp)
    (bkgnddata.row_cutoff_sigmas, bkgnddata.row_ignore_fraction, bkgnddata.row_blur,
     bkgnddata.ring_lower_limit, bkgnddata.ring_upper_limit, bkgnddata.column_cutoff_sigmas,
     bkgnddata.column_inside_background_pixels, bkgnddata.column_outside_background_pixels, bkgnddata.degree) = bkgnd_data
    bkgnd_metadata_fp.close()


def bkgnd_png_path_spec(ring_radius, radius_inner, radius_outer,
                        radius_resolution, longitude_resolution,
                        radial_zoom_amount, longitude_zoom_amount,
                        obsid, ring_type, png_type, make_dirs=False):
    bkgnd_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d' % (
                       ring_radius, radius_inner, radius_outer,
                       radius_resolution, longitude_resolution,
                       radial_zoom_amount, longitude_zoom_amount))
    png_dir = file_clean_join(RING_RESULTS_ROOT,
                              'png_bkgnd_'+png_type+'_'+ring_type)
    if make_dirs and not os.path.exists(png_dir):
        os.mkdir(png_dir)
    png_path = file_clean_join(png_dir,
                               obsid+bkgnd_res_data+'-bkgnd-'+png_type+'.png')
    return png_path

def bkgnd_png_path(arguments, obsid, png_type, make_dirs=False):
    return bkgnd_png_path_spec(arguments.ring_radius,
                               arguments.radius_inner_delta,
                               arguments.radius_outer_delta,
                               arguments.radius_resolution,
                               arguments.longitude_resolution,
                               arguments.radial_zoom_amount,
                               arguments.longitude_zoom_amount,
                               obsid, arguments.ring_type,
                               png_type,
                               make_dirs=make_dirs)

def ew_paths_spec(ring_radius, radius_inner, radius_outer,
                  radius_resolution, longitude_resolution,
                  radial_zoom_amount, longitude_zoom_amount,
                  obsid, ring_type, make_dirs=False):
    ew_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d_1' % (
                   ring_radius, radius_inner, radius_outer,
                   radius_resolution, longitude_resolution,
                   radial_zoom_amount, longitude_zoom_amount))
    ew_dir = file_clean_join(RING_RESULTS_ROOT, 'ew_'+ring_type)
    if make_dirs and not os.path.exists(ew_dir):
        os.mkdir(ew_dir)
    data_path = file_clean_join(ew_dir, obsid+ew_res_data)
    metadata_path = file_clean_join(ew_dir, obsid+ew_res_data+'-METADATA.dat')

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

    ew_metadata_fp = open(ew_metadata_filename, 'wb')
    pickle.dump(ew_metadata, ew_metadata_fp)
    ew_metadata_fp.close()


################################################################################
#
# F Ring computations
#
################################################################################

ROTATING_ET = julian.tdb_from_tai(julian.tai_from_iso("2007-01-01"))
FRING_MEAN_MOTION = 581.964

def f_ring_longitude_shift(img_ET):
    return - (FRING_MEAN_MOTION * ((img_ET - ROTATING_ET) / 86400.)) % 360.

def f_ring_inertial_to_corotating(longitude, ET):
    return (longitude + f_ring_longitude_shift(ET)) % 360.

def f_ring_corotating_to_inertial(co_long, ET):
    return (co_long - f_ring_longitude_shift(ET)) % 360.

def f_ring_corotating_to_true_anomaly(co_long, ET):
    return (co_long - f_ring_longitude_shift(ET) - 2.7007*(ET/86400.)) % 360.
