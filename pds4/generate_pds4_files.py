##########################################################################################
# Create all files for the PDS4 achive including binary, tabular, and label.
##########################################################################################

import argparse
from datetime import datetime
import logging
import os
import pickle
import re
import sys
import traceback

import msgpack
import msgpack_numpy
import numpy as np
import numpy.ma as ma
from PIL import Image

import julian
import pdslogger
import pdstemplate

pdslogger.TIME_FMT = '%Y-%m-%d %H:%M:%S'

from pdsparser import PdsLabel

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'external'))

import f_ring_util.f_ring as f_ring

# XML directory structure:
#   bundle.xml                                      [RMS]
#   readme.txt                                      [RF writes]
#   browse_mosaic/
#     collection_browse_mosaic.csv                 +[generated: [P|S], LIDVID]
#     collection_browse_mosaic.xml                  [RMS]
#     OBSID/
#       OBSID_browse_mosaic_full.png               +[generated]
#       OBSID_browse_mosaic_med.png                +[generated]
#       OBSID_browse_mosaic_small.png              +[generated]
#       OBSID_browse_mosaic_thumb.png              +[generated]
#       OBSID_browse_mosaic.xml                    +[template mosaic-browse-image.xml]
#   browse_mosaic_bkg_sub/
#     collection_browse_mosaic_bkg_sub.csv         +[generated: [P|S], LIDVID]
#     collection_browse_mosaic_bkg_sub.xml          [RMS]
#     OBSID/
#       OBSID_browse_mosaic_bkg_sub_full.png       +[generated]
#       OBSID_browse_mosaic_bkg_sub_med.png        +[generated]
#       OBSID_browse_mosaic_bkg_sub_small.png      +[generated]
#       OBSID_browse_mosaic_bkg_sub_thumb.png      +[generated]
#       OBSID_browse_mosaic_bkg_sub.xml            +[template mosaic-browse-image.xml]
#   browse_reproj_img/
#     collection_browse_reproj_img.csv             +[generated: [P|S], LIDVID]
#     collection_browse_reproj_img.xml              [RMS]
#     OBSID/
#       IMG_browse_reproj_img_full.png             +[generated]
#       IMG_browse_reproj_img_thumb.png            +[generated]
#       IMG_browse_reproj_img.xml                  +[template reproj-browse-image.xml]
#   context/
#     [written by RMS]
#   data_mosaic/
#     collection_data_mosaic.csv                   +[generated: [P|S], LIDVID]
#     collection_data_mosaic.xml                    [RMS]
#     OBSID/
#       OBSID_mosaic.img                           +[generated]
#       OBSID_mosaic.xml                           +[template mosaic.xml]
#       OBSID_mosaic_metadata_src_imgs.tab         +[generated]
#       OBSID_mosaic_metadata_params.tab           +[generated]
#       OBSID_mosaic_metadata.xml                  +[template mosaic-metadata.xml]
#   data_mosaic_bkg_sub/
#     collection_data_mosaic_bkg_sub.csv           +[generated: [P|S], LIDVID]
#     collection_data_mosaic_bkg_sub.xml            [RMS]
#     OBSID/
#       OBSID_mosaic_bkg_sub.img                   +[generated]
#       OBSID_mosaic_bkg_sub.xml                   +[template mosaic.xml]
#       OBSID_mosaic_bkg_sub_metadata_src_imgs.tab +[generated]
#       OBSID_mosaic_bkg_sub_metadata_params.tab   +[generated]
#       OBSID_mosaic_bkg_sub_metadata.xml          +[template mosaic-metadata.xml]
#   data_reproj_img/
#     collection_data_reproj_img.csv               +[generated: [P|S], LIDVID]
#     collection_data_reproj_img.xml                [RMS]
#     OBSID/
#       IMG_reproj_img.img                         +[generated]
#       IMG_reproj_img.xml                         +[template reproj-img.xml]
#       IMG_reproj_img_metadata_params.tab         +[generated]
#       IMG_reproj_img_metadata.xml                +[template reproj-img-metadata.xml]
#   document/
#     collection_document.csv                       [RMS]
#     collection_document.xml                       [RMS]
#     document-01.pdf                               [RF writes]
#     document-01.xml                               [RMS]
#   xml_schema/
#     [writted by RMS]                              [RMS]
#
# Internal_Reference:
#   Mosaic: Mosaic Metadata, Mosaic Browse, BSMosaic
#   Mosaic Metadata: Mosaic
#   Mosaic Browse: Mosaic
#   BSMosaic: BSMosaic Metadata, BSMosaic Browse, Mosaic
#   BSMosaic Metadata: BSMosaic
#   BSMosaic Browse: BSMosaic
#   Reproj Image: Reproj Image Metadata, Reproj Image Browse, Mosaic, BSMosaic
#   Reproj Image Metadata: Reproj Image
#   Reproj Image Browse: Reproj Image


##########################################################################################
#
# COMMAND LINE ARGUMENT PROCESSING
#
##########################################################################################


cmd_line = sys.argv[1:]

parser = argparse.ArgumentParser()

parser.add_argument('--output-dir', type=str, default='bundle',
                    help='The root directory for all output files')
parser.add_argument('--log-dir', type=str, default='logs',
                    help='The root directory for all log files')

parser.add_argument('--generate-reproj-labels',
                    action='store_true', default=False,
                    help='Generate reproj labels')
parser.add_argument('--generate-reproj-images',
                    action='store_true', default=False,
                    help='Generate reproj image files')
parser.add_argument('--generate-reproj-collections',
                    action='store_true', default=False,
                    help='Generate reproj collections files')
parser.add_argument('--generate-reproj',
                    action='store_true', default=False,
                    help='Generate reproj image files and labels')

parser.add_argument('--generate-reproj-metadata-labels',
                    action='store_true', default=False,
                    help='Generate reproj table labels')
parser.add_argument('--generate-reproj-metadata-tables',
                    action='store_true', default=False,
                    help='Generate reproj tables')
parser.add_argument('--generate-reproj-metadata',
                    action='store_true', default=False,
                    help='Generate reproj tables and labels')

parser.add_argument('--generate-reproj-browse-labels',
                    action='store_true', default=False,
                    help='Generate reproj browse labels')
parser.add_argument('--generate-reproj-browse-images',
                    action='store_true', default=False,
                    help='Generate reproj browse image files')
parser.add_argument('--generate-reproj-browse-collections',
                    action='store_true', default=False,
                    help='Generate reproj browse image '
                         'collections files')
parser.add_argument('--generate-reproj-browse',
                    action='store_true', default=False,
                    help='Generate reproj browse image files and labels')

parser.add_argument('--generate-all-reproj',
                    action='store_true', default=False,
                    help='Generate all reproj image, metadata, and browse files with '
                         'labels as well as associated collections files')


parser.add_argument('--generate-mosaic-labels',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub mosaic labels')
parser.add_argument('--generate-mosaic-images',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub mosaic image files')
parser.add_argument('--generate-mosaic-collections',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub mosaic collections files')
parser.add_argument('--generate-mosaics',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub mosaic image files and labels')

parser.add_argument('--generate-mosaic-metadata-labels',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub metadata table labels')
parser.add_argument('--generate-mosaic-metadata-tables',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub metadata tables')
parser.add_argument('--generate-mosaic-metadata',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub metadata tables and labels')

parser.add_argument('--generate-mosaic-browse-labels',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub mosaic browse labels')
parser.add_argument('--generate-mosaic-browse-images',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub mosaic browse image files')
parser.add_argument('--generate-mosaic-browse-collections',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub mosaic browse image '
                         'collections files')
parser.add_argument('--generate-mosaic-browse',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub mosaic browse image files and '
                         'labels')

parser.add_argument('--generate-all-mosaics',
                    action='store_true', default=False,
                    help='Generate all mosaic image, metadata, and browse files with '
                         'labels as well as associated collections files')

parser.add_argument('--generate-all',
                    action='store_true', default=False,
                    help='Generate all files and labels')

f_ring.add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)

f_ring.init(arguments)


CALIBRATED_DIR = '/data/pdsdata/holdings/calibrated' # XXX
REPROJ_DIR = '/data/cb-results/fring/ring_mosaic/ring_repro' # XXX

GENERATE_REPROJ_IMAGES = False
GENERATE_REPROJ_IMAGE_LABELS = False
GENERATE_REPROJ_METADATA_TABLES = False
GENERATE_REPROJ_METADATA_LABELS = False
GENERATE_REPROJ_COLLECTIONS = False
GENERATE_REPROJ_BROWSE_IMAGES = False
GENERATE_REPROJ_BROWSE_LABELS = False
GENERATE_REPROJ_BROWSE_COLLECTIONS = False

GENERATE_MOSAIC_IMAGES = False
GENERATE_MOSAIC_IMAGE_LABELS = False
GENERATE_MOSAIC_METADATA_TABLES = False
GENERATE_MOSAIC_METADATA_LABELS = False
GENERATE_MOSAIC_COLLECTIONS = False
GENERATE_MOSAIC_BROWSE_IMAGES = False
GENERATE_MOSAIC_BROWSE_LABELS = False
GENERATE_MOSAIC_BROWSE_COLLECTIONS = False

if arguments.generate_reproj_labels:
    GENERATE_REPROJ_IMAGE_LABELS = True
if arguments.generate_reproj_images:
    GENERATE_REPROJ_IMAGES = True
if (arguments.generate_reproj_collections or
    arguments.generate_all_reproj or
    arguments.generate_all):
    GENERATE_REPROJ_COLLECTIONS = True
if (arguments.generate_reproj or
    arguments.generate_all_reproj or
    arguments.generate_all):
    GENERATE_REPROJ_IMAGE_LABELS = True
    GENERATE_REPROJ_IMAGES = True

if arguments.generate_reproj_metadata_labels:
    GENERATE_REPROJ_METADATA_LABELS = True
if arguments.generate_reproj_metadata_tables:
    GENERATE_REPROJ_METADATA_TABLES = True
if (arguments.generate_reproj_metadata or
    arguments.generate_all_reproj or
    arguments.generate_all):
    GENERATE_REPROJ_METADATA_LABELS = True
    GENERATE_REPROJ_METADATA_TABLES = True

if arguments.generate_reproj_browse_labels:
    GENERATE_REPROJ_BROWSE_LABELS = True
if arguments.generate_reproj_browse_images:
    GENERATE_REPROJ_BROWSE_IMAGES = True
if (arguments.generate_reproj_browse_collections or
    arguments.generate_all_reproj or
    arguments.generate_all):
    GENERATE_REPROJ_BROWSE_COLLECTIONS = True
if (arguments.generate_reproj_browse or
    arguments.generate_all_reproj or
    arguments.generate_all):
    GENERATE_REPROJ_BROWSE_LABELS = True
    GENERATE_REPROJ_BROWSE_IMAGES = True

if arguments.generate_mosaic_labels:
    GENERATE_MOSAIC_IMAGE_LABELS = True
if arguments.generate_mosaic_images:
    GENERATE_MOSAIC_IMAGES = True
if (arguments.generate_mosaic_collections or
    arguments.generate_all_mosaics or
    arguments.generate_all):
    GENERATE_MOSAIC_COLLECTIONS = True
if (arguments.generate_mosaics or
    arguments.generate_all_mosaics or
    arguments.generate_all):
    GENERATE_MOSAIC_IMAGE_LABELS = True
    GENERATE_MOSAIC_IMAGES = True

if arguments.generate_mosaic_metadata_labels:
    GENERATE_MOSAIC_METADATA_LABELS = True
if arguments.generate_mosaic_metadata_tables:
    GENERATE_MOSAIC_METADATA_TABLES = True
if (arguments.generate_mosaic_metadata or
    arguments.generate_all_mosaics or
    arguments.generate_all):
    GENERATE_MOSAIC_METADATA_LABELS = True
    GENERATE_MOSAIC_METADATA_TABLES = True

if arguments.generate_mosaic_browse_labels:
    GENERATE_MOSAIC_BROWSE_LABELS = True
if arguments.generate_mosaic_browse_images:
    GENERATE_MOSAIC_BROWSE_IMAGES = True
if (arguments.generate_mosaic_browse_collections or
    arguments.generate_all_mosaics or
    arguments.generate_all):
    GENERATE_MOSAIC_BROWSE_COLLECTIONS = True
if (arguments.generate_mosaic_browse or
    arguments.generate_all_mosaics or
    arguments.generate_all):
    GENERATE_MOSAIC_BROWSE_LABELS = True
    GENERATE_MOSAIC_BROWSE_IMAGES = True


##########################################################################################
#
# LOGGER INITIALIZATION
#
##########################################################################################

LOGGER = pdslogger.PdsLogger('fring.pds4')

LOG_DIR = arguments.log_dir
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_INFO = os.path.join(LOG_DIR, 'generate_pd4.log')
LOG_FILE_DEBUG = os.path.join(LOG_DIR, 'generate_pds4_debug.log')

info_handler = pdslogger.file_handler(LOG_FILE_INFO, level=logging.INFO,
                                      rotation='ymdhms')
debug_handler = pdslogger.file_handler(LOG_FILE_DEBUG, level=logging.DEBUG,
                                       rotation='ymdhms')

LOGGER.add_handler(info_handler)
LOGGER.add_handler(debug_handler)
LOGGER.add_handler(pdslogger.stdout_handler)

handler = pdslogger.warning_handler(LOG_DIR, rotation='none')
LOGGER.add_handler(handler)

handler = pdslogger.error_handler(LOG_DIR, rotation='none')
LOGGER.add_handler(handler)

pdstemplate.PdsTemplate.set_logger(LOGGER)


##########################################################################################
#
# LONG STATIC XML STRINGS
#
##########################################################################################

TARGET_PROMETHEUS = """
        <Target_Identification>
            <!-- Use when appropriate -->
            <name>Prometheus</name>
            <alternate_designation>Saturn XVI (Prometheus)</alternate_designation>
            <alternate_designation>S/1980 S 27</alternate_designation>
            <alternate_designation>NAIF ID 616</alternate_designation>
            <type>Satellite</type>
            <description>
                NAIF ID: 616;
                Center of motion: Saturn;
                LID of central body: urn:nasa:pds:context:target:planet.saturn;
                NAIF ID of central body: 699.
            </description>
            <Internal_Reference>
                <lid_reference>urn:nasa:pds:context:target:satellite.saturn.prometheus</lid_reference>
                <reference_type>ancillary_to_target</reference_type>
            </Internal_Reference>
        </Target_Identification>
"""

TARGET_PANDORA = """
        <Target_Identification>
            <!-- Use when appropriate -->
            <name>Pandora</name>
            <alternate_designation>Saturn XVII (Pandora)</alternate_designation>
            <alternate_designation>S/1980 S 26</alternate_designation>
            <alternate_designation>NAIF ID 617</alternate_designation>
            <type>Satellite</type>
            <description>
                NAIF ID: 617;
                Center of motion: Saturn;
                LID of central body: urn:nasa:pds:context:target:planet.saturn;
                NAIF ID of central body: 699.
            </description>
            <Internal_Reference>
                <lid_reference>urn:nasa:pds:context:target:satellite.saturn.pandora</lid_reference>
                <reference_type>ancillary_to_target</reference_type>
            </Internal_Reference>
        </Target_Identification>
"""


##########################################################################################
#
# UTILITIY FUNCTIONS
#
##########################################################################################

class ObsIdFailedException(Exception):
    """Fatal error with current obsid. Can continue to next."""
    pass


def et_to_datetime(et):
    """Convert a SPICE ET to a datetime like 2020-01-01T00:00:00Z."""
    return julian.ymdhms_format_from_tai(julian.tai_from_tdb(et)) + 'Z'


def downsample(img, amt0, amt1):
    """Downsample an image by taking the mean of slices across longitude."""
    # Crop to an integral multiple of amt
    cropped_size0 = (img.shape[0] // amt0) * amt0
    cropped_size1 = (img.shape[1] // amt1) * amt1
    img = img[:cropped_size0, :cropped_size1]
    img = np.mean(img.reshape(img.shape[0]//amt0, amt0, -1), axis=1)
    img = np.mean(img.reshape(-1, img.shape[1]//amt1, amt1), axis=2)
    return img


def pad_image(image, margin):
    """Pad an image with a zero-filled margin on each edge."""
    if margin[0] == 0 and margin[1] == 0:
        return image
    new_image = np.zeros((image.shape[0]+margin[0]*2,image.shape[1]+margin[1]*2),
                         dtype=image.dtype)
    new_image[margin[0]:margin[0]+image.shape[0],
              margin[1]:margin[1]+image.shape[1], ...] = image
    return new_image


def img_to_repro_path(image_path):
    components = image_path.split('/')
    vol = components[-4]
    sclk_dir = components[-2]
    image_name = components[-1]
    image_name = image_name.replace('_CALIB.IMG',
                                    '_140220_-01000_001000_05.000_0.020_10_1-REPRO.DAT')
    return os.path.join(REPROJ_DIR, vol, sclk_dir, image_name)


def populate_template(obsid, template_name, output_path, xml_metadata):
    """Copy a template to an output file after making substitutions."""
    template = pdstemplate.PdsTemplate(os.path.join('templates', template_name))
    xml = template.generate(xml_metadata)#, terminator='\n')
    remaining = re.findall(r'\$([^$]+)\$', xml)
    if remaining:
        for remain in remaining:
            LOGGER.error(f'{obsid}: Template {template_name} - Missed metadata '
                         f'field "{remain}"')

    with open(output_path, 'w') as output_fp:
        output_fp.write(xml)


def fixup_byte_to_str(data):
    """Fixup a msgpack'd metadata structure to use Unicode strings not bytes."""
    if (isinstance(data, (str, float, int, bool,
                          np.bool_, np.float32, np.ndarray))
        or data is None):
        return data
    if isinstance(data, bytes):
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            # This will happen for things like image overlays
            return data
    if isinstance(data, list):
        return [fixup_byte_to_str(x) for x in data]
    if isinstance(data, tuple):
        return tuple([fixup_byte_to_str(x) for x in data])
    if isinstance(data, dict):
        new_data = {}
        for key in data:
            new_data[key.decode('utf-8')] = fixup_byte_to_str(data[key])
        return new_data
    LOGGER.error('Unknown type in fixup_byte_to_str', type(data))
    return data


def read_mosaic(data_path, metadata_path, *, bkg_sub=False, read_img=True):
    """Read a main or background-subtracted mosaic and associated metadata."""
    try:
        with open(metadata_path, 'rb') as metadata_fp:
            metadata = msgpack.unpackb(metadata_fp.read(),
                                       object_hook=msgpack_numpy.decode)
    except UnicodeDecodeError:
        with open(metadata_path, 'rb') as metadata_fp:
            metadata = msgpack.unpackb(metadata_fp.read(),
                                       object_hook=msgpack_numpy.decode,
                                       raw=True)
            metadata = fixup_byte_to_str(metadata)

    if read_img:
        if bkg_sub:
            with np.load(data_path) as npz:
                metadata['img'] = ma.MaskedArray(**npz)
                # The background image mask shows the "bad pixels"
                metadata['img'].mask = False
        else:
            metadata['img'] = ma.MaskedArray(np.load(data_path))

    metadata['longitudes'] = (np.arange(len(metadata['long_mask'])) *
                              arguments.longitude_resolution)
    metadata['inertial_longitudes'] = np.degrees(f_ring.fring_corotating_to_inertial(
                                np.radians(metadata['longitudes']),
                                metadata['time']))

    return metadata


def read_bkgnd_metadata(model_path, metadata_path):
    """Read background model metadata."""
    metadata = {}
    with open(metadata_path, 'rb') as bkgnd_metadata_fp:
        bkgnd_data = pickle.load(bkgnd_metadata_fp)
    metadata['row_cutoff_sigmas'] = bkgnd_data[0]
    metadata['row_ignore_fraction'] = bkgnd_data[1]
    metadata['row_blur'] = bkgnd_data[2]
    metadata['ring_lower_limit'] = bkgnd_data[3]
    metadata['ring_upper_limit'] = bkgnd_data[4]
    metadata['column_cutoff_sigmas'] = bkgnd_data[5]
    metadata['column_inside_background_pixels'] = bkgnd_data[6]
    metadata['column_outside_background_pixels'] = bkgnd_data[7]
    metadata['degree'] = bkgnd_data[8]
    with np.load(model_path) as npz:
        metadata['bkgnd_model'] = ma.MaskedArray(**npz)


def read_reproj(metadata_path):
    """Read a reproject image metadata."""
    try:
        with open(metadata_path, 'rb') as metadata_fp:
            metadata = msgpack.unpackb(metadata_fp.read(),
                                       object_hook=msgpack_numpy.decode)
    except UnicodeDecodeError:
        with open(metadata_path, 'rb') as metadata_fp:
            metadata = msgpack.unpackb(metadata_fp.read(),
                                       object_hook=msgpack_numpy.decode,
                                       raw=True)
            metadata = fixup_byte_to_str(metadata)

    metadata['longitudes'] = (np.arange(len(metadata['long_mask'])) *
                              arguments.longitude_resolution)
    metadata['inertial_longitudes'] = np.degrees(f_ring.fring_corotating_to_inertial(
                                np.radians(metadata['longitudes']),
                                metadata['time']))

    return metadata


def mosaic_has_prometheus(metadata):
    """Return True if Prometheus is present in the mosaic."""
    return False # XXX


def mosaic_has_pandora(metadata):
    """Return True if Pandora is present in the mosaic."""
    return False # XXX


def remap_image_indexes(metadata):
    """Remap the image indexes to be contiguous starting at 0.

    This is necessary in case any of the images that went into building the
    mosaic didn't actually get used. This also is going to limit which
    reprojected images we include, because if an image wasn't used to make the
    mosaic, we never checked to see if it was navigated properly.
    """
    image_indexes = metadata['image_number']
    image_name_list = metadata['image_name_list']
    image_path_list = metadata['image_path_list']
    used_indexes = sorted(set(image_indexes) - set([SENTINEL]))
    number_map = {SENTINEL: SENTINEL}
    for i in range(len(used_indexes)):
        number_map[used_indexes[i]] = i
    new_image_indexes = [number_map[x] for x in image_indexes]
    metadata['image_number'] = np.array(new_image_indexes)

    # Only include images that we actually used in the name list
    new_image_name_list = [image_name_list[number_map[x]]
                               for x in number_map.keys() if x != SENTINEL]
    metadata['image_name_list'] = new_image_name_list
    new_image_path_list = [image_path_list[number_map[x]]
                               for x in number_map.keys() if x != SENTINEL]
    metadata['image_path_list'] = new_image_path_list


def image_name_to_lidvid(name): ### Convert to LIDVID ??? XXX
    """Convert Cassini ISS image name to a LIDVID.

    urn:nasa:pds:cassini_iss_saturn:data_raw:1454725799n
    """
    name = name.lower()
    return ( 'urn:nasa:pds:cassini_iss_saturn:data_calibrated:'
            f'{name[1:11]}{name[0]}::1.0')


def image_name_to_reproj_lid(name):
    """Convert Cassini ISS image name to a reprojected image LID.

    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_reproj_img:
    1551253524n_reproj_img
    """
    name = name.lower()
    return ( 'urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_reproj_img:'
            f'{name[1:11]}{name[0]}_reproj_img')


def image_name_to_reproj_lidvid(name):
    """Convert Cassini ISS image name to a reprojected image LIDVID.

    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_reproj_img:
    1551253524n_reproj_img::1.0
    """
    return image_name_to_reproj_lid(name)+'::1.0'


def image_name_to_reproj_metadata_lid(name):
    """Convert Cassini ISS image name to a reprojected image metadata LID.

    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_reproj_img:
    1551253524n_reproj_img_metadata
    """
    name = name.lower()
    return ( 'urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_reproj_img:'
            f'{name[1:11]}{name[0]}_reproj_img_metadata')


def image_name_to_reproj_metadata_lidvid(name):
    """Convert Cassini ISS image name to a reprojected image metadata LIDVID.

    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_reproj_img:
    1551253524n_reproj_img_metadata::1.0
    """
    return image_name_to_reproj_metadata_lid(name)+'::1.0'


def image_name_to_reproj_browse_lid(name):
    """Convert Cassini ISS image name to a reprojected browse image LID.

    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:browse_reproj_img:
    1551253524n_browse_reproj_img
    """
    name = name.lower()
    return ( 'urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:browse_reproj_img:'
            f'{name[1:11]}{name[0]}_browse_reproj_img')


def image_name_to_reproj_browse_lidvid(name):
    """Convert Cassini ISS image name to a reprojected browse image LIDVID.

    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:browse_reproj_img:
    1551253524n_browse_reproj_img::1.0
    """
    return image_name_to_reproj_browse_lid(name)+'::1.0'


def obsid_to_mosaic_lid(obsid, bkg_sub):
    """Convert OBSID IOSIC_276RB_COMPLITB4001_SI to a mosaic or bsm LID.

    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_mosaic:
    iosic_276rb_complitb4001_si_mosaic
        or
    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_mosaic_bkg_sub:
    iosic_276rb_complitb4001_si_mosaic_bkg_sub
    """
    sfx = '_bkg_sub' if bkg_sub else ''
    obsid = obsid.lower()
    return ( 'urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:'
            f'data_mosaic{sfx}:{obsid}_mosaic{sfx}')


def obsid_to_mosaic_lidvid(obsid, bkg_sub):
    """Convert OBSID IOSIC_276RB_COMPLITB4001_SI to a mosaic or bsm LIDVID.

    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_mosaic:
    iosic_276rb_complitb4001_si_mosaic::1.0
        or
    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_mosaic_bkg_sub:
    iosic_276rb_complitb4001_si_mosaic_bkg_sub::1.0
    """
    return obsid_to_mosaic_lid(obsid, bkg_sub)+'::1.0'


def obsid_to_mosaic_metadata_lid(obsid, bkg_sub):
    """Convert OBSID to a mosaic or bsm metadata LID.

    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_mosaic:
    iosic_276rb_complitb4001_si_mosaic_metadata
        or
    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_mosaic_bkg_sub:
    iosic_276rb_complitb4001_si_mosaic_bkg_sub_metadata
    """
    sfx = '_bkg_sub' if bkg_sub else ''
    obsid = obsid.lower()
    return ( 'urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:'
            f'data_mosaic{sfx}:{obsid}_mosaic{sfx}_metadata')


def obsid_to_mosaic_metadata_lidvid(obsid, bkg_sub):
    """Convert OBSID to a mosaic or bsm metadata LIDVID.

    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_mosaic:
    iosic_276rb_complitb4001_si_metadata::1.0
        or
    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_mosaic_bkg_sub:
    iosic_276rb_complitb4001_si_metadata_bkg_sub::1.0
    """
    return obsid_to_mosaic_metadata_lid(obsid, bkg_sub)+'::1.0'


def obsid_to_mosaic_browse_lid(obsid, bkg_sub):
    """Convert OBSID to a mosaic or bsm browse LID.

    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:browse_mosaic:
    iosic_276rb_complitb4001_si_browse_mosaic
        or
    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:browse_mosaic_bkg_sub:
    iosic_276rb_complitb4001_si_browse_mosaic_bkg_sub
    """
    sfx = '_bkg_sub' if bkg_sub else ''
    obsid = obsid.lower()
    return ( 'urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:'
            f'browse_mosaic{sfx}:{obsid}_browse_mosaic{sfx}')


def obsid_to_mosaic_browse_lidvid(obsid, bkg_sub):
    """Convert OBSID to a mosaic or bsm browse LIDVID.

    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:browse_mosaic:
    iosic_276rb_complitb4001_si_browse_mosaic::1.0
        or
    urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:browse_mosaic_bkg_sub:
    iosic_276rb_complitb4001_si_browse_mosaic_bkg_sub::1.0
    """
    return obsid_to_mosaic_browse_lid(obsid, bkg_sub)+'::1.0'


def et_to_tour(et):
    """Convert ET to PDS4 Cassini Tour name.

    See https://github.com/pds-data-dictionaries/ldd-cassini/blob/main/src/
        PDS4_CASSINI_IngestLDD.xml
    """
    datetime = et_to_datetime(et)
    if datetime <= '2004-12-24':
        return 'Tour Pre-Huygens'
    if datetime <= '2008-06-30':
        return 'Tour'
    if datetime <= '2010-09-29':
        return 'Extended Mission'
    return 'Extended-Extended Mission'


def read_label(image_name):
    """Return the PDS3 label for the given image name.

    This is needed to lookup the spacecraft clock counts, which aren't
    stored in the mosaic metadata.
    """
    components = image_name.split('/')[-5:]
    image_path = os.path.join(CALIBRATED_DIR, *components)
    label_path = image_path.replace('.IMG', '.LBL')
    return PdsLabel.from_file(label_path)


##########################################################################################
#
# PROCESS A MOSAIC
#
##########################################################################################

def xml_metadata_for_image(obsid, metadata, img_type):
    """Generate the common template substitions for all image types.

        img_type is 'm' (basic mosasic), 'b' (bsm), or 'r' (reprojected image).
    """
    ret = BASIC_XML_METADATA.copy()

    long_mask = metadata['long_mask']

    match = re.search(r'^(.*)_(\d+)$', obsid)
    partial_obsid = bool(match)
    root_obsid = obsid
    if partial_obsid:
        root_obsid = match[1]
        obsid_chunk = match[2]

    sfx = None
    cap_bkg = None
    num_images = None

    if img_type == 'r':
        max_et = min_et = metadata['time']
    else:
        sfx = '_bkg_sub' if img_type == 'b' else ''
        cap_bkg = 'Background-subtracted ' if img_type == 'b' else ''
        num_images = len(metadata['image_path_list'])

        ETs = metadata['time'][long_mask]
        min_et = np.min(ETs)
        max_et = np.max(ETs)

    ret['START_DATE_TIME'] = start_date = et_to_datetime(min_et)
    ret['STOP_DATE_TIME'] = stop_date = et_to_datetime(max_et)
    total_hours = (max_et - min_et) / 3600

    ret['TOUR'] = et_to_tour(min_et)

    num_good_long = np.sum(long_mask)
    min_corot_long = np.where(long_mask)[0][0] / len(long_mask) * 360
    max_corot_long = np.where(long_mask)[0][-1] / len(long_mask) * 360
    diff_corot = max_corot_long - min_corot_long + 1 / len(long_mask) * 360
    deg_good_long = num_good_long / len(long_mask) * 360
    inertial_longitudes = metadata['inertial_longitudes'][long_mask]
    min_inertial = np.min(inertial_longitudes)
    max_inertial = np.max(inertial_longitudes)
    diff_inertial = max_inertial - min_inertial

    ret['OBSERVATION_ID'] = obsid
    ret['FILTER1'] = 'CL1'
    ret['FILTER2'] = 'CL2'
    ret['NUM_VALID_LONGITUDES'] = str(np.sum(long_mask))
    ret['MOSAIC_LID'] = obsid_to_mosaic_lid(obsid, img_type == 'b')
    ret['MOSAIC_ORIGINAL_LID'] = obsid_to_mosaic_lid(obsid, False)
    ret['MOSAIC_BKG_SUB_LID'] = obsid_to_mosaic_lid(obsid, True)
    ret['MOSAIC_METADATA_LID'] = obsid_to_mosaic_metadata_lid(obsid,
                                                              img_type == 'b')
    ret['MOSAIC_BROWSE_LID'] = obsid_to_mosaic_browse_lid(obsid, img_type == 'b')
    if img_type == 'b':
        ret['MOSAIC_OTHER_LID'] = ret['MOSAIC_ORIGINAL_LID']
        ret['MOSAIC_OTHER_REFERENCE_COMMENT'] = """
            The original mosaic without the background subtracted."""
        ret['MOSAIC_REFERENCE_COMMENT'] = 'The mosaic with the background subtracted.'
        ret['MOSAIC_BROWSE_COMMENT'] = """
            Browse images of the background-subtracted mosaic in multiple sizes
            in PNG format."""
    else:
        ret['MOSAIC_OTHER_LID'] = ret['MOSAIC_BKG_SUB_LID']
        ret['MOSAIC_REFERENCE_COMMENT'] = """
            The original mosaic without the background subtracted."""
        ret['MOSAIC_OTHER_REFERENCE_COMMENT'] = """
            The mosaic with the background subtracted."""
        ret['MOSAIC_BROWSE_COMMENT'] = """
            Browse images of the mosaic in multiple sizes in PNG format."""

    if img_type == 'r':
        max_image_path = min_image_path = metadata['image_path']
        image_name = metadata['image_name']
        ret['REPROJ_METADATA_LID'] = image_name_to_reproj_lid(image_name)
        ret['REPROJ_TITLE'] = f"""
Reprojected version of Cassini ISS calibrated image {image_name} from
observation {root_obsid} taken at {start_date}
"""
        ret['REPROJ_METADATA_TITLE'] = f"""
Metadata for the reprojected version of Cassini ISS calibrated image
{image_name} from observation {root_obsid} taken at {start_date}
"""
        ret['REPROJ_LID'] = image_name_to_reproj_lid(image_name)
        ret['REPROJ_BROWSE_LID'] = image_name_to_reproj_browse_lid(image_name)
        ret['REPROJ_METADATA_LID'] = image_name_to_reproj_metadata_lid(image_name)

        ret['REPROJ_DESCRIPTION'] = ret['REPROJ_TITLE']

        ret['REPROJ_COMMENT'] = f"""
This data file is an individual reprojected image of Saturn's F ring from
Cassini ISS image {image_name} taken at {start_date}. In this image, Cassini
observed an area of space covering {diff_inertial:.2f} degrees of inertial
longitude from {min_inertial:.2f} to {max_inertial:.2f}. The source image was
calibrated using CISSCAL 4.0 and the data values are in units of I/F. The
mosaics, in the data_mosaic and data_mosaic_bkg_sub collections, were generated
by stitching together reprojected, calibrated images such as this.


The reprojection takes the image space and reprojects it onto a regular
radius/longitude grid, where the longitude (sampled at 0.02 degrees) is
co-rotating with the core of the F ring and the radius (sampled at 5 km) is
relative to the position of the core at that longitude and time given a
particular model of the F ring's orbit (in other words, even though the F ring
is eccentric, in the mosaic it looks like a straight line at constant radius).
We use the F ring orbit from Albers et al. (2009), fit #2. The co-rotating
longitude is calculated using the epoch 2007-01-01T00:00:00Z, meaning this was
the instant when co-rotating and inertial longitudes were the same. This
reprojected image contains valid data for a total of {deg_good_long:.2f} degrees
of co-rotating longitude spanning the {diff_corot:.2f} degrees from
{min_corot_long:.2f} to {max_corot_long:.2f}.
"""
        ret['REPROJ_RINGS_DESCRIPTION'] = ret['REPROJ_COMMENT'] + f"""


The following parameters in this class use the Albers 2009 model:
epoch_reprojection_basis_utc is the date and time of zero longitude of the
rotating frame corotation_rate is the mean corotation rate


Mean/min/max values are based upon the aggregate of the source images for the
following parameters: phase angle, observed ring elevation, ring
longitude,...etc XXX
"""

        ret['REPROJ_METADATA_DESCRIPTION'] = ret['REPROJ_METADATA_TITLE']
        ret['REPROJ_METADATA_COMMENT'] = f"""
One file containing metadata parameters per corotating longitude for the
reprojected version of the Cassini ISS calibrated image {image_name} from
{root_obsid} taken at {start_date}.
"""
        ret['REPROJ_METADATA_RINGS_DESCRIPTION'] = ret['REPROJ_METADATA_DESCRIPTION']

        ret['REPROJ_IMG_FILENAME'] = f'{image_name.lower()}_reproj_img.img'

    else:
        # Find the image names at the starting and ending ETs
        image_indexes = metadata['image_number'][long_mask]
        image_path_list = metadata['image_path_list']
        image_name_list = metadata['image_name_list']
        idx_min = np.argmin(ETs)
        idx_max = np.argmax(ETs)
        min_image_path = image_path_list[image_indexes[idx_min]]
        min_image_name = image_name_list[image_indexes[idx_min]]
        max_image_path = image_path_list[image_indexes[idx_max]]
        max_image_name = image_name_list[image_indexes[idx_max]]

        ret['MOSAIC_TITLE'] = f"""
{cap_bkg}F Ring mosaic created from reprojected Cassini ISS calibrated images
from observation {root_obsid} spanning {start_date} ({min_image_name}) to
{stop_date} ({max_image_name})
"""
        ret['MOSAIC_METADATA_TITLE'] = f"""
Metadata for the {cap_bkg.lower()}F Ring mosaic created from reprojected Cassini
ISS calibrated images from observation {root_obsid} spanning {start_date}
({min_image_name}) to {stop_date} ({max_image_name})
"""

        ret['MOSAIC_DESCRIPTION'] = ret['MOSAIC_TITLE']

        partial_comment = ''
        if partial_obsid:
            partial_comment = f"""
Because Cassini observed multiple distinct inertial longitudes during
{root_obsid}, each making its own "movie", we have split the observation into
multiple chunks. This mosaic consists of {root_obsid} chunk {obsid_chunk}. Other
mosaics are available in this bundle for {root_obsid} with different date ranges
representing the other available observation chunks.
"""
        bkg_comment = ''
        if img_type == 'b':
            bkg_comment = """

Background subtraction was performed by creating, for each longitude, a linear
model based on the available data from 750 to 1000 km on either side of the F
ring core. Obviously bad pixels (such as stars or moons) were ignored. If
insufficient data was available to generate the model, that longitude was marked
as invalid and removed from the mosaic. As such, the longitudes available in the
background-subtracted mosaic may be less than those available in the original
mosaic.
"""
        ret['MOSAIC_COMMENT'] = f"""
This data file is a {cap_bkg.lower()}mosaic of Saturn's F ring, stitched
together from reprojections of {num_images} source images from Cassini
Observation Name {root_obsid} spanning {start_date} ({min_image_name}) to
{stop_date} ({max_image_name}). During this time, Cassini repeatedly observed an
area of space covering {diff_inertial:.2f} degrees of inertial longitude from
{min_inertial:.2f} to {max_inertial:.2f} while the ring rotated under it for
{total_hours:.2f} hours. The source images were calibrated using CISSCAL 4.0 and
the data values are in units of I/F.{partial_comment}


The reprojection takes the image space and reprojects it onto a regular
radius/longitude grid, where the longitude (sampled at 0.02 degrees) is
co-rotating with the core of the F ring and the radius (sampled at 5 km) is
relative to the position of the core at that longitude and time given a
particular model of the F ring's orbit (in other words, even though the F ring
is eccentric, in the mosaic it looks like a straight line at constant radius).
We use the F ring orbit from Albers et al. (2009), fit #2. The co-rotating
longitude is calculated using the epoch 2007-01-01T00:00:00Z, meaning this was
the instant when co-rotating and inertial longitudes were the same. This mosaic
image contains valid data for a total of {deg_good_long:.2f} degrees of
co-rotating longitude spanning the {diff_corot:.2f} degrees from
{min_corot_long:.2f} to {max_corot_long:.2f}.{bkg_comment}
"""
        ret['MOSAIC_RINGS_DESCRIPTION'] = ret['MOSAIC_COMMENT'] + f"""


The following parameters in this class use the Albers 2009 model:
epoch_reprojection_basis_utc is the date and time of zero longitude of the
rotating frame corotation_rate is the mean corotation rate


Mean/min/max values are based upon the aggregate of the source images for the
following parameters: phase angle, observed ring elevation, ring
longitude,...etc XXX
"""

        ret['MOSAIC_METADATA_DESCRIPTION'] = ret['MOSAIC_METADATA_TITLE']
        ret['MOSAIC_METADATA_COMMENT'] = f"""
Two files containing metadata for the {cap_bkg.lower()}mosaics created from
reprojected Cassini ISS calibrated images from {root_obsid}, {start_date} to
{stop_date}:

    1) Indices and LIDs of source images

    2) Metadata parameters per corotating longitude
"""
        ret['MOSAIC_METADATA_RINGS_DESCRIPTION'] = ret['MOSAIC_METADATA_DESCRIPTION']

        ret['MOSAIC_IMG_FILENAME'] = f'{obsid.lower()}_mosaic{sfx}.img'

    try:
        min_label = read_label(min_image_path)
    except FileNotFoundError:
        LOGGER.error(f'{obsid}: Failed to open label file {min_image_path}')
        raise ObsIdFailedException
    ret['SPACECRAFT_CLOCK_START_COUNT'] = str(min_label['SPACECRAFT_CLOCK_START_COUNT'])
    if min_image_path == max_image_path:
        ret['SPACECRAFT_CLOCK_STOP_COUNT'] = str(min_label['SPACECRAFT_CLOCK_STOP_COUNT'])
    else:
        try:
            max_label = read_label(max_image_path)
        except FileNotFoundError:
            LOGGER.error(f'{obsid}: Failed to open label file {max_image_path}')
            raise ObsIdFailedException
        ret['SPACECRAFT_CLOCK_STOP_COUNT'] = str(max_label['SPACECRAFT_CLOCK_STOP_COUNT'])

    if img_type == 'r':
        incidence_angle = np.degrees(metadata['incidence'])
    else:
        incidence_angle = np.degrees(metadata['mean_incidence'])

    ret['INCIDENCE_ANGLE'] = f'{incidence_angle:.6f}'

    if img_type == 'r':
        emission_angles = np.degrees(metadata['mean_emission'])
        phase_angles = np.degrees(metadata['mean_phase'])
        resolutions = metadata['mean_resolution']
    else:
        emission_angles = np.degrees(metadata['mean_emission'][long_mask])
        phase_angles = np.degrees(metadata['mean_phase'][long_mask])
        resolutions = metadata['mean_resolution'][long_mask]

    # XXX Implement difference between emission angle and observation ring elevation
    ret['MEAN_OBS_RING_ELEV'] = f'{np.mean(emission_angles):.6f}'
    ret['MIN_OBS_RING_ELEV'] = f'{np.min(emission_angles):.6f}'
    ret['MAX_OBS_RING_ELEV'] = f'{np.max(emission_angles):.6f}'

    ret['MEAN_PHASE_ANGLE'] = f'{np.mean(phase_angles):.6f}'
    ret['MIN_PHASE_ANGLE'] = f'{np.min(phase_angles):.6f}'
    ret['MAX_PHASE_ANGLE'] = f'{np.max(phase_angles):.6f}'

    ret['MEAN_REPROJ_GRID_RAD_RES'] = f'{np.mean(resolutions):.6f}'
    ret['MIN_REPROJ_GRID_RAD_RES'] = f'{np.min(resolutions):.6f}'
    ret['MAX_REPROJ_GRID_RAD_RES'] = f'{np.max(resolutions):.6f}'

    if img_type != 'r':
        image_name_list = metadata['image_name_list']
        ret['NUM_IMAGES'] = str(len(image_name_list))
        image_name0 = metadata['image_name_list'][0]
    else:
        _, image_name0 = os.path.split(metadata['image_path'])
    camera = image_name0[0]
    if camera not in ('N', 'W'):
        LOGGER.fatal(f'Unknown camera for image {image_name0}')
        sys.exit(-1)
    if img_type != 'r':
        for image_name in image_name_list:
            if image_name[0] != camera:
                LOGGER.error(f'{obsid}: Inconsistent cameras for images '
                            f'{image_name0} and {image_name}')
                break
    if image_name0[0] == 'N':
        ret['CAMERA_WIDTH'] = 'Narrow'
        ret['CAMERA_WN_UC'] = 'N'
        ret['CAMERA_WN_LC'] = 'n'
    else:
        ret['CAMERA_WIDTH'] = 'Wide'
        ret['CAMERA_WN_UC'] = 'W'
        ret['CAMERA_WN_LC'] = 'w'

    if img_type == 'r':
        ret['MIN_RING_COROTATING_LONG'] = f'{min_corot_long:.2f}'
        ret['MAX_RING_COROTATING_LONG'] = f'{max_corot_long:.2f}'
    else:
        # Mosaics are always written out to their full extent even if not all
        # longitudes are populated.
        ret['MIN_RING_COROTATING_LONG'] = '0.00'
        ret['MAX_RING_COROTATING_LONG'] = '360.00'
    ret['MIN_RING_INERTIAL_LONG'] = f'{min_inertial:.2f}'
    ret['MAX_RING_INERTIAL_LONG'] = f'{max_inertial:.2f}'

    return ret


def generate_image(obsid, output_dir, metadata, xml_metadata, img_type):
    """Create mosaic images and labels and mosaic metadata tables and labels.

    Inputs:
        obsid          The observation name.
        output_dir      The directory in which to put all output files.
        metadata        The metadata for a mosaic, background-subtracted
                        mosaic, or reprojected image.
        xml_metadata    The XML substitutions.
        img_type        The img_type of data being provided:
                            'm' = Mosaic
                            'b' = Background-subtracted mosaic
                            'r' = Reprojected image

    The global flags are used to determine which output files to create:

    img_type = 'm':

      data_mosaic/
        OBSID/
          OBSID_mosaic.img                            [GENERATE_MOSAIC_IMAGES]
          OBSID_mosaic.xml                            [GENERATE_MOSAIC_IMAGE_LABELS]
          OBSID_mosaic_metadata_src_imgs.tab          [GENERATE_MOSAIC_METADATA_TABLES]
          OBSID_mosaic_metadata_params.tab            [GENERATE_MOSAIC_METADATA_TABLES]
          OBSID_mosaic_metadata.xml                   [GENERATE_MOSAIC_METADATA_LABELS]

    img_type = 'b':

      data_mosaic_bkg_sub/
        OBSID/
          OBSID_mosaic_bkg_sub.img                    [GENERATE_MOSAIC_IMAGES]
          OBSID_mosaic_bkg_sub.xml                    [GENERATE_MOSAIC_IMAGE_LABELS]
          OBSID_mosaic_bkg_sub_metadata_src_imgs.tab  [GENERATE_MOSAIC_METADATA_TABLES]
          OBSID_mosaic_bkg_sub_metadata_params.tab    [GENERATE_MOSAIC_METADATA_TABLES]
          OBSID_mosaic_bkg_sub_metadata.xml           [GENERATE_MOSAIC_METADATA_LABELS]

    img_type = 'r':

      data_reproj_img/
        OBSID/
          IMG_reproj_img.img                          [GENERATE_REPROJ_IMAGES]
          IMG_reproj_img.xml                          [GENERATE_REPROJ_IMAGE_LABELS]
          IMG_reproj_img_metadata_params.tab          [GENERATE_REPROJ_METADATA_TABLES]
          IMG_reproj_img_metadata.xml                 [GENERATE_REPROJ_METADATA_LABELS]
    """
    os.makedirs(output_dir, exist_ok=True)

    if img_type == 'r':
        image_name = metadata['image_name']
    else:
        sfx = '_bkg_sub' if img_type == 'b' else ''

    long_mask = metadata['long_mask']
    longitudes = metadata['longitudes'][long_mask]
    if img_type == 'r':
        emission_angles = np.degrees(metadata['mean_emission'])
        phase_angles = np.degrees(metadata['mean_phase'])
        resolutions = metadata['mean_resolution']
    else:
        emission_angles = np.degrees(metadata['mean_emission'][long_mask])
        phase_angles = np.degrees(metadata['mean_phase'][long_mask])
        resolutions = metadata['mean_resolution'][long_mask]
    inertial_longitudes = metadata['inertial_longitudes']

    if img_type == 'r':
        pass
    else:
        ETs = metadata['time'][long_mask]
        image_indexes = metadata['image_number'][long_mask]
        image_name_list = metadata['image_name_list']

    target_id = ''
    if mosaic_has_prometheus(metadata):
        target_id += TARGET_PROMETHEUS
    if mosaic_has_pandora(metadata):
        target_id += TARGET_PANDORA
    xml_metadata['TARGET_IDENTIFICATION'] = target_id

    # XXX DO SOMETHING WITH STARS
        # <Target_Identification>
        #     <!-- If appropriate, include occulted star -->
        #     <name>Spica</name>
        #     <alternate_designation>Alpha Virginis</alternate_designation>
        #     <alternate_designation>Alpha Vir</alternate_designation>
        #     <type>Star</type>
        #     <Internal_Reference>
        #         <lid_reference>urn:nasa:pds:context:target:star.alf_vir</lid_reference>
        #         <reference_type>ancillary_to_target</reference_type>
        #     </Internal_Reference>
        # </Target_Identification>


            ###############################
            ###     METADATA_TABLES     ###
            ###############################

    if img_type == 'r':
        params_filename = f'{image_name.lower()}_reproj_img_metadata_params.tab'
        xml_metadata['METADATA_PARAMS_TABLE_FILENAME'] = params_filename
        metadata_params_table_path = os.path.join(output_dir, params_filename)
    else:
        params_filename = f'{obsid.lower()}_mosaic{sfx}_metadata_params.tab'
        xml_metadata['METADATA_PARAMS_TABLE_FILENAME'] = params_filename
        metadata_params_table_path = os.path.join(output_dir, params_filename)
        src_imgs_filename = f'{obsid.lower()}_mosaic{sfx}_metadata_src_imgs.tab'
        xml_metadata['IMAGE_TABLE_FILENAME'] = src_imgs_filename
        image_table_path = os.path.join(output_dir, src_imgs_filename)

    if ((img_type == 'r' and GENERATE_REPROJ_METADATA_TABLES) or
        (img_type != 'r' and GENERATE_MOSAIC_METADATA_TABLES)):
        # OBSID_mosaic_metadata_params.tab or
        # IMG_reproj_img_metadata_params.tab
        with open(metadata_params_table_path, 'w') as fp:
            if img_type == 'r':
                fp.write('Corotating Longitude, '
                         'Inertial Longitude, Resolution, Phase Angle, '
                         'Emission Angle\n')
            else:
                fp.write('Corotating Longitude, Image Index, Mid-time SPICE ET, '
                         'Inertial Longitude, Resolution, Phase Angle, '
                         'Emission Angle\n')
            for idx in range(len(longitudes)):
                longitude = longitudes[idx]
                inertial = inertial_longitudes[idx]
                resolution = resolutions[idx]
                phase = phase_angles[idx]
                emission = emission_angles[idx]
                if img_type == 'r':
                    row = (f'{longitude:6.2f}, '
                           f'{inertial:7.3f}, {resolution:10.5f}, '
                           f'{phase:10.6f}, {emission:10.6f}')
                else:
                    et = ETs[idx]
                    image_idx = image_indexes[idx]
                    row = (f'{longitude:6.2f}, {image_idx:4d}, {et:13.3f}, '
                           f'{inertial:7.3f}, {resolution:10.5f}, '
                           f'{phase:10.6f}, {emission:10.6f}')
                fp.write(row+'\n')

        if img_type != 'r':
            # mosaic_metadata_src_imgs.tab
            with open(image_table_path, 'w') as fp:
                fp.write('Source Image Index, LIDVID\n')
                for idx in range(len(image_name_list)):
                    lidvid = image_name_to_lidvid(image_name_list[idx])
                    row = f'{idx:4d}, {lidvid}'
                    fp.write(row+'\n')


            ###############################
            ###     METADATA_LABELS     ###
            ###############################

    if ((img_type == 'r' and GENERATE_REPROJ_METADATA_LABELS) or
        (img_type != 'r' and GENERATE_MOSAIC_METADATA_LABELS)):
        xml_metadata['METADATA_PARAMS_TABLE_PATH'] = metadata_params_table_path
        if img_type != 'r':
            xml_metadata['IMAGE_TABLE_PATH'] = image_table_path
        if img_type == 'r':
            metadata_label_output_path = os.path.join(output_dir,
                                   f'{image_name.lower()}_reproj_img_metadata.xml')
            populate_template(obsid, 'reproj-img-metadata.xml',
                              metadata_label_output_path, xml_metadata)
        else:
            metadata_label_output_path = os.path.join(output_dir,
                                   f'{obsid.lower()}_mosaic{sfx}_metadata.xml')
            populate_template(obsid, 'mosaic-metadata.xml',
                              metadata_label_output_path, xml_metadata)


            ###############################
            ###  MOSAIC/REPROJ_IMAGES   ###
            ###############################

    img = ma.filled(metadata['img'], SENTINEL).astype('float32')
    xml_metadata['IMG_NUM_SAMPLES'] = str(img.shape[1])
    xml_metadata['IMG_NUM_LINES'] = str(img.shape[0])
    if img_type == 'r':
        image_output_path = os.path.join(output_dir, xml_metadata['REPROJ_IMG_FILENAME'])
        label_output_path = os.path.join(output_dir,
                                         f'{image_name}_reproj_img.xml')
    else:
        image_output_path = os.path.join(output_dir, xml_metadata['MOSAIC_IMG_FILENAME'])
        label_output_path = os.path.join(output_dir,
                                         f'{obsid.lower()}_mosaic{sfx}.xml')

    if ((img_type == 'r' and GENERATE_REPROJ_IMAGES) or
        (img_type != 'r' and GENERATE_MOSAIC_IMAGES)):
        img.tofile(image_output_path)


            ###############################
            ###  MOSAIC/REPROJ_LABELS   ###
            ###############################

    if ((img_type == 'r' and GENERATE_REPROJ_IMAGE_LABELS) or
        (img_type != 'r' and GENERATE_MOSAIC_IMAGE_LABELS)):
        try:
            xml_metadata['IMG_PATH'] = image_output_path
        except FileNotFoundError:
            pass
        if img_type == 'r':
            populate_template(obsid, 'reproj-img.xml', label_output_path, xml_metadata)
        else:
            populate_template(obsid, 'mosaic.xml', label_output_path, xml_metadata)


def generate_browse(obsid, browse_dir, metadata, xml_metadata, img_type):
    """Create mosaic browse images. These are only from bkg-sub mosaics.

    Inputs:
        obsid          The observation name.
        browse_dir      The directory in which to put all browse files.
        metadata        The metadata for a background-subtracted mosaic.
        xml_metadata    The XML substitutions.
        img_type        The img_type of data being provided:
                            'm' = Mosaic
                            'b' = Background-subtracted mosaic
                            'r' = Reprojected image

    The global flags like GENERATE_MOSAIC_BROWSE_IMAGES are used to determine
    which output files to create:

    img_type == 'm':

      browse_mosaic/
        OBSID/
          OBSID_browse_mosaic_full.png
          OBSID_browse_mosaic_med.png
          OBSID_browse_mosaic_small.png
          OBSID_browse_mosaic_thumb.png
          OBSID_browse_mosaic.xml

    img_type == 'b':

      browse_mosaic_bkg_sub/
        OBSID/
          OBSID_browse_mosaic_bkg_sub_full.png
          OBSID_browse_mosaic_bkg_sub_med.png
          OBSID_browse_mosaic_bkg_sub_small.png
          OBSID_browse_mosaic_bkg_sub_thumb.png
          OBSID_browse_mosaic_bkg_sub.xml

    img_type == 'r':

      browse_mosaic/
        OBSID/
          IMG_browse_reproj_img_full.png
          IMG_browse_reproj_img_med.png
          IMG_browse_reproj_img_small.png
          IMG_browse_reproj_img_thumb.png
          IMG_browse_reproj_img.xml
    """
    os.makedirs(browse_dir, exist_ok=True)

    match = re.search(r'^(.*)_(\d+)$', obsid)
    partial_obsid = bool(match)
    root_obsid = obsid
    if partial_obsid:
        root_obsid = match[1]
        obsid_chunk = match[2]

    cap_bkg = 'Background-subtracted ' if img_type == 'b' else ''

    if img_type != 'r':
        sfx = '_bkg_sub' if img_type == 'b' else ''
        sizes = (('full',  1,   1, 401, 18000),
                 ('med',   1,  10, 400,  1800),
                 ('small', 1,  45, 400,   400),
                 ('thumb', 4, 180, 100,   100))
    else:
        image_name = metadata['image_name']
        sizes = (('full',  1,    1, 401, 18000),
                 ('thumb', 4, None, 100,   100))


            ###############################
            ###      BROWSE_IMAGES      ###
            ###############################

    if ((img_type == 'r' and GENERATE_REPROJ_BROWSE_IMAGES) or
        (img_type != 'r' and GENERATE_MOSAIC_BROWSE_IMAGES)):
        img = ma.filled(metadata['img'], 0)
        valid_cols = np.sum(img, axis=0) != 0
        subimg = img[:, valid_cols]
        blackpoint = max(np.min(subimg), 0)
        whitepoint_ignore_frac = 0.995
        img_sorted = sorted(list(subimg.flatten()))
        whitepoint = img_sorted[np.clip(int(len(img_sorted)*
                                            whitepoint_ignore_frac),
                                        0, len(img_sorted)-1)]
        gamma = 0.5
        if whitepoint < blackpoint:
            whitepoint = blackpoint
        if whitepoint == blackpoint:
            whitepoint += 0.00001

        for size, sub0, sub1, crop0, crop1 in sizes:
            if sub1 is None:
                # Figure out how much to downsample
                sub1 = img.shape[1] // crop1
                if img.shape[1] != sub1 * crop1:
                    sub1 += 1
                    pad = (sub1 * crop1) - img.shape[1]
                    img = pad_image(img, (0, pad))
                downsampled_img = downsample(img, sub0, sub1)
            else:
                downsampled_img = downsample(img, sub0, sub1)
            if crop0 is not None:
                downsampled_img = downsampled_img[:crop0, :]
            if crop1 is not None:
                downsampled_img = downsampled_img[:, :crop1]

            greyscale_img = np.floor((np.maximum(downsampled_img-blackpoint, 0)/
                                     (whitepoint-blackpoint))**gamma*256)
            greyscale_img = np.clip(greyscale_img, 0, 255)
            scaled_img = np.cast['int8'](greyscale_img[::-1,:])
            pil_img = Image.frombuffer('L', (scaled_img.shape[1],
                                             scaled_img.shape[0]),
                                       scaled_img, 'raw', 'L', 0, 1)
            if img_type == 'r':
                png_path = os.path.join(browse_dir,
                            f'{image_name.lower()}_browse_reproj_img_{size}.png')
            else:
                png_path = os.path.join(browse_dir,
                            f'{obsid.lower()}_browse_mosaic{sfx}_{size}.png')
            pil_img.save(png_path, 'PNG')


            ###############################
            ###      BROWSE_LABELS      ###
            ###############################

    start_date = xml_metadata['START_DATE_TIME']
    stop_date = xml_metadata['STOP_DATE_TIME']

    if img_type == 'r':
        xml_metadata['REPROJ_BROWSE_LID'] = image_name_to_reproj_browse_lid(image_name)
        xml_metadata['REPROJ_BROWSE_TITLE'] = f"""
Browse images for the reprojected Cassini ISS calibrated image {image_name}
taken at {start_date}
"""
        xml_metadata['REPROJ_BROWSE_DESCRIPTION'] = f"""
These browse images correspond to the reprojected image of Cassini ISS
calibrated image {image_name}. The reprojected image is in units of I/F. The
browse images map I/F to 8-bit greyscale and are contrast-stretched for easier
viewing, using a blackpoint at the minimum image value, a whitepoint at the
99.5% maximum image value, and a gamma of 0.5. Browse images are available in
two sizes: full (equal in size to the reprojected image) and thumb (100x100,
padded as necessary). The browse images omit longitudes that have no data
available; if the available longitudes are discontinuous, the browse image will
show the longitudes as being adjacent. Pixels with no data available are shown
as black.
"""
    else:
        # Find the image names at the starting and ending ETs
        long_mask = metadata['long_mask']
        image_indexes = metadata['image_number'][long_mask]
        image_name_list = metadata['image_name_list']
        ETs = metadata['time'][long_mask]
        idx_min = np.argmin(ETs)
        idx_max = np.argmax(ETs)
        min_image_name = image_name_list[image_indexes[idx_min]]
        max_image_name = image_name_list[image_indexes[idx_max]]

        xml_metadata['MOSAIC_BROWSE_LID'] = obsid_to_mosaic_browse_lid(obsid,
                                                                       img_type == 'b')
        xml_metadata['MOSAIC_BROWSE_TITLE'] = f"""
Browse images for the {cap_bkg.lower()}F Ring mosaic created from reprojected
Cassini ISS calibrated images from observation {root_obsid} spanning
{start_date} ({min_image_name}) to {stop_date} ({max_image_name})
"""
        xml_metadata['MOSAIC_BROWSE_DESCRIPTION'] = f"""
These browse images correspond to the {cap_bkg.lower()}F Ring mosaic created
from reprojected Cassini ISS calibrated images from observation {root_obsid}.
The mosaic is in units of I/F. The browse images map I/F to 8-bit greyscale and
are contrast-stretched for easier viewing, using a blackpoint at the minimum
mosaic value, a whitepoint at the 99.5% maximum mosaic value, and a gamma of
0.5. Browse images are available in four sizes: full (18000x401), med
(1800x400), small (400x400), and thumb (100x100). The full longitude range is
shown even when no images cover that area. Pixels with no data available are
shown as black.
"""

    if ((img_type != 'r' and GENERATE_REPROJ_BROWSE_LABELS) or
        (img_type == 'r' and GENERATE_MOSAIC_BROWSE_LABELS)):
        for size, sub0, sub1, crop0, crop1 in (('full',  1,   1, 401, 18000),
                                               ('med',   1,  10, 401,  1800),
                                               ('small', 1,  45, 400,   400),
                                               ('thumb', 4, 180, 100,   100)):
            if img_type == 'r':
                browse_filename = f'{image_name.lower()}_browse_reproj_img_{size}.png'
            else:
                browse_filename = f'{obsid.lower()}_browse_mosaic{sfx}_{size}.png'
            xml_metadata[f'BROWSE_{size.upper()}_FILENAME'] = browse_filename
            png_path = os.path.join(browse_dir, browse_filename)
            try:
                xml_metadata[f'BROWSE_{size.upper()}_PATH'] = png_path
            except FileNotFoundError:
                pass

        if img_type == 'r':
            output_path = os.path.join(browse_dir,
                                       f'{image_name.lower()}_browse_reproj_img.xml')
        else:
            output_path = os.path.join(browse_dir,
                                       f'{obsid.lower()}_browse_mosaic{sfx}.xml')
        if img_type == 'r':
            populate_template(obsid, 'reproj-browse-image.xml', output_path, xml_metadata)
        else:
            populate_template(obsid, 'mosaic-browse-image.xml', output_path, xml_metadata)


def generate_mosaic(obsid,
                    mosaic_dir, bsm_dir,
                    mosaic_browse_dir, bsm_browse_dir,
                    mosaic_metadata, bsm_metadata, bkgnd_metadata):
    """Create all files related to mosaics.

    Inputs:
        obsid              The observation name.
        mosaic_dir          The directory in which to put all mosaic files.
        bsm_dir             The directory in which to put all bsm files.
        mosaic_browse_dir   The directory in which to put mosaic browse files.
        bsm_browse_dir      The directory in which to put bsm browse files.
        mosaic_metadata     The metadata for the mosaic.
        bsm_metadata        The metadata for the background-subtracted mosaic.
        bkgnd_metadata      The metadata for the background subtraction model.
    """
    # Do plain mosaics first
    xml_metadata = xml_metadata_for_image(obsid, mosaic_metadata, 'm')
    if (GENERATE_MOSAIC_METADATA_TABLES or GENERATE_MOSAIC_METADATA_LABELS or
        GENERATE_MOSAIC_IMAGES or GENERATE_MOSAIC_IMAGE_LABELS):
        generate_image(obsid, mosaic_dir, mosaic_metadata, xml_metadata, 'm')
    if GENERATE_MOSAIC_BROWSE_IMAGES or GENERATE_MOSAIC_BROWSE_LABELS:
        generate_browse(obsid, mosaic_browse_dir, mosaic_metadata,
                        xml_metadata, 'm')

    # Now do BSM
    xml_metadata = xml_metadata_for_image(obsid, bsm_metadata, 'b')
    if (GENERATE_MOSAIC_METADATA_TABLES or GENERATE_MOSAIC_METADATA_LABELS or
        GENERATE_MOSAIC_IMAGES or GENERATE_MOSAIC_IMAGE_LABELS):
        generate_image(obsid, bsm_dir, bsm_metadata, xml_metadata, 'b')
    if GENERATE_MOSAIC_BROWSE_IMAGES or GENERATE_MOSAIC_BROWSE_LABELS:
        generate_browse(obsid, bsm_browse_dir, bsm_metadata, xml_metadata, 'b')


def generate_reproj(obsid, reproj_dir, reproj_browse_dir, reproj_metadata):
    """Create all files related to mosaics.

    Inputs:
        obsid              The observation name.
        reproj_dir          The directory in which to put all reproj files.
        reproj_browse_dir   The directory in which to put all reproj browse
                            files.
        reproj_metadata     The metadata for the reprojected images.
    """
    xml_metadata = xml_metadata_for_image(obsid, reproj_metadata, 'r')
    if (GENERATE_REPROJ_METADATA_TABLES or GENERATE_REPROJ_METADATA_LABELS or
        GENERATE_REPROJ_IMAGES or GENERATE_REPROJ_IMAGE_LABELS):
        generate_image(obsid, reproj_dir, reproj_metadata, xml_metadata, 'r')
    if GENERATE_REPROJ_BROWSE_IMAGES or GENERATE_REPROJ_BROWSE_LABELS:
        generate_browse(obsid, reproj_browse_dir, reproj_metadata,
                        xml_metadata, 'r')


##########################################################################################
#
# MAIN OBSID LOOP
#
##########################################################################################

def handle_one_obsid(obsid, reproj_collection_fp, browse_reproj_collection_fp):
    mosaic_dir = os.path.join(arguments.output_dir, 'data_mosaic',
                              obsid.lower())
    bsm_dir = os.path.join(arguments.output_dir, 'data_mosaic_bkg_sub',
                           obsid.lower())

    # Paths for the mosaic image and the mosaic metadata
    (mosaic_path, mosaic_metadata_path) = f_ring.mosaic_paths(arguments, obsid)
    if not os.path.exists(mosaic_path):
        LOGGER.error(f'File not found: {mosaic_path}')
        return
    if not os.path.exists(mosaic_metadata_path):
        LOGGER.error(f'File not found: {mosaic_metadata_path}')
        return

    # Paths for the background-subtracted-mosaic image and metadata
    (bsm_path, bsm_metadata_path) = f_ring.bkgnd_sub_mosaic_paths(arguments, obsid)
    if not os.path.exists(bsm_path):
        LOGGER.error(f'File not found: {bsm_path}')
        return
    if not os.path.exists(bsm_metadata_path):
        LOGGER.error(f'File not found: {bsm_metadata_path}')
        return

    mosaic_metadata = None
    bsm_metadata = None

    if (GENERATE_MOSAIC_IMAGES or GENERATE_MOSAIC_IMAGE_LABELS or
        GENERATE_MOSAIC_METADATA_TABLES or GENERATE_MOSAIC_METADATA_LABELS or
        GENERATE_MOSAIC_BROWSE_IMAGES or GENERATE_MOSAIC_BROWSE_LABELS):
        mosaic_browse_dir = os.path.join(arguments.output_dir, 'browse_mosaic',
                                         obsid.lower())
        bsm_browse_dir = os.path.join(arguments.output_dir,
                                      'browse_mosaic_bkg_sub', obsid.lower())

        # Paths for the background model and background model metadata
        (bkgnd_model_path, bkgnd_metadata_path) = f_ring.bkgnd_paths(arguments, obsid)
        if not os.path.exists(bkgnd_model_path):
            LOGGER.error(f'File not found: {bkgnd_model_path}')
            return
        if not os.path.exists(bkgnd_metadata_path):
            LOGGER.error(f'File not found: {bkgnd_metadata_path}')
            return

        mosaic_metadata = read_mosaic(mosaic_path, mosaic_metadata_path, bkg_sub=False)
        bsm_metadata = read_mosaic(bsm_path, bsm_metadata_path, bkg_sub=True)
        bkgnd_metadata = read_bkgnd_metadata(bkgnd_model_path, bkgnd_metadata_path)

        if not all(obsid == x for x in mosaic_metadata['obsid_list']):
            LOGGER.error(f'Not all mosaic OBSIDs are {obsid}')
            return
        if not all(obsid == x for x in bsm_metadata['obsid_list']):
            LOGGER.error(f'Not all background-sub mosaic OBSIDs are {obsid}')
            return

        remap_image_indexes(mosaic_metadata)
        remap_image_indexes(bsm_metadata)

        generate_mosaic(obsid,
                        mosaic_dir, bsm_dir,
                        mosaic_browse_dir, bsm_browse_dir,
                        mosaic_metadata, bsm_metadata, bkgnd_metadata)

    if (GENERATE_REPROJ_IMAGES or GENERATE_REPROJ_IMAGE_LABELS or
        GENERATE_REPROJ_METADATA_TABLES or GENERATE_REPROJ_METADATA_LABELS or
        GENERATE_REPROJ_BROWSE_IMAGES or GENERATE_REPROJ_BROWSE_LABELS):
        if mosaic_metadata is None:
            mosaic_metadata = read_mosaic(mosaic_path, mosaic_metadata_path,
                                          bkg_sub=False, read_img=False)
            remap_image_indexes(mosaic_metadata)
        reproj_dir = os.path.join(arguments.output_dir, 'data_reproj_img',
                                  obsid.lower())
        reproj_browse_dir = os.path.join(arguments.output_dir, 'browse_reproj_img',
                                         obsid.lower())
        for image_path in mosaic_metadata['image_path_list']:
            reproj_path = img_to_repro_path(image_path)
            reproj_metadata = read_reproj(reproj_path)
            reproj_metadata['image_path'] = image_path
            reproj_metadata['image_name'] = image_name = \
                image_path.split('/')[-1].replace('_CALIB.IMG', '')

            if GENERATE_REPROJ_COLLECTIONS:
                reproj_lidvid = image_name_to_reproj_lidvid(image_name)
                reproj_metadata_lidvid = image_name_to_reproj_metadata_lidvid(image_name)
                reproj_collection_fp.write(f'P,{reproj_lidvid}\n')
                reproj_collection_fp.write(f'P,{reproj_metadata_lidvid}\n')
            if GENERATE_REPROJ_BROWSE_COLLECTIONS:
                browse_reproj_lidvid = image_name_to_reproj_browse_lidvid(image_name)
                browse_reproj_collection_fp.write(f'P,{browse_reproj_lidvid}\n')

            generate_reproj(obsid, reproj_dir, reproj_browse_dir, reproj_metadata)


NOW = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
SENTINEL = -999

BASIC_XML_METADATA = {
    'KEYWORD': 'saturn rings, f ring',
    'PUBLICATION_YEAR': '2023',
    'MODIFICATION_DATE': NOW[:10], # UTC
    'NOW': NOW, # UTC
    'MIN_RING_RADIUS': f'{arguments.ring_radius+arguments.radius_inner_delta:.0f}',
    'MAX_RING_RADIUS': f'{arguments.ring_radius+arguments.radius_outer_delta:.0f}',
    'USERGUIDE_LID': 'urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:document:rob-detailed-users-guide', # XXX
    'USERGUIDE_COMMENT': """Detailed User's Guide for the F Ring Mosaics and
Reprojected Images in this bundle.""",
    'SENTINEL': str(SENTINEL)
}


os.makedirs(os.path.join(arguments.output_dir, 'data_mosaic'), exist_ok=True)
os.makedirs(os.path.join(arguments.output_dir, 'data_mosaic_bkg_sub'), exist_ok=True)
os.makedirs(os.path.join(arguments.output_dir, 'data_reproj_img'), exist_ok=True)
os.makedirs(os.path.join(arguments.output_dir, 'browse_mosaic'), exist_ok=True)
os.makedirs(os.path.join(arguments.output_dir, 'browse_mosaic_bkg_sub'), exist_ok=True)
os.makedirs(os.path.join(arguments.output_dir, 'browse_reproj_img'), exist_ok=True)

mosaic_collection_fp = None
bsm_collection_fp = None
if GENERATE_MOSAIC_COLLECTIONS:
    mosaic_collection_fp = open(os.path.join(arguments.output_dir,
                                             'data_mosaic',
                                             'collection_data_mosaic.csv'),
                                'w')
    bsm_collection_fp = open(os.path.join(arguments.output_dir,
                                          'data_mosaic_bkg_sub',
                                          'collection_data_mosaic_bkg_sub.csv'),
                             'w')
browse_mosaic_collection_fp = None
browse_bsm_collection_fp = None
if GENERATE_MOSAIC_BROWSE_COLLECTIONS:
    browse_mosaic_collection_fp = open(os.path.join(arguments.output_dir,
                                                    'browse_mosaic',
                                                    'collection_browse_mosaic.csv'),
                                       'w')
    browse_bsm_collection_fp = open(os.path.join(arguments.output_dir,
                                                 'browse_mosaic_bkg_sub',
                                                 'collection_browse_mosaic_bkg_sub.csv'),
                                    'w')

reproj_collection_fp = None
if GENERATE_REPROJ_COLLECTIONS:
    reproj_collection_fp = open(os.path.join(arguments.output_dir,
                                             'data_reproj_img',
                                             'collection_data_reproj_img.csv'),
                                'w')
browse_reproj_collection_fp = None
if GENERATE_REPROJ_BROWSE_COLLECTIONS:
    browse_reproj_collection_fp = open(os.path.join(arguments.output_dir,
                                                    'browse_reproj_img',
                                                    'collection_browse_reproj_img.csv'),
                                    'w')


for obsid in f_ring.enumerate_obsids(arguments):
    # LOGGER.open(f'OBSID {obsid}')
    try:
        handle_one_obsid(obsid, reproj_collection_fp, browse_reproj_collection_fp)
    except ObsIdFailedException:
        # A logged failure
        pass
    except KeyboardInterrupt:
        # Ctrl-C should be honored
        raise
    except SystemExit:
        # sys.exit() should be honored
        raise
    except:
        # Anything else
        LOGGER.error('Uncaught exception:\n' + traceback.format_exc())

    if GENERATE_MOSAIC_COLLECTIONS:
        mosaic_lidvid = obsid_to_mosaic_lidvid(obsid, False)
        mosaic_metadata_lidvid = obsid_to_mosaic_metadata_lidvid(obsid, False)
        mosaic_collection_fp.write(f'P,{mosaic_lidvid}\n')
        mosaic_collection_fp.write(f'P,{mosaic_metadata_lidvid}\n')
        bsm_lidvid = obsid_to_mosaic_lidvid(obsid, True)
        bsm_metadata_lidvid = obsid_to_mosaic_metadata_lidvid(obsid, True)
        bsm_collection_fp.write(f'P,{bsm_lidvid}\n')
        bsm_collection_fp.write(f'P,{bsm_metadata_lidvid}\n')
    if GENERATE_MOSAIC_BROWSE_COLLECTIONS:
        browse_mosaic_lidvid = obsid_to_mosaic_browse_lidvid(obsid, False)
        browse_mosaic_collection_fp.write(f'P,{browse_mosaic_lidvid}\n')
        browse_bsm_lidvid = obsid_to_mosaic_browse_lidvid(obsid, True)
        browse_bsm_collection_fp.write(f'P,{browse_bsm_lidvid}\n')

    # LOGGER.close()

if GENERATE_MOSAIC_COLLECTIONS:
    mosaic_collection_fp.close()
    bsm_collection_fp.close()
if GENERATE_MOSAIC_BROWSE_COLLECTIONS:
    browse_mosaic_collection_fp.close()
    browse_bsm_collection_fp.close()
if GENERATE_REPROJ_COLLECTIONS:
    reproj_collection_fp.close()
if GENERATE_REPROJ_BROWSE_COLLECTIONS:
    browse_reproj_collection_fp.close()


"""
- How do we include SPICE kernel information with the reprojected images?
- How do we include pointing navigation information with the reprojected images?

    I'm not sure of the answer to these - there's ongoing discussion with Boris about how
    to include SPICE information (and I see you've been cc'ed on that)

- What LID should I use for the Cassini original images, since I want to reference the
  calibrated images?

   I'd contacted @Mitch Gordon about this and we need to discuss this with @Matt Tiscareno
   (he/him). I could also initiate a wider discussion in Slack, but as we're planning to
   meet this week about something else that may be a good time.

- How does indenting of text work? You didn't like having the text left-justified, so I
  assume it has to be indented at the appropriate XML level. But then how do you want it
  wrapped? I've been wrapping to 80 characters left-justified, like this:
        <title>
F Ring mosaic created from reprojected Cassini ISS calibrated images from observation
ISS_079RI_FMONITOR002_PRIME spanning 2008-08-02T01:25:19Z (W1596333808_1) to
2008-08-02T01:54:07Z (W1596335548_1)
        </title>
so this is what you'll see in the latest sample. Do you want it indented, but then still
wrapped at 80 characters? Or something else?
        <title>
            F Ring mosaic created from reprojected Cassini ISS calibrated images from
            observation ISS_079RI_FMONITOR002_PRIME spanning 2008-08-02T01:25:19Z
            (W1596333808_1) to 2008-08-02T01:54:07Z (W1596335548_1)
        </title>

    As far as I know, there are no hard and fast rules about indenting, so don't worry too
    much about it. I prefer when the contents of the xml element are indented to the same
    amount as the tags. Wrapping at 80 characters looks good, i.e. as in your second
    example:

- Why did you remove the mean incidence angle and mean ring opening angle, but not the
  mean phase angle? I actually think all of the means are useful, because they aren't just
  (min+max)/2, they are the mean of the metadata available for all the longitudes.

    I removed the mean incidence angle because it doesn't exist in the Rings dictionary
    (only min/max incidence angles are defined) and I removed the mean ring opening angle
    because thought that wouldn't vary much. However, if you think they would be useful
    then we should talk about including them.

- You commented that you now have the Cassini ISS User's Guide DOI, but I don't know how
  to actually include it in the XML.

    <Internal_Reference>s don’t have a <doi> element like <External_Reference>s do, so to
    include a DOI please use the <comment>: <comment>The Cassini ISS Data User's Guide
    (PDS3); DOI: 10.17189/1504135</comment>

- When providing the <offset> field for metadata, should the offset include the newline?
  Your comment says "<!--mjtm: This should match <Header>.<object_length> -->". However,
  the header object length is currently just the length of the text, right? Not including
  the newline. So shouldn't the offset include the newline so that if you seek to that
  byte in the file you actually start at the proper location?

    I thought  <Header>.<offset> included the newline, so that <Table_Character>.<offset>
    is the same number of bytes. Requesting 2nd opinion from @Mitch Gordon!

- You comment "<!--mjtm: the data_type above is single precision float, so the
  missing_constant needs to have appropriate number of decimal places -->". But I don't
  know what that means. What is the "appropriate number of decimal places" for a
  single-precision float? Since it's a floating point number, by definition the number of
  decimal places changes based on the magnitude of the number. Would it be OK just to say
  -999.0?

    Good question, I thought that maybe you were outputting values like 123.456, 789.012,
    ... etc in which case you would use -999.000. I think that -999.0 should be OK

- Should we include all of the Cassini ISS metadata fields for the reprojected images?
  (edited)

    I think it makes sense to include whatever metadata fields you think would be useful



Full Cassini image info for reproj image metadata?

SPICE kernel and navigation info for reproj image?

Mosaic metadata_src_imgs is using data_calibrated.

Fix up intertial and co-rotating longitude limits

File creation date should really be the creation date of the file, not NOW

corotating or co-rotating?

How do we wrap and indent?
            <rings:description> <!-- mjtm: indentation and wrapping-->
Metadata for F Ring mosaics of reprojected Cassini ISS calibrated images from observation
ISS_039RF_FMOVIE001_VIMS, 2007-02-27T07:18:39Z to 2007-02-27T22:58:23Z
            </rings:description>

But mean phase angle is?
            <rings:minimum_observed_ring_elevation
            unit="deg">143.665878</rings:minimum_observed_ring_elevation> <!--mjtm: I
            opted to remove mean_observed_ring_elevation as not useful, also not present
            in Rings dictionary-->
And why isn't mean incidence angle?

Add Cassini ISS User's Manual DOI <!--mjtm DOI: 10.17189/1504135 -->

Should this offset include the NEWLINE? <offset unit="byte">26</offset> <!--mjtm: This
should match <Header>.<object_length> -->

Is IEEE754LSBSingle the correct data type?

<!--mjtm: the data_type above is single precision float, so the missing_constant needs to
have appropriate number of decimal places --> What is the number of decimal places for a
float?

"""