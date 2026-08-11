# TODO For obs that follow one co-rot, include all reproj images!

##########################################################################################
# Create all files for the PDS4 achive including binary, tabular, and label.
##########################################################################################

import argparse
import datetime
import logging
import math
import os
import pickle
import pyparsing
import re
import shutil
import sys
import traceback

import msgpack
import msgpack_numpy
import numpy as np
import numpy.ma as ma
from PIL import Image, ImageDraw, ImageFont

import cspyce
import julian
import oops
import oops.hosts.cassini.iss as coiss
import pdslogger
import pdstemplate

pdslogger.TIME_FMT = '%Y-%m-%d %H:%M:%S'

from pdsparser import Pds3Label

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'external'))

import f_ring_util.f_ring as f_ring

BUNDLE_NAME = 'cassini_iss_fring_mosaics_rsfrench2025'
DATA_MOSAIC_COLLECTION_LID = f'urn:nasa:pds:{BUNDLE_NAME}:data_mosaic'
DATA_MOSAIC_BKG_SUB_COLLECTION_LID =f'urn:nasa:pds:{BUNDLE_NAME}:data_mosaic_bkg_sub'
DATA_REPROJ_COLLECTION_LID = f'urn:nasa:pds:{BUNDLE_NAME}:data_reproj_img'
BROWSE_MOSAIC_COLLECTION_LID = f'urn:nasa:pds:{BUNDLE_NAME}:browse_mosaic'
BROWSE_MOSAIC_BKG_SUB_COLLECTION_LID = f'urn:nasa:pds:{BUNDLE_NAME}:browse_mosaic_bkg_sub'
BROWSE_REPROJ_COLLECTION_LID = f'urn:nasa:pds:{BUNDLE_NAME}:browse_reproj_img'
CONTEXT_COLLECTION_LID = f'urn:nasa:pds:{BUNDLE_NAME}:context'
DOCUMENT_COLLECTION_LID = f'urn:nasa:pds:{BUNDLE_NAME}:document'
MISCELLANEOUS_COLLECTION_LID = f'urn:nasa:pds:{BUNDLE_NAME}:miscellaneous'
USERGUIDE_LID = f'urn:nasa:pds:{BUNDLE_NAME}:document:f-ring-mosaics-user-guide'
GLOBAL_MOSAIC_INDEX_LID = f'urn:nasa:pds:{BUNDLE_NAME}:miscellaneous:global_mosaic_index'
GLOBAL_MOSAIC_BKG_SUB_INDEX_LID = f'urn:nasa:pds:{BUNDLE_NAME}:miscellaneous:global_mosaic_bkg_sub_index'
GLOBAL_REPROJ_INDEX_LID = f'urn:nasa:pds:{BUNDLE_NAME}:miscellaneous:global_reproj_img_index'
SPICE_KERNELS_COLLECTION_LID = f'urn:nasa:pds:{BUNDLE_NAME}:spice_kernels'
KERNELS_LID = f'urn:nasa:pds:{BUNDLE_NAME}:spice_kernels:kernels'
XML_SCHEMA_COLLECTION_LID = f'urn:nasa:pds:{BUNDLE_NAME}:xml_schema'


OBSERVATION_LIST_PATH = 'observation_list.csv'

# XML directory structure:
#   bundle.lblx                                     [RMS]
#   readme.txt                                      [RF writes]
#   browse_mosaic/
#     collection_browse_mosaic.lblx                 [RMS]
#     collection_browse_mosaic.csv                 +[generated: [P|S], LIDVID]
#     OBSID/
#       OBSID_browse_mosaic.lblx                   +[template mosaic-browse-image.lblx]
#       OBSID_browse_mosaic_full.png               +[generated]
#       OBSID_browse_mosaic_med.png                +[generated]
#       OBSID_browse_mosaic_small.png              +[generated]
#       OBSID_browse_mosaic_thumb.png              +[generated]
#   browse_mosaic_bkg_sub/
#     collection_browse_mosaic_bkg_sub.lblx         [RMS]
#     collection_browse_mosaic_bkg_sub.csv         +[generated: [P|S], LIDVID]
#     OBSID/
#       OBSID_browse_mosaic_bkg_sub.lblx           +[template mosaic-browse-image.lblx]
#       OBSID_browse_mosaic_bkg_sub_full.png       +[generated]
#       OBSID_browse_mosaic_bkg_sub_med.png        +[generated]
#       OBSID_browse_mosaic_bkg_sub_small.png      +[generated]
#       OBSID_browse_mosaic_bkg_sub_thumb.png      +[generated]
#   browse_reproj_img/
#     collection_browse_reproj_img.lblx             [RMS]
#     collection_browse_reproj_img.csv             +[generated: [P|S], LIDVID]
#     OBSID/
#       IMG_browse_reproj_img.lblx                 +[template reproj-browse-image.lblx]
#       IMG_browse_reproj_img_full.png             +[generated]
#       IMG_browse_reproj_img_thumb.png            +[generated]
#   context/
#     collection_context.csv                        [RMS - boilerplate]
#     collection_context.lblx                       [RMS - boilerplate]
#   data_mosaic/
#     collection_data_mosaic.lblx                   [RMS]
#     collection_data_mosaic.csv                   +[generated: [P|S], LIDVID]
#     OBSID/
#       OBSID_mosaic.lblx                          +[template mosaic.lblx]
#       OBSID_mosaic.img                           +[generated]
#       OBSID_mosaic_metadata_src_imgs.tab         +[generated]
#       OBSID_mosaic_metadata_params.tab           +[generated]
#   data_mosaic_bkg_sub/
#     collection_data_mosaic_bkg_sub.lblx           [RMS]
#     collection_data_mosaic_bkg_sub.csv           +[generated: [P|S], LIDVID]
#     OBSID/
#       OBSID_mosaic_bkg_sub.lblx                  +[template mosaic.lblx]
#       OBSID_mosaic_bkg_sub.img                   +[generated]
#       OBSID_mosaic_bkg_sub_metadata_src_imgs.tab +[generated]
#       OBSID_mosaic_bkg_sub_metadata_params.tab   +[generated]
#   data_reproj_img/
#     collection_data_reproj_img.lblx               [RMS]
#     collection_data_reproj_img.csv               +[generated: [P|S], LIDVID]
#     OBSID/
#       IMG_reproj_img.lblx                        +[template reproj-img.lblx]
#       IMG_reproj_img.img                         +[generated]
#       IMG_reproj_img_suppl.txt                   +[generated]
#       IMG_reproj_img_metadata_params.tab         +[generated]
#   document/
#     collection_document.lblx                      [RMS - boilerplate]
#     collection_document.csv                       [RMS - boilerplate]
#     user_guide/
#       f-ring-mosaics-user-guide.lblx              [RMS]
#       f-ring-mosaics-user-guide.pdf              +[RF writes]
#   miscellaneous/
#     collection_miscellaneous.lblx                [RMS]
#     collection_miscellaneous.csv                +[generated: [P|S], LIDVID]
#     global_mosaic_index.lblx                     [RF writes]
#     global_mosaic_index.tab                     +[generated]
#     global_mosaic_bkg_sub_index.lblx             [RF writes]
#     global_mosaic_bkg_sub_index.tab             +[generated]
#     global_reproj_img_index.lblx                 [RF writes]
#     global_reproj_img_index.tab                 +[generated]
#   spice_kernels/
#     collection_spice_kernels.lblx                 [RMS - boilerplate]
#     collection_spice_kernels.csv                  [RMS - boilerplate]
#     kernels.ker                                  +[RF writes]
#   xml_schema/
#     collection_xml_schema.lblx                    [RMS - boilerplate]
#     collection_xml_schema.csv                     [RMS - boilerplate]
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
parser.add_argument('--generate-reproj-suppl-files',
                    action='store_true', default=False,
                    help='Generate reproj supplemental files')
parser.add_argument('--generate-reproj-collections',
                    action='store_true', default=False,
                    help='Generate reproj collections files')
parser.add_argument('--generate-reproj-global-index',
                    action='store_true', default=False,
                    help='Generate reproj global index file')
parser.add_argument('--generate-reproj',
                    action='store_true', default=False,
                    help='Generate reproj image files and labels')

parser.add_argument('--generate-reproj-metadata-tables',
                    action='store_true', default=False,
                    help='Generate reproj tables')
parser.add_argument('--generate-reproj-metadata',
                    action='store_true', default=False,
                    help='Generate reproj metadata tables only; labels must be '
                         'regenerated separately to refresh their embedded '
                         'checksums')

parser.add_argument('--generate-reproj-browse-labels',
                    action='store_true', default=False,
                    help='Generate reproj browse labels')
parser.add_argument('--generate-reproj-browse-images',
                    action='store_true', default=False,
                    help='Generate reproj browse image files')
parser.add_argument('--generate-reproj-browse',
                    action='store_true', default=False,
                    help='Generate reproj browse images and labels')
parser.add_argument('--generate-reproj-browse-collections',
                    action='store_true', default=False,
                    help='Generate reproj browse image collections files')

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
parser.add_argument('--generate-mosaic-global-index',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub mosaic global index file')
parser.add_argument('--generate-mosaics',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub mosaic image files and labels')

parser.add_argument('--generate-mosaic-metadata-tables',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub metadata tables')
parser.add_argument('--generate-mosaic-metadata',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub metadata tables only; '
                         'labels must be regenerated separately to refresh '
                         'their embedded checksums')

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

parser.add_argument('--generate-all-images',
                    action='store_true', default=False,
                    help='Generate all images (mosaics, reproj, browse)')
parser.add_argument('--generate-all-labels',
                    action='store_true', default=False,
                    help='Generate all labels')

parser.add_argument('--generate-support-files',
                    action='store_true', default=False,
                    help='Generate the support files like bundle.lblx and context')

parser.add_argument('--generate-all',
                    action='store_true', default=False,
                    help='Generate all files and labels')

f_ring.add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)

f_ring.init(arguments)


# These hardcoded paths are for the machine ringlet
CALIBRATED_DIR = '/data/pdsdata/holdings/calibrated'
REPROJ_DIR = '/data/cb-results/fring/ring_mosaic/ring_repro'
OFFSETS_DIR = '/data/cb-results/fring/offsets'

GENERATE_REPROJ_IMAGE_LABELS = (arguments.generate_reproj_labels or
                                arguments.generate_reproj or
                                arguments.generate_all_reproj or
                                arguments.generate_all_labels or
                                arguments.generate_all)
GENERATE_REPROJ_IMAGES = (arguments.generate_reproj_images or
                          arguments.generate_reproj or
                          arguments.generate_all_reproj or
                          arguments.generate_all_images or
                          arguments.generate_all)
GENERATE_REPROJ_SUPPL_FILES = (arguments.generate_reproj_suppl_files or
                               arguments.generate_reproj or
                               arguments.generate_all_reproj or
                               arguments.generate_all)

GENERATE_REPROJ_METADATA_TABLES = (arguments.generate_reproj_metadata_tables or
                                   arguments.generate_reproj_metadata or
                                   arguments.generate_all_reproj or
                                   arguments.generate_all)

GENERATE_REPROJ_COLLECTIONS = (arguments.generate_reproj_collections or
                               arguments.generate_all_reproj or
                               arguments.generate_all)
GENERATE_REPROJ_GLOBAL_INDEX = (arguments.generate_reproj_global_index or
                                arguments.generate_all_reproj or
                                arguments.generate_all)

GENERATE_BROWSE_REPROJ_LABELS = (arguments.generate_reproj_browse_labels or
                                 arguments.generate_reproj_browse or
                                 arguments.generate_all_reproj or
                                 arguments.generate_all_labels or
                                 arguments.generate_all)
GENERATE_BROWSE_REPROJ_IMAGES = (arguments.generate_reproj_browse_images or
                                 arguments.generate_reproj_browse or
                                 arguments.generate_all_reproj or
                                 arguments.generate_all_images or
                                 arguments.generate_all)
GENERATE_BROWSE_REPROJ_COLLECTIONS = (arguments.generate_reproj_browse_collections or
                                      arguments.generate_all_reproj or
                                      arguments.generate_all)

GENERATE_MOSAIC_IMAGE_LABELS = (arguments.generate_mosaic_labels or
                                arguments.generate_mosaics or
                                arguments.generate_all_mosaics or
                                arguments.generate_all_labels or
                                arguments.generate_all)
GENERATE_MOSAIC_IMAGES = (arguments.generate_mosaic_images or
                          arguments.generate_mosaics or
                          arguments.generate_all_mosaics or
                          arguments.generate_all_images or
                          arguments.generate_all)
GENERATE_MOSAIC_COLLECTIONS = (arguments.generate_mosaic_collections or
                               arguments.generate_all_mosaics or
                               arguments.generate_all)
GENERATE_MOSAIC_GLOBAL_INDEX = (arguments.generate_mosaic_global_index or
                                arguments.generate_all_mosaics or
                                arguments.generate_all)

GENERATE_MOSAIC_METADATA_TABLES = (arguments.generate_mosaic_metadata_tables or
                                   arguments.generate_mosaic_metadata or
                                   arguments.generate_all_mosaics or
                                   arguments.generate_all)

GENERATE_BROWSE_MOSAIC_LABELS = (arguments.generate_mosaic_browse_labels or
                                 arguments.generate_mosaic_browse or
                                 arguments.generate_all_mosaics or
                                 arguments.generate_all_labels or
                                 arguments.generate_all)
GENERATE_BROWSE_MOSAIC_IMAGES = (arguments.generate_mosaic_browse_images or
                                 arguments.generate_mosaic_browse or
                                 arguments.generate_all_mosaics or
                                 arguments.generate_all_images or
                                 arguments.generate_all)
GENERATE_BROWSE_MOSAIC_COLLECTIONS = (arguments.generate_mosaic_browse_collections or
                                      arguments.generate_all_mosaics or
                                      arguments.generate_all)

GENERATE_SUPPORT_FILES = (arguments.generate_support_files or
                          arguments.generate_all)


##########################################################################################
#
# LOGGER INITIALIZATION
#
##########################################################################################

LOGGER = pdslogger.PdsLogger('fring.pds4')

LOG_DIR = arguments.log_dir
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_INFO = os.path.join(LOG_DIR, 'generate_pds4.log')
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

# Create a separate logger for pdstemplate that only handles warnings and errors
# but writes to the same log files as the main logger
pdstemplate_logger = pdslogger.PdsLogger('fring.pdstemplate')

# Create handlers that write to the same files but with WARNING and ERROR levels
pdstemplate_warning_handler = pdslogger.file_handler(LOG_FILE_INFO, level=logging.WARNING,
                                                    rotation='ymdhms')
pdstemplate_error_handler = pdslogger.file_handler(LOG_FILE_INFO, level=logging.ERROR,
                                                  rotation='ymdhms')

# Create a custom stdout handler that only shows warnings and errors
pdstemplate_stdout_handler = logging.StreamHandler(sys.stdout)
pdstemplate_stdout_handler.setLevel(logging.WARNING)
pdstemplate_stdout_formatter = logging.Formatter('%(levelname)s: %(message)s')
pdstemplate_stdout_handler.setFormatter(pdstemplate_stdout_formatter)

pdstemplate_logger.add_handler(pdstemplate_warning_handler)
pdstemplate_logger.add_handler(pdstemplate_error_handler)
pdstemplate_logger.add_handler(pdstemplate_stdout_handler)

pdstemplate.PdsTemplate.set_logger(pdstemplate_logger)


##########################################################################################
#
# LONG STATIC XML STRINGS
#
##########################################################################################

TARGET_PROMETHEUS = """
        <Target_Identification>
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
                <reference_type>data_to_target</reference_type>
            </Internal_Reference>
        </Target_Identification>"""

TARGET_PANDORA = """
        <Target_Identification>
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
                <reference_type>data_to_target</reference_type>
            </Internal_Reference>
        </Target_Identification>"""


##########################################################################################
#
# UTILITIY FUNCTIONS
#
##########################################################################################

class ObsIdFailedException(Exception):
    """Fatal error with current obsid. Can continue to next."""
    pass


def et_to_datetime(et, dec=None):
    """Convert a SPICE ET to a datetime like 2020-01-01T00:00:00Z."""
    return julian.ymdhms_format_from_tai(julian.tai_from_tdb(et), digits=dec) + 'Z'


def utc2et(s):
    """Convert a date/time in UTC format to SPICE Ephemeris Time."""
    return julian.tdb_from_tai(julian.tai_from_iso(s))


# F ring orbit from Albers 2012
FRING_ROTATING_ET = utc2et('2007-01-01')
FRING_MEAN_MOTION = 581.964  # deg/day
FRING_A = 140221.3
FRING_E = 0.00235
FRING_W0 = 24.2
FRING_DW = 2.70025
FRING_OMEGA0 = 15.0
FRING_DOMEGA = -2.68778

def _compute_fring_longitude_shift(et):
    return - (FRING_MEAN_MOTION * (et - FRING_ROTATING_ET) / 86400.) % 360


def fring_inertial_to_corotating(longitude, et):
    """Convert inertial longitude to corotating."""
    return (longitude + _compute_fring_longitude_shift(et)) % 360.


def fring_corotating_to_inertial(co_long, et):
    """Convert corotating longitude to inertial."""
    return (co_long - _compute_fring_longitude_shift(et)) % 360.


def fring_longitude_of_pericenter(et):
    """Return the longitude of pericenter at the given time."""
    return (FRING_W0 + FRING_DW*et/86400.) % 360.


def fring_true_anomaly(longitude, et):
    """Return the true anomaly at the given time and inertial longitude."""
    curly_w = fring_longitude_of_pericenter(et)
    return (longitude - curly_w) % 360.


def fring_longitude_of_ascending_node(et):
    """Return the longitude of ascending node at the given time."""
    return (FRING_OMEGA0 + FRING_DOMEGA*et/86400.) % 360.


def fring_radius_at_longitude(longitude, et):
    """Return the radius (km) of the F ring core at inertial longitude."""
    true_anomaly = fring_true_anomaly(longitude, et)

    radius = (FRING_A * (1-FRING_E**2) /
              (1 + FRING_E * np.cos(np.radians(true_anomaly))))

    return radius


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

def saturn_to_prometheus_corot(et):
    et_arr = np.asarray(et, dtype=np.float64)

    def _one(t):
        (prometheus_j2000, lt) = cspyce.spkez(PROMETHEUS_ID, t, 'J2000', 'NONE', SATURN_ID)
        prometheus_sat = np.dot(J2000_TO_SATURN, prometheus_j2000[0:3])
        dist = np.sqrt(prometheus_sat[0]**2.+prometheus_sat[1]**2.+prometheus_sat[2]**2.)
        longitude = np.degrees(math.atan2(prometheus_sat[1], prometheus_sat[0])) % 360
        longitude = fring_inertial_to_corotating(longitude, t)
        return dist, longitude

    if et_arr.ndim == 0:
        return _one(float(et_arr))
    flat = et_arr.ravel()
    n = flat.size
    dist_out = np.empty(n, dtype=np.float64)
    long_out = np.empty(n, dtype=np.float64)
    for i in range(n):
        dist_out[i], long_out[i] = _one(float(flat[i]))
    return dist_out.reshape(et_arr.shape), long_out.reshape(et_arr.shape)


def saturn_to_pandora_corot(et):
    et_arr = np.asarray(et, dtype=np.float64)

    def _one(t):
        (pandora_j2000, lt) = cspyce.spkez(PANDORA_ID, t, 'J2000', 'NONE', SATURN_ID)
        pandora_sat = np.dot(J2000_TO_SATURN, pandora_j2000[0:3])
        dist = np.sqrt(pandora_sat[0]**2.+pandora_sat[1]**2.+pandora_sat[2]**2.)
        longitude = np.degrees(math.atan2(pandora_sat[1], pandora_sat[0])) % 360
        longitude = fring_inertial_to_corotating(longitude, t)
        return dist, longitude

    if et_arr.ndim == 0:
        return _one(float(et_arr))
    flat = et_arr.ravel()
    n = flat.size
    dist_out = np.empty(n, dtype=np.float64)
    long_out = np.empty(n, dtype=np.float64)
    for i in range(n):
        dist_out[i], long_out[i] = _one(float(flat[i]))
    return dist_out.reshape(et_arr.shape), long_out.reshape(et_arr.shape)


def wrapped_minmax(lon):
    lon = np.asarray(lon) % 360
    lon = np.sort(lon)

    # gaps between consecutive sorted angles, including wraparound
    gaps = np.diff(np.r_[lon, lon[0] + 360])

    # largest gap
    k = np.argmax(gaps)

    # interval is the complement of that gap
    min_lon = lon[(k + 1) % len(lon)]
    max_lon = lon[k]

    return min_lon, max_lon


def wrapped_mean(lon_deg):
    lon = np.asarray(lon_deg, dtype=float) % 360
    lon = np.sort(lon)

    # Find largest gap
    gaps = np.diff(np.r_[lon, lon[0] + 360])
    k = np.argmax(gaps)

    # Start interval just after the largest gap
    start = lon[(k + 1) % len(lon)]

    # Unwrap so all points lie in one contiguous interval
    unwrapped = (lon - start) % 360 + start

    # Ordinary mean in that unwrapped interval
    mean = np.mean(unwrapped) % 360
    return mean


def img_to_repro_path(image_path):
    """Convert a calibrated image path to a reprojected image path.

    Parameters:
        image_path (str): path to a calibrated image like
            /data/pdsdata/holdings/calibrated/COISS_2xxx/COISS_2001/data/
            1454725799_1455008789/N1454725799_1_CALIB.IMG

    Returns:
        str: path to a reprojected image like
            /data/cb-results/fring/ring_mosaic/ring_repro/COISS_2001/
            1454725799_1455008789/N1454725799_1_140220_-01000_001000_05.000_0.020_10_1-REPRO.DAT
    """
    components = image_path.split('/')
    vol = components[-4]
    sclk_dir = components[-2]
    image_name = components[-1]
    repro_res_data = ('_%06d_%06d_%06d_%06.3f_%05.3f_%d_%d-REPRO.DAT' % (
                      arguments.ring_radius, arguments.radius_inner_delta, arguments.radius_outer_delta,
                      arguments.radius_resolution, arguments.longitude_resolution,
                      arguments.radial_zoom_amount, arguments.longitude_zoom_amount))
    image_name = image_name.replace('_CALIB.IMG', repro_res_data)
    return os.path.join(REPROJ_DIR, vol, sclk_dir, image_name)


def img_to_offset_path(image_path):
    """Convert a calibrated image path to an offset path.

    Parameters:
        image_path (str): path to a calibrated image like
            /data/pdsdata/holdings/calibrated/COISS_2xxx/COISS_2001/data/
            1454725799_1455008789/N1454725799_1_CALIB.IMG

    Returns:
        str: path to a offset like
            /data/cb-results/fring/offsets/COISS_2001/1454725799_1455008789/N1454725799_1-OFFSET.dat
    """
    components = image_path.split('/')
    vol = components[-4]
    sclk_dir = components[-2]
    image_name = components[-1]
    image_name = image_name.replace('_CALIB.IMG',
                                    '-OFFSET.dat')
    return os.path.join(OFFSETS_DIR, vol, sclk_dir, image_name)


def read_offset_metadata_path(offset_path):
    """Read and decompress metadata given an offset file path.

    overlay             True to include the overlay in the metadata dict.
    """
    try:
        with open(offset_path, "rb") as offset_fp:
            metadata = msgpack.unpackb(offset_fp.read(),
                                       max_str_len=100000000,
                                       object_hook=msgpack_numpy.decode)
    except UnicodeDecodeError: # Python2 msgpack file
        with open(offset_path, "rb") as offset_fp:
            metadata = msgpack.unpackb(offset_fp.read(),
                                       max_str_len=100000000,
                                       object_hook=msgpack_numpy.decode,
                                       raw=True)
            metadata = fixup_byte_to_str(metadata)
    except:
        LOGGER.error("Failed to read %s:\n%s", offset_path,
                     traceback.format_exc())
        raise ObsIdFailedException

    return metadata


def copy_file(template_dir_name, output_path):
    """Copy a file from the templates directory to an output path."""
    shutil.copy(os.path.join('templates', template_dir_name), output_path)


def populate_template(template_name, output_path, xml_metadata):
    """Copy a template to an output file after making substitutions.

    Parameters:
        template_name (str): name of the template file to find in the templates directory
        output_path (str): path to the output file
        xml_metadata (dict): XML metadata
    """
    template = pdstemplate.PdsTemplate(os.path.join('templates', template_name))
    LOGGER.info(f'Writing {output_path}')
    template.write(xml_metadata, output_path)


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


def add_orbital_metadata(metadata):
    """Add ring orbital information to mosaic or reprojected image metadata.

    Parameters:
        metadata (dict): The metadata to mutate.
    """
    long_antimask = metadata['long_antimask']
    longitudes = metadata['inertial_longitudes'][long_antimask]
    ETs = metadata['time']  # Image midtime, scalar for reprojected images, array for mosaics
    if isinstance(ETs, np.ndarray):
        ETs = ETs[long_antimask]
    metadata['core_radius'] = fring_radius_at_longitude(longitudes, ETs)
    metadata['long_asc'] = fring_longitude_of_ascending_node(ETs)
    metadata['long_peri'] = fring_longitude_of_pericenter(ETs)
    metadata['true_anomaly'] = fring_true_anomaly(longitudes, ETs)  # Always an array
    metadata['prometheus_dist'], metadata['prometheus_corot_long'] = saturn_to_prometheus_corot(ETs)
    metadata['pandora_dist'], metadata['pandora_corot_long'] = saturn_to_pandora_corot(ETs)


def read_mosaic(data_path, metadata_path, *, bkg_sub=False, read_img=True):
    """Read a main or background-subtracted mosaic and associated metadata.

    Parameters:
        data_path (str): path to the mosaic data file
        metadata_path (str): path to the mosaic metadata file
        bkg_sub (bool): whether to read a background-subtracted mosaic
        read_img (bool): whether to read the image data

    Returns:
        dict: metadata
    """
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

    if 'mean_resolution' in metadata: # Old format
        LOGGER.error(f'{obsid}: Old format metadata found for {metadata_path}')
        raise ObsIdFailedException
        # metadata['mean_radial_resolution'] = res = metadata['mean_resolution']
        # del metadata['mean_resolution']
        # metadata['mean_angular_resolution'] = np.zeros(res.shape)
    if 'long_mask' in metadata: # Old format
        LOGGER.error(f'{obsid}: Old format metadata found for {metadata_path}')
        raise ObsIdFailedException
        # metadata['long_antimask'] = metadata['long_mask']
        # del metadata['long_mask']

    if read_img:
        if bkg_sub:
            with np.load(data_path) as npz:
                metadata['img'] = ma.MaskedArray(**npz)
                # The background image mask shows the "bad pixels"
                # The missing data in the original mosaic has already been
                # converted to the sentinel value.
                metadata['img'].mask = False
        else:
            metadata['img'] = ma.MaskedArray(np.load(data_path))

    long_antimask = metadata['long_antimask']
    longitudes = (np.arange(len(long_antimask)) * arguments.longitude_resolution)
    inertial_longitudes = fring_corotating_to_inertial(longitudes, metadata['time'])
    inertial_longitudes[~long_antimask] = 0
    metadata['inertial_longitudes'] = inertial_longitudes
    metadata['longitudes'] = longitudes

    add_orbital_metadata(metadata)
    return metadata


def read_bkgnd_metadata(model_path, metadata_path):
    """Read background model metadata.

    Parameters:
        model_path (str): path to the background model file
        metadata_path (str): path to the background model metadata file

    Returns:
        dict: metadata
    """
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

    return metadata


def read_reproj(metadata_path):
    """Read reprojected image metadata.

    Parameters:
        metadata_path (str): path to the reprojected image metadata file

    Returns:
        dict: metadata
    """
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

    if 'mean_resolution' in metadata: # Old format
        LOGGER.error(f'{obsid}: Old format metadata found for {metadata_path}')
        raise ObsIdFailedException
        # metadata['mean_radial_resolution'] = res = metadata['mean_resolution']
        # del metadata['mean_resolution']
        # metadata['mean_angular_resolution'] = np.zeros(res.shape)
    if 'long_mask' in metadata: # Old format
        LOGGER.error(f'{obsid}: Old format metadata found for {metadata_path}')
        raise ObsIdFailedException
        # metadata['long_antimask'] = metadata['long_mask']
        # del metadata['long_mask']

    long_antimask = metadata['long_antimask']
    longitudes = (np.arange(len(long_antimask)) * arguments.longitude_resolution)
    inertial_longitudes = fring_corotating_to_inertial(longitudes, metadata['time'])
    inertial_longitudes[~long_antimask] = 0
    metadata['inertial_longitudes'] = inertial_longitudes
    metadata['longitudes'] = longitudes

    # Reprojected images are longitude-compressed, with only the valid lontitudes
    # having image data. We expand to full size for later cropping.
    old_img = metadata['img']
    new_img = ma.zeros((old_img.shape[0], len(metadata['long_antimask'])),
                       dtype=np.float32)
    new_img[:, :] = ma.masked

    new_img[:, metadata['long_antimask']] = old_img
    metadata['img'] = new_img

    add_orbital_metadata(metadata)
    return metadata


def _image_has_satellite(metadata, satellite_dist, satellite_long):
    """Return True if the satellite is present in the image."""
    long_antimask = metadata['long_antimask']
    longitudes = metadata['longitudes'][long_antimask]  # Valid corotating longitudes
    inertial_longitudes = metadata['inertial_longitudes'][long_antimask]
    ETs = metadata['time']  # Scalar for reprojected images, array for mosaics
    if isinstance(ETs, np.ndarray):
        ETs = ETs[long_antimask]

    # Find the closest longitude to the satellite longitude, accounting for
    # the wraparound at 0/360 degrees.
    long_diff = np.abs((longitudes - satellite_long + 180.) % 360. - 180.)
    closest_index = np.argmin(long_diff)
    closest_diff = long_diff[closest_index]
    # Must be within two longitude bins of a valid longitude; this gives us a
    # little leeway for missing data.
    if closest_diff > 2 * arguments.longitude_resolution:
        return False
    # Must have at least two valid longitudes on either side
    # We don't account for wraparound; it's unlikely to matter
    if closest_index < 2 or closest_index >= len(longitudes) - 2:
        return False

    if isinstance(ETs, np.ndarray):
        closest_ET = ETs[closest_index]
        closest_sat_dist = satellite_dist[closest_index]
    else:
        closest_ET = ETs
        closest_sat_dist = satellite_dist
    # fring_radius_at_longitude expects an inertial longitude, not corotating.
    closest_radius = fring_radius_at_longitude(inertial_longitudes[closest_index],
                                               closest_ET)
    radius_sat_dist = closest_radius - closest_sat_dist  # + Prometheus, - Pandora
    return ((radius_sat_dist < 0 and  # Pandora
             radius_sat_dist > -arguments.radius_outer_delta) or
            (radius_sat_dist > 0 and  # Prometheus
             radius_sat_dist < -arguments.radius_inner_delta))


def image_has_prometheus(metadata):
    """Return True if Prometheus is present in the mosaic/reproj imaged."""
    ETs = metadata['time']
    if isinstance(ETs, np.ndarray):
        ETs = ETs[metadata['long_antimask']]
    prometheus_dist, prometheus_corot_long = saturn_to_prometheus_corot(ETs)
    return _image_has_satellite(metadata, prometheus_dist, prometheus_corot_long)


def image_has_pandora(metadata):
    """Return True if Pandora is present in the mosaic/reproj imaged."""
    ETs = metadata['time']
    if isinstance(ETs, np.ndarray):
        ETs = ETs[metadata['long_antimask']]
    pandora_dist, pandora_corot_long = saturn_to_pandora_corot(ETs)
    return _image_has_satellite(metadata, pandora_dist, pandora_corot_long)


def mosaic_has_visual_prometheus(obsid):
    """Return True if Prometheus is visually present in the mosaic."""
    return OBSERVATION_INFO[obsid]['prometheus'] == 'Y'


def mosaic_has_visual_pandora(obsid):
    """Return True if Pandora is visually present in the mosaic."""
    return OBSERVATION_INFO[obsid]['pandora'] == 'Y'


def reformat_iss_name(name):
    """Reformat W1234567890_1 as 1234567890w"""
    name = name.lower()
    return f'{name[1:11]}{name[0]}'


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

    # XXX Change for occultations? or all mosaics?
    # Only include images that we actually used in the name list
    new_image_name_list = [reformat_iss_name(image_name_list[x])
                               for x in number_map.keys() if x != SENTINEL]
    metadata['image_name_list'] = new_image_name_list
    new_image_path_list = [image_path_list[x]
                               for x in number_map.keys() if x != SENTINEL]
    metadata['image_path_list'] = new_image_path_list


def reslice_reproj_img(img, min_corotating_longitude, max_corotating_longitude):
    """Reslice the reprojected image if necessary when min > max."""
    min_corot_long_idx = int(np.round(min_corotating_longitude /
                                      arguments.longitude_resolution))
    max_corot_long_idx = int(np.round(max_corotating_longitude /
                                      arguments.longitude_resolution))
    if min_corotating_longitude < max_corotating_longitude:
        return img[:, min_corot_long_idx:max_corot_long_idx+1]

    # If min > max, there is wraparound, so we need to put the image in
    # the correct spot and then wrap around the end of the image to the
    # beginning.
    # 340 -> 30
    num_long = (max_corot_long_idx - min_corot_long_idx + 1) % img.shape[1]
    if num_long == 0:
        # Full 360-degree coverage; rotate so that column 0 is at the
        # minimum longitude
        return np.roll(img, -min_corot_long_idx, axis=1)
    ret_img = np.zeros((img.shape[0], num_long), dtype=np.float32)
    slice1_size = img.shape[1] - min_corot_long_idx
    ret_img[:, :slice1_size] = img[:, min_corot_long_idx:min_corot_long_idx+slice1_size]
    ret_img[:, slice1_size:] = img[:, :max_corot_long_idx+1]
    return ret_img


def image_name_to_calib_lidvid(name):
    """Convert Cassini ISS image name to a calibrated source product LIDVID.

    urn:nasa:pds:cassini_iss_saturn:data_calibrated:1455008633n_calib::1.0
    """
    name = name.lower()
    return ( 'urn:nasa:pds:cassini_iss_saturn:data_calibrated:'
            f'{name}_calib::1.0')


def image_name_to_reproj_lid(name):
    """Convert Cassini ISS image name to a reprojected image LID.

    urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:data_reproj_img:1551253524n_reproj_img
    """
    name = name.lower()
    return (f'urn:nasa:pds:{BUNDLE_NAME}:data_reproj_img:{name}_reproj_img')


def image_name_to_reproj_lidvid(name):
    """Convert Cassini ISS image name to a reprojected image LIDVID.

    urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:data_reproj_img:1551253524n_reproj_img::1.0
    """
    return image_name_to_reproj_lid(name)+'::1.0'


def image_name_to_reproj_browse_lid(name):
    """Convert Cassini ISS image name to a reprojected browse image LID.

    urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:browse_reproj_img:1551253524n_browse_reproj_img
    """
    name = name.lower()
    return (f'urn:nasa:pds:{BUNDLE_NAME}:browse_reproj_img:{name}_browse_reproj_img')


def image_name_to_reproj_browse_lidvid(name):
    """Convert Cassini ISS image name to a reprojected browse image LIDVID.

    urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:browse_reproj_img:1551253524n_browse_reproj_img::1.0
    """
    return image_name_to_reproj_browse_lid(name)+'::1.0'


def obsid_to_mosaic_lid(obsid, bkg_sub):
    """Convert OBSID IOSIC_276RB_COMPLITB4001_SI to a mosaic or bsm LID.

    urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:data_mosaic:
    iosic_276rb_complitb4001_si_mosaic
        or
    urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025_mosaic_rsfrench2025:data_mosaic_bkg_sub:iosic_276rb_complitb4001_si_mosaic_bkg_sub
    """
    sfx = '_bkg_sub' if bkg_sub else ''
    obsid = obsid.lower()
    return (f'urn:nasa:pds:{BUNDLE_NAME}:data_mosaic{sfx}:{obsid}_mosaic{sfx}')


def obsid_to_mosaic_lidvid(obsid, bkg_sub):
    """Convert OBSID IOSIC_276RB_COMPLITB4001_SI to a mosaic or bsm LIDVID.

    urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:data_mosaic:iosic_276rb_complitb4001_si_mosaic::1.0
        or
    urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025_mosaic_rsfrench2025:data_mosaic_bkg_sub:iosic_276rb_complitb4001_si_mosaic_bkg_sub::1.0
    """
    return obsid_to_mosaic_lid(obsid, bkg_sub)+'::1.0'


def obsid_to_mosaic_browse_lid(obsid, bkg_sub):
    """Convert OBSID to a mosaic or bsm browse LID.

    urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:browse_mosaic:iosic_276rb_complitb4001_si_browse_mosaic
        or
    urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025_mosaic_rsfrench2025:browse_mosaic_bkg_sub:
    iosic_276rb_complitb4001_si_browse_mosaic_bkg_sub
    """
    sfx = '_bkg_sub' if bkg_sub else ''
    obsid = obsid.lower()
    return (f'urn:nasa:pds:{BUNDLE_NAME}:browse_mosaic{sfx}:{obsid}_browse_mosaic{sfx}')


def obsid_to_mosaic_browse_lidvid(obsid, bkg_sub):
    """Convert OBSID to a mosaic or bsm browse LIDVID.

    urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:browse_mosaic:iosic_276rb_complitb4001_si_browse_mosaic::1.0
        or
    urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:browse_mosaic_bkg_sub:iosic_276rb_complitb4001_si_browse_mosaic_bkg_sub::1.0
    """
    return obsid_to_mosaic_browse_lid(obsid, bkg_sub)+'::1.0'


TOUR_PRE_HUYGENS_END_ET = utc2et('2004-359T00:00:00.000')
TOUR_END_ET = utc2et('2008-183T00:00:00.000')
EQUINOX_MISSION_END_ET = utc2et('2010-273T00:00:00.000')

def et_to_tour(et):
    """Convert ET to PDS4 Cassini Tour name.

    See https://github.com/pds-data-dictionaries/ldd-cassini/blob/main/src/
        PDS4_CASSINI_IngestLDD.xml
    """
    if et < TOUR_PRE_HUYGENS_END_ET:
        return 'TOUR PRE-HUYGENS'
    if et < TOUR_END_ET:
        return 'TOUR'
    if et < EQUINOX_MISSION_END_ET:
        return 'EQUINOX MISSION'
    return 'SOLSTICE MISSION'


def read_label(image_name):
    """Return the PDS3 label for the given image name.

    This is needed to lookup various image metadata which isn't stored in the
    mosaic metadata.
    """
    components = image_name.split('/')[-5:]
    image_path = os.path.join(CALIBRATED_DIR, *components)
    label_path = image_path.replace('.IMG', '.LBL')
    return Pds3Label(label_path, method='fast')


def compute_mid_sclk(start_sclk, stop_sclk):
    """Compute the mid-time SCLK from the start and stop SCLKs.

    Cassini SCLKs have an integer part and a fractional part. The fractional
    part is 3 digits 0-255.
    """
    start_sclk_int, start_sclk_frac = start_sclk.split('.')
    stop_sclk_int, stop_sclk_frac = stop_sclk.split('.')
    start_sclk = int(start_sclk_int) + int(start_sclk_frac)/256
    stop_sclk = int(stop_sclk_int) + int(stop_sclk_frac)/256
    mid_sclk = (start_sclk + stop_sclk)/2
    mid_sclk_int = int(mid_sclk)
    mid_sclk_frac = int((mid_sclk - mid_sclk_int)*256)
    return f'{mid_sclk_int}.{mid_sclk_frac:03d}'


def ra_dec_from_cmat(cmat):
    z = unit(cmat[2, :])
    ra = np.arctan2(z[1], z[0])
    dec = np.arcsin(z[2])
    if ra < 0: ra += 2 * np.pi
    return ra, dec


def unit(v):
    return v / np.linalg.norm(v)


def extract_roll_from_cmat(cmat):
    z = unit(cmat[2, :])
    x = unit(cmat[0, :])

    # Reference X-axis for constructing ideal frame (Z-perp)
    temp = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_ref = unit(np.cross(temp, z))
    y_ref = np.cross(z, x_ref)

    # Project original X into the Z-plane and normalize
    x_proj = unit(x - np.dot(x, z) * z)

    # Compute roll angle between projected X and ref X/Y
    roll = np.arctan2(np.dot(x_proj, y_ref), np.dot(x_proj, x_ref))
    return roll


def rebuild_cmatrix_from_ra_dec_roll(ra, dec, roll):
    # Construct Z axis from RA/DEC
    z = np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])
    z = unit(z)

    # Construct reference X-axis
    temp = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_ref = unit(np.cross(temp, z))
    y_ref = np.cross(z, x_ref)

    # Apply roll about Z
    cos_r = np.cos(roll)
    sin_r = np.sin(roll)

    x = cos_r * x_ref + sin_r * y_ref
    y = -sin_r * x_ref + cos_r * y_ref

    # Assemble new matrix (X, Y, Z as rows)
    return np.stack([x, y, z], axis=0)


def ra_rad_to_hms(ra):
    """Convert right ascension in radians to a pretty string."""
    ra_deg = ra*oops.DPR/15 # In hours
    hh = int(ra_deg)
    mm = int((ra_deg-hh)*60)
    ss = (ra_deg-hh-mm/60.)*3600
    return f"{hh:02d}h{mm:02d}m{ss:05.3f}s"


def dec_rad_to_deg(dec):
    """Convert declination in radians to a pretty string."""
    dec_deg = dec*oops.DPR # In degrees
    neg = "+"
    if dec_deg < 0.:
        neg = "-"
        dec_deg = -dec_deg
    dd = int(dec_deg)
    mm = int((dec_deg-dd)*60)
    ss = (dec_deg-dd-mm/60.)*3600
    return f"{neg}{dd:03d}d{mm:02d}m{ss:05.3f}s"


def write_suppl_file(output_path, metadata, xml_metadata):
    """Write the supplemental file for the reprojected image."""
    offset_path = img_to_offset_path(metadata['image_path'])
    offset_metadata = read_offset_metadata_path(offset_path)

    if 'manual_offset' in offset_metadata:
        nav_type = 'Manual'
        offset = offset_metadata['manual_offset']
    elif offset_metadata['offset_winner'] == 'MODEL':
        nav_type = 'Ring and/or Satellite Models'
        offset = offset_metadata['offset']
    else:
        nav_type = 'Stars'
        offset = offset_metadata['offset']

    image_path = metadata['image_path']

    # coiss.initialize(ck='predicted', spk='predicted')
    obs = coiss.from_file(image_path, fast_distortion=True, return_all_planets=True)

    if False:
        # offset = (0, 0)
        print(f'Offset U = {offset[0]:.6f} pixels')
        print(f'Offset V = {offset[1]:.6f} pixels')

        # 1726771710w
        # cmat_matt = np.array([[0.34458504, -0.37933220, -0.85870148],
        #                       [0.75498380,  0.65560755,  0.013349307],
        #                       [0.55790735, -0.65290567,  0.51230222]])

        # 1726808835w
        cmat_matt = np.array([[0.34517947, -0.38065358, -0.85787760],
                            [0.72756155,  0.68594401, -0.011619060],
                            [0.59287884, -0.62014810,  0.51372270]])

        cmat_matt[0, :] = -cmat_matt[0, :]
        cmat_matt[1, :] = -cmat_matt[1, :]
        ra_matt, dec_matt = ra_dec_from_cmat(cmat_matt)
        roll_matt = extract_roll_from_cmat(cmat_matt)

        cmat_matt_neg = cmat_matt.copy()
        cmat_matt_neg[0, :] = -cmat_matt_neg[0, :]
        cmat_matt_neg[1, :] = -cmat_matt_neg[1, :]
        ra_matt_neg, dec_matt_neg = ra_dec_from_cmat(cmat_matt_neg)
        roll_matt_neg = extract_roll_from_cmat(cmat_matt_neg)

        print()
        print('Matt CMAT:')
        print(cmat_matt)
        print('Matt RA/Dec:')
        print(f'RA  = {np.rad2deg(ra_matt):.6} deg ({ra_rad_to_hms(ra_matt)})')
        print(f'Dec = {np.rad2deg(dec_matt):.6} deg ({dec_rad_to_deg(dec_matt)})')
        print(f'Roll = {np.rad2deg(roll_matt):.6} deg')
        boresight_inst_frame = [0, 0, 1]
        cmat_matt_T = cmat_matt.T
        boresight_j2000 = cspyce.mxv(cmat_matt_T, boresight_inst_frame)
        (range_, ra_matt2, dec_matt2) = cspyce.recrad(boresight_j2000)
        print('Matt RA/Dec using CSPYCE:')
        print(f'RA  = {np.rad2deg(ra_matt2):.6} deg ({ra_rad_to_hms(ra_matt2)})')
        print(f'Dec = {np.rad2deg(dec_matt2):.6} deg ({dec_rad_to_deg(dec_matt2)})')

        meshgrid_ctr = oops.Meshgrid.for_fov(obs.fov,
                                            origin=(obs.shape[1]//2, obs.shape[0]//2),
                                            limit=(obs.shape[1]//2, obs.shape[0]//2),
                                            swap=True)
        bp = oops.backplane.Backplane(obs)
        bp_ctr = oops.backplane.Backplane(obs, meshgrid=meshgrid_ctr)

        oops_ra_app = bp.right_ascension(apparent=True)
        oops_dec_app = bp.declination(apparent=True)

        ra_min = oops_ra_app.min()
        ra_max = oops_ra_app.max()
        if ra_max-ra_min > oops.PI:
            # Wrap around
            ra_min = oops_ra_app[np.where(oops_ra_app>np.pi)].min()
            ra_max = oops_ra_app[np.where(oops_ra_app<np.pi)].max()

        dec_min = oops_dec_app.min()
        dec_max = oops_dec_app.max()
        if dec_max-dec_min > oops.PI:
            # Wrap around
            dec_min = oops_dec_app[np.where(oops_dec_app>np.pi)].min()
            dec_max = oops_dec_app[np.where(oops_dec_app<np.pi)].max()

        ra_min = ra_min.vals
        ra_max = ra_max.vals
        dec_min = dec_min.vals
        dec_max = dec_max.vals

        oops_ra_app_ctr = bp_ctr.right_ascension(apparent=True).vals[0][0]
        oops_dec_app_ctr = bp_ctr.declination(apparent=True).vals[0][0]

        print()
        print('Freshly calculated non-navigated apparent=True:')
        print(f'RA  Min = {np.rad2deg(ra_min):.6} deg ({ra_rad_to_hms(ra_min)})')
        print(f'RA  Max = {np.rad2deg(ra_max):.6} deg ({ra_rad_to_hms(ra_max)})')
        print(f'Dec Min = {np.rad2deg(dec_min):.6} deg ({dec_rad_to_deg(dec_min)})')
        print(f'Dec Max = {np.rad2deg(dec_max):.6} deg ({dec_rad_to_deg(dec_max)})')
        print(f'RA  Ctr = {np.rad2deg(oops_ra_app_ctr):.6} deg ({ra_rad_to_hms(oops_ra_app_ctr)})')
        print(f'Dec Ctr = {np.rad2deg(oops_dec_app_ctr):.6} deg ({dec_rad_to_deg(oops_dec_app_ctr)})')

        print('Offset file apparent=True:')
        ra_min, ra_max, dec_min, dec_max = offset_metadata['ra_dec_corner_app']
        print(f'RA  Min = {np.rad2deg(ra_min):.6} deg ({ra_rad_to_hms(ra_min)})')
        print(f'RA  Max = {np.rad2deg(ra_max):.6} deg ({ra_rad_to_hms(ra_max)})')
        print(f'Dec Min = {np.rad2deg(dec_min):.6} deg ({dec_rad_to_deg(dec_min)})')
        print(f'Dec Max = {np.rad2deg(dec_max):.6} deg ({dec_rad_to_deg(dec_max)})')
        ra_ctr, dec_ctr = offset_metadata['ra_dec_center_app']
        print(f'RA  Ctr = {np.rad2deg(ra_ctr):.6} deg ({ra_rad_to_hms(ra_ctr)})')
        print(f'Dec Ctr = {np.rad2deg(dec_ctr):.6} deg ({dec_rad_to_deg(dec_ctr)})')

        oops_ra_non_app = bp.right_ascension(apparent=False)
        oops_dec_non_app = bp.declination(apparent=False)

        ra_min = oops_ra_non_app.min()
        ra_max = oops_ra_non_app.max()
        if ra_max-ra_min > oops.PI:
            # Wrap around
            ra_min = oops_ra_non_app[np.where(oops_ra_non_app>np.pi)].min()
            ra_max = oops_ra_non_app[np.where(oops_ra_non_app<np.pi)].max()

        dec_min = oops_dec_non_app.min()
        dec_max = oops_dec_non_app.max()
        if dec_max-dec_min > oops.PI:
            # Wrap around
            dec_min = oops_dec_non_app[np.where(oops_dec_non_app>np.pi)].min()
            dec_max = oops_dec_non_app[np.where(oops_dec_non_app<np.pi)].max()

        ra_min = ra_min.vals
        ra_max = ra_max.vals
        dec_min = dec_min.vals
        dec_max = dec_max.vals

        oops_ra_non_app_ctr = bp_ctr.right_ascension(apparent=False).vals[0][0]
        oops_dec_non_app_ctr = bp_ctr.declination(apparent=False).vals[0][0]

        print()
        print('Freshly calculated non-navigated apparent=False:')
        print(f'RA  Min = {np.rad2deg(ra_min):.6} deg ({ra_rad_to_hms(ra_min)})')
        print(f'RA  Max = {np.rad2deg(ra_max):.6} deg ({ra_rad_to_hms(ra_max)})')
        print(f'Dec Min = {np.rad2deg(dec_min):.6} deg ({dec_rad_to_deg(dec_min)})')
        print(f'Dec Max = {np.rad2deg(dec_max):.6} deg ({dec_rad_to_deg(dec_max)})')
        print(f'RA  Ctr = {np.rad2deg(oops_ra_non_app_ctr):.6} deg ({ra_rad_to_hms(oops_ra_non_app_ctr)})')
        print(f'Dec Ctr = {np.rad2deg(oops_dec_non_app_ctr):.6} deg ({dec_rad_to_deg(oops_dec_non_app_ctr)})')
        print('OOPS center apparent=False:')
        ra_ctr, dec_ctr = offset_metadata['ra_dec_center']
        print(f'RA  Ctr = {np.rad2deg(ra_ctr):.6} deg ({ra_rad_to_hms(ra_ctr)})')
        print(f'Dec Ctr = {np.rad2deg(dec_ctr):.6} deg ({dec_rad_to_deg(dec_ctr)})')

        # print('OOPS ra dec', ra, dec)
        cmat = cspyce.pxform('J2000', 'CASSINI_ISS_WAC', obs.midtime)
        xform_j2000_wac = cmat
        xform_wac_j2000 = cmat.T
        xform_nac_j2000 = cspyce.pxform('CASSINI_ISS_NAC', 'J2000', obs.midtime)
        ra_cmat, dec_cmat = ra_dec_from_cmat(cmat)
        print()
        print('CSPICE center:')
        print(f'RA  Ctr = {np.rad2deg(ra_cmat):.6} deg ({ra_rad_to_hms(ra_cmat)})')
        print(f'Dec Ctr = {np.rad2deg(dec_cmat):.6} deg ({dec_rad_to_deg(dec_cmat)})')
        roll = extract_roll_from_cmat(cmat)
        print(f'Roll = {np.rad2deg(roll):.6} deg')
        cmat_new = rebuild_cmatrix_from_ra_dec_roll(ra_cmat, dec_cmat, roll)
        ra_recon, dec_recon = ra_dec_from_cmat(cmat_new)
        print('CSPICE reconstructed center (should be the same):')
        print(f'RA  Ctr = {np.rad2deg(ra_recon):.6} deg ({ra_rad_to_hms(ra_recon)})')
        print(f'Dec Ctr = {np.rad2deg(dec_recon):.6} deg ({dec_rad_to_deg(dec_recon)})')
        print()
        print('CMAT from pxform:')
        print(cmat)
        print('Reconstructed CMAT (should be the same):')
        print(cmat_new)
        print()
        ISS_PIXEL = 60e-6
        ang_dist = np.arccos(np.sin(dec_matt)*np.sin(dec_recon) + np.cos(dec_matt)*np.cos(dec_recon)*np.cos(ra_matt-ra_recon))
        print('Angular distance (deg):', np.rad2deg(ang_dist))
        print('Angular distance (pix):', ang_dist / ISS_PIXEL)

        obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)
        meshgrid_ctr = oops.Meshgrid.for_fov(obs.fov,
                                            origin=(obs.shape[1]//2, obs.shape[0]//2),
                                            limit=(obs.shape[1]//2, obs.shape[0]//2),
                                            swap=True)
        bp_ctr_nav = oops.backplane.Backplane(obs, meshgrid=meshgrid_ctr)

        oops_ra_app_ctr_nav = bp_ctr_nav.right_ascension(apparent=True).vals[0][0]
        oops_dec_app_ctr_nav = bp_ctr_nav.declination(apparent=True).vals[0][0]

        print()
        print('Navigated apparent=True:')
        print(f'RA  Ctr = {np.rad2deg(oops_ra_app_ctr_nav):.6} deg ({ra_rad_to_hms(oops_ra_app_ctr_nav)})')
        print(f'Dec Ctr = {np.rad2deg(oops_dec_app_ctr_nav):.6} deg ({dec_rad_to_deg(oops_dec_app_ctr_nav)})')
        ang_dist = np.arccos(np.sin(dec_matt)*np.sin(oops_dec_app_ctr_nav) + np.cos(dec_matt)*np.cos(oops_dec_app_ctr_nav)*np.cos(ra_matt-oops_ra_app_ctr_nav))
        print('Angular distance (deg):', np.rad2deg(ang_dist))
        print('Angular distance (pix):', ang_dist / ISS_PIXEL)

        oops_ra_non_app_ctr_nav = bp_ctr_nav.right_ascension(apparent=False).vals[0][0]
        oops_dec_non_app_ctr_nav = bp_ctr_nav.declination(apparent=False).vals[0][0]

        print()
        print('Navigated apparent=False:')
        print(f'RA  Ctr = {np.rad2deg(oops_ra_non_app_ctr_nav):.6} deg ({ra_rad_to_hms(oops_ra_non_app_ctr_nav)})')
        print(f'Dec Ctr = {np.rad2deg(oops_dec_non_app_ctr_nav):.6} deg ({dec_rad_to_deg(oops_dec_non_app_ctr_nav)})')
        ang_dist = np.arccos(np.sin(dec_matt)*np.sin(oops_dec_non_app_ctr_nav) + np.cos(dec_matt)*np.cos(oops_dec_non_app_ctr_nav)*np.cos(ra_matt-oops_ra_non_app_ctr_nav))
        print('Angular distance (deg):', np.rad2deg(ang_dist))
        print('Angular distance (pix):', ang_dist / ISS_PIXEL)

        cmat_nav = rebuild_cmatrix_from_ra_dec_roll(oops_ra_non_app_ctr_nav, oops_dec_non_app_ctr_nav, roll)

        print()
        print('CMAT from navigated apparent=False:')
        print(cmat_nav)

        print()
        cmatt_matt_in_wac = np.dot(xform_j2000_wac, np.dot(xform_nac_j2000, cmat_matt))
        print('Matt CMAT in WAC:')
        print(cmatt_matt_in_wac)
        ra_matt_in_wac, dec_matt_in_wac = ra_dec_from_cmat(cmatt_matt_in_wac)
        print('Matt RA/Dec in WAC:')
        print(f'RA  = {np.rad2deg(ra_matt_in_wac):.6} deg ({ra_rad_to_hms(ra_matt_in_wac)})')
        print(f'Dec = {np.rad2deg(dec_matt_in_wac):.6} deg ({dec_rad_to_deg(dec_matt_in_wac)})')
        roll_matt_in_wac = extract_roll_from_cmat(cmatt_matt_in_wac)
        print(f'Roll = {np.rad2deg(roll_matt_in_wac):.6} deg')
        ang_dist = np.arccos(np.sin(dec_matt)*np.sin(dec_matt_in_wac) + np.cos(dec_matt)*np.cos(dec_matt_in_wac)*np.cos(ra_matt-ra_matt_in_wac))
        print('Angular distance (deg):', np.rad2deg(ang_dist))
        print('Angular distance (pix):', ang_dist / ISS_PIXEL)

        print()
        print()
        print('** Final comparison **')
        print('Matt CMAT (apparent=False):')
        print(cmat_matt)
        print()
        print('My navigated CMAT (apparent=False):')
        print(cmat_nav)
        print()
        ang_dist = np.arccos(np.sin(dec_matt)*np.sin(oops_dec_non_app_ctr_nav) + np.cos(dec_matt)*np.cos(oops_dec_non_app_ctr_nav)*np.cos(ra_matt-oops_ra_non_app_ctr_nav))
        ang_dist_neg = np.arccos(np.sin(dec_matt_neg)*np.sin(oops_dec_non_app_ctr_nav) + np.cos(dec_matt_neg)*np.cos(oops_dec_non_app_ctr_nav)*np.cos(ra_matt_neg-oops_ra_non_app_ctr_nav))
        ang_dist_pix = ang_dist / ISS_PIXEL
        ang_dist_pix_neg = ang_dist_neg / ISS_PIXEL
        print('                      RA         DEC         ROLL     DIFF (pixels)')
        print(f'Rob (app=False) {np.rad2deg(oops_ra_non_app_ctr_nav):10.6}  {np.rad2deg(oops_dec_non_app_ctr_nav):10.6}  {np.rad2deg(roll):10.6}')
        print(f'Rob (app=True)  {np.rad2deg(oops_ra_app_ctr_nav):10.6}  {np.rad2deg(oops_dec_app_ctr_nav):10.6}  {np.rad2deg(roll):10.6}')
        print(f'Matt (current)  {np.rad2deg(ra_matt):10.6}  {np.rad2deg(dec_matt):10.6}  {np.rad2deg(roll_matt):10.6}  {ang_dist_pix:10.6}')
        print(f'Matt (negated)  {np.rad2deg(ra_matt_neg):10.6}  {np.rad2deg(dec_matt_neg):10.6}  {np.rad2deg(roll_matt_neg):10.6}  {ang_dist_pix_neg:10.6}')
        print()
        return 0

    obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=offset)
    meshgrid_ctr = oops.Meshgrid.for_fov(obs.fov,
                                         origin=(obs.shape[1]//2, obs.shape[0]//2),
                                         limit=(obs.shape[1]//2, obs.shape[0]//2),
                                         swap=True)
    bp_ctr_nav = oops.backplane.Backplane(obs, meshgrid=meshgrid_ctr)

    oops_ra_non_app_ctr_nav = bp_ctr_nav.right_ascension(apparent=False).vals[0][0]
    oops_dec_non_app_ctr_nav = bp_ctr_nav.declination(apparent=False).vals[0][0]

    image_name = metadata['image_name']

    if image_name[-1] == 'w':
        cmat = cspyce.pxform('J2000', 'CASSINI_ISS_WAC', obs.midtime)
    elif image_name[-1] == 'n':
        cmat = cspyce.pxform('J2000', 'CASSINI_ISS_NAC', obs.midtime)
    else:
        assert False, f'Unknown image name camera {image_name}'

    roll = extract_roll_from_cmat(cmat)

    cmat_nav = rebuild_cmatrix_from_ra_dec_roll(oops_ra_non_app_ctr_nav, oops_dec_non_app_ctr_nav, roll)

    start_date = xml_metadata['START_DATE_TIME_3']
    partition = xml_metadata['SPACECRAFT_CLOCK_CNT_PARTITION']
    start_sclk = xml_metadata['SPACECRAFT_CLOCK_START_COUNT']
    mid_date = xml_metadata['MIDTIME_DATE_TIME_3']
    mid_sclk = xml_metadata['SPACECRAFT_CLOCK_MID_COUNT']
    stop_date = xml_metadata['STOP_DATE_TIME_3']
    stop_sclk = xml_metadata['SPACECRAFT_CLOCK_STOP_COUNT']
    hdr_text = 'This file contains a C-matrix that describes the rotation from the J2000 reference\n'
    hdr_text += 'frame to the camera pointing based upon analysis of the contents of the image.\n\n'
    hdr_text += f'Source Data Product ID = {image_name}_calib\n'
    hdr_text += f'Image Start Time (SCLK) = {partition}/{start_sclk}\n'
    hdr_text += f'Image Start Time (UTC) = {start_date}\n'
    hdr_text += f'Image Mid Time (SCLK) = {partition}/{mid_sclk}\n'
    hdr_text += f'Image Mid Time (UTC) = {mid_date}\n'
    hdr_text += f'Image Stop Time (SCLK) = {partition}/{stop_sclk}\n'
    hdr_text += f'Image Stop Time (UTC) = {stop_date}\n'
    hdr_text += f'Trajectory Kernels Query Time = Observation mid time\n'
    hdr_text += 'Stellar Aberration Correction = No\n'
    hdr_text += 'Light Travel Time Correction = No\n'
    hdr_text += f'Navigation Type = {nav_type}\n'
    hdr_text += f'Navigated Boresight RA = {np.rad2deg(oops_ra_non_app_ctr_nav):.6} deg ({ra_rad_to_hms(oops_ra_non_app_ctr_nav)})\n'
    hdr_text += f'Navigated Boresight Dec = {np.rad2deg(oops_dec_non_app_ctr_nav):.6} deg ({dec_rad_to_deg(oops_dec_non_app_ctr_nav)})\n'
    hdr_text += f'Navigated Boresight Roll = {np.rad2deg(roll):.6} deg\n'
    hdr_text += 'C-Matrix = \n'

    c_matrix_text = ''
    for i in range(3):
        for j in range(3):
            c_matrix_text += f'{cmat_nav[i,j]:16.10f}'
        c_matrix_text += '\n'

    LOGGER.info(f'Writing supplemental file for {image_name} to {output_path}')
    with open(output_path, 'w') as f:
        f.write(hdr_text)
        f.write(c_matrix_text)

    return len(hdr_text)


def read_observation_list():
    """Read the observation list from the observation list CSV file.

    The CSV file has the following columns:
        Observation,Inertial,Nav qual,Bkgnd qual,Prometheus,Pandora,Notes,

    Notes:
        B: Background-subtracted mosaic is missing data due to insufficient radial extent
        C: Some source images have corrupted or missing data
        E: Some areas may be overexposed
        M1: Multiple contiguous observations of the same inertial longitude range
        M2: One of a pair of observations taken at inertial longitudes roughly 180 degrees apart
        M3: Multiple observations of the same co-rotating longitude range but different inertial
        M4: Observations of different co-rotating and different inertial longitudes
        N: Non-inertial
        O: Occultation
        R: Follows one co-rotating longitude range with different inertial longitudes
    """
    global OBSERVATION_INFO
    OBSERVATION_INFO = {}
    # Use the csv module to read the observation list CSV file.
    import csv
    with open(OBSERVATION_LIST_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row.
        for row in reader:
            obsid, inertial, nav_qual, bkgnd_qual, prometheus, pandora, notes, _ = row
            OBSERVATION_INFO[obsid] = {
                'inertial': inertial,
                'nav_qual': nav_qual,
                'bkgnd_qual': bkgnd_qual,
                'prometheus': prometheus,
                'pandora': pandora,
                'notes': notes,
            }


##########################################################################################
#
# PROCESS A MOSAIC
#
##########################################################################################

def xml_metadata_for_image(obsid, metadata, bkgnd_metadata, img_type):
    """Generate the common template substitions for all image types.

    img_type is 'm' (basic mosasic), 'b' (bsm), or 'r' (reprojected image).
    """
    ret = BASIC_XML_METADATA.copy()

    ret['CURRENT_DATE'] = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d')
    ret['CURRENT_DATE_TIME'] = datetime.datetime.now(datetime.UTC).strftime(
        '%Y-%m-%dT%H:%M:%SZ')

    long_antimask = metadata['long_antimask']

    match = re.search(r'^(.*)_(\d+)$', obsid)
    partial_obsid = bool(match)
    root_obsid = obsid
    obsid_chunk = None
    if partial_obsid:
        root_obsid = match[1]
        obsid_chunk = match[2]

    ret['FULL_OBSERVATION_ID'] = obsid
    ret['MOSAIC_OBSERVATION_ID_ROOT'] = root_obsid
    ret['MOSAIC_OBSERVATION_ID_CHUNK'] = obsid_chunk

    if img_type == 'r':
        max_image_path = min_image_path = metadata['image_path']
    else:
        ETs = metadata['time'][long_antimask]

        # Find the image names at the starting and ending ETs
        image_indexes = metadata['image_number'][long_antimask]
        image_path_list = metadata['image_path_list']
        image_name_list = metadata['image_name_list']
        idx_min = np.argmin(ETs)
        idx_max = np.argmax(ETs)
        min_image_path = image_path_list[image_indexes[idx_min]]
        ret['MIN_IMAGE_NAME'] = image_name_list[image_indexes[idx_min]]
        max_image_path = image_path_list[image_indexes[idx_max]]
        ret['MAX_IMAGE_NAME'] = image_name_list[image_indexes[idx_max]]

    xml_add_pds3_label_info(ret, obsid, min_image_path, max_image_path)

    ret['NUM_VALID_LONGITUDES'] = np.sum(long_antimask)
    ret['MOSAIC_LID'] = obsid_to_mosaic_lid(obsid, img_type == 'b')
    ret['MOSAIC_ORIGINAL_LID'] = obsid_to_mosaic_lid(obsid, False)
    ret['MOSAIC_BKG_SUB_LID'] = obsid_to_mosaic_lid(obsid, True)
    ret['BROWSE_MOSAIC_LID'] = obsid_to_mosaic_browse_lid(obsid, img_type == 'b')

    xml_add_comments(ret, img_type, obsid, metadata, bkgnd_metadata)

    if img_type == 'r':
        incidence_angle = np.degrees(metadata['incidence'])
    else:
        incidence_angle = np.degrees(metadata['mean_incidence'])

    ret['MEAN_INCIDENCE_ANGLE'] = f'{incidence_angle:.3f}'
    ret['MEAN_INCIDENCE_ANGLE_FIXED'] = f'{incidence_angle:6.3f}'

    if img_type == 'r':
        emission_angles = np.degrees(metadata['mean_emission'])
        phase_angles = np.degrees(metadata['mean_phase'])
        rad_resolutions = metadata['mean_radial_resolution']
        ang_resolutions = np.degrees(metadata['mean_angular_resolution'])
    else:
        emission_angles = np.degrees(metadata['mean_emission'][long_antimask])
        phase_angles = np.degrees(metadata['mean_phase'][long_antimask])
        rad_resolutions = metadata['mean_radial_resolution'][long_antimask]
        ang_resolutions = np.degrees(metadata['mean_angular_resolution'][long_antimask])

    # XXX Implement difference between emission angle and observed ring elevation
    ret['MEAN_EMISSION_ANGLE'] = f'{np.mean(emission_angles):.3f}'
    ret['MEAN_EMISSION_ANGLE_FIXED'] = f'{np.mean(emission_angles):7.3f}'
    ret['MIN_EMISSION_ANGLE'] = f'{np.min(emission_angles):.3f}'
    ret['MIN_EMISSION_ANGLE_FIXED'] = f'{np.min(emission_angles):7.3f}'
    ret['MAX_EMISSION_ANGLE'] = f'{np.max(emission_angles):.3f}'
    ret['MAX_EMISSION_ANGLE_FIXED'] = f'{np.max(emission_angles):7.3f}'

    ret['MEAN_PHASE_ANGLE'] = f'{np.mean(phase_angles):.3f}'
    ret['MEAN_PHASE_ANGLE_FIXED'] = f'{np.mean(phase_angles):7.3f}'
    ret['MIN_PHASE_ANGLE'] = f'{np.min(phase_angles):.3f}'
    ret['MIN_PHASE_ANGLE_FIXED'] = f'{np.min(phase_angles):7.3f}'
    ret['MAX_PHASE_ANGLE'] = f'{np.max(phase_angles):.3f}'
    ret['MAX_PHASE_ANGLE_FIXED'] = f'{np.max(phase_angles):7.3f}'

    ret['MEAN_REPROJ_GRID_RAD_RES'] = f'{np.mean(rad_resolutions):.3f}'
    ret['MEAN_REPROJ_GRID_RAD_RES_FIXED'] = f'{np.mean(rad_resolutions):8.3f}'
    ret['MIN_REPROJ_GRID_RAD_RES'] = f'{np.min(rad_resolutions):.3f}'
    ret['MIN_REPROJ_GRID_RAD_RES_FIXED'] = f'{np.min(rad_resolutions):8.3f}'
    ret['MAX_REPROJ_GRID_RAD_RES'] = f'{np.max(rad_resolutions):.3f}'
    ret['MAX_REPROJ_GRID_RAD_RES_FIXED'] = f'{np.max(rad_resolutions):8.3f}'

    ret['MEAN_REPROJ_GRID_ANG_RES'] = f'{np.mean(ang_resolutions):.5f}'
    ret['MEAN_REPROJ_GRID_ANG_RES_FIXED'] = f'{np.mean(ang_resolutions):7.5f}'
    ret['MIN_REPROJ_GRID_ANG_RES'] = f'{np.min(ang_resolutions):.5f}'
    ret['MIN_REPROJ_GRID_ANG_RES_FIXED'] = f'{np.min(ang_resolutions):7.5f}'
    ret['MAX_REPROJ_GRID_ANG_RES'] = f'{np.max(ang_resolutions):.5f}'
    ret['MAX_REPROJ_GRID_ANG_RES_FIXED'] = f'{np.max(ang_resolutions):7.5f}'

    inertial_longitudes = metadata['inertial_longitudes'][long_antimask]
    if img_type == 'r':
        radii = fring_radius_at_longitude(inertial_longitudes,
                                          metadata['time'])
    else:
        radii = fring_radius_at_longitude(inertial_longitudes,
                                          metadata['time'][long_antimask])
    min_radius = np.min(radii)
    max_radius = np.max(radii)
    mean_radius = np.mean(radii)
    ret['MIN_CORE_RADIUS'] = f'{min_radius:.3f}'
    ret['MIN_CORE_RADIUS_FIXED'] = f'{min_radius:10.3f}'
    ret['MAX_CORE_RADIUS'] = f'{max_radius:.3f}'
    ret['MAX_CORE_RADIUS_FIXED'] = f'{max_radius:10.3f}'
    ret['MEAN_CORE_RADIUS'] = f'{mean_radius:.3f}'
    ret['MEAN_CORE_RADIUS_FIXED'] = f'{mean_radius:10.3f}'
    ret['MIN_RING_RADIUS'] = f'{min_radius+arguments.radius_inner_delta:.3f}'
    ret['MIN_RING_RADIUS_FIXED'] = f'{min_radius+arguments.radius_inner_delta:10.3f}'
    ret['MAX_RING_RADIUS'] = f'{max_radius+arguments.radius_outer_delta:.3f}'
    ret['MAX_RING_RADIUS_FIXED'] = f'{max_radius+arguments.radius_outer_delta:10.3f}'

    if img_type != 'r':
        image_name_list = metadata['image_name_list']
        ret['NUM_IMAGES'] = len(image_name_list)
        image_name0 = metadata['image_name_list'][0]
    else:
        _, image_name0 = os.path.split(metadata['image_path'])
        image_name0 = reformat_iss_name(image_name0)
    camera = image_name0[-1]
    if camera not in ('n', 'w'):
        LOGGER.fatal(f'Unknown camera for image {image_name0}')
        raise ObsIdFailedException
    if img_type != 'r':
        for image_name in image_name_list:
            if image_name[-1] != camera:
                LOGGER.error(f'{obsid}: Inconsistent cameras for images '
                            f'{image_name0} and {image_name}')
                break
    if camera == 'n':
        ret['CAMERA_WIDTH'] = 'Narrow'
        ret['CAMERA_WN_UC'] = 'N'
        ret['CAMERA_WN_LC'] = 'n'
    else:
        ret['CAMERA_WIDTH'] = 'Wide'
        ret['CAMERA_WN_UC'] = 'W'
        ret['CAMERA_WN_LC'] = 'w'

    return ret


def xml_add_pds3_label_info(ret, obsid, min_image_path, max_image_path):
    """Add PDS3 label information to the XML metadata."""
    try:
        min_label = read_label(min_image_path)
    except FileNotFoundError:
        LOGGER.error(f'{obsid}: Failed to open label file {min_image_path}')
        raise ObsIdFailedException
    except pyparsing.exceptions.ParseException:
        LOGGER.error(f'{obsid}: Failed to parse label file {min_image_path}')
        raise ObsIdFailedException
    ret['SPACECRAFT_CLOCK_START_COUNT'] = str(min_label['SPACECRAFT_CLOCK_START_COUNT'])
    ret['SPACECRAFT_CLOCK_CNT_PARTITION'] = min_label['SPACECRAFT_CLOCK_CNT_PARTITION']
    ret['START_TIME_DOY'] = julian.iso_from_tai(julian.tai_from_tdb(
        julian.tdb_from_iso(min_label['START_TIME'])),
        ymd=False, digits=3)
    if min_image_path == max_image_path:
        ret['SPACECRAFT_CLOCK_STOP_COUNT'] = str(min_label['SPACECRAFT_CLOCK_STOP_COUNT'])
        ret['STOP_TIME_DOY'] = julian.iso_from_tai(julian.tai_from_tdb(
        julian.tdb_from_iso(min_label['STOP_TIME'])),
        ymd=False, digits=3)
    else:
        try:
            max_label = read_label(max_image_path)
        except FileNotFoundError:
            LOGGER.error(f'{obsid}: Failed to open label file {max_image_path}')
            raise ObsIdFailedException
        except pyparsing.exceptions.ParseException:
            LOGGER.error(f'{obsid}: Failed to parse label file {max_image_path}')
            raise ObsIdFailedException
        ret['SPACECRAFT_CLOCK_STOP_COUNT'] = str(max_label['SPACECRAFT_CLOCK_STOP_COUNT'])
        ret['STOP_TIME_DOY'] = julian.iso_from_tai(julian.tai_from_tdb(
            julian.tdb_from_iso(max_label['STOP_TIME'])),
            ymd=False, digits=3)
    ret['SPACECRAFT_CLOCK_MID_COUNT'] = compute_mid_sclk(ret['SPACECRAFT_CLOCK_START_COUNT'],
                                                         ret['SPACECRAFT_CLOCK_STOP_COUNT'])

    et_start_time = julian.tdb_from_iso(ret['START_TIME_DOY'])
    et_stop_time = julian.tdb_from_iso(ret['STOP_TIME_DOY'])
    ret['START_DATE_TIME'] = et_to_datetime(et_start_time)
    ret['START_DATE_TIME_3'] = et_to_datetime(et_start_time, dec=3)
    ret['STOP_DATE_TIME'] = et_to_datetime(et_stop_time)
    ret['STOP_DATE_TIME_3'] = et_to_datetime(et_stop_time, dec=3)
    ret['MIDTIME_ET'] = (et_start_time + et_stop_time)/2
    ret['MIDTIME_DATE_TIME'] = et_to_datetime((et_start_time + et_stop_time)/2)
    ret['MIDTIME_DATE_TIME_3'] = et_to_datetime((et_start_time + et_stop_time)/2, dec=3)

    ret['ANTIBLOOMING_STATE_FLAG'] = min_label['ANTIBLOOMING_STATE_FLAG']
    ret['BIAS_STRIP_MEAN'] = min_label['BIAS_STRIP_MEAN']
    ret['CALIBRATION_LAMP_STATE_FLAG'] = min_label['CALIBRATION_LAMP_STATE_FLAG']
    ret['COMMAND_FILE_NAME'] = min_label['COMMAND_FILE_NAME']
    ret['COMMAND_SEQUENCE_NUMBER'] = min_label['COMMAND_SEQUENCE_NUMBER']
    ret['DARK_STRIP_MEAN'] = min_label['DARK_STRIP_MEAN']
    ret['DATA_CONVERSION_TYPE'] = min_label['DATA_CONVERSION_TYPE']
    ret['DELAYED_READOUT_FLAG'] = min_label['DELAYED_READOUT_FLAG']
    ret['DETECTOR_TEMPERATURE'] = str(min_label['DETECTOR_TEMPERATURE']).split(' ')[0]
    ret['EARTH_RECEIVED_START_TIME'] = julian.iso_from_tai(julian.tai_from_tdb(
        julian.tdb_from_iso(min_label['EARTH_RECEIVED_START_TIME'])),
        ymd=False, digits=3)
    ret['EARTH_RECEIVED_STOP_TIME'] = julian.iso_from_tai(julian.tai_from_tdb(
        julian.tdb_from_iso(min_label['EARTH_RECEIVED_STOP_TIME'])),
        ymd=False, digits=3)
    ret['ELECTRONICS_BIAS'] = min_label['ELECTRONICS_BIAS']
    ret['EXPECTED_MAXIMUM'] = min_label['EXPECTED_MAXIMUM']
    ret['EXPECTED_PACKETS'] = min_label['EXPECTED_PACKETS']
    ret['EXPOSURE_DURATION'] = min_label['EXPOSURE_DURATION']
    if str(min_label['FILTER_NAME'][0]) != 'CL1' or str(min_label['FILTER_NAME'][1]) != 'CL2':
        LOGGER.error(f'{obsid}: Filter name is not CL1 and CL2: {min_label["FILTER_NAME"]}')
        raise ObsIdFailedException
    ret['FILTER1'] = 'CL1'
    ret['FILTER2'] = 'CL2'
    ret['FILTER_TEMPERATURE'] = min_label['FILTER_TEMPERATURE']
    ret['FLIGHT_SOFTWARE_VERSION_ID'] = min_label['FLIGHT_SOFTWARE_VERSION_ID']
    ret['GAIN_MODE_ID'] = str(min_label['GAIN_MODE_ID']).split(' ')[0]
    ret['IMAGE_MID_TIME'] = julian.iso_from_tai(julian.tai_from_tdb(
        julian.tdb_from_iso(min_label['IMAGE_MID_TIME'])),
        ymd=False, digits=3)
    ret['IMAGE_NUMBER'] = min_label['IMAGE_NUMBER']
    ret['IMAGE_OBSERVATION_TYPE'] = str(min_label['IMAGE_OBSERVATION_TYPE']).strip("{}'")
    ret['IMAGE_TIME'] = julian.iso_from_tai(julian.tai_from_tdb(
        julian.tdb_from_iso(min_label['IMAGE_TIME'])),
        ymd=False, digits=3)
    ret['INSTRUMENT_DATA_RATE'] = min_label['INSTRUMENT_DATA_RATE']
    ret['INSTRUMENT_HOST_NAME'] = min_label['INSTRUMENT_HOST_NAME']
    ret['INSTRUMENT_ID'] = min_label['INSTRUMENT_ID']
    ret['INSTRUMENT_MODE_ID'] = min_label['INSTRUMENT_MODE_ID']
    ret['INST_CMPRS_PARAM'] = ['999' if str(x) == 'N/A' else str(x) for x in
                               min_label['INST_CMPRS_PARAM']]
    ret['INST_CMPRS_RATE'] = min_label['INST_CMPRS_RATE']
    ret['INST_CMPRS_RATIO'] = min_label['INST_CMPRS_RATIO']
    ret['INST_CMPRS_TYPE'] = min_label['INST_CMPRS_TYPE']
    ret['LIGHT_FLOOD_STATE_FLAG'] = min_label['LIGHT_FLOOD_STATE_FLAG']
    ret['METHOD_DESC'] = min_label['METHOD_DESC']
    ret['MISSING_LINES'] = -1 if min_label['MISSING_LINES'] == 'N/A' else min_label['MISSING_LINES']
    ret['MISSING_LINES_COMMENT'] = ' <!--A value of -1 indicates that the value in the original PDS3 label was N/A -->' if ret['MISSING_LINES'] == -1 else ''
    ret['MISSING_PACKET_FLAG'] = min_label['MISSING_PACKET_FLAG']
    ret['MISSION_NAME'] = min_label['MISSION_NAME']
    ret['MISSION_PHASE_NAME'] = min_label['MISSION_PHASE_NAME']
    ret['OBSERVATION_ID'] = min_label['OBSERVATION_ID']
    if ret['OBSERVATION_ID'].upper() != ret['MOSAIC_OBSERVATION_ID_ROOT'].upper():
        LOGGER.error(f'{obsid}: Observation ID {ret["OBSERVATION_ID"]} does not match mosaic observation ID root {ret["MOSAIC_OBSERVATION_ID_ROOT"]}')
        raise ObsIdFailedException
    ret['OPTICS_TEMPERATURE'] = min_label['OPTICS_TEMPERATURE']
    ret['ORDER_NUMBER'] = min_label['ORDER_NUMBER']
    ret['PARALLEL_CLOCK_VOLTAGE_INDEX'] = min_label['PARALLEL_CLOCK_VOLTAGE_INDEX']
    ret['PREPARE_CYCLE_INDEX'] = min_label['PREPARE_CYCLE_INDEX']
    ret['PRODUCT_CREATION_TIME'] = min_label['PRODUCT_CREATION_TIME']
    ret['PRODUCT_ID'] = min_label['PRODUCT_ID']
    ret['PRODUCT_VERSION_TYPE'] = min_label['PRODUCT_VERSION_TYPE']
    ret['READOUT_CYCLE_INDEX'] = min_label['READOUT_CYCLE_INDEX']
    ret['RECEIVED_PACKETS'] = min_label['RECEIVED_PACKETS']
    ret['SENSOR_HEAD_ELEC_TEMPERATURE'] = min_label['SENSOR_HEAD_ELEC_TEMPERATURE']
    ret['SEQUENCE_ID'] = min_label['SEQUENCE_ID']
    ret['SEQUENCE_NUMBER'] = min_label['SEQUENCE_NUMBER']
    ret['SEQUENCE_TITLE'] = min_label['SEQUENCE_TITLE']
    ret['SHUTTER_MODE_ID'] = min_label['SHUTTER_MODE_ID']
    ret['SHUTTER_STATE_ID'] = min_label['SHUTTER_STATE_ID']
    ret['SOFTWARE_VERSION_ID'] = min_label['SOFTWARE_VERSION_ID']
    ret['TARGET_DESC'] = min_label['TARGET_DESC']
    ret['TELEMETRY_FORMAT_ID'] = min_label['TELEMETRY_FORMAT_ID']


def xml_add_comments(ret, img_type, obsid, metadata, bkgnd_metadata):
    """Add descriptions, comments, and references to the XML metadata."""
    root_obsid = ret['MOSAIC_OBSERVATION_ID_ROOT']
    obsid_chunk = ret['MOSAIC_OBSERVATION_ID_CHUNK']

    start_date_time = ret['START_DATE_TIME']
    stop_date_time = ret['STOP_DATE_TIME']
    et_start_time = julian.tdb_from_iso(start_date_time)
    et_stop_time = julian.tdb_from_iso(stop_date_time)

    global EARLIEST_START_DATE_TIME, LATEST_STOP_DATE_TIME
    if EARLIEST_START_DATE_TIME is None:
        EARLIEST_START_DATE_TIME = et_start_time
    else:
        EARLIEST_START_DATE_TIME = min(EARLIEST_START_DATE_TIME, et_start_time)
    if LATEST_STOP_DATE_TIME is None:
        LATEST_STOP_DATE_TIME = et_stop_time
    else:
        LATEST_STOP_DATE_TIME = max(LATEST_STOP_DATE_TIME, et_stop_time)

    total_secs = et_stop_time - et_start_time

    ret['TOUR'] = et_to_tour(et_start_time)

    long_antimask = metadata['long_antimask']
    num_good_long = np.sum(long_antimask)
    corot_longitudes = metadata['longitudes'][long_antimask]
    min_corot_long, max_corot_long = wrapped_minmax(corot_longitudes)
    diff_corot = (max_corot_long - min_corot_long) % 360
    deg_good_long = num_good_long / len(long_antimask) * 360
    inertial_longitudes = metadata['inertial_longitudes'][long_antimask]
    min_inertial, max_inertial = wrapped_minmax(inertial_longitudes)
    diff_inertial = (max_inertial - min_inertial) % 360

    if img_type == 'r':
        ret['MIN_RING_COROTATING_LONG'] = f'{min_corot_long:.2f}'
        ret['MAX_RING_COROTATING_LONG'] = f'{max_corot_long:.2f}'
    else:
        # Mosaics are always written out to their full extent even if not all
        # longitudes are populated.
        ret['MIN_RING_COROTATING_LONG'] = '0.00'
        ret['MAX_RING_COROTATING_LONG'] = \
            f'{360 - arguments.longitude_resolution:.2f}'
    ret['MIN_RING_COROTATING_LONG_FIXED'] = f'{min_corot_long:6.2f}'
    ret['MAX_RING_COROTATING_LONG_FIXED'] = f'{max_corot_long:6.2f}'
    ret['MIN_RING_INERTIAL_LONG'] = f'{min_inertial:.3f}'
    ret['MAX_RING_INERTIAL_LONG'] = f'{max_inertial:.3f}'
    ret['MIN_RING_INERTIAL_LONG_FIXED'] = f'{min_inertial:7.3f}'
    ret['MAX_RING_INERTIAL_LONG_FIXED'] = f'{max_inertial:7.3f}'

    # References to browse images and other mosaic variation
    if img_type == 'b':
        # Background-subtracted mosaic
        ret['MOSAIC_OTHER_LID'] = ret['MOSAIC_ORIGINAL_LID']
        ret['MOSAIC_OTHER_REFERENCE_COMMENT'] = """
            The mosaic without the background subtracted."""
        ret['MOSAIC_REFERENCE_COMMENT'] = """
            The mosaic with the background subtracted."""
        ret['BROWSE_MOSAIC_COMMENT'] = """
            Browse images of the background-subtracted mosaic in multiple sizes
            in PNG format."""
    else:
        # Original mosaic or reprojected image
        ret['MOSAIC_OTHER_LID'] = ret['MOSAIC_BKG_SUB_LID']
        ret['MOSAIC_REFERENCE_COMMENT'] = """
            The mosaic without the background subtracted."""
        ret['MOSAIC_OTHER_REFERENCE_COMMENT'] = """
            The mosaic with the background subtracted."""
        ret['BROWSE_MOSAIC_COMMENT'] = """
            Browse images of the mosaic in multiple sizes in PNG format."""

    target_id = ''
    has_prometheus = image_has_prometheus(metadata)
    has_pandora = image_has_pandora(metadata)
    if img_type != 'r':
        if has_prometheus != mosaic_has_visual_prometheus(obsid):
            LOGGER.warning(f'{obsid}/{img_type}: Prometheus is {has_prometheus} in the image but {mosaic_has_visual_prometheus(obsid)} visually confirmed')
        if has_pandora != mosaic_has_visual_pandora(obsid):
            LOGGER.warning(f'{obsid}/{img_type}: Pandora is {has_pandora} in the image but {mosaic_has_visual_pandora(obsid)} visually confirmed')
    if has_prometheus:
        target_id += TARGET_PROMETHEUS
    if has_pandora:
        target_id += TARGET_PANDORA
    ret['TARGET_IDENTIFICATION'] = target_id

    if img_type == 'r':
        xml_add_reproj_comments(ret, metadata, root_obsid, obsid, start_date_time,
                                deg_good_long,
                                min_inertial, max_inertial, diff_inertial,
                                min_corot_long, max_corot_long, diff_corot)
    else:
        xml_add_mosaic_comments(ret, metadata, bkgnd_metadata, img_type, root_obsid,
                                obsid, start_date_time, stop_date_time, total_secs,
                                deg_good_long,
                                min_inertial, max_inertial, diff_inertial,
                                min_corot_long, max_corot_long, diff_corot)


def xml_add_reproj_comments(ret, metadata, root_obsid, obsid, start_date_time,
                            deg_good_long,
                            min_inertial, max_inertial, diff_inertial,
                            min_corot_long, max_corot_long, diff_corot):
    """Add comments to the XML metadata for a reprojected image."""
    image_name = metadata['image_name']
    ret['REPROJ_TITLE'] = f"""
Reprojected Version of Cassini ISS Calibrated Image {image_name} from
Observation {root_obsid}
"""
    ret['REPROJ_METADATA_TITLE'] = f"""
Metadata for the Reprojected Version of Cassini ISS Calibrated Image
{image_name} from Observation {root_obsid}
"""
    ret['REPROJ_LID'] = image_name_to_reproj_lid(image_name)
    ret['CALIB_LIDVID'] = image_name_to_calib_lidvid(image_name)
    ret['BROWSE_REPROJ_LID'] = image_name_to_reproj_browse_lid(image_name)

    ret['REPROJ_DESCRIPTION'] = f"""
Reprojected version of Cassini ISS calibrated image {image_name} from
Cassini observation {root_obsid}. This reprojected image was used to
create mosaic {obsid.lower()}.

This derived data product is part of bundle cassini_iss_fring_mosaics_rsfrench2025,
created by Robert S. French et al., and archived at the Ring-Moon Systems Node.
For full citation information, see collection or bundle labels.
"""

    nav_qual_str = {'G': 'good', 'F': 'fair', 'P': 'poor'}[
        OBSERVATION_INFO[obsid]['nav_qual']]

    ret['REPROJ_COMMENT'] = f"""
This data file is an individual reprojected image of Saturn's F ring from Cassini ISS
image {image_name} taken at {start_date_time}. In this image, Cassini observed an area of
space covering {diff_inertial:.3f} degrees of inertial longitude from {min_inertial:.3f}
to {max_inertial:.3f}. The source image was calibrated using CISSCAL 4.0 and the data
values are in units of I/F. The mosaics, in the data_mosaic and data_mosaic_bkg_sub
collections, were generated by stitching together reprojected, calibrated images such as
this, and this reprojected image is used in the mosaic named {obsid.lower()}.


The reprojection takes the image space and reprojects it onto a regular radius/longitude
grid, where the longitude (sampled at 0.02 degrees) is co-rotating with the core of the F
ring and the radius (sampled at 5 km) is relative to the position of the core at that
longitude and time using the model of the F ring's orbit from Albers et al. (2012), Table 3,
fit #2 (in other words, even though the F ring is eccentric, in the mosaic it looks like a
straight line at constant radius). The co-rotating longitude is calculated using the epoch
2007-01-01T00:00:00Z, meaning this was the instant when co-rotating and inertial
longitudes were the same. This reprojected image contains valid data for a total of
{deg_good_long:.2f} degrees of co-rotating longitude spanning the (possibly discontinuous)
{diff_corot:.2f} degrees from {min_corot_long:.2f} to {max_corot_long:.2f}.


Before reprojecting, the pointing specified by the available SPICE kernels was refined by
using known features in the image. In some cases, manual intervention was required. The
details of the navigation can be found in the supplemental file. The subjective quality of
the navigation for all of the images for mosaic {obsid.lower()} is "{nav_qual_str}".
"""
    if 'Prometheus' in ret['TARGET_IDENTIFICATION']:
        ret['REPROJ_COMMENT'] += """

This reprojected F-ring image includes Prometheus within the valid data range, although
its presence has not been visually confirmed.
"""
    if 'Pandora' in ret['TARGET_IDENTIFICATION']:
        ret['REPROJ_COMMENT'] += """

This reprojected F-ring image includes Pandora within the valid data range, although
its presence has not been visually confirmed.
"""


    ret['REPROJ_RINGS_DESCRIPTION'] = f"""
The parameters in this class are derived as follows:


epoch_reprojection_basis_utc is the date and time when the inertial longitude and
co-rotating longitude are the same. It is arbitrarily chosen to be a time near Cassini's
arrival at Saturn and is the same for all reprojected images.


corotation_rate is the mean corotation rate of the F ring core taken from Albers et al.
(2012), Table 3, fit #2.


The minimum, maximum, and mean values for phase angle and emission angle are computed by
looking at every longitude that contains valid data. Because the incidence angle changes
very slowly, the minimum and maximum incidence angle are set to the mean incidence angle.


The minimum and maximum co-rotating longitude are the limits that contain valid data. If
the reprojection wraps around then the minimum will be greater than the maximum.


The minimum and maximum ring radius are the actual radii (distance from Saturn) of the F
ring core -1000km and +1000km at each inertial longitude containing valid data at the time
of the observation.
"""

    ret['REPROJ_METADATA_DESCRIPTION'] = f"""
Metadata for the reprojected version of Cassini ISS calibrated image
{image_name} from observation {root_obsid}.
"""

    ret['REPROJ_METADATA_COMMENT'] = f"""
One file containing metadata parameters per valid corotating longitude for the reprojected
version of the Cassini ISS calibrated image {image_name} from {root_obsid} taken at
{start_date_time}.
"""
    ret['REPROJ_METADATA_RINGS_DESCRIPTION'] = ret['REPROJ_METADATA_DESCRIPTION']

    ret['REPROJ_IMG_FILENAME'] = f'{image_name.lower()}_reproj_img.img'
    ret['REPROJ_IMG_SUPPL_FILENAME'] = f'{image_name.lower()}_reproj_img_suppl.txt'


def xml_add_mosaic_comments(ret, metadata, bkgnd_metadata, img_type, root_obsid,
                            obsid, start_date_time, stop_date_time, total_secs,
                            deg_good_long,
                            min_inertial, max_inertial, diff_inertial,
                            min_corot_long, max_corot_long, diff_corot):
    """Add comments to the XML metadata for a mosaic."""
    full_obsid = ret['FULL_OBSERVATION_ID']
    obsid_chunk = ret['MOSAIC_OBSERVATION_ID_CHUNK']
    notes = OBSERVATION_INFO[full_obsid]['notes']

    sfx = '_bkg_sub' if img_type == 'b' else ''
    cap_bkg = 'Background-subtracted ' if img_type == 'b' else ''
    title_bkg = 'Background-Subtracted ' if img_type == 'b' else ''
    num_images = len(metadata['image_path_list'])
    min_image_name = ret['MIN_IMAGE_NAME']
    max_image_name = ret['MAX_IMAGE_NAME']
    nav_qual_str = {'G': 'good', 'F': 'fair', 'P': 'poor'}[
        OBSERVATION_INFO[full_obsid]['nav_qual']]
    bkgnd_qual_str = {'G': 'good', 'F': 'fair', 'P': 'poor'}[
        OBSERVATION_INFO[full_obsid]['bkgnd_qual']]

    total_hours = total_secs / 3600

    ret['MIN_IMAGE_NAME'] = min_image_name
    ret['MAX_IMAGE_NAME'] = max_image_name
    ret['MOSAIC_TITLE'] = f"""
{title_bkg}F Ring Mosaic Created from Reprojected, Calibrated Cassini ISS Images
from Observation {root_obsid} Spanning {min_image_name} ({start_date_time}) to
{max_image_name} ({stop_date_time})
"""
    ret['MOSAIC_METADATA_TITLE'] = f"""
Metadata for the {title_bkg}F Ring Mosaic Created from Reprojected,
Calibrated Cassini ISS Images from Observation {root_obsid} Spanning
{min_image_name} ({start_date_time}) to {max_image_name} ({stop_date_time})
"""

    ret['MOSAIC_DESCRIPTION'] = f"""
{cap_bkg}F Ring mosaic created from reprojected, calibrated Cassini ISS images
from observation {root_obsid} spanning {min_image_name} ({start_date_time}) to
{max_image_name} ({stop_date_time}).

This derived data product is part of bundle cassini_iss_fring_mosaics_rsfrench2025,
created by Robert S. French et al., and archived at the Ring-Moon Systems Node.
For full citation information, see collection or bundle labels.
"""

    additional_notes = []

    partial_comment = ''
    if obsid_chunk is not None:
        if 'M1' in notes:
            partial_comment = f"""

Because Cassini observed the same inertial longitudes for more than one orbit of the
F ring during {root_obsid}, we have split the observation into multiple chunks, each
corresponding to one complete orbit of the F ring. This mosaic consists of {root_obsid}
chunk {obsid_chunk}. Other mosaics are available in this bundle for {root_obsid} with
different date ranges representing the other available observation chunks.
"""
        elif 'M2' in notes:
            partial_comment = f"""

Because observation {root_obsid} consists of two distinct "movies" consisting of
approximately the same co-rotating longitudes but taken at inertial longitudes roughly 180
degrees apart, we have split the observation into two chunks. This mosaic consists of
{root_obsid} chunk {obsid_chunk}. The other mosaic is available as
{root_obsid.lower()}_{3-int(obsid_chunk)}.
"""
        elif 'M3' in notes:
            partial_comment = f"""

Because observation {root_obsid} consists of multiple "movies" consisting of approximately
the same co-rotating longitudes but taken at different inertial longitudes (not 180
degrees apart), we have split the observation into multiple chunks. This mosaic consists
of {root_obsid} chunk {obsid_chunk}. Other mosaics are available in this bundle for
{root_obsid} with different date ranges representing the other available observation
chunks.
"""
        elif 'M4' in notes:
            partial_comment = f"""

Because Cassini observed multiple distinct inertial and co-rotating longitudes during
{root_obsid}, each making its own "movie", we have split the observation into multiple
chunks. This mosaic consists of {root_obsid} chunk {obsid_chunk}. Other mosaics are
available in this bundle for {root_obsid} with different date ranges representing the
other available observation chunks.
"""
        else:
            LOGGER.error(f'{full_obsid}: Multi-chunk observation is missing "M" note')
            raise ObsIdFailedException

    bkg_comment = ''
    if img_type == 'b':
        # The background limits are stored as mosaic row numbers, so each row
        # spans one radial resolution element.
        radial_res = arguments.radius_resolution
        lower_limit = int(-arguments.radius_inner_delta -
                          bkgnd_metadata['ring_lower_limit']*radial_res)
        ret['BKGND_LOWER_LIMIT'] = lower_limit
        num_limit_rows = int((arguments.radius_outer_delta -
                              arguments.radius_inner_delta) // radial_res)
        upper_limit = int(arguments.radius_outer_delta -
                          (num_limit_rows-bkgnd_metadata['ring_upper_limit'])*radial_res)
        ret['BKGND_UPPER_LIMIT'] = upper_limit
        bkg_comment = f"""


Background subtraction was performed by creating, for each longitude, a linear model based
on the available data from {lower_limit} to 1000 km closer to Saturn and {upper_limit} to
1000 km further from Saturn. Statistically bad pixels (such as stars or moons) were
ignored. If insufficient data was available to generate the model, that longitude was
marked as invalid and removed from the mosaic. As such, the number of longitudes available
in the background-subtracted mosaic may be fewer than those available in the original
mosaic."""
        if lower_limit != 750 or upper_limit != 750:
            bkg_comment += f"""


Note that the background limits for this mosaic are non-standard and were manually chosen
because the standard values of 750-1000 km did not work for reasons such as insufficient
data, bad data, or encroachment of the F ring dust sheet into the background area due to
low-resolution source images. All efforts were made to preserve the photometric
consistency of the background-subtracted mosaic with other mosaics using the "standard"
parameters and we do not expect the use of non-standard parameters to change the resulting
data values by more than a few percent."""

        bkg_comment += f"""

The subjective quality of the background modeling and subtraction process for this
mosaic is "{bkgnd_qual_str}"."""

        if 'B' in notes:
            additional_notes.append("""This background-subtracted mosaic contains
substantially fewer valid longitudes than the original mosaic due to insufficient data
being available to create a background model.""")

    if 'C' in notes:
        additional_notes.append("""Some source images contained corrupted or missing data
that could not be repaired during the construction of the mosaic.""")
    if 'E' in notes:
        additional_notes.append("""Some source images may have been overexposed and the
data values clipped; use caution when using this mosaic for photometry.""")
    if 'O' in notes:
        additional_notes.append("""The sequence of source images used to create this
mosaic were designed to observe a stellar occultation of the F ring core. As such, a star
is present in each source image and may appear in the mosaic multiple times depending on
how the reprojected images were stitched together. In addition, the source images were
taken at roughly the same co-rotating longitudes and thus have significant overlap in the
mosaic. To fully explore the occultation, use the reprojected images.""")


    if 'R' in notes:
        ret['MOSAIC_COMMENT'] = f"""
This data file is a {cap_bkg.lower()}mosaic of Saturn's F ring, stitched together from
reprojections of {num_images} source images from Cassini Observation Name {root_obsid}
spanning {min_image_name} ({start_date_time}) to {max_image_name} ({stop_date_time}).
During this time, Cassini followed one co-rotating longitude for {total_secs:,.0f} seconds
({total_hours:.5f} hours) by observing multiple inertial longitudes covering the (possibly
discontinuous) {diff_inertial:.3f} degrees from {min_inertial:.3f} to {max_inertial:.3f}.
The source images were calibrated using CISSCAL 4.0 and the data values are in units of
I/F.
"""
    elif 'N' in notes:
        ret['MOSAIC_COMMENT'] = f"""
This data file is a {cap_bkg.lower()}mosaic of Saturn's F ring, stitched together from
reprojections of {num_images} source images from Cassini Observation Name {root_obsid}
spanning {min_image_name} ({start_date_time}) to {max_image_name} ({stop_date_time}).
During this time, Cassini observed multiple co-rotating longitudes at multiple inertial
longitudes for {total_secs:,.0f} seconds ({total_hours:.5f} hours). The inertial
longitudes covered the (possibly discontinuous) {diff_inertial:.3f} degrees from
{min_inertial:.3f} to {max_inertial:.3f}. The source images were calibrated using CISSCAL
4.0 and the data values are in units of I/F.
"""
    else:
        ret['MOSAIC_COMMENT'] = f"""
This data file is a {cap_bkg.lower()}mosaic of Saturn's F ring, stitched
together from reprojections of {num_images} source images from Cassini
Observation Name {root_obsid} spanning {min_image_name} ({start_date_time}) to
{max_image_name} ({stop_date_time}). During this time, Cassini repeatedly observed an
area of space covering {diff_inertial:.3f} degrees of inertial longitude from
{min_inertial:.3f} to {max_inertial:.3f} while the ring rotated under it for
{total_secs:,.0f} seconds ({total_hours:.5f} hours).
"""

    ret['MOSAIC_COMMENT'] += partial_comment

    ret['MOSAIC_COMMENT'] += f"""

The reprojection takes the image space and reprojects it onto a regular radius/longitude
grid, where the longitude (sampled at 0.02 degrees) is co-rotating with the core of the F
ring and the radius (sampled at 5 km) is relative to the position of the core at that
longitude and time using the model of the F ring's orbit from Albers et al. (2012), Table 3,
fit #2 (in other words, even though the F ring is eccentric, in the mosaic it looks like a
straight line at constant radius). The co-rotating longitude is calculated using the epoch
2007-01-01T00:00:00Z, meaning this was the instant when co-rotating and inertial
longitudes were the same. This mosaic image contains valid data for a total of
{deg_good_long:.2f} degrees of co-rotating longitude spanning the (possibly discontinuous)
{diff_corot:.2f} degrees from {min_corot_long:.2f} to {max_corot_long:.2f}. The source
images were calibrated using CISSCAL 4.0 and the data values are in units of
I/F.

Before reprojecting, the pointing specified by the available SPICE kernels was refined by
using known features in the image. In some cases, manual intervention was required. The
details of the navigation can be found in the supplemental file for each source reprojected
image. The subjective quality of the navigation for all of the images for this mosaic
is "{nav_qual_str}".{bkg_comment}
"""

    if additional_notes:
        ret['MOSAIC_COMMENT'] += '\n\nNotes:'
        for additional_note in additional_notes:
            ret['MOSAIC_COMMENT'] += f'\n\n- {additional_note}'

    ret['MOSAIC_RINGS_DESCRIPTION'] = f"""
The parameters in this class are derived as follows:

- epoch_reprojection_basis_utc is the date and time when the inertial longitude and
co-rotating longitude are the same. It is arbitrarily chosen to be a time near Cassini's
arrival at Saturn and is the same for all reprojected images.

- corotation_rate is the mean corotation rate of the F ring core taken from Albers et al.
(2012), Table 3, fit #2.

- The minimum, maximum, and mean values for phase angle and emission angle are computed
by looking at every longitude that contains valid data. Because the incidence angle
changes very slowly, the minimum and maximum incidence angle are set to the mean
incidence angle.

- The minimum and maximum co-rotating longitude always span the full extent of the
mosaic, even if not all longitudes contain valid data.

- The minimum and maximum ring radius are the actual radii (distance from Saturn) of the F
ring core -1000km and +1000km at each inertial longitude containing valid data at the time
of the observation.
"""
    if 'Prometheus' in ret['TARGET_IDENTIFICATION']:
        ret['MOSAIC_RINGS_DESCRIPTION'] += """

This mosaic includes Prometheus within the valid data range, although its presence has not
been visually confirmed.
"""
    if 'Pandora' in ret['TARGET_IDENTIFICATION']:
        ret['MOSAIC_RINGS_DESCRIPTION'] += """

This mosaic includes Pandora within the valid data range, although its presence has not
been visually confirmed.
"""

    ret['MOSAIC_IMG_FILENAME'] = f'{obsid.lower()}_mosaic{sfx}.img'


def generate_image(obsid, output_dir, metadata, xml_metadata, global_index_fp,
                   img_type):
    """Create mosaic/reproj images and labels and mosaic metadata tables and labels.

    Inputs:
        obsid               The observation name.
        output_dir          The directory in which to put all output files.
        metadata            The metadata for a mosaic, background-subtracted
                            mosaic, or reprojected image.
        xml_metadata        The XML substitutions.
        global_index_fp     The file pointer for the global index.
        img_type            The img_type of data being provided:
                                'm' = Mosaic
                                'b' = Background-subtracted mosaic
                                'r' = Reprojected image

    The global flags are used to determine which output files to create:

    img_type = 'm':

      data_mosaic/
        OBSID/
          OBSID_mosaic.img                            [GENERATE_MOSAIC_IMAGES]
          OBSID_mosaic.lblx                           [GENERATE_MOSAIC_IMAGE_LABELS]
          OBSID_mosaic_metadata_src_imgs.tab          [GENERATE_MOSAIC_METADATA_TABLES]
          OBSID_mosaic_metadata_params.tab            [GENERATE_MOSAIC_METADATA_TABLES]

    img_type = 'b':

      data_mosaic_bkg_sub/
        OBSID/
          OBSID_mosaic_bkg_sub.img                    [GENERATE_MOSAIC_IMAGES]
          OBSID_mosaic_bkg_sub.lblx                   [GENERATE_MOSAIC_IMAGE_LABELS]
          OBSID_mosaic_bkg_sub_metadata_src_imgs.tab  [GENERATE_MOSAIC_METADATA_TABLES]
          OBSID_mosaic_bkg_sub_metadata_params.tab    [GENERATE_MOSAIC_METADATA_TABLES]

    img_type = 'r':

      data_reproj_img/
        OBSID/
          IMG_reproj_img.img                          [GENERATE_REPROJ_IMAGES]
          IMG_reproj_img.lblx                         [GENERATE_REPROJ_IMAGE_LABELS]
          IMG_reproj_img_metadata_params.tab          [GENERATE_REPROJ_METADATA_TABLES]
          IMG_reproj_img_suppl.txt                    [GENERATE_REPROJ_SUPPL_FILES]
    """
    os.makedirs(output_dir, exist_ok=True)

    if img_type == 'r':
        image_name = metadata['image_name']
    else:
        sfx = '_bkg_sub' if img_type == 'b' else ''

    long_antimask = metadata['long_antimask']
    longitudes = metadata['longitudes'][long_antimask]
    if img_type == 'r':
        incidence = np.degrees(metadata['incidence'])
        emission_angles = np.degrees(metadata['mean_emission'])
        phase_angles = np.degrees(metadata['mean_phase'])
        rad_resolutions = metadata['mean_radial_resolution']
        ang_resolutions = np.degrees(metadata['mean_angular_resolution'])
    else:
        incidence = np.degrees(metadata['mean_incidence'])
        emission_angles = np.degrees(metadata['mean_emission'][long_antimask])
        phase_angles = np.degrees(metadata['mean_phase'][long_antimask])
        rad_resolutions = metadata['mean_radial_resolution'][long_antimask]
        ang_resolutions = np.degrees(metadata['mean_angular_resolution'][long_antimask])
    inertial_longitudes = metadata['inertial_longitudes'][long_antimask]

    if img_type == 'r':
        pass
    else:
        ETs = metadata['time'][long_antimask]
        image_indexes = metadata['image_number'][long_antimask]
        image_name_list = metadata['image_name_list']


            ###############################
            ###     METADATA_TABLES     ###
            ###############################

    if img_type == 'r':
        params_filename = f'{image_name.lower()}_reproj_img_metadata_params.tab'
        xml_metadata['METADATA_PARAMS_TABLE_FILENAME'] = params_filename
        metadata_params_table_path = os.path.join(output_dir, params_filename)
        xml_metadata['METADATA_PARAMS_TABLE_PATH'] = metadata_params_table_path
    else:
        params_filename = f'{obsid.lower()}_mosaic{sfx}_metadata_params.tab'
        xml_metadata['METADATA_PARAMS_TABLE_FILENAME'] = params_filename
        metadata_params_table_path = os.path.join(output_dir, params_filename)
        xml_metadata['METADATA_PARAMS_TABLE_PATH'] = metadata_params_table_path
        src_imgs_filename = f'{obsid.lower()}_mosaic{sfx}_metadata_src_imgs.tab'
        xml_metadata['IMAGE_TABLE_FILENAME'] = src_imgs_filename
        image_table_path = os.path.join(output_dir, src_imgs_filename)
        xml_metadata['IMAGE_TABLE_PATH'] = image_table_path

    if ((img_type == 'r' and GENERATE_REPROJ_METADATA_TABLES) or
        (img_type != 'r' and GENERATE_MOSAIC_METADATA_TABLES)):
        # OBSID_mosaic_metadata_params.tab or
        # IMG_reproj_img_metadata_params.tab
        if img_type == 'r':
            LOGGER.info(f'Writing metadata table for reprojected image {image_name} to {metadata_params_table_path}')
        else:
            LOGGER.info(f'Writing metadata table for mosaic {obsid} to {metadata_params_table_path}')
        with open(metadata_params_table_path, 'w') as fp:
            if img_type == 'r':
                fp.write('rings:corotating_ring_longitude,'
                         'rings:observed_event_tdb,'
                         'rings:inertial_ring_longitude,'
                         'rings:radial_resolution,'
                         'rings:longitudinal_resolution,'
                         'rings:incidence_angle,'
                         'rings:phase_angle,'
                         'rings:emission_angle,'
                         'core_radius,'
                         'longitude_ascending_node,'
                         'longitude_pericenter,'
                         'true_anomaly,'
                         'corotating_longitude_prometheus,'
                         'radius_prometheus,'
                         'corotating_longitude_pandora,'
                         'radius_pandora'
                         '\n')
            else:
                fp.write('rings:corotating_ring_longitude,'
                         'image_index,'
                         'rings:observed_event_tdb,'
                         'rings:inertial_ring_longitude,'
                         'rings:radial_resolution,'
                         'rings:longitudinal_resolution,'
                         'rings:incidence_angle,'
                         'rings:phase_angle,'
                         'rings:emission_angle,'
                         'core_radius,'
                         'longitude_ascending_node,'
                         'longitude_pericenter,'
                         'true_anomaly,'
                         'corotating_longitude_prometheus,'
                         'radius_prometheus,'
                         'corotating_longitude_pandora,'
                         'radius_pandora'
                         '\n')
            for idx in range(len(longitudes)):
                longitude = longitudes[idx]
                inertial = inertial_longitudes[idx]
                rad_resolution = rad_resolutions[idx]
                ang_resolution = ang_resolutions[idx]
                phase = phase_angles[idx]
                emission = emission_angles[idx]
                if img_type == 'r':
                    et = xml_metadata['MIDTIME_ET']
                else:
                    et = ETs[idx]
                core_radius = fring_radius_at_longitude(inertial, et)
                long_asc = fring_longitude_of_ascending_node(et)
                long_peri = fring_longitude_of_pericenter(et)
                true_anomaly = fring_true_anomaly(inertial, et)
                prometheus_dist, prometheus_corot_long = saturn_to_prometheus_corot(et)
                pandora_dist, pandora_corot_long = saturn_to_pandora_corot(et)

                row = f'{longitude:6.2f}, '
                if img_type != 'r':
                    image_idx = image_indexes[idx]
                    row += f'{image_idx:4d}, '
                row += f'{et:13.3f}, {inertial:7.3f}, '
                row += (f'{rad_resolution:8.3f}, '
                        f'{ang_resolution:8.5f}, '
                        f'{incidence:7.3f}, {phase:7.3f}, {emission:7.3f}, ')
                row += (f'{core_radius:10.3f}, {long_asc:7.3f}, {long_peri:7.3f}, '
                        f'{true_anomaly:7.3f}, '
                        f'{prometheus_corot_long:7.3f}, {prometheus_dist:10.3f}, '
                        f'{pandora_corot_long:7.3f}, {pandora_dist:10.3f}')
                fp.write(row+'\n')

        if img_type != 'r':
            # mosaic_metadata_src_imgs.tab
            LOGGER.info(f'Writing image list for mosaic {obsid} to {image_table_path}')
            with open(image_table_path, 'w') as fp:
                fp.write('image_index,LIDVID\n')
                for idx in range(len(image_name_list)):
                    lidvid = image_name_to_reproj_lidvid(image_name_list[idx])
                    row = f'{idx:4d}, {lidvid}'
                    fp.write(row+'\n')


            ###############################
            ###  MOSAIC/REPROJ_IMAGES   ###
            ###############################

    img = ma.filled(metadata['img'], SENTINEL).astype('float32')
    if img_type == 'r':
        min_corot_long = float(xml_metadata['MIN_RING_COROTATING_LONG'])
        max_corot_long = float(xml_metadata['MAX_RING_COROTATING_LONG'])
        total_long = (max_corot_long - min_corot_long) % 360
        total_long_idx = int(np.round(total_long / arguments.longitude_resolution)) + 1
        img = reslice_reproj_img(img, min_corot_long, max_corot_long)
        if total_long_idx != img.shape[1]:
            LOGGER.error(
                f'{obsid}/{image_name}: Total longitude index {total_long_idx} does not '
                f'match image shape {img.shape[1]}')
        image_output_path = os.path.join(output_dir, xml_metadata['REPROJ_IMG_FILENAME'])
        label_output_path = os.path.join(output_dir,
                                         f'{image_name.lower()}_reproj_img.lblx')
    else:
        image_output_path = os.path.join(output_dir, xml_metadata['MOSAIC_IMG_FILENAME'])
        label_output_path = os.path.join(output_dir,
                                         f'{obsid.lower()}_mosaic{sfx}.lblx')
    xml_metadata['IMG_PATH'] = image_output_path
    xml_metadata['IMG_NUM_SAMPLES'] = str(img.shape[1])
    xml_metadata['IMG_NUM_LINES'] = str(img.shape[0])

    if ((img_type == 'r' and GENERATE_REPROJ_IMAGES) or
        (img_type != 'r' and GENERATE_MOSAIC_IMAGES)):
        img.tofile(image_output_path)


            ###############################
            ###  REPROJ IMG SUPPL FILE  ###
            ###############################

    if img_type == 'r':
        suppl_output_path = os.path.join(output_dir, xml_metadata['REPROJ_IMG_SUPPL_FILENAME'])
        xml_metadata['SUPPL_PATH'] = suppl_output_path
        if GENERATE_REPROJ_SUPPL_FILES:
            hdr_length = write_suppl_file(suppl_output_path, metadata, xml_metadata)
            xml_metadata['SUPPL_HEADER_LENGTH'] = hdr_length
        elif os.path.exists(suppl_output_path):
            # The C-matrix table is exactly 3 records of 49 bytes at the end of
            # the file; everything before it is the header
            xml_metadata['SUPPL_HEADER_LENGTH'] = (
                os.path.getsize(suppl_output_path) - 3*49)
        elif GENERATE_REPROJ_IMAGE_LABELS:
            LOGGER.error(f'{obsid}/{image_name}: Cannot generate reproj label '
                         f'because supplemental file {suppl_output_path} does '
                         f'not exist; generate the supplemental files first')
            raise ObsIdFailedException


            ###############################
            ###  MOSAIC/REPROJ_LABELS   ###
            ###############################

    if ((img_type == 'r' and GENERATE_REPROJ_IMAGE_LABELS) or
        (img_type != 'r' and GENERATE_MOSAIC_IMAGE_LABELS)):
        if img_type == 'r':
            populate_template('data_reproj_img.lblx', label_output_path, xml_metadata)
        else:
            populate_template('data_mosaic.lblx', label_output_path, xml_metadata)


            ####################################
            ###  MOSAIC/REPROJ_GLOBAL_INDEX  ###
            ####################################

    if GENERATE_MOSAIC_GLOBAL_INDEX or GENERATE_REPROJ_GLOBAL_INDEX:
        orig_obsid = xml_metadata['MOSAIC_OBSERVATION_ID_ROOT'].upper()
        mean_incidence = xml_metadata['MEAN_INCIDENCE_ANGLE_FIXED']
        mean_emission = xml_metadata['MEAN_EMISSION_ANGLE_FIXED']
        min_emission = xml_metadata['MIN_EMISSION_ANGLE_FIXED']
        max_emission = xml_metadata['MAX_EMISSION_ANGLE_FIXED']
        mean_phase = xml_metadata['MEAN_PHASE_ANGLE_FIXED']
        min_phase = xml_metadata['MIN_PHASE_ANGLE_FIXED']
        max_phase = xml_metadata['MAX_PHASE_ANGLE_FIXED']
        mean_reproj_grid_rad_res = xml_metadata['MEAN_REPROJ_GRID_RAD_RES_FIXED']
        min_reproj_grid_rad_res = xml_metadata['MIN_REPROJ_GRID_RAD_RES_FIXED']
        max_reproj_grid_rad_res = xml_metadata['MAX_REPROJ_GRID_RAD_RES_FIXED']
        mean_reproj_grid_ang_res = xml_metadata['MEAN_REPROJ_GRID_ANG_RES_FIXED']
        min_reproj_grid_ang_res = xml_metadata['MIN_REPROJ_GRID_ANG_RES_FIXED']
        max_reproj_grid_ang_res = xml_metadata['MAX_REPROJ_GRID_ANG_RES_FIXED']
        min_radius = xml_metadata['MIN_RING_RADIUS_FIXED']
        max_radius = xml_metadata['MAX_RING_RADIUS_FIXED']
        num_valid_longitudes = xml_metadata['NUM_VALID_LONGITUDES']
        min_corotating_longitude = xml_metadata['MIN_RING_COROTATING_LONG_FIXED']
        max_corotating_longitude = xml_metadata['MAX_RING_COROTATING_LONG_FIXED']
        min_inertial_longitude = xml_metadata['MIN_RING_INERTIAL_LONG_FIXED']
        max_inertial_longitude = xml_metadata['MAX_RING_INERTIAL_LONG_FIXED']
        filespec = '/'.join(label_output_path.split('/')[-3:])
        start_date = xml_metadata['START_DATE_TIME']
        stop_date = xml_metadata['STOP_DATE_TIME']
        current_date_time = xml_metadata['CURRENT_DATE_TIME']
        sclk_start = xml_metadata['SPACECRAFT_CLOCK_START_COUNT']
        sclk_stop = xml_metadata['SPACECRAFT_CLOCK_STOP_COUNT']
        notes = OBSERVATION_INFO[obsid]['notes']
        mean_core_radius = xml_metadata['MEAN_CORE_RADIUS_FIXED']
        min_core_radius = xml_metadata['MIN_CORE_RADIUS_FIXED']
        max_core_radius = xml_metadata['MAX_CORE_RADIUS_FIXED']
        long_asc = metadata['long_asc']
        long_peri = metadata['long_peri']
        true_anomaly = metadata['true_anomaly']
        min_true_anomaly, max_true_anomaly = wrapped_minmax(true_anomaly)
        prometheus_corot_long = metadata['prometheus_corot_long']
        prometheus_dist = metadata['prometheus_dist']
        pandora_corot_long = metadata['pandora_corot_long']
        pandora_dist = metadata['pandora_dist']
        nav_qual_str = OBSERVATION_INFO[obsid]['nav_qual']
        bkgnd_qual_str = OBSERVATION_INFO[obsid]['bkgnd_qual']
        perc_valid_longitudes = (num_valid_longitudes / len(long_antimask)) * 100

        if img_type == 'r':
            lid = xml_metadata['REPROJ_LID']
        else:
            lid = xml_metadata['MOSAIC_LID']
        row = (f'{lid:117},'
               f'{orig_obsid:29},'
               f'{filespec:101},'
               f'{start_date},'
               f'{stop_date},'
               f'{sclk_start},'
               f'{sclk_stop},'
               f'{num_valid_longitudes:5d},'
               f'{perc_valid_longitudes:7.3f},'
               f'{min_corotating_longitude},'
               f'{max_corotating_longitude},'
               f'{min_inertial_longitude},'
               f'{max_inertial_longitude},'
               f'{mean_phase},'
               f'{min_phase},'
               f'{max_phase},'
               f'{mean_incidence},'
               f'{mean_emission},'
               f'{min_emission},'
               f'{max_emission},'
               f'{mean_reproj_grid_rad_res},'
               f'{min_reproj_grid_rad_res},'
               f'{max_reproj_grid_rad_res},'
               f'{mean_reproj_grid_ang_res},'
               f'{min_reproj_grid_ang_res},'
               f'{max_reproj_grid_ang_res},'
               f'{min_radius},'
               f'{max_radius},'
               f'{mean_core_radius},'
               f'{min_core_radius},'
               f'{max_core_radius},')

        if img_type == 'r':
            row += (f'{long_asc:7.3f},'
                    f'{long_peri:7.3f},'
                    f'{min_true_anomaly:7.3f},'
                    f'{max_true_anomaly:7.3f},'
                    f'{prometheus_corot_long:7.3f},'
                    f'{prometheus_dist:10.3f},'
                    f'{pandora_corot_long:7.3f},'
                    f'{pandora_dist:10.3f},')
        else:
            mean_long_asc = wrapped_mean(long_asc)
            min_long_asc, max_long_asc = wrapped_minmax(long_asc)
            mean_long_peri = wrapped_mean(long_peri)
            min_long_peri, max_long_peri = wrapped_minmax(long_peri)
            mean_prometheus_corot_long = wrapped_mean(prometheus_corot_long)
            min_prometheus_corot_long, max_prometheus_corot_long = wrapped_minmax(prometheus_corot_long)
            mean_prometheus_dist = np.mean(prometheus_dist)
            min_prometheus_dist = np.min(prometheus_dist)
            max_prometheus_dist = np.max(prometheus_dist)
            mean_pandora_corot_long = wrapped_mean(pandora_corot_long)
            min_pandora_corot_long, max_pandora_corot_long = wrapped_minmax(pandora_corot_long)
            mean_pandora_dist = np.mean(pandora_dist)
            min_pandora_dist = np.min(pandora_dist)
            max_pandora_dist = np.max(pandora_dist)
            row += (f'{mean_long_asc:7.3f},'
                    f'{min_long_asc:7.3f},'
                    f'{max_long_asc:7.3f},'
                    f'{mean_long_peri:7.3f},'
                    f'{min_long_peri:7.3f},'
                    f'{max_long_peri:7.3f},'
                    f'{min_true_anomaly:7.3f},'
                    f'{max_true_anomaly:7.3f},'
                    f'{mean_prometheus_corot_long:7.3f},'
                    f'{min_prometheus_corot_long:7.3f},'
                    f'{max_prometheus_corot_long:7.3f},'
                    f'{mean_prometheus_dist:10.3f},'
                    f'{min_prometheus_dist:10.3f},'
                    f'{max_prometheus_dist:10.3f},'
                    f'{mean_pandora_corot_long:7.3f},'
                    f'{min_pandora_corot_long:7.3f},'
                    f'{max_pandora_corot_long:7.3f},'
                    f'{mean_pandora_dist:10.3f},'
                    f'{min_pandora_dist:10.3f},'
                    f'{max_pandora_dist:10.3f},')

        row += (f'{current_date_time},'
                f'{nav_qual_str:1},')

        if img_type == 'r' and GENERATE_REPROJ_GLOBAL_INDEX:
            row += (f'{notes:4}')
            global_index_fp.write(row+'\n')

        elif img_type in 'bm' and GENERATE_MOSAIC_GLOBAL_INDEX:
            min_image_name = xml_metadata['MIN_IMAGE_NAME']
            max_image_name = xml_metadata['MAX_IMAGE_NAME']
            num_images = xml_metadata['NUM_IMAGES']
            if img_type == 'm':
                row += (
                    f'{notes:4},'
                    f'{num_images:4d},'
                    f'{min_image_name:11},'
                    f'{max_image_name:11}')

            else:
                lower_limit = -xml_metadata['BKGND_LOWER_LIMIT']
                upper_limit = xml_metadata['BKGND_UPPER_LIMIT']
                row += (f'{bkgnd_qual_str:1},'
                        f'{notes:4},'
                        f'{num_images:4d},'
                        f'{min_image_name:11},'
                        f'{max_image_name:11},'
                        f'{lower_limit:5d},'
                        f'{upper_limit:4d}')
            global_index_fp.write(row+'\n')


def generate_browse(obsid, browse_dir, metadata, xml_metadata, img_type):
    """Create mosaic browse images.

    Inputs:
        obsid           The observation name.
        browse_dir      The directory in which to put all browse files.
        metadata        The metadata for a background-subtracted mosaic.
        xml_metadata    The XML substitutions.
        img_type        The img_type of data being provided:
                            'm' = Mosaic
                            'b' = Background-subtracted mosaic
                            'r' = Reprojected image

    The global flags like GENERATE_BROWSE_MOSAIC_IMAGES are used to determine
    which output files to create:

    img_type == 'm':

      browse_mosaic/
        OBSID/
          OBSID_browse_mosaic_full.png
          OBSID_browse_mosaic_med.png
          OBSID_browse_mosaic_small.png
          OBSID_browse_mosaic_thumb.png
          OBSID_browse_mosaic.lblx

    img_type == 'b':

      browse_mosaic_bkg_sub/
        OBSID/
          OBSID_browse_mosaic_bkg_sub_full.png
          OBSID_browse_mosaic_bkg_sub_med.png
          OBSID_browse_mosaic_bkg_sub_small.png
          OBSID_browse_mosaic_bkg_sub_thumb.png
          OBSID_browse_mosaic_bkg_sub.lblx

    img_type == 'r':

      browse_mosaic/
        OBSID/
          IMG_browse_reproj_img_full.png
          IMG_browse_reproj_img_med.png
          IMG_browse_reproj_img_small.png
          IMG_browse_reproj_img_thumb.png
          IMG_browse_reproj_img.lblx
    """
    os.makedirs(browse_dir, exist_ok=True)

    match = re.search(r'^(.*)_(\d+)$', obsid)
    partial_obsid = bool(match)
    root_obsid = obsid
    if partial_obsid:
        root_obsid = match[1]
        obsid_chunk = match[2]
    obsid_lc = obsid.lower()
    obsid_split = obsid_lc.split('_')
    obsid_partial_lc = obsid_lc
    if len(obsid_split) == 5:
        obsid_partial_lc = f'{obsid_split[1]}_{obsid_split[2]}/{obsid_split[4]}'
    else:
        obsid_partial_lc = f'{obsid_split[1]}_{obsid_split[2]}'

    cap_bkg = 'Background-subtracted ' if img_type == 'b' else ''
    title_bkg = 'Background-Subtracted ' if img_type == 'b' else ''

    # The browse sizes below are hardcoded for the standard mosaic geometry.
    # A data set with a different geometry has to be handled explicitly.
    num_rad, num_long = metadata['img'].shape
    assert num_rad == 401, f'Unexpected number of radial rows: {num_rad}'
    if img_type != 'r':
        assert num_long == 18000, f'Unexpected number of longitudes: {num_long}'
        sfx = '_bkg_sub' if img_type == 'b' else ''
        sizes = (('full',  401, 18000),
                 ('med',   400,  1800),
                 ('small', 200,   200),
                 ('thumb', 100,   100))
    else:
        image_name = metadata['image_name']
        sizes = (('full',  401,  None),
                 ('med',   400,  None),
                 ('small', 200,   200),
                 ('thumb', 100,   100))


            ###############################
            ###      BROWSE_IMAGES      ###
            ###############################

    if ((img_type == 'r' and GENERATE_BROWSE_REPROJ_IMAGES) or
        (img_type != 'r' and GENERATE_BROWSE_MOSAIC_IMAGES)):
        img = ma.filled(metadata['img'], SENTINEL)
        valid_antimask = img != SENTINEL
        valid_cols = np.any(valid_antimask, axis=0)
        if not np.any(valid_cols):
            if img_type == 'r':
                LOGGER.error(f'No valid columns in reprojected image {image_name}')
            else:
                LOGGER.error(f'No valid columns in mosaic {obsid}')
            raise ObsIdFailedException
        subimg = img[:, valid_cols].copy()  # Make contiguous
        valid_pixels = subimg[valid_antimask[:, valid_cols]]
        blackpoint = max(np.min(valid_pixels), 0)
        whitepoint_ignore_frac = 0.998
        img_sorted = sorted(list(valid_pixels.flatten()))
        whitepoint = img_sorted[np.clip(int(len(img_sorted)*
                                            whitepoint_ignore_frac),
                                        0, len(img_sorted)-1)]
        gamma = 0.5
        if whitepoint < blackpoint:
            whitepoint = blackpoint
        if whitepoint == blackpoint:
            whitepoint += 0.00001

        src_img = img  # Mosaics we display all 18000 longitudes
        if img_type == 'r':  # Reprojected images we display only the valid longitudes
            src_img = subimg

        for size, height, width in sizes:
            if width is None:
                if size == 'full':
                    width = max(src_img.shape[1], 800)
                elif size == 'med':
                    width = max(src_img.shape[1] // 10, 400)
            greyscale_img = np.floor((np.maximum(src_img-blackpoint, 0)/
                                     (whitepoint-blackpoint))**gamma*256)
            greyscale_img = np.clip(greyscale_img, 0, 255)
            scaled_img = np.asarray(greyscale_img[::-1,:], dtype=np.uint8)
            pil_img = Image.frombuffer('L', (scaled_img.shape[1],
                                             scaled_img.shape[0]),
                                       scaled_img, 'raw', 'L', 0, 1)

            pil_img = pil_img.resize((width, height))
            font = TITLE_FONTS[(img_type, size)]
            if font is not None:
                if size in ('thumb', 'small'):
                    if img_type == 'b':
                        title = f'{obsid_partial_lc}\nbkgnd-sub mosaic'
                    elif img_type == 'm':
                        title = f'{obsid_partial_lc}\nmosaic'
                    else:
                        title = f'{image_name.lower()}\nreproj img'
                    corner = (5, 3)
                else:
                    if img_type == 'b':
                        title = f'{obsid_lc}\nbkgnd-sub mosaic'
                    elif img_type == 'm':
                        title = f'{obsid_lc}\nmosaic'
                    else:
                        title = f'{obsid_lc} / {image_name.lower()}\nreproj img'
                    corner = (5, 5)
                draw = ImageDraw.Draw(pil_img)
                draw.text(corner, title, fill=255, font=TITLE_FONTS[(img_type, size)])

            if img_type == 'r':
                png_path = os.path.join(browse_dir,
                            f'{image_name.lower()}_browse_reproj_img_{size}.png')
                LOGGER.info(f'Writing browse image for reprojected image {image_name} '
                            f'to {png_path}')
            else:
                png_path = os.path.join(browse_dir,
                            f'{obsid.lower()}_browse_mosaic{sfx}_{size}.png')
                LOGGER.info(f'Writing browse image for mosaic {obsid} to {png_path}')

            pil_img.save(png_path, 'PNG')


            ###############################
            ###      BROWSE_LABELS      ###
            ###############################

    start_date = xml_metadata['START_DATE_TIME']
    stop_date = xml_metadata['STOP_DATE_TIME']

    if img_type == 'r':
        xml_metadata['BROWSE_REPROJ_LID'] = image_name_to_reproj_browse_lid(image_name)
        xml_metadata['BROWSE_REPROJ_TITLE'] = f"""
Browse Images for the Reprojected Version of Cassini ISS Calibrated Image
{image_name} from Observation {root_obsid}
"""
        xml_metadata['BROWSE_REPROJ_DESCRIPTION'] = f"""
These browse images correspond to the reprojected, calibrated Cassini ISS image
{image_name} from observation {root_obsid} taken at {start_date}. The original
reprojected image is in units of I/F. The browse images map I/F to 8-bit
greyscale and are contrast-stretched for easier viewing, using a blackpoint at
the minimum image value, a whitepoint at the 99.8% maximum image value, and a
gamma of 0.5. Browse images are available in four sizes: full (containing only
the longitudes with valid data at full resolution, and thus possibly narrower
than the reprojected image data array when the coverage is discontinuous, with a
minimum width of 800 pixels), med (downsampled by
10 in longitude, with a minimum width of 400 pixels and a height of 400 pixels),
small (200x200), and thumb (100x100). The browse images omit longitudes that
have no data available; if the available longitudes are discontinuous, the
browse image will show the longitudes as being adjacent. Pixels with no data
available are shown as black.


This derived data product is part of bundle
cassini_iss_fring_mosaics_rsfrench2025, created by Robert S. French et al., and
archived at the Ring-Moon Systems Node. For full citation information, see
collection or bundle labels.
"""
    else:
        # Find the image names at the starting and ending ETs
        long_antimask = metadata['long_antimask']
        image_indexes = metadata['image_number'][long_antimask]
        image_name_list = metadata['image_name_list']
        ETs = metadata['time'][long_antimask]
        idx_min = np.argmin(ETs)
        idx_max = np.argmax(ETs)
        min_image_name = image_name_list[image_indexes[idx_min]]
        max_image_name = image_name_list[image_indexes[idx_max]]

        xml_metadata['BROWSE_MOSAIC_LID'] = obsid_to_mosaic_browse_lid(obsid,
                                                                       img_type == 'b')
        xml_metadata['BROWSE_MOSAIC_TITLE'] = f"""
Browse Images for the {title_bkg}F Ring Mosaic Created from Cassini
Observation {root_obsid} ({min_image_name} to {max_image_name})
"""
        xml_metadata['BROWSE_MOSAIC_DESCRIPTION'] = f"""
These browse images correspond to the {cap_bkg.lower()}F Ring mosaic created
from reprojected, calibrated Cassini ISS images from observation {root_obsid}.
The images used range from {min_image_name} ({start_date}) to {max_image_name}
({stop_date}). The original mosaic data are in units of I/F. The browse images map I/F to
8-bit greyscale and are contrast-stretched for easier viewing, using a
blackpoint at the minimum mosaic value, a whitepoint at the 99.8% maximum mosaic
value, and a gamma of 0.5. Browse images are available in four sizes: full
(18000x401), med (1800x400), small (200x200), and thumb (100x100). The full
longitude range is shown even when no images cover that area. Pixels with no
data available are shown as black.


This derived data product is part of bundle cassini_iss_fring_mosaics_rsfrench2025,
created by Robert S. French et al., and archived at the Ring-Moon Systems Node.
For full citation information, see collection or bundle labels.
"""

    if ((img_type == 'r' and GENERATE_BROWSE_REPROJ_LABELS) or
        (img_type != 'r' and GENERATE_BROWSE_MOSAIC_LABELS)):
        for size in ('full', 'med', 'small', 'thumb'):
            if img_type == 'r':
                browse_filename = f'{image_name.lower()}_browse_reproj_img_{size}.png'
            else:
                browse_filename = f'{obsid.lower()}_browse_mosaic{sfx}_{size}.png'
            xml_metadata[f'BROWSE_{size.upper()}_FILENAME'] = browse_filename
            png_path = os.path.join(browse_dir, browse_filename)
            xml_metadata[f'BROWSE_{size.upper()}_PATH'] = png_path

        if img_type == 'r':
            output_path = os.path.join(browse_dir,
                                       f'{image_name.lower()}_browse_reproj_img.lblx')
        else:
            output_path = os.path.join(browse_dir,
                                       f'{obsid.lower()}_browse_mosaic{sfx}.lblx')
        if img_type == 'r':
            populate_template('browse_reproj_img.lblx', output_path, xml_metadata)
        else:
            populate_template('browse_mosaic.lblx', output_path, xml_metadata)


def generate_mosaic(obsid,
                    mosaic_dir, bsm_dir,
                    mosaic_browse_dir, bsm_browse_dir,
                    mosaic_metadata, bsm_metadata, bkgnd_metadata,
                    global_mosaic_index_fp, global_bsm_index_fp):
    """Create all files related to mosaics.

    Inputs:
        obsid                   The observation name.
        mosaic_dir              The directory in which to put all mosaic files.
        bsm_dir                 The directory in which to put all bsm files.
        mosaic_browse_dir       The directory in which to put mosaic browse files.
        bsm_browse_dir          The directory in which to put bsm browse files.
        mosaic_metadata         The metadata for the mosaic.
        bsm_metadata            The metadata for the background-subtracted mosaic.
        bkgnd_metadata          The metadata for the background subtraction model.
        mosaic_global_index_fp  The file pointer for the mosaic global index.
        bsm_global_index_fp     The file pointer for the bsm global index.
    """
    # Do plain mosaics first
    xml_metadata = xml_metadata_for_image(obsid, mosaic_metadata, bkgnd_metadata, 'm')
    if (GENERATE_MOSAIC_METADATA_TABLES or GENERATE_MOSAIC_IMAGES or
        GENERATE_MOSAIC_IMAGE_LABELS or GENERATE_MOSAIC_GLOBAL_INDEX):
        generate_image(obsid, mosaic_dir, mosaic_metadata, xml_metadata,
                       global_mosaic_index_fp, 'm')
    if GENERATE_BROWSE_MOSAIC_IMAGES or GENERATE_BROWSE_MOSAIC_LABELS:
        generate_browse(obsid, mosaic_browse_dir, mosaic_metadata,
                        xml_metadata, 'm')

    # Now do BSM
    xml_metadata = xml_metadata_for_image(obsid, bsm_metadata, bkgnd_metadata, 'b')
    if (GENERATE_MOSAIC_METADATA_TABLES or GENERATE_MOSAIC_IMAGES or
        GENERATE_MOSAIC_IMAGE_LABELS or GENERATE_MOSAIC_GLOBAL_INDEX):
        generate_image(obsid, bsm_dir, bsm_metadata, xml_metadata,
                       global_bsm_index_fp, 'b')
    if GENERATE_BROWSE_MOSAIC_IMAGES or GENERATE_BROWSE_MOSAIC_LABELS:
        generate_browse(obsid, bsm_browse_dir, bsm_metadata, xml_metadata, 'b')


def generate_reproj(obsid, reproj_dir, reproj_browse_dir, reproj_metadata,
                    global_reproj_index_fp):
    """Create all files related to reprojected images.

    Inputs:
        obsid                   The observation name.
        reproj_dir              The directory in which to put all reproj files.
        reproj_browse_dir       The directory in which to put all reproj browse
                                files.
        reproj_metadata         The metadata for the reprojected images.
        global_reproj_index_fp  The file pointer for the reproj global index.
    """
    xml_metadata = xml_metadata_for_image(obsid, reproj_metadata, None, 'r')
    if (GENERATE_REPROJ_METADATA_TABLES or GENERATE_REPROJ_IMAGES or
        GENERATE_REPROJ_IMAGE_LABELS or GENERATE_REPROJ_SUPPL_FILES or
        GENERATE_REPROJ_GLOBAL_INDEX):
        generate_image(obsid, reproj_dir, reproj_metadata, xml_metadata,
                       global_reproj_index_fp, 'r')
    if GENERATE_BROWSE_REPROJ_IMAGES or GENERATE_BROWSE_REPROJ_LABELS:
        generate_browse(obsid, reproj_browse_dir, reproj_metadata,
                        xml_metadata, 'r')


##########################################################################################
#
# MAIN OBSID LOOP
#
##########################################################################################

def handle_one_obsid(obsid, reproj_collection_fp, browse_reproj_collection_fp,
                     global_mosaic_index_fp, global_bsm_index_fp,
                     global_reproj_index_fp):
    """Process one obsid.

    Returns True if the mosaic products were processed successfully; the caller
    uses this to decide whether to write the mosaic collection inventory rows.
    """
    mosaic_dir = os.path.join(arguments.output_dir, 'data_mosaic',
                              obsid.lower())
    bsm_dir = os.path.join(arguments.output_dir, 'data_mosaic_bkg_sub',
                           obsid.lower())

    # Paths for the mosaic image and the mosaic metadata
    (mosaic_path, mosaic_metadata_path) = f_ring.mosaic_paths(arguments, obsid)
    if not os.path.exists(mosaic_path):
        LOGGER.error(f'File not found: {mosaic_path}')
        return False
    if not os.path.exists(mosaic_metadata_path):
        LOGGER.error(f'File not found: {mosaic_metadata_path}')
        return False

    # Paths for the background-subtracted-mosaic image and metadata
    (bsm_path, bsm_metadata_path) = f_ring.bkgnd_sub_mosaic_paths(arguments, obsid)
    if not os.path.exists(bsm_path):
        LOGGER.error(f'File not found: {bsm_path}')
        return False
    if not os.path.exists(bsm_metadata_path):
        LOGGER.error(f'File not found: {bsm_metadata_path}')
        return False

    mosaic_metadata = None
    bsm_metadata = None

    if (GENERATE_MOSAIC_IMAGES or GENERATE_MOSAIC_IMAGE_LABELS or
        GENERATE_MOSAIC_METADATA_TABLES or GENERATE_BROWSE_MOSAIC_IMAGES or
        GENERATE_BROWSE_MOSAIC_LABELS or GENERATE_MOSAIC_GLOBAL_INDEX):
        mosaic_browse_dir = os.path.join(arguments.output_dir, 'browse_mosaic',
                                         obsid.lower())
        bsm_browse_dir = os.path.join(arguments.output_dir,
                                      'browse_mosaic_bkg_sub', obsid.lower())

        # Paths for the background model and background model metadata
        (bkgnd_model_path, bkgnd_metadata_path) = f_ring.bkgnd_paths(arguments, obsid)
        if not os.path.exists(bkgnd_model_path):
            LOGGER.error(f'File not found: {bkgnd_model_path}')
            return False
        if not os.path.exists(bkgnd_metadata_path):
            LOGGER.error(f'File not found: {bkgnd_metadata_path}')
            return False

        mosaic_metadata = read_mosaic(mosaic_path, mosaic_metadata_path, bkg_sub=False)
        bsm_metadata = read_mosaic(bsm_path, bsm_metadata_path, bkg_sub=True)
        bkgnd_metadata = read_bkgnd_metadata(bkgnd_model_path, bkgnd_metadata_path)

        if not all(obsid == x for x in mosaic_metadata['obsid_list']):
            LOGGER.error(f'Not all mosaic OBSIDs are {obsid}')
            return False
        if not all(obsid == x for x in bsm_metadata['obsid_list']):
            LOGGER.error(f'Not all background-sub mosaic OBSIDs are {obsid}')
            return False

        remap_image_indexes(mosaic_metadata)
        remap_image_indexes(bsm_metadata)

        generate_mosaic(obsid,
                        mosaic_dir, bsm_dir,
                        mosaic_browse_dir, bsm_browse_dir,
                        mosaic_metadata, bsm_metadata, bkgnd_metadata,
                        global_mosaic_index_fp, global_bsm_index_fp)

    if (GENERATE_REPROJ_IMAGES or GENERATE_REPROJ_IMAGE_LABELS or
        GENERATE_REPROJ_METADATA_TABLES or GENERATE_BROWSE_REPROJ_IMAGES or
        GENERATE_REPROJ_COLLECTIONS or GENERATE_BROWSE_REPROJ_COLLECTIONS or
        GENERATE_BROWSE_REPROJ_LABELS or GENERATE_REPROJ_SUPPL_FILES or
        GENERATE_REPROJ_GLOBAL_INDEX):
        if mosaic_metadata is None:
            mosaic_metadata = read_mosaic(mosaic_path, mosaic_metadata_path,
                                          bkg_sub=False, read_img=False)
            remap_image_indexes(mosaic_metadata)
        reproj_dir = os.path.join(arguments.output_dir, 'data_reproj_img',
                                  obsid.lower())
        reproj_browse_dir = os.path.join(arguments.output_dir, 'browse_reproj_img',
                                         obsid.lower())
        for image_path in mosaic_metadata['image_path_list']:
            try:
                reproj_path = img_to_repro_path(image_path)
                reproj_metadata = read_reproj(reproj_path)
                reproj_metadata['image_path'] = image_path
                reproj_metadata['image_name'] = image_name = \
                    reformat_iss_name(image_path.split('/')[-1].replace('_CALIB.IMG', ''))

                generate_reproj(obsid, reproj_dir, reproj_browse_dir, reproj_metadata,
                                global_reproj_index_fp)
            except ObsIdFailedException:
                # Already logged; skip this image's inventory rows without
                # losing the mosaic products generated above
                continue

            # Only list products in the inventories once they have actually
            # been generated
            if GENERATE_REPROJ_COLLECTIONS:
                reproj_lidvid = image_name_to_reproj_lidvid(image_name)
                reproj_collection_fp.write(f'P,{reproj_lidvid}\n')
            if GENERATE_BROWSE_REPROJ_COLLECTIONS:
                browse_reproj_lidvid = image_name_to_reproj_browse_lidvid(image_name)
                browse_reproj_collection_fp.write(f'P,{browse_reproj_lidvid}\n')

    return True


##########################################################################################
#
# GENERATE COLLECTION XMLs
#
##########################################################################################

def generate_mosaic_collection_xml(coll_data_mosaic_csv_path,
                                   coll_bsm_data_mosaic_csv_path):
    """Generate the data_mosaic and data_mosaic_bkg_sub collection xml files."""
    metadata = BASIC_XML_METADATA.copy()

    if EARLIEST_START_DATE_TIME is None or LATEST_STOP_DATE_TIME is None:
        LOGGER.error('Cannot generate data_mosaic collection labels without '
                     'traversing products; run with product generation enabled')
        return
    metadata['EARLIEST_START_DATE_TIME'] = et_to_datetime(EARLIEST_START_DATE_TIME)
    metadata['LATEST_STOP_DATE_TIME'] = et_to_datetime(LATEST_STOP_DATE_TIME)

    coll_data_mosaic_xml_path = coll_data_mosaic_csv_path.replace('.csv', '.lblx')
    coll_bsm_data_mosaic_xml_path = coll_bsm_data_mosaic_csv_path.replace('.csv', '.lblx')

    metadata['DATA_MOSAIC_COLLECTION_LID'] = DATA_MOSAIC_COLLECTION_LID
    metadata['DATA_MOSAIC_COLLECTION_CSV_PATH'] = coll_data_mosaic_csv_path
    metadata['DATA_MOSAIC_COLLECTION_TITLE'] = """
Collection for the (Non Background-Subtracted) F Ring Mosaics
Created from Reprojected, Calibrated Cassini ISS Images
"""
    metadata['DATA_MOSAIC_COLLECTION_DESCRIPTION'] = """
This is the collection of (non background-subtracted) F Ring mosaics
created from reprojected, calibrated Cassini ISS images, and
associated metadata.
"""
    metadata['DATA_MOSAIC_COLLECTION_CSV_NAME'] = 'collection_data_mosaic.csv'
    populate_template('collection_data_mosaic.lblx', coll_data_mosaic_xml_path, metadata)
    metadata['DATA_MOSAIC_COLLECTION_LID'] = DATA_MOSAIC_BKG_SUB_COLLECTION_LID
    metadata['DATA_MOSAIC_COLLECTION_CSV_PATH'] = coll_bsm_data_mosaic_csv_path
    metadata['DATA_MOSAIC_COLLECTION_TITLE'] = """
Collection for the Background-Subtracted F Ring Mosaics Created from
Reprojected, Calibrated Cassini ISS Images
"""
    metadata['DATA_MOSAIC_COLLECTION_DESCRIPTION'] = """
This is the collection of background-subtracted F Ring mosaics created from
reprojected, calibrated Cassini ISS images, and associated metadata.
"""
    metadata['DATA_MOSAIC_COLLECTION_CSV_NAME'] = 'collection_data_mosaic_bkg_sub.csv'
    populate_template('collection_data_mosaic.lblx', coll_bsm_data_mosaic_xml_path, metadata)


def generate_mosaic_browse_collection_xml(coll_browse_mosaic_csv_path,
                                          coll_bsm_browse_mosaic_csv_path):
    """Generate the browse_mosaic and browse_mosaic_bkg_sub collection xml files."""
    metadata = BASIC_XML_METADATA.copy()

    coll_browse_mosaic_xml_path = coll_browse_mosaic_csv_path.replace('.csv', '.lblx')
    coll_bsm_browse_mosaic_xml_path = coll_bsm_browse_mosaic_csv_path.replace('.csv', '.lblx')

    metadata['BROWSE_MOSAIC_COLLECTION_LID'] = BROWSE_MOSAIC_COLLECTION_LID
    metadata['BROWSE_MOSAIC_COLLECTION_CSV_PATH'] = coll_browse_mosaic_csv_path
    metadata['BROWSE_MOSAIC_COLLECTION_TITLE'] = """
Collection for the Browse Products for the (Non Background-Subtracted) F Ring
Mosaics Created from Reprojected, Calibrated Cassini ISS Images
"""
    metadata['BROWSE_MOSAIC_COLLECTION_DESCRIPTION'] = """
This is the collection of browse products for the (non background-subtracted) F
Ring mosaics created from reprojected, calibrated Cassini ISS images.
    """
    metadata['BROWSE_MOSAIC_COLLECTION_CSV_NAME'] = 'collection_browse_mosaic.csv'
    populate_template('collection_browse_mosaic.lblx', coll_browse_mosaic_xml_path, metadata)
    metadata['BROWSE_MOSAIC_COLLECTION_LID'] = BROWSE_MOSAIC_BKG_SUB_COLLECTION_LID
    metadata['BROWSE_MOSAIC_COLLECTION_CSV_PATH'] = coll_bsm_browse_mosaic_csv_path
    metadata['BROWSE_MOSAIC_COLLECTION_TITLE'] = """
Collection for the Browse Products for the Background-Subtracted F Ring
Mosaics Created from Reprojected, Calibrated Cassini ISS Images
"""
    metadata['BROWSE_MOSAIC_COLLECTION_DESCRIPTION'] = """
This is the collection of browse products for the background-subtracted F
Ring mosaics created from reprojected, calibrated Cassini ISS images.
"""
    metadata['BROWSE_MOSAIC_COLLECTION_CSV_NAME'] = 'collection_browse_mosaic_bkg_sub.csv'
    populate_template('collection_browse_mosaic.lblx', coll_bsm_browse_mosaic_xml_path, metadata)


def generate_reproj_collection_xml(coll_data_reproj_csv_path):
    """Generate the data_reproj collection xml file."""
    metadata = BASIC_XML_METADATA.copy()

    if EARLIEST_START_DATE_TIME is None or LATEST_STOP_DATE_TIME is None:
        LOGGER.error('Cannot generate data_reproj_img collection label without '
                     'traversing products; run with product generation enabled')
        return
    metadata['EARLIEST_START_DATE_TIME'] = et_to_datetime(EARLIEST_START_DATE_TIME)
    metadata['LATEST_STOP_DATE_TIME'] = et_to_datetime(LATEST_STOP_DATE_TIME)
    coll_data_reproj_xml_path = coll_data_reproj_csv_path.replace('.csv', '.lblx')

    metadata['DATA_REPROJ_COLLECTION_LID'] = DATA_REPROJ_COLLECTION_LID
    metadata['DATA_REPROJ_COLLECTION_CSV_PATH'] = coll_data_reproj_csv_path
    metadata['DATA_REPROJ_COLLECTION_TITLE'] = """
Collection of Reprojected, Calibrated Cassini ISS Images
"""
    metadata['DATA_REPROJ_COLLECTION_DESCRIPTION'] = """
This is the collection of reprojected, calibrated Cassini ISS images.
"""
    metadata['DATA_REPROJ_COLLECTION_CSV_NAME'] = 'collection_data_reproj_img.csv'
    populate_template('collection_data_reproj_img.lblx', coll_data_reproj_xml_path, metadata)


def generate_reproj_browse_collection_xml(coll_browse_reproj_csv_path):
    """Generate the browse_reproj_img collection xml file."""
    metadata = BASIC_XML_METADATA.copy()

    coll_browse_reproj_xml_path = coll_browse_reproj_csv_path.replace('.csv', '.lblx')

    metadata['BROWSE_REPROJ_COLLECTION_LID'] = BROWSE_REPROJ_COLLECTION_LID
    metadata['BROWSE_REPROJ_COLLECTION_CSV_PATH'] = coll_browse_reproj_csv_path
    metadata['BROWSE_REPROJ_COLLECTION_TITLE'] = """
Collection for the Browse Products for the Reprojected, Calibrated Cassini ISS
Images
"""
    metadata['BROWSE_REPROJ_COLLECTION_DESCRIPTION'] = """
This is the collection of browse products for the reprojected, calibrated Cassini
ISS images.
"""
    metadata['BROWSE_REPROJ_COLLECTION_CSV_NAME'] = 'collection_browse_reproj_img.csv'
    populate_template('collection_browse_reproj_img.lblx', coll_browse_reproj_xml_path, metadata)


def generate_global_index_xml(global_index_csv_path, hdr, img_type):
    """Generate a global index xml file."""
    metadata = BASIC_XML_METADATA.copy()

    if img_type == 'r':
        metadata['GLOBAL_INDEX_LID'] = GLOBAL_REPROJ_INDEX_LID
        metadata['GLOBAL_INDEX_TITLE'] = 'Global Reprojected Image Index'
        metadata['GLOBAL_INDEX_DESCRIPTION'] = """
Index table containing metadata for all reprojected images in the F-ring mosaic dataset.
        """
    elif img_type == 'm':
        metadata['GLOBAL_INDEX_LID'] = GLOBAL_MOSAIC_INDEX_LID
        metadata['GLOBAL_INDEX_TITLE'] = 'Global Mosaic Index'
        metadata['GLOBAL_INDEX_DESCRIPTION'] = """
Index table containing metadata for all mosaics in the F-ring mosaic dataset.
        """
    elif img_type == 'b':
        metadata['GLOBAL_INDEX_LID'] = GLOBAL_MOSAIC_BKG_SUB_INDEX_LID
        metadata['GLOBAL_INDEX_TITLE'] = 'Global Background-Subtracted Mosaic Index'
        metadata['GLOBAL_INDEX_DESCRIPTION'] = """
Index table containing metadata for all background-subtracted mosaics in the F-ring mosaic dataset.
        """
    else:
        raise ValueError(f'Invalid image type: {img_type}')

    global_index_xml_path = global_index_csv_path.replace('.tab', '.lblx')

    metadata['CURRENT_DATE'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
    metadata['HEADER_LENGTH'] = len(hdr)+1
    metadata['GLOBAL_INDEX_CSV_PATH'] = global_index_csv_path
    metadata['GLOBAL_INDEX_TABLE_FILENAME'] = global_index_csv_path.split('/')[-1]
    metadata['IS_MOSAIC'] = img_type in 'mb'
    metadata['IS_BKGND_SUB'] = img_type == 'b'
    populate_template('global_index.lblx', global_index_xml_path, metadata)


##########################################################################################
#
# GENERATE SUPPORT FILES
#
##########################################################################################

def generate_support_files():
    """Generate the support files."""
    # readme.txt
    copy_file('readme.txt', os.path.join(arguments.output_dir, 'readme.txt'))

    # context/collection_context.csv
    # context/collection_context.lblx
    metadata = BASIC_XML_METADATA.copy()
    context_dir = os.path.join(arguments.output_dir, 'context')
    csv_name = 'collection_context.csv'
    csv_path = os.path.join(context_dir, csv_name)
    metadata['CONTEXT_COLLECTION_LID'] = CONTEXT_COLLECTION_LID
    metadata['COLLECTION_CONTEXT_CSV_NAME'] = csv_name
    metadata['COLLECTION_CONTEXT_CSV_PATH'] = csv_path
    copy_file(csv_name, csv_path)
    populate_template('collection_context.lblx', csv_path.replace('.csv', '.lblx'),
                      metadata)

    # document/collection_document.csv
    # document/collection_document.lblx
    metadata = BASIC_XML_METADATA.copy()
    document_dir = os.path.join(arguments.output_dir, 'document')
    csv_name = 'collection_document.csv'
    csv_path = os.path.join(document_dir, csv_name)
    metadata['DOCUMENT_COLLECTION_LID'] = DOCUMENT_COLLECTION_LID
    metadata['COLLECTION_DOCUMENT_CSV_NAME'] = csv_name
    metadata['COLLECTION_DOCUMENT_CSV_PATH'] = csv_path
    copy_file(csv_name, csv_path)
    populate_template('collection_document.lblx', csv_path.replace('.csv', '.lblx'),
                      metadata)

    # spice_kernels/collection_spice_kernels.csv
    # spice_kernels/collection_spice_kernels.lblx
    metadata = BASIC_XML_METADATA.copy()
    spice_kernels_dir = os.path.join(arguments.output_dir, 'spice_kernels')
    csv_name = 'collection_spice_kernels.csv'
    csv_path = os.path.join(spice_kernels_dir, csv_name)
    metadata['SPICE_KERNELS_COLLECTION_LID'] = SPICE_KERNELS_COLLECTION_LID
    metadata['COLLECTION_SPICE_KERNELS_CSV_NAME'] = csv_name
    metadata['COLLECTION_SPICE_KERNELS_CSV_PATH'] = csv_path
    copy_file(csv_name, csv_path)
    populate_template('collection_spice_kernels.lblx',
                      csv_path.replace('.csv', '.lblx'), metadata)

    # spice_kernels/kernels.lblx
    metadata = BASIC_XML_METADATA.copy()
    spice_kernels_dir = os.path.join(arguments.output_dir, 'spice_kernels')
    kernels_name = 'kernels.ker'
    kernels_path = os.path.join(spice_kernels_dir, kernels_name)
    metadata['KERNELS_LID'] = KERNELS_LID
    metadata['KERNELS_NAME'] = kernels_name
    metadata['KERNELS_PATH'] = kernels_path
    copy_file(kernels_name, kernels_path)
    populate_template('kernels.lblx', kernels_path.replace('.ker', '.lblx'), metadata)

    # xml_schema/collection_xml_schema.csv
    # xml_schema/collection_xml_schema.lblx
    metadata = BASIC_XML_METADATA.copy()
    schema_dir = os.path.join(arguments.output_dir, 'xml_schema')
    csv_name = 'collection_xml_schema.csv'
    csv_path = os.path.join(schema_dir, csv_name)
    metadata['XML_SCHEMA_COLLECTION_LID'] = XML_SCHEMA_COLLECTION_LID
    metadata['COLLECTION_XML_SCHEMA_CSV_NAME'] = csv_name
    metadata['COLLECTION_XML_SCHEMA_CSV_PATH'] = csv_path
    copy_file(csv_name, csv_path)
    populate_template('collection_xml_schema.lblx',
                      os.path.join(schema_dir, csv_name.replace('.csv', '.lblx')),
                      metadata)

    # f-ring-mosaics-user-guide.lblx
    metadata = BASIC_XML_METADATA.copy()
    user_guide_dir = os.path.join(arguments.output_dir, 'document', 'user_guide')
    pdf_name = 'f-ring-mosaics-user-guide.pdf'
    pdf_path = os.path.join(user_guide_dir, pdf_name)
    metadata['USERGUIDE_PDF_NAME'] = pdf_name
    metadata['USERGUIDE_PDF_PATH'] = pdf_path
    copy_file(pdf_name, pdf_path)

    for example_name in ('display_reproj_img.py', 'plot_ews_ma.py', 'plot_ews_df.py',
                         'mosaic_utils.py', 'find_prometheus_closest_approaches.py'):
        example_path = os.path.join(user_guide_dir, example_name)
        metadata[example_name.replace('.', '_').upper() + '_NAME'] = example_name
        metadata[example_name.replace('.', '_').upper() + '_PATH'] = example_path
        copy_file(f'examples/{example_name}', example_path)

    populate_template('f-ring-mosaics-user-guide.lblx', pdf_path.replace('.pdf', '.lblx'),
                      metadata)

    # bundle.lblx
    if EARLIEST_START_DATE_TIME is None or LATEST_STOP_DATE_TIME is None:
        LOGGER.error('Cannot generate bundle.lblx without traversing products; '
                     'run with product generation enabled')
        return
    metadata = BASIC_XML_METADATA.copy()
    bundle_dir = arguments.output_dir
    bundle_name = 'bundle.lblx'
    bundle_path = os.path.join(bundle_dir, bundle_name)
    metadata['BUNDLE_LID'] = f'urn:nasa:pds:{BUNDLE_NAME}'
    metadata['EARLIEST_START_DATE_TIME'] = et_to_datetime(EARLIEST_START_DATE_TIME)
    metadata['LATEST_STOP_DATE_TIME'] = et_to_datetime(LATEST_STOP_DATE_TIME)
    metadata['BROWSE_MOSAIC_COLLECTION_LID'] = BROWSE_MOSAIC_COLLECTION_LID
    metadata['BROWSE_MOSAIC_BKG_SUB_COLLECTION_LID'] = BROWSE_MOSAIC_BKG_SUB_COLLECTION_LID
    metadata['BROWSE_REPROJ_COLLECTION_LID'] = BROWSE_REPROJ_COLLECTION_LID
    metadata['CONTEXT_COLLECTION_LID'] = CONTEXT_COLLECTION_LID
    metadata['DATA_MOSAIC_COLLECTION_LID'] = DATA_MOSAIC_COLLECTION_LID
    metadata['DATA_MOSAIC_BKG_SUB_COLLECTION_LID'] = DATA_MOSAIC_BKG_SUB_COLLECTION_LID
    metadata['DATA_REPROJ_COLLECTION_LID'] = DATA_REPROJ_COLLECTION_LID
    metadata['DOCUMENT_COLLECTION_LID'] = DOCUMENT_COLLECTION_LID
    metadata['MISCELLANEOUS_COLLECTION_LID'] = MISCELLANEOUS_COLLECTION_LID
    metadata['SPICE_KERNELS_COLLECTION_LID'] = SPICE_KERNELS_COLLECTION_LID
    metadata['XML_SCHEMA_COLLECTION_LID'] = XML_SCHEMA_COLLECTION_LID
    populate_template('bundle.lblx', bundle_path, metadata)


def generate_miscellaneous_support_files():
    # miscellaneous/collection_miscellaneous.csv
    # miscellaneous/collection_miscellaneous.lblx
    metadata = BASIC_XML_METADATA.copy()
    miscellaneous_dir = os.path.join(arguments.output_dir, 'miscellaneous')
    csv_name = 'collection_miscellaneous.csv'
    csv_path = os.path.join(miscellaneous_dir, csv_name)
    metadata['MISCELLANEOUS_COLLECTION_LID'] = MISCELLANEOUS_COLLECTION_LID
    metadata['COLLECTION_MISCELLANEOUS_CSV_NAME'] = csv_name
    metadata['COLLECTION_MISCELLANEOUS_CSV_PATH'] = csv_path
    copy_file(csv_name, csv_path)
    populate_template('collection_miscellaneous.lblx', csv_path.replace('.csv', '.lblx'),
                      metadata)


##########################################################################################
#
# TOP LEVEL
#
##########################################################################################

TITLE_FONTS = {
    ('r', 'thumb'): ImageFont.truetype('/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf', 11),
    ('r', 'small'): ImageFont.truetype('/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf', 14),
    ('r', 'med'): ImageFont.truetype('/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf', 14),
    ('r', 'full'): None,
    ('b', 'thumb'): ImageFont.truetype('/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf', 8),
    ('b', 'small'): ImageFont.truetype('/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf', 14),
    ('b', 'med'): ImageFont.truetype('/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf', 14),
    ('b', 'full'): None,
    ('m', 'thumb'): ImageFont.truetype('/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf', 8),
    ('m', 'small'): ImageFont.truetype('/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf', 14),
    ('m', 'med'): ImageFont.truetype('/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf', 14),
    ('m', 'full'): None,
}

EARLIEST_START_DATE_TIME = None
LATEST_STOP_DATE_TIME = None
SENTINEL = -999
OBSERVATION_INFO = None

BASIC_XML_METADATA = {
    'INFORMATION_MODEL_VERSION': '1.24.0.0',
    'PDS4_PDS_SCHEMA_XSD': 'https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1O00.xsd',
    'PDS4_PDS_SCHEMA': 'https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1O00.sch',
    'PDS4_RINGS_SCHEMA_XSD': 'https://pds.nasa.gov/pds4/rings/v1/PDS4_RINGS_1O00_1E00.xsd',
    'PDS4_RINGS_SCHEMA': 'https://pds.nasa.gov/pds4/rings/v1/PDS4_RINGS_1O00_1E00.sch',
    'PDS4_DISP_SCHEMA_XSD': 'https://pds.nasa.gov/pds4/disp/v1/PDS4_DISP_1O00_1510.xsd',
    'PDS4_DISP_SCHEMA': 'https://pds.nasa.gov/pds4/disp/v1/PDS4_DISP_1O00_1510.sch',
    'PDS4_CASSINI_SCHEMA_XSD': 'https://pds.nasa.gov/pds4/mission/cassini/v1/PDS4_CASSINI_1O00_1800.xsd',
    'PDS4_CASSINI_SCHEMA': 'https://pds.nasa.gov/pds4/mission/cassini/v1/PDS4_CASSINI_1O00_1800.sch',
    'PDS4_GEOM_SCHEMA_XSD': 'https://pds.nasa.gov/pds4/geom/v1/PDS4_GEOM_1O00_19B0.xsd',
    'PDS4_GEOM_SCHEMA': 'https://pds.nasa.gov/pds4/geom/v1/PDS4_GEOM_1O00_19B0.sch',
    'BUNDLE_DOI': '10.17189/3tfh-th07',
    'KEYWORDS': ['saturn rings', 'f ring', 'cassini iss'],
    'KEYWORDS_MOSAIC': ['saturn rings', 'f ring', 'cassini iss', 'mosaic'],
    'KEYWORDS_REPROJ': ['saturn rings', 'f ring', 'cassini iss', 'reprojected image'],
    'KEYWORDS_MOSAIC_REPROJ': ['saturn rings', 'f ring', 'cassini iss', 'mosaic', 'reprojected image'],
    'PUBLICATION_YEAR': datetime.datetime.now(datetime.UTC).strftime('%Y'),
    'USERGUIDE_LID': USERGUIDE_LID,
    'USERGUIDE_DOI': '10.17189/ajhh-aj88',
    'USERGUIDE_PDF_NAME': 'f-ring-mosaics-user-guide.pdf',
    'USERGUIDE_PDF_PATH': os.path.join('document', 'user_guide', 'f-ring-mosaics-user-guide.pdf'),
    'USERGUIDE_COMMENT': "Detailed User's Guide for the F Ring Mosaics and Reprojected Images in this bundle.",
    'XML_SCHEMA_COLLECTION_LID': f'urn:nasa:pds:{BUNDLE_NAME}:xml_schema',
    'CASSINI_USER_GUIDE_LID': 'urn:nasa:pds:cassini_iss_saturn:document:iss-data-user-guide',
    'CASSINI_USER_GUIDE_DESC': "The Cassini ISS Data User's Guide (PDS3); DOI: 10.17189/1504135",
    'SENTINEL': str(SENTINEL),
    'AUTHORS': 'Robert S. French, Matthew M. Hedman',
    'EDITORS': 'Mia J.T. Mace, Mitchell K. Gordon, Matthew S. Tiscareno, Emilie R. Simpson',
}


if (GENERATE_REPROJ_GLOBAL_INDEX or GENERATE_MOSAIC_GLOBAL_INDEX):
    # Index files
    os.makedirs(os.path.join(arguments.output_dir, 'miscellaneous'), exist_ok=True)

if (GENERATE_MOSAIC_IMAGE_LABELS or
    GENERATE_MOSAIC_IMAGES or
    GENERATE_MOSAIC_METADATA_TABLES or
    GENERATE_MOSAIC_COLLECTIONS):
    os.makedirs(os.path.join(arguments.output_dir, 'data_mosaic'), exist_ok=True)
    os.makedirs(os.path.join(arguments.output_dir, 'data_mosaic_bkg_sub'), exist_ok=True)

if (GENERATE_BROWSE_MOSAIC_LABELS or
    GENERATE_BROWSE_MOSAIC_IMAGES or
    GENERATE_BROWSE_MOSAIC_COLLECTIONS):
    os.makedirs(os.path.join(arguments.output_dir, 'browse_mosaic'), exist_ok=True)
    os.makedirs(os.path.join(arguments.output_dir, 'browse_mosaic_bkg_sub'),
                exist_ok=True)

if (GENERATE_REPROJ_METADATA_TABLES or
    GENERATE_REPROJ_COLLECTIONS):
    os.makedirs(os.path.join(arguments.output_dir, 'data_reproj_img'), exist_ok=True)

if (GENERATE_BROWSE_REPROJ_LABELS or
    GENERATE_BROWSE_REPROJ_IMAGES or
    GENERATE_BROWSE_REPROJ_COLLECTIONS):
    os.makedirs(os.path.join(arguments.output_dir, 'browse_reproj_img'), exist_ok=True)

if GENERATE_SUPPORT_FILES:
    os.makedirs(os.path.join(arguments.output_dir, 'context'), exist_ok=True)
    os.makedirs(os.path.join(arguments.output_dir, 'document/user_guide'), exist_ok=True)
    os.makedirs(os.path.join(arguments.output_dir, 'spice_kernels'), exist_ok=True)
    os.makedirs(os.path.join(arguments.output_dir, 'xml_schema'), exist_ok=True)

BASE_INDEX_HDR = ('pds:logical_identifier,'
                  'cassini:observation_id,'
                  'file_spec,'
                  'pds:start_date_time,'
                  'pds:stop_date_time,'
                  'cassini:spacecraft_clock_start_count,'
                  'cassini:spacecraft_clock_stop_count,'
                  'num_valid_longitudes,'
                  'percent_coverage,'
                  'rings:minimum_corotating_ring_longitude,'
                  'rings:maximum_corotating_ring_longitude,'
                  'rings:minimum_inertial_ring_longitude,'
                  'rings:maximum_inertial_ring_longitude,'
                  'rings:mean_phase_angle,'
                  'rings:minimum_phase_angle,'
                  'rings:maximum_phase_angle,'
                  'rings:mean_incidence_angle,'
                  'rings:mean_emission_angle,'
                  'rings:minimum_emission_angle,'
                  'rings:maximum_emission_angle,'
                  'rings:mean_radial_resolution,'
                  'rings:minimum_radial_resolution,'
                  'rings:maximum_radial_resolution,'
                  'rings:mean_longitudinal_resolution,'
                  'rings:minimum_longitudinal_resolution,'
                  'rings:maximum_longitudinal_resolution,'
                  'rings:minimum_ring_radius,'
                  'rings:maximum_ring_radius,'
                  'mean_core_radius,'
                  'minimum_core_radius,'
                  'maximum_core_radius,'
                  )

GLOBAL_REPROJ_INDEX_HDR = BASE_INDEX_HDR + (
                  'longitude_ascending_node,'
                  'longitude_pericenter,'
                  'minimum_true_anomaly,'
                  'maximum_true_anomaly,'
                  'corotating_longitude_prometheus,'
                  'radius_prometheus,'
                  'corotating_longitude_pandora,'
                  'radius_pandora,'
                  'pds:creation_date_time,'
                  'nav_quality,'
                  'notes')

GLOBAL_MOSAIC_INDEX_HDR_BASE = BASE_INDEX_HDR + (
                  'mean_longitude_ascending_node,'
                  'minimum_longitude_ascending_node,'
                  'maximum_longitude_ascending_node,'
                  'mean_longitude_pericenter,'
                  'minimum_longitude_pericenter,'
                  'maximum_longitude_pericenter,'
                  'minimum_true_anomaly,'
                  'maximum_true_anomaly,'
                  'mean_corotating_longitude_prometheus,'
                  'minimum_corotating_longitude_prometheus,'
                  'maximum_corotating_longitude_prometheus,'
                  'mean_radius_prometheus,'
                  'minimum_radius_prometheus,'
                  'maximum_radius_prometheus,'
                  'mean_corotating_longitude_pandora,'
                  'minimum_corotating_longitude_pandora,'
                  'maximum_corotating_longitude_pandora,'
                  'mean_radius_pandora,'
                  'minimum_radius_pandora,'
                  'maximum_radius_pandora,'
                  'pds:creation_date_time,'
                  'nav_quality,')

global_mosaic_index_fp = None
global_bsm_index_fp = None
if GENERATE_MOSAIC_GLOBAL_INDEX:
    global_mosaic_index_csv_path = os.path.join(arguments.output_dir,
                                                'miscellaneous',
                                                'global_mosaic_index.tab')
    global_mosaic_index_fp = open(global_mosaic_index_csv_path, 'w')
    GLOBAL_MOSAIC_INDEX_HDR = GLOBAL_MOSAIC_INDEX_HDR_BASE + ('notes,'
                                                              'num_images,'
                                                              'min_image_name,'
                                                              'max_image_name')
    global_mosaic_index_fp.write(GLOBAL_MOSAIC_INDEX_HDR+'\n')
    global_bsm_index_csv_path = os.path.join(arguments.output_dir,
                                             'miscellaneous',
                                             'global_mosaic_bkg_sub_index.tab')
    global_bsm_index_fp = open(global_bsm_index_csv_path, 'w')
    GLOBAL_BSM_INDEX_HDR = GLOBAL_MOSAIC_INDEX_HDR_BASE + ('bkgnd_quality,'
                                                           'notes,'
                                                           'num_images,'
                                                           'min_image_name,'
                                                           'max_image_name,'
                                                           'bkgnd_lower_limit,'
                                                           'bkgnd_upper_limit')
    global_bsm_index_fp.write(GLOBAL_BSM_INDEX_HDR+'\n')

mosaic_collection_fp = None
bsm_collection_fp = None
if GENERATE_MOSAIC_COLLECTIONS:
    mosaic_collection_csv_path = os.path.join(arguments.output_dir,
                                              'data_mosaic',
                                              'collection_data_mosaic.csv')
    mosaic_collection_fp = open(mosaic_collection_csv_path, 'w')
    bsm_collection_csv_path = os.path.join(arguments.output_dir,
                                           'data_mosaic_bkg_sub',
                                           'collection_data_mosaic_bkg_sub.csv')
    bsm_collection_fp = open(bsm_collection_csv_path, 'w')

browse_mosaic_collection_fp = None
browse_bsm_collection_fp = None
if GENERATE_BROWSE_MOSAIC_COLLECTIONS:
    browse_mosaic_collection_csv_path = os.path.join(arguments.output_dir,
                                                     'browse_mosaic',
                                                     'collection_browse_mosaic.csv')

    browse_mosaic_collection_fp = open(browse_mosaic_collection_csv_path, 'w')
    browse_bsm_collection_csv_path = os.path.join(arguments.output_dir,
                                                  'browse_mosaic_bkg_sub',
                                                  'collection_browse_mosaic_bkg_sub.csv')

    browse_bsm_collection_fp = open(browse_bsm_collection_csv_path, 'w')

global_reproj_index_fp = None
if GENERATE_REPROJ_GLOBAL_INDEX:
    global_reproj_index_csv_path = os.path.join(arguments.output_dir,
                                                'miscellaneous',
                                                'global_reproj_img_index.tab')
    global_reproj_index_fp = open(global_reproj_index_csv_path, 'w')
    global_reproj_index_fp.write(GLOBAL_REPROJ_INDEX_HDR+'\n')

reproj_collection_fp = None
if GENERATE_REPROJ_COLLECTIONS:
    reproj_collection_csv_path = os.path.join(arguments.output_dir,
                                              'data_reproj_img',
                                              'collection_data_reproj_img.csv')
    reproj_collection_fp = open(reproj_collection_csv_path, 'w')

browse_reproj_collection_fp = None
if GENERATE_BROWSE_REPROJ_COLLECTIONS:
    browse_reproj_collection_csv_path = os.path.join(arguments.output_dir,
                                                     'browse_reproj_img',
                                                     'collection_browse_reproj_img.csv')
    browse_reproj_collection_fp = open(browse_reproj_collection_csv_path, 'w')

read_observation_list()

for obsid in f_ring.enumerate_obsids(arguments):
    with LOGGER.open(f'OBSID {obsid}'):
        try:
            mosaic_ok = handle_one_obsid(
                             obsid, reproj_collection_fp, browse_reproj_collection_fp,
                             global_mosaic_index_fp, global_bsm_index_fp,
                             global_reproj_index_fp)
        except ObsIdFailedException:
            # A logged failure
            continue
        except KeyboardInterrupt:
            # Ctrl-C should be honored
            raise
        except SystemExit:
            # sys.exit() should be honored
            raise
        except:
            # Anything else
            LOGGER.error(f'{obsid}: Uncaught exception:\n' + traceback.format_exc())
            continue

        if not mosaic_ok:
            # Don't list the mosaic products in the inventories if they weren't
            # processed successfully
            continue

        if GENERATE_MOSAIC_COLLECTIONS:
            mosaic_lidvid = obsid_to_mosaic_lidvid(obsid, False)
            mosaic_collection_fp.write(f'P,{mosaic_lidvid}\n')
            bsm_lidvid = obsid_to_mosaic_lidvid(obsid, True)
            bsm_collection_fp.write(f'P,{bsm_lidvid}\n')
        if GENERATE_BROWSE_MOSAIC_COLLECTIONS:
            browse_mosaic_lidvid = obsid_to_mosaic_browse_lidvid(obsid, False)
            browse_mosaic_collection_fp.write(f'P,{browse_mosaic_lidvid}\n')
            browse_bsm_lidvid = obsid_to_mosaic_browse_lidvid(obsid, True)
            browse_bsm_collection_fp.write(f'P,{browse_bsm_lidvid}\n')

if GENERATE_MOSAIC_GLOBAL_INDEX:
    global_mosaic_index_fp.close()
    generate_global_index_xml(global_mosaic_index_csv_path, GLOBAL_MOSAIC_INDEX_HDR,
                              img_type='m')
    global_bsm_index_fp.close()
    generate_global_index_xml(global_bsm_index_csv_path, GLOBAL_BSM_INDEX_HDR,
                              img_type='b')
if GENERATE_MOSAIC_COLLECTIONS:
    mosaic_collection_fp.close()
    bsm_collection_fp.close()
    generate_mosaic_collection_xml(mosaic_collection_csv_path,
                                   bsm_collection_csv_path)
if GENERATE_BROWSE_MOSAIC_COLLECTIONS:
    browse_mosaic_collection_fp.close()
    browse_bsm_collection_fp.close()
    generate_mosaic_browse_collection_xml(browse_mosaic_collection_csv_path,
                                          browse_bsm_collection_csv_path)
if GENERATE_REPROJ_GLOBAL_INDEX:
    global_reproj_index_fp.close()
    generate_global_index_xml(global_reproj_index_csv_path, GLOBAL_REPROJ_INDEX_HDR,
                              img_type='r')
if GENERATE_REPROJ_COLLECTIONS:
    reproj_collection_fp.close()
    generate_reproj_collection_xml(reproj_collection_csv_path)
if GENERATE_BROWSE_REPROJ_COLLECTIONS:
    browse_reproj_collection_fp.close()
    generate_reproj_browse_collection_xml(browse_reproj_collection_csv_path)

if GENERATE_SUPPORT_FILES:
    generate_support_files()

if (GENERATE_REPROJ_GLOBAL_INDEX or GENERATE_MOSAIC_GLOBAL_INDEX):
    generate_miscellaneous_support_files()

# Support for OPUS index files:
#
# Occultation Constraints: None

#   General Constraints
# Planet - Saturn
# Intended Target Name - Saturn Rings
# Nominal Target Class - Ring
# Mission - Cassini
# Instrument Host Name - Cassini
# Instrument Name - Cassini ISS
# Observation Type - TBD: Reprojected Image? Mosaic?
# Observation Time - From start_date_time and stop_date_time
# Observation Duration - stop_date_time - start_date_time
# Measurement Quantity - Reflectivity
# Right Ascension - N/A
# Declination - N/A

#   PDS Constraints
# Volume ID - "urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:data_mosaic"
# Product Creation Time - product_creation_time
# Primary File Spec - file_spec
# OPUS ID - TBD
# Note - N/A

#   Wavelength Constraints
# Wavelength - Derive from CLEAR
# Wavelength Resolution - Derived
# Wavenumber - Derived
# Wavenumber Resolution - Derived
# Spectral Information Flag - No
# Spectrum Size - N/A
# Polarization Type - None

#   Ring Geo Constraints
# Observed Ring Radius - min_radius to max_radius
# Observed Longitude - min_inertial_longitude to max_inertial_longitude
# Sub-Solar J2000 Longitude - N/A
# Observed Solar Hour Angle - N/A
# Sub-Observer J2000 Longitude - N/A
# Longitude WRT Observer - N/A
# Azimuth WRT Observer - N/A
# Observed Distance to Ring Int - N/A
# Ring Center Distance - N/A
# Observed Resolution - N/A
# Projected Radial Resolution - min_radial_resolution to max_radial_resolution
# Observed Phase Angle - min_phase_angle to max_phase_angle
# Observed Incidence Angle - mean_incidence_angle
# Observed Emission Angle - min_emission_angle to max_emission_angle
# Observed North-Based Incidence - Derived
# Observed North-Based Emission - Derived
# Solar Ring Elevation - Derived
# Observer Ring Elevation - Derived
# Ring Center Phase - Copy from Observed Phase Angle?
# Ring Center Incidence - Copy from Observed Incidence Angle?
# Ring Center Emission - Copy from Observed Emission Angle?
# Ring Center North-Based Incidence - Copy from Observed North-Based Incidence?
# Ring Center North-Based Emission - Copy from Observed North-Based Emission?
# Ring Center Opening Angle to Sun - Copy from Observed Opening Angle to Sun?
# Ring Center Opening Angle to Observer - Copy from Observed Opening Angle to Observer?
# Edge-On ***
# Ring Event Time - N/A

#   Cassini Mission
# Observation Name - observation_id
# Activity Name - Derived
# Mission Phase - Derived
# Cassini Target Code - Derived
# Cassini Original Target Name - N/A
# Saturn Orbit Number - Derived
# Primary Instrument - Derived
# Is Prime - Derived
# Sequence ID - N/A
# Spacecraft Clock Start Count - spacecraft_clock_start_count
# Spacecraft Clock Stop Count - spacecraft_clock_stop_count
# Earth Received Start Time (YMDhms) - N/A
# Earth Received Stop Time (YMDhms) - N/A

#   Cassini ISS Constraints
# Camera - N or W
# Filter - CLEAR
# Shutter Mode - N/A
# Shutter State - Enabled
# Compression Type - N/A
# Data Conversion Type - N/A
# Gain Mode - N/A
# Instrument Mode - FULL
# Missing Lines - N/A
# Image Number - N/A
# Target Description - N/A
# Image Observation Type - N/A
# Exposure Duration [Image] (secs) - N/A


# Copy boilerplate
# Update f-ring-mosaics-user-guide.lblx
# kernels.lblx
