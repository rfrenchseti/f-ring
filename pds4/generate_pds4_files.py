##########################################################################################
# Create all files for the PDS4 achive including binary, tabular, and label.
##########################################################################################

import argparse
import csv
from datetime import datetime
import hashlib
import logging
import os
import pickle
import re
import sys
import traceback

import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy
import numpy as np
import numpy.ma as ma
from PIL import Image

import julian
import pdslogger
pdslogger.TIME_FMT = '%Y-%m-%d %H:%M:%S'

from pdsparser import PdsLabel

import f_ring_util.f_ring as f_ring

# XML directory structure:
#   bundle.xml                                      [RMS]
#   readme.txt                                      [RF writes]
#   browse_mosaics/
#     OBSID/
#       collection_browse_mosaics.csv               [generated: [P|S], LIDVID]
#       collection_browse_mosaics.xml               [RMS]
#       OBSID_browse_mosaic_full.png                [generated]
#       OBSID_browse_mosaic_med.png                 [generated]
#       OBSID_browse_mosaic_small.png               [generated]
#       OBSID_browse_mosaic_thumb.png               [generated]
#       OBSID_browse_mosaic.xml                     [template browse-image.xml]
#   browse_mosaics_bkg_sub/
#     OBSID/
#       collection_browse_mosaics_bkg_sub.csv       [generated: [P|S], LIDVID]
#       collection_browse_mosaics_bkg_sub.xml       [RMS]
#       OBSID_browse_mosaic_bkg_sub_full.png        [generated]
#       OBSID_browse_mosaic_bkg_sub_med.png         [generated]
#       OBSID_browse_mosaic_bkg_sub_small.png       [generated]
#       OBSID_browse_mosaic_bkg_sub_thumb.png       [generated]
#       OBSID_browse_mosaic_bkg_sub.xml             [template browse-image.xml]
#   context/
#     [written by RMS]
#   document/
#     collection_document.csv                       [RMS]
#     collection_document.xml                       [RMS]
#     document-01.pdf                               [RF writes]
#     document-01.xml                               [RMS]
#   data_mosaics/
#     collection_data_mosaics.csv                   [generated: [P|S], LIDVID]
#     collection_data_mosaics.xml                   [RMS]
#     OBSID/
#       OBSID_mosaic.img                            [generated]
#       OBSID_mosaic.xml                            [template mosaic.xml]
#       OBSID_mosaic_metadata_src_imgs.tab          [generated]
#       OBSID_mosaic_metadata_params.tab            [generated]
#       OBSID_mosaic_metadata.xml                   [template mosaic-metadata.xml]
#   data_mosaics_bkg_sub/
#     collection_data_mosaics_bkg_sub.csv           [generated: [P|S], LIDVID]
#     collection_data_mosaics_bkg_sub.xml           [RMS]
#     OBSID/
#       OBSID_mosaic_bkg_sub.img                    [generated]
#       OBSID_mosaic_bkg_sub.xml                    [template mosaic.xml]
#       OBSID_mosaic_bkg_sub_metadata_src_imgs.tab  [generated]
#       OBSID_mosaic_bkg_sub_metadata_params.tab    [generated]
#       OBSID_mosaic_bkg_sub_metadata.xml           [template mosaic-metadata.xml]
#   data_reproj_imgs/
#     collection_data_reproj_imgs.csv               [generated: [P|S], LIDVID]
#     collection_data_reproj_imgs.xml               [RMS]
#     OBSID/
#       xxxx_reproj.img                             [generated]
#       xxxx_reproj.xml                             [template TBD]
#   xml_schema/
#     [writted by RMS]


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

parser.add_argument('--generate-mosaic-labels',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub mosaic labels')
parser.add_argument('--generate-mosaic-images',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub mosaic image files')
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
parser.add_argument('--generate-mosaic-browse',
                    action='store_true', default=False,
                    help='Generate mosaic and bkgnd-sub mosaic browse image files and '
                         'labels')

parser.add_argument('--generate-all-mosaics',
                    action='store_true', default=False,
                    help='Generate all mosaic image, metadata, and browse files and '
                         'labels')

parser.add_argument('--generate-all',
                    action='store_true', default=False,
                    help='Generate all files and labels')

f_ring.add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)


CALIBRATED_DIR = '/data/pdsdata/holdings/calibrated' # XXX

GENERATE_REPROJ_IMAGE = False
GENERATE_REPROJ_LABELS = False
GENERATE_REPROJ_BROWSE_IMAGE = False
GENERATE_REPROJ_BROWSE_LABELS = False

GENERATE_MOSAIC_IMAGES = False
GENERATE_MOSAIC_IMAGE_LABELS = False
GENERATE_MOSAIC_METADATA_TABLES = False
GENERATE_MOSAIC_METADATA_LABELS = False
GENERATE_MOSAIC_BROWSE_IMAGE = False
GENERATE_MOSAIC_BROWSE_LABELS = False

if arguments.generate_mosaic_labels:
    GENERATE_MOSAIC_LABELS = True
if arguments.generate_mosaic_images:
    GENERATE_MOSAIC_IMAGES = True
if (arguments.generate_mosaics or
    arguments.generate_all_mosaics or
    arguments.generate_all):
    GENERATE_MOSAIC_LABELS = True
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


##########################################################################################
#
# LONG XML STRINGS
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


def populate_template(obs_id, template_name, output_path, xml_metadata):
    """Copy a template to an outut file after making $$ substitutions."""
    with open(os.path.join('templates', template_name), 'r') as template_fp:
        template = template_fp.read()

    for key, val in xml_metadata.items():
        template = template.replace(f'${key}$', val)

    remaining = re.findall(r'\$([^$]+)\$', template)
    if remaining:
        for remain in remaining:
            LOGGER.error(f'{obs_id}: Template {template_name} - Missed metadata '
                         f'field "{remain}"')

    with open(output_path, 'w') as output_fp:
        output_fp.write(template)


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


def read_mosaic(data_path, metadata_path, bkg_sub):
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

    if bkg_sub:
        with np.load(data_path) as npz:
            metadata['img'] = ma.MaskedArray(**npz)
    else:
        metadata['img'] = ma.MaskedArray(np.load(data_path))

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


def image_name_to_lidvid(name): ### Convert to LIDVID
    """Convert name like N1454725799_1 to a LID.

    LID is like: urn:nasa:pds:cassini_iss_saturn:data_raw:1454725799n
    """
    return ('urn:nasa:pds:cassini_iss_saturn:data_calibrated:'
            + name[1:11] + name[0].lower())


def obsid_to_mosaic_lid(obs_id):
    """Convert OBSID like IOSIC_276RB_COMPLITB4001_SI to a mosaic LID.

    LID is like: urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:
                 data_mosaics:IOSIC_276RB_COMPLITB4001_SI
    """
    return ( 'urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:'
            f'data_mosaics:{obs_id}')


def obsid_to_bsm_lid(obs_id):
    """Convert OBSID like IOSIC_276RB_COMPLITB4001_SI to a bsm LID.

    LID is like: urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:
                 data_mosaics_bkg_sub:IOSIC_276RB_COMPLITB4001_SI_bkg_sub
    """
    return ( 'urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:'
            f'data_mosaics_bkg_sub:{obs_id}_bkg_sub')


def obsid_to_mosaic_metadata_lid(obs_id):
    """Convert OBSID like IOSIC_276RB_COMPLITB4001_SI to a mosaic metadata LID.

    LID is like: urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:
                 data_mosaics:IOSIC_276RB_COMPLITB4001_SI_metadata
    """
    return ( 'urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_mosaics:'
            f'{obs_id}_metadata')


def obsid_to_bsm_metadata_lid(obs_id):
    """Convert OBSID like IOSIC_276RB_COMPLITB4001_SI to a bsm metadata LID.

    LID is like: urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:
                 data_mosaics_bkg_sub:IOSIC_276RB_COMPLITB4001_SI_metadata_bkg_sub
    """
    return ( 'urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:data_mosaics_bkg_sub:'
            f'{obs_id}_metadata_bkg_sub')


def obsid_to_mosaic_browse_lid(obs_id):
    """Convert OBSID like IOSIC_276RB_COMPLITB4001_SI to a mosaic browse LID.

    LID is like: urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:
                 browse_mosaics:IOSIC_276RB_COMPLITB4001_SI_browse_mosaic
    """
    return ( 'urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:browse_mosaics:'
            f'{obs_id}_browse_mosaic')


def obsid_to_bsm_browse_lid(obs_id):
    """Convert OBSID like IOSIC_276RB_COMPLITB4001_SI to a bsm browse LID.

    LID is like: urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:
                 browse_mosaics_bkg_sub:IOSIC_276RB_COMPLITB4001_SI_browse_mosaic_bkg_sub
    """
    return ( 'urn:nasa:pds:cdap2020_hedman_saturn_dusty_rings:'
            f'browse_mosaics_bkg_sub:{obs_id}_browse_mosaic_bkg_sub_')


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
    """Return the PDS3 label for the given image name."""
    components = image_name.split('/')[-5:]
    image_path = os.path.join(CALIBRATED_DIR, *components)
    label_path = image_path.replace('.IMG', '.LBL')
    return PdsLabel.from_file(label_path)


##########################################################################################
#
# PROCESS A MOSAIC
#
##########################################################################################

def xml_metadata_for_mosaic(obs_id, metadata, bkg_sub):
    """Generate the common template substitions for all mosaic types."""
    ret = BASIC_XML_METADATA.copy()

    sfx = '_bkg_sub' if bkg_sub else ''
    bkg_name = 'background-subtracted ' if bkg_sub else ''

    num_images = len(metadata['image_path_list'])
    long_mask = metadata['long_mask']

    ETs = metadata['time'][long_mask]
    ret['START_DATE_TIME'] = start_date = et_to_datetime(np.min(ETs))
    ret['STOP_DATE_TIME'] = stop_date = et_to_datetime(np.max(ETs))
    ret['TOUR'] = et_to_tour(ETs[0])

    ret['OBSERVATION_ID'] = obs_id
    ret['FILTER1'] = 'CL1'
    ret['FILTER2'] = 'CL2'

    if bkg_sub:
        ret['MOSAIC_LID'] = obsid_to_bsm_lid(obs_id)
        ret['MOSAIC_BROWSE_LID'] = obsid_to_bsm_browse_lid(obs_id)
        ret['MOSAIC_TITLE'] = f"""\
Background-subtracted F Ring mosaic of reprojected Cassini ISS \
calibrated images from observation {obs_id}, {start_date} to {stop_date}\
"""
        ret['MOSAIC_METADATA_LID'] = obsid_to_bsm_metadata_lid(obs_id)
        ret['MOSAIC_REFERENCE_COMMENT'] = 'Mosaic with background subtraction'
        ret['MOSAIC_METADATA_TITLE'] = f"""\
Metadata for background-subtracted F Ring mosaics of reprojected Cassini ISS \
calibrated images from observation {obs_id}, {start_date} to {stop_date}\
"""
    else:
        ret['MOSAIC_LID'] = obsid_to_mosaic_lid(obs_id)
        ret['MOSAIC_BROWSE_LID'] = obsid_to_mosaic_browse_lid(obs_id)
        ret['MOSAIC_TITLE'] = f"""\
F Ring mosaic of reprojected Cassini ISS calibrated images from \
observation {obs_id}, {start_date} to {stop_date}\
"""
        ret['MOSAIC_METADATA_LID'] = obsid_to_mosaic_metadata_lid(obs_id)
        ret['MOSAIC_REFERENCE_COMMENT'] = 'Mosaic (without background subtraction)'
        ret['MOSAIC_METADATA_TITLE'] = f"""\
Metadata for F Ring mosaics of reprojected Cassini ISS calibrated images from \
observation {obs_id}, {start_date} to {stop_date}\
"""

    ret['MOSAIC_DESCRIPTION'] = ret['MOSAIC_TITLE']

    ret['MOSAIC_COMMENT'] = f"""\
This data file is a mosaic of Saturn's F ring, stitched together from reprojected, calibrated images.
The reprojection takes the image space and reprojects it onto a regular radius/longitude grid,
where the longitude is co-rotating with the core of the F ring and the radius is really the offset
from the core at that longitude and time given a particular model of the F ring's orbit (in other
words, even though the F ring is eccentric, in the mosaic it looks like a straight line at constant
radius). Thus rather than "radius" it would be more accurate to describe it as "radial offset" from
the orbital solution.\
"""
    ret['MOSAIC_RINGS:DESCRIPTION'] = f"""\
This data file is a mosaic, stitched together from {num_images} reprojected, calibrated images.
The reprojection takes the image space and reprojects it onto a regular radius/longitude grid,
where the longitude is co-rotating with the core of the F ring and the radius is really the offset
from the core at that longitude and time given a particular model (Albers et al, 2009) of the
F ring's orbit (in other
words, even though the F ring is eccentric, in the mosaic it looks like a straight line at constant
radius). Thus rather than "radius" it would be more accurate to describe it as "radial offset" from
the orbital solution.

The following parameters in this class use the Albers 2009 model:
epoch_reprojection_basis_utc is the date and time of zero longitude of the rotating frame
corotation_rate is the mean corotation rate

Mean/min/max values are based upon the aggregate of the source images for the following parameters:
phase angle, observed ring elevation, ring longitude,...etc\
"""

    ret['MOSAIC_METADATA_DESCRIPTION'] = ret['MOSAIC_METADATA_TITLE']
    ret['MOSAIC_METADATA_COMMENT'] = f"""\
Two files containing metadata for the {bkg_name}mosaics of reprojected Cassini ISS
calibrated images from {obs_id}, {start_date} to {stop_date}:\
    1) Indices and LIDs of source images
    2) Metadata parameters per corotating longitude\
"""
    ret['MOSAIC_METADATA_RINGS:DESCRIPTION'] = ret['MOSAIC_METADATA_DESCRIPTION']

    ret['MOSAIC_IMG_FILENAME'] = f'{obs_id}_mosaic{sfx}.img'

    # Find the image names at the starting and ending ETs
    image_indexes = metadata['image_number'][long_mask]
    image_path_list = metadata['image_path_list']
    idx_min = np.argmin(ETs)
    idx_max = np.argmax(ETs)
    try:
        min_image_path = image_path_list[image_indexes[idx_min]]
    except IndexError:
        LOGGER.error(f'{obs_id}: Failed to lookup image path for earliest time: '
                     f'longitude idx {idx_min} gives image number '
                     f'{image_indexes[idx_min]} paths length '
                     f'{len(image_path_list)}: {image_path_list}')
        raise ObsIdFailedException
    try:
        max_image_path = image_path_list[image_indexes[idx_max]]
    except IndexError:
        LOGGER.error(f'{obs_id}: Failed to lookup image path for latest time: '
                     f'longitude idx {idx_min} gives image number '
                     f'{image_indexes[idx_min]} paths length '
                     f'{len(image_path_list)}: {image_path_list}')
        raise ObsIdFailedException
    try:
        min_label = read_label(min_image_path)
    except FileNotFoundError:
        LOGGER.error(f'{obs_id}: Failed to open label file {min_image_path}')
        raise ObsIdFailedException
    ret['SPACECRAFT_CLOCK_START_COUNT'] = str(min_label['SPACECRAFT_CLOCK_START_COUNT'])
    try:
        max_label = read_label(max_image_path)
    except FileNotFoundError:
        LOGGER.error(f'{obs_id}: Failed to open label file {max_image_path}')
        raise ObsIdFailedException
    ret['SPACECRAFT_CLOCK_STOP_COUNT'] = str(max_label['SPACECRAFT_CLOCK_STOP_COUNT'])

    ret['NUM_VALID_LONGITUDES'] = str(len(ETs))

    incidence_angle = np.degrees(metadata['mean_incidence'])
    ret['INCIDENCE_ANGLE'] = f'{incidence_angle:.6f}'

    # XXX Implement difference between emission angle and observation ring elevation
    emission_angles = np.degrees(metadata['mean_emission'][long_mask])
    ret['MEAN_OBS_RING_ELEV'] = f'{np.mean(emission_angles):.6f}'
    ret['MIN_OBS_RING_ELEV'] = f'{np.min(emission_angles):.6f}'
    ret['MAX_OBS_RING_ELEV'] = f'{np.max(emission_angles):.6f}'

    phase_angles = np.degrees(metadata['mean_phase'][long_mask])
    ret['MEAN_PHASE_ANGLE'] = f'{np.mean(phase_angles):.6f}'
    ret['MIN_PHASE_ANGLE'] = f'{np.min(phase_angles):.6f}'
    ret['MAX_PHASE_ANGLE'] = f'{np.max(phase_angles):.6f}'

    resolutions = metadata['mean_resolution'][long_mask]
    ret['MEAN_REPROJ_GRID_RAD_RES'] = f'{np.mean(resolutions):.6f}'
    ret['MIN_REPROJ_GRID_RAD_RES'] = f'{np.min(resolutions):.6f}'
    ret['MAX_REPROJ_GRID_RAD_RES'] = f'{np.max(resolutions):.6f}'

    image_name_list = metadata['image_name_list']
    ret['NUM_IMAGES'] = str(len(image_name_list))

    image_name0 = metadata['image_name_list'][0]
    camera = image_name0[0]
    if camera not in ('N', 'W'):
        LOGGER.fatal(f'Unknown camera for image {image_name0}')
        sys.exit(-1)
    for image_name in image_name_list:
        if image_name[0] != camera:
            LOGGER.error(f'{obs_id}: Inconsistent cameras for images '
                         f'{image_name0} and {image_name}')
            break
    if metadata['image_name_list'][0][0] == 'N':
        ret['CAMERA_WIDTH'] = 'Narrow'
        ret['CAMERA_WN_UC'] = 'N'
        ret['CAMERA_WN_LC'] = 'n'
    else:
        ret['CAMERA_WIDTH'] = 'Wide'
        ret['CAMERA_WN_UC'] = 'W'
        ret['CAMERA_WN_LC'] = 'w'

    # Mosaics are always written out to their full extent even if not all
    # longitudes are populated.
    ret['MIN_RING_COROT_LONG'] = '0'
    ret['MAX_RING_COROT_LONG'] = '360'
    ret['MIN_RING_INERTIAL_LONG'] = '0'   # XXX Deprecated - not used?
    ret['MAX_RING_INERTIAL_LONG'] = '360'

    return ret


def generate_mosaic_or_bsm(obs_id, mosaic_dir, metadata, xml_metadata, *,
                           bkg_sub=False):
    """Create mosaic images and labels and mosaic metadata tables and labels.

    Inputs:
        obs_id          The observation name.
        mosaic_dir      The directory in which to put all mosaic files.
        metadata        The metadata for a mosaic or background-subtracted
                        mosaic.
        bkg_sub         If True, this is a background-subtracted mosaic.

    The global flags like GENERATE_MOSAIC_LABELS are used to determine which
    output files to create:

        data_mosaics/
            OBSID/
                OBSID_mosaic.img
                OBSID_mosaic.xml
                OBSID_mosaic_metadata_src_imgs.tab
                OBSID_mosaic_metadata_params.tab
                OBSID_mosaic_metadata.xml
        data_mosaics_bkg_sub/
            OBSID/
                OBSID_mosaic_bkg_sub.img
                OBSID_mosaic_bkg_sub.xml
                OBSID_mosaic_bkg_sub_metadata_src_imgs.tab
                OBSID_mosaic_bkg_sub_metadata_params.tab
                OBSID_mosaic_bkg_sub_metadata.xml
    """
    long_mask = metadata['long_mask']
    longitudes = np.arange(len(long_mask)) * arguments.longitude_resolution
    longitudes = longitudes[long_mask]
    ETs = metadata['time'][long_mask]
    image_indexes = metadata['image_number'][long_mask]
    image_name_list = metadata['image_name_list']
    emission_angles = np.degrees(metadata['mean_emission'][long_mask])
    phase_angles = np.degrees(metadata['mean_phase'][long_mask])
    resolutions = metadata['mean_resolution'][long_mask]
    inertial_longitudes = np.degrees(f_ring.fring_corotating_to_inertial(
                                                    np.radians(longitudes),
                                                    ETs))

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

    sfx = '_bkg_sub' if bkg_sub else ''


            ###############################
            ###     METADATA_TABLES     ###
            ###############################

    params_filename = f'{obs_id}_mosaic{sfx}_metadata_params.tab'
    xml_metadata['METADATA_PARAMS_TABLE_FILENAME'] = params_filename
    metadata_params_table_path = os.path.join(mosaic_dir, params_filename)
    src_imgs_filename = f'{obs_id}_mosaic{sfx}_metadata_src_imgs.tab'
    xml_metadata['IMAGE_TABLE_FILENAME'] = src_imgs_filename
    image_table_path = os.path.join(mosaic_dir, src_imgs_filename)

    if GENERATE_MOSAIC_METADATA_TABLES:
        # mosaic_params.tab
        with open(metadata_params_table_path, 'w') as fp:
            fp.write('Corotating Longitude, Image Index, Mid-time SPICE ET, '
                     'Inertial Longitude, Resolution, Phase Angle, '
                     'Emission Angle\n')
            for idx in range(len(ETs)):
                longitude = longitudes[idx]
                image_idx = image_indexes[idx]
                et = ETs[idx]
                inertial = inertial_longitudes[idx]
                resolution = resolutions[idx]
                phase = phase_angles[idx]
                emission = emission_angles[idx]
                row = (f'{longitude:6.2f}, {image_idx:4d}, {et:13.3f}, {inertial:7.3f}, '
                       f'{resolution:10.5f}, {phase:10.6f}, {emission:10.6f}')
                fp.write(row+'\n')

        # mosaic_params.tab
        with open(image_table_path, 'w') as fp:
            fp.write('Source Image Index, LIDVID\n')
            for idx in range(len(image_name_list)):
                lidvid = image_name_to_lidvid(image_name_list[idx])
                row = f'{idx:4d}, {lidvid}'
                fp.write(row+'\n')


            ###############################
            ###     METADATA_LABELS     ###
            ###############################

    if GENERATE_MOSAIC_METADATA_LABELS:
        try:
            with open(metadata_params_table_path, 'rb') as fp:
                hash = hashlib.md5(fp.read()).hexdigest();
            xml_metadata['METADATA_PARAMS_TABLE_HASH'] = hash
        except FileNotFoundError:
            pass
        try:
            with open(image_table_path, 'rb') as fp:
                hash = hashlib.md5(fp.read()).hexdigest();
            xml_metadata['IMAGE_TABLE_HASH'] = hash
        except FileNotFoundError:
            pass
        output_path = os.path.join(mosaic_dir,
                                   f'{obs_id}_mosaic{sfx}_metadata.xml')
        populate_template(obs_id, 'mosaic-metadata.xml', output_path,
                          xml_metadata)


            ###############################
            ###      MOSAIC_IMAGES      ###
            ###############################

    img = ma.filled(metadata['img'], SENTINEL).astype('float32')
    xml_metadata['MOSAIC_NUM_SAMPLES'] = str(img.shape[1])
    xml_metadata['MOSAIC_NUM_LINES'] = str(img.shape[0])
    mosaic_image_path = os.path.join(mosaic_dir, xml_metadata['MOSAIC_IMG_FILENAME'])

    if GENERATE_MOSAIC_IMAGES:
        img.tofile(mosaic_image_path)


            ###############################
            ###      MOSAIC_LABELS      ###
            ###############################

    if GENERATE_MOSAIC_LABELS:
        try:
            with open(mosaic_image_path, 'rb') as fp:
                hash = hashlib.md5(fp.read()).hexdigest();
            xml_metadata['MOSAIC_IMG_HASH'] = hash
        except FileNotFoundError:
            pass
        output_path = os.path.join(mosaic_dir,
                                   f'{obs_id}_mosaic{sfx}.xml')
        populate_template(obs_id, 'mosaic.xml', output_path, xml_metadata)


def generate_mosaic_or_bsm_browse(obs_id, browse_dir, metadata, xml_metadata,
                                  bkg_sub=False):
    """Create mosaic browse images. These are only from bkg-sub mosaics.

    Inputs:
        obs_id          The observation name.
        browse_dir      The directory in which to put all browse files.
        metadata        The metadata for a background-subtracted mosaic.

    The global flags like GENERATE_MOSAIC_BROWSE_IMAGES are used to determine
    which output files to create:

    browse_mosaics/
      OBSID/
        OBSID_browse_mosaic_full.png
        OBSID_browse_mosaic_med.png
        OBSID_browse_mosaic_small.png
        OBSID_browse_mosaic_thumb.png
        OBSID_browse_mosaic.xml
    browse_mosaics_bkg_sub/
      OBSID/
        OBSID_browse_mosaic_bkg_sub_full.png
        OBSID_browse_mosaic_bkg_sub_med.png
        OBSID_browse_mosaic_bkg_sub_small.png
        OBSID_browse_mosaic_bkg_sub_thumb.png
        OBSID_browse_mosaic_bkg_sub.xml
    """
    sfx = '_bkg_sub' if bkg_sub else ''

            ###############################
            ###      BROWSE_IMAGES      ###
            ###############################

    if GENERATE_MOSAIC_BROWSE_IMAGES:
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
        greyscale_img = np.floor((np.maximum(img-blackpoint, 0)/
                                  (whitepoint-blackpoint))**gamma*256)
        greyscale_img = np.clip(greyscale_img, 0, 255)
        scaled_mosaic = np.cast['int8'](greyscale_img[::-1,:])

        for size, sub0, sub1, dim0, dim1 in (
                                ('full',    1, 1, 18000, 401),
                                ('med',    10, 1,  1800, 400),
                                ('small',  90, 2,   200, 200),
                                ('thumb', 180, 4,   100, 100)):
            # +0 makes a copy
            sub_img = scaled_mosaic[::sub1,::sub0][:dim1,:dim0]+0
            pil_img = Image.frombuffer('L', (sub_img.shape[1],
                                            sub_img.shape[0]),
                                    sub_img, 'raw', 'L', 0, 1)
            png_path = os.path.join(browse_dir,
                                    f'{obs_id}_browse_mosaic{sfx}_{size}.png')
            pil_img.save(png_path, 'PNG')


            ###############################
            ###      BROWSE_LABELS      ###
            ###############################

    start_date = xml_metadata['START_DATE_TIME']
    stop_date = xml_metadata['STOP_DATE_TIME']

    if bkg_sub:
        xml_metadata['MOSAIC_BROWSE_LID'] = obsid_to_bsm_browse_lid(obs_id)
        xml_metadata['MOSAIC_BROWSE_TITLE'] = f"""\
Browse images for background-subtracted F Ring
mosaic of reprojected Cassini ISS calibrated images from observation
{obs_id}, {start_date} to {stop_date}. Images are available in
four sizes: thumb (100x100), small (200x200), med (1800x400) and
full (18000x401).
"""
    else:
        xml_metadata['MOSAIC_BROWSE_LID'] = obsid_to_mosaic_browse_lid(obs_id)
        xml_metadata['MOSAIC_BROWSE_TITLE'] = f"""\
Browse images for F Ring mosaic of reprojected
Cassini ISS calibrated images from observation {obs_id},
{start_date} to {stop_date}. Images are available in
four sizes: thumb (100x100), small (200x200), med (1800x400) and
full (18000x401).
"""

    if GENERATE_MOSAIC_BROWSE_LABELS:
        for size, sub0, sub1, dim0, dim1 in (
                                ('full',    1, 1, 18000, 401),
                                ('med',    10, 1,  1800, 400),
                                ('small',  90, 2,   200, 200),
                                ('thumb', 180, 4,   100, 100)):
            browse_filename = f'{obs_id}_browse_mosaic{sfx}_{size}.png'
            xml_metadata[f'MOSAIC_BROWSE_{size.upper()}_FILENAME'] = browse_filename
            png_path = os.path.join(browse_dir, browse_filename)
            try:
                with open(png_path, 'rb') as fp:
                    hash = hashlib.md5(fp.read()).hexdigest();
                xml_metadata[f'MOSAIC_BROWSE_{size.upper()}_HASH'] = hash
            except FileNotFoundError:
                pass

        output_path = os.path.join(browse_dir,
                                   f'{obs_id}_browse_mosaic{sfx}.xml')
        populate_template(obs_id, 'browse-image.xml', output_path, xml_metadata)


def generate_mosaic(obs_id,
                    mosaic_dir, bsm_dir,
                    mosaic_browse_dir, bsm_browse_dir,
                    mosaic_metadata, bsm_metadata, bkgnd_metadata):
    """Create all files related to mosaics.

    Inputs:
        obs_id          The observation name.
        mosaic_dir      The directory in which to put all mosaic files.
        bsm_dir         The directory in which to put all bsm files.
        mosaic_metadata The metadata for the mosaic.
        bsm_metadata    The metadata for the background-subtracted mosaic.
        bkgnd_metadata  The metadata for the background subtraction model.
    """
    # Do plain mosaics first
    xml_metadata = xml_metadata_for_mosaic(obs_id, mosaic_metadata, False)
    if (GENERATE_MOSAIC_METADATA_TABLES or GENERATE_MOSAIC_METADATA_LABELS or
        GENERATE_MOSAIC_IMAGES or GENERATE_MOSAIC_IMAGE_LABELS):
        generate_mosaic_or_bsm(obs_id, mosaic_dir, mosaic_metadata,
                               xml_metadata)
    if GENERATE_MOSAIC_BROWSE_IMAGE or GENERATE_MOSAIC_BROWSE_LABELS:
        generate_mosaic_or_bsm_browse(obs_id, mosaic_browse_dir, mosaic_metadata,
                                      xml_metadata)

    # Now do BSM
    xml_metadata = xml_metadata_for_mosaic(obs_id, bsm_metadata, True)
    if (GENERATE_MOSAIC_METADATA_TABLES or GENERATE_MOSAIC_METADATA_LABELS or
        GENERATE_MOSAIC_IMAGES or GENERATE_MOSAIC_IMAGE_LABELS):
        generate_mosaic_or_bsm(obs_id, bsm_dir, bsm_metadata,
                               xml_metadata, bkg_sub=True)
    if GENERATE_MOSAIC_BROWSE_IMAGE or GENERATE_MOSAIC_BROWSE_LABELS:
        generate_mosaic_or_bsm_browse(obs_id, bsm_browse_dir, bsm_metadata,
                                      xml_metadata, bkg_sub=True)


##########################################################################################
#
# MAIN OBSID LOOP
#
##########################################################################################

def handle_one_obsid(obs_id):
    # Paths for the mosaic image and the mosaic metadata
    (mosaic_path, mosaic_metadata_path) = f_ring.mosaic_paths(arguments, obs_id)
    if not os.path.exists(mosaic_path):
        LOGGER.error(f'File not found: {mosaic_path}')
        return
    if not os.path.exists(mosaic_metadata_path):
        LOGGER.error(f'File not found: {mosaic_metadata_path}')
        return

    # Paths for the background-subtracted-mosaic image and metadata
    (bsm_path, bsm_metadata_path) = f_ring.bkgnd_sub_mosaic_paths(arguments, obs_id)
    if not os.path.exists(bsm_path):
        LOGGER.error(f'File not found: {bsm_path}')
        return
    if not os.path.exists(bsm_metadata_path):
        LOGGER.error(f'File not found: {bsm_metadata_path}')
        return

    # Paths for the background model and background model metadata
    (bkgnd_model_path, bkgnd_metadata_path) = f_ring.bkgnd_paths(arguments, obs_id)
    if not os.path.exists(bkgnd_model_path):
        LOGGER.error(f'File not found: {bkgnd_model_path}')
        return
    if not os.path.exists(bkgnd_metadata_path):
        LOGGER.error(f'File not found: {bkgnd_metadata_path}')
        return

    mosaic_dir = os.path.join(arguments.output_dir, 'data_mosaics', obs_id)
    os.makedirs(mosaic_dir, exist_ok=True)
    bsm_dir = os.path.join(arguments.output_dir, 'data_mosaics_bkg_sub', obs_id)
    os.makedirs(bsm_dir, exist_ok=True)
    reproj_dir = os.path.join(arguments.output_dir, 'data_reproj_imgs', obs_id)
    os.makedirs(reproj_dir, exist_ok=True)
    mosaic_browse_dir = os.path.join(arguments.output_dir, 'browse_mosaics',
                                     obs_id)
    os.makedirs(mosaic_browse_dir, exist_ok=True)
    bsm_browse_dir = os.path.join(arguments.output_dir,
                                  'browse_mosaics_bkg_sub', obs_id)
    os.makedirs(bsm_browse_dir, exist_ok=True)

    mosaic_metadata = read_mosaic(mosaic_path, mosaic_metadata_path, False)
    bsm_metadata = read_mosaic(bsm_path, bsm_metadata_path, True)
    bkgnd_metadata = read_bkgnd_metadata(bkgnd_model_path, bkgnd_metadata_path)

    if not all(obs_id == x for x in mosaic_metadata['obsid_list']):
        LOGGER.error(f'Not all mosaic OBSIDs are {obs_id}')
        return
    if not all(obs_id == x for x in bsm_metadata['obsid_list']):
        LOGGER.error(f'Not all background-sub mosaic OBSIDs are {obs_id}')
        return

    remap_image_indexes(mosaic_metadata)
    remap_image_indexes(bsm_metadata)

    # if (GENERATE_REPROJ_IMAGE or GENERATE_REPROJ_LABELS or
    #     GENERATE_REPROJ_BROWSE_IMAGE or GENERATE_REPROJ_BROWSE_LABELS):
    #     generate_reproj()

    if (GENERATE_MOSAIC_IMAGES or GENERATE_MOSAIC_IMAGE_LABELS or
        GENERATE_MOSAIC_METADATA_TABLES or GENERATE_MOSAIC_METADATA_LABELS or
        GENERATE_MOSAIC_BROWSE_IMAGE or GENERATE_MOSAIC_BROWSE_LABELS):
        generate_mosaic(obs_id,
                        mosaic_dir, bsm_dir,
                        mosaic_browse_dir, bsm_browse_dir,
                        mosaic_metadata, bsm_metadata, bkgnd_metadata)


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
    'SENTINEL': str(SENTINEL)
}




# XXX Make bundle/collection files here
for obs_id in f_ring.enumerate_obsids(arguments):
    LOGGER.open(f'OBSID {obs_id}')
    try:
        handle_one_obsid(obs_id)
    except ObsIdFailedException:
        # A logged failure
        pass
    except KeyboardInterrupt:
        # Ctrl-C should be honored
        pass
    except SystemExit:
        # sys.exit() should be honored
        raise
    except:
        # Anything else
        LOGGER.error('Uncaught exception:\n' + traceback.format_exc())

    LOGGER.close()
    break
