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

import julian

import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy
import numpy as np
import numpy.ma as ma

import pdslogger
pdslogger.TIME_FMT = '%Y-%m-%d %H:%M:%S'

import f_ring_util.f_ring as f_ring

# Metadata:
# ['ring_lower_limit', 'ring_upper_limit', 'radius_resolution',
#  'longitude_resolution', 'long_mask', 'image_numbers', 'ETs',
#  'emission_angles', 'incidence_angle', 'phase_angles', 'resolutions',
#  'longitudes', 'obsid_list', 'image_name_list', 'image_path_list',
#  'repro_path_list']


# XML directory structure:
# - bundle.xml
# - readme.txt
# - browse/
#     - collection_browse.csv
#     - collection_browse.xml
#     - CSV - [P|S], LIDVID
#     - xxx_browse_mosaic.png
#     - xxx_browse_mosaic.xml            [template]
#     - xxx_browse_mosaic_bkg_sub.png
#     - xxx_browse_mosaic_bkg_sub.xml            [template]
# - context/
#     - [written by RMS]
# - document/
#     - collection_document.[csv,xml]
#     - document-01.pdf
#     - document-01.xml
# - xml_schema/
#     - [RMS]
#
#  - data_mosaics/
#       - OBSID/
#             - xxxx_mosaic.img
#             - xxxx_mosaic.xml
#             - xxxx_mosaic_bkg_sub.img
#             - xxxx_mosaic_bkg_sub.xml
#             - xxxx_mosaic_metadata_src_imgs.tab
#             - xxxx_mosaic_metadata_params.tab
#             - xxxx_mosaic_metadata_bkg_sub_src_imgs.tab
#             - xxxx_mosaic_metadata_bkg_sub_params.tab
#             - xxxx_mosaic_metadata.xml #this describes both metadata tables
#             ... etc
# - data_reproj_imgs/
#       - OBSID/
#            - xxxx_reproj.img
#            - xxxx_reproj.xml
#             ...etc


cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
   cmd_line = []

parser = argparse.ArgumentParser()

parser.add_argument('--output-dir', type=str, default='.',
                    help='The root directory for all output files')

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



f_ring.add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)


GENERATE_REPROJ_IMAGE = False
GENERATE_REPROJ_LABELS = False
GENERATE_REPROJ_BROWSE_IMAGE = False
GENERATE_REPROJ_BROWSE_LABELS = False
GENERATE_MOSAIC_IMAGE = False
GENERATE_MOSAIC_IMAGE_LABELS = False
GENERATE_MOSAIC_METADATA_TABLES = False
GENERATE_MOSAIC_METADATA_LABELS = False
GENERATE_MOSAIC_BROWSE_IMAGE = False
GENERATE_MOSAIC_BROWSE_LABELS = False

if arguments.generate_mosaic_labels:
    GENERATE_MOSAIC_LABELS = True
if arguments.generate_mosaic_images:
    GENERATE_MOSAIC_IMAGES = True
if arguments.generate_mosaics:
    GENERATE_MOSAIC_LABELS = True
    GENERATE_MOSAIC_IMAGES = True

if arguments.generate_mosaic_metadata_labels:
    GENERATE_MOSAIC_METADATA_LABELS = True
if arguments.generate_mosaic_metadata_tables:
    GENERATE_MOSAIC_METADATA_TABLES = True
if arguments.generate_mosaic_metadata:
    GENERATE_MOSAIC_METADATA_LABELS = True
    GENERATE_MOSAIC_METADATA_TABLES = True

if arguments.generate_mosaic_browse_labels:
    GENERATE_MOSAIC_BROWSE_LABELS = True
if arguments.generate_mosaic_browse_images:
    GENERATE_MOSAIC_BROWSE_IMAGES = True
if arguments.generate_mosaic_browse:
    GENERATE_MOSAIC_BROWSE_LABELS = True
    GENERATE_MOSAIC_BROWSE_IMAGES = True


LOGGER = pdslogger.PdsLogger('fring.pds4')

LOG_DIR = os.path.join(arguments.output_dir, 'logs')
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



def datetime_from_et(et):
    return julian.ymdhms_format_from_tai(julian.tai_from_tdb(et)) + 'Z'

def populate_template(template_name, output_path, xml_metadata):
    with open(os.path.join('templates', template_name), 'r') as template_fp:
        template = template_fp.read()
    for key, val in xml_metadata.items():
        template = template.replace(f'${key}$', val)

    remaining = re.findall(r'\$([^$]+)\$', template)
    if remaining:
        for remain in remaining:
            LOGGER.error(f'Template {template_name} - Missed metadata field "{remain}"')

    with open(output_path, 'w') as output_fp:
        output_fp.write(template)

def fixup_byte_to_str(data):
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

# For both main mosaic and background-subtracted mosaic
def read_mosaic(data_path, metadata_path):
    print(metadata_path)
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

    metadata['img'] = np.load(data_path)

    return metadata

def read_bkgnd_metadata(model_path, metadata_path):
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
    return False

def mosaic_has_pandora(metadata):
    return False

def remap_image_indexes(metadata):
    """Remap the image indexes to be contiguous starting at 0.

    This is necessary in case any of the images that went into building the
    mosaic didn't actually get used. This also is going to limit which
    reprojected images we include, because if an image wasn't used to make the
    mosaic, we never checked to see if it was navigated properly.
    """
    image_indexes = metadata['image_number']
    image_name_list = metadata['image_name_list']
    used_indexes = sorted(set(image_indexes))
    number_map = {}
    for i in range(len(used_indexes)):
        number_map[used_indexes[i]] = i
    new_image_indexes = [number_map[x] for x in image_indexes]
    new_image_name_list = [image_name_list[number_map[x]]
                               for x in range(len(used_indexes))]
    metadata['image_number'] = np.array(new_image_indexes)
    metadata['image_name_list'] = new_image_name_list



##########################################################################################
#
# PROCESS A MOSAIC
#
##########################################################################################

def xml_metadata_for_mosaic(obs_id, metadata):
    """Generate the common template substitions for all mosaic types.

    Inputs:
        obs_id          The observation name.
        metadata        The metadata for a mosaic or background-subtracted
                        mosaic.

    Returns:
        A dictionary with all common template substitions.
    """
    ret = BASIC_XML_METADATA.copy()

    ret['OBSERVATION_ID'] = obs_id
    ret['LID'] = 'XXXLIDXXX'
    ret['FILTER1'] = 'CL1'
    ret['FILTER2'] = 'CL2'

    long_mask = metadata['long_mask']

    ETs = metadata['time'][long_mask]
    ret['START_DATE_TIME'] = datetime_from_et(np.min(ETs))
    ret['STOP_DATE_TIME'] = datetime_from_et(np.max(ETs))

    ret['NUM_VALID_LONGITUDES'] = str(len(ETs))

    incidence_angle = np.degrees(metadata['mean_incidence'])
    ret['INCIDENCE_ANGLE'] = f'{incidence_angle:.6f}'

    # XXX Implement difference between emission angle and observation ring elevation
    emission_angles = np.degrees(metadata['mean_emission'][long_mask])
    ret['MEAN_OBS_RING_ELEV'] = f'{np.mean(emission_angles):.6f}'
    ret['MIN_OBS_RING_ELEV'] = f'{np.mean(emission_angles):.6f}'
    ret['MAX_OBS_RING_ELEV'] = f'{np.mean(emission_angles):.6f}'

    phase_angles = np.degrees(metadata['mean_phase'][long_mask])
    ret['MEAN_PHASE_ANGLE'] = f'{np.mean(phase_angles):.6f}'
    ret['MIN_PHASE_ANGLE'] = f'{np.mean(phase_angles):.6f}'
    ret['MAX_PHASE_ANGLE'] = f'{np.mean(phase_angles):.6f}'

    resolutions = metadata['mean_resolution'][long_mask]
    ret['MEAN_REPROJ_GRID_RAD_RES'] = f'{np.mean(phase_angles):.6f}'
    ret['MIN_REPROJ_GRID_RAD_RES'] = f'{np.mean(phase_angles):.6f}'
    ret['MAX_REPROJ_GRID_RAD_RES'] = f'{np.mean(phase_angles):.6f}'

    image_name_list = metadata['image_name_list']
    ret['NUM_IMAGES'] = str(len(image_name_list))

    image_name0 = metadata['image_name_list'][0]
    camera = image_name0[0]
    if camera not in ('N', 'W'):
        LOGGER.fatal(f'Unknown camera for image {image_name0}')
        sys.exit(-1)
    for image_name in image_name_list:
        if image_name[0] != camera:
            LOGGER.error('Inconsistent cameras for images {image_name0} and {image_name}')
            break
    if metadata['image_name_list'][0][0] == 'N':
        ret['CAMERA_WIDTH'] = 'Narrow'
        ret['CAMERA_WN_UC'] = 'N'
        ret['CAMERA_WN_LC'] = 'n'
    else:
        ret['CAMERA_WIDTH'] = 'Wide'
        ret['CAMERA_WN_UC'] = 'W'
        ret['CAMERA_WN_LC'] = 'w'

    ret['MIN_RING_LONG'] = '0'
    ret['MAX_RING_LONG'] = '360'


    return ret


def generate_mosaic_or_bsm(obs_id, mosaic_dir, metadata, *, bkgnd_sub=False):
    """Create mosaic images and labels and mosaic metadata tables and labels.

    Inputs:
        obs_id          The observation name.
        mosaic_dir      The directory in which to put all mosaic files.
        metadata        The metadata for a mosaic or background-subtracted
                        mosaic.
        bkgnd_sub       If True, this is a background-subtracted mosaic.

    The global flags like GENERATE_MOSAIC_LABELS are used to determine which
    output files to create:

        OBSID_mosaic.img
        OBSID_mosaic.xml
        OBSID_mosaic_bkg_sub.img
        OBSID_mosaic_bkg_sub.xml
        OBSID_mosaic_metadata_src_imgs.tab
        OBSID_mosaic_metadata_params.tab
        OBSID_mosaic_metadata.xml
        OBSID_mosaic_bkg_sub_metadata_src_imgs.tab
        OBSID_mosaic_bkg_sub_metadata_params.tab
        OBSID_mosaic_bkg_sub_metadata.xml
    """
    xml_metadata = xml_metadata_for_mosaic(obs_id, metadata)

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

    start_date = datetime_from_et(np.min(ETs))
    end_date = datetime_from_et(np.max(ETs))

    target_id = ''
    if mosaic_has_prometheus(metadata):
        target_id += """\
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
    if mosaic_has_pandora(metadata):
        target_id += """\
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

    sfx = '_bkg_sub' if bkgnd_sub else ''

            ###############################

    if GENERATE_MOSAIC_METADATA_TABLES:
        # mosaic_params.tab
        params_filename = f'{obs_id}_mosaic{sfx}_metadata_params.tab'
        xml_metadata['METADATA_PARAMS_TAB_FILENAME'] = params_filename
        output_path = os.path.join(mosaic_dir, params_filename)
        with open(output_path, 'w') as fp:
            fp.write('Corotating Longitude, Image Index, Mid-time SPICE ET, '
                     'Inertial Longitude, Resolution, Phase Angle, '
                     'Emission Angle\r\n')
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
                fp.write(row+'\r\n')
        with open(output_path, 'rb') as fp:
            hash = hashlib.md5(fp.read()).hexdigest();
        xml_metadata['METADATA_TABLE_HASH'] = hash

        # mosaic_params.tab
        src_imgs_filename = f'{obs_id}_mosaic{sfx}_metadata_src_imgs.tab'
        output_path = os.path.join(mosaic_dir, src_imgs_filename)
        xml_metadata['IMAGE_TAB_FILENAME'] = src_imgs_filename
        with open(output_path, 'w') as fp:
            fp.write('Source Image Index, LIDVID\r\n')
            for idx in range(len(image_name_list)):
                lidvid = image_name_list[idx] # XXX
                row = f'{idx:4d}, {lidvid}'
                fp.write(row+'\r\n')
        with open(output_path, 'rb') as fp:
            hash = hashlib.md5(fp.read()).hexdigest();
        xml_metadata['IMAGE_TABLE_HASH'] = hash

            ###############################

    if GENERATE_MOSAIC_METADATA_LABELS:
        if bkgnd_sub:
            xml_metadata['TITLE'] = f"""\
Metadata for background-subtracted F Ring Mosaics of reprojected Cassini ISS \
calibrated Images from observation {obs_id} from {start_date} to {end_date}\
"""
        else:
            xml_metadata['TITLE'] = f"""\
Metadata for F Ring Mosaics of reprojected Cassini ISS calibrated Images from \
observation {obs_id} from {start_date} to {end_date}\
"""
        xml_metadata['DESCRIPTION'] = xml_metadata['TITLE']
        xml_metadata['COMMENT'] = f"""\
Metadata for the mosaics.
Two files containing:
    1) Indices and LIDs of source images,
    2) Metadata parameters per corotating longitude,
for the mosaics of reprojected Cassini ISS Calibrated Images from {obs_id} \
from {start_date} to {end_date}.\
"""
        xml_metadata['RINGS:DESCRIPTION'] = xml_metadata['DESCRIPTION']

        output_path = os.path.join(mosaic_dir,
                                   f'{obs_id}_mosaic{sfx}_metadata.xml')
        populate_template('mosaic-metadata.xml', output_path, xml_metadata)


def generate_mosaic(obs_id, mosaic_dir, mosaic_metadata, bsm_metadata, bkgnd_metadata):
    if (GENERATE_MOSAIC_IMAGE or
        GENERATE_MOSAIC_IMAGE_LABELS or
        GENERATE_MOSAIC_METADATA_TABLES or
        GENERATE_MOSAIC_METADATA_LABELS):
        generate_mosaic_or_bsm(obs_id, mosaic_dir, mosaic_metadata)
        generate_mosaic_or_bsm(obs_id, mosaic_dir, bsm_metadata, bkgnd_sub=True)
    if GENERATE_MOSAIC_BROWSE_IMAGE or GENERATE_MOSAIC_BROWSE_LABELS:
        generate_mosaic_browse()


# SPACECRAFT_CLOCK_START_COUNT
# SPACECRAFT_CLOCK_STOP_COUNT


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
    reproj_dir = os.path.join(arguments.output_dir, 'data_reproj_imgs', obs_id)
    os.makedirs(reproj_dir, exist_ok=True)

    mosaic_metadata = read_mosaic(mosaic_path, mosaic_metadata_path)
    bsm_metadata = read_mosaic(bsm_path, bsm_metadata_path)
    bkgnd_metadata = read_bkgnd_metadata(bkgnd_model_path, bkgnd_metadata_path)

    if not all(obs_id == x for x in mosaic_metadata['obsid_list']):
        LOGGER.error(f'Not all mosaic OBSIDs are {obs_id}')
        return
    if not all(obs_id == x for x in bsm_metadata['obsid_list']):
        LOGGER.error(f'Not all background-sub mosaic OBSIDs are {obs_id}')
        return

    remap_image_indexes(mosaic_metadata)
    remap_image_indexes(bsm_metadata)

    if (GENERATE_REPROJ_IMAGE or GENERATE_REPROJ_LABELS or
        GENERATE_REPROJ_BROWSE_IMAGE or GENERATE_REPROJ_BROWSE_LABELS):
        generate_reproj()

    if (GENERATE_MOSAIC_IMAGE or GENERATE_MOSAIC_IMAGE_LABELS or
        GENERATE_MOSAIC_METADATA_TABLES or GENERATE_MOSAIC_METADATA_LABELS or
        GENERATE_MOSAIC_BROWSE_IMAGE or GENERATE_MOSAIC_BROWSE_LABELS):
        generate_mosaic(obs_id, mosaic_dir, mosaic_metadata, bsm_metadata, bkgnd_metadata)


NOW = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

BASIC_XML_METADATA = {
    'KEYWORD': 'saturn rings',
    'PUBLICATION_YEAR': '2023',
    'MODIFICATION_DATE': NOW[:10], # UTC
    'NOW': NOW, # UTC
    'MIN_RING_RADIUS': f'{arguments.ring_radius+arguments.radius_inner_delta:.0f}',
    'MAX_RING_RADIUS': f'{arguments.ring_radius+arguments.radius_outer_delta:.0f}'
}




# XXX Make bundle/collection files here
for obs_id in f_ring.enumerate_obsids(arguments):
    LOGGER.open(f'OBSID {obs_id}')
    handle_one_obsid(obs_id)
    LOGGER.close()
