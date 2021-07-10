import numpy as np
import os
import pickle

# Given a directory, this returns a list of all the filenames in that directory
# that end with "-METADATA.dat", but with that suffix stripped. We call that
# stripped filename a "root" filename.
def get_root_list(dir):
    root_list = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith('-METADATA.dat'):
                root_list.append(filename.replace('-METADATA.dat', ''))
    root_list.sort()
    return root_list

# Given a root filename, read and return the mosaic data file.
def read_mosaic(dir, root):
    return np.load(os.path.join(dir, root+'-MOSAIC.npy'))

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
#   longitudes              The inertial longitude (in radians) in the source
#                           image for each co-rotating longitude.
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
def read_metadata(dir, root):
    with open(os.path.join(dir, root+'-METADATA.dat'), 'rb') as fp:
        metadata = pickle.load(fp, encoding='latin1')
    return metadata

# Return the valid portion of a mosaic.
def valid_mosaic_subset(mosaic, meadata):
    valid_longitudes = get_valid_longitudes(mosaic, metadata):
    lower_limit = metadata['ring_lower_limit']
    upper_limit = metadata['ring_upper_limit']
    return mosaic[lower_limit:upper_limit+1, valid_longitudes]

# Given a mosaic, figure out which longitudes have valid data in ALL radial
# locations that are outside of the radii used to compute the background
# gradient. This returns a 1-D boolean array of longitudes where True means
# that longitude has valid data.
def get_valid_longitudes(mosaic, metadata):
    lower_limit = metadata['ring_lower_limit']
    upper_limit = metadata['ring_upper_limit']
    return np.all(mosaic[lower_limit:upper_limit+1, :], axis=0)
