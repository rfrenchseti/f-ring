"""Utility functions for working with files in the PDS4 bundle
cassini_iss_fring_mosaics_rsfrench2025.
"""

import numpy as np
import numpy.ma as ma
import pds4_tools  # SBN label/table reader


################################################################################
#                                                                              #
# General utilities                                                            #
#                                                                              #
################################################################################

def get_element(label, element_name, return_type=float):
    """Return the value of the given element anywhere in the label.

    Searches for the element name in all child elements of the label and
    returns the value of the first element found, coerced to the specified type
    (defaults to float).

    Arguments:
        label (pds4_tools.reader.label_objects.Label): The PDS4 label structure.
        element_name (str): The name of the element to return.
        return_type (type, optional): The type to return for the value, as
            expressed as a function that converts a string to the desired type.
            Defaults to float.

    Returns:
        return_type: The value of the element as the specified type.

    Raises:
        ValueError: If the element is not found in the label.
    """
    element = label.find(f'.//{element_name}')
    if element is None:
        raise ValueError(f"Element {element_name} not found in label")
    return return_type(element.text)


def contrast_stretch(image_ma_data, black_point=None, white_point=None,
                     gamma=0.5, white_point_ignore_frac=0.02):
    """Contrast-stretch the image data so it's easier to see faint features.

    Arguments:
        image_ma_data (numpy.ma.MaskedArray): The image data to
            contrast-stretch.
        black_point (float, Optional): The black point to use. Defaults to the
            minimum value of the image data.
        white_point (float, Optional): The white point to use. Defaults to the
            value of the image data at the white point ignore fraction.
        white_point_ignore_frac (float, Optional): The image's upper tail
            fraction to ignore when determining the white point. Set to 0.0
            to force the white point to be the maximum value of the image
            data. However, this is not recommended because stray bright pixels,
            stars, or moons will skew the white point of the faint ring.
        gamma (float, Optional): The gamma to use.

    Returns:
        numpy.ma.MaskedArray: The contrast-stretched image data. The stretch is
        performed in the original data units and remains in floating point, with
        a range of [0, 1]. Note that unless gamma=1 the stretch is nonlinear.
    """
    if black_point is None:
        black_point = max(ma.min(image_ma_data), 0)
    if white_point is None:
        if white_point_ignore_frac == 0.0:
            white_point = ma.max(image_ma_data)
        else:
            vals = ma.compressed(image_ma_data)
            if vals.size == 0:
                white_point = black_point
            else:
                k = int(vals.size * (1 - white_point_ignore_frac))
                k = int(np.clip(k, 0, vals.size - 1))
                white_point = np.partition(vals, k)[k]
    if black_point == white_point:
        raise ValueError('Black point and white point are the same')
    return np.clip((ma.maximum(image_ma_data-black_point, 0) /
                    (white_point-black_point))**gamma, 0, 1)


################################################################################
#                                                                              #
# Read and manipulate mosaics                                                  #
#                                                                              #
################################################################################

def read_mosaic_ma(mosaic_label_path, include_image_table=False):
    """Read a mosaic as a masked_array, with metadata as a numpy array.

    Returns the full width and height mosaic image as a masked array. Any
    invalid or missing data is masked.

    Note that the pds4_tools module replaces ``:`` in field names with ``_``.
    Thus, a column name like ``rings:corotating_ring_longitude`` will be
    converted to ``rings_corotating_ring_longitude`` in the returned metadata
    parameters.

    Arguments:
        mosaic_label_path (str): The path to the mosaic label file.
        include_image_table (bool, Optional): Whether to include the image table
            in the returned tuple. Defaults to False. This is provided for
            efficiency since most uses will not require the image table.

    Returns:
        tuple: A tuple containing: 1) the label structure for the entire label,
        2) the mosaic image data (as a ma.masked_array, index 0 is radius, index
        1 is longitude), and 3) the metadata parameters as a structured numpy
        array. If include_image_table is True, then 4) the mapping of image
        numbers to image LIDVIDs as a dict.
    """
    # Read the mosaic label and all associated images and tables.
    mosaic_pkg = pds4_tools.read(mosaic_label_path)
    label = mosaic_pkg.label  # Label structure for the entire label.

    image = mosaic_pkg['mosaic'].data  # Image data as a PDS_ndarray array.
    image_meta_data = image.meta_data

    # Convert from the PDS_marray returned by read() to a standard numpy masked
    # array. We do this because some of the indexing behavior of PDS_ndarrays
    # is different from that of numpy masked arrays and we want to provide the
    # user with a standard interface. Also the returned image may not be
    # contiguous, which makes future slicing difficult.
    image = ma.masked_array(np.ascontiguousarray(image))

    # Find the "missing constant" and convert every instance of it in the
    # image to a masked value.
    sentinel = None
    try:
        sentinel = image_meta_data['Special_Constants']['missing_constant']
    except KeyError:
        pass
    if sentinel is None:
        image_ma_data = image
    else:
        image_ma_data = ma.masked_equal(image, sentinel)

    # Retrieve the metadata params and (maybe) image number tables.
    metadata_params = mosaic_pkg['metadata_params'].data

    if not include_image_table:
        return (label, image_ma_data, metadata_params)

    mosaic_img_table = mosaic_pkg['image_table'].data

    mosaic_img_table_dict = {int(x['image_index']): str(x['LIDVID']).strip()
                             for x in mosaic_img_table}
    return (label, image_ma_data, metadata_params, mosaic_img_table_dict)


def read_mosaic_ma_df(mosaic_label_path, include_image_table=False):
    """Read a mosaic as a masked_array, with metadata as a pandas DataFrame.

    This is the same as read_mosaic_ma, except that the metadata is converted to
    a pandas DataFrame.

    Note that the pds4_tools module replaces ``:`` in field names with ``_``.
    Thus, a column name like ``rings:corotating_ring_longitude`` will be
    converted to ``rings_corotating_ring_longitude`` in the returned metadata
    parameters.

    Arguments:
        mosaic_label_path (str): The path to the mosaic label file.
        include_image_table (bool, Optional): Whether to include the image table
            in the returned tuple. Defaults to False. This is provided for
            efficiency since most uses will not require the image table.

    Returns:
        tuple: A tuple containing: 1) the label structure for the entire label,
        2) the mosaic image data (as a ma.masked_array, index 0 is radius, index
        1 is longitude), and 3) the metadata parameters (as a pandas DataFrame).
        If include_image_table is True, then 4) the mapping of image numbers to
        image LIDVIDs as a dict.
    """
    # We import pandas here to avoid requiring it as a dependency for users
    # who don't want to use pandas.
    import pandas as pd

    (label, image_ma_data, metadata_params,
     *other) = read_mosaic_ma(mosaic_label_path,
                              include_image_table=include_image_table)

    metadata_params_df = pd.DataFrame(metadata_params)

    return (label, image_ma_data, metadata_params_df, *other)


def get_mosaic_name_from_lid(mosaic_lid):
    """Get the name of the mosaic from the LID.

    This relies on the format for the LID:
        urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:data_mosaic_bkg_sub:
        <OBSID>_mosaic[_bkg_sub]

    Arguments:
        mosaic_lid (str): The LID of the mosaic.

    Returns:
        str: The name of the mosaic.
    """
    return mosaic_lid.split(':')[-1].split('_mosaic')[0]


def get_mosaic_name_from_mosaic_label(mosaic_label):
    """Get the name of the mosaic from the label.

    This relies on the format for the LID:
        urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:data_mosaic_bkg_sub:
        <OBSID>_mosaic[_bkg_sub]

    Arguments:
        mosaic_label (pds4_tools.reader.label_objects.Label): The PDS4 label
            for the mosaic.

    Returns:
        str: The name of the mosaic.
    """
    lid = get_element(mosaic_label, 'logical_identifier', return_type=str)
    return get_mosaic_name_from_lid(lid)


################################################################################
#                                                                              #
# Read and manipulate reprojected images                                       #
#                                                                              #
################################################################################

def read_reproj_img_ma(reproj_img_label_path):
    """Read a reproj image as a masked_array, with metadata as a numpy array.

    Returns the full width and height reprojected image as a masked array. Any
    invalid or missing data is masked. The reprojected image is the same full
    size as a mosaic and may be extremely sparse if the reprojection takes up
    only a small range of corotating longitudes.

    Note that the pds4_tools module replaces ``:`` in field names with ``_``.
    Thus, a column name like ``rings:corotating_ring_longitude`` will be
    converted to ``rings_corotating_ring_longitude`` in the returned metadata
    parameters.

    Arguments:
        reproj_img_label_path (str): The path to the reprojected image label
            file.

    Returns:
        tuple: A tuple containing: 1) the label structure for the entire label,
        2) the reproj image data (as a ma.masked_array, index 0 is radius, index
        1 is longitude), and 3) the metadata parameters as a structured numpy
        array.
    """
    # Read the reprojected image label and all associated images and tables.
    reproj_img_pkg = pds4_tools.read(reproj_img_label_path)
    label = reproj_img_pkg.label  # Label structure for the entire label.

    # Image data as a PDS_ndarray array.
    image = reproj_img_pkg['reproj_image'].data
    image_meta_data = image.meta_data

    # Convert from the PDS_marray returned by read() to a standard numpy masked
    # array. We do this because some of the indexing behavior of PDS_ndarrays
    # is different from that of numpy masked arrays and we want to provide the
    # user with a standard interface. Also the returned image may not be
    # contiguous, which makes future slicing difficult.
    image = ma.masked_array(np.ascontiguousarray(image))

    # Find the "missing constant" and convert every instance of it in the
    # image to a masked value.
    sentinel = None
    try:
        sentinel = image_meta_data['Special_Constants']['missing_constant']
    except KeyError:
        pass
    if sentinel is None:
        image_ma_data = image
    else:
        image_ma_data = ma.masked_equal(image, sentinel)

    metadata_params = reproj_img_pkg['metadata_params'].data

    # Unlike mosaics, reprojected images do not fill the entire image with data,
    # because of the amount of disk space that would be required. Instead,
    # they are limited to the extent of the minimum and maximum valid corotating
    # longitudes, possibly with wrapping around at 360.
    # Here we take the part of the image provided and place it in the correct
    # location in a full-width image.
    long_interval = get_element(
        label, 'rings:reprojection_grid_longitudinal_sampling_interval')
    num_long = int(np.round(360 / long_interval))
    min_corot_long = get_element(
        label, 'rings:minimum_corotating_ring_longitude')
    min_corot_long_idx = int(np.round(min_corot_long / long_interval))
    max_corot_long = get_element(
        label, 'rings:maximum_corotating_ring_longitude')
    max_corot_long_idx = int(np.round(max_corot_long / long_interval))
    full_reproj_img_ma_data = ma.masked_all((image_ma_data.shape[0], num_long))
    if min_corot_long_idx < max_corot_long_idx:
        # If min < max, there is no wraparound so we can just put the image
        # in the correct spot.
        full_reproj_img_ma_data[:, min_corot_long_idx:max_corot_long_idx+1] = \
            image_ma_data
    else:
        # If min > max, there is wraparound, so we need to put the image in
        # the correct spot and then wrap around the end of the image to the
        # beginning.
        slice1_size = num_long - min_corot_long_idx
        image_ma_data_slice1 = image_ma_data[:, :slice1_size]  # Min -> 360
        image_ma_data_slice2 = image_ma_data[:, slice1_size:]  #   0 -> Max
        full_reproj_img_ma_data[:, min_corot_long_idx:] = image_ma_data_slice1
        full_reproj_img_ma_data[:, :max_corot_long_idx+1] = image_ma_data_slice2

    return (label, full_reproj_img_ma_data, metadata_params)


def read_reproj_img_ma_df(reproj_img_label_path):
    """Read a reproj image as a masked_array, with metadata as a pandas
    DataFrame.

    This is the same as read_reproj_img_ma, except that the metadata is
    converted to a pandas DataFrame.

    Note that the pds4_tools module replaces ``:`` in field names with ``_``.
    Thus, a column name like ``rings:corotating_ring_longitude`` will be
    converted to ``rings_corotating_ring_longitude`` in the returned metadata
    parameters.

    Arguments:
        reproj_img_label_path (str): The path to the reprojected image label
            file.

    Returns:
        tuple: A tuple containing: 1) the label structure for the entire label,
        2) the reproj image data (as a ma.masked_array, index 0 is radius, index
        1 is longitude), and 3) the metadata parameters as a pandas DataFrame.
    """
    # We import pandas here to avoid requiring it as a dependency for users
    # who don't want to use pandas.
    import pandas as pd

    (label, image_ma_data,
     metadata_params) = read_reproj_img_ma(reproj_img_label_path)

    metadata_params_df = pd.DataFrame(metadata_params)

    return (label, image_ma_data, metadata_params_df)


def get_reproj_img_name_from_lid(reproj_img_lid):
    """Get the name of the reprojected image from the LID.

    This relies on the format for the LID:
        urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:data_reproj_img:
        <OBSID>_reproj_img

    Arguments:
        reproj_img_lid (str): The LID of the reprojected image.

    Returns:
        str: The name of the reprojected image.
    """
    return reproj_img_lid.split(':')[-1].split('_reproj_img')[0]


def get_reproj_img_name_from_label(reproj_img_label):
    """Get the name of the reprojected image from the label.

    This relies on the format for the LID:
        urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:data_reproj_img:
        <OBSID>_reproj_img

    Arguments:
        reproj_img_label (pds4_tools.reader.label_objects.Label): The PDS4 label
            for the reprojected image.

    Returns:
        str: The name of the reprojected image.
    """
    lid = get_element(reproj_img_label, 'logical_identifier', return_type=str)
    return get_reproj_img_name_from_lid(lid)


def get_mosaic_name_from_reproj_img_label(label):
    """Mosaic OBSID string from the reproj label's data_to_derived reference."""
    ref = label.find(
        './/Internal_Reference[reference_type="data_to_derived_product"]/'
        'lid_reference')
    if ref is None or not ref.text:
        return 'Unknown'
    return get_mosaic_name_from_lid(ref.text)


################################################################################
#                                                                              #
# Read and manipulate global indices                                           #
#                                                                              #
################################################################################

def read_index_np(global_index_label_path):
    """Read a reprojected image or mosaic global index as a numpy array.

    Note that the pds4_tools module replaces ``:`` in field names with ``_``.
    Thus, a column name like ``rings:corotating_ring_longitude`` will be
    converted to ``rings_corotating_ring_longitude`` in the returned metadata
    parameters.

    Arguments:
        global_index_label_path (str): Path to the global index label file.

    Returns:
        numpy.ndarray: Structured array of one row per reprojected image or
        mosaic, as appropriate.
    """
    index_pkg = pds4_tools.read(global_index_label_path)

    # We don't know if this is a reprojected image index, normal mosaic index,
    # or background-subtracted mosaic index, so just try each of them in order.
    for index_structure_id in ['global_reproj_img_index',
                               'global_mosaic_index',
                               'global_mosaic_bkg_sub_index']:
        try:
            data = index_pkg[index_structure_id].data
        except KeyError:
            continue
        # Fields such as LID and filespec may have trailing spaces, so strip
        # them
        for name, (dt, _) in data.dtype.fields.items():
            if dt.kind in 'SU':
                data[name] = np.char.strip(data[name])
        return data

    raise ValueError(f'No index structure found in {global_index_label_path}')


def read_index_df(global_index_label_path):
    """Read a reprojected image or mosaic global index as a pandas DataFrame.

    This is the same as read_index_np, except that the data is converted to a
    pandas DataFrame.

    Note that the pds4_tools module replaces ``:`` in field names with ``_``.
    Thus, a column name like ``rings:corotating_ring_longitude`` will be
    converted to ``rings_corotating_ring_longitude`` in the returned metadata
    parameters.

    Arguments:
        global_index_label_path (str): Path to the global index label file.

    Returns:
        pandas.DataFrame: One row per reprojected image or mosaic, as
        appropriate.
    """
    import pandas as pd

    return pd.DataFrame(read_index_np(global_index_label_path))
