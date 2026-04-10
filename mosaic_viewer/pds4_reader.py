"""PDS4 data loading for F ring mosaic bundles."""
from __future__ import annotations

import contextlib
import os
from typing import Any

import numpy as np
import numpy.ma as ma
import pds4_tools


def _pds4_read(label_path: str) -> Any:
    """Read a PDS4 label; suppress pds4_tools logging and any stray stdio."""
    with open(os.devnull, 'w', encoding='utf-8') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return pds4_tools.read(label_path, quiet=True)


def get_element(label, element_name: str, return_type=float):
    """Return the value of the given element from a PDS4 label."""
    element = label.find(f'.//{element_name}')
    if element is None:
        raise ValueError(f"Element '{element_name}' not found in label")
    return return_type(element.text)


def _extract_image_ma(pkg_data, struct_name: str) -> tuple[ma.MaskedArray, object]:
    """Extract a masked array image from a pds4_tools structure."""
    image = pkg_data[struct_name].data
    image_meta = image.meta_data
    image = ma.masked_array(np.ascontiguousarray(image))
    sentinel = None
    try:
        sentinel = image_meta['Special_Constants']['missing_constant']
    except (KeyError, TypeError):
        pass
    image_ma = ma.masked_equal(image, sentinel) if sentinel is not None else image
    return image_ma, image_meta


def read_mosaic_ma(
    mosaic_label_path: str,
    include_image_table: bool = False,
) -> tuple:
    """Read a mosaic from a PDS4 .lblx file.

    Returns:
        (label, image_ma, metadata_params) or
        (label, image_ma, metadata_params, image_table_dict) if
        include_image_table is True.
    """
    pkg = _pds4_read(mosaic_label_path)
    label = pkg.label
    image_ma, _ = _extract_image_ma(pkg, 'mosaic')
    metadata_params = pkg['metadata_params'].data

    if not include_image_table:
        return (label, image_ma, metadata_params)

    img_tbl = pkg['image_table'].data
    img_table_dict = {
        int(x['image_index']): str(x['LIDVID']).strip()
        for x in img_tbl
    }
    return (label, image_ma, metadata_params, img_table_dict)


def read_reproj_img_ma(reproj_img_label_path: str) -> tuple:
    """Read a reprojected image from a PDS4 .lblx file.

    The returned image is expanded to full 360-degree width (same coordinate
    system as the parent mosaic), with most pixels masked.

    Returns:
        (label, full_image_ma, metadata_params)
    """
    pkg = _pds4_read(reproj_img_label_path)
    label = pkg.label
    image_ma, _ = _extract_image_ma(pkg, 'reproj_image')
    metadata_params = pkg['metadata_params'].data

    long_interval = get_element(
        label, 'rings:reprojection_grid_longitudinal_sampling_interval')
    num_long = int(np.round(360.0 / long_interval))
    min_corot = get_element(label, 'rings:minimum_corotating_ring_longitude')
    max_corot = get_element(label, 'rings:maximum_corotating_ring_longitude')
    min_idx = int(np.round(min_corot / long_interval))
    max_idx = int(np.round(max_corot / long_interval))

    full_image = ma.masked_all((image_ma.shape[0], num_long))
    if min_idx <= max_idx:
        full_image[:, min_idx:max_idx + 1] = image_ma
    else:
        # Wrap-around case
        slice1 = num_long - min_idx
        full_image[:, min_idx:] = image_ma[:, :slice1]
        full_image[:, :max_idx + 1] = image_ma[:, slice1:]

    return (label, full_image, metadata_params)


def read_index_np(global_index_label_path: str, table_local_id: str) -> np.ndarray:
    """Read a PDS4 index table as a structured numpy array.

    ``table_local_id`` must match the label (e.g. ``global_mosaic_index`` or
    ``global_mosaic_bkg_sub_index``).
    """
    pkg = _pds4_read(global_index_label_path)
    data = pkg[table_local_id].data
    for name, (dt, _) in data.dtype.fields.items():
        if dt.kind in 'SU':
            data[name] = np.char.strip(data[name])
    return data


def get_mosaic_name_from_lid(lid: str) -> str:
    """Extract observation name from a mosaic LID string."""
    return lid.split(':')[-1].split('_mosaic')[0]


def get_mosaic_name_from_mosaic_label(label) -> str:
    """Extract observation name from a loaded mosaic label."""
    lid = get_element(label, 'logical_identifier', return_type=str)
    return get_mosaic_name_from_lid(lid)


def reproj_product_stem_from_label(label) -> str:
    """Last segment of reproj logical id, e.g. ``...:1538168640n_reproj_img``."""
    lid = get_element(label, 'logical_identifier', return_type=str)
    return lid.split(':')[-1]


def get_mosaic_name_from_reproj_img_label(label) -> str:
    """Mosaic OBSID from reproj label ``data_to_derived_product`` reference."""
    ref = label.find(
        './/Internal_Reference[reference_type="data_to_derived_product"]/'
        'lid_reference')
    if ref is None or not ref.text:
        raise ValueError(
            'reproj label missing Internal_Reference '
            '[reference_type="data_to_derived_product"]/lid_reference')
    return get_mosaic_name_from_lid(ref.text)


def lidvid_to_reproj_name(lidvid: str) -> str:
    """Extract reprojected image name from a LIDVID string.

    E.g. 'urn:...:1538168640n_reproj_img::1.0' → '1538168640n_reproj_img'
    """
    lid = lidvid.split('::')[0]
    return lid.split(':')[-1]


def compute_default_stretch(
    image_ma: ma.MaskedArray,
    white_point_ignore_frac: float = 0.005,
) -> tuple[float, float]:
    """Compute default black and white points from the image data."""
    valid = ma.compressed(image_ma)
    if valid.size == 0:
        return 0.0, 1.0
    black = max(float(np.min(valid)), 0.0)
    k = int(np.clip(valid.size * (1.0 - white_point_ignore_frac), 0, valid.size - 1))
    white = float(np.partition(valid, k)[k])
    if white <= black:
        white = black + 1e-6
    return black, white
