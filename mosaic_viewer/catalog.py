"""Mosaic catalog: loads the global index and provides filtering/sorting."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Optional

from pds4_reader import read_index_np


@dataclass
class FilterCriteria:
    """Filter criteria for mosaic selection."""
    nav_quality_g: bool = True
    nav_quality_f: bool = True
    nav_quality_p: bool = True
    bkgnd_quality_g: bool = True
    bkgnd_quality_f: bool = True
    bkgnd_quality_p: bool = True
    min_radial_res: Optional[float] = None
    max_radial_res: Optional[float] = None
    min_long_res: Optional[float] = None
    max_long_res: Optional[float] = None
    min_prometheus_dist: Optional[float] = None
    max_prometheus_dist: Optional[float] = None
    min_pandora_dist: Optional[float] = None
    max_pandora_dist: Optional[float] = None

    def copy(self) -> FilterCriteria:
        return copy.deepcopy(self)


@dataclass
class MosaicRecord:
    """One entry from the global mosaic index."""
    name: str           # observation dir name, e.g. iss_029rf_fmovie001_vims
    obsid: str          # Cassini OBSID, e.g. ISS_029RF_FMOVIE001_VIMS
    file_spec: str      # relative path to .lblx from bundle root
    start_datetime: str  # ISO 8601 string for sorting
    nav_quality: str    # G / F / P
    bkgnd_quality: str  # G / F / P / '' (empty for non-bkg-sub)
    radial_res: float
    long_res: float
    prometheus_dist: float  # |mean core − min Prometheus orbital r| (km)
    pandora_dist: float     # |mean core − max Pandora orbital r| (km)
    notes: str = ''         # free-text from global index ``notes`` column


class MosaicCatalog:
    """Shared catalog of all available mosaics, loaded once.

    Passed by reference to every MosaicWindow.  Each window applies its own
    FilterCriteria to obtain a filtered list.
    """

    def __init__(
        self,
        bundle_path: str,
        bkg_sub: bool = True,
        name_filter: Optional[list[str]] = None,
        obsid_filter: Optional[list[str]] = None,
        start_obsid: str = '',
        end_obsid: str = '',
    ) -> None:
        self.bundle_path = bundle_path
        self.bkg_sub = bkg_sub
        self._records: list[MosaicRecord] = []
        self._load_index(
            bundle_path, bkg_sub,
            name_filter, obsid_filter,
            start_obsid, end_obsid,
        )

    def _load_index(
        self,
        bundle_path: str,
        bkg_sub: bool,
        name_filter: Optional[list[str]],
        obsid_filter: Optional[list[str]],
        start_obsid: str,
        end_obsid: str,
    ) -> None:
        misc_dir = os.path.join(bundle_path, 'miscellaneous')
        index_file = (
            'global_mosaic_bkg_sub_index.lblx'
            if bkg_sub else
            'global_mosaic_index.lblx'
        )
        index_label = os.path.join(misc_dir, index_file)
        table_id = (
            'global_mosaic_bkg_sub_index' if bkg_sub else 'global_mosaic_index'
        )
        data = read_index_np(index_label, table_id)

        name_filter_lower = (
            {n.lower() for n in name_filter} if name_filter else None
        )
        obsid_filter_upper = (
            {o.upper() for o in obsid_filter} if obsid_filter else None
        )

        for row in data:
            obsid = str(row['cassini_observation_id']).strip()
            file_spec = str(row['file_spec']).strip()

            # Derive observation directory name from file_spec path.
            # e.g. 'data_mosaic_bkg_sub/iss_029rf_.../iss_029rf_..._mosaic.lblx'
            parts = file_spec.replace('\\', '/').split('/')
            obs_name = parts[1] if len(parts) >= 2 else obsid.lower()

            if name_filter_lower and obs_name.lower() not in name_filter_lower:
                continue
            if obsid_filter_upper and obsid.upper() not in obsid_filter_upper:
                continue
            if start_obsid and obsid.upper() < start_obsid.upper():
                continue
            if end_obsid and obsid.upper() > end_obsid.upper():
                continue

            nav_q = str(row['nav_quality']).strip()
            bkgnd_q = (
                str(row['bkgnd_quality']).strip() if bkg_sub else ''
            )

            mean_core = row['mean_core_radius']
            min_prom_moon_r = row['minimum_radius_prometheus']
            max_pand_moon_r = row['maximum_radius_pandora']
            prom_sep = abs(mean_core - min_prom_moon_r)
            pand_sep = abs(mean_core - max_pand_moon_r)

            notes_s = str(row['notes']).strip()

            rec = MosaicRecord(
                name=obs_name,
                obsid=obsid,
                file_spec=file_spec,
                start_datetime=str(row['pds_start_date_time']).strip(),
                nav_quality=nav_q,
                bkgnd_quality=bkgnd_q,
                radial_res=row['rings_mean_radial_resolution'],
                long_res=row['rings_mean_longitudinal_resolution'],
                prometheus_dist=prom_sep,
                pandora_dist=pand_sep,
                notes=notes_s,
            )
            self._records.append(rec)

        self._records.sort(key=lambda r: r.start_datetime)

    def filter(self, criteria: FilterCriteria) -> list[MosaicRecord]:
        """Return records matching the filter criteria (datetime-sorted)."""
        allowed_nav = set()
        if criteria.nav_quality_g:
            allowed_nav.add('G')
        if criteria.nav_quality_f:
            allowed_nav.add('F')
        if criteria.nav_quality_p:
            allowed_nav.add('P')

        allowed_bkgnd = set()
        if criteria.bkgnd_quality_g:
            allowed_bkgnd.add('G')
        if criteria.bkgnd_quality_f:
            allowed_bkgnd.add('F')
        if criteria.bkgnd_quality_p:
            allowed_bkgnd.add('P')

        def _check(val: Optional[float], lo: Optional[float], hi: Optional[float]) -> bool:
            if lo is not None and val < lo:
                return False
            if hi is not None and val > hi:
                return False
            return True

        result = []
        for rec in self._records:
            if rec.nav_quality not in allowed_nav:
                continue
            if self.bkg_sub and rec.bkgnd_quality and rec.bkgnd_quality not in allowed_bkgnd:
                continue
            if not _check(rec.radial_res, criteria.min_radial_res, criteria.max_radial_res):
                continue
            if not _check(rec.long_res, criteria.min_long_res, criteria.max_long_res):
                continue
            if not _check(rec.prometheus_dist, criteria.min_prometheus_dist, criteria.max_prometheus_dist):
                continue
            if not _check(rec.pandora_dist, criteria.min_pandora_dist, criteria.max_pandora_dist):
                continue
            result.append(rec)

        return result

    def all_records(self) -> list[MosaicRecord]:
        return list(self._records)

    def label_path(self, record: MosaicRecord) -> str:
        """Return the absolute path to a mosaic's .lblx file."""
        return os.path.join(self.bundle_path, record.file_spec)
