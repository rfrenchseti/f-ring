# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research code for analyzing Saturn's F ring using mosaics built from Cassini ISS
and Voyager ISS images. The pipeline reprojects calibrated images into a co-rotating
ring frame, stitches them into longitude-vs-radius mosaics, fits and subtracts a
background, then derives photometry (equivalent widths, widths, clumps, gradients,
phase curves). The repo also generates a PDS4 archive bundle of those mosaics and
a PyQt6 viewer for browsing the bundle. There is no test suite; everything is run
as scripts/notebooks against on-disk data.

## Required environment

Most code requires these env vars (set before running anything that touches data):

- `FRING_DATA_ROOT` — root of the on-disk mosaic data (used by `f_ring_util.f_ring`).
  Subdirs are derived per ring type: `mosaic_<RING_TYPE>`, `bkgnd_<RING_TYPE>`,
  `bkgnd_sub_mosaic_<RING_TYPE>`, `ring_repro`, `png_polar_<RING_TYPE>`.
- `CB_SOURCE_ROOT`, `CB_RESULTS_ROOT` — required by the mosaic-building pipeline
  in `mosaics/ring/ring_util.py` (the rms-nav `nav.*` modules are pulled in).
- `FRING_BUNDLE_PATH` (optional) — bundle root for `pds4_mosaic_viewer`; falls
  back to `<repo>/pds4/bundle`.

`--ring-type` (default `FMOVIE`) selects the dataset. Common values:
`FMOVIE`, `FMOVIE_SENSITIVITY`, `FRING_VOYAGER1`, `FRING_VOYAGER2`. Each pairs
with a list under `mosaics/ring_mosaic_lists/FILELIST_*`.

## Repo layout (big picture)

The work splits into four largely independent pipelines that share `f_ring_util/`:

1. **Mosaic build** (`mosaics/`): per-image offset → reprojection → mosaic →
   background model → background-subtracted mosaic. Driven by the `ring_ui_*.py`
   scripts under `mosaics/`, which import the heavyweight `ring/ring_util.py`
   (depends on rms-nav `nav.*`). `ring_ui_toplevel.py` is the Tkinter GUI;
   `ring_ui_reproject.py`/`ring_ui_mosaic.py`/`ring_ui_bkgnd.py` are the per-stage
   batch entry points. Mosaic file paths are deterministic and encode the
   `ring_radius / inner_delta / outer_delta / radius_res / longitude_res /
   radial_zoom / longitude_zoom` parameters, so any consumer must pass the same
   args (see `mosaic_paths`/`bkgnd_paths`/`bkgnd_sub_mosaic_paths` in
   `f_ring_util/f_ring.py`).

2. **Photometry** (`photometry/`): `create_ews.py` reads
   background-subtracted mosaics and emits per-slice EW CSVs into `data_files/`
   (e.g. `cass_ew_0_1.csv`, `cass_ew_60_0.csv`, `v1_ew_0_1.csv`). `create_eds.py`
   produces equivalent-depth tables for occultations. The `update_all_ews_*.sh`
   scripts are the canonical invocations (Cassini/Voyager/sensitivity sweeps).
   Output CSVs are inputs to the analysis notebooks.

3. **PDS4 bundle generation** (`pds4_bundle_gen/`): `generate_pds4_files.py`
   ingests `observation_list.csv` and the mosaic/repro on-disk files and emits
   the full bundle (data, browse, document, miscellaneous, spice_kernels,
   xml_schema collections) under `pds4_bundle_gen/bundle/` using
   `rms-pdstemplate` against the `templates/*.lblx` files. The bundle name
   `cassini_iss_fring_mosaics_rsfrench2025` and collection LIDs are constants
   at the top of the script.

4. **PDS4 mosaic viewer** (`pds4_mosaic_viewer/`): PyQt6 GUI; entry point
   `pds4_mosaic_viewer.py`. Reads a generated bundle (no other dependencies on
   the build pipeline). `--show-radii` accepts negative values via the comma form
   `--show-radii=-100`; the script rewrites `--show-radii -100` to that form
   before argparse sees it.

Other directories: `notebooks/` (analysis Jupyter notebooks, numbered by topic
group: 01 phase functions, 11–13 calibration/coverage/sensitivity, 21–25
phase curves & Voyager comparison, 31–32 clumps/version-compare),
`sim/` (rebound-based collisional and channelized-stream simulations),
`clump/` (current and 2014-paper clump detection/tracking code; the
`2014-clump/` subdir is legacy),
`utilities/` (one-off tools — polar mosaics, mosaic display, geometry-matched
image finding, calibration cross-checks under `calibration/coiss_vgiss_cross_calib/`),
`external/gravity.py` (vendored), `archive/` (frozen older versions —
do not edit), `f_ring_util/` (shared utilities; `f_ring.py` for paths and ring
geometry, `moons.py` for Prometheus/Pandora positions),
`papers_posters/` (LaTeX/figure assets), `observation_lists/` (master CSVs
of Cassini and Voyager observations), `data_files/` (small CSVs that are
inputs/outputs of the photometry stage and inputs to notebooks).

`mosaics/ring/` is a *different* `ring_util.py` from `f_ring_util/f_ring.py` —
the former is the heavyweight mosaic-build version (depends on rms-nav and
operates on the live `MosaicData`/`OffRepData`/`BkgndData` objects), the latter
is the lightweight reader used by photometry, PDS4 generation, and notebooks.
Don't conflate them.

## Common commands

```bash
# Install (uses pip-tools; requirements.txt is compiled from requirements.in)
pip install -r requirements.txt

# Build mosaics for FMOVIE (default ring type) — top-level Tkinter GUI
python mosaics/ring_ui_toplevel.py

# Per-stage batch (see create_sensitivity_mosaics.sh for the canonical sweep)
python mosaics/ring_ui_reproject.py --ring-type FMOVIE --verbose
python mosaics/ring_ui_mosaic.py    --ring-type FMOVIE
python mosaics/ring_ui_bkgnd.py     --ring-type FMOVIE

# Generate the standard EW CSVs into data_files/
cd photometry && ./update_all_ews_cassini.sh    # Cassini, slice=0/60
cd photometry && ./update_all_ews_voyager.sh    # Voyager 1 and 2

# PDS4 bundle generation
python pds4_bundle_gen/generate_pds4_files.py

# Mosaic viewer (PyQt6) — reads $FRING_BUNDLE_PATH or pds4/bundle by default
python pds4_mosaic_viewer/pds4_mosaic_viewer.py [--bundle-path PATH] \
    [--no-bkg-sub] [--obsid OBSID ...] [mosaic_name ...]
```

Add a single mosaic to the batch by passing its OBSID positionally to any of
the `ring_ui_*.py` scripts; they share `f_ring.add_parser_arguments` /
`f_ring.enumerate_obsids` for OBSID selection.

## When editing

- The `mosaic_paths` / `bkgnd_paths` filename templates encode all geometry
  args. Don't change a default (`ring_radius`, `*_delta`, `*_resolution`,
  `*_zoom_amount`) without realizing all derived files become unreadable —
  callers must pass the matching args.
- `f_ring_util.f_ring.init(arguments)` mutates module-level globals
  (`MOSAIC_DIR`, etc.) based on `--ring-type`. Always call it before any path
  helper. Switching ring types mid-run requires re-`init`.
- The PDS4 templates (`pds4_bundle_gen/templates/*.lblx`) are RMS-managed in
  parts and locally generated in others — `generate_pds4_files.py` has comments
  marking which is which (`[RMS]` vs `[generated]`).
- Notebooks under `archive/notebooks.old/` and the `clump/2014-clump/`
  directory are frozen; prefer the top-level `notebooks/` and `clump/*.py`
  for current work.
