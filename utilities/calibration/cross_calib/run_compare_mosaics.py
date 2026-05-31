#!/usr/bin/env python3
"""Run compare_mosaics.py for every matched image pair in a matches CSV.

Each row in the matches CSV (output of find_geometry_matched_images.py)
pairs one image from "set1" with one image from "set2". This script knows
how to map each row onto the two on-disk reprojected FITS files and pipe
them through compare_mosaics.py, capturing the JSON statistics.

Two dataset "kinds" are currently understood:

  coiss   reprojected COISS file named  <MOON>_<STEM>_CALIB_reproj.fits
          where <STEM> is the image name from setN_file (e.g.
          N1597244970_3 or W1572256382_1).
  vgiss   reprojected VGISS file named  <MOON>_<ID>_GEOMED_reproj.fits
          where <ID> is the last "-" component of setN_opus_id, uppercased
          (e.g. C3494502).

Use cases:

  COISS vs VGISS cross-cal (default):
    --set1-kind coiss --set2-kind vgiss
    --data-root /seti/research/f-ring/calibration/coiss_vgiss_cross_calib
    (set1 files in <root>/coiss/, set2 in <root>/vgiss/)

  COISS NAC vs COISS WAC cross-cal:
    --set1-kind coiss --set2-kind coiss
    --data-root /seti/research/f-ring/calibration/coiss_nac_vs_wacc_calibration
    --set1-dir <root> --set2-dir <root>      # both kinds share one dir

The output JSON is named results/<moon>_<id1>_<id2>.json where idN is the
short opus_id form for that kind (e.g. n1597244970, c3494502, w1572256382).

Usage:
  python run_compare_mosaics.py [--dry-run] [--matches PATH] [--results-dir DIR]
                                [--photometry MODE] [--max-pixel-incidence DEG]
                                [--data-root PATH]
                                [--set1-kind {coiss,vgiss}] [--set1-dir DIR]
                                [--set2-kind {coiss,vgiss}] [--set2-dir DIR]
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

COMPARE_MOSAICS = "/seti/newnav/rms-nav/src/experiments/compare_mosaics.py"

# Defaults match the COISS-vs-VGISS cross-cal layout, which was the
# original use case for this script.
DEFAULT_DATA_ROOT = Path("/seti/research/f-ring/calibration/coiss_vgiss_cross_calib")

SCRIPT_DIR = Path(__file__).parent
DEFAULT_MATCHES = SCRIPT_DIR / "matches.csv"
DEFAULT_RESULTS = SCRIPT_DIR / "results"

# Per-kind lookup. ``stem_source`` says how to derive the on-disk image
# stem (the part between MOON_ and the suffix in the FITS filename) from
# a matches CSV row, and ``id_source`` says how to derive the short opus
# ID used in the output JSON filename.
KINDS = {
    "coiss": {
        "subdir":      "coiss",
        "fits_suffix": "_CALIB_reproj.fits",
        "stem_source": "file",          # Path(setN_file).stem (keeps version)
        "id_strip":    "co-iss-",       # opus_id minus this prefix, lowercased
    },
    "vgiss": {
        "subdir":      "vgiss",
        "fits_suffix": "_GEOMED_reproj.fits",
        "stem_source": "opus_last",     # last "-" component, uppercased
        "id_strip":    None,            # use last "-" component, lowercased
    },
}


def stem_for_set(row: dict, side: int, kind_meta: dict) -> str:
    """Return the on-disk image stem for one side of a matches-CSV row."""
    src = kind_meta["stem_source"]
    if src == "file":
        return Path(row[f"set{side}_file"]).stem
    if src == "opus_last":
        return row[f"set{side}_opus_id"].split("-")[-1].upper()
    raise ValueError(f"unknown stem_source: {src!r}")


def id_for_set(row: dict, side: int, kind_meta: dict) -> str:
    """Return the short image ID used in the output JSON filename."""
    opus = row[f"set{side}_opus_id"]
    strip = kind_meta["id_strip"]
    if strip is not None and opus.startswith(strip):
        return opus[len(strip):].lower()
    return opus.split("-")[-1].lower()


def fits_path(set_dir: Path, moon: str, stem: str, kind_meta: dict) -> Path:
    """Build the full FITS path for one side of a pair."""
    fname = f"{moon.upper()}_{stem}{kind_meta['fits_suffix']}"
    return set_dir / fname


def output_path(results_dir: Path, moon: str, id1: str, id2: str) -> Path:
    """results/<moon>_<id1>_<id2>.json — short, unique per pair."""
    return results_dir / f"{moon.lower()}_{id1}_{id2}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--matches", default=DEFAULT_MATCHES,
                        help=f"Path to matches CSV (default: {DEFAULT_MATCHES})")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS,
                        help=f"Directory for output JSON files (default: {DEFAULT_RESULTS})")
    parser.add_argument("--photometry", default=None,
                        choices=["as_saved", "uncorrected", "intrinsic",
                                 "lambert", "lommel_seeliger", "minnaert"],
                        help="Photometric model to apply (passed through to "
                             "compare_mosaics.py). Default: leave to compare_mosaics "
                             "(which uses 'as_saved').")
    parser.add_argument("--max-pixel-incidence", type=float, default=None,
                        metavar="DEG",
                        help="Drop pixels whose absolute incidence in either "
                             "image exceeds DEG before computing the ratio "
                             "(passed through to compare_mosaics.py). "
                             "Suggested: 70.")

    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                        help=f"Root directory holding the per-kind subdirectories "
                             f"of reprojected FITS files. Used to compute the "
                             f"defaults of --setN-dir if those are not given. "
                             f"(default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--set1-kind", default="coiss", choices=list(KINDS),
                        help="Dataset kind for set 1 (default: coiss). "
                             "Determines the FITS filename pattern and how to "
                             "derive image stem/ID from matches CSV columns.")
    parser.add_argument("--set2-kind", default="vgiss", choices=list(KINDS),
                        help="Dataset kind for set 2 (default: vgiss).")
    parser.add_argument("--set1-dir", default=None,
                        help="Directory holding set 1 reprojected FITS files. "
                             "(default: <data-root>/<set1-kind subdir>)")
    parser.add_argument("--set2-dir", default=None,
                        help="Directory holding set 2 reprojected FITS files. "
                             "(default: <data-root>/<set2-kind subdir>)")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    set1_meta = KINDS[args.set1_kind]
    set2_meta = KINDS[args.set2_kind]
    set1_dir = (Path(args.set1_dir) if args.set1_dir
                else data_root / set1_meta["subdir"])
    set2_dir = (Path(args.set2_dir) if args.set2_dir
                else data_root / set2_meta["subdir"])

    results_dir = Path(args.results_dir)
    if not args.dry_run:
        results_dir.mkdir(parents=True, exist_ok=True)

    print(f"set1: kind={args.set1_kind}, dir={set1_dir}")
    print(f"set2: kind={args.set2_kind}, dir={set2_dir}")

    skipped = 0
    ran = 0

    with open(args.matches, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            moon = row["TARGET_NAME"].strip()
            stem1 = stem_for_set(row, 1, set1_meta)
            stem2 = stem_for_set(row, 2, set2_meta)
            file1 = fits_path(set1_dir, moon, stem1, set1_meta)
            file2 = fits_path(set2_dir, moon, stem2, set2_meta)
            id1 = id_for_set(row, 1, set1_meta)
            id2 = id_for_set(row, 2, set2_meta)
            out = output_path(results_dir, moon, id1, id2)

            # Skip pairs where either input file is missing.
            missing = [p for p in (file1, file2) if not p.exists()]
            if missing and not args.dry_run:
                for p in missing:
                    print(f"SKIP (missing): {p}", file=sys.stderr)
                skipped += 1
                continue

            cmd = [
                sys.executable, COMPARE_MOSAICS,
                str(file1), str(file2),
                "--output-statistics", str(out),
            ]
            if args.photometry:
                cmd.extend(["--photometry", args.photometry])
            if args.max_pixel_incidence is not None:
                cmd.extend(["--max-pixel-incidence", str(args.max_pixel_incidence)])

            if args.dry_run:
                print(" ".join(cmd))
            else:
                print(f"Running: {file1.name}  x  {file2.name}")
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"  ERROR: exit code {result.returncode}", file=sys.stderr)
            ran += 1

    if not args.dry_run:
        print(f"\nDone: {ran} run(s), {skipped} skipped.")


if __name__ == "__main__":
    main()
