#!/usr/bin/env python3
"""Summarise compare_mosaics output JSON files.

Each per-moon table is split by COISS camera (NAC vs WAC for image 1) since
the two cameras have separate CISSCAL calibration chains and historically
show a few-percent brightness offset. The image-2 (set2) category is
derived from the image-name prefix:

  C3xxxxxx -> "Voyager 1"   (VGISS NAC)
  C4xxxxxx -> "Voyager 2"   (VGISS NAC)
  Nxxxxxx  -> "COISS NAC"
  Wxxxxxx  -> "COISS WAC"

so the script handles both COISS-vs-VGISS cross-cal (set2 = VGISS) and
COISS NAC-vs-COISS WAC cross-cal (set2 = COISS WAC) without code changes.

Columns per row:
  Image1          set1 image stem (e.g. N1549195987_1 or W1572256382_1)
  Image2          set2 image ID   (e.g. C3493023, W1597185826_1)
  SS-Lat1/Lon1    sub-solar lat & lon for image 1 (deg)
  SS-Lat2/Lon2    sub-solar lat & lon for image 2 (deg)
  SO-Lat1/Lon1    sub-observer lat & lon for image 1 (deg)
  SO-Lat2/Lon2    sub-observer lat & lon for image 2 (deg)
  MaxΔPhase       max pixel-wise phase difference (deg)
  MaxΔInc         max pixel-wise incidence difference (deg)
  MaxΔEmiss       max pixel-wise emission difference (deg)
  MedRatio(1/2)   median per-pixel ratio (Image1 / Image2)

Footer per (moon, camera) block: mean and sample stddev of MedRatio(1/2).

Usage:
  python summarize_results.py [results_dir]
                              [--sub-pole-max-deg DEG]
                              [--no-binning]
                              [--plot [PATH]]
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
DEFAULT_RESULTS = SCRIPT_DIR / "results"

CAMERAS = ("NAC", "WAC")
# Default ordering for known set2 categories. Any category not in this list
# is appended in alphabetical order. Real categories shown are derived from
# the data, so NAC-WAC mode automatically gets "COISS WAC" (etc.) here.
_SET2_CATEGORY_ORDER = ("Voyager 1", "Voyager 2", "COISS NAC", "COISS WAC")

# Bin edges in degrees. Lower inclusive, upper exclusive; last bin is open.
PHASE_BIN_EDGES = (0.0, 2.0, 5.0, 10.0, 20.0, float("inf"))
INC_BIN_EDGES = (0.0, 5.0, 10.0, 20.0, 30.0, float("inf"))


def short_name(filepath: str) -> str:
    """Extract a compact image identifier from a full FITS path.

    e.g.  .../coiss/DIONE_N1549195987_1_CALIB_reproj.fits  ->  N1549195987_1
          .../vgiss/DIONE_C3493023_GEOMED_reproj.fits       ->  C3493023
    """
    stem = Path(filepath).stem          # e.g. DIONE_N1549195987_1_CALIB_reproj
    parts = stem.split("_")
    stop_words = {"CALIB", "GEOMED", "reproj"}
    body = []
    for p in parts[1:]:
        if p in stop_words:
            break
        body.append(p)
    return "_".join(body)


def camera_from_img1(img1: str) -> str:
    """Map a COISS image stem to its camera: 'NAC' for N-prefix, 'WAC' for W-prefix."""
    if not img1:
        return "UNK"
    first = img1[0].upper()
    if first == "N":
        return "NAC"
    if first == "W":
        return "WAC"
    return "UNK"


def get_scalar(angle_deltas: list, name: str):
    """Return (value_file1, value_file2) for a scalar angle entry."""
    for entry in angle_deltas:
        if entry["name"] == name and entry["kind"] == "scalar":
            return entry["value_file1_deg"], entry["value_file2_deg"]
    return None, None


def get_pixel_max(angle_deltas: list, name: str):
    """Return the max pixel-wise difference for a pixel-kind angle entry."""
    for entry in angle_deltas:
        if entry["name"] == name and entry["kind"] == "pixel":
            return entry["stats"]["max"]
    return None


def set2_category(img2: str) -> str:
    """Return the set2 category label derived from the image-name prefix.

    Handles both VGISS (C3/C4 -> 'Voyager 1'/'Voyager 2') and COISS
    (N -> 'COISS NAC', W -> 'COISS WAC') so the same script works for
    COISS-vs-VGISS and COISS NAC-vs-COISS WAC datasets.
    """
    if not img2:
        return "Unknown"
    s = img2.upper()
    if s.startswith("C3"):
        return "Voyager 1"
    if s.startswith("C4"):
        return "Voyager 2"
    if s.startswith("N"):
        return "COISS NAC"
    if s.startswith("W"):
        return "COISS WAC"
    return "Unknown"


def sorted_set2_categories(by_moon: dict) -> list[str]:
    """Return the distinct set2 categories present in the data, ordered."""
    seen = set()
    for rows in by_moon.values():
        for r in rows:
            seen.add(r["voyager"])
    known = [c for c in _SET2_CATEGORY_ORDER if c in seen]
    extra = sorted(seen - set(known))
    return known + extra


def load_results(results_dir: Path) -> tuple[dict[str, list], set[str],
                                              set[float | None], list[float]]:
    """Load all JSON files.

    Returns:
      by_moon: dict moon -> list of row dicts
      photometry_modes: set of photometry strings seen
      inc_caps: set of pixel-incidence caps seen (None if not applied)
      kept_fractions: list of overlap-survival fractions (one per pair where
                      the incidence mask was applied)
    """
    by_moon: dict[str, list] = defaultdict(list)
    photometry_modes: set[str] = set()
    inc_caps: set[float | None] = set()
    kept_fractions: list[float] = []

    for path in sorted(results_dir.glob("*.json")):
        with open(path) as fh:
            data = json.load(fh)

        meta = data["metadata"]
        moon = meta["body_name"].capitalize()
        ad = data["angle_deltas"]
        overlap = data.get("overlap", {})

        if "photometry" in meta:
            photometry_modes.add(str(meta["photometry"]))
        inc_caps.add(meta.get("max_pixel_incidence_deg"))

        n_after = overlap.get("n_overlap")
        n_before = overlap.get("n_overlap_before_inc_mask")
        if n_before and n_after is not None:
            kept_fractions.append(n_after / n_before)

        ss_lat1, ss_lat2 = get_scalar(ad, "sub_solar_lat")
        ss_lon1, ss_lon2 = get_scalar(ad, "sub_solar_lon")
        so_lat1, so_lat2 = get_scalar(ad, "sub_observer_lat")
        so_lon1, so_lon2 = get_scalar(ad, "sub_observer_lon")

        img1 = short_name(meta["file1"])
        img2 = short_name(meta["file2"])
        by_moon[moon].append({
            "img1":      img1,
            "img2":      img2,
            "camera":    camera_from_img1(img1),
            "voyager":   set2_category(img2),
            "ss_lat1":   ss_lat1,
            "ss_lon1":   ss_lon1,
            "ss_lat2":   ss_lat2,
            "ss_lon2":   ss_lon2,
            "so_lat1":   so_lat1,
            "so_lon1":   so_lon1,
            "so_lat2":   so_lat2,
            "so_lon2":   so_lon2,
            "max_phase": get_pixel_max(ad, "phase"),
            "max_inc":   get_pixel_max(ad, "incidence"),
            "max_emiss": get_pixel_max(ad, "emission"),
            "med_ratio": data["ratio_of_pixels"]["p50"] if data.get("ratio_of_pixels") else None,
        })

    return by_moon, photometry_modes, inc_caps, kept_fractions


def passes_sub_pole(row: dict, max_deg: float | None) -> bool:
    """True if both ΔSSLat and ΔSOLat are within ``max_deg``."""
    if max_deg is None:
        return True
    try:
        delta_ss = abs(row["ss_lat1"] - row["ss_lat2"])
        delta_so = abs(row["so_lat1"] - row["so_lat2"])
    except TypeError:
        return False
    return max(delta_ss, delta_so) <= max_deg


def filter_sub_pole(by_moon: dict, max_deg: float | None) -> tuple[dict, int, int]:
    """Return (filtered_by_moon, n_in, n_out)."""
    if max_deg is None:
        n_total = sum(len(v) for v in by_moon.values())
        return by_moon, n_total, n_total
    out: dict[str, list] = defaultdict(list)
    n_in = 0
    n_out = 0
    for moon, rows in by_moon.items():
        for r in rows:
            n_in += 1
            if passes_sub_pole(r, max_deg):
                out[moon].append(r)
                n_out += 1
    return out, n_in, n_out


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_COL_SPEC = [
    ("Image1",    14, None),
    ("Image2",    14, None),
    ("SSLat1",     7,    1),
    ("SSLat2",     7,    1),
    ("SSLon1",     7,    1),
    ("SSLon2",     7,    1),
    ("SOLat1",     7,    1),
    ("SOLat2",     7,    1),
    ("SOLon1",     7,    1),
    ("SOLon2",     7,    1),
    ("MaxΔPhase",  7,    2),
    ("MaxΔInc",    7,    2),
    ("MaxΔEmiss",  7,    2),
    ("MedRatio(1/2)", 8, 4),
]
COLS = [(hdr, max(len(hdr), dw), dec) for hdr, dw, dec in _COL_SPEC]


def fmt_f(val, width, decimals=1):
    if val is None:
        return " " * width
    return f"{val:{width}.{decimals}f}"


def header_line() -> str:
    return "  ".join(hdr.center(w) for hdr, w, _ in COLS)


def separator_line() -> str:
    return "  ".join("-" * w for _, w, _ in COLS)


def data_line(row: dict) -> str:
    vals = [
        row["img1"],
        row["img2"],
        row["ss_lat1"], row["ss_lat2"],
        row["ss_lon1"], row["ss_lon2"],
        row["so_lat1"], row["so_lat2"],
        row["so_lon1"], row["so_lon2"],
        row["max_phase"],
        row["max_inc"],
        row["max_emiss"],
        row["med_ratio"],
    ]
    parts = []
    for (_, w, dec), val in zip(COLS, vals):
        if dec is None:
            parts.append((val or "").ljust(w)[:w])
        else:
            parts.append(fmt_f(val, w, dec))
    return "  ".join(parts)


def _stats_text(ratios: list[float]) -> str:
    """Return 'Mean MedRatio(1/2): X.XXXX  StdDev: Y.YYYY' for a list of ratios."""
    _, _, dec = COLS[-1]
    mean = sum(ratios) / len(ratios)
    if len(ratios) > 1:
        std_str = f"{statistics.stdev(ratios):.{dec}f}"
    else:
        std_str = " " * (dec + 1) + "---"
    return f"Mean MedRatio(1/2): {mean:.{dec}f}  StdDev: {std_str}"


def _mean_line(ratios: list[float]) -> str:
    return _stats_text(ratios).rjust(len(header_line()))


def print_subtable(label: str, rows: list) -> list[float]:
    """Print one labeled subtable (one moon × one camera) and return its ratios."""
    print(f"\n  {label}")
    print(header_line())
    print(separator_line())
    ratios = []
    for row in sorted(rows, key=lambda r: (r["img1"], r["img2"])):
        print(data_line(row))
        if row["med_ratio"] is not None:
            ratios.append(row["med_ratio"])
    print(separator_line())
    if ratios:
        print(_mean_line(ratios))
    return ratios


def print_mission_section(mission: str, by_moon: dict) -> dict[str, list[float]]:
    """Print every (moon, camera) subtable for one set2 category.

    ``mission`` is whatever ``set2_category`` returned: "Voyager 1",
    "Voyager 2", "COISS NAC", "COISS WAC", etc. Returns a dict
    ``{camera: [all ratios across moons for that camera]}``.
    """
    total_width = len(header_line())
    print(f"\n{'=' * total_width}")
    print(f"  SET2 = {mission.upper()}")
    print(f"{'=' * total_width}")

    per_camera: dict[str, list[float]] = {cam: [] for cam in CAMERAS}
    for moon in sorted(by_moon):
        for camera in CAMERAS:
            rows = [r for r in by_moon[moon]
                    if r["voyager"] == mission and r["camera"] == camera]
            if not rows:
                continue
            label = f"{moon.upper()}  ·  set1=COISS {camera}  ·  set2={mission}"
            per_camera[camera] += print_subtable(label, rows)
    return per_camera


# ---------------------------------------------------------------------------
# Final summary, matrix, and binned stats
# ---------------------------------------------------------------------------


def _bucket_index(value: float | None, edges: tuple[float, ...]) -> int | None:
    """Return bucket index in [0, len(edges)-2] or None if value is None / out of range."""
    if value is None:
        return None
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return None


def _bin_label(edges: tuple[float, ...], i: int) -> str:
    lo, hi = edges[i], edges[i + 1]
    if hi == float("inf"):
        return f">={lo:.0f}°"
    return f"{lo:.0f}-{hi:.0f}°"


def _print_binned_one(label: str, rows: list, key: str,
                      edges: tuple[float, ...]) -> None:
    """Print mean MedRatio(1/2) binned by ``key`` (e.g. 'max_phase')."""
    dec = COLS[-1][2]
    bin_w = max(len("Bin"),
                max(len(_bin_label(edges, i)) for i in range(len(edges) - 1)))
    n_w = max(4, len("N"))
    mean_w = max(8, dec + 4)
    std_w = max(8, dec + 4)
    row_w = bin_w + 2 + n_w + 2 + mean_w + 2 + std_w
    line_w = max(row_w, len(label) + 2)

    print(f"\n{'=' * line_w}")
    print(f"  {label}")
    print(f"{'=' * line_w}")
    print(f"{'Bin':>{bin_w}}  {'N':>{n_w}}  {'Mean':>{mean_w}}  {'StdDev':>{std_w}}")
    print(f"{'-' * bin_w}  {'-' * n_w}  {'-' * mean_w}  {'-' * std_w}")
    for i in range(len(edges) - 1):
        bin_rows = [r for r in rows
                    if r["med_ratio"] is not None
                    and _bucket_index(r[key], edges) == i]
        ratios = [r["med_ratio"] for r in bin_rows]
        bl = _bin_label(edges, i)
        if not ratios:
            print(f"{bl:>{bin_w}}  {0:>{n_w}d}  {'---':>{mean_w}}  {'---':>{std_w}}")
            continue
        mean = sum(ratios) / len(ratios)
        if len(ratios) > 1:
            std_str = f"{statistics.stdev(ratios):.{dec}f}"
        else:
            std_str = "---"
        print(f"{bl:>{bin_w}}  {len(ratios):>{n_w}d}  "
              f"{mean:>{mean_w}.{dec}f}  {std_str:>{std_w}}")
    print(f"{'=' * line_w}")


def _print_binned_section(by_moon: dict) -> None:
    """For every (set2_cat, camera) group, show MedRatio binned by ΔPhase and ΔInc."""
    for mission in sorted_set2_categories(by_moon):
        for camera in CAMERAS:
            rows = [r for moon_rows in by_moon.values() for r in moon_rows
                    if r["voyager"] == mission and r["camera"] == camera]
            if not rows:
                continue
            _print_binned_one(
                f"set2={mission}  ·  COISS {camera}  binned by MaxΔPhase",
                rows, "max_phase", PHASE_BIN_EDGES,
            )
            _print_binned_one(
                f"set2={mission}  ·  COISS {camera}  binned by MaxΔInc",
                rows, "max_inc", INC_BIN_EDGES,
            )


_CATEGORY_MARKERS = {
    "Voyager 1": "C3xxxxxx",
    "Voyager 2": "C4xxxxxx",
    "COISS NAC": "Nxxxxxxxxx",
    "COISS WAC": "Wxxxxxxxxx",
}


def _print_final_summary(per_mission_camera: dict[tuple[str, str], list[float]],
                         missions: list[str]) -> None:
    """Print the final summary block split by (set2_cat, camera) and overall."""
    total_width = len(header_line())
    print(f"\n{'=' * total_width}")
    print("  FINAL SUMMARY  —  mean MedRatio(1/2) across all moons")
    print(f"{'=' * total_width}")

    for mission in missions:
        marker = _CATEGORY_MARKERS.get(mission, "")
        marker_str = f" ({marker})" if marker else ""
        for camera in CAMERAS:
            ratios = per_mission_camera.get((mission, camera), [])
            if not ratios:
                continue
            line = (f"set2={mission}{marker_str}  set1=COISS {camera}  "
                    f"{len(ratios):3d} pairs  " + _stats_text(ratios))
            print(line.rjust(total_width))

    # Roll-up per camera across all set2 categories. Only meaningful when
    # there is more than one set2 category present.
    if len(missions) > 1:
        for camera in CAMERAS:
            ratios = []
            for mission in missions:
                ratios += per_mission_camera.get((mission, camera), [])
            if not ratios:
                continue
            line = (f"All set2  set1=COISS {camera}  "
                    f"{len(ratios):3d} pairs  " + _stats_text(ratios))
            print(line.rjust(total_width))

    all_ratios = [r for rs in per_mission_camera.values() for r in rs]
    if all_ratios:
        line = f"All pairs  {len(all_ratios):3d} pairs  " + _stats_text(all_ratios)
        print(line.rjust(total_width))
    print(f"{'=' * total_width}")


def _aggregate_by_image2(by_moon: dict
                         ) -> dict[tuple[str, str, str], dict]:
    """Return ``{(img2_id, set2_cat, camera): {moon, ratios}}``.

    Group rows by image-2 (and COISS camera, since the same image-2 can be
    paired against both NAC and WAC images-1). Used to compute the
    cross-image-2 cal scale that's not biased by repeated image-1 imaging
    of the same target image-2.
    """
    grouped: dict[tuple[str, str, str], dict] = {}
    for moon, rows in by_moon.items():
        for r in rows:
            if r["med_ratio"] is None:
                continue
            key = (r["img2"], r["voyager"], r["camera"])
            entry = grouped.setdefault(key, {"moon": moon, "ratios": []})
            entry["ratios"].append(r["med_ratio"])
    return grouped


def _print_by_image2_section(by_moon: dict, missions: list[str]) -> None:
    """Print per-image-2 means and the cross-image-2 aggregate.

    Rationale: every (image1, image2) pair gives one MedRatio. The per-moon
    mean is biased by which moons happen to have many image-1 samples
    backing a single image-2 (e.g. Mimas: many COISS images paired with one
    VGISS image). Reducing first within each image-2 gives equal weight to
    each independent set2 calibration sample.
    """
    grouped = _aggregate_by_image2(by_moon)
    if not grouped:
        return

    dec = COLS[-1][2]
    cols = [
        ("Image2",      14, "img"),
        ("Moon",        10, "moon"),
        ("Set2",         11, "mission"),
        ("Set1 cam",     8, "camera"),
        ("N pairs",      8, "n"),
        ("Mean",         8, "mean"),
        ("StdDev",       8, "std"),
    ]
    widths = [(hdr, max(len(hdr), w), key) for hdr, w, key in cols]
    header = "  ".join(hdr.center(w) for hdr, w, _ in widths)
    sep = "  ".join("-" * w for _, w, _ in widths)
    title = "  BY-IMAGE-2 MEAN MedRatio(1/2)  (one row per image-2 · set1 camera)"
    line_w = max(len(header), len(title))

    print(f"\n{'=' * line_w}")
    print(title)
    print(f"{'=' * line_w}")
    print(header)
    print(sep)

    def sort_key(item):
        (vgiss, mission, camera), entry = item
        return (mission, camera, entry["moon"], vgiss)

    # Per (mission, camera) lists of per-image means, used for the aggregate.
    per_mc: dict[tuple[str, str], list[float]] = defaultdict(list)

    for (vgiss, mission, camera), entry in sorted(grouped.items(), key=sort_key):
        ratios = entry["ratios"]
        mean = sum(ratios) / len(ratios)
        per_mc[(mission, camera)].append(mean)
        if len(ratios) > 1:
            std_str = f"{statistics.stdev(ratios):.{dec}f}"
        else:
            std_str = " " * (dec + 1) + "---"
        values = {
            "img":     vgiss,
            "moon":    entry["moon"],
            "mission": mission,
            "camera":  camera,
            "n":       f"{len(ratios):d}",
            "mean":    f"{mean:.{dec}f}",
            "std":     std_str,
        }
        parts = []
        for hdr, w, key in widths:
            v = values[key]
            if key in ("img", "moon", "mission", "camera"):
                parts.append(v.ljust(w)[:w])
            else:
                parts.append(v.rjust(w))
        print("  ".join(parts))
    print(f"{'=' * line_w}")

    # Cross-image-2 aggregate per (set2_cat, camera).
    title = ("  CROSS-IMAGE-2 AGGREGATE  "
             "(mean of per-image-2 means; one image-2 = one sample)")
    print(f"\n{'=' * len(title)}")
    print(title)
    print(f"{'=' * len(title)}")
    for mission in missions:
        for camera in CAMERAS:
            means = per_mc.get((mission, camera), [])
            if not means:
                continue
            agg_mean = sum(means) / len(means)
            if len(means) > 1:
                agg_std = f"{statistics.stdev(means):.{dec}f}"
            else:
                agg_std = " " * (dec + 1) + "---"
            print(f"  set2={mission}  set1=COISS {camera}  "
                  f"{len(means):3d} image-2 samples  "
                  f"Mean: {agg_mean:.{dec}f}  StdDev: {agg_std}")
    # Roll-up across set2 categories, per camera. Only useful if there is
    # more than one set2 category present.
    if len(missions) > 1:
        for camera in CAMERAS:
            means = []
            for mission in missions:
                means += per_mc.get((mission, camera), [])
            if not means:
                continue
            agg_mean = sum(means) / len(means)
            if len(means) > 1:
                agg_std = f"{statistics.stdev(means):.{dec}f}"
            else:
                agg_std = " " * (dec + 1) + "---"
            print(f"  All set2  set1=COISS {camera}  "
                  f"{len(means):3d} image-2 samples  "
                  f"Mean: {agg_mean:.{dec}f}  StdDev: {agg_std}")
    print(f"{'=' * len(title)}")


def _print_per_moon_matrix(by_moon: dict, missions: list[str]) -> None:
    """Print a (set2_cat × camera) × moon matrix of mean MedRatio(1/2) values."""
    moons = sorted(by_moon)
    if not moons:
        return
    dec = COLS[-1][2]
    row_labels = [f"{m} {c}" for m in missions for c in CAMERAS]
    mission_w = max([len("Set2 · cam")] + [len(s) for s in row_labels])
    col_w = max(7, max(len(m) for m in moons))
    title = "  PER-MOON MEAN MedRatio(1/2)  (rows: set2 · set1 camera; cols: moons)"
    matrix_width = max(mission_w + len(moons) * (col_w + 2), len(title))

    print(f"\n{'=' * matrix_width}")
    print(title)
    print(f"{'=' * matrix_width}")
    print("Set2 · cam".ljust(mission_w) + "  "
          + "  ".join(m.center(col_w) for m in moons))
    print("-" * mission_w + "  "
          + "  ".join("-" * col_w for _ in moons))
    for mission in missions:
        for camera in CAMERAS:
            cells = []
            any_data = False
            for moon in moons:
                ratios = [r["med_ratio"] for r in by_moon[moon]
                          if r["voyager"] == mission and r["camera"] == camera
                          and r["med_ratio"] is not None]
                if ratios:
                    any_data = True
                    cells.append(f"{sum(ratios)/len(ratios):{col_w}.{dec}f}")
                else:
                    cells.append("---".rjust(col_w))
            if not any_data:
                continue
            print(f"{mission} {camera}".ljust(mission_w) + "  "
                  + "  ".join(cells))
    print(f"{'=' * matrix_width}")


# ---------------------------------------------------------------------------
# Multi-panel plot
# ---------------------------------------------------------------------------


def _delta_ss_lat(row: dict):
    a, b = row.get("ss_lat1"), row.get("ss_lat2")
    return None if a is None or b is None else abs(a - b)


def _delta_ss_lon(row: dict):
    a, b = row.get("ss_lon1"), row.get("ss_lon2")
    if a is None or b is None:
        return None
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


# Marker + linestyle key for each (set2_cat, camera) combination. Within a
# moon, color is shared; set2 · camera is conveyed by linestyle and marker.
# Combinations not in this table fall back to ``_FALLBACK_STYLES`` indexed
# by their first-seen order.
_FALLBACK_STYLES = [
    {"linestyle": "-",  "marker": "o"},
    {"linestyle": "--", "marker": "s"},
    {"linestyle": ":",  "marker": "^"},
    {"linestyle": "-.", "marker": "D"},
    {"linestyle": "-",  "marker": "v"},
    {"linestyle": "--", "marker": "P"},
    {"linestyle": ":",  "marker": "X"},
    {"linestyle": "-.", "marker": "*"},
]
_GROUP_STYLE = {
    ("Voyager 1", "NAC"): _FALLBACK_STYLES[0],
    ("Voyager 1", "WAC"): _FALLBACK_STYLES[1],
    ("Voyager 2", "NAC"): _FALLBACK_STYLES[2],
    ("Voyager 2", "WAC"): _FALLBACK_STYLES[3],
}


def _resolve_group_style(combo: tuple[str, str],
                         style_index: dict[tuple[str, str], dict]) -> dict:
    """Return a style dict for ``combo``, assigning a fallback if unknown.

    ``style_index`` is a mutable cache so the same combo always renders
    with the same style across all panels of the figure.
    """
    if combo in style_index:
        return style_index[combo]
    if combo in _GROUP_STYLE:
        style = _GROUP_STYLE[combo]
    else:
        style = _FALLBACK_STYLES[len(style_index) % len(_FALLBACK_STYLES)]
    style_index[combo] = style
    return style


def _make_plot(by_moon: dict, output_path: str | None,
               sub_pole_max_deg: float | None) -> None:
    """Multi-panel plot of MedRatio(1/2) vs five geometry deltas.

    One series per (moon, mission, camera) combination. Moon → color (tab10),
    mission · camera → linestyle + marker per ``_GROUP_STYLE``.

    output_path:
      None or empty string -> show interactively.
      Any other string     -> save the figure to that path (PNG/PDF by extension).
    """
    import matplotlib
    if output_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    panels = [
        ("MaxΔPhase (deg)",     lambda r: r["max_phase"]),
        ("MaxΔInc (deg)",       lambda r: r["max_inc"]),
        ("MaxΔEmiss (deg)",     lambda r: r["max_emiss"]),
        ("|ΔSubSolarLat| (deg)", _delta_ss_lat),
        ("|ΔSubSolarLon| (deg)", _delta_ss_lon),
    ]

    moons = sorted(by_moon)
    cmap = plt.colormaps.get_cmap("tab10")
    moon_colors = {m: cmap(i % 10) for i, m in enumerate(moons)}
    missions = sorted_set2_categories(by_moon)

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    flat = list(axes.flatten())

    populated_groups: set[tuple[str, str]] = set()
    style_index: dict[tuple[str, str], dict] = {}

    for ax, (xlabel, xfn) in zip(flat, panels):
        for moon in moons:
            for mission in missions:
                for camera in CAMERAS:
                    data = []
                    for r in by_moon[moon]:
                        if r["voyager"] != mission or r["camera"] != camera:
                            continue
                        x = xfn(r)
                        y = r["med_ratio"]
                        if x is None or y is None:
                            continue
                        data.append((x, y))
                    if not data:
                        continue
                    populated_groups.add((mission, camera))
                    data.sort(key=lambda d: d[0])
                    xs = [d[0] for d in data]
                    ys = [d[1] for d in data]
                    style = _resolve_group_style((mission, camera), style_index)
                    ax.plot(xs, ys,
                            color=moon_colors[moon],
                            linestyle=style["linestyle"],
                            marker=style["marker"],
                            markersize=5, linewidth=1.2, alpha=0.85)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.6)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("MedRatio(1/2) = COISS / VGISS", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=10)

    # 6th panel: two legends (moon→color and mission·camera→line/marker).
    legend_ax = flat[-1]
    legend_ax.axis("off")

    moon_handles = [mpatches.Patch(color=moon_colors[m], label=m) for m in moons]
    moon_legend = legend_ax.legend(
        handles=moon_handles, loc="upper left",
        title="Moon (color)", fontsize=12, title_fontsize=13, frameon=True,
    )
    legend_ax.add_artist(moon_legend)

    style_handles = []
    for mission in missions:
        for camera in CAMERAS:
            if (mission, camera) not in populated_groups:
                continue
            style = _resolve_group_style((mission, camera), style_index)
            style_handles.append(mlines.Line2D(
                [], [], color="black",
                linestyle=style["linestyle"], marker=style["marker"],
                markersize=6, linewidth=1.4,
                label=f"set2={mission} · set1=COISS {camera}",
            ))
    if style_handles:
        legend_ax.legend(
            handles=style_handles, loc="lower right",
            title="Set2 · Set1 camera (linestyle + marker)",
            fontsize=12, title_fontsize=13, frameon=True,
        )

    title = ("MedRatio(1/2) vs geometry mismatch  "
             "(series = moon · set2 · set1 camera)")
    if sub_pole_max_deg is not None:
        title += f"   [sub-pole filter: |ΔSSLat|, |ΔSOLat| ≤ {sub_pole_max_deg}°]"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", nargs="?", default=str(DEFAULT_RESULTS),
                        help=f"Directory of compare_mosaics JSON files "
                             f"(default: {DEFAULT_RESULTS})")
    parser.add_argument("--sub-pole-max-deg", type=float, default=None,
                        help="Drop pairs where |ΔSSLat| or |ΔSOLat| exceeds this "
                             "value (deg). Removes hemisphere-mismatch pairs that "
                             "see different parts of the moon. Suggested: 10.")
    parser.add_argument("--no-binning", action="store_true",
                        help="Skip the per-(mission,camera) MaxΔPhase / MaxΔInc "
                             "binned-mean tables.")
    parser.add_argument("--plot", nargs="?", const="", default=None, metavar="PATH",
                        help="Make a multi-panel plot of MedRatio(1/2) vs geometry "
                             "deltas, one colored series per moon. Bare --plot shows "
                             "the figure interactively; --plot foo.png (or .pdf) "
                             "saves it to that path. The sub-pole filter applies.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    by_moon, photometry_modes, inc_caps, kept_fractions = load_results(results_dir)
    if not by_moon:
        print("No JSON files found.", file=sys.stderr)
        sys.exit(1)

    # Header banner with photometry mode and sub-pole filter info.
    if photometry_modes:
        modes = ", ".join(sorted(photometry_modes))
        print(f"Photometry mode (per JSON metadata): {modes}")
    else:
        print("Photometry mode: (not recorded in JSON metadata)")

    caps_set = inc_caps - {None}
    if caps_set:
        caps_str = ", ".join(f"{c:g}°" for c in sorted(caps_set))
        if None in inc_caps:
            caps_str += "  (some pairs unfiltered)"
        if kept_fractions:
            mean_kept = sum(kept_fractions) / len(kept_fractions)
            min_kept = min(kept_fractions)
            print(f"Per-pixel incidence cap: {caps_str}  "
                  f"(mean overlap kept: {mean_kept:.2%}, min: {min_kept:.2%})")
        else:
            print(f"Per-pixel incidence cap: {caps_str}")
    else:
        print("Per-pixel incidence cap: (none — pass --max-pixel-incidence DEG "
              "to compare_mosaics)")

    by_moon, n_in, n_out = filter_sub_pole(by_moon, args.sub_pole_max_deg)
    if args.sub_pole_max_deg is not None:
        print(f"Sub-pole filter: |ΔSSLat|<={args.sub_pole_max_deg}° and "
              f"|ΔSOLat|<={args.sub_pole_max_deg}°  "
              f"({n_out}/{n_in} pairs kept)")
    else:
        print("Sub-pole filter: (none — use --sub-pole-max-deg DEG to apply)")

    if not by_moon:
        print("No pairs survived the sub-pole filter.", file=sys.stderr)
        sys.exit(1)

    missions = sorted_set2_categories(by_moon)
    print(f"Set2 categories present: {', '.join(missions)}")

    per_mission_camera: dict[tuple[str, str], list[float]] = {}
    for mission in missions:
        per_camera = print_mission_section(mission, by_moon)
        for camera, ratios in per_camera.items():
            if ratios:
                per_mission_camera[(mission, camera)] = ratios

    _print_final_summary(per_mission_camera, missions)
    _print_per_moon_matrix(by_moon, missions)
    _print_by_image2_section(by_moon, missions)
    if not args.no_binning:
        _print_binned_section(by_moon)

    if args.plot is not None:
        _make_plot(by_moon, args.plot or None, args.sub_pole_max_deg)


if __name__ == "__main__":
    main()
