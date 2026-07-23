# Critique: `pds4_bundle_gen` vs. the Cassini ISS F Ring Mosaics Users Guide

**Date:** 2026-07-19
**Software reviewed:** `/seti/research/f-ring/f-ring/pds4_bundle_gen/` — `generate_pds4_files.py` (3,840 lines), `templates/*.lblx` (~4,700 lines), `templates/examples/*.py` (5 scripts), `templates/readme.txt`, `observation_list.csv`
**Users guide compared against:** `~/DS/f-ring-users-guide-latex/sections/*.tex` (current LaTeX sources, V1.0)
**Ground truth:** the generated bundle at `/data/fring-bundles/pds4/` (and the `pds4-full-*` builds). All HIGH findings and most MEDIUM findings were verified directly against the source code and/or the generated bundle, not just reported by review passes. The four runnable example scripts were executed end-to-end against the real bundle (including the 360° wraparound path) and all ran without crashing.

Line numbers refer to the current files.

---

## Executive summary

The pipeline is in good shape overall — the orbit model, corotating-frame math, wraparound min/max logic, SCLK arithmetic, time-system chain, array geometry, and all three global-index field sets were checked and are correct and consistent with the guide. But the review found **four high-severity bugs in the generator** (two of which have already corrupted metadata in every generated bundle: all NAC products are attributed to the Wide Angle Camera, and Pandora is never added as a target), **one high-severity science bug in the shipped example scripts** (degrees passed to `np.cos`, so the showcased equivalent-width plots are quantitatively wrong), **two label-validity problems** (missing miscellaneous collection inventory; `<files>5</files>` vs six files), and a set of naming/size disagreements between the software and the users guide, plus guide-side errata from the docx→LaTeX conversion.

---

## 1. High-severity bugs in `generate_pds4_files.py`

### 1.1 Every product is labeled "Wide Angle Camera" — NAC attribution is impossible ⚠ VERIFIED IN BUNDLE
`generate_pds4_files.py:1765-1782`. The camera letter is correctly extracted (`camera = image_name0[-1]`, since `reformat_iss_name` puts the letter last, e.g. `1545556618n`), but the branch three lines later tests the **first** character:

```python
if image_name0[0] == 'n':          # first char is always a digit
    ret['CAMERA_WIDTH'] = 'Narrow'
else:
    ret['CAMERA_WIDTH'] = 'Wide'   # always taken
```

Every label therefore gets `CAMERA_WIDTH='Wide'`, `CAMERA_WN_LC='w'`, which feeds the instrument name and the context LID (`iss$CAMERA_WN_LC$a.co`). Verified in the bundle: `data_mosaic/iss_036rf_fmovie001_vims/iss_036rf_fmovie001_vims_mosaic.lblx:110` says `Cassini Imaging Science Subsystem - Wide Angle Camera` / `isswa.co` even though every source image in that observation is NAC (`1545556618n…`). The guide (§1) emphasizes that the archive contains both NAC and WAC clear-filter images; the archive as generated attributes all of them to the WAC. Fix: test `camera == 'n'`.

### 1.2 Pandora can never be detected as a target (sign error) ⚠ VERIFIED IN BUNDLE
`generate_pds4_files.py:984-987`:

```python
radius_sat_dist = closest_radius - closest_sat_dist
return ((radius_sat_dist < 0 and                        # Pandora
         radius_sat_dist > arguments.radius_outer_delta) or   # +1000: impossible
        (radius_sat_dist > 0 and                        # Prometheus
         radius_sat_dist < -arguments.radius_inner_delta))    # -(-1000): correct
```

Pandora orbits outside the core, so `radius_sat_dist < 0`; requiring it to also be `> +1000` is unsatisfiable. The Pandora clause should be `radius_sat_dist > -arguments.radius_outer_delta`. Verified in the bundle: 111 of 302 mosaic labels carry a Prometheus `Target_Identification`; **zero** carry Pandora, even though `observation_list.csv` marks several observations Pandora=Y. This also means the visual-vs-computed consistency warning (line 1971-1973) fires spuriously for every Pandora=Y observation. This directly contradicts the CHANGELOG claim that satellite identification "is far superior to the previous method." (TODO.txt's "Deal with Prometheus and Pandora in reprojected images" is adjacent but does not describe this sign bug.)

### 1.3 `remap_image_indexes` compacts the image lists with the wrong index
`generate_pds4_files.py:1044-1049`:

```python
new_image_name_list = [reformat_iss_name(image_name_list[number_map[x]])
                           for x in number_map.keys() if x != SENTINEL]
```

`number_map` maps old index → new index; to build the compacted list the subscript must be the **old** index `x`, not `number_map[x]`. As written, the new lists are simply the first N entries of the original lists, which is correct only when the used indexes happen to be the contiguous prefix 0..N-1 — precisely the case where this function is a no-op. Whenever an *interior* image is dropped from a mosaic (the exact situation the docstring says this handles), the min/max image names, PDS3 start/stop times and SCLKs, the `_metadata_src_imgs.tab` LIDVID table, and the set of reprojected-image products generated for the observation (line 3147 iterates `image_path_list`) all come from the wrong images — silently. The current bundle appears healthy, which suggests no interior image has been dropped yet, but this is a landmine for any future regeneration.

### 1.4 Browse-label generation flags are crossed (mosaic ↔ reproj) ⚠ VERIFIED IN SOURCE
`generate_pds4_files.py:2973-2974`:

```python
if ((img_type != 'r' and GENERATE_BROWSE_REPROJ_LABELS) or
    (img_type == 'r' and GENERATE_BROWSE_MOSAIC_LABELS)):
```

Compare the browse-*image* gate at 2829-2830, which pairs the flags correctly. Here mosaic browse labels are gated by the reproj flag and vice versa. Invisible under `--generate-all` (both flags true), but running `--generate-mosaic-browse-labels` or `--generate-reproj-browse-labels` alone silently produces no labels.

---

## 2. Medium-severity bugs in `generate_pds4_files.py`

### 2.1 Global-index `rings:minimum_ring_radius` uses the mean core radius ⚠ VERIFIED IN BUNDLE
`generate_pds4_files.py:1753-1754`:

```python
ret['MIN_RING_RADIUS']       = f'{min_radius+arguments.radius_inner_delta:.3f}'
ret['MIN_RING_RADIUS_FIXED'] = f'{mean_radius+arguments.radius_inner_delta:10.3f}'   # mean, not min
```

The `_FIXED` variant is what goes into the global index. Verified: for `iss_036rf_fmovie001_vims`, the label says minimum_ring_radius = 139439.689 km while the index says 139454.997 km (= mean − 1000). The core radius varies by ±330 km, so index values can be wrong by hundreds of km, and label and index disagree for the same product. `MAX_RING_RADIUS_FIXED` (1756) is correct.

### 2.2 Supplemental-file preamble is silently discarded
`generate_pds4_files.py:1563-1565`: two lines of explanatory header are built with `=`/`+=`, then line 1565 *reassigns* (`hdr_text = f'Source Data Product ID = ...'`) instead of appending, so the preamble ("This file contains a C-matrix that describes...") never appears in any `*_reproj_suppl.txt`. The label's `SUPPL_HEADER_LENGTH` is computed from the same truncated string, so nothing catches it. (The guide's §4 sample of this file matches the truncated output, so the guide inadvertently documents the bug.)

### 2.3 `miscellaneous/` collection inventory is never generated ⚠ VERIFIED IN BUNDLE
`generate_pds4_files.py:3357-3365` sets up `collection_miscellaneous.csv/.lblx` metadata but — unlike the sibling context/document/spice/xml_schema blocks — is missing the `copy_file(...)` and `populate_template(...)` calls. Verified: `/data/fring-bundles/pds4/miscellaneous/` and both `pds4-full-*` builds contain only the six index files; there is no `collection_miscellaneous.csv/.lblx`, although `bundle.lblx` declares the collection (`bundle_has_miscellaneous_collection`), the templates exist, and the guide (§4.4) documents both files. **This will fail PDS4 validation.** Also, `generate_support_files()` never creates the `miscellaneous` directory itself (only the global-index code path does, line 3505-3507).

### 2.4 Failed obsids are still registered in collection CSVs
`generate_pds4_files.py:3702-3711`: the collection-membership rows are written after the `except ObsIdFailedException: pass` handler, outside the `try`, so an obsid whose products failed is still written as a Primary member of the data/browse collection CSVs — the inventory then references nonexistent products. The reproj equivalent (3154-3159) writes the CSV row *before* `generate_reproj()` runs, with the same effect on failure.

### 2.5 `--generate-reproj-collections` alone produces an empty collection
`generate_pds4_files.py:3135-3138`: the gate around the reproj loop tests seven GENERATE_REPROJ_* flags but omits `GENERATE_REPROJ_COLLECTIONS` and `GENERATE_BROWSE_REPROJ_COLLECTIONS`, so running only those flags opens the CSV, writes nothing, and generates a label for an empty inventory — silently.

### 2.6 Support-only runs stamp garbage dates into `bundle.lblx`
`generate_pds4_files.py:3467-3468` initialize `EARLIEST_START_DATE_TIME = 1e38`, `LATEST_STOP_DATE_TIME = 0`; they are only updated during per-obsid label generation (1912-1914). A `--generate-support-files`-only run feeds the sentinels straight into `et_to_datetime` for the bundle description (3430-3431) with no guard.

### 2.7 bkg-sub mosaic mask deliberately zeroed before SENTINEL fill
`generate_pds4_files.py:860-863`: after loading the bkg-sub `.npz`, `metadata['img'].mask = False` destroys the bad-pixel mask, so the later `ma.filled(..., SENTINEL)` (2537) and browse fill (2831) are no-ops. If the stored arrays already have −999 baked in this is benign, but the comment ("The background image mask shows the 'bad pixels'") plus the asymmetry with the plain-mosaic path suggests an unresolved decision. Worth confirming against an actual `.npz` before the next regeneration.

### 2.8 One dict key, two meanings: `OBSERVATION_ID`
`generate_pds4_files.py:1660` sets `ret['OBSERVATION_ID'] = root_obsid` (chunk-stripped, from the observation list); `xml_add_pds3_label_info` at 1881 then overwrites it with the PDS3 label's `OBSERVATION_ID`. The global-index `cassini:observation_id` column and the comment-note text both read the clobbered value. This is why the index can show `IOSIC_276RB_COMPLITB3001_SI` for products filed under `iosic_276rb_complitb4001_si` — the PDS3 labels of the source images say B3001 while the bundle's observation list says B4001. If that mismatch is intentional (real Cassini obsid vs. curated grouping name), it deserves a comment in the code and a sentence in the guide; the guide's §7 examples silently reproduce the discrepancy, which will confuse users.

### 2.9 Stale wraparound text in every data label
`generate_pds4_files.py:2082-2083` and 2326-2327: the label boilerplate still says "If the reprojection wraps around then they will be 0 and 359.98", but the CHANGELOG explicitly changed to wrapped min/max semantics (min > max possible), and the code writes wrapped values (1931-1932). The guide (§7) documents the *new* semantics; the generated labels describe the *old* ones.

### 2.10 Browse label text disagrees with the actual browse images ⚠ VERIFIED IN BUNDLE
`generate_pds4_files.py:2963` label text: "full (18000x401), med (1800x400), **small (400x400)**, and thumb (100x100)" — actual small mosaics are 200×200 (sizes table at 2813-2816; PNGs confirmed 200×200). Related dead code: the tuple table at 2975-2978 (`('small', 1, 45, 400, 400)`) carries downsample/crop values that the loop body never uses — stale numbers drifted from the live `sizes` table.

### 2.11 Params-table ET vs. label ET can disagree
`generate_pds4_files.py:2498`: for reprojected images the params table recomputes orbital quantities at `MIDTIME_ET` (derived from the PDS3 label start/stop), while `add_orbital_metadata` (816-825) computed the same quantities at `metadata['time']` (the reprojection's own ET) for the label and global index. Any difference between the two times makes table, label, and index mutually inconsistent for the same product.

### 2.12 Reprojection grid parameters hardcoded in one path
`generate_pds4_files.py:690`: `img_to_repro_path` bakes `_140220_-01000_001000_05.000_0.020_10_1-REPRO.DAT` into the filename while everything else derives paths from the configurable arguments. Running with non-default grid parameters would silently mix grids.

### 2.13 Old-format inputs fabricate zero longitudinal resolution
`generate_pds4_files.py:850-853, 927-930`: the old-format fallback sets `mean_angular_resolution = np.zeros(...)`, so `rings:longitudinal_resolution` would be published as 0.0 with no warning — a physically impossible value. Should log loudly or refuse.

---

## 3. Low-severity / code-quality issues in `generate_pds4_files.py`

- **Duplicate function:** `fixup_byte_to_str` is defined twice (715-736, 783-805); the first is dead, references an undefined global in its error path, and both pass a stray positional arg to `LOGGER.error`.
- **Dead orbit constant / duplicated orbit code:** `FRING_ORBIT_EPOCH` (483) is never used; the orbit functions (476-529) duplicate `f_ring_util/f_ring.py:307-346`, and the file *mixes* the two copies (869, 937 call the `f_ring` versions; `add_orbital_metadata` uses the local ones). The constants currently agree; the pattern invites drift. The local `fring_corotating_to_inertial` (501) appears unused.
- **Citation mismatch:** code comment says the orbit is "Albers 2012" (481, matching the guide §3.4.1's "Albers et al. 2012 Table 3 fit #2"), but the *label text* users will read says "Albers et al. (2009), fit #2" (2037, 2291). The archive text is presumably the wrong one.
- **`_image_has_satellite`:** 0.04° threshold is silently 2× the default longitude resolution (969); the "two valid longitudes on either side" check is off by one at the top end (`> len-2` should be `> len-3`, 973); satellite positions are recomputed via SPICE though `add_orbital_metadata` already stored them.
- **`et_to_tour` boundary bug:** (1165-1178) `et_to_datetime(et) <= '2004-12-24'` compares a full ISO timestamp against a bare date, so the boundary day itself lands in the next mission phase. Same for every subsequent boundary.
- **Asymmetric error handling:** `max_label` read (1808-1812) lacks the `ParseException` handler its `min_label` twin has (1794-1796); unknown camera does `sys.exit(-1)` (1768) where every comparable per-obsid failure raises `ObsIdFailedException`.
- **Hidden magic numbers:** `num_valid_longitudes/180` for percent coverage (2646) works only because 18000/100 = 180; `'359.98'` literals; `1000-(400-...)*5` background-limit math (2200-2202); browse-size tables.
- **Dead `try/except FileNotFoundError`** around a dict assignment (2985-2988) — leftover from code that presumably stat'ed the PNG; as written, labels are generated even for missing browse PNGs.
- **~225-line `if False:` block** in `write_suppl_file` (1305-1531) with per-image hardcoded matrices and a mid-function `return 0` — dead code that obscures the live path.
- **Hardcoded environment paths:** `/data/pdsdata/holdings/calibrated` (279-281), `kdir = '/home/rfrench/DS/Shared/OOPS-Resources/SPICE'` (532), and a font path loaded at *import time* (3452-3464) that kills the script on any machine without `NimbusSans-Bold.otf` even for runs that never make browse images. `templates/` and `observation_list.csv` are cwd-relative.
- **Python-version drift:** `datetime.timezone.utc` (1645) vs `datetime.UTC` (3469, Python ≥3.11 only); `NOW` (3469) unused; `numpy` re-imported mid-file (612).
- **Fragile CSV parsing:** (1620-1621) unpacks exactly 8 unquoted fields; a comma in "Additional Comments" crashes the run. Quality codes are dict-indexed with no validation (2021, 2123-2126).
- **Duplicate log handlers:** WARNING and ERROR file handlers on the same path with `rotation='ymdhms'` (397-400) — errors written twice, potentially to differently-stamped files.
- **Misc:** LID-helper docstrings show the wrong bundle name `fring_mosaic_rsfrench2025` (1085-1162); `downsample()`/`pad_image()` are dead (633-670); `populate_template` docstring documents a nonexistent parameter; `reslice_reproj_img`'s non-wrap branch (1052-1072) returns correctly only via an accidental modulo cancellation.

---

## 4. Template issues (`templates/*.lblx`)

### High
- **Dangling internal reference:** `data_reproj_img.lblx:209` — `<local_identifier_reference>image</local_identifier_reference>` inside `rings:Ring_Reprojection`, but the array's identifier is `reproj_image` (line 311). Present in every generated reproj label. The Display_Settings reference (178) uses the correct name, and the mosaic template is self-consistent — this is copy-drift. Likely a validation error; certainly broken semantics. ⚠ VERIFIED
- **File count wrong:** `f-ring-mosaics-user-guide.lblx:242` — `<files>5</files>` but six `Document_File` entries follow (PDF + 5 Python scripts). Present in the deployed label. ⚠ VERIFIED

### Medium
- **Contributor sequence numbers:** in `bundle.lblx` (and repeated in `global_index.lblx`, all `collection_*.lblx`, `f-ring-mosaics-user-guide.lblx`), the four `List_Contributor` Persons are numbered 1, 2, 2, 2 (Gordon, Tiscareno, Simpson all `<sequence_number>2</sequence_number>`); should be 1-4. ⚠ VERIFIED in `bundle.lblx:60,78,96,114`
- **`kernels.lblx`** is missing the `<?xml version="1.0" encoding="UTF-8"?>` declaration every other template has; its `Observing_System` names only the WAC although the metakernel covers NAC+WAC processing; and it mixes `collection_to_investigation` (line 54) with `data_to_target` (82, 90) reference types in one product label.
- **Observing_System drift between the two data templates:** `data_reproj_img.lblx:77` says "Cassini Orbiter Imaging Science Subsystem" (generic) while `data_mosaic.lblx:73` says "Cassini Imaging Science Subsystem - $CAMERA_WIDTH$ Angle Camera". Parallel products describe the same system two different ways (and per finding 1.1, the substituted value is always "Wide").
- **True-anomaly descriptions in the reproj global index** (`global_index.lblx:1031,1042`) say "over the image mid-time span" — a single reprojected image has one mid-time; the min/max is across valid longitudes (as the guide §7 correctly says).

### Low
- `browse_reproj_img.lblx` has a `<keyword>browse products</keyword>` its mosaic twin lacks; `bundle.lblx:16` hardcodes `<publication_year>2025</publication_year>` where other templates use `$PUBLICATION_YEAR$`; Hedman's ORCID uses `http://` while all others use `https://`; collection titles inconsistently end in a period; date macros differ (`$CURRENT_DATE$` vs `$CURRENT_ZULU(date_only=True)$`).
- **LID vs filename word-order flip:** products are `miscellaneous:mosaic_global_index` but files/local_identifiers are `global_mosaic_index.*`. Legal but a foot-gun; the guide only ever uses the file names.
- `templates/readme.txt` never mentions the background-subtracted mosaic collections, which the guide treats as a headline product type. Its citation reads "French, R.S. and Hedman, M.M." while the guide's title page lists only Robert S. French — worth aligning.

---

## 5. Example-script issues (`templates/examples/`, shipped in `document/user_guide/`)

### High — degrees passed to `np.cos`: the equivalent-width plots are quantitatively wrong ⚠ VERIFIED IN SOURCE
`plot_ews_df.py:63-64` and `plot_ews_ma.py:65-75`:

```python
emission = metadata_df['rings_emission_angle'].to_numpy()   # degrees, per the label
mu = np.abs(np.cos(emission))                               # np.cos expects radians
```

The labels declare `rings:emission_angle` in degrees (values ~122-126° in the test mosaic). `|cos(123.7 rad)| = 0.361` vs. the correct `|cos(radians(123.7°))| = 0.555` — a ~35% error that varies nonlinearly with angle, so the EW×μ profiles produced by the very scripts the guide showcases (its `screenshot-plot-ews.png` figures) are wrong, not just rescaled. Fix: `np.cos(np.radians(emission))` in both scripts. These files ship inside the archive itself, so this is worth fixing before delivery.

### Low
- All four runnable scripts' docstrings say they "must be run from the ``examples`` directory," but in the delivered bundle they live in `document/user_guide/` and no `examples` directory exists.
- `imshow` extents treat pixel centers as edges (`display_reproj_img.py:73`, `plot_ews_ma.py:156`, `plot_ews_df.py:144`, `find_prometheus_closest_approaches.py:235`) — half-pixel (0.01°, 2.5 km) misregistration; the Prometheus marker is plotted in these shifted coordinates.
- `find_prometheus_closest_approaches.py:169-174` sorts on signed (not absolute) core−Prometheus separation; works only because Prometheus is always interior to the core.

### Verified working (for the record)
All structure IDs, column names (after pds4_tools's `:`→`_` mapping), the `+1` slice bounds, and the wraparound reconstruction were validated against the real bundle; scripts have no hardcoded paths, complete dependency docstrings, and no deprecated-API problems under numpy 2.4 / pandas 3.0.

---

## 6. Software ↔ users guide disagreements

Where the two disagree, the likely-correct side is noted.

| # | Topic | Guide says | Software/bundle does | Likely fix |
|---|-------|-----------|----------------------|------------|
| 6.1 | Supplemental filename | `IMGID_reproj_img_suppl.txt` (§2 L29, §3 L251, §4 L83+L178) | `IMGID_reproj_suppl.txt` (all 20,315 files + labels) | Fix guide (renaming archive files is costlier); or rename for consistency with every other product suffix |
| 6.2 | Browse mosaic filenames | `OBSID_browse_mosaic[_bkg_sub]_img.lblx` / `_img_*.png` (§2 L41-43) | No `_img` component: `..._browse_mosaic_full.png` | Fix guide (spurious `_img`) |
| 6.3 | Reprojected-image count | 20,303 (§1 L14, §3 L61) | 20,315 products (index rows, `.lblx` files, and `.img` files all agree) | Fix guide (count drifted after it was written); mosaic count 302 matches |
| 6.4 | "Med" browse height | "always 401 pixels in the Y dimension… maximum size 1800×401" (§4 L223, L296) | Med images are 400 px tall (code `sizes` tables at 2814/2820; PNGs confirmed 1800×400, 497×400) | Decide: either resize to 401 in code or fix guide to 400. Note 401→400 also slightly distorts the radial scale |
| 6.5 | Small/thumb "padded" | "downsampled **and padded** as necessary" (§4 L223) | `pil_img.resize((width, height))` — plain anisotropic stretch, no padding; the `pad_image()` helper exists but is dead code (633-670) | Align guide wording with the code (or restore padding) |
| 6.6 | Full reproj browse width | "always the same size as the reprojected image" (§4 L223) | `width = max(shape[1], 800)` — images narrower than 800 px are stretched (2859) | Document the 800 px floor in the guide |
| 6.7 | `src_imgs` table header | `Source Image Index, LIDVID` (§4 L276) | `image_index,LIDVID` (template `data_mosaic.lblx:252` and actual files) | Fix guide excerpt |
| 6.8 | Example script name | `plot_ews_ma_df.py` (§5 L108) | Ships `plot_ews_df.py` | Fix guide |
| 6.9 | mosaic_utils function names | `get_reproj_name_from_label`, `get_mosaic_name_from_reproj_img` (§5 L101-102) | Actual: `get_reproj_img_name_from_label`, `get_mosaic_name_from_reproj_img_label` | Fix guide |
| 6.10 | `read_*_ma` metadata type | "metadata returned as a Python dict" (§5 L98) | Structured numpy array (dict is only the LIDVID map) | Fix guide |
| 6.11 | Array local_identifier | `<local_identifier>image</local_identifier>` (§4 L135 excerpt) | `reproj_image` — though see the dangling `image` reference bug in §4 above; the two errors are related | Fix template ref → then fix guide excerpt |
| 6.12 | document/ tree | §4 L307-314 lists only `.lblx` + `.pdf` under `user_guide/` | Also contains the five `.py` example programs (which §4 L320 and §5 L91 do mention) | Fix guide tree listing |
| 6.13 | Node regression rate sign | "Ω̇ = 2.68778 °/day" (§3 L277, positive) | Label text: "Omega regression rate = -2.68778 deg/day" | State the sign convention on one side |
| 6.14 | Wraparound min/max semantics | min > max possible, wraps at 360° (§7, correct) | Label boilerplate still says wrap → "0 and 359.98" (see finding 2.9) | Fix generator text |
| 6.15 | §3 processing lists | Omit `longitude_ascending_node` from the per-longitude parameter lists (§3 L314-322, L333-348) | Column exists in tables and §7 documents it | Fix guide lists |
| 6.16 | Shipped guide PDF | Current LaTeX build: `main.pdf`, 1.87 MB, built 2026-07-19 | `templates/f-ring-mosaics-user-guide.pdf`: 0.70 MB, dated 2025-09-23 — ~10 months stale | Replace the template PDF with the current build before the next bundle generation |

Verified consistent (no action): orbit elements and corotation rate (581.964°/day = 0.006735694444°/s), epoch ET 220881665.1839181, array geometry (401 radial × N longitudinal, Last Index Fastest, IEEE754LSBSingle, I/F, −999 sentinel), metadata-params and all three global-index field sets (names, order, and record lengths match §7 exactly), collection/bundle structure, the M1/M2/M3/M4/B/C/E/O/R/N flag assignments (all 153 flagged mosaics in the index match the guide's per-flag lists, including the combination flags M4;R, M4;O, M3;R, E;M2, B;N).

---

## 7. Guide-side errata (independent of the software)

1. **Mojibake in §4 label excerpts** (docx→LaTeX damage): `un”t="b”te"` for `unit="byte"` (L130, L136), `unit=”km"` (L106-107), and `—--` where `<!--` belongs (L94, L133, L144, L149, L154). These render as garbage in the published PDF; the templates themselves are correct.
2. **The "N" note is not a list item** (§4 L359): missing `\item`, so it typesets as a continuation of the "R" bullet.
3. **Copy-pasteable commands aren't** (§5 L114, L120): the example command lines wrap paths across lines with embedded space runs and no shell continuation; pasting them yields three arguments and a usage error.
4. **§3.1 says INST is "always 'ISS'"** but the bundle (and the guide's own §7 examples) include `IOSIC_276RB_COMPLITB4001_SI`. A sentence explaining the IOSIC joint-observation prefix would help.
5. **§7 example inconsistency**: `cassini:observation_id` example is `IOSIC_276RB_COMPLITB3001_SI` while the LID/file_spec examples use `complitb4001` — the B3001/B4001 mismatch is real in the data (see finding 2.8) but is presented without explanation.
6. Guide title page lists only Robert S. French; the bundle's `List_Author` and the readme citation include M.M. Hedman. Probably intentional (guide vs. data authorship) but worth a conscious check.

---

## 8. Items already tracked in `TODO.txt` (not re-counted as findings)

kernels.lblx / collection_spice_kernels updates; document collection file; full List_Author/List_Contributor classes; navigation check of `iss_199rf_fmovie002_prime`; bundle.xml; Prometheus/Pandora in reprojected images (nb: the sign bug in finding 1.2 is *not* what this TODO describes); star targets for occultations; per-collection camera/target limiting; the Cassini-LDD diffs; rings-dictionary description; wraparound longitude limits (done per CHANGELOG — only the stale label text of finding 2.9 remains).

---

## 9. Suggested priority order

1. **Before any regeneration:** fix 1.1 (camera), 1.2 (Pandora), 2.1 (index min ring radius), 2.3 (miscellaneous inventory), 5-High (emission-angle radians in both EW scripts), template `<files>5</files>` and the dangling `image` reference — these all corrupt or invalidate delivered archive content.
2. **Same pass, cheap:** 2.2 (suppl preamble), 2.9/2.10 (stale label text), contributor sequence numbers, kernels.lblx XML declaration, refresh the shipped user-guide PDF (6.16).
3. **Guide edits:** rows 6.1-6.15 above plus the §7 errata — most are one-line fixes in the LaTeX sources.
4. **Hardening for future runs:** 1.3 (remap), 1.4 (crossed flags), 2.4-2.6 (collection/CSV gating), 2.12 (hardcoded grid), CSV parsing, font-at-import.
