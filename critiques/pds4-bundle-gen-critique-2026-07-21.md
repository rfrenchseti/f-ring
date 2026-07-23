# Critique: `pds4_bundle_gen` — follow-up review

**Date:** 2026-07-21
**Software reviewed:** `/seti/research/f-ring/f-ring/pds4_bundle_gen/` — `generate_pds4_files.py` (3,817 lines), `templates/*.lblx`, `templates/examples/*.py`, `templates/*.csv`
**Users guide compared against:** `/seti/research/f-ring/f-ring/users_guide/sections/*.tex` (LaTeX sources; `main.pdf` built 2026-07-19, md5 `e6e7eef933…`, shipped identically in the bundle)
**Ground truth:** the regenerated bundle at `/data/fring-bundles/pds4/` (rebuilt 2026-07-21 18:23).
**Predecessor:** `critiques/pds4-bundle-gen-critique-2026-07-19.md`. Line numbers refer to the current files.

---

## Executive summary

Excellent progress. **All four high-severity generator bugs are fixed** (camera attribution, the Pandora sign error, `remap_image_indexes`, and the crossed browse-label flags), **both high-severity template bugs are fixed** (the dangling `image` array reference and `<files>5</files>`), and the **high-severity science bug in the example scripts is fixed** (degrees→radians in both EW plotters). Most medium/low findings are also resolved, several verified directly in the rebuilt bundle (NAC products now correctly say "Narrow Angle Camera"; 24 labels now carry a Pandora `Target_Identification`; the global-index minimum ring radius now matches the product labels; browse text now says 200×200).

However, the fix for the old miscellaneous-collection findings introduced **two new PDS4 validation errors** in that same collection, and there is **one new latent crash** from a copy-paste. A handful of medium/low items remain open, and five guide-vs-bundle disagreements persist (the product counts drifted again after the rebuild). Nothing here re-corrupts the science content of the archive, but the two new miscellaneous-collection errors **will fail `validate`** and must be fixed before delivery.

---

## 1. New issues (introduced since 2026-07-19)

### 1.1 ⚠ HIGH — `miscellaneous` inventory LIDs don't match the products (dangling references + orphans)
The LID constants were renamed to put `global_` first (`generate_pds4_files.py:53-55`), which correctly fixed the old LID-vs-filename word-order finding — the three products now declare:

```
…:miscellaneous:global_mosaic_index
…:miscellaneous:global_mosaic_bkg_sub_index
…:miscellaneous:global_reproj_img_index
```

But the **static** inventory `templates/collection_miscellaneous.csv` (copied verbatim into the bundle at `generate_pds4_files.py:3416` via `copy_file`) was *not* updated and still lists the old word order:

```
P,…:miscellaneous:mosaic_bkg_sub_global_index::1.0
P,…:miscellaneous:mosaic_global_index::1.0
P,…:miscellaneous:reproj_img_global_index::1.0
```

Verified in `/data/fring-bundles/pds4/miscellaneous/collection_miscellaneous.csv` vs the three `*_index.lblx` `<logical_identifier>` values. **None of the three primary inventory members resolve to a product, and all three real global-index products are orphaned.** `validate` will flag both. Fix: regenerate/rewrite `templates/collection_miscellaneous.csv` to the `global_*` order (or, better, generate those three rows from the LID constants instead of shipping a static file, so they can never drift again).

### 1.2 ⚠ HIGH — `collection_miscellaneous.lblx` `<records>` mismatch (12 vs 5)
`templates/collection_miscellaneous.lblx`:

```
141:  <records>$FILE_RECORDS(COLLECTION_MISCELLANEOUS_CSV_PATH)$</records>   # File → 12 (correct)
147:  <records>5</records>                                                   # Inventory → hardcoded, stale
```

The `<File_Area_Inventory>/<File>/<records>` is computed correctly (12, matching the 12-row CSV), but the sibling `<Inventory>/<records>` is a hardcoded `5` (left over from when the collection had 5 rows). The two must be equal. Compare any sibling collection (e.g. `collection_data_mosaic.lblx` uses the row count, 300, in both places). Fix: line 147 should use the same `$FILE_RECORDS(...)$` macro (→12).

### 1.3a MEDIUM (decision needed) — two observations were silently dropped by the new OBSERVATION_ID check
The mosaic count fell from **302** (build `pds4-full-260518`) to **300**. The two dropped observations are `iosic_276rb_complitb4001_si` and `iss_134ri_spkmvdfhp001_prime`. Both were rejected by the new consistency check added to fix old finding 2.8 (`generate_pds4_files.py:1827-1829`, which now `raise ObsIdFailedException`): their source-image PDS3 labels carry a different `OBSERVATION_ID` than the curated obsid grouping —

```
IOSIC_276RB_COMPLITB4001_SI: label says IOSIC_276RB_COMPLITB3001_SI  (B3001 ≠ B4001)
ISS_134RI_SPKMVDFHP001_PRIME: label says ISS_134RI_SPKMVDFHP003_PRIME (003 ≠ 001)
```

This is the exact B3001/B4001 discrepancy documented in the old critique (§2.8) and guide §7. The check now **discards the whole observation** rather than publishing a mismatched ID. These are valid mosaics that shipped in prior bundles, so this is a **decision**, not a defect: either (a) accept the drop and update the guide's counts / note the exclusion, or (b) relax the check to a warning for the known joint-observation (IOSIC / renumbered) cases so the observations are retained with a documented ID convention. Confirm which is intended before delivery.

### 1.3 MEDIUM — duplicate unconditional `LATEST_STOP_DATE_TIME` assignment reintroduces the support-only crash
`generate_pds4_files.py:3383-3391` (inside `generate_support_files()`):

```python
if LATEST_STOP_DATE_TIME is None:
    metadata['LATEST_STOP_DATE_TIME'] = 'N/A'
else:
    metadata['LATEST_STOP_DATE_TIME'] = et_to_datetime(LATEST_STOP_DATE_TIME)
metadata['LATEST_STOP_DATE_TIME'] = et_to_datetime(LATEST_STOP_DATE_TIME)   # 3391: clobbers the guard
```

Line 3391 unconditionally re-runs `et_to_datetime(LATEST_STOP_DATE_TIME)`, defeating the `None` guard that was added to fix old finding 2.6. On a `--generate-support-files`-only run (no per-obsid labels → `LATEST_STOP_DATE_TIME` stays `None`), this calls `et_to_datetime(None)` and crashes. The `EARLIEST_START_DATE_TIME` block just above has no such duplicate — this is a copy-paste leftover. Fix: delete line 3391.

### 1.4 LOW — `collection_spice_kernels.lblx` hardcodes the publication year
Old finding (§4 Low) flagged `bundle.lblx` hardcoding `<publication_year>2025</publication_year>`; that is now `$PUBLICATION_YEAR$`. But `collection_spice_kernels.lblx:16` still hardcodes `2025` — the only one of the 17 templates that doesn't use the macro. The bug just relocated.

### 1.5 LOW — `kernels.lblx` residual reference-type / naming issues
XML declaration and the dual NAC+WAC `Observing_System` were fixed, but two issues remain: the product label mixes `collection_to_investigation` (line ~55) into a `Product_SPICE_Kernel` (should be `data_to_investigation`), and the `Observing_System` `<name>` still reads "…Wide Angle Camera" even though the block now lists both NAC and WAC components.

---

## 2. Status of prior findings

### 2.1 High-severity generator bugs — ALL FIXED ✅
| # | Prior finding | Status | Evidence |
|---|---------------|--------|----------|
| 1.1 | Camera always "Wide" | **FIXED** | `camera = image_name0[-1]`; `if camera == 'n'` (1707-1724). Bundle: the all-NAC `iss_036rf_fmovie001_vims` now says "Narrow Angle Camera" / `issna.co`. |
| 1.2 | Pandora sign error | **FIXED** | `radius_sat_dist > -arguments.radius_outer_delta` (927). Bundle: 24 labels now carry a Pandora `Target_Identification` (110 Prometheus); was zero. |
| 1.3 | `remap_image_indexes` wrong index | **FIXED** | `image_name_list[x]` / `image_path_list[x]` over old-index keys (986-991). |
| 1.4 | Crossed browse-label flags | **FIXED** | `img_type=='r'`↔`REPROJ`, `!='r'`↔`MOSAIC` (2930-2931). |

### 2.2 Medium-severity generator bugs
| # | Prior finding | Status | Notes |
|---|---------------|--------|-------|
| 2.1 | Index min ring radius used the mean | **FIXED** | `MIN_RING_RADIUS_FIXED` now uses `min_radius` (1696). Bundle: index and label both 139439.689 for the test mosaic. |
| 2.2 | Suppl preamble discarded | **FIXED** | `hdr_text` built additively incl. the C-matrix preamble (1505-1534); bundle suppl file confirmed complete. |
| 2.3 | `miscellaneous` inventory never generated | **FIXED (but see 1.1, 1.2)** | `generate_miscellaneous_support_files()` now does `copy_file` + `populate_template` (3406-3418, called 3719). The .csv/.lblx now exist — but with the two new defects above. |
| 2.6 | Support-only garbage dates | **FIXED (but see 1.3)** | `None`-sentinels + `'N/A'` guards (3128-3135, 3205-3212, 3383-3390) — except the reintroduced clobber at 3391. |
| 2.7 | bkg-sub mask zeroed | **RESOLVED (deliberate)** | Now documented ("missing data … already converted to the sentinel value", 803-806) and a verification item was added to `TODO.txt`. |
| 2.8 | `OBSERVATION_ID` double meaning | **FIXED** | Distinct keys `FULL_OBSERVATION_ID` / `MOSAIC_OBSERVATION_ID_ROOT` (1601-1602) plus a mismatch warning (1827-1828). The underlying B3001/B4001 data discrepancy still exists but is now logged, not silently clobbered. |
| 2.9 | Stale wraparound label text | **FIXED** | Labels now say "the minimum will be greater than the maximum" (2037, 2283). |
| 2.10 | Browse text vs actual sizes | **FIXED** | "small (200x200)" (2884, 2920); bundle PNGs confirmed 200×200. |
| 2.12 | Hardcoded reproj grid path | **FIXED** | Filename now derived from `arguments` (648-651); the old literal survives only in the docstring example. |
| 2.11 | Params-table ET vs label ET | **NOT AN ISSUE** | `metadata['time']` for a reprojected image is `offrepdata.repro_time` = the OOPS `obs.midtime`, i.e. the image midtime — the same physical quantity as `MIDTIME_ET` (the PDS3-label start/stop midpoint). They agree to sub-second precision, so the params table (uses `MIDTIME_ET`) and the label/index (use `metadata['time']`) are consistent. Withdrawn per user confirmation. |
| 2.13 | Old-format zero longitudinal resolution | **PARTIALLY FIXED** | Mosaic reader now *raises* `ObsIdFailedException` on old-format input (787-797). The reproj reader (`read_reproj`, 869-872) still silently sets `mean_angular_resolution = np.zeros(...)`. Harmless as long as current inputs are new-format, but latent. |
| 2.4 | Failed obsids still in collection CSVs | **NOT RE-VERIFIED** | Not manifest in this build (300 products = 300 CSV rows). Still worth the gating fix. |
| 2.5 | `--generate-reproj-collections` alone → empty | **NOT RE-VERIFIED** | Gating not re-audited. |

### 2.3 Low-severity generator items
- **FIXED:** duplicate `fixup_byte_to_str` (now one definition, 720); Albers citation (label text now "Albers et al. (2012)", 1991/2247 — matches the guide and the code comment); `et_to_tour` boundary (now compares full `YYYY-DDDTHH:MM:SS.mmm` timestamps, 1114-1120).
- **ACCEPTED (won't fix):** hardcoded absolute paths and the font-at-import path (3428-3435) — the generator is run by one person on one machine, so these are fine. (Note a `cmunssdc.ttf` was added to `templates/` but the code still references the system NimbusSans path — harmless.) Duplicate WARNING/ERROR log handlers writing the same file (378-401) also remain but are cosmetic.

### 2.4 Template issues (`templates/*.lblx`)
| Prior finding | Status |
|---------------|--------|
| Dangling `image` array reference (`data_reproj_img.lblx`) | **FIXED** — all references now `reproj_image` (178, 209, 311); verified in bundle. |
| `<files>5</files>` in user-guide label | **FIXED** — now `<files>6</files>` with 6 `Document_File` entries. |
| Observing_System drift reproj vs mosaic | **FIXED** — both now "…- $CAMERA_WIDTH$ Angle Camera". |
| Reproj global-index true-anomaly "over the image mid-time span" | **FIXED** — now "over valid longitudes" (1031, 1042). |
| `kernels.lblx` missing XML declaration / WAC-only Observing_System | **FIXED** (declaration + dual NAC/WAC); reference-type + name residue remains → see 1.5. |
| Hedman ORCID `http://`; `browse_reproj` extra keyword | **FIXED.** |
| LID vs filename word-order | **"FIXED" in code** (LIDs + filenames agree on `global_mosaic_index`) — **but broke the static inventory CSV → 1.1.** |
| **Contributor sequence numbers 1,2,2,2** | **NOT A BUG (withdrawn)** — the 1,2,2,2 numbering (Mace=1; Gordon/Tiscareno/Simpson=2) is intentional per user confirmation. |
| `rings:minimum/maximum_incidence_angle` = `$MEAN_INCIDENCE_ANGLE$` | **NOT A BUG** — deliberate and documented (`data_mosaic.lblx:405-409`: "Due to the slow rate of change of the incidence angle, we do not compute a separate incidence angle for each corotating longitude"). |

### 2.5 Example scripts (`templates/examples/`)
| Prior finding | Status |
|---------------|--------|
| **Degrees passed to `np.cos`** (EW plots wrong) | **FIXED** — `np.cos(np.radians(...))` in `plot_ews_df.py:64` and `plot_ews_ma.py:75`; these are the only trig calls in all five scripts. |
| Docstrings said "run from the `examples` directory" | **FIXED** — now say `document/user_guide`, matching the bundle. |
| `imshow` extents treat pixel centers as edges | **ACCEPTED (won't fix)** — half-pixel offset deemed irrelevant. |
| Signed (not absolute) core−Prometheus sort in `find_prometheus_closest_approaches.py` | **ACCEPTED (won't fix)** — correct given Prometheus is always interior; works with the code. |

### 2.6 Guide ↔ bundle disagreements
**Now agree (fixed):** 6.1 suppl filename, 6.2 browse names, 6.6 800 px width floor (now documented), 6.7 `src_imgs` header, 6.8 `plot_ews_df.py`, 6.9 mosaic_utils fn names, 6.10 "structured numpy array", 6.13 node-rate sign (guide now −2.68778), 6.16 PDF (identical build shipped), plus the §7 mojibake and the §3.1 "INST always ISS" softening.

**Still disagree:**
| # | Guide | Bundle | Fix |
|---|-------|--------|-----|
| 6.3 | "20,303 reprojected images … 302 … mosaics" (§1 L14) | **20,291** reproj products, **300** mosaics | Update the guide counts (they drifted further after the rebuild; note mosaics dropped 302→300 — confirm this is intentional). |
| 6.4 | "1800×401" for Med browse (§4 L223, L296) | Med PNG = 1800×**400** | The Y-dim prose was fixed to 400 but "1800×401" still prints twice. |
| 6.5 | Small "downsampled and **padded**" (§4 L223) | plain `resize()` stretch, no padding | Align wording with code. |
| 6.12 | §4 tree lists 4 `.py`, omits **`mosaic_utils.py`** | bundle ships 5 `.py` incl. `mosaic_utils.py` | Add it to the tree. |
| 6.15 | §3 mosaic parameter list omits "Longitude of Ascending Node" | column present in the table and reproj list | Add it to the mosaic list. |

---

## 3. Spot-check of the rebuilt bundle (validation-oriented)

One product of each type inspected; all array geometry, offsets, record lengths, `<file_size>`, and `<records>` were self-consistent and the XML well-formed, **except** the two miscellaneous-collection defects (1.1, 1.2). Highlights:

- **Mosaic** `data_mosaic/iss_036rf_fmovie001_vims`: 401×18000×4 = 28,872,000 B matches `.img`; NAC/`issna` correct; F Ring + Saturn Rings + **Pandora** targets present. `src_imgs.tab` and `metadata_params.tab` sizes check out.
- **Bkg-sub mosaic**: same dims, same radii as the plain mosaic, NAC correct.
- **Reprojected image** `data_reproj_img/iss_029rf_fmovie001_vims/1538168640n`: 401×368×4 = 590,272 B; array id and both `local_identifier_reference`s all `reproj_image`; suppl header 809 B with full preamble.
- **Browse** `browse_mosaic/iss_036rf_fmovie001_vims`: label text (full 18000×401, med 1800×400, small 200×200, thumb 100×100) matches actual PNG dimensions.
- **All 11 collections** have both `collection_*.csv` and `collection_*.lblx`.
- **Global index** `miscellaneous/global_mosaic_index.tab`: 300 rows; minimum/maximum ring radius match the product label exactly (old 2.1 mismatch gone).
- **Counts:** 20,291 reproj `.img`/`.tab`/`.txt`/`.lblx` (the 20,292nd `_reproj_img.lblx` is the collection label — not an orphan); 300 mosaic products.

---

## 4. Suggested priority order

1. **Before any delivery / re-run of `validate`:** fix 1.1 (miscellaneous inventory LIDs — the static `collection_miscellaneous.csv`) and 1.2 (`<records>5</records>` → macro). These two **will fail PDS4 validation**.
2. **Same pass, cheap:** 1.3 (delete the duplicate line 3391), 1.4 (spice-kernels publication year macro), 1.5 (kernels reference-type / name).
3. **Decision:** 1.3a — confirm whether the two dropped observations (`complitb4001`, `spkmvdfhp001`) should stay dropped or be retained via a relaxed OBSERVATION_ID check.
4. **Guide edits:** 6.3 counts (20,291 / 300), 6.4 (1800×400 ×2), 6.5 (padding wording), 6.12 (add `mosaic_utils.py`), 6.15 (add ascending node to the mosaic list).
5. **Hardening for future runs:** 2.13 (reproj old-format guard), 2.4/2.5 (collection/CSV gating).

---

## Appendix A — Source-image tables corrected by the `remap_image_indexes` fix (finding 1.3)

Diffing the source-image metadata tables (`<obsid>_mosaic_metadata_src_imgs.tab` and the `_mosaic_bkg_sub_` twin) between the current bundle (`/data/fring-bundles/pds4`, 2026-07-21) and the immediate predecessor (`/data/fring-bundles/pds4-full-20260518`, 2026-05-18):

- **267** of the 300 common mosaic observations are byte-identical.
- **33** changed — in each, the image **count is unchanged**; a contiguous block of images was replaced by an equal-sized block later in time (the old buggy compaction took the first *N* contiguous entries instead of the actually-used indexes, so any mosaic that dropped an interior image got the wrong LIDVID list). The "Imgs corrected" column is the number of interior-dropped images that were previously mislisted. Background-subtracted twins changed for the same 33.
- **2** observations present only in the old build (`iosic_276rb_complitb4001_si`, `iss_134ri_spkmvdfhp001_prime`) were dropped by the new OBSERVATION_ID check (finding 1.3a) — unrelated to the source-image fix.

| # | Observation | Total imgs | Imgs corrected |
|---|---|---|---|
| 1 | `iss_174ri_spokemov002_prime` | 44 | 44 |
| 2 | `iss_260rf_fmovie001_prime` | 264 | 41 |
| 3 | `iss_000ri_satsrchap001_prime` | 79 | 38 |
| 4 | `iss_213rf_fmovie001_prime` | 116 | 37 |
| 5 | `iss_044rf_fmovie001_vims` | 134 | 33 |
| 6 | `iss_173ri_spokemov003_prime` | 131 | 32 |
| 7 | `iss_289rf_fmovie001_prime` | 127 | 24 |
| 8 | `iss_246rf_fmovie002_prime` | 87 | 19 |
| 9 | `iss_199rf_fmovie002_prime` | 23 | 17 |
| 10 | `iss_041rf_fmovie002_vims` | 146 | 17 |
| 11 | `iss_262rf_fmovie001_prime_12` | 17 | 16 |
| 12 | `iss_253rf_fmovie001_prime_2` | 130 | 13 |
| 13 | `iss_173rf_hiresfrng001_prime` | 45 | 11 |
| 14 | `iss_100ri_subms20lp001_cirs` | 24 | 10 |
| 15 | `iss_181rf_fmovie001_prime` | 131 | 9 |
| 16 | `iss_256ri_hiresafrg002_prime` | 51 | 8 |
| 17 | `iss_105ri_tdifs20hp001_cirs` | 21 | 8 |
| 18 | `iss_036rf_fmovie001_vims` | 109 | 8 |
| 19 | `iss_211rf_fmovie001_prime` | 101 | 7 |
| 20 | `iss_174ri_spokemov001_prime` | 92 | 7 |
| 21 | `iss_091ri_apomosl109_vims` | 23 | 5 |
| 22 | `iss_075rf_fmovie002_vims` | 106 | 5 |
| 23 | `iss_043rf_fmovie001_vims` | 82 | 5 |
| 24 | `iss_109ri_tdifs20hp001_cirs` | 19 | 4 |
| 25 | `iss_039rf_fmovie001_vims` | 135 | 4 |
| 26 | `iss_179rf_fmovie001_prime` | 140 | 3 |
| 27 | `iss_268rf_fmovie001_prime_1` | 16 | 2 |
| 28 | `iss_098ri_tmapn30lp001_cirs` | 24 | 2 |
| 29 | `iss_253ri_hiresafrg001_pie` | 40 | 1 |
| 30 | `iss_253rf_fmovie001_prime_3` | 5 | 1 |
| 31 | `iss_196rf_fmovie003_prime` | 129 | 1 |
| 32 | `iss_105ri_tmapn45lp001_cirs_4` | 1 | 1 |
| 33 | `iss_007ri_lphrlfmov001_prime` | 233 | 1 |
