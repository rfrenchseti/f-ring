# Full code review: `pds4_bundle_gen`, `f_ring_util`, `mosaics`, `pds4_mosaic_viewer`, `user_guide`

**Date:** 2026-08-04
**Scope:** every line of every project file in the five directories (≈25,000 lines), plus the
consumer/producer code they interact with (`photometry/`, `utilities/`, and the rms-nav
`nav/ring_mosaic.py` where the on-disk contract is defined there). Excluded: `user_guide/sty/`
(vendored third-party LaTeX packages), `archive/`, and — per instruction — the generated bundle
itself (it is mid-regeneration; a bundle spot-check should be a second pass).
**Emphasis:** correctness of intermediate and archived data; PDS4 validity; cross-program
consistency of units, masks, and sentinel values; code-vs-guide consistency.
**Method:** nine parallel line-by-line reviews by area, followed by independent re-verification of
every critical/major claim against the source (and, where marked, numerically or empirically).
**Predecessor:** `critiques/pds4-bundle-gen-critique-2026-07-21.md` (status of its findings in §6).

> **Fix status (updated 2026-08-04):** every finding in this report was fixed on branch
> `code_review_fixes` — commits `52cb449` (mosaics), `115574b` (f_ring_util + photometry),
> `fe4f8f5` (generator), `7f47619` (templates + examples), `ba555cb` (viewer),
> `1a85209` (user guide) — **except** the items explicitly marked **NOT FIXED** below:
> §2.2 (upstream in `/seti/nav/rms-csmithing`) was subsequently fixed there on branch
> `fix_mosaic_time_float64` (commit `f156fdb`); the `mean_incidence` item in §5 was withdrawn
> (the incidence angle is assumed constant over one mosaic, per user confirmation);
> and three deliberate deferrals remain
> (INST_CMPRS_PARAM/dead-key content decisions, collection-CSV context-member styling,
> and the stale `mosaics/obs_list.csv` snapshot, which needs a data run to regenerate).
> One additional bug was found and fixed during the fix pass; see the addendum at the
> end of §5.

Deliberately not re-flagged (accepted per prior review): hardcoded single-machine paths;
`metadata['time']` = midtime for reprojected images; imshow half-pixel extents; signed
closest-approach sort; contributor sequence 1,2,2,2; the strict OBSERVATION_ID failure check.

---

## Executive summary

Two **critical** findings: a coefficient swap in the quadratic background fit
(`ring_model_bkgnd.py`) that silently corrupts any degree-2 background-subtracted mosaic —
**verified numerically; a scan of all 305 production `BKGND-METADATA` pickles shows every
observation currently uses degree 1, so no production data is affected today** — and a wrong
core-radius equation printed in the user guide (the longitude term is missing entirely).

The largest cluster of **major** findings is in archive metadata correctness: the Cassini
`mission_phase_name` is wrong for every observation in the second half of each boundary year
(YMD-vs-DOY string comparison, verified empirically); per-column mosaic times are quantized to
16–64 s by an upstream float32 array, which propagates into archived inertial longitudes at up to
~10× the longitude bin size; the labels' `vertical_display_direction` contradicts the actual
array orientation; two document-collection label errors will fail `validate`; and several
flag-gating holes can produce inventories that disagree with the products actually generated.

The mosaic pipeline, viewer, and utility library each have a handful of real but narrower bugs
(details below). The heavily used core paths — grid construction, corotating-frame conversion,
best-pixel mosaic selection, sentinel/mask propagation, filename round-tripping, the linear
(degree-1) background fit, and the three copies of the F-ring orbit model — were checked closely
and are **sound and mutually consistent**.

---

## 1. Critical

### 1.1 `fit_quadratic` evaluates the model with constant and quadratic coefficients swapped — **FIXED** (`52cb449`)
`mosaics/ring/ring_model_bkgnd.py:269` (evaluation), 203–256 (solver).
The normal equations as written solve `y = a + b·x + c·x²` (so `a` is the constant and `c` the
quadratic coefficient), but the returned model is `a*x*x + b*x + c`. **Verified numerically**:
fitting exact data `y = 2x² + 3x + 5` returns the model `5x² + 3x + 2`. Any background computed
with degree 2 (selectable in the `ring_ui_bkgnd.py` GUI, persisted per-obsid in the
`BKGND-METADATA` pickle and silently re-applied on every recompute) is wrong by
`(a_const − a_quad)·(x² − 1)`-shaped error across the radial profile, corrupting the
background-subtracted mosaic, every EW derived from it, and the PDS4 products for that obsid —
with no warning anywhere.
**Production impact check:** all 305 `bkgnd_FMOVIE/*BKGND-METADATA.dat` files carry degree = 1
(the linear fit, which is correct), so no current data is affected. The bug is a landmine for the
first time degree 2 is ever used. Fix: `model = a + b*x + c*x*x` (or rename the solver outputs).

### 1.2 The user guide's core-radius equation is missing the longitude term — **FIXED** (`1a85209`)
`user_guide/sections/03-image-selection-and-processing.tex:293`.
The guide prints
`r_core = a(1−e²) / (1 + e·cos(ϖ₀ + ϖ̇·ET/86400))` — the cosine of the longitude of pericenter
alone. The code (all three copies, e.g. `generate_pds4_files.py:506-529`) computes the true
anomaly first: `f = (λ_inertial − (ϖ₀ + ϖ̇·ET/86400)) mod 360`, then `r = a(1−e²)/(1+e·cos f)`.
As printed, the formula yields a radius independent of longitude — contradicted by the guide's own
sample table of per-longitude `core_radius` values (04:169-173). Sections 03:315 and 03:342 refer
back to "the formula above", propagating the error. Since the guide is archived in the bundle,
this is a permanent, citable error in a published equation. Fix: insert `λ −` inside the cosine.

---

## 2. Major — archived data and label correctness (generator + templates)

### 2.1 `et_to_tour` compares YMD strings against DOY cutoffs → wrong `mission_phase_name` — **FIXED** (`fe4f8f5`)
`pds4_bundle_gen/generate_pds4_files.py:1116-1129`. `et_to_datetime` returns YMD format
(`2008-08-15T12:00:00Z`) but the cutoffs are DOY strings (`'2008-183T00:00:00.000'`). Month
digits (01–12) always compare lexicographically below DOY digits, so **every date in a boundary
year lands on the "before" side**. Verified empirically: `'2008-08-15T12:00:00Z' <
'2008-183T00:00:00.000'` is `True`. Consequence in archived labels
(`<cassini:mission_phase_name>$TOUR$`): Dec 24–31 2004 → `TOUR PRE-HUYGENS` instead of `TOUR`;
Jul 1–Dec 31 2008 → `TOUR` instead of `EQUINOX MISSION`; Sep 30–Dec 31 2010 → `EQUINOX MISSION`
instead of `SOLSTICE MISSION`. The FMOVIE set spans 2004–2017, so real products are affected, and
`validate` cannot catch it. (Ironically introduced by the prior review's "compare full
timestamps" fix, which assumed `et_to_datetime` returned DOY format.) Fix: convert the cutoffs
with `utc2et`/compare ETs, or format the datetime as year-DOY before comparing.

### 2.2 Mosaic per-column ET is float32 → archived times and inertial longitudes quantized — **FIXED upstream** (rms-csmithing branch `fix_mosaic_time_float64`, commit `f156fdb`; existing mosaics retain the quantized values until rebuilt)
Root cause upstream: `nav/ring_mosaic.py:1129` allocates the mosaic `'time'` array as
`np.float32` (verified in source; float64 midtimes are stored into it at `:1215`). At Cassini-era
ET (2.6–5.5×10⁸ s) the float32 spacing is 16–64 s, so every per-column time in every mosaic
metadata file is quantized by up to ±8–32 s. Downstream in the archive:
`generate_pds4_files.py:812` computes `rings:inertial_ring_longitude` from these ETs
(581.964°/day ≈ 0.0067°/s → up to ~±0.2°, i.e. ~10 longitude bins of 0.02°), `:2465-2477` writes
the ET into `metadata_params.tab` with false millisecond precision (`{et:13.3f}`), and the same
error enters `core_radius` and the satellite corotating longitudes. Reprojected-image products
are unaffected (their `time` is a float64 scalar). Fix in `rings_mosaic_init` (float64), then
mosaics must be rebuilt for the archive to improve; alternatively document the precision.

### 2.3 Labels declare `vertical_display_direction` "Top to Bottom" but row 0 = inner radius — **FIXED** (`7f47619`)
`templates/data_mosaic.lblx:131`, `templates/data_reproj_img.lblx:185` (verified). The archived
arrays have Line 0 at −1000 km (inner edge); the browse PNGs are row-flipped before writing
(`generate_pds4_files.py:2831`) so +1000 km is at top, and all four example scripts render with
`origin='lower'`. A label-compliant viewer therefore displays the ring **radially mirrored**
relative to the bundle's own browse products and example scripts. Fix: `Bottom to Top` in both
templates (the data, scripts, and browse all agree with each other; the label is the outlier).

### 2.4 `document_standard_id` "Python" is not a valid PDS4 enumeration — **FIXED** (`7f47619`; two scripts contain a non-ASCII `≈` and use `UTF-8 Text`)
`templates/f-ring-mosaics-user-guide.lblx:253,259,265,271,277` (verified ×5). IM 1.24's
permissible values do not include "Python" (`7-Bit ASCII Text`, `UTF-8 Text`, `PDF`, `PDF/A`, …).
`validate` will reject the user-guide product label for all five example-script
`Document_File`s. Fix: `7-Bit ASCII Text` (or `UTF-8 Text`).

### 2.5 `collection_document.lblx` hardcodes `Inventory` `<records>5</records>` — CSV has 10 rows — **FIXED** (`7f47619`)
`templates/collection_document.lblx:147` vs `templates/collection_document.csv` (verified: 10
records — the CSV later grew the 8 context entries and the Cassini user guide). The `File`-level
`<records>` on line 141 already uses `$FILE_RECORDS(...)$`; the Inventory count was left behind.
Same bug class as the miscellaneous-collection `<records>5</records>` fixed last round, in the
sibling label. Guaranteed `validate` error. Fix: use the macro.

### 2.6 Collection inventories can disagree with the products actually generated — **FIXED** (`fe4f8f5`)
`generate_pds4_files.py:3684-3693` (verified against the failure paths at 3039-3084 and
3115-3122). Three scenarios:
(a) `handle_one_obsid` bails with a plain `return` (missing mosaic/bsm/bkgnd file, `obsid_list`
mismatch) — the main loop still writes `P,<lidvid>` rows for that obsid's mosaic and browse
products: inventory members with no product → `validate` referential-integrity failure.
(b) A generic exception is logged (3680-3682) without `continue` — same fall-through.
(c) In the reproj loop, the collection row is written **before** `generate_reproj` runs; if it
raises `ObsIdFailedException`, that reproj row stays (product never written) *and* the outer
`continue` then skips 3684-3693, so the obsid's successfully generated **mosaic** products are
missing from their inventories. Fix: write inventory rows only after the corresponding product
generation succeeds (track success per product class).

### 2.7 Index-only runs silently produce empty global indexes — **FIXED** (`fe4f8f5`)
Verified: `handle_one_obsid`'s outer guards (3058-3060, 3095-3103) include
`GENERATE_*_GLOBAL_INDEX`, but the inner guards in `generate_mosaic`/`generate_reproj`
(2983-2985, 2994-2996, 3014-3017) around `generate_image` — the only place index rows are
written — list only the metadata/images/labels flags. Running `--generate-mosaic-global-index`
(or the reproj equivalent) alone reads all the metadata, then writes header-only `.tab` files,
and `generate_global_index_xml` happily writes labels declaring 0 records. Fix: add the index
flags to the inner guards.

### 2.8 Reproj labels reference `SUPPL_PATH` that is only set when suppl files are generated — **FIXED** (`fe4f8f5`)
`generate_pds4_files.py:2532-2537` (verified) vs `templates/data_reproj_img.lblx:340-361`. The
template unconditionally uses `$FILE_ZULU(SUPPL_PATH)$`, `$FILE_BYTES$`, `$FILE_MD5$`, and
`$SUPPL_HEADER_LENGTH$`, but the dict keys are set only under `GENERATE_REPROJ_SUPPL_FILES` —
which (lines 294-297) is *not* implied by `--generate-all-labels` or `--generate-reproj-labels`.
Those invocations produce substitution errors/broken headers in every reproj label even when the
`_suppl.txt` files already exist on disk. Fix: always compute `SUPPL_PATH`, and read the existing
file's header length when not regenerating it (as is already done for `IMG_PATH`).

---

## 3. Major — programs

### 3.1 Viewer filter computes moon separation from the wrong orbital extreme — **FIXED** (`ba555cb`)
`pds4_mosaic_viewer/catalog.py:127-130` (verified). `prom_sep = |mean_core −
minimum_radius_prometheus|` and `pand_sep = |mean_core − maximum_radius_pandora|`. Prometheus
orbits *inside* the core, so its minimum radius is its **farthest** point from the core; Pandora
orbits *outside*, so its maximum radius is likewise farthest. Both filter values are therefore the
maximum separation over the observation, not the closest approach that the filter UI implies and
that the repo convention uses everywhere else (`moons.prometheus_close_approach`,
`create_moon_dist.py`). A user filtering "Prometheus to core ≤ 500 km" gets essentially nothing
while genuine close approaches are excluded. Fix: use `maximum_radius_prometheus` and
`minimum_radius_pandora` (both already in the index).

### 3.2 Background staleness check ignores the products downstream actually consumes — **FIXED** (`52cb449`)
`mosaics/ring_ui_bkgnd.py:186-188` (verified). `all_bkgnd_files_exist` checks only
`-BKGND-MODEL.npz` and `-BKGND-METADATA.dat`, but `save_bkgnd_results` also writes
`-BKGND-SUB-MOSAIC.npz`/`-BKGND-SUB-METADATA.dat` — the files photometry and the PDS4 generator
read. A crash between the two writes (or a wiped `bkgnd_sub_mosaic_*` directory) leaves the model
current; every later batch run early-returns "output files already exist and are current" and the
missing bkgnd-sub product is never regenerated (and the toplevel GUI shows `B`/up-to-date). Fix:
include the bkgnd-sub pair (and ideally the PNGs) in the existence/mtime check.

### 3.3 "Display Background" crashes for any obsid without a computed background — **FIXED** (`52cb449`)
`mosaics/ring_ui_bkgnd.py:554` (verified). The toplevel button runs `--display-bkgnd --no-bkgnd`;
with no `-BKGND-MODEL.npz` on disk, `display_bkgnd` (837-845) never computes
`corrected_mosaic_img`, and `setup_bkgnd_window` dereferences
`bkgnddata.corrected_mosaic_img.copy()` → `AttributeError: 'NoneType' object has no attribute
'copy'` instead of a window.

### 3.4 Missing offset file → image silently reprojected at zero offset — **FIXED** (`52cb449`; the fallback is retained but now prints a prominent UNNAVIGATED warning)
`mosaics/ring_ui_reproject.py:483-488`. When the offset file doesn't exist the abort is commented
out and `the_offset = (0., 0.)` is substituted with nothing recorded in the repro metadata. An
unnavigated image can then win mosaic columns and enter the archive indistinguishable from
navigated data. If intentional, it needs at minimum a metadata flag (the suppl file's
`Navigation Type` would otherwise claim a navigation that never happened).

### 3.5 `f_ring_util` API breaks against its own callers (3 crashes) — **FIXED** (`115574b`)
All verified against source:
- `f_ring.py:277` — `polar_png_path(arguments, obsid)` lost its `make_dirs` parameter;
  `utilities/create_polar_mosaics.py:133` passes `make_dirs=True` → `TypeError` (and the
  directory is never created even without it).
- `moons.py:100` — `_close_approach` now returns a 4-tuple, but
  `photometry/create_moon_dist.py:90,92,137,139` still unpacks 2 → `ValueError` every run
  (`create_ews.py` was migrated; `create_moon_dist.py` was missed).
- `moons.py:91-99` — with an empty search window the loop body never runs and `min_dist_long` is
  unbound; `photometry/create_eds.py:594` passes `max_et=0`, which is empty for any Cassini-era
  `min_et` → `UnboundLocalError` on every equivalent-depth run through that path. (If `0` was
  meant to mean "default one orbit", the sentinel should be `None`.)

### 3.6 `ring_plot_offsets.py` crashes on obsids lacking a navigation type — **FIXED** (`52cb449`; also saves per-obsid `offsets_<obsid>.png`)
`mosaics/ring_plot_offsets.py:75-82`. The plot loop iterates fixed keys `'STARS'`/`'MODEL'` of
`symbol_by_winner` but `num_by_winner`/`xs_by_winner` only contain winners actually seen →
`KeyError: 'STARS'` for any obsid with no star-navigated image (common for F-ring movies). Other
winner values are silently dropped, and every obsid overwrites the same `figure.png`.

### 3.7 `FILELIST_BRING_MOUNTAINS/defaults.txt` is short two lines → startup crash for that type — **FIXED** (`52cb449`; zooms 10/1 assumed, no existing data files constrain them)
The file has 6 lines; `ring_init` (`mosaics/ring/ring_util.py:226-233`) unconditionally reads 8
(`readline()` at EOF returns `''` → `int('')` `ValueError`). Every `ring_ui_*` script crashes at
startup for `--ring-type BRING_MOUNTAINS` (radial/longitude zoom lines missing).

---

## 4. Major — user guide vs bundle

(4.1 = §1.2 above, the core-radius equation.)

### 4.2 Product counts are stale after the joint-observation split — **FIXED** (`1a85209`; 305 mosaics, image count re-verify after regeneration)
`sections/01-introduction.tex:14` and `03:61`: "20,303 reprojected images … 302 … mosaics". The
split of the two joint observations makes it **305** mosaics (`observation_list.csv` and
`FILELIST_FMOVIE` both have 305 entries — verified identical in both directions), and the reproj
count will change with the restored observations. Update after the regeneration completes, from
the actual bundle.

### 4.3 Both quick-start commands are broken as printed — **FIXED** (`1a85209`)
`sections/05-reading-labels-and-data-product-files.tex:114-116, 121-123` (verified). The label
path is wrapped as `.../data_reproj_img/ \` + newline + `iss_036rf...` — the space before the
backslash makes the continuation a *second* shell argument, and both scripts require exactly one
(`len(sys.argv) != 2` → usage exit). A user copying either command verbatim gets the usage
message. Join the path (no space before the line break) or unwrap.

### 4.4 "Each line refers, in sequence, to an X index in the image array" is false in two cases — **FIXED** (`1a85209`)
`sections/04-bundle-organization-and-directory-structure.tex:176`. For wraparound reproj images
the `.img` is resliced to (min→359.98, 0→max) order while `metadata_params.tab` rows are in plain
ascending-longitude order; and the table contains only *valid* longitudes while the image retains
interior sentinel columns. Row *i* ≠ column *i* in general; readers must key on
`rings:corotating_ring_longitude` (as `mosaic_utils.py` correctly does). Reword.

### 4.5 `core_radius` is described as "(constant for one image)" — **FIXED** (`1a85209`)
`sections/07-metadata-and-global-index-file-fields.tex:33`. It is computed per row from that
row's inertial longitude (`generate_pds4_files.py:2466`), and the guide's own sample table shows
it varying. Drop the annotation.

### 4.6 Broken hyperlink target left over from the docx conversion — **FIXED** (`1a85209`)
`sections/03:245` (verified): `\href{../customXml/item1.xml}{https://pds-rings.seti.org/pds4/...}`
— the link target is a docx-internal relationship path; the archived PDF gets a dead link.

---

## 5. Minor findings

> **Fix status:** every item in this section was fixed in the commits listed in the header
> note, except the four bullets explicitly marked **NOT FIXED** below.

### Generator (`generate_pds4_files.py`)
- **`'N/A'` substituted into `<start_date_time>`/`<stop_date_time>`** (3138-3144, 3215-3221,
  3392-3399 fallbacks) in `bundle.lblx`/collection labels without `xsi:nil` — invalid
  `ASCII_Date_Time_YMD_UTC`, triggered by support/collections-only runs that traversed no
  products (contrast `kernels.lblx:38-39`, which nils correctly). Related: bundle date range is
  accumulated only from obsids processed in the current run, so support files regenerated during
  a subset run silently narrow the archived date range.
- **`reslice_reproj_img` full-coverage wrap case returns the unrotated image** (1016-1018): a
  360°-covered wraparound image should be rotated left by `min_idx` columns to start at the
  label's stated minimum longitude; the unrotated array is returned instead (width matches, so
  the shape check passes). Physically requires a full-ring single image — latent.
- **Hardcoded grid constants**: `'0.00'/'359.98'` mosaic corot min/max (1899-1900), the
  `400/5/1000` background-limit arithmetic (2163-2168), and `401/18000` browse dimensions
  (2779-2788) are only correct for the default geometry; any other `--ring-type` geometry writes
  wrong label values without error (partially TODO-flagged at 2163).
- **`read_reproj` performs no `0 → −999` normalization** (888-896) while the mosaic builder does
  it unconditionally (`ring_ui_mosaic.py:187-192`); a legacy 0-convention repro file would
  archive 0.0 as apparently valid I/F while the mosaic marks the same pixels −999. Latent
  (old-format files are rejected at 869-877; a current file was spot-checked clean).
- **Browse stretch statistics include sentinel pixels** (2797-2811): `valid_cols = sum(img,axis=0)
  != 0` doesn't exclude −999s, so the blackpoint is always clamped to 0 when sentinels are
  present and the 99.8% whitepoint percentile shifts with coverage. Cosmetic (browse only).
- **`--generate-mosaic-metadata`/`--generate-reproj-metadata` help says "tables and labels"** but
  the flags regenerate only the tables (191-193, 233-235 vs 299-302, 342-345) — leaving stale
  `FILE_MD5`/`FILE_ZULU` in existing labels after a tables-only rerun.
- **NOT FIXED (content decision)** — **`INST_CMPRS_PARAM` is prepared (with a deliberate `'N/A'→'999'` transform) but no template
  references it** (1823-1824) — looks like an accidentally dropped label field. Assorted other
  dead keys (`REPROJ_METADATA_TITLE/...`, `MOSAIC_METADATA_TITLE`, `AUTHORS`/`EDITORS`) invite
  drift; author lists are hardcoded per-template in ~8 places. Whether to add the label field or
  delete the dead keys changes archive content, so it was deferred.

### Templates
- `data_mosaic.lblx:158-159`: mosaic corot min/max are hardcoded 0.00/359.98 (full extent), but
  the embedded `rings:description` says min/max "are the limits that contain valid data … if the
  reprojection wraps … minimum will be greater than maximum" — false for every partially covered
  mosaic; the wraparound sentence is meaningless for a fixed range.
- Both `RINGS_DESCRIPTION`s (2041-2042, 2288-2289) say min/max/mean are computed "for phase angle
  and observed_ring_elevation" — the labels actually carry phase, incidence, and emission; no
  `observed_ring_elevation` exists (open TODO at code line 1659), and min/max incidence being the
  mean is documented in the field descriptions but not here.
- Reproj browse description "full (equal in size to the reprojected image…)" (2884-2896) is
  strictly wrong for discontinuous coverage (browse drops gap columns; the `.img` keeps them).
- `collection_spice_kernels.lblx:136`: leftover curator instruction comment (`<!-- MJTM: please
  update … -->`) ships in the archived label.
- **NOT FIXED (content decision)** — `collection_document.csv:3-10` / `collection_miscellaneous.csv:4-12`: context products (and one
  Product_Document) listed as secondary members of Document/Miscellaneous collections — duplicates
  the context collection's role, may draw `validate` warnings; also LID-only vs LIDVID styling is
  inconsistent with `collection_context.csv`. Changing collection membership is an archive-content
  decision, so it was deferred.

### Example scripts (archived in the bundle)
- `find_prometheus_closest_approaches.py:120,169-170`: separation and marker use
  `mean_core_radius` (image-mean) instead of the core radius at Prometheus's longitude — several
  pixels/tens of km of error over a several-degree image; documented in the docstring, but worth
  using the per-longitude `core_radius` that is already in the metadata table.
- `plot_ews_df.py:70` / `plot_ews_ma.py:83`: EW sums include partially covered columns (interior
  −999 pixels shrink the integral), producing spuriously low EW points at coverage edges; the
  pipeline's own `create_ews.py` filters these (`--maximum-bad-pixels-percentage`). A one-line
  filter or a warning comment would stop users treating those points as real.

### f_ring_util
- `f_ring.py:19-72`: the metadata-key documentation block names the *old* keys (`long_mask`,
  `ETs`, `emission_angles`, `resolutions`, `image_numbers`) instead of what writers emit
  (`long_antimask`, `time`, `mean_emission`, `mean_radial_resolution`, `image_number`). Not
  purely cosmetic: `photometry/create_moon_dist.py:132` follows the documented `long_mask` and
  KeyErrors on every current-format file.
- `f_ring.py:309`: `FRING_ORBIT_EPOCH` is unused, and its value (`utc2et('2000-01-01T12:00:00')`)
  is TDB-of-UTC-noon, not the ET=0 zero point the pericenter formulas implicitly use — a trap if
  someone "fixes" the formulas to reference it.
- `moons.py:51-83` vs `generate_pds4_files.py:551-592`: same-named
  `saturn_to_prometheus_corot`/`saturn_to_pandora_corot` with different semantics (core-relative
  delta vs absolute Saturn distance; the moons.py pair also uses opposite subtraction orders so
  both come out positive). Each is used consistently at home, but the shared names invite a
  ~140,000 km porting error.
- `f_ring.py:487-489`: `fit_hg_phase_function`'s documented `nstd` outlier-rejection mode is
  `assert False` followed by dead code that would `AttributeError` anyway.

### Mosaic pipeline
- `ring_model_bkgnd.py:401-402`: the in-loop "too few pixels now" recheck counts the *original*
  mask, which never changes — the recheck is a no-op, so a column progressively deweighted below
  the `below`/`above` minimums is never rejected.
- `ring_model_bkgnd.py:304` vs 353-354/401-402: row `rmax` is background for the fit but counted
  toward neither ring nor outside minimum (off-by-one, one radial row of bias in the reject test).
- `ring_model_bkgnd.py:290-292, 306-307, 318-322`: the non-tuple `ring_rows` paths are broken
  (int path centers the band on the array *edge* and produces float slice bounds → `TypeError`;
  `None` path leaves `rmin/rmax` undefined → `NameError`). Latent — the only caller passes a
  tuple.
- `ring_model_bkgnd.py:420-421`: `model.mask = image.mask` aliases and then mutates the caller's
  mask in place (`|=`) — harmless today, a trap for any future caller reusing the input.
- **NOT A BUG (withdrawn)** — `nav/ring_mosaic.py:1143`: `mean_incidence` is overwritten by *each added image*, so
  the stored value is the last image in the list — even one contributing zero columns — not a
  mean. Readers archive it as `MEAN_INCIDENCE_ANGLE`. Withdrawn per user confirmation: the
  incidence angle is assumed constant over one mosaic (the same simplification the labels
  document), so which image supplies the stored value is immaterial.
- `ring_ui_bkgnd.py:193-210`: `write_bkgnd` runs *before* the ring-limit sliders are re-read,
  `write_bkgnd_sub_mosaic` after — moving a slider and pressing Commit without Recalc makes the
  two metadata files disagree, and the archived limits (`generate_pds4_files.py:2165-2167` reads
  the bkgnd-sub side) describe values never used in the fit.
- `ring_ui_bkgnd.py:127-146`: dead `_update_metadata` assigns recomputed longitudes to the wrong
  target (`bkgnddata.longitudes`, not `metadata['longitudes']`) — a trap if revived.
- `ring_ui_bkgnd.py:343,356`: mouse-handler bound check `>` should be `>=` (`x == shape[1]`
  passes and IndexErrors); `:543` vs `:311`: mask-overlay red channel 1 vs 255 — the initial mask
  display is effectively invisible until the first recalc.
- `ring_ui_reproject.py:88-89`: `--allow-exception` is `store_true` with `default=True` — can
  never be disabled, so batch runs abort on the first exception (opposite of `ring_ui_mosaic`).
- `ring_ui_reproject.py:556-559`: the F-ring-core overlay row uses `outer_delta` where
  `−inner_delta` is meant; the extra flip cancels only for symmetric ranges — asymmetric ranges
  draw the core mirrored. `:576-581`: Py3 `TypeError` from float `/2` slice indices in the
  oversized-overlay path. `:1247`: `image_log_filehander` typo (harmless).
- `ring_ui_mosaic.py:598` / `ring_ui_reproject.py:1183`: "Abs Radius" adds the grid offset to the
  *eccentric* core radius; the grid is relative to the fixed 140220 km reference, so the display
  is wrong by core(λ)−140220, up to ±330 km. GUI-only. `ring_ui_mosaic.py:615-624`:
  `command_show_longitudes` misspells its parameter (works via the module global by accident) and
  its block test drops the last element of every block.
- `ring/ring_util.py:332-333`: `ring_basic_cmd_line` emits `--radius-inner`/`--radius-outer`,
  which only work via argparse prefix matching — adding any new `--radius-inner*` option breaks
  every spawned subprocess.
- `ring_ui_toplevel.py:438,939`: `char_skip=16` doesn't reach the image-name field of the list
  format (churn/lost selection only); `:451-452`: empty-list click → `IndexError`; `:399-408`:
  `--planet` (and `--verbose`) not forwarded to stage subprocesses; `:404-405`: resolutions
  forwarded at `%.3f` — a finer GUI-entered value is silently rounded (paths stay consistent).
- `ring_make_mosaic_list.py:25-29`: re-`ring_init` after the first ring type is a no-op for
  already-populated geometry — latent until the commented-out ring types are re-enabled
  (**FIXED**: geometry args are now reset before each `ring_init`).
  **NOT FIXED**: `mosaics/obs_list.csv` remains a stale snapshot (302 obsids; missing the three
  split additions) — regenerating it requires running the script against the data.
- All F-ring `defaults.txt` files carry an unread 9th line (`1`) — `ring_init` stops at 8;
  visually matches the hardcoded `_1` filename suffix but is not parsed (trap). Informational
  only; deliberately left as-is.

### Viewer
- `mosaic_window.py:68`: "Emission (abs)" color scale is clipped at 90°, but emission spans
  0–180° (southern viewing) — south-side mosaics saturate solid red. Should be `(0.0, 180.0)`.
- `mosaic_window.py:1377-1411`: wraparound reproj images (valid span crossing corot 0°) open
  zoomed to the full ring centered on empty longitudes instead of fit to the data.
- Radial-offset readouts say "offset from **mean core**" (`pds4_mosaic_viewer.py:71-73`,
  `mosaic_window.py:663,870-873`) — the grid is core-following (offset from the *local* elliptical
  core), so the wording misstates absolute radii by up to ±330 km.

### User guide (beyond §4)
- 07:59-63 and 07:153-157: examples pair image `1874525875w` / a B3001 `observation_id` with
  `..._complitb4001_si` paths and LIDs — stale after the joint-observation split (the generator
  now guarantees these agree).
- 04:135: excerpt shows `local_identifier` `reproj_img`; the template emits `reproj_image`.
- 04:95-103: excerpt lists phase → emission → incidence (template order is phase → incidence →
  emission) and shows 6-decimal values where the generator writes `.3f`.
- 04:223: reproj Med browse omits the 400-px minimum *width* (`max(src//10, 400)`) — the typical
  few-degree image is not a ÷10 downsample.
- 04:33: "details of the stretching process are not provided" — the browse labels do specify
  blackpoint/whitepoint/gamma.
- 03:318: reproj per-longitude list claims "Min and Max True Anomaly"; the table has a single
  `true_anomaly` column (min/max are index/label-level).
- 02:51: `_metadata_src_imgs.tab` maps image numbers to the reproj image **LIDVID**, not "name"
  (04:273 states it correctly).
- 04:94,133,144,149,154: XML comments printed as `<-- … -->` instead of `<!-- … -->` (docx
  artifact, acknowledged in README but still wrong in verbatim excerpts).


### Addendum — found and fixed during the fix pass

- `photometry/create_moon_dist.py` applied `np.degrees()` to the close-approach longitude
  before writing it to the output CSV, but `moons.saturn_to_prometheus`/`saturn_to_pandora`
  already return the longitude in degrees — every longitude in the CSV was inflated by a
  factor of 180/π. The dead `if False:` debug block above it was separately broken (caught
  the whole return tuple in `dist` and referenced an undefined `long`). Both **FIXED**
  (`115574b`).

---

## 6. Status of 2026-07-21 findings

| Prior | Status |
|---|---|
| 1.1 miscellaneous inventory LIDs | **FIXED** — CSV LIDs match the generator constants (re-verified by template review). |
| 1.2 miscellaneous `<records>` hardcoded | **FIXED** — macro in place. (But the same bug was found un-fixed in `collection_document.lblx` → §2.5.) |
| 1.3 duplicate `LATEST_STOP_DATE_TIME` clobber (~3391) | **FIXED** — independently confirmed by both generator reviews; all three assignment sites are now None-guarded. Residual: the `'N/A'` fallback is schema-invalid (§5, generator). |
| 1.3a dropped observations decision | **RESOLVED** — joint observations split; `FILELIST_FMOVIE` (305) and `observation_list.csv` (305) verified identical in both directions. |
| 1.4 spice-kernels year/date | **FIXED**. |
| 1.5 kernels.lblx reference type / name | **FIXED** (template review found kernels.lblx sound; one leftover curator comment remains in `collection_spice_kernels.lblx`). |
| 2.4/2.5 collection/CSV gating | **CONFIRMED and expanded** → §2.6, §2.7 (now with concrete failure paths). |
| 2.13 reproj old-format fallback | Old-format files are now rejected (869-877); the related 0-sentinel normalization gap is noted in §5. |
| 6.3 guide counts | **STILL STALE and drifted further** → §4.2 (now 305 mosaics). |
| 6.4 Med browse 1800×401 / 6.5 "padded" / 6.12 `mosaic_utils.py` / 6.15 ascending node | **ALL FIXED** — re-verified sound by the guide review. |

---

## 7. Checked and found sound (condensed)

- **Orbit model**: the three copies of the F-ring model (`f_ring.py:308-338`,
  `mosaics/ring/ring_util.py:736-765`, `generate_pds4_files.py:483-529`) are byte-identical in
  constants (n = 581.964°/day, a = 140221.3, e = 0.00235, ϖ₀ = 24.2, ϖ̇ = 2.70025, epoch
  2007-01-01 TDB) and algebraically identical in the conversions; Ω₀/Ω̇ in the generator match the
  guide. No drift anywhere.
- **Grid & frames**: 18000×401 grid construction, bin-left-edge convention, negative inner delta,
  corot↔inertial round trips, `%360` wraparound, and the `(Δ+180)%360−180` idiom are consistent
  across writer, readers, generator, viewer, and example scripts.
- **Sentinels/masks**: −999 convention (incl. legacy 0→−999 conversion at reprojection and mosaic
  assembly), `long_antimask` = True-is-valid, and mask propagation through background subtraction
  are honored identically by `create_ews.py`, the generator, the viewer (via
  `Special_Constants`), and `mosaic_utils.py`.
- **Best-pixel mosaic selection** (`rings_mosaic_add`): most-valid-pixels wins, ties by better
  radial resolution, dataless columns can never enter `long_antimask`, and image/provenance
  arrays update atomically under one column mask — `image_index` bookkeeping feeding the archive
  source-image tables is correct.
- **File formats**: every filename template round-trips byte-identically between
  `mosaics/ring/ring_util.py` writers and `f_ring_util/f_ring.py` readers; bkgnd 9-tuple pickle
  and NPZ key contracts match all readers; msgpack metadata keys match all consumers.
- **PDS4 table arithmetic**: every field_location/field_length/record length in the mosaic
  (17-field), reproj (16-field), src_imgs, suppl, and all three global-index layouts
  (42/57/60 fields; records 625/762/775) was summed against the generator's format strings and
  matches exactly; LID/LIDVID lengths fit their fixed-width fields at the longest obsid; all
  `$VARIABLE$`s are provided on the populating code path (except §2.8); reference_types are valid;
  schema versions are coherent (IM 1.24 / 1O00-series) and match `collection_xml_schema.csv`.
- **Mosaic-vs-reproj template pairs**: diffed; no crossed "mosaic"/"reproj" text or sizes.
- **Viewer**: axis order, row direction, sentinel re-masking, stretch, cursor/EW/profile unit
  handling, pds4_tools field-name sanitization, and the wraparound reproj re-expansion all match
  the generator exactly (§3.1's filter columns are the one exception).
- **Example scripts**: EW math (μ = |cos e|, degrees→radians), index/label parsing, longitude→
  column recovery, and wrap handling verified against generator and templates; docstring "run
  from document/user_guide" placement is correct.
- **Guide**: directory trees, all §7 field tables, browse sizes, observation-class lists, LIDs,
  DOIs, orbit constants, epoch ET value (recomputed exactly), and the processing description all
  match the code except as listed in §4/§5.
- **Lists**: all F-ring `defaults.txt` geometry matches the documented defaults; FMOVIE lists ↔
  `observation_list.csv` are in perfect sync.

---

## 8. Suggested priority order

1. **Before the in-progress regeneration is used for anything:** §2.1 (`et_to_tour` — wrong
   mission phases are being written *right now*), §2.4 and §2.5 (guaranteed `validate`
   failures), §2.3 (display direction), §2.8 if any label-only reruns are planned.
   *All done — but the regeneration started before these fixes must be redone.*
2. **Same pass, generator hygiene:** §2.6/§2.7 (inventory/index gating), `'N/A'` date fallback.
   *Done.*
3. **Code landmines (no current data impact, cheap to fix):** §1.1 `fit_quadratic` swap, §3.5
   `f_ring_util` API breaks, §3.2/§3.3 bkgnd staleness/crash, §3.1 viewer filter columns.
   *Done.*
4. **Guide edits for the next PDF build:** §1.2 equation (most important — published math),
   §4.2–§4.6, plus the §5 guide minors. *Done; the guide rebuilds cleanly.*
5. **Decision needed:** §2.2 (float32 mosaic times — fixing properly means a nav-code change and
   mosaic rebuild; alternatively document the ±16–32 s precision), §3.4 (zero-offset fallback —
   intentional?). *§2.2 fixed upstream in rms-csmithing (`f156fdb`); archived per-column times
   and inertial longitudes improve only after mosaics are rebuilt. §3.4: fallback kept, now
   warns loudly.*
6. **Second pass after regeneration:** spot-check the new bundle (deferred from this review).
