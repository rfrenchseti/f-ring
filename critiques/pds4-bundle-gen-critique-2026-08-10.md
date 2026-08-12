# PDS4 bundle & repository state critique — 2026-08-10

**Scope.** Complete analysis of the repository state with emphasis on PDS4 bundle
generation: the generated bundle at `pds4_bundle_gen/bundle` (→
`/data/fring-bundles/pds4`, built today 2026-08-10 13:52–16:06), the generator
(`pds4_bundle_gen/generate_pds4_files.py` + `templates/`), the user guide
(`user_guide/sections/*.tex` and the shipped PDF), the generation logs
(`/data/fring-bundles/logs/`), and the on-disk pipeline data feeding the bundle.

**Method.** Seven parallel review passes plus orchestrator verification:
- Every inventory row of all 11 collections reconciled against disk (42,107
  `.lblx`, 21,045 `.img`, 84,180 `.png`, 21,658 `.tab`, 20,436 `.txt`).
- All 42,107 labels parsed as XML; 77 labels (all collection/index/support labels
  plus 60 sampled products) validated against the official 1O00-series XSDs
  (schematron not checked — `validate` still required).
- Byte-level verification of labels vs files (file_size, md5, record counts,
  field offsets) for all data labels globally and 60 products in depth,
  including numeric reads of the binary arrays.
- All PNG dimensions parsed from IHDR and checked against labels and code.
- Live PDS registry and RMS-holdings queries for every external LID/LIDVID.
- Every guide section read line-by-line and every concrete guide claim
  (listings, field tables, counts, verbatim excerpts, example scripts)
  re-executed or re-derived against the actual bundle.
- Key claims independently re-verified by the orchestrator (float32 ET
  quantization, background mask handling, phantom inventory row).

**Supersedes** the 2026-07-19 and 2026-07-21 critiques (moved to
`critiques/archive/`) and complements `critiques/code-review-2026-08-04.md`,
which lived on the `code_review_fixes` branch (merged to main 2026-08-11 as
`9fc7045`, *after* this bundle was generated).

Accepted non-issues (per prior user feedback) are not re-flagged: hardcoded
paths; `metadata['time']` = image midtime; imshow half-pixel extent; signed
core−Prometheus sort; contributor sequence 1,2,2,2; incidence angle treated as
constant per mosaic (min/max = mean).

---

## 1. State of the repository — the context for everything below

The single most important fact: **today's bundle was built from pre-fix code
against pre-fix data.**

1. **`code_review_fixes` was not merged when this bundle was built.**
   *(Update 2026-08-11: merged to main as `9fc7045`, together with the
   review-feedback revisions in `c1168cb`.)* The branch carried every fix from
   the 2026-08-04 review — the PDS4 label-validation fixes
   (`vertical_display_direction`, `document_standard_id`, mission-phase
   boundaries, inventory-integrity gating, template/example fixes) and the
   guide corrections (1a85209). **None of it is in this bundle**, which was
   generated from `9612b2e`; a regeneration is required for any of it to
   appear in the archive.
2. **The rms-csmithing float64 time fix was not merged and the data was not
   rebuilt when this bundle was built.** Every mosaic/bkgnd file under
   `/data/cb-results/fring/ring_mosaic/` was dated **Jul 23**, before the fix,
   so the bundle inherits float32-quantized times (§3.2).
   *(Update 2026-08-11: `fix_mosaic_time_float64` merged as rms-csmithing
   `365621c`, and all 305 mosaics and backgrounds rebuilt. Verified in the new
   data: `metadata['time']` is now `float64`, values are no longer
   float32-representable, and consecutive per-image times differ by
   533.746/538.247 s — the true cadence — instead of the quantized 528/544 s
   steps seen in the shipped bundle. The archive still carries the old values
   until it is regenerated.)*
3. Working tree: only the user's uncommitted `pds4_bundle_gen/TODO.txt` edit
   (which now also contains a pasted copy of today's crash traceback).
4. Generation logs for today's run: 1 uncaught exception (§3.1) and 73
   Prometheus/Pandora visibility warnings (§4.4); nothing else.

**Consequence:** even with zero new findings, this bundle could not be final —
it must be regenerated after the branch merges and the data is rebuilt. The
findings below are therefore split into (a) issues the existing branch already
fixes, (b) new issues needing new code/text changes, and (c) decisions.

---

## 2. Release blockers — summary

In recommended fix order (details in the cited sections):

| # | Blocker | Where fixed |
|---|---------|-------------|
| B1 | ~~`code_review_fixes` unmerged~~ → **merged 2026-08-11 (`9fc7045`)**; all known label-validation errors are still present in *this build*, which predates the merge, so a regeneration is required | done — regenerate (§6) |
| B2 | float32-quantized `rings:observed_event_tdb` in all mosaic params tables (±8–32 s → up to ~0.2° derived-longitude error) | ~~merge rms-csmithing fix~~, ~~rebuild mosaics + bkgnd~~ — **both done 2026-08-11 and verified**; regenerate the bundle to pick them up (§3.2) |
| ~~B3~~ | ~~bkgnd-sub mosaics archive masked pixels as valid I/F~~ — **withdrawn, not a bug** (§3.3): the mask marks gradient-fit exclusions, which are real data | none |
| B4 | `ISS_287RI_PROPRETRG001_PRIME` incomplete: 6 reproj products missing, 1 phantom inventory row ×2 collections, 6 dangling src_imgs LIDVIDs ×2 tables | ~~duplicate-keyword tolerance~~ **fixed 2026-08-11 and verified (all 19 products, clean inventory)**; regenerate to clear it from the archive (§3.1) |
| ~~B5~~ | ~~`__pycache__` with 5 `.pyc` files inside `document/user_guide/`~~ — **resolved**; it was created by this review running the example scripts in place, not by the generator, and the user has deleted it (§4.1) | none |
| B6 | Dangling external LIDVID `iss-data-user-guide::1.0` (only `::1.1` exists) | one-character template/CSV fix (§4.2) |
| B7 | Guide: missing core concepts (array axis direction, inertial-longitude definition, bkgnd-limit semantics, IMGID convention) | guide edits (§4.7–4.10; §4.6 withdrawn) |
| B8 | Guide: every §4 verbatim numeric excerpt and the product counts disagree with the real bundle | re-capture from the **final** bundle, rebuild PDF (§4.11) |
| B9 | Moon `Target_Identification` wrong for the 12+8 "visually confirmed but geometrically rejected" mosaics; "not visually confirmed" disclaimer emitted even when confirmed | new generator fix / policy decision (§4.4) |
| B10 | Decisions required: `data_calibrated` forward references (§4.3), source-product VIDs `::1.0` vs PDS3 versions 2–9 (§4.5), xml_schema LIDVIDs (§4.12), open TODO items (§7) | user |

---

## 3. Critical findings

### 3.1 ISS_287RI_PROPRETRG001_PRIME: generation crashed mid-obsid; the bundle ships an incomplete, self-inconsistent product set — NEW root cause

Today's run aborted this obsid's reproj stage with an uncaught
`KeyError: 'SPACECRAFT_CLOCK_START_COUNT'` at `generate_pds4_files.py:1748`.

**Root cause (new bug, distinct from the known §2.6 gating issue).** The source
PDS3 label
`/data/pdsdata/holdings/calibrated/COISS_2xxx/COISS_2115/data/1880794265_1880940914/N1880796883_3_CALIB.LBL`
— the only version-`_3` image in the obsid — is malformed at the archive level:
a ~14-keyword block (`SEQUENCE_TITLE` … `TELEMETRY_FORMAT_ID`, including all
`SPACECRAFT_CLOCK_*`, `START_TIME`, `STOP_TIME`, `TARGET_*`) appears **twice**
at top level with identical values. `pdsparser` renames duplicated keys to
`KEY_1`/`KEY_2`, so no unsuffixed `SPACECRAFT_CLOCK_START_COUNT` exists.
`xml_add_pds3_label_info` (lines 1740–1747) catches only `FileNotFoundError`
and `pyparsing.ParseException`, so the `KeyError` escapes to the top-level bare
`except`, which logs and skips the rest of the obsid.

- **Scope:** a scripted scan of all 21,046 unique source labels across all 305
  mosaics found exactly this one duplicated label. No other obsid affected.
- **Regression:** the previous build
  (`/data/fring-bundles/cassini_iss_fring_mosaics_rsfrench2025.old2`) contains
  this product, so a pdsparser/environment change altered duplicate-key
  behavior between builds.

**Shipped damage (all verified in the bundle):**
- 6 of 19 reproj products missing (`1880796883n`, `1880796976n`, `1880797069n`,
  `1880797162n`, `1880797255n`, `1880797348n`) from both `data_reproj_img/` and
  `browse_reproj_img/`.
- One **phantom Primary inventory row** for `1880796883n…::1.0` in *both*
  `collection_data_reproj_img.csv` (row 20,436) and
  `collection_browse_reproj_img.csv` — inventory rows are written before
  `generate_reproj` runs. Guaranteed `validate` referential-integrity failure.
  The other five images have no rows anywhere (silent data loss).
- Both 287RI mosaic products' `*_metadata_src_imgs.tab` list all 19 LIDVIDs —
  indexes 13–18 dangle — and the mosaics *use* those images' data (image_index
  13 supplies 32 longitude columns of the params table), so the archived
  mosaics permanently reference source products that were never archived.
- `global_mosaic_index.tab`/`global_mosaic_bkg_sub_index.tab` carry
  `num_images=19`, `max_image_name=1880797348n` for this obsid while only 13
  reproj products exist.

**The branch fix is NOT sufficient.** `fe4f8f5`'s per-image guard catches only
`ObsIdFailedException`; this `KeyError` would still escape to the outer generic
`except → continue`, which on the branch would then also skip the obsid's
*mosaic* inventory rows even though the mosaic products were written — the
inverse validate failure (orphan products). Required fix:

1. ~~Duplicate-tolerant PDS3 lookup~~ — **FIXED 2026-08-11** by reading the
   label with `Pds3Label(..., first_suffix=False)`, which keeps the first
   occurrence of a duplicated keyword under its plain name. This covers the
   whole duplicated block at once, which matters because `START_TIME`,
   `STOP_TIME`, `SEQUENCE_TITLE`, `TARGET_DESC`, `TELEMETRY_FORMAT_ID`, and
   `SOFTWARE_VERSION_ID` are read a few lines later and were all suffixed in
   this label. Verified: all eight affected keys now resolve, the `_1`/`_2`
   values agree, normal labels are unaffected, and regenerating
   `ISS_287RI_PROPRETRG001_PRIME` produces all **19** reprojected products
   (was 13) with 19 inventory rows, 19 products on disk, and zero dangling or
   orphan entries.
2. ~~Backstop: convert unexpected exceptions in `xml_add_pds3_label_info` into
   `ObsIdFailedException`~~ — **FIXED 2026-08-11**; it now logs an error naming
   the label file first.
3. ~~Make the per-image guard survive unexpected exceptions without aborting
   the obsid's inventory bookkeeping~~ — **FIXED 2026-08-11**. A
   `sys.excepthook` was also added so that a failure anywhere outside the
   per-OBSID loop is logged as an error instead of dying silently.
   Fault-injection verified: one forced exception yields one logged error, 18
   of 19 products, and consistent inventories with no dangling or orphan
   rows.
4. Regenerate the obsid (products, both collection CSVs, all three global
   indexes) — covered by the full regeneration.

### 3.2 float32-quantized times contaminate every mosaic params table — data rebuild required (verified numerically)

In the shipped bundle, every `rings:observed_event_tdb` value in the sampled
mosaic params tables is *exactly* float32-representable: at ET ≈ 2.1×10⁸ the
representable spacing is 16 s, and consecutive unique ETs in
`iss_029rf_fmovie001_vims_mosaic_metadata_params.tab` differ by 528/544 s
instead of the true smooth cadence. At 2017-era ETs (≈5.5×10⁸ s) the spacing is
64 s. Consequences: per-longitude times are wrong by up to ±8–32 s, and the
generator computes `rings:inertial_ring_longitude`, `core_radius`,
`longitude_ascending_node`/pericenter, and `true_anomaly` **from these
quantized ETs**, so those columns inherit errors up to ~0.2° / ~1 km.
Reprojected-image tables are unaffected (per-image scalar float64 midtime;
verified <0.01° round-trip).

The fix (`rms-csmithing` `fix_mosaic_time_float64`, f156fdb) existed but the
mosaic (`ring_ui_mosaic.py`) and background (`ring_ui_bkgnd.py`) stages had
not been rerun since (all data files Jul 23). **Reprojection does not need to
be rerun.** Required order: merge the fix → rebuild mosaics → rebuild
backgrounds → regenerate the bundle.

**RESOLVED in the data 2026-08-11** (fix merged as rms-csmithing `365621c`;
all 305 mosaics and backgrounds rebuilt). Verified in the new metadata:
`time` is `float64`, the values are no longer float32-representable, and for
`ISS_029RF_FMOVIE001_VIMS` consecutive per-image times now differ by
533.746/538.247 s versus the 528/544 s quantized steps in the shipped bundle —
a shift of several seconds per column, as predicted. The archive itself still
carries the old quantized values until the bundle is regenerated.

### 3.3 NOT A BUG (withdrawn) — background-subtracted mosaics and the npz mask

The original finding claimed that `generate_pds4_files.py` wrongly discards the
background-subtracted mosaic's mask (`metadata['img'].mask = False`), archiving
"bad" pixels as valid I/F. **That was a misreading of what the mask means.**

Per user confirmation, the mask marks only the pixels that were excluded when
*fitting the background gradient* — stars, moons, and similar contaminants.
Those pixels are real data and belong in the archive; discarding the mask is
the correct behavior, and sentinelling them would destroy valid science data.

The requirement that does matter — **pixels missing from the original mosaic
must be sentinels, not merely masked** — was checked directly and **holds**:
across all 305 mosaic/background-subtracted pairs, every pixel that is −999 in
the original mosaic is also −999 in the background-subtracted data, with zero
exceptions. `mosaics/ring_ui_bkgnd.py:150` guarantees this
(`corrected[mosaic_img == -999] = -999`), and line 152 additionally sentinels
whole columns whose background model is entirely masked. Mosaics represent
missing data as −999 (254 of 305 files contain it; the other 51 have complete
coverage), so nothing is lost at the reprojection-limited mosaic edges.

TODO line 64 ("verify that the sentinel values are present in the background
mosaics from the original mosaic's bad pixels") is therefore **satisfied**.

---

## 4. Major findings

### 4.1 RESOLVED, self-inflicted — `__pycache__` inside the document collection
`document/user_guide/__pycache__/` held five `.cpython-312.pyc` files. **This
was an artifact of this review, not of the generator**: the mtime (20:19) is
hours after generation finished (16:06), and it matches the point at which
the review agents ran the shipped example scripts in place inside the bundle
to check that they work. The generator never creates these files.

**Deleted by the user 2026-08-11.** No code change is needed. The only lasting
lesson is procedural: run the shipped example scripts from a copy, never from
inside the bundle tree, and check for stray files before delivery.

### 4.2 Dangling external LIDVID: `iss-data-user-guide::1.0` — NEW
`document/collection_document.csv:2` and
`miscellaneous/collection_miscellaneous.csv:4` (sources:
`templates/collection_document.csv`, `templates/collection_miscellaneous.csv`)
reference `urn:nasa:pds:cassini_iss_saturn:document:iss-data-user-guide::1.0`.
The PDS registry and the live RMS inventory have only **`::1.1`**; `::1.0`
does not exist → referential-integrity error at ingestion. Change to `::1.1`
(or drop the VID). `bundle.lblx` is safe (LID-only reference).

### 4.3 All 20,435 reproj labels reference `cassini_iss_saturn:data_calibrated` — DECISION NEEDED
Every reproj label's `Source_Product_Internal` points at
`urn:nasa:pds:cassini_iss_saturn:data_calibrated:<img>_calib::1.0`, but the
archived `cassini_iss_saturn::1.1` bundle contains no `data_calibrated`
collection (only browse_raw/context/data_raw/document/xml_schema; registry
lookups return not-found). If a calibrated-ISS delivery is coordinated with
RMS this is a deliberate forward reference; otherwise every reproj label ships
a dangling source-product LIDVID. Confirm with RMS before final.

### 4.4 Moon `Target_Identification` policy: geometric test overrides the human "visually confirmed" flags — NEW
`generate_pds4_files.py:1929–1941` warns on disagreement but the geometric
test always wins. Today's 73 warnings decompose as Prometheus 46 geo-True/vis-
False + 12 geo-False/vis-True; Pandora 7 + 8.
- The **geo-False/vis-True** cases yield wrong labels: a moon the author
  visually confirmed is omitted from `Target_Identification`. Verified for 4
  obsids: the moon-to-modeled-core separation is 1010.5–1027.6 km — just past
  the hard ±1000 km window (`_image_has_satellite`, :934–938), which ignores
  the moon's own radius (~40–70 km) and the ~±50 km scatter of the real core
  about the Albers model. The visual flag is right; the window is too strict.
- The "presence has not been visually confirmed" sentence (:2014–2025,
  :2298–2309) is emitted **unconditionally** — including for the common case
  where `observation_list.csv` says `Y`. The archived text then contradicts
  the author's own records.
- The same observation can get different target lists in its two mosaic
  variants (verified: `iss_105ri_tmapn45lp001_cirs_4` mosaic lists Pandora,
  its bkg_sub twin doesn't) because background subtraction drops columns and
  changes the antimask.
- Latent: the edge check (:922) uses compressed-array indices with no
  wraparound — a moon near corot 0° in a full-360° mosaic is wrongly rejected.

Suggested: widen the window by moon radius + model tolerance (or let the
visual flag override for mosaics), and condition the disclaimer sentence on
the flag. (Commit 83b46eb's inertial-longitude core radius was verified
correct; residual error ~0.05 km.)

### 4.5 `Source_Product_Internal` hardcodes `::1.0` while 165 source images are PDS3 versions 2–9 — VERIFY EXTERNALLY
`image_name_to_calib_lidvid` (:1026–1033) always emits `::1.0`. The mosaics use
165 source images with PDS3 version suffix ≥ 2 (v2:37 … v9:2). If the migrated
calibrated bundle assigns VIDs per PDS3 version (RMS convention), those 165
reproj labels reference superseded versions. Check against the actual
`cassini_iss_saturn` calibrated delivery (not resolvable from this machine);
combine with the §4.3 decision.

### 4.6 NOT A BUG (withdrawn) — guide's "emission < 90° = lit side" statement
The original finding assumed a north-based emission convention. Per user
confirmation, the standard rings convention is that incidence and emission are
referenced to the lit-side ring-plane normal, so emission < 90° means lit-side
viewing by definition. Verified in the bundle: `rings:mean_incidence_angle` is
63.3°–89.8° (never > 90°) across all 305 mosaics spanning 2004–2017, and mean
emission straddles 90° in both the pre- and post-equinox eras — exactly as the
lit-side convention predicts. The guide statements at 07:93 and 07:187 are
correct as written. (Optional polish only: one sentence in §3 or §7 stating
the lit-side-normal convention for readers from outside the rings community.)

### 4.7 Guide: the radial (Line) axis direction of the binary arrays is stated nowhere in the archive — NEW
The guide never says row 0 = Δr = −1000 km (inner edge) with radius increasing
along the Line axis (04:163 gives only the range; 03:299 gives the core row).
Meanwhile the shipped labels' `vertical_display_direction` is wrong (known,
unmerged fix), so **no document in the current archive states the true array
orientation**. One sentence at 04:163 fixes the guide half.

### 4.8 Guide: "inertial longitude" is never defined — NEW
The corot↔inertial formulas (03:279–289) are given, but the zero point and
direction of inertial longitude (J2000 reference, measured how, from where)
appear nowhere; 07:35/37 ("relative to J2000") is the only hint. Without it a
user cannot independently place a moon, verify `true_anomaly`, or compare with
other published F-ring longitudes. Define it once in §3.

### 4.9 Guide: `bkgnd_lower_limit`/`bkgnd_upper_limit` semantics contradictory — NEW
03:357 calls them "pixel limits"; 07:269–271 calls them "lower ring delta
radius"; the archived value (generator :2163–2168) is actually a positive,
unsigned distance in km from the core (750 ⇒ region Δr −1000…−750 used
interior). Reading "+750" as a signed delta radius puts the background region
on the wrong side of the ring. State: km, unsigned offset interior (lower) /
exterior (upper), background region between that offset and 1000 km. The same
text should replace the identical ambiguity in `templates/global_index.lblx`
(:897, :908).

### 4.10 Guide: the `IMGID` naming convention is never defined — NEW
Used ~20 times with examples (`1622049830n`, `1874525875w`) but never
explained: digits = Cassini spacecraft clock count, suffix = camera
(`n` NAC / `w` WAC). It is the primary key of every reproj product and the
link back to COISS; a peer reviewer will ask.

### 4.11 Guide: every §4 verbatim numeric excerpt — and the product counts — disagree with the real bundle — NEW
All sampled values in the guide's excerpts come from a pre-fix build and the
*bundle* is the correct side (agent recomputed the orbit model:
e.g. reproj sample corot 320.08 → inertial 61.809 in both model and bundle;
guide prints 101.729. Label excerpt for `1622049830n`: samples 655 vs actual
649, file_size 1050620 vs 1040996, corot range 286.14/308.60 vs 186.60/199.56,
plus resolution-format drift 3 → 5 decimals). Counts: guide says "20,303
reprojected images … 302 mosaics"; the bundle has 20,435 archived (20,441
intended once §3.1 is fixed) and 305 mosaics. **All excerpts and counts must be
re-captured from the final regenerated bundle**, then the PDF rebuilt — do this
last (note the ordering dependency: bundle → excerpts → PDF → document
collection re-copy).

### 4.12 xml_schema collection pins five LIDVIDs that don't resolve in the PDS registry — NEW
`collection_xml_schema.csv` pins `pds-xml_schema::1.24`, `disp::1.15`,
`geom::1.19`, `rings::1.14`, `cassini::1.18`; none of those exact VIDs exist in
the registry (and newer EN registrations use different LID patterns). The VIDs
look guessed. Safer: LID-only secondary entries (as `collection_context.csv`
already does) or confirm exact LIDVIDs with EN/RMS.

---

## 5. Minor findings

### Bundle / support collections
1. SPICE collection `Time_Coordinates` stop is `2017-09-01`, before the
   bundle's own data stop (2017-09-07) and the kernels' actual coverage
   (≈Sep 19). Placeholder dates on the same line as the known leftover MJTM
   comment (`collection_spice_kernels.lblx:136–139`).
2. `mission.cassini-huygens::1.4` cited; registry latest is `::1.5` (valid but
   stale at delivery time) — `collection_document.csv:6`,
   `collection_miscellaneous.csv:8`.
3. Inventory LIDVID-style inconsistency: `collection_context.csv` uses bare
   LIDs for secondary members; document/miscellaneous CSVs use `::1.x` for the
   same products. Both legal; normalize.
4. Cosmetic: trailing-period inconsistency in collection titles;
   "Initial version" with/without period across `Modification_Detail`s; stray
   tab at `bundle.lblx:131`.

### Generator / templates (main)
5. 559/20,435 reproj "full" browse PNGs are narrower than their data product
   (interior gap columns dropped, disclosed in the description text) — but the
   same description's size list says "equal in size to the reprojected image",
   strictly false for these 2.7%. (Quantifies a known wording issue.)
6. `Special_Constants` comment in `data_mosaic.lblx:233–237` /
   `data_reproj_img.lblx:329–333` describes only reproj semantics ("off the
   edge of the FOV / transmission error"); for mosaics the dominant meaning is
   "no image covered this longitude", and for bkg_sub also "background model
   invalid".
7. `spacecraft_clock_count_partition` hardcoded `1` in both data templates
   while the generator parses the real `SPACECRAFT_CLOCK_CNT_PARTITION` (dead
   key). Always 1 for Cassini; still, use the variable.
8. Global bkg-sub index writes `{upper_limit:4d}` (:2708): a value of −1000
   would be 5 chars and shift the fixed-width record. Current data ranges
   150–905 — latent only.
9. `read_mosaic`/`read_reproj` error paths reference `obsid` (:787, :793,
   :870, :876), not a parameter — works only via the top-level loop's global;
   `NameError` from any other caller.
10. Inconsistent-camera check (:1720–1725) logs an error but doesn't raise; a
    mixed-camera mosaic would silently archive under image 0's camera.
11. Docstring at :1088 shows a garbled example LID
    (`…rsfrench2025_mosaic_rsfrench2025…`); emitted LIDs are correct.
12. `pds4_bundle_gen/data_dictionaries/` holds GEOM `19A0` / RINGS `1D00` XSDs
    while labels reference GEOM `19B0` / RINGS `1E00` — offline validation
    against the local copies tests the wrong LDD versions.
13. Comment drift: `generate_pds4_files.py:88–90` claims reproj browse has only
    full+thumb (four sizes are made); `generate_browse` docstring (:2750) files
    reproj output under `browse_mosaic/`.
14. Dead-key drift persists (deferred decision last round): `SOFTWARE_VERSION_ID`,
    `INST_CMPRS_PARAM`, `MISSION_PHASE_NAME`, `PRODUCT_*` parsed but never
    templated. Note: TODO's two "FROM WHERE?" questions have answers —
    `ground_software_version_id` = PDS3 `SOFTWARE_VERSION_ID` (already parsed,
    currently dead), and `valid_maximum_full_well`/`valid_maximum_DN_sat` = the
    two elements of PDS3 `VALID_MAXIMUM`.

### User guide (beyond §4.6–4.11; file:line into `user_guide/sections/`)
15. 03:178 "Like the M3 mosaics, these mosaics are not particularly useful" —
    the not-useful statement was made about the **R** class (03:166), not M3.
16. 03:20 rev described as numeric 000–293, but the bundle contains lettered
    rev `ISS_00ARI_SPKMOVPER001_PRIME`; a `[REV][TI]` parser following the
    guide mis-splits it.
17. UNIQUENAME acronym list (03:22–42) lacks `SPOKEMOV` (42 mosaics — the
    second-most-common stem; the list's `SPK*` entry doesn't match it).
18. R and N classes get no obsid lists while M1–M4 and O do (03:163–172).
19. `nav_quality`/`bkgnd_quality` cross-refs (07:139, 07:257/259) point to §3
    sections that never define the G/F/P grading criteria.
20. Incidence min/max text contradicts itself: 07:91 "not supplied" vs label
    excerpt 04:101–103 showing them; 07:185 gives a different rationale than
    07:91.
21. 03:301 promises §5 wraparound-handling examples; §5 has none. Related:
    §5's "off-the-shelf" coverage has recipes for labels/.tab/.csv but no
    numpy recipe for the binary `.img` (byte order, reshape) — the one format
    users can't open with a text editor.
22. 03:263–277 element epoch unstated (ϖ₀, Ω₀ are J2000/ET=0 values; the
    corotation epoch 2007-01-01 is defined separately — say so).
23. Negative I/F in bkg-sub mosaics never mentioned (~half of all background
    pixels are necessarily < 0 after subtraction).
24. No "cite this bundle as…", no modification-history/errata policy
    (frontmatter has V1.0 + DOI only).
25. 02:56 "miscellaneous bundle" → collection.
26. 05:139 links `github.com/SETI/fring-mosaics-bundle-software` — confirm the
    repo exists/is populated before archiving the PDF (permanent link).
27. Moon ephemeris source (SPICE kernels in the bundle's own
    `spice_kernels` collection) never stated (03:319–322, 07:41–47).
28. §7 gives units for resolutions but not for times ("seconds past J2000
    TDB"), longitudes, or radii — uneven.
29. The suppl.txt verbatim listing (04:180–199) omits the file's actual 3-line
    preamble ("This file contains a C-matrix …" + blank line).
30. The known "row i ↔ column i" caveat fix (04:176) should also cover the
    mosaic section (04:261–273) and state the mapping
    `col = round(longitude / 0.02°)` explicitly.
31. Typos: 03:55 "due the"; 03:87 "each complete or partial final orbit"
    (garbled); 04:6 "reprojected image" → images; 04:40 "top-Level"; 05:62
    missing period; 06:65,67 "AASDivision"; 01:18 stray space in
    `\textit{User Guide }`; 04:65 stray `\textbf{ }`.

---

## 6. Status of the 2026-08-04 review — merged 2026-08-11, bundle predates it

Everything below is **fixed in the code** (`code_review_fixes`, merged to main
2026-08-11 as `9fc7045`) but confirmed **still present in this build**, which
was generated before the merge (not re-reported above): mission_phase_name wrong
across boundary-year second halves (verified live: all 2008-07…12 mosaics say
`TOUR`); `vertical_display_direction` "Top to Bottom" in all data labels;
`document_standard_id` "Python" ×5; `collection_document.lblx` Inventory
records 5 vs 10; inventory-row gating (§2.6 — whose reproj half manifested as
§3.1's phantom row); the guide's core-radius equation missing the longitude
term; stale 302/20,303 counts; broken quick-start commands; `<--` comments;
excerpt field order; `local_identifier` mismatch; `core_radius "(constant)"`;
dead customXml href; B3001/B4001 mixed example pairing; "Min and Max True
Anomaly"; src_imgs "name" → LIDVID; med-browse 400-px note; stretch
description. **None of this reaches the archive until the bundle is
regenerated** — and note §3.1: the merged per-image guard still needs
strengthening, and §4.11: the merged guide numbers must themselves be
re-checked against the final bundle (intended reproj count 20,441,
mosaics 305).

Still unmerged and load-bearing: rms-csmithing `fix_mosaic_time_float64`
(§3.2, open as PR #5). Also merged with the branch: the quadratic-bkgnd-fit
fix (52cb449 — confirmed harmless to current data: all 305 production
backgrounds are degree 1), and the review-feedback revisions in `c1168cb`
(moons `core_to_*_corot` rename, `--allow-exception` default restored to
True, dead zero-to-sentinel conversion removed after verifying no
reprojected file on disk uses 0, background limits keyed to the radial
resolution, browse sizes re-hardcoded behind geometry asserts, example
scripts simplified, SPICE `Time_Coordinates` removed — which also resolves
§5.1).

---

## 7. TODO.txt assessment (archive-readiness of open items)

| Item | Verdict |
|---|---|
| Mia: kernels/spice labels, document collection CSV, List_Author/Contributor | Appears done in current templates/bundle. |
| Check navigation of `iss_199rf_fmovie002_prime` | Required before final (data QA). |
| Stars as targets for occultation ('O'/'R') products | Polish, not validity-blocking — the mosaic label comment already discloses the star. Open completeness decision at `:993` XXX (unused images of 'R' obsids excluded from archive). |
| Limit cameras/targets to those present | Largely satisfied (labels are per-camera; collections genuinely contain both moons/cameras). Optional. |
| Cassini field diffs vs original bundle | Content decisions; two "FROM WHERE?" answers found — see §5 item 14. |
| `rings:description` update | Blocked on external dictionary cleanup; defer. |
| Verify sentinels in bkgnd mosaics | **Done — PASSES** (§3.3): all 305 pairs check out. |
| Wrap-around limits / example labels / bundle.lblx / moons | Marked DONE; consistent with code. |

---

## 8. Verified sound (what does NOT need attention)

- **Structure:** all 11 collections' inventories reconcile with disk exactly
  (sole exception §3.1); zero orphan labels; obsid sets byte-identical across
  all six data/browse collections; bundle.lblx member entries complete/valid;
  bundle time range exactly spans the data; file census ties out
  (21,045/84,180/21,658/20,436/42,107).
- **Labels vs files:** file_size and md5 match for every checked file; all
  labels well-formed XML; XSD-valid (sampled, all namespaces, zero version
  drift across 42k references); `IEEE754LSBSingle` confirmed correct by
  numeric read; date formats uniform `…Z`; no sentinel/N/A leakage.
- **Data self-consistency:** params tables ↔ image columns exact (including
  dropped-column bkg_sub cases); label min/max/mean stats match table-derived
  values; core radius matches the ellipse model < 1 km; `true_anomaly` =
  (inertial − ϖ) % 360; angles/longitudes in range; SCLK/ET orderings correct;
  C-matrices orthonormal; suppl header byte math exact.
- **Browse:** all 84,180 PNGs match labels (names, sizes, dims, 8-bit
  greyscale); dimension rules match the code everywhere; content sanity-checked.
- **Support:** document label covers all 6 files (MD5s match); PDF intact;
  scripts byte-compile and **run to completion against this bundle** (all four
  examples, incl. the full 20,435-row index scan); context collection exactly
  bijective with label references, all 8 registry-verified; SPICE kernel
  complement complete and spanning; global-index field definitions verified
  byte-exact against all rows.
- **Constants/formulas:** orbit constants match `f_ring_util` everywhere;
  guide arithmetic checks; corotation rate exact; commit 83b46eb (inertial
  longitude for satellite core radius) verified correct.
- **Guide:** all filename patterns/trees correct (except `__pycache__`);
  params/global-index column tables match actual headers name-for-name;
  M1–M4/O lists match index notes exactly; browse sizes match (2f4808c);
  LaTeX build clean (no unresolved refs, no overfull boxes).

---

## 9. Recommended path to a final bundle

1. ~~**Merge** `code_review_fixes` into main (`9fc7045`); merge rms-csmithing
   `fix_mosaic_time_float64` (`365621c`)~~ — **both done 2026-08-11.**
2. **New code fixes** (this critique): ~~duplicate-keyword-tolerant PDS3
   lookup, `KeyError` backstop, and a stronger per-image
   guard (§3.1)~~ (all done — see §3.1); still to do: moon window/disclaimer
   (§4.4); `iss-data-user-guide::1.1` (§4.2); xml_schema LIDVID style
   (§4.12); ~~SPICE Time_Coordinates (§5.1)~~ (done in `c1168cb`);
   ~~`__pycache__` prune (§4.1)~~ (not a generator issue); small
   template/text items (§5).
3. **Decisions:** `data_calibrated` forward reference (§4.3); source-product
   VIDs for v2–v9 images (§4.5); stars-as-targets and 'R'-obsid completeness;
   cassini: dead-key fields (§5.14); `iss_199rf_fmovie002_prime` navigation
   check.
4. ~~**Rebuild data:** `ring_ui_mosaic.py` then `ring_ui_bkgnd.py` for all 305
   obsids (reprojection does NOT need rerunning), so mosaic times become
   float64.~~ — **done 2026-08-11**, all 305 rebuilt and the float64 times
   verified.
5. **Guide edits:** §4.7–4.10 concept fixes + §5 minors (on top of the
   already-merged 1a85209).
6. **Regenerate the bundle**; verify ERRORS.log empty; then **re-capture the
   guide's §4 excerpts and counts from this final bundle**, rebuild the PDF,
   and re-run (or re-copy) the document collection.
7. Delete any stray files, run PDS4 `validate` (schematron — XSD alone was
   checked here), and spot-check the 287RI obsid end-to-end.
