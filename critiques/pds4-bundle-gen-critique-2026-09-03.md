# PDS4 bundle and user guide critique, 2026-09-03

**Subject.** The bundle at `pds4_bundle_gen/bundle` (`/data/fring-bundles/pds4`),
generated 2026-09-03 14:17 to 16:39 from `generate_pds4_files.py` at commit
`70aefd4` (ERRORS.log empty, 63 moon-visibility warnings), and the user guide
LaTeX sources under `user_guide/` (`main.pdf` rebuilt 2026-09-03 15:32). The
PDF inside the bundle is the July build and was reviewed only to list what it
lacks.

**Method.** Six independent review passes plus orchestrator spot checks, all
read-only, all scripted where the population allowed it:

- Every one of the 42,405 labels parsed and XSD-validated with lxml against
  the five schemas the labels reference (a wrapper schema importing PDS
  1O00, DISP 1510, GEOM 19B0, RINGS 1E00, CASSINI 1800). PDS4 enumerations
  and reference-type rules checked from the assertion lists embedded in the
  schemas. The official `validate` tool could not run here (no Java), so
  Schematron proper is still outstanding.
- Every file referenced by a label (148,380 references, 58 GB) checked for
  existence, size and MD5; every table parsed record by record against its
  fixed-width field definitions; every array read with its declared dtype
  and shape; every PNG header read.
- All 305 mosaics and 305 background-subtracted mosaics read and compared
  with their params tables, src_imgs tables, labels, index rows, browse
  images and the on-disk pipeline files; orbit quantities and moon
  positions recomputed (cspyce, the generator's kernel set).
- All 20,584 reprojected images: label versus params table versus array,
  suppl.txt parsed and the C-matrices compared with the CK kernels, 671
  products compared field by field with their PDS3 source labels, targets
  and star handling re-derived.
- Support products, the three global indexes (every row, every field), the
  PDS registry for every context and schema LIDVID, NAIF for the kernel
  names, and the full text of one label of every kind.
- The guide: every checkable claim re-derived against the bundle and code,
  every verbatim excerpt diffed, every script run from a scratch copy, and a
  separate line-by-line editorial pass.

Accepted non-issues from earlier reviews (hardcoded paths, midtime as
`metadata['time']`, imshow extent, signed Prometheus sort, contributor
sequence numbers, constant incidence per mosaic, the bkg_sub mask, the
lit-side angle convention, the two forward references, the missing L2 Puppis
context product as such) are not re-reported. This review was done without
reading the 2026-08-10 critique; section 6 compares the two afterwards.

Paths are relative to `pds4_bundle_gen/bundle/` for bundle files and to the
repo root otherwise. `G:` means `pds4_bundle_gen/generate_pds4_files.py`.

---

## 1. Summary

The archive is structurally sound: every label validates, every inventory
reconciles, every checksum matches, every array and table is internally
consistent, and the orbit, timing and geometry numbers reproduce from the
stated constants. The problems are in a small number of generator code paths
that write wrong or self-contradicting values into otherwise good products,
in the C-matrix supplemental files, and in a user guide whose numbers,
excerpts and several claims lag the bundle.

### 1.1 Must fix before delivery

| # | Item | Products affected |
|---|------|-------------------|
| B1 | `cassini:image_observation_type` holds the literal string `SUPPORT', 'SCIENCE` (a Python set stringified) | 65 reprojected images |
| B2 | Nine images archived with "Navigation Type = Manual" and a zero offset; the pipeline used automatic offsets of up to 55 px for them, so the archived C-matrix is not the pointing that made the product | 9 reprojected images |
| B3 | One archived C-matrix is rotated ~76° about the boresight; the "Navigated Boresight Roll" header is inconsistent for products with a boresight near declination −64° | 1 product (matrix), a handful (roll header) |
| B5 | 143 reprojected images archived without contributing to a mosaic say "was not used to create mosaic X" in the description and "is used in the mosaic named X" in the comment, and reference both mosaics as derived products | 143 |
| B9 | Seven background-subtracted mosaics list, count and reference source images that contribute no data after the background fit; 99 reprojected images cite a bkg_sub mosaic whose time span excludes them | 7 mosaics, 102 src_imgs rows, 99 reproj labels |
| B24 | The five `xml_schema` inventory LIDVIDs do not exist in the PDS registry, which now registers schema products under LIDs that embed the IM version | 5 rows (Engineering Node) |
| B26 | The document collection ships the July PDF (41 pages) while the sources are at 44 pages with every change since; the document label gives the guide two author lists and three titles | 1 product |
| G1–G3 | Guide: both image counts wrong (20,303 versus 20,584), every section-4 excerpt stale (two contradict section 3's own formulas), the star claim false for L2 Puppis | guide |

### 1.2 Decisions for the author

- B4: the archived C-matrices are not the physical instrument orientation from
  the CK kernels; they are that orientation rotated by the stellar aberration
  at the image time (up to about 5.5 NAC pixels) plus the navigation offset.
  Either document the convention in the suppl header, the label and the
  guide, or build the matrix from the apparent boresight.
- B7: every label and the guide say "CISSCAL 4.0"; every PDS3 source label
  says "Calibrated using CISSCAL 4.0beta".
- B16: `cassini:mission_phase_name` uses EQUINOX MISSION and SOLSTICE MISSION
  after 2008-07-01 where the PDS3 source labels say EXTENDED MISSION and
  EXTENDED-EXTENDED MISSION. Both vocabularies are legal in the LDD.
- B19: `ISS_191RI_RCASOCCB001_VIMS` is named as an R Cassiopeiae occultation
  but carries no O note and no star target; its sibling
  `ISS_180RI_RCASOCC001_VIMS` has both.
- B15: whole-second `Time_Coordinates` are rounded, so 9,972 reprojected
  images state a start after the shutter opened; keep milliseconds or
  floor/ceil.
- B21: the global-index SCLK columns are typed `ASCII_Real` although the
  fraction is in 1/256-second ticks.

### 1.3 Counts

| Area | Blocker/major | Minor | Cosmetic | Passed checks (see section 3) |
|------|---------------|-------|----------|-------------------------------|
| Labels and structure | 1 (B24) | 4 | 6 | 42,405 labels, 148,380 files, 11 inventories, 296,759 references |
| Reprojected images | 5 (B1–B5) | 3 | 2 | 20,584 products, 671 PDS3 comparisons |
| Mosaics | 1 (B9) | 3 | 3 | 610 products, 5.7 M table rows |
| Support and indexes | 1 (B26) | 5 | 4 | 21,194 index rows, 16 registry lookups |
| Guide, technical | 3 (G1–G3) | 12 | 1 | class lists, formulas, scripts, LaTeX build |
| Guide, editorial | 0 | 44 | 84 | spelling, terminology, references |

---
## 2. Bundle findings

Severity: **blocker** stops delivery or fails `validate`; **major** puts a
wrong value or a self-contradiction into an archived product; **minor** is
misleading or imprecise but not wrong in the data; **cosmetic** is style.
Every finding below was reproduced by at least one reviewer and, where
marked, again by the orchestrator.

### 2.1 Reprojected images (`data_reproj_img`)

**B1. `cassini:image_observation_type` is a stringified Python set. Major,
65 products.** Example
`data_reproj_img/iss_105ri_tmapn45lp001_cirs_3/1614945040w_reproj_img.lblx:177`
reads `<cassini:image_observation_type>SUPPORT', 'SCIENCE</...>`. Values over
all 20,584 labels: SCIENCE 20,510, SUPPORT 9, this string 65 (largest groups
`iss_105ri_tdifs20hp001_cirs` 21, `iss_283ra_complita2001_cirs` 10,
`iss_109ri_tdifs20hp001_cirs` 9). The PDS3 keyword is multi-valued
(`{"SCIENCE","SUPPORT"}`); pdsparser returns a set and `G:2050`
`str(...).strip("{}'")` strips only the outer characters. Set iteration
order is hash-dependent, so the string is also not reproducible between
runs. The LDD declares the element `maxOccurs="unbounded"`
(`data_dictionaries/PDS4_CASSINI_1O00_1800.xsd:216`); emit one element per
value. It is XSD-valid, so no validator will catch it. Confirmed by the
orchestrator.

**B2. Nine products archived with a zero navigation offset and "Navigation
Type = Manual". Major, 9 products.** `G:1444-1446` tests
`if 'manual_offset' in offset_metadata`, but these nine offset files contain
the key with value `None`; the generator then passes `offset=None` to
`oops.fov.OffsetFOV` (`G:1688`), which is a (0, 0) offset. The reprojection
pipeline tests the value (`mosaics/ring_ui_reproject.py:413-416`) and used
the automatic offset. Products and the offsets actually used:
`iss_007ri_lphrlfmov001_prime/1493652861n` (stars, 54.91/0.44 px),
`iss_091ri_apomosl109_vims/1604291743w, 1604291858w, 1604292268w,
1604292383w` (model, about 2/1.5 px), `iss_173ri_spokemov002_prime/1728612560w`
(stars), `iss_233ri_hiresafrg001_prime/1836263420n` (stars, 18.0/−6.3 px),
`iss_244ri_propretrg001_prime/1854445005n` (stars, −6.4/7.5 px),
`iss_286ri_casdivlit001_cirs/1880157547n` (stars, 3.2/2.0 px). The archived
boresight in each equals the (de-aberrated) CK boresight to 0.00 px, so the
suppl.txt describes pointing that was never used. Fix: test
`offset_metadata.get('manual_offset') is not None`. For the other 20,575
products the archived navigation type matches the offset-file logic exactly
(Stars 15,469, Ring/Satellite Models 4,177, Manual 938).

**B3. One C-matrix is twisted about the boresight; the roll header flips
for boresights near declination −64°. Major, 1 product plus a few
headers.** In
`data_reproj_img/iss_180rf_hiresfrng001_prime/1738428217n_reproj_img_suppl.txt`
the first two matrix rows (0.734, 0.627, −0.260 / 0.534, −0.770, −0.350)
bear no relation to those of the neighbouring images 1738428033n and
1738428401n (0.697, −0.594, −0.402 / −0.583, −0.795, 0.164), while the
third row agrees; the camera X and Y axes are 75.85° from the CK axes.
Cause: `extract_roll_from_cmat` (`G:1376-1390`) and
`rebuild_cmatrix_from_ra_dec_roll` (`G:1393-1415`) each pick the roll
reference vector with `[0,0,1] if abs(z[2]) < 0.9 else [0,1,0]`, but on
different z vectors (CK boresight z[2] = −0.900058, navigated boresight
z[2] = −0.899884), so the roll is measured against one reference and applied
against the other. The same code makes the "Navigated Boresight Roll" header
jump by ~76° between otherwise identical neighbours (1738428401n reports
−67.1252° where its neighbours report −143.3°) whenever |z[2]| crosses 0.9.
Eleven products have |z[2]| within 0.002 of 0.9. Use one reference vector
for both functions, chosen from one of the two z vectors. Confirmed by the
orchestrator from the three files.

**B4. Every archived C-matrix has stellar aberration folded in, and nothing
says so. Major as documentation, decision as data, 20,584 products.** The
archived boresight is not "CK boresight plus navigation offset": over all
products with an offset the discrepancy is median 2.7 NAC-equivalent px, p99
7.1 px, max 10.0 px. If the CK boresight is first de-aberrated (true
direction of a source that appears on the axis, using Cassini's velocity
from the bundle's own kernels) the discrepancy drops to median 0.05 px, max
2.0 px. The cause is `G:1695-1696`, which take the navigated boresight from
`right_ascension(apparent=False)` / `declination(apparent=False)` and
rebuild the matrix from it. The label calls the file "a C-matrix rotation
from the J2000 reference frame to the camera frame" and points to the NAIF
CK documentation; a user who applies SPICE aberration corrections with this
matrix double-counts up to about 33 µrad (5.5 NAC px, 0.55 WAC px), the same
order as the navigation corrections the file exists to convey (median 11
px). The header line `Stellar Aberration Correction = No` is the only hint,
and the guide reproduces the header without explanation (`04:181-200`).
Either use `apparent=True` or state in the header, the File comment and the
guide that the matrix reproduces apparent positions with no aberration
correction applied. The reprojected data are unaffected.

**B5. 143 labels contradict themselves about mosaic membership. Major, 143
products.** These are exactly the images in no src_imgs table (20,584 −
20,441): `iss_199rf_fmovie002_prime` 97 of 120, `iss_262rf_fmovie001_prime_12`
27, `iss_256ri_hiresafrg002_prime` 16, `iss_268rf_fmovie001_prime_1` 3, all
"R" observations whose unused images are archived by design (`G:1167-1183`).
In `data_reproj_img/iss_199rf_fmovie002_prime/1765020030n_reproj_img.lblx`
lines 45-47 say "but was not used to create mosaic iss_199rf_fmovie002_prime,
which covers only part of the observation", lines 70-71 say "this
reprojected image is used in the mosaic named iss_199rf_fmovie002_prime",
the nav-quality sentence says "for all of the images for mosaic ...", and
lines 323-336 reference both mosaics with `data_to_derived_product`. Cause:
`G:2231-2237` branch the description on `used_in_mosaic`; the comment at
`G:2258` and the Reference_List block in `templates/data_reproj_img.lblx:266-278`
are unconditional. The guide promises at `03:228-230` that the label states
which case applies. Confirmed by the orchestrator (143 of 143).

**B6. Undisclosed −999 in physical-quantity fields. Minor, 2 products (plus
2,754 by LDD convention).** `iss_007ri_lphrlfmov001_prime/1493625366n` and
`1493638821n` carry `<cassini:filter_temperature unit="degC">-999.` and
`sensor_head_electronics_temperature` = −999 (copied verbatim from PDS3,
`G:2043, 2083`) with nothing in the label marking them as fill. Every WAC
image (2,754) has `optics_temperature_back` = −999, which the LDD documents
as the WAC convention. The other placeholders (`calibration_lamp_state_flag`
N/A in 17,830, `telemetry_format_id` UNK in 1,593, `missing_lines` −1 in
1,072 with an in-label comment) are string flags or LDD-documented.

**B7. "CISSCAL 4.0" versus "4.0beta". Decision, 21,194 labels and the
guide.** All 1,137 PDS3 calibrated labels checked say `Calibrated using
CISSCAL 4.0beta:` (product creation 2019). Every comment (`G:2255, 2505,
2544`) and `01:12` say "CISSCAL 4.0".

**B8. The per-image moon sentence always says "not visually confirmed".
Cosmetic, 454 products.** `G:2280-2288` append "although its presence has
not been visually confirmed" to every reprojected-image label that lists a
moon, including the 417 (Prometheus) and 37 (Pandora) whose parent mosaic
says "has been visually confirmed". Accurate per image, but it reads as a
contradiction of the mosaic label.

### 2.2 Mosaics (`data_mosaic`, `data_mosaic_bkg_sub`)

**B9. Background-subtracted mosaics count source images that contribute no
data. Major, 7 products.** After the background fit drops every longitude an
image supplied, the image survives in the bkg_sub src_imgs table, in the
comment's "reprojections of N source images", and in the index `num_images`:

| product | listed | contributing |
|---|---|---|
| iss_007ri_hpmrdfmov001_prime | 169 | 105 |
| iss_007ri_lphrlfmov001_prime | 233 | 205 |
| iss_098ri_tmapn30lp001_cirs | 24 | 22 |
| iss_100ri_subms20lp001_cirs | 24 | 23 |
| iss_105ri_tdifs20hp001_cirs | 21 | 17 |
| iss_172ri_betpegocc001_vims | 9 | 7 |
| iss_197rf_fmovie002_prime | 78 | 77 |

Cause: `remap_image_indexes()` (`G:1151`)
`used_indexes = sorted(set(image_indexes) - set([SENTINEL]))` ignores
`long_antimask`; the dropped columns keep their image numbers in the on-disk
`BKGND-SUB-METADATA.dat` (5,039 invalid columns carry indexes 0-63 for
hpmrdfmov001). Time_Coordinates are computed from valid columns only
(`G:1830-1840`), so 99 of the 102 phantom images have start/stop times
outside the bkg_sub mosaic's Time_Coordinates, and their labels cite that
bkg_sub mosaic as `data_to_derived_product` although none of their pixels
are in it. The 305 plain mosaics have zero such rows. Fix: mask the image
indexes with `long_antimask` before taking the set.

**B10. A mosaic lists a moon that none of its source images lists. Minor,
3 observations (6 products).** `iss_057rf_fmovie001_vims` (Prometheus,
"presence has been visually confirmed"), `iss_085rf_fmovie003_prime_1`
(Pandora), `iss_105rf_fmovie002_prime` (Prometheus). The mosaic test admits
a visually confirmed moon up to `SATELLITE_VISUAL_RADIAL_TOLERANCE` = 50 km
outside the radial limits (`G:1011, 1105-1111`); the per-image test runs
with tolerance 0 (`G:2182-2183`). Since the mosaic holds exactly the images'
pixels, the two should agree. The remaining 22 mosaic/image disagreements
are explained (6 bkg_sub products drop the longitudes, all logged; 16
where the mosaic pixel at the moon's longitude came from a different image).

**B11. Full-coverage mosaics report an arbitrary wrapped span. Minor, 54
mosaics, 32 bkg_sub, 86 index rows.** For every product with 18,000 valid
longitudes the comment says, e.g.
`data_mosaic/iss_000ri_satsrchap001_prime/..._mosaic.lblx:77` "a total of
360.00 degrees ... spanning the (possibly discontinuous) 359.98 degrees from
256.04 to 256.02", and the index rows carry 256.04/256.02 as the
minimum/maximum corotating longitude. `wrapped_minmax` (`G:696-710`) takes
argmax over gaps that are all 0.02, and float rounding in
`np.arange(18000)*0.02` makes the gap at 256.02→256.04 win. Expected "from
0.00 to 359.98". Related: where several gaps tie exactly (three spoke-movie
bkg_sub products), the chosen endpoints are the first tie.

**B12. "total of X degrees ... spanning Y degrees" has X = Y + 0.02 for
every contiguous product. Minor, 20,003 reproj + 268 mosaic + 219 bkg_sub
labels.** `G:2137` counts bins (n × 0.02) while `G:2136` computes max − min,
which omits the last bin (e.g. 252 samples: "total of 5.04 ... spanning
5.02 from 75.00 to 80.02"). Use (max − min) mod 360 + 0.02, or say the span
is exclusive.

**B13. The CISSCAL sentence appears twice in R and N mosaic labels.
Cosmetic, 42 labels.** The 9 "R" and 12 "N" mosaics and their bkg_sub twins
say "The source images were calibrated using CISSCAL 4.0 and the data values
are in units of I/F." twice (`G:2505-2506` or `2516-2517`, then `2543-2545`).
Example `data_mosaic/iss_191rf_fmovie001_prime_1/..._mosaic.lblx`.

**B14. Angles printed as 360.000, and tie-broken endpoints. Cosmetic.**
`true_anomaly` = 360.000 in `iss_110rf_fmovie002_prime` (corot 292.54) and
`iss_201ri_spokemov001_prime_1` (337.72); inertial longitude 360.000 in two
rows of `iss_176rf_fmovie001_prime`; same rows in the bkg_sub twins.
`%7.3f` after `% 360` (`G:2771-2782`) rounds 359.9996 up. A range check on
[0, 360) will flag them.

### 2.3 All data products

**B15. Time_Coordinates are rounded to the nearest second. Minor, 20,584 +
610 labels, 3 collections, bundle, index columns.** `et_to_datetime`
(`G:573-575`) calls `julian.ymdhms_format_from_tai(..., digits=None)`, which
rounds; `start_date_time − start_time_doy` is uniform on [−0.5, +0.5] s.
9,972 reprojected images state a start after the shutter opened, 9,015 a
stop before it closed, 2,269 a start later than their own
`cassini:image_mid_time`, and 4,375 have start = stop. Example
`data_reproj_img/iosic_276rb_complitb3001_si/1874525875w_reproj_img.lblx:93`
start 2017-05-26T20:29:10Z with `start_time_doy` 2017-146T20:29:09.814.
Millisecond strings already exist (`START_DATE_TIME_3`, `G:2012/2014`, used
in suppl.txt); `ASCII_Date_Time_YMD_UTC` accepts them. The bundle span
2004-06-20T19:15:31Z to 2017-09-07T21:51:58Z happens to bracket the true
extremes (19:15:31.427, 21:51:57.901). Confirmed by the orchestrator.

**B16. `cassini:mission_phase_name` vocabulary. Decision, 16,886 labels.**
The generator derives the phase from the date (`et_to_tour`, `G:1304-1320`)
and writes EQUINOX MISSION (56 + 56 + 4,944 labels) and SOLSTICE MISSION
(214 + 214 + 11,402). The PDS3 calibrated labels and the raw-volume index
say EXTENDED MISSION and EXTENDED-EXTENDED MISSION for the same images
(checked: `COISS_2060/.../N1644997934_1_CALIB.LBL`, 2010-047, "EXTENDED
MISSION"). Boundaries agree (no crossed cases in 1,137 labels checked);
TOUR PRE-HUYGENS and TOUR agree with PDS3. All four strings are in the LDD
enumeration. If the PDS3 vocabulary is wanted, the parsed
`MISSION_PHASE_NAME` (`G:2069`) is already available.

**B17. Array orientation is stated only in XML comments. Minor, 21,194
labels.** The Line axis carries `<!-- vertical (delta radius) -->` and the
`disp` block says Bottom to Top, but no element text says Line 0 is
Δr = −1000 km or that Sample 0 is corotating longitude 0.00 (mosaics) or the
minimum longitude (reprojected images). The guide now states it (`04:163`)
and the data confirm it (moons on the expected side in 100 of 100 mosaic
detections). One sentence in the `Array_2D_Image` description would make
the labels self-contained.

### 2.4 Occultation targets

**B18. L2 Puppis: the labels do not even name the star. Known open item,
new detail; 8 mosaic-type + 37 reproj labels.** The four L2PUPOCC mosaics
carry the O note ("a star is present in each source image ... To fully
explore the occultation, use the reprojected images") while
`Target_Identification` lists only the two rings, and "Puppis" appears
nowhere in any of their labels; the only trace is
`cassini:sequence_title`. `G:499-513` stores the name with LID `None` and
`target_star()` (`G:537-545`) then returns nothing, discarding the name
along with the LID. Until the context product exists, the comment should at
least name the star. The other 10 occultation observations carry their star
(117 reproj labels, 8 star LIDs, all in `collection_context.csv`).

**B19. `ISS_191RI_RCASOCCB001_VIMS` is not treated as an occultation.
Question, 2 mosaic + 9 reproj labels.** `observation_list.csv:146` gives it
no O note and `OCCULTATION_STARS` has no entry, so its labels carry no
R Cassiopeiae target and no occultation note, while
`ISS_180RI_RCASOCC001_VIMS` has both. The mosaic is 9 images over 476 s at
one inertial longitude; the browse images show no obvious star. If the
omission is deliberate, section 3.1.5 of the guide could say so.

### 2.5 Browse products

**B20. Browse label descriptions misstate the rendering. Minor, 21,194
labels.**
- "a blackpoint at the minimum mosaic value" (`G:3244`, and `3203-3205` for
  reprojected images) versus `G:3121` `blackpoint = max(np.min(valid_pixels),
  0)`. All 305 bkg_sub mosaics and 233 plain mosaics have negative valid
  pixels, so their blackpoint is 0. Regenerating every full PNG from the
  code formula matched the shipped files pixel for pixel (mosaics exact,
  bkg_sub within one grey level).
- Reprojected "full" images are described as "containing only the
  longitudes with valid data at full resolution (minimum width 800)" and
  "med" as "downsampled by 10 in longitude (minimum width 400)". The rule is
  full width = max(valid columns, 800), med width = max(valid columns // 10,
  400): 14,598 of 20,584 full PNGs are stretched horizontally (up to 22.9×
  for 35-column images), 19,087 med PNGs are not downsampled by 10, and 559
  full PNGs are narrower than the image because interior sentinel-only
  columns are dropped (`G:3113-3135`).
- No browse label says which way is up (rows are flipped at `G:3146` so
  radius increases upward) or that med/small/thumb PNGs carry a burned-in
  title (`G:3152-3171`).
The PNG dimensions themselves follow the code rule in every product.

### 2.6 Global index tables (`miscellaneous/`)

**B21. SCLK columns are typed `ASCII_Real`. Minor, 3 labels.**
`global_mosaic_index.lblx:235-256` (and the other two):
`cassini:spacecraft_clock_start_count` / `stop_count` are `ASCII_Real`,
`field_length` 14, `unit` none. The values are SCLK strings without the
partition ("1549801215.222") whose fraction is in 1/256-second ticks (all
42,388 values have fractions ≤ 255); read as a Real, .222 means 0.222 s
instead of 0.867 s. Type them `ASCII_String` and say so in the description.

**B22. Index column names and units diverge from the labels. Minor.**
`rings:minimum/maximum_corotating_ring_longitude` in the mosaic indexes hold
the valid-data extremes (e.g. 109.32/80.02) while the product labels hold
0.00/359.98 for the same attribute; 164 of 305 rows have min > max, as do
inertial longitude (6), true anomaly (28), pericenter (1), Prometheus (2)
and Pandora (4) longitudes, and the wrap rule (`wrapped_minmax`) is stated
in none of the field descriptions (`global_mosaic_index.lblx:280-300,
576-590, 600-612, 657-669`). The index resolution fields say km/pixel and
deg/pixel where the labels say km and deg. `<unit>none</unit>` appears on
640 fields (`templates/data_mosaic.lblx:336`, 18 places in
`templates/global_index.lblx`); "none" is not a unit_of_measure value,
though it validates.

**B23. `pds:creation_date_time` differs from the label by 1–2 s. Cosmetic,
409 rows.** The index writes `CURRENT_DATE_TIME` (`G:1812`); the label
writes the file's mtime (`$FILE_ZULU$`, template line 209).

### 2.7 Support collections and bundle level

**B24. `xml_schema/collection_xml_schema.csv` names five products that do
not exist. Blocker for `validate`, Engineering Node action.** The rows are
`pds-xml_schema::1.24`, `disp-xml_schema::1.15`, `geom-xml_schema::1.19`,
`rings-xml_schema::1.14`, `cassini-xml_schema::1.18`
(`templates/collection_xml_schema.csv`, copied at `G:3701`). All five return
404 from the registry. The bare LIDs exist only up to `pds-xml_schema::1.21`
(= PDS 1.15.0.0), `disp ::1.16` (DISP 1.10.1.0), `geom ::1.16`, `rings ::1.9`,
`cassini ::1.17`; since IM 1.14 the schema products are registered under
LIDs that embed the versions, e.g. `pds-xml_schema_1.23.0.0::1.0`,
`disp-xml_schema_1.21.0.0_1.5.1.0::1.27`,
`rings-xml_schema_1.23.0.0_1.13.0::1.0`. No IM-1.24 schema products are
registered yet. Three of the five VIDs also do not follow the shipped
dictionary versions (DISP 1.5.1.0, GEOM 1.9.11.0, CASSINI 1.8.0.0).

**B25. `spice_kernels/kernels.ker` and its label. Minor.**
- Line 9 `cpck15Dec2017_saturn_only.tpc` exists only in the author's tree;
  NAIF holds `cpck15Dec2017.tpc` and `cpck15Dec2017_Nav.tpc`. Every other
  named kernel that could be checked exists at NAIF.
- `kernels.lblx:23` says the file lists "all kernels used to create" the
  products, but the moon columns in every params table and index were
  computed with `pck00010_edit_v01.tpc` (`G:637`, `f_ring_util/moons.py:15`),
  which is not listed; with the bundle's own PCK the moon corotating
  longitudes shift by 0.0022°.
- The file has no comment header and no `PATH_VALUES`/`PATH_SYMBOLS`; the
  `.ker` extension is unusual (NAIF metakernels are `.tm`). Its
  `Time_Coordinates` are nil while the kernels cover 2004-06-19 to
  2017-09-19. The label title is just "Metakernel".

**B26. Document product. Major (stale PDF), minor (metadata).**
- The shipped PDF is the July build (see section 4.4).
- `document/user_guide/f-ring-mosaics-user-guide.lblx`: `<title>Cassini ISS
  F Ring Mosaics User's Guide` (line 13), `<document_name>F Ring Mosaics
  User's Guide` (line 230), and the PDF/guide title "Cassini ISS F Ring
  Mosaics User Guide"; `Citation_Information/List_Author` (lines 31-56)
  names French and Hedman, `Document/List_Author` (lines 282-295) French
  only, and the guide's own citation names French only. Confirmed by the
  orchestrator.
- The PDF is PDF 1.5 from pdfTeX with no PDF/A metadata; the PDS4 Standards
  Reference asks for PDF/A for archived documents. `document_standard_id`
  values are right (two scripts contain one "≈" each and are UTF-8; replace
  it with "~" and all five become 7-bit ASCII).

**B27. `readme.txt` citation lacks the year and version. Cosmetic.** Lines
28-31 give "French, R.S. and Hedman, M.M., ... PDS Ring-Moon Systems (RMS).
DOI 10.17189/3tfh-th07." with no year; the guide recommends "(2026) ...
(Version 1.0) [Data set]. NASA Planetary Data System."

**B28. Cosmetic set, all verified.**
- Five collection labels (browse_mosaic, browse_mosaic_bkg_sub,
  browse_reproj_img, context, xml_schema) use CRLF line endings; the other
  42,400 labels are LF (`templates/collection_browse_*.lblx`,
  `collection_context.lblx`, `collection_xml_schema.lblx`).
- Leading tabs in 21,198 labels: the `xsi:schemaLocation` continuation lines
  of every data label (`templates/data_mosaic.lblx:18-21`,
  `data_reproj_img.lblx:21-25`), `global_index.lblx:9`, `bundle.lblx:131`.
- `Modification_Detail` description "Initial version" (bundle, context,
  document, miscellaneous) versus "Initial version." (13 other support
  labels and all products).
- `Observing_System_Component` order NAC then WAC in `bundle.lblx`, WAC then
  NAC in the document label, `kernels.lblx` and the data/browse collection
  labels.
- The document label declares an unused `xmlns:pds`; `bundle.lblx:340-342`
  is the only `File` without `creation_date_time`.
- Context products appear as bare LIDs in `collection_context.csv` and as
  LIDVIDs in the document and miscellaneous inventories.
- Cassini DOY times (6 × 20,584 values) carry no `Z`; valid, but every other
  time in the bundle has one.
- suppl.txt declination/RA seconds are not zero-padded ("-055d16m0.002s",
  6,388 files; `G:1424, 1437` `{ss:05.3f}` should be `06.3f`).
- `rings:longitudinal_resolution` is printed `%8.5f`, i.e. three significant
  digits for values near 0.003.
- `observation_list.csv`: `ISS_213RF_FMOVIE001_PRIME` has a notes field of
  one space; the "Inertial" column is read (`G:1776-1783`) and never used.

### 2.8 Generator-level, no effect on this build

- The global index row is written (`G:2864-3017`) before `generate_browse`
  runs; a browse failure raises `ObsIdFailedException` and skips the
  inventory rows (`G:3443-3446, 4025-4039`) but leaves the index row.
- Several fixed-width index fields are exactly filled by the longest
  current values (bkg_sub LID 117/117, `file_spec` 101/101,
  `mean_incidence` 6/6, `num_images` 4/4 at 1,030); a longer obsid or a
  5-character notes string would shift columns silently (`G:2912-3016`).
- `PDS4_CASSINI_1O00_1800.xsd:23-24` imports GEOM `1O00_19A0` and CART
  `1O00_1970`; the labels bind GEOM `19B0`. A validator that follows imports
  loads two GEOM versions of one namespace. Not a label defect.

---

## 3. Verified sound

Everything below passed for the whole population unless a count says
otherwise.

**Labels.** 42,405 well-formed and XSD-valid (0 errors); 4 PI/schemaLocation
variants, all the expected 1O00/1E00/1510/19B0/1800 pairs; every enumeration
and unit value in the harvested LDD lists (0 violations); reference_type
values valid for every product class; 42,405 unique LIDs and LIDVIDs with
correct grammar, collection segment = directory, product segment = file
basename; version 1.0, IM 1.24.0.0, publication year 2026, one
Modification_Detail dated 2026-09-03 everywhere; 0 leftover template
artifacts (`$`, XXX, TODO, None, nan, `<--`, doubled spaces, empty elements,
non-ASCII).

**Inventories and references.** 11 collections; every Primary row maps to
exactly one label and back (305/305/20,584/305/305/20,584/1/3/1); `records`
= CSV lines; collection_type matches the bundle member entry; 296,759
internal references resolve with matching VIDs; all 16 context LIDs are
referenced and all references are in `collection_context.csv`; mosaic ↔
bkg_sub, data ↔ browse, reproj → mosaic/bkg_sub/browse links mutual; every
reproj cites `cassini_iss_saturn:data_calibrated:<imgid>_calib::1.0` with the
right imgid; the 8 context products and 8 star LIDs resolve in the registry
at the cited versions.

**Files.** 148,380 referenced files exist (case-sensitive); file_size and MD5
match for all of them (every `.img` included); every Header `object_length`
equals its first line; offset + records × record_length = size for all
42,391 tables; every record LF-terminated; fields tile every record exactly;
402,845 field columns parse as their declared type; 21,194 arrays have the
declared shape with 0 NaN/Inf over 13.08 G pixels; 84,776 PNGs have the
stated dimensions, 8-bit greyscale, none blank.

**Mosaics (610).** Arrays bitwise equal to the on-disk `MOSAIC.npy` /
`BKGND-SUB-MOSAIC.npz`; every −999 in a mosaic is −999 in its bkg_sub twin;
bkg_sub = float32(mosaic − model) bit-exact, the model degree-1 per column
within 0.63 ulp, limits and degree equal to the on-disk background files;
params-table longitudes = valid columns exactly; fixed-width parse = CSV
parse; inertial longitude, core radius, node, pericenter, true anomaly and
moon positions reproduce from the label constants (max 0.0005°, 0.0034 km;
moons versus cspyce 0.0005°); ET per image equals the reprojected image's
ET; every rings min/max/mean, ring radius, Time_Coordinates, SCLK, filter
and camera equals the member images; every comment number (image count,
spans, seconds, hours, quality word) recomputes except B9/B11/B12; src_imgs
contiguous, time-ordered, resolving within the obsid; global index 57 of 60
fields equal the labels for all 305 rows (exceptions B9, B11, B23); browse
PNGs equal the recomputed stretch, vertically flipped, so "Bottom to Top"
is right; Line 0 is the inner edge (moon detections 100/100 on the expected
side); core at rows 196-204 everywhere.

**Reprojected images (20,584).** Every label statistic equals its params
table; sample count = span/0.02 + 1; wrap handled (1,314 products, column 0
= minimum longitude); `observed_event_tdb` = (start + stop)/2 within 0.5 ms;
stop − start = exposure within 2 ms; SCLK consistent within 3 ticks; 671
products compared with PDS3: every copied field equal except B1; mission
phase boundaries consistent; instrument, mode, filters, size all right;
suppl.txt orthonormal (max |RRᵀ − I| 1.7e−10, det 1) with header times and
RA/Dec consistent with the matrix; navigation type matches the offset files
except B2; the F ring lies in the field of view for 606/606 sampled products
with the bundle's own metakernel; each used image in exactly one mosaic's
src_imgs and its Time_Coordinates inside the mosaic's; no image entirely
sentinel, none single-row/column, none constant; browse dimensions follow
the code rule; `ISS_287RI_PROPRETRG001_PRIME` complete (19 products, 19 rows
everywhere).

**Global indexes.** 21,194 rows parse per their labels; every field of every
row equals its source in the product label or table (B9, B11, B23 excepted);
percent_coverage = valid/18000 × 100; min ≤ mean ≤ max; ring radius = core ±
1000 km; notes, nav_quality and bkgnd_quality equal `observation_list.csv`
for every row; background limits (−750/750 in 217 rows, 88 non-standard) agree
with the on-disk models and the label sentences.

**Support.** Citation blocks identical across bundle, 11 collections,
document and index labels; bundle and collection Time_Coordinates equal the
product extremes; `readme.txt` 7-bit ASCII with LF; document label file list,
MD5s and standards match disk; kernel set complete and gap-free for
2004-2017 (971 CKs, 156 SCPSE SPKs); the example scripts run.

**Moon and star targets.** The geometric rule re-implemented on the tables
agrees with the labels for 20,583 of 20,584 images (the one exception is a
rounding-boundary case); "not visually confirmed" text present exactly when
the observation list says N, and every omitted confirmed moon has a
WARNINGS.log line; star targets present for exactly the 10 named-star
occultations.

---
## 4. User guide

The guide reviewed is the LaTeX under `user_guide/` (`main.pdf` rebuilt
2026-09-03 15:32, 44 pages). The PDF inside the bundle
(`document/user_guide/f-ring-mosaics-user-guide.pdf`, byte-identical to
`pds4_bundle_gen/templates/f-ring-mosaics-user-guide.pdf`, dated Jul 23, 41
pages) is an older build; section 4.4 lists what it lacks.

Line references are `section-file:line` into `user_guide/sections/`
(`01` = `01-introduction.tex`, and so on) or into the named file.

### 4.1 Claims that disagree with the bundle

**G1. The reprojected-image count is wrong twice.** `01:14` "In total 20,303
reprojected images are provided along with 305 associated mosaics" and
`03:61` "In total, 20,303 images were chosen for inclusion." The bundle
archives 20,584 reprojected images (17,830 NAC, 2,754 WAC); 20,441 of them
contribute to a mosaic and 143 are the unused images of the four "R"
observations that section 3.1.6 says are archived deliberately. Neither
number in the guide matches either count. The stale PDF has the same 20,303
and "302 mosaics"; only the mosaic count was updated.

**G2. The star claim is false for L2 Puppis.** `03:217-222` lists the three
L2 Puppis entries and then states that each star "is named in a
Target_Identification block in the label of every data product belonging to
its observation, and is linked there to the PDS context product for the
star; the bundle and collection labels list the full set of stars observed."
The four L2PUPOCC mosaics (8 mosaic-type labels) and their 37 reprojected
images carry no star target, the word "Puppis" appears in no label, and the
bundle and collection labels list eight stars. The missing context product is
a known open item; the guide asserts the opposite of the shipped state, and
the labels do not even name the star in free text (see B18).

**G3. Every verbatim label and table excerpt in section 4.2 is stale, and two
contradict the guide's own formulas.**
- `04:90-121` (rings excerpt): the numbers belong to
  `iss_111rf_fmovie002_prime/1622033938n`, not to the `1622049830n` shown in
  the file excerpt below it; the three longitudinal resolutions are printed
  with 3 decimals (0.016/0.018/0.017) where the label has 5
  (0.01554/0.01784/0.01696); the two sampling-interval values are shown on
  continuation lines that do not exist in the label.
- `04:126-160` (File_Area excerpt for `1622049830n`): current label has
  `<elements>649` (guide 655), `file_size` 1040996 (guide 1050620), a 2026
  creation date (guide 2025-07-25), a different md5, and axis comments
  `vertical (delta radius)` / `horizontal (co-rotating longitude)` (guide
  "vertical (radial)" / "horizontal (longitude)").
- `04:167-174` (reproj params rows): the source is
  `iss_292ri_propretrg001_prime/1883514713n`. The bundle's row 2 is
  `320.08, 558091398.772,  61.809,    5.232,  0.00556,  63.312, 129.712,  67.484, 140406.759, 293.578, 186.171, 235.638, ...`;
  the guide prints inertial 101.729, longitudinal resolution 0.00010 (a
  radian value), core radius 140188.617 and true anomaly 275.558. Applying
  section 3.4.1's own conversion to the printed ET and corotating longitude
  gives 61.809 and 235.638, so the excerpt contradicts section 3.
- `04:266-274` (mosaic params rows): no mosaic table in the bundle contains ET
  558093184.000 (a float32-quantized value from an earlier build); the
  current row is `308.04,    8, 558093186.744,  61.812, ...`.
- `04:180-202` (suppl.txt): image `1736795703n` is not in the bundle. The
  format matches the shipped files apart from the trailing space after
  `C-Matrix =`.
- `04:281-285` (src_imgs excerpt) matches `iss_111rf_fmovie002_prime`.
All excerpts must be re-captured from the final bundle, after the generator
fixes in section 2, because several of them (file size, sample count, md5,
creation date) change with every regeneration.

**G4. Notes are said to be described in every reprojected-image label.**
`04:356` "Each note will also be described in text in the comment section of
the label's `<Observation_Area>`" for "each reprojected image and mosaic."
Only mosaic labels carry the note text (`generate_pds4_files.py:2381-2470`);
6,610 reprojected-image index rows have a non-empty `notes` field and none of
their labels explain it.

**G5. The "Full" browse-image rule is stated wrongly, twice.** `04:226`
"'Full' images are always the same size as the reprojected image (with a
minimum of 800 pixels in the X dimension)" contradicts itself whenever the
image has fewer than 800 columns, which is the case for 14,598 of 20,584
images (they are stretched horizontally, up to 22.9x for a 35-column image).
In addition the generator drops interior sentinel-only columns
(`generate_pds4_files.py:3113-3135`), so 559 full PNGs are narrower than
their image. "Med" images are described as "always 400 pixels in the Y
dimension" (the source has 401 rows; they are resampled). `04:301` calls the
same product "Medium" while the file suffix and `04:226` say "Med". The med,
small and thumb PNGs also carry a drawn title (`:3157-3175`), which no text
mentions.

**G6. The stretch blackpoint is not "the minimum valid value".** `04:33`
versus `generate_pds4_files.py:3121` `blackpoint = max(np.min(valid_pixels),
0)`. All 305 background-subtracted mosaics and 233 of 305 plain mosaics
contain negative valid pixels, so their blackpoint is 0 and every negative
pixel renders black. The browse labels repeat the same wording (B20).

**G7. The moon-position kernel statement is inaccurate.** `03:377` says the
Prometheus and Pandora positions "come from the Saturn system ephemerides
sat368.bsp, sat393.bsp, sat427.bsp, and sat428.bsp, and planetary positions
from de438.bsp". The generator loads only `naif0012.tls`, `de438.bsp`,
`sat393.bsp` and `pck00010_edit_v01.tpc` for the tabulated positions
(`generate_pds4_files.py:633-637`; the same four in `f_ring_util/moons.py:12-15`),
and every params-table field description says "computed using the sat393.bsp
SPICE kernel". The metakernel also lists `sat393_daphnis.bsp` and
`sat393-rocks_pan.bsp`, which the guide omits, and does not list the PCK the
generator actually used (B25).

**G8. Figure captions carry stale or ambiguous numbers.** `03:253` third
chunk "136°−185°" versus the index 135.88–186.00. `03:264` "Image #1, #5,
#9" are `image_index` 0, 4, 8 in the src_imgs table (the guide itself
documents the index as 0-based at `04:282`), and the quoted inertial ranges
6.0–9.2, 9.6–12.9, 13.3–16.6 are now 5.706–9.606, 9.307–13.327,
12.935–17.075; "the same 3°" spans are 3.90–4.14°. Figure 5's numbers agree.

**G9. The plot_ews screenshot no longer matches the script.** `05:127-132`
`figures/screenshot-plot-ews.png`: running the shipped `plot_ews_ma.py` on
`iss_036rf_fmovie001_vims` (the quick-start command) now peaks near 21 km with
a baseline near 5 km; the screenshot peaks near 35 km with a baseline near 0.
The display and find-Prometheus screenshots match current output.

**G10. Title and authorship disagree with the document label.** The guide is
titled "Cassini ISS F Ring Mosaics User Guide" (`frontmatter-body.tex`,
`preamble.tex`) and its citation block names French alone. The document label
`document/user_guide/f-ring-mosaics-user-guide.lblx` has `<title>Cassini ISS
F Ring Mosaics User's Guide`, `<document_name>F Ring Mosaics User's Guide`,
lists French and Hedman in `Citation_Information/List_Author`, and French
alone in `Document/List_Author`. Three names and two author lists for one
document; pick one of each.

**G11. Wraparound handling is referred to the wrong section.** `03:352` and
`04:176` send the reader to Section 5 "for examples on how to handle this
case in software". Section 5 never mentions wraparound; the column formulas
that handle it are at `04:176` itself, and none of the example scripts
demonstrate it.

**G12. IMGID and OBSID are never defined.** Both appear from `02:19` and
`02:41` onward in every file-name pattern, but nothing says IMGID is the
10-digit spacecraft clock count of the source image followed by `n` (NAC) or
`w` (WAC), which is the only key linking a product to its COISS source.
`04:128` shows `1622049830n` without explanation. LID and LIDVID are likewise
used (`02:51`, `05:107`) before being defined.

**G13. The inertial-longitude definition names the wrong plane.** `03:326`
"the ascending node of Saturn's invariable (equatorial) plane on the Earth's
mean equator of J2000". Saturn's invariable (Laplace) plane and its equatorial
plane are different planes; the labels declare `reprojection_plane = Equator`.
Say "equatorial plane".

**G14. "Rides with the mosaic" wording for nav_quality is wrong for the 143
unused images.** `07:141` and `07:143` describe `nav_quality` and `notes` of
a reprojected image as those of "the mosaic that uses it"; 143 archived
images are used by no mosaic (they inherit the observation's grade).

**G15. Smaller factual points.**
- `03:412` "the innermost 50 pixels (Δr = −1000 to −750 km)": the code uses
  rows [0, 50) = −1000 to −755 km on the inner side and rows [350, 401) =
  +750 to +1000 km (51 rows) on the outer side, so the two margins differ by
  one row and neither is exactly what the sentence says.
- `03:29` glosses FMONITOR as "CIRS scanned the A and B rings while ISS
  observed the F ring", but six of the nine FMONITOR mosaics in the N list are
  `_prime` observations.
- `03:38-40` gloss SUBM*, TDIFS* and TMAP* identically as "Temperature map
  taken by CIRS".
- `03:363` "an incidence angle below 90° places the Sun on the lit side (as it
  is for every observation in this bundle)": with the lit-side convention the
  angle is below 90° by definition, so the parenthetical is circular.
- `01:8` "June 30, 2004" for Saturn orbit insertion is the Pacific date; the
  UTC date is July 1.
- `03:65` says each class "is indicated ... by a note in the global index
  file"; standard F movies carry no note, and notes B, C and E exist only in
  the list at `04:363-365`.
- `04:359-368` print the generator's placeholders (`{root_obsid}`,
  `{total_secs}`, ...) without saying they are substituted, and paraphrase the
  strings (the labels say "co-rotating" and quote "movies").
- `06:25` and `06:73` cite the Cassini ISS Data User's Guide under two titles
  and two locators.
- `README.md:4` "41-page PDF" (44 pages); `README.md:74-77` says autocorrect
  artifacts remain in section 4 that no longer exist.

### 4.2 Verified against the bundle (passed)

Class lists M1 (8 root observations, 19 mosaics), M2 (25 pairs), M3 (31),
M4 (18), R (9), N (12) and O (14) match the `notes` column of the global
mosaic index exactly; the eight star names match the generator table; 42
SPOKEMOV mosaics, all WAC; rev `00A` and target codes RA/RB/RF/RI/ST all
present. Directory trees and file-name patterns for all eleven collections
are correct. Every params-table and global-index column list in section 7
matches the actual headers name-for-name and in order. Orbit constants,
the corotation rate (0.006735694444 deg/s = 581.964 deg/day), the epoch ET
220881665.1839181 s, the core-radius formula and the a(1∓e) range
139,892–140,551 km all agree with `f_ring_util/f_ring.py`, the generator and
the labels. The stitching rule at `03:385` (most valid radii, tie broken by
smaller radial resolution) matches `nav/ring_mosaic.py:1135-1224`. Line 0 is
the inner edge, as `04:163` states (moon detections on the expected side in
100 of 100 mosaics). The four example scripts are byte-identical to
`templates/examples/`, export the 13 functions listed at `05:103-111`, and
run to completion on the quick-start paths. LaTeX: no undefined references,
no overfull boxes, all 15 figures present, no `.tex` newer than `main.pdf`.
All DOIs and URLs resolve except the two bundle DOIs (unregistered until
release); four URLs redirect (`saturn.jpl.nasa.gov/mission/`, the two
`pds-rings` camera descriptions, `adsabs`).

### 4.3 Editorial review

A line-by-line proofreading pass produced 132 items: 3 spelling, 30
grammar, 43 typography/LaTeX, 21 terminology, 5 acronym, 3 numeric, 20
clarity, 2 frontmatter, 5 README. Appendix A lists all of them with
file:line and a replacement. The ones a reader will notice:

- `03:35` "sub-subspacecraft latitude" (the one true misspelling).
- `07:12` `IMG_reproj_img_metadata_params.tab` for `IMGID_...`; `07:19` a
  file name without `.tab`; `07:201` "populate mosaic pixel".
- Parenthetical citations used as nouns: "See (French et al. 2012)",
  "the orbit fit from (Albers et al. 2012) Table 3 fit #2", "(Murray et al.
  2008) used ..." (`03:280`, `03:308`, `03:338`).
- `03:310-322` orbit elements set half in text italics and half in math
  mode, with a spaced degree sign and an en dash as a minus sign.
- U+2212 (minus) used as a range dash in `03:141-143`, `03:253`, `03:264`,
  `03:271` and as a sentence dash in `04:53`; `preamble.tex:51` maps it to a
  math minus, so these print wrongly.
- The `_1/2/3` chunk shorthand (`03:90-97` and elsewhere) is never
  explained.
- `02:41` is missing a `\newline`, so two collection names run together in
  the quick-start table.
- Directory names appear as `\textit`, `\code` and bold in different places;
  program and function names in section 5 are bold where every other file
  name is `\code`; `\textbf{ }` at `05:101`.
- Acronyms used before definition: CDAP, CIRS, UVIS, ET, TDB, UTC, SCLK, LID,
  LIDVID; CSV is defined twice.
- `\zcite` citation years out of order (`03:73`); Zotero artifacts in three
  references (`06:53`, `06:61`, `06:65`).
- `06:6` "Saturn F Ring Cassini ISS Mosaic Bundle" for the bundle's name.
- `Section \ref` with a breakable space in 96 places versus `Figure~\ref`.

### 4.4 What the archived PDF lacks

The shipped PDF (Jul 21 build, 41 pages) predates every guide change since
July. Relative to `main.pdf` it lacks: the citation and errata block; the
305-mosaic count (it says 302); the rev `00A` note; the R and N mosaic lists
and the stars in the O list; the nav_quality and bkgnd_quality grade
definitions; the ϖ0/Ω0 epoch statement and the inertial-longitude
definition; the λ term in the core-radius formula; the lit-side convention
sentence; the SPICE-kernel sentence; the signed `bkgnd_lower/upper_limit`
semantics and the negative-values paragraph; the stretch description; the
Line-0 orientation paragraph and the column formulas; `mosaic_utils.py` in
the document tree; the units paragraph in section 7. It also still says
"miscellaneous bundle", "src_imgs maps to name", "Min and Max True
Anomaly", "Details of the stretching process are not provided", and shows a
transposed C-matrix in its excerpt. The document collection must be
regenerated after the final PDF is copied to `templates/`.
## 5. Recommended order of work

1. Generator fixes that change product content: B1 (observation type), B2
   (manual-offset test), B3 (roll reference vector), B4 (aberration
   convention, or its documentation), B5 (conditional comment and
   references), B9 (mask indexes with `long_antimask`), B11/B12 (span
   arithmetic), B13 (duplicate sentence), B15 (millisecond times), B17
   (orientation sentence), B18 (name the star), B20 (browse text), B21/B22
   (index typing and descriptions). Decide B7, B16, B19.
2. Ask the Engineering Node for the registered schema LIDVIDs (B24) and the
   L2 Puppis context product; fix `kernels.ker` and its label (B25).
3. Regenerate the bundle; confirm ERRORS.log is empty.
4. Guide: fix G1–G15 and the editorial list, then re-capture every
   section-4 excerpt and figure-caption number from the regenerated bundle
   and re-take the plot_ews screenshot; align the title and author list
   with the document label (G10/B26); rebuild `main.pdf`; copy it to
   `templates/`; regenerate the document collection.
5. Run `validate` with Schematron. Expect the two known forward-reference
   failures and, until step 2 completes, the xml_schema failures.

---

## 6. Comparison with the 2026-08-10 critique

This review was written without consulting the earlier critique. Read
afterwards, the two agree on the state of the bundle where they overlap,
and this review adds a set of findings the earlier one could not see
because they lie inside values that validate.

**Earlier findings now verified fixed in this build.**
- 287RI crash and phantom rows: all 19 products present, inventories clean.
- float32-quantized mosaic times: every mosaic ET equals the per-image
  float64 midtime.
- Moon policy (50 km tolerance, wrap-aware edge test, conditional
  disclaimer): in place; warnings down to 63 and every remaining one
  explained.
- `iss-data-user-guide::2.0`, `mission.cassini-huygens::1.5`, SPICE
  Time_Coordinates, Special_Constants wording, the `__pycache__` directory:
  all resolved.
- Guide: axis orientation, inertial-longitude definition, signed background
  limits, quality-grade definitions, negative-value paragraph, citation and
  errata block, element epoch, units paragraph, suppl preamble, column
  mapping, SPOKEMOV and rev 00A, the R and N lists, the "Like the R
  mosaics" fix, and every listed typo except the `\textbf{ }` at `05:101`.

**Earlier findings still open.**
- IMGID never defined (G12); section-4 excerpts stale and image counts
  wrong (G1, G3), which the earlier critique said to redo only after the
  final regeneration, so this is expected; xml_schema LIDVIDs (B24);
  inventory VID style, "Initial version" period and the tab at
  `bundle.lblx:131` (B28); 559 narrower full browse PNGs, now described with
  different but still inaccurate wording (B20); wraparound examples promised
  for section 5 (G11); the moon-kernel sentence now exists but is inaccurate
  (G7).

**New in this review** (not in the earlier critique): B1, B2, B3, B4, B5,
B6, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B19, B21, B22, B23, B25
(pck and cpck), B26 (author lists, three titles, PDF/A), B27, and on the
guide side G2, G4, G5, G6, G8, G9, G10, G13, G14 and most of section 4.3.
The earlier critique's section 5 generator items (dead keys, `{:4d}` width,
`obsid` global, camera check) were not re-examined here; section 2.8's fixed-width
note overlaps its item 8.

---

## Appendix A. Editorial findings in the user guide

Per-file line numbers. Each item gives the observed text and the intended
replacement; severity is minor unless marked cosmetic.

### frontmatter-body.tex
- `:6` `V1.0` versus "Version 1.0" at lines 29, 35, 40. Cosmetic.
- `:27-30` bundle citation names French and Hedman; the title page and the
  guide citation name French alone (see G10).
- `:28-29` the formal bundle title appears only here; the body says
  "Cassini ISS F Ring Mosaics bundle" and `06:6` "Saturn F Ring Cassini ISS
  Mosaic Bundle". Introduce the formal title once at `01:10`.

### preamble.tex, fontmacros.tex, Makefile, README.md
- `preamble.tex:51` maps U+2212 to `\ensuremath{-}`, so every U+2212 used
  as a range dash (`03:141-143, 253, 264, 271`; `04:53`) prints as a math
  minus. Use en dashes there.
- `fontmacros.tex:2`, `Makefile:1` say "User's Guide" in comments. Cosmetic.
- `README.md:4` "41-page PDF" (44). `README.md:74-77` says autocorrect
  artifacts remain in section 4; none do. `README.md:3` still describes the
  sources as a Word conversion. `README.md:64` "14 citation fields" (15
  references). Cosmetic.

### 01-introduction.tex
- `:8` "fly-bys" → "flybys"; "quality images" → "good-quality images".
  Cosmetic.
- `:10` straight apostrophe in "Saturn's" inside a curly-quoted title;
  CDAP never expanded; "FMOVIEs" versus "F movie(s)" elsewhere.
- `:14` "In total 20,303" → "In total, 20,584" (G1).
- `:18` "First time users" → "First-time users". Cosmetic.

### 02-quick-start.tex
- `:6` footnote repeated verbatim at `04:6`. Cosmetic.
- `:8` `\textit{etc}.`; italic "e.g." in 07 but roman in 03/05. Cosmetic.
- `:25`, `:47` "N×401" / "18000×401" with lowercase x/y where `04:163` and
  `05:62` describe (401, N) and "X axis"/"Y axis". State once: a 401-row by
  N-column array.
- `:34` "[and background-subtracted mosaics]" square brackets in prose.
  Cosmetic.
- `:41` missing `\newline` after "[original]" so the two collection names run
  together (sibling `:45` has it).
- `:56`, `:58`, `04:34`, `04:330` "User Guide" plain where elsewhere italic.
  Cosmetic.

### 03-image-selection-and-processing.tex
- `:20` "at the end of mission" → "of the mission". Cosmetic.
- `:21` "F Ring", "A Ring", "B Ring" capitalized. Cosmetic.
- `:22`, `:24` missing commas before "often". Cosmetic.
- `:24-41` en dash between name and gloss; `:208-219` em dash. Cosmetic.
- `:29` FMONITOR gloss versus the six `_prime` FMONITOR mosaics (G15).
- `:35` "sub-subspacecraft" → "sub-spacecraft".
- `:36`, `:103` straight quotes. Cosmetic.
- `:38-40` SUBM*, TDIFS*, TMAP* glossed identically (G15).
- `:43` "If PINST is not "PRIME" then" → add comma; "prime instrument" here,
  "primary instrument" at `:201`.
- `:45`, `:53`, `:350` `\textasciitilde{}` → `$\sim$`. Cosmetic.
- `:53` "It had a minimum radial resolution for the F ring of at most ~80
  km/pixel" → "Its best radial resolution at the F ring was no coarser than
  ~80 km/pixel."
- `:59` "42 SPOKEMOV sequences" are 42 mosaics from 31 observations.
- `:61` "20,303" (G1).
- `:63` "lower case" → "lowercase". Cosmetic.
- `:65` "Each is indicated ... by a note" (G15).
- `:73` "the resulting image sequence and mosaic does not" → "do not";
  citation years out of order.
- `:87` "an observation observed" → "stared at". Cosmetic.
- `:90-97` and elsewhere: the `_1/2` and `_01−17` chunk shorthand is never
  explained; `:141-143` use U+2212.
- `:151` "the stellar position in the two has no relation" → "the star's
  position during ingress and during egress bears no fixed relation".
- `:166` figure label `fig:m4-reproj-imgs` for the R example. Cosmetic.
- `:180` comma splice "doesn't actually follow anything, it just takes";
  "the WAC instrument" → "the WAC".
- `:201` "the VIMS or UVIS instruments were the primary instrument" → "was
  the prime instrument"; CIRS never expanded, UVIS used at `:41` before its
  expansion.
- `:203` uppercase observation names not in `\code`, unlike `:154, 156, 212,
  217`.
- `:228` "are the exception" → "exceptions". Cosmetic.
- Captions `:236`, `:430` end with a period; `:244, 253, 264, 271` do not;
  obsids `\code` in two, plain in four. Cosmetic.
- `:253` "progressive" → "successively increasing"; U+2212 ranges. Cosmetic.
- `:264` double space; "the same 3° of corotating longitudes"; "#1/#5/#9"
  versus 0-based `image_index` (G8).
- `:271` U+2212 and hyphen in one caption. Cosmetic.
- `:280` "The calibrated files at the time of writing are being migrated"
  word order; `\textbf` bundle names → `\code`.
- `:280` footnote, `:308`, `:338` parenthetical citations used as nouns.
- `:286` file name split across `\textbf{\literal{}}` pieces; `\textbf{data_reproj_img}`;
  missing comma before "such as stars".
- `:286`, `02:29` "C matrix" versus "C-matrix" in the shipped text. Cosmetic.
- `:310-322` orbit elements half italic text, half math; spaced degree sign;
  en dash as minus at `:322`.
- `:324` "defined in the next paragraph" (two paragraphs later); ET used
  before its definition.
- `:326` "invariable (equatorial) plane" (G13); "prime meridian".
- `:330`, `:334` formulas mix U+2212 and en dash, "*" and "×", "% 360" and
  "mod 360"; only the second has `\quad`.
- `:338` "the new reference frame" → "a common reference frame"; French et
  al. (2014a) used this bundle's epoch, so it is not a counter-example.
- `:342` stray `\ ` after "="; `ET` in math mode typesets as a product.
  Cosmetic.
- `:350` "Δ\textit{r }= 0" trailing space in italics. Cosmetic.
- `:352` Section 5 reference for wraparound (G11).
- `:357-362` sentence case versus `:368-375` Title Case; "computed for each
  pixel" versus "Mean ..." items. Cosmetic.
- `:363` circular parenthetical (G15).
- `:377`, `04:36` "metakernel" versus `04:374` "meta kernel". Cosmetic.
- `:387` dangling "Similar to reprojected images, for each corotating
  longitude several parameters are stored".
- `:412` "50 pixels (−1000 to −750 km)" (G15); "a linear model which is then
  subtracted" comma; "50 pixel range", "401 pixel radial slice", "very low
  resolution images" hyphens; "Whenever possible the new margins" comma.

### 04-bundle-organization-and-directory-structure.tex
- `:6` the collections sentence is garbled ("browse images for each,
  documentation, miscellaneous ..."); "a PDS4 collection of a different type
  of derived data product" is wrong for document, context and xml_schema.
- `:33-35` space inside `\textbf{...: }`; `:35` missing period. Cosmetic.
- `:36` "needed for most scientific users" → "by". Cosmetic.
- `:46` double space. Cosmetic.
- Directory names italic in headings `:50, 307, 333, 371`, plain in `:70,
  207, 229, 293, 327`, italic in body `:330, 354`, `\code` elsewhere.
- `:53` U+2212 as a sentence dash. Cosmetic.
- Ellipsis "…" at `:61, 62, 84, 285` versus "..." at nine other places.
  Cosmetic.
- `:65` and throughout: OBSID and IMGID undefined (G12).
- `:88` missing comma; `:123` file name not in `\code`. Cosmetic.
- `:163` "IEEE754" versus `05:62` "IEEE 754"; "floating point" hyphen.
  Cosmetic.
- `:165, 178, 262, 264, 278` "IMGID1_"/"OBSID1_" in prose. Cosmetic.
- `:176` "stored with its two sections reordered" misleads (storage runs
  from the minimum longitude through 360°); Section 5 reference (G11).
- `:226` "Full" rule self-contradicting; "Med ... always 400 pixels in the Y
  dimension"; `:301` "Medium" versus "Med" (G5).
- `:262` trailing space inside `\code`; comma before "since". Cosmetic.
- `:276` "and the mapping is simpler" → "but". Cosmetic.
- `:278` "image_index" in curly quotes. Cosmetic.
- `:288`, `05:50` CSV expanded twice; first used at `:37`.
- `:330` `\textit{user_guide}` → `\code`. Cosmetic.
- `:333` heading uses a spaced en dash where others use a colon. Cosmetic.
- `:354` "three global index files" while section 7 has tables for two and
  never names the files.
- `:356` `\code{<Observation_Area>}` with angle brackets; "described in text
  in the comment section" → "in an XML comment".
- `:359-368` placeholders unexplained (G15); `:360-361` "movies consisting
  of" (also in `G:2408-2418`); `:366` "The sequence ... were designed" →
  "was designed" (also in `G:2489-2490`, which is what the labels carry).
- `:374` "meta kernel". Cosmetic.

### 05-reading-labels-and-data-product-files.tex
- `:15` "is defined by PDS4 label" → "by a PDS4 label". Cosmetic.
- `:17` comma before "and readers". Cosmetic.
- `:31` "For the best results, the use of a text editor that understands XML
  is helpful." Cosmetic.
- `:50` "(see below)" points at nothing; "In this case ... in this case".
- `:56` "fixed width" → "fixed-width". Cosmetic.
- `:79, 80` "The read data can easily be used"; `:92` "endeavors to be able
  to read". Cosmetic.
- `:101-116` program and function names in `\textbf` where every other file
  name is `\code`; `\textbf{ }` at `:101`; trailing spaces inside bold at
  `:105, 106`; colon and space inside bold at `:113-116`.
- `:144` "In addition to ... additional software"; "GitHub repo". Cosmetic.

### 06-external-references-and-further-reading.tex
- `:6` "the Saturn F Ring Cassini ISS Mosaic Bundle".
- `:25` and `:73` one document under two titles and two locators. Cosmetic.
- `:45` `\clearpage` inside the itemize. Cosmetic.
- `:53` "AGU Fall Meet Abstr, 54, P54B"; `:61` "(American Association for the
  Advancement of Science)"; `:65` "Abstr 46, 46". Cosmetic.

### 07-metadata-and-global-index-file-fields.tex
- `:10` "percentages are in percent"; TDB and UTC never expanded. Cosmetic.
- `:12` `IMG_reproj_img_metadata_params.tab` → `IMGID_`.
- `:19` first file name lacks ".tab". Cosmetic.
- `:21-49` "source observation" / "original observation" / "source image"
  mixed in one table. Cosmetic.
- `:43, 47` "The longitude of Prometheus/Pandora" → "The corotating longitude
  of".
- `:63, 65, 67, 157, 159, 161` `\textit{e.g. }` with the space inside.
  Cosmetic.
- `:77, 171` field name not in `\code`; "*" and "18000". Cosmetic.
- `:81, 175` "it's possible". Cosmetic.
- `:85` "the distance between the min and max corotating longitude and min
  and max inertial longitude will be identical" → "the two ranges have the
  same width".
- `:141`, `:143` 60-word sentence; "the mosaic that uses it" (G14).
- `:149` heading covers two files.
- `:157` "source image" → "source images" (mosaic index). Cosmetic.
- `:201` "populate mosaic pixel" → "each mosaic pixel". Cosmetic.
- `:271, 273` only rows ending with a period; "core-1000"/"core+1000" with
  a hyphen as minus and an undefined "core". Cosmetic.
- `:141, 259, 261` "G=Good, F=Fair, P=Poor" spacing. Cosmetic.

### Cross-file
- `Section \ref` with a breakable space in 96 places versus `Figure~\ref`.
- "18,000" in prose at five places, "18000" at four.
- Dash usage: en dash, em dash, U+2212 and hyphen all used for ranges or
  sentence breaks in different places.
- "greyscale" (`04:226, 304`) in an otherwise US-spelled document.
- "X axis"/"Y axis" versus "X dimension"/"Y dimension" versus "(x)"/"(y)".
- "prime instrument" versus "primary instrument".
- "F Ring" capitalized outside titles at `03:21`, `06:6`.
- Used before or without definition: CDAP, OBSID, IMGID, LIDVID, LID, CIRS,
  UVIS, ET, TDB, UTC, SCLK.

### Verified sound in the editorial pass
aspell over all sections flags only proper names and identifiers;
"background-subtracted" and "corotating" consistent throughout; every
`\zcite` key has a reference entry and every entry is cited; all `\ref`
targets resolve; the numbers that appear in more than one place (305, 42,
18,000 × 401, 0.02°, 5 km, ±1000 km, row 200, 139,892–140,551 km, the epoch
ET, the corotation rate, browse sizes, the three-file counts, the background
limit semantics) agree with each other; the 13 function names match
`mosaic_utils.py`; README figure count (15) and `sty/` list match the
directory; the two DOIs are distinct and used consistently.
