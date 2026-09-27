# PDS4 bundle critique, 2026-09-10

Bundle reviewed: `/data/fring-bundles/pds4/`, the full regeneration of
2026-09-09 (data collections 14:21 to 16:41 local, support collections 23:41
UTC) from `main` at `79b4f03`. The user guide in the bundle
(`document/user_guide/f-ring-mosaics-user-guide.pdf`, 45 pages) is
byte-identical to `user_guide/main.pdf`, so the guide was reviewed from the
LaTeX sources under `user_guide/sections/` and the PDF together.

This review was done with fresh eyes: none of the earlier critiques was read
until section 6, which compares this report with the 2026-09-03 one.

## 1. How the review was done

Six parallel reviewers, each with a written brief, each restricted to
read-only access, each required to give path, count, evidence and a
VERIFIED/SUSPECTED status for every finding. Every finding rated MAJOR or
above was then re-checked by hand before being written here.

| Pass | Coverage |
|---|---|
| Labels, schemas, inventories, checksums | All 42,405 labels parsed and XSD-validated with lxml against the five local dictionaries (PDS 1O00, DISP 1O00_1510, GEOM 1O00_19B0, RINGS 1O00_1F00, CASSINI 1O00_1800). All 11 inventories reconciled. All 148,380 file references checked for existence and size; MD5 checked for all 127,185 non-image files, all 610 mosaic arrays and 300 random reprojected arrays. All 296,883 internal references resolved. All 42,391 tables checked structurally. |
| Mosaic products | All 610 mosaic and background-subtracted arrays read in full; every label value, table value and index row recomputed. |
| Reprojected-image products | All 20,584 labels, tables, supplemental files, browse labels and index rows checked in full (60.9 million table rows). Pixel content read for 3,588 arrays (every observation, all 1,314 wraparound images, all 163 occultation images, all moon candidates). 150 images cross-checked against their PDS3 calibrated labels. 40 images checked with SPICE for boresight direction. |
| Support collections | Every file outside the per-observation collections read end to end; 26 external identifiers resolved against the PDS registry; all 1,153 kernel names checked against NAIF directory listings; the five example programs run against the bundle from scratch copies. |
| User guide, technical | Every count, path, excerpt, formula, field table and figure caption compared with the bundle. |
| User guide, editorial | Full proofread of the sources and the rendered PDF. |

Schematron rules could not be evaluated here (no Java), so the PDS4
`validate` run remains the one check this review cannot substitute for.

## 2. Headline

The archive is structurally clean. Every label validates, every inventory
reconciles, every file reference matches on size and checksum, every
cross-reference resolves, and the orbit, timing and geometry numbers reproduce
from the constants the labels state. No BLOCKER was found.

Three items are rated MAJOR. One is in the data products: the printed
boresight roll in 161 supplemental files uses a different reference axis from
the other 20,423, and neither convention is stated anywhere. Two are in the
user guide: section 7.3 describes the mosaic index longitude fields as the
edges of the mosaic when the index actually holds the valid-data range, and
section 3.6 gives three mutually inconsistent statements of the
background-fit row range.

Everything else is wording, documentation gaps, or presentation.

## 3. Findings in the bundle

### B1. MAJOR. Boresight roll uses two conventions across the supplemental files

- Where: `data_reproj_img/<obs>/<img>_reproj_img_suppl.txt`, line
  `Navigated Boresight Roll`. 161 of 20,584 files, all in three
  observations: `iss_096rf_fmovie004_prime` (76), `iss_093rf_fmovie001_prime`
  (71), `iss_180rf_hiresfrng001_prime` (14). These are exactly the images
  whose boresight declination has |Dec| above about 64.3°.
- Evidence: in 20,423 files the printed roll equals
  `atan2(C[1,3], C[2,3])`, the J2000 z-components of the camera X and Y rows
  of the printed matrix. In the 161 files it equals `atan2(C[1,2], C[2,2])`,
  the y-components. The two differ by 69° to 180°. Re-checked by hand:

  | File | Printed roll | z-convention | y-convention |
  |---|---|---|---|
  | `iss_093rf_fmovie001_prime/1605530283n` (Dec −78.75°) | −77.1551 | 78.0451 | −77.1551 |
  | `iss_180rf_hiresfrng001_prime/1738425645n` (Dec −65.49°) | −142.1960 | −72.8858 | −142.1960 |
  | `iss_111rf_fmovie002_prime/1622049830n` (Dec −11.80°) | −44.9209 | −44.9209 | −126.0980 |
  | `iss_000ri_satsrchap001_prime/1466451581n` (Dec +9.66°) | −90.0468 | −90.0468 | 7.4721 |

  The C-matrices themselves are sound in all 20,584 files: orthonormal to
  1.7e-10, determinant +1, and the RA/Dec recomputed from the third row
  agrees with the printed RA/Dec to 0.0005° in every file. A SPICE check of
  40 images confirms the third row points at the F ring in every case.
- Why it is a problem: the same named quantity is defined two ways in the
  archive, and the label describes the file only as "C-matrix pointing
  information and relevant parameters" with a pointer to the NAIF C-kernel
  documentation. A reader using the printed roll for a high-declination
  image gets a number that is not comparable with the rest of the bundle.
  Either print the roll with one convention throughout, or state in the
  label which axis the roll is measured against and when it switches. The
  same label sentence should also say that the matrix rows are the camera
  axes expressed in J2000 and that the boresight is the third row, since
  that is what the files contain.
- Status: VERIFIED.

### B2. MINOR. Wraparound products: table rows and browse columns run from 0°, the array runs from the minimum longitude, and no label says so

- Where: the 1,314 reprojected images whose minimum co-rotating longitude
  exceeds the maximum. `data_reproj_img/<obs>/<img>_reproj_img_metadata_params.tab`
  and `browse_reproj_img/<obs>/<img>_browse_reproj_img_*.png`.
- Evidence: the array description in every label says Sample 0 is the
  minimum longitude and the longitude "wraps through 360 degrees when the
  minimum is greater than the maximum", and the arrays do exactly that
  (checked on all 1,314: the data continue contiguously across the boundary).
  The params table, however, is sorted by longitude from 0.00 in all 20,584
  files, so for a wraparound image row k is not Sample k. The browse image
  follows the table order: for `iosic_276rb_complitb3001_si/1874525875w`
  (338.04° to 77.60°, 4,979 samples), the correlation between the browse
  column means and the array column means is −0.31 in array order and +0.76
  in longitude-sorted order. The user guide (section 4.2.1) documents the
  table ordering; nothing documents the browse ordering, and the label's
  `Table_Character` description does not mention either.
- Why it is a problem: a reader who looks at the browse image to orient
  themselves in the array sees the two halves of every wraparound product
  swapped. One sentence in the table description and one in the browse
  description would remove the ambiguity.
- Status: VERIFIED.

### B3. MINOR. Small and thumb browse images do not carry the name the labels say they carry

- Where: every browse label (`browse_mosaic`, `browse_mosaic_bkg_sub`,
  `browse_reproj_img`) says "The med, small and thumb images carry the
  observation name [and image name] drawn in the upper left corner". The
  user guide repeats it (04:235, 04:313).
- Evidence, from the rendered PNGs of `iss_111rf_fmovie002_prime`: the
  mosaic small and thumb images read "111rf_fmovie002 / mosaic" (the `iss_`
  prefix and `_prime` suffix dropped); the reprojected small image reads
  "1622049830n / reproj img" with no observation name at all; a split
  observation's small image reads "112rf_fmovie002/1". Only the med images
  carry the full text.
- Why it is a problem: the label statement is inaccurate for two of the
  three sizes. Either describe the abbreviation or drop the claim for small
  and thumb.
- Status: VERIFIED.

### B4. MINOR. The co-rotating span in every comment is 0.02° larger than its own end points imply

- Where: `Observation_Area/comment` in all 20,584 reprojected-image labels
  and all 610 mosaic labels.
- Evidence: "covering 99.560 degrees of inertial longitude from 14.095 to
  113.655" is the difference of the ends, but "spanning the (possibly
  discontinuous) 99.58 degrees from 338.04 to 77.60" is samples × 0.02
  (4,979 × 0.02), while (77.60 − 338.04) mod 360 = 99.56. Every label
  follows this pattern.
- Why it is a problem: the two spans in one paragraph use different
  conventions and the reader who checks the arithmetic finds a 0.02°
  discrepancy with no explanation. Saying "4,979 samples of 0.02 degrees"
  or "from 338.04 to 77.60 inclusive" would settle it.
- Status: VERIFIED.

### B5. MINOR. "Within the valid data range" is looser than it reads

- Where: the satellite sentence "This reprojected F-ring image includes
  Prometheus|Pandora within the valid data range, although its presence has
  not been visually confirmed" (772 Prometheus and 74 Pandora listings) and
  its mosaic counterpart.
- Evidence: recomputed from each product's own params table and array, 47
  listed moons lie 1,000 to 1,050 km from the core, outside the −1000 to
  +1000 km array (the 50 km tolerance adopted for the visibility test), and
  37 lie on a longitude column whose pixel at the moon's radius holds no
  data. Three moons that do sit on valid pixels (`iss_000ri_satsrchap001_prime/1466466941n`,
  `iss_093rf_fmovie003_prime/1605385758n`, `iss_111rf_fmovie002_prime/1622035856n`,
  all within two columns of the array edge) are not listed.
- Why it is a problem: the tolerance and the column-level test are
  reasonable design choices, but the sentence claims more than the test
  checks. "Within 1050 km of the core at a longitude covered by this image"
  is what the data support.
- Status: VERIFIED.

### B6. MINOR. Metakernel label declares its time range inapplicable

- Where: `spice_kernels/kernels.lblx` lines 37 to 40: `start_date_time` and
  `stop_date_time` carry `xsi:nil="true" nilReason="inapplicable"`;
  `collection_spice_kernels.lblx` has a `Context_Area` with no
  `Time_Coordinates` at all.
- Evidence: the listed CKs run from `04171_04179rb.bc` to `17257_17262ra.bc`
  and the SPKs from 2004-036 to 2017-258; every other label with a
  `Context_Area` gives the bundle span 2004-06-20T19:15:31Z to
  2017-09-07T21:51:58Z.
- Why it is a problem: a time range is applicable, and NAIF's own PDS4
  metakernel labels carry real start and stop times.
- Status: VERIFIED.

### B7. MINOR. Publication year 2025 against a publication date of 2026-09-09

- Where: all 42,405 labels say `publication_year` 2025 (pinned by
  decision). `document/user_guide/f-ring-mosaics-user-guide.lblx` also says
  `Document/publication_date` 2026-09-09, and every `Modification_History`
  says 2026-09-09. `readme.txt` and the guide's title page cite "(2025)".
- Evidence: values as stated. Neither DOI resolves yet, so the year on the
  DOI record cannot be checked.
- Why it is a problem: one label says the document was published in 2025
  and on 2026-09-09. Whichever year the DOI record carries, the two fields
  should agree.
- Status: VERIFIED.

### B8. MINOR. Reprojected-image index copies the mosaic-only longitude caveat

- Where: `miscellaneous/global_reproj_img_index.lblx`, fields 10 and 11.
- Evidence: both descriptions say "a mosaic label reports the full grid, 0
  to 359.98 degrees, because a mosaic is always written out to its full
  extent". For reprojected images the index value equals the label value
  in every row checked (25 random rows, plus the full-pass agreement in the
  reprojected-image review).
- Why it is a problem: the sentence describes a discrepancy that does not
  exist in this table and refers to "a mosaic label" in a table of
  reprojected images.
- Status: VERIFIED.

### B9. MINOR. Wording defects in label prose

All VERIFIED.

- `spice_kernels/kernels.lblx` line 116 expands NAIF as "Navigation and
  Ancillary Information Node of NASA PDS". It is the Navigation and
  Ancillary Information Facility.
- `rings:description` in all 21,194 data labels: "-1000km and +1000km"
  (no space before the unit).
- Field description of `rings:observed_event_tdb` in all 20,584
  reprojected-image labels: "at middle of the exposure".
- Supplemental `Table_Character` description: "containing C-matrix pointing
  information and relevant parameters". The three-record table holds the
  matrix only; the parameters are in the header.
- Background-subtracted mosaic comment: "If insufficient data was
  available" (the rest of the bundle treats data as plural).
- `bundle.lblx` and the three data collection labels call the referenced
  ISS Data User's Guide "(PDS3)". Its LID is a PDS4 document product; the
  data it describes are PDS3.
- Target and title phrasing drift across the 17 support labels: "F ring"
  (11 uses), "F Ring" (30), "F-ring" (15, only in the three index labels);
  "Reprojected Versions of Cassini ISS Calibrated Images" in the support
  collection titles against "Calibrated Cassini ISS Images" in the bundle
  and data collection titles.
- `document/user_guide/mosaic_utils.py` lines 352 to 354 and 368 to 370
  describe the reprojected-image LID as
  `...:data_reproj_img:<OBSID>_reproj_img`; the field is the image name,
  e.g. `1874525875w_reproj_img`. The code is right; the comment is wrong.
- `readme.txt` line 20 is 86 characters (the document LID); every other
  line is 80 or fewer. It cannot wrap, so either accept it or restructure
  the sentence.
- `readme.txt` and the guide title page give the recommended citation with
  different punctuation ("French, R.S. and Hedman, M.M. (2025). ... DOI
  10.17189/3tfh-th07." against "French, R. S., & Hedman, M. M. (2025). ...
  https://doi.org/10.17189/3tfh-th07").

### B10. NOTE. Observations the reviewers recorded without asking for action

- The bundle description (three places in `bundle.lblx`) names mosaics,
  reprojected images, metadata, documentation and index files; it omits
  the background-subtracted mosaics and the browse products that
  `readme.txt` does mention.
- The three global index labels carry no `Reference_List`, although every
  `notes` field says "See the User Guide for details"; every other product
  label references the guide.
- Context products appear as secondary members three times: as bare LIDs in
  the context inventory and as pinned LIDVIDs in the document and
  miscellaneous inventories. Nothing in the miscellaneous collection
  references the products its S rows name. The document-collection rows
  were added at the PDS reviewer's suggestion (see `pds4_bundle_gen/CHANGELOG.txt`),
  so this is recorded, not raised.
- Mosaic labels do not say that non-contributing reprojected images exist.
  143 archived images in four observations are listed in no `src_imgs`
  table; their own labels explain it, the mosaic labels do not.
- "Valid longitude" means any non-sentinel pixel in the column: 19,608
  valid columns in 251 mosaics have no data at the core line and 27
  mosaics contain a column with exactly one valid pixel. The
  background-subtracted products remove almost all of these. The
  definition is not stated in the labels.
- Prometheus and Pandora radii in the `iosic_276rb_complitb3001_si` mosaic
  table differ from the same images' own tables by 0.001 km (12 values).
  Every other per-image constant is byte-identical across all 40,780
  image/product pairs.
- `readme.txt` and the six `Document_File` entries carry no `file_size`,
  and `readme.txt` no MD5. Both are optional.

## 4. Findings in the user guide

### G1. MAJOR. Section 7.3 misdescribes the mosaic index longitude fields

- Where: `sections/07-metadata-and-global-index-file-fields.tex:177,179`.
- Evidence: the guide says the minimum co-rotating longitude "represent[s]
  the left edge of the mosaic" and the maximum "the right edge". In
  `global_mosaic_index.tab` only 54 of 305 rows are 0.00/359.98; 110 rows
  have minimum greater than maximum; `iss_111rf_fmovie002_prime` reads
  186.60 / 26.44 while its product label reads 0.00 / 359.98. The index
  label's own field description says the index holds "the range of
  co-rotating longitude that contains valid data" and warns that it differs
  from the product label. Section 4.2.3 of the guide itself says a mosaic
  always spans the full grid.
- Why it is a problem: a reader following section 7.3 expects the mosaic
  edges and either misreads the valid-data range or concludes that mosaics
  are stored wrapped. The reprojected-image table (07:81, 07:83) is
  correct; the mosaic table needs the same "range containing valid data"
  wording plus the note that it differs from the product label.
- Status: VERIFIED.

### G2. MAJOR. Section 3.6 gives three inconsistent background-row ranges

- Where: `sections/03-image-selection-and-processing.tex:405`;
  `sections/07-metadata-and-global-index-file-fields.tex:275-277`.
- Evidence: one paragraph says "the innermost 50 rows (Δr = −1000 to −755
  km) and the outermost 51 rows (Δr = +750 to +1000 km)", then "the
  standard range of 50 pixels on each side", then "the interior background
  region runs from −1000 km out to `bkgnd_lower_limit` (normally −750 km)".
  −1000 to −750 inclusive is 51 rows; −1000 to −755 is 50 rows. The 256
  default-margin background-subtracted labels say "from 750 to 1000 km".
- Why it is a problem: a reader reproducing the background fit cannot
  tell whether the −750 km row is in the fit. The bundle cannot settle
  which count is right; the author can.
- Status: VERIFIED (inconsistency); which statement is true is SUSPECTED.

### G3. MINOR. Claims that disagree with the bundle

All VERIFIED unless marked.

- 03:287, 03:405, 03:411: `nav_quality`, `bkgnd_quality`,
  `bkgnd_lower_limit` and `bkgnd_upper_limit` are described as fields "of
  the mosaic label". No label has elements of those names. The grades and
  margins appear only as prose in the `Observation_Area/comment`; the coded
  fields exist only in the index files.
- 04:177-181 and 04:277-281: the two params-table excerpts differ from the
  shipped files in the Prometheus and Pandora columns by 0.001 to 0.003 (for
  example 274.743 → 274.741, 68.058 → 68.056, 142262.214 → 142262.216).
  Every other column matches. The mosaic excerpt starts at row 4.60
  (image_index 8) although section 4.2 says every excerpt comes from image
  `1622049830n`. 04:127: the excerpt's `creation_date_time` is
  2026-09-08T19:31:25Z; the bundle's is 2026-09-09T22:09:20Z. The
  timestamp staleness was accepted on 2026-09-08; the moon columns changed
  afterwards with the kernel substitution and were not re-captured.
- 04:235, 04:313: the small and thumb overlay claim (see B3).
- 04:365-378: the observation-class notes are called "XML comments" and
  the text implies the index carries the substituted sentences. They are
  prose in the PDS4 `<comment>` element; the index `notes` column holds
  codes only.
- 07:115, 07:117, 07:211, 07:213: `rings:minimum_ring_radius` and
  `maximum_ring_radius` are described as radii "in the original image".
  In every row of both indexes they equal the core radius minus and plus
  1000 km, the grid edges, which is what the product labels say.
- 03:41: the PINST rule ("if PINST is not PRIME, the ISS images were a
  ride-along" with PINST the prime instrument) fails for
  `IOSIC_276RB_COMPLITB3001_SI`, `IOSIC_276RB_COMPLITB4001_SI` (SI) and
  `ISS_253RI_HIRESAFRG001_PIE` (PIE). PINST values in the bundle: PRIME
  237, VIMS 47, CIRS 17, SI 2, UVIS 1, PIE 1.
- 03:221: "the bundle and collection labels list the full set of stars"
  holds for the bundle label and the three data collection labels only;
  the browse and support collection labels list no targets and the SPICE
  collection lists the rings only.
- 03:270: Figure 5 caption "93°–309°"; the index gives 93.030 to 309.750.
- The guide never explains the labels' "visually confirmed" statement
  (88 mosaic labels confirmed, 35 not); the reader is given no definition
  of what was confirmed or how.
- Figures 1 to 6 are renderings from an earlier build (no overlay,
  different stretch; Figure 5 differs from the current browse image by a
  mean of 43 of 255 levels because the current one shows the black
  silhouette region). The captions do not claim they are browse products,
  so nothing is wrong, but readers comparing with the bundle will notice.
  The three screenshots in section 5.3.4 are identical to the plots the
  shipped scripts produce on the current bundle.

### G4. MINOR. Editorial and presentation

The full list, including a 24-row table of trivial typos and style
inconsistencies, is in the editorial reviewer's report. The items that
affect a reader:

- 04:185 cites section 5 for the wraparound modulo formula; section 5 does
  not contain it (3.4.2 and the example programs do).
- LID, LIDVID, OBSID, IMGID, I/F and COISS are used in the quick-start
  section before they are defined (definitions arrive on pages 16 and 23);
  LID and LIDVID are never defined.
- Figure 1 lands four pages after its reference, inside section 3.1.5,
  splitting a sentence about occultations; Figures 2 to 5 land three to
  five pages after their references and leave two pages nearly blank.
- Verbatim blocks wrap inside tokens and mark the break with an
  unexplained "⌋" glyph (`rings:radial_ ⌋ / resolution`,
  `iss_036r ⌋ / f_fmovie001_vims_mosaic_bkg_sub.lblx`, pages 25, 27, 34).
- The table-file excerpts on pages 25 and 28 wrap each row onto two lines,
  so the fixed-width layout they are meant to show is not visible.
- The two quick-start tables break collection names inside the name
  (`browse_reproj_` / `img`), split across pages with no repeated header,
  and have no visually distinct header row. No table in the guide is
  captioned or numbered; the section 7 tables have no header row; the
  three screenshots in 5.3.4 are uncaptioned.
- Page footers on the landscape pages 38 to 45 are rotated along the left
  edge.
- The M1 to M4 definitions (3.1.2.1 to 3.1.2.4) are in neither the table of
  contents nor the PDF bookmarks.
- Directory names are typeset three ways in headings (italic, plain,
  Courier).
- The suffix-range notation `_2–9` (03:139-141) is never defined and
  renders with a dash indistinguishable from a hyphen.
- 04:235 says a full browse image "is stretched horizontally" and then
  "Every size other than Full is resampled"; stretching is resampling.
- "observation" means a campaign in section 3 and a single image in
  section 7 (07:21, 07:29, 07:37, 07:39, 07:45, 07:49).
- The title page has no date; the citation year 2025 sits beside a 2026
  label excerpt.
- Grammar: 03:71 "The resulting image sequence and mosaic therefore do
  not ... Instead, it provides"; 05:50 "while also permitting the wide
  range of software" (no object); 01:14 "Of these" has the mosaics as its
  antecedent; 05:116 "at that time" dangles.
- 04:31 says mosaics are "stitched together from multiple reprojected
  images"; 03:56 and 03:378 say one or more, and single-image mosaics
  exist.
- Array dimensions are written 401×N, 401×18,000, 18,000 × 401 and
  18,000×401 in different places.
- Two of fifteen reference-list entries carry URLs, the rest none; journal
  names mix abbreviated and full forms.
- One underfull line (`main.log:975`); no overfull boxes; no undefined or
  duplicated references; all 102 cross-references resolve; the TOC and 52
  bookmarks match the headings.

## 5. Verified sound

The reviewers' positive results, kept because they are what a release
decision rests on.

- Well-formedness and XSD validity: 42,405 of 42,405 labels, zero errors.
  Every label references the same five dictionary versions; the
  `xml_schema` inventory carries the five LIDVIDs the published dictionary
  labels declare, character for character.
- Identifiers: 42,405 distinct LIDVIDs, no duplicates; every product LID
  matches its directory and file base name; every reprojected-image name
  agrees with its instrument LID (17,830 NAC, 2,754 WAC); no observation
  mixes cameras.
- Inventories: primary rows equal product labels one to one in all nine
  product collections (305, 305, 20,584, 305, 305, 20,584, 1, 3, 1);
  record counts, sizes and MD5s match in all 11; every secondary row is
  referenced by at least one label; the eight pinned context LIDVIDs are
  the versions the registry reports as current.
- Files: 148,380 of 148,380 referenced files exist with the stated size;
  MD5 matches for all 127,185 non-array files and all 910 arrays checked;
  every `creation_date_time` equals the file's UTC modification time to
  the second; every non-label file is referenced by exactly one label; no
  orphans.
- Cross-references: 296,883 of 296,883 internal references resolve. Every
  `src_imgs` LIDVID resolves inside its own observation; every reprojected
  label points at its own mosaic, background-subtracted mosaic and browse
  product, and they point back; every browse label points at a data
  product of the same base name. Global indexes: every row's LID, path and
  creation time match the product.
- Tables: 42,391 of 42,391 pass every structural check (header length
  equals offset, record count and length, field positions, data types,
  no overflow, no CR, no non-ASCII). No longitude prints 360.00; the only
  360.000 values are radial resolutions in km/pixel passing through 360.
- Template artefacts: no `$`, `{`, `}`, `None`, `nan`, `inf`, quoted
  Python strings, `TODO`, curator comments or double spaces in any label.
  `image_observation_type` is a plain value in every label (SCIENCE
  19,853 + 9 SUPPORT + 51 labels with both as separate elements, matching
  the PDS3 source in 674 of 674 checked).
- Mosaic arrays: 610 of 610 are 401 × 18000 little-endian float32 with
  −999 as the only sentinel, no NaN or infinity, MD5 matching. The valid
  column count, the coverage sentence, the circular min/max, the index row
  and the label statistics agree in 610 of 610. Per-image constants in the
  mosaic tables are byte-identical to the images' own tables in 40,780
  of 40,780 pairs (with the 0.001 km exception noted). Time coordinates
  equal the earliest start and latest stop of the listed images in 610 of
  610; SCLK counts likewise. True anomaly, core radius, inertial
  longitude, node and pericenter reproduce from the stated constants to
  0.001° and 0.004 km.
- Background-subtracted mosaics: valid columns are a subset of the
  mosaic's in 305 of 305; the seven products that lose whole images list
  only the contributing ones; label values are recomputed from their own
  reduced tables in 305 of 305; the sentinel comment differs from the
  mosaic's as it should.
- Satellite targets: the listing decision reproduces from the params
  tables under the same-image rule (1,050 km radial, longitude among the
  image's valid columns) in 1,219 of 1,220 mosaic product-moon cases, and
  the remaining case is the exclusion logged in `WARNINGS.log`. The 82
  warnings contradict no label. Every listed moon has its sentence and
  vice versa (846 of 846 reprojected, 1,220 of 1,220 mosaic).
- Stars: exactly the 163 images and 30 mosaic products of the 15
  occultation mosaics carry a star target; the nine name/alias/LID triples
  match the registered context products (L2 Puppis pending registration);
  every star in the context inventory is used.
- Reprojected arrays: size equals 401 × samples × 4 and samples equals
  round(((max − min) mod 360)/0.02) + 1 in 20,584 of 20,584; all 1,314
  wraparound arrays continue contiguously across 360°; first and last
  columns are the min and max longitudes in all 3,588 arrays read; the
  browse stretch recipe reproduces the full PNG in 1,288 of 1,288 tested.
- Reprojected tables and labels: 60.9 million rows parse; per-image
  constants constant; inertial longitude, true anomaly, core radius, node
  and pericenter reproduce to 0.0006°, 0.001°, 0.0034 km, 0.0005°;
  `observed_event_tdb` equals the TDB of the mid-time to 2 ms; label
  statistics recompute from the table in 20,584 of 20,584; start and stop
  times floor and ceil the exposure in every label; the image name equals
  the integer part of the stop SCLK count in every label.
- Supplemental files: 20,584 parsed; every C-matrix orthonormal with
  determinant +1; RA/Dec from row 3 within 0.0005° of the printed values;
  the SCLK and UTC lines equal the label's; 40 images checked with SPICE
  point at the F ring within the camera field.
- Cassini fields: all 57 mapped attributes agree with the PDS3 calibrated
  labels for 150 of 150 sampled images; mission phase agrees with the
  date boundaries in every label; full-well values follow the gain mode
  (4095 for 20,529, 9896 for the 55 gain-12 images).
- Browse: 84,776 PNGs are 8-bit greyscale of the documented sizes
  (mosaic 18000×401, 1800×400, 200×200, 100×100; reprojected max(V, 800)
  × 401, max(V/10, 400) × 400, 200×200, 100×100) in every product; no-data
  columns are dropped as stated; no-data pixels are black; the display
  direction is bottom to top as declared.
- Support collections: bundle time span equals the extreme product times;
  the 13 bundle targets are exactly the union of product targets;
  `Bundle_Member_Entry` matches the 11 collections; `reference_type`
  values match the Schematron lists; author and contributor blocks are
  byte-identical across the 16 labels that carry them; `kernels.ker`
  parses under cspyce, has no line over 36 characters, no duplicates, and
  every one of its 1,153 names is a real NAIF file; the CK and SPK coverage
  brackets the data span and no image falls in a coverage gap; the
  document label's six files, standards and MD5s are right; all five
  example programs run to completion on the bundle and reproduce the
  guide's screenshots.
- User guide, technical: every count, directory tree, file name, LID,
  DOI, orbit constant, class list, star assignment, field name, unit and
  field order agrees with the bundle except as listed in section 4; the
  column-mapping formulas hold for all 20,584 labels and for 50 arrays
  read; the R-class and O-class behaviour is as described (20,441
  contributing images, 143 non-contributing in the four R observations).

## 6. Comparison with the 2026-09-03 critique

Written after everything above.

Fixed since 2026-09-03 (all 28 bundle findings and all 15 guide findings
were reported closed in PR #8; this review confirms the closures it can
see): stringified `image_observation_type`; manual-offset misattribution
(the supplemental headers now carry the navigation type used); stellar
aberration folded into the C-matrices (the SPICE check of 40 images finds
the third row pointing at the ring with no aberration offset visible at
the 0.02° level); phantom source images in seven background-subtracted
mosaics (the source lists now equal the contributing images); start and
stop times rounding into the exposure (now floor and ceil in every label);
the satellite test tolerance (the same-image 1,050 km rule reproduces every
decision); the xml_schema LIDVIDs; the stale PDF; the mission-phase
vocabulary; L2 Puppis; the array orientation description; the four global
index label defects; the publication year.

Still open, restated: the roll-axis discrepancy. The 2026-09-03 review (its
B3) found one matrix in `iss_180rf_hiresfrng001_prime` twisted by the
reference-axis switch at |z| = 0.9 and noted in the same finding that the
"Navigated Boresight Roll" header jumps by about 76° whenever a boresight
crosses that threshold. The matrix is now right in every file, but the
printed roll in the 161 high-declination files is still computed against
the other axis (B1 above). The fix corrected the matrix and left the
derived number.

Partly fixed: the 2026-09-03 B12 (co-rotating span 0.02° short of the
sample count) is closed; both co-rotating numbers now agree. The inertial
sentence in the same paragraph still uses the end-point difference, which
is B4 above. The 2026-09-03 B20 asked the browse labels to mention the
burned-in title; they now do, but the wording overstates what the small
and thumb sizes carry, which is B3 above.

New in this review: B2 (browse and table order for wraparound products),
B5 (the "within the valid data range" wording), B6 (nil metakernel time
range), B7 (publication year against publication date), B8 (index caveat
copied into the wrong index), G1 (section 7.3 mosaic index fields), G2
(section 3.6 row ranges), and the G3 and G4 lists. G3's excerpt drift in
the moon columns is a consequence of the kernel substitution made after
the excerpts were re-captured on 2026-09-08; the 2026-09-03 G3 (all
excerpts stale, two contradicting the guide's own formulas) is otherwise
closed, as are G1 (counts), G2 (L2 Puppis), G5 to G9 (browse rules,
blackpoint, kernels, captions, screenshot) and G10 (title).

Raised by the reviewers but already decided by the data provider, so not
counted as findings here: the document label's two author lists (Citation
French and Hedman, Document French alone) are the house rule; the mosaic
incidence angle being the last image's value under a "mean" description is
the accepted single-value simplification (maximum discrepancy 0.016°); the
absence of a header block in `kernels.ker` is by ruling; the per-image
"not visually confirmed" wording is correct; the 1,2,2,2 contributor
sequence numbers are deliberate.

## 7. What remains before release

1. Decide B1: one roll convention throughout, or a stated convention.
   This is the only finding that changes a number a reader might use.
2. Fix G1 and G2 in the guide, rebuild the PDF, and re-copy it into the
   bundle's document collection (which regenerates the document label's
   checksum and publication date).
3. Settle B7 (publication year against publication date) with the DOI
   record.
4. The rest of sections 3 and 4 are wording changes that can ride along
   with any regeneration.
5. Run PDS4 `validate` with Schematron on the final tree. The four expected
   referential failures are unchanged: the ISS user guide `::2.0`, the
   `data_calibrated` source products, `star.l02_pup` until the Engineering
   Node registers it, and the five IM-1.24 schema products until the
   registry ingests Build 15.1.
