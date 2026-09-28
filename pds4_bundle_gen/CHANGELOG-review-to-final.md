# Change log for `urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025`

## How the delivered bundle differs from the peer-review copy

Peer-review copy: generated 2025-09-22. Delivered bundle: generated
2026-09-27. Bundle LID `urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025`,
version 1.0, information model 1.24.0.0 in both.

This document lists every difference a reader of the two bundles would see,
grouped by part of the bundle, for the peer reviewer and the Ring-Moon
Systems Node archivist. Every count comes from a comparison run over the
complete contents of both bundles. Three appendices list affected
observations and columns by name.

Identifiers, versions and authorship are unchanged: the bundle LID, the
collection LIDs, every product LID present in both copies, all version
identifiers, the bundle DOI 10.17189/3tfh-th07, the bundle title, and the
author and contributor lists. Nothing here changes how the bundle is cited.

## Summary

- **Size.** 302 observations, 20,303 reprojected images, 188,194 files,
  55.8 GB became 305 observations, 20,584 reprojected images, 190,785
  files, 61.9 GB. Three observations were added, none removed; 671
  reprojected images were added and 390 removed.
- **Source-image attribution corrected.** In the review copy, 36 mosaics
  attributed 420,401 of their 2,851,152 valid longitudes (14.7 percent) to
  the wrong source image, and 108,065 of those longitudes came from images
  the review copy did not archive. In the delivered bundle every mosaic
  longitude is byte-identical to the column of the image its metadata
  names. The 390 removed images were never sources of any mosaic; 601 of
  the 671 added images are the sources that were missing (Appendix A).
- **New collection and files.** A `miscellaneous` collection holds the
  three global index tables, which were in `document/supplemental/`. A
  `readme.txt` was added at the root. Five example Python programs were
  added to the document collection.
- **Dictionary.** The Rings dictionary moved from 1O00_1E00 (1.14.0.0) to
  1O00_1F00 (1.15.0.0) in every data label.
- **Every data label changed.** Camera identification (every review-copy
  mosaic and reprojected image named the Wide Angle Camera; 231 mosaics and
  17,830 images are Narrow Angle), display direction, mission-phase
  capitalisation, longitude limits, four new Cassini attributes, array and
  table descriptions, navigation and background quality statements,
  Prometheus and Pandora statements, and nine stellar occultation targets.
- **Metadata tables rebuilt.** The per-longitude tables went from 8 (mosaic)
  and 6 (reprojected image) columns to 17 and 16, with dictionary-style
  names, nine new orbit and moon columns, an inertial-longitude column that
  was wrong in the review copy, a longitudinal-resolution column that was
  in the wrong unit, and exposure mid-times to the millisecond. The index
  tables went from 33, 35 and 30 columns to 57, 60 and 42.
- **Arrays.** 296 of 302 common mosaic arrays and 18,351 of 19,913 common
  reprojected-image arrays are byte-identical. 1,065 wrapped reprojected
  images have the same pixels in a different column order; 495 have the
  same pixels with empty longitudes inserted; 2 changed. All 302
  background-subtracted arrays changed: the subtracted background differs
  by a per-column linear function of typical size 2e-7 I/F (0.08 percent of
  the pixel value), and 263 products lost valid longitudes (81,125 in
  total, 2.9 percent, up to 38 percent in one product; section 7.2).
- **Pointing.** The C-matrix, right ascension, declination and roll changed
  in every common supplemental file; the boresight moved by a median of 6
  arcsec. The file was renamed and gained an explanatory preamble.
- **Browse images.** Same sizes, overlays and orientation; the grey-level
  mapping changed (darker backgrounds) in most images; background-subtracted
  browse images show the dropped longitudes as black.
- **User guide.** Rewritten from 19 pages, 5 sections and no bookmarks or
  links to 47 pages, 7 sections, 56 bookmarks and 251 links, now PDF/A-1b
  (the review copy was PDF/A-2b). Three sections and one subsection are new.
- **Formatting defects removed.** CRLF line endings in nine labels, tab
  characters inside `schemaLocation` in every product label, a missing XML
  declaration in the kernel label, malformed identifiers in the document
  and schema inventories, and an author-review comment inside the SPICE
  collection label are gone.

### Notes for the reviewer

Two statements in the delivered bundle are listed here as shipped and are
also the subject of open findings in the critique of 2026-09-27:

- The supplemental pointing files and the reprojected-image labels define
  the roll as measured from increasing declination, positive towards
  increasing right ascension. The printed roll is the angle measured from
  increasing right ascension, positive towards increasing declination. The
  matrices are correct; the sentence is not.
- The document and miscellaneous inventories list the Cassini ISS Data
  User's Guide as `::2.0`. The PDS registry holds `::1.1` (2023-07-20) as
  the current version, and `::2.0` does not resolve.

## 1. Bundle level

- `readme.txt` (1,771 bytes, 7-bit ASCII) added at the root and declared in
  `bundle.lblx` as `File_Area_Text`. It gives the bundle and user-guide
  LIDs, the citation with DOI 10.17189/3tfh-th07, and contact information
  for the Node and the author.
- `bundle.lblx`: `Bundle_Member_Entry` list 10 to 11 (adds `miscellaneous`,
  `bundle_has_miscellaneous_collection`). `Time_Coordinates` start
  2004-06-20T18:19:31Z to 2004-06-20T19:15:31Z (the review copy's earliest
  image was one of the misattributed ones); stop unchanged at
  2017-09-07T21:51:58Z. Nine `Star` targets added (R Lyrae, Mirach, Scheat,
  Algenib, L2 Puppis, Herschel's Garnet Star, R Cassiopeiae, R Hydrae,
  W Hydrae), each with a `bundle_to_target` reference; the two ring and two
  moon targets unchanged. The three descriptions now end "...associated
  metadata, documentation, and global index files. The images and mosaics
  cover the period from 2004-06-20T19:15:31Z to 2017-09-07T21:51:58Z."
- Reference comments reworded in every label that carries them: "Detailed
  User's Guide" to "Detailed User Guide"; the ISS Data User's Guide comment
  now reads "The Cassini ISS Data User's Guide, which describes the PDS3
  form of the source images; DOI: 10.17189/1504135".
- Hedman's ORCID changed from `http://orcid.org/...` to
  `https://orcid.org/...` in all 14 labels listing authors.
- `modification_date` 2025-09-22 became 2026-09-28 in the bundle,
  collection, document, index and kernel labels; "Initial version" became
  "Initial version." where the period was missing.
- Unchanged: LID, title, `version_id`, information model version,
  `publication_year` 2025, DOI, keywords, authors, the four DataCurator
  contributors, investigation, observing system.

## 2. Collections and inventories

- Collections 10 to 11: `miscellaneous` added (`collection_type`
  Miscellaneous; inventory of 3 primary members, the index products, and 9
  secondary members: the ISS Data User's Guide and the eight Cassini
  context products). `document/supplemental/` and its six files removed.
- Collection titles for context, document, spice_kernels and xml_schema:
  "... F Ring Mosaics and Associated Reprojected Versions of Cassini ISS
  Calibrated Images" to "... F Ring Mosaics and Associated Reprojected
  Images Created from Calibrated Cassini ISS Images" (a trailing period on
  the document title removed). Descriptions "reprojected versions of Cassini
  ISS calibrated images" to "reprojected versions of calibrated Cassini ISS
  images".
- Data collection labels (three): start time as in the bundle; the ISSNA
  and ISSWA components now listed NAC first; nine star targets added.
  Browse collection labels: only dates and inventory fields changed.
- Inventories: data and browse collections 302 to 305 (mosaic types) and
  20,303 to 20,584 (reprojected types) primary members, all `::1.0`.
  Context 8 to 17 secondary members (nine stars added). Document 5 to 10
  rows: the three index entries (written `...:document:global_mosaic_index:1.0`
  with a single colon, and not matching the index labels' own LIDs)
  removed; the ISS Data User's Guide entry changed from `::1.0` to `::2.0`;
  eight Cassini context products added as secondary members. xml_schema:
  the five entries rewritten from `pds-xml_schema::1.24`,
  `disp-xml_schema::1.15`, `geom-xml_schema::1.19`, `rings-xml_schema::1.14`,
  `cassini-xml_schema::1.18` to `pds-xml_schema_1.24.0.0::1.0`,
  `disp-xml_schema_1.24.0.0_1.5.1.0::1.0`, `geom-xml_schema_1.24.0.0_1.9.11.0::1.0`,
  `rings-xml_schema_1.24.0.0_1.15.0::1.0`, `cassini-xml_schema_1.24.0.0_1.8.0.0::1.0`,
  which are the identifiers the dictionary products carry at PDS.
  spice_kernels inventory unchanged.
- The `spice_kernels` collection label lost a `Time_Coordinates` block
  (2004-01-01 to 2017-09-01) that carried an XML comment addressed to the
  archivist ("MJTM: please update to describe kernel temporal coverage..."),
  and its inventory creation time gained the missing `Z`.

## 3. Context products and targets

- Nine stellar occultation targets are named. Each of the 15 occultation
  mosaics (`O` note) names its star in `Target_Identification`, as do the
  15 background-subtracted counterparts and all 163 reprojected images of
  those observations (L2 Puppis 37, R Cassiopeiae 35, R Lyrae 21, R Hydrae
  18, Mirach 17, Scheat 9, Herschel's Garnet Star 9, W Hydrae 9, Algenib 8).
  The bundle label and the three data collection labels list all nine; the
  context inventory lists their LIDs. The review copy had no star targets.
  Appendix B lists the observations.
- Prometheus and Pandora targets are attached only where the moon's
  predicted position, at the time of the image that supplied the data,
  falls inside the product's valid longitudes within 1050 km of the core.
  Reprojected images: Prometheus 8,755 to 772 labels, Pandora 1,896 to 74.
  Mosaics: Prometheus 77 to 103, Pandora 14 to 22. Background-subtracted
  mosaics: 77 to 75 and 14 to 12 (Appendix B). Each such label carries a
  sentence in `rings:description` (mosaics) or the observation comment
  (images) giving the moon, the 1050 km criterion, and whether "its
  presence has been visually confirmed" (mosaics: 89 confirmed, 36 not;
  background-subtracted: 58 and 29) or "has not been visually confirmed"
  (all 846 images). The review copy's reprojected-image labels all ended
  with "Some reprojected F-ring images may include Prometheus and/or
  Pandora, but their presence has not been visually confirmed."
- The context inventory lists exactly the 17 context products referenced
  anywhere in the bundle; the versions cited in the document and
  miscellaneous inventories are the current ones at PDS (mission 1.5, host
  1.4, instruments 1.2, rings 1.1, moons 1.2).

## 4. Product set: observations and images

### 4.1 Observations

- Added: `iosic_276rb_complitb3001_si` (4 images, 2017-05-26, 47 percent
  coverage), `iss_134ri_spkmvdfhp002_prime` (55 images, 2010-07-01/02, full
  coverage), `iss_134ri_spkmvdfhp003_prime` (15 images, 2010-07-02/03, full
  coverage). None removed, none renamed.
- The review copy's `iosic_276rb_complitb4001_si` held 9 images and was
  labelled and indexed with observation ID IOSIC_276RB_COMPLITB3001_SI. Its
  first four images (1874525875w to 1874536491w) are now the new
  `iosic_276rb_complitb3001_si`; the remaining five stay under
  `iosic_276rb_complitb4001_si`, now with observation ID
  IOSIC_276RB_COMPLITB4001_SI, start time 2017-05-27T02:19:39Z and 9,332
  valid longitudes (was 16,532).
- The review copy's `iss_134ri_spkmvdfhp001_prime` was built from 15 images
  that belong to the third of the three SPKMVDFHP observations. The
  delivered bundle has the three observations with 65, 55 and 15 images.

### 4.2 Reprojected images

- 20,303 to 20,584: 19,913 common, 671 added, 390 removed, 4 moved between
  the two `iosic` directories.
- 601 images were added to 35 existing observations and 390 removed from 28
  of them (Appendix A). In every case the added images are later than all
  kept images. None of the 390 removed arrays matches any added array, by
  checksum or by column content: they are different images.
- The review copy's source attribution in 36 mosaics did not match the
  archived data. Comparing each mosaic column with the columns of the
  reprojected images archived for that observation: 2,430,751 of 2,851,152
  columns matched the image named in the metadata, 312,336 matched a
  different archived image, and 108,065 matched no archived image. Sixteen
  of the 36 mosaics matched their named images in no column at all (for
  example `iss_000ri_satsrchap001_prime`: 11,500 columns from other images,
  6,500 from images not archived; `iss_134ri_spkmvdfhp001_prime` and
  `iss_174ri_spokemov002_prime`: all 18,000 columns from images not
  archived). In the delivered bundle all 2,888,402 mosaic columns match the
  named image. Each named image can also be checked by testing whether its
  exposure window contains the mid-time recorded in the mosaic's metadata
  table; every named image in all 305 mosaics passes.
- The four "R" observations `iss_199rf_fmovie002_prime`,
  `iss_256ri_hiresafrg002_prime`, `iss_262rf_fmovie001_prime_12` and
  `iss_268rf_fmovie001_prime_1` now archive every image (23 to 120, 51 to
  67, 17 to 44 and 16 to 19). For the R observations, where Cassini follows
  one co-rotating longitude range across different inertial longitudes, and
  the O observations (stellar occultations), the reprojected images are the
  product of interest, so every image is archived whether or not it
  supplied data to the mosaic. 143 of the 20,584 images did not contribute
  to a mosaic (97, 16, 27 and 3 in those four observations); each appears
  in no source-image table and its own label says so. The other five R
  observations and all 15 O observations already archived every image.
- Net change, 20,303 to 20,584:

  | Reason | Images |
  |---|---|
  | The 2010 SPKMVDFHP observations archived in full as three observations | +120 |
  | R observations archiving every image | +143 |
  | Two observations gaining images that contribute to their mosaics: `iss_007ri_hpmrdfmov001_prime` (+14), `iss_007ri_lphrlfmov001_prime` (+4) | +18 |
  | The IOSIC_276RB separation: four images moved to a new observation | 0 |
  | The 28 observations with corrected source images: 390 replaced by 390 | 0 |
  | **Net** | **+281** |

## 5. Reprojected images

### 5.1 Labels

- Dictionary: Rings 1O00_1E00 to 1O00_1F00 in the `xml-model` and
  `schemaLocation`; PDS, DISP, GEOM, CASSINI unchanged.
- Camera: the review copy named "Cassini Orbiter Imaging Science Subsystem"
  with a single ISSWA component in all 20,303 labels. The delivered bundle
  names "Cassini Imaging Science Subsystem - Narrow Angle Camera" with an
  ISSNA component (17,830 labels) or "- Wide Angle Camera" with ISSWA
  (2,754), matching the image's `n`/`w` suffix.
- Citation description states whether the image was used: "This reprojected
  image was used to create mosaic X." (20,441) or "... is part of
  observation X but was not used to create mosaic x, which covers only part
  of the observation." (143).
- Observation comment: "this reprojected image is used in the mosaic named
  X" to "is associated with the mosaic named X"; "Albers et al. (2009), fit
  #2" to "Albers et al. (2012), Table 3, fit #2" (every label); the
  longitude sentence gives the true span ("valid data for a total of 30.00
  degrees of co-rotating longitude spanning the (possibly discontinuous)
  30.00 degrees from 356.46 to 26.44, measured to the outer edges of those
  two longitude bins") where the review copy said "360.00 degrees from 0.00
  to 359.98" for every wrapped image; the closing sentence about Prometheus
  and Pandora replaced by "The subjective quality of the navigation for all
  of the images for mosaic X is "good"" (good 14,845, fair 4,058, poor
  1,681) and, in 846 labels, the moon sentence of section 3.
- `rings:description`: "phase angle and observed_ring_elevation" to "phase
  angle and emission angle", with the added sentence "Because the incidence
  angle changes very slowly, the minimum and maximum incidence angle are set
  to the mean incidence angle"; "If the reprojection wraps around then they
  will be 0 and 359.98" to "then the minimum will be greater than the
  maximum". Resolution statistics reordered to mean, minimum, maximum.
- `rings:minimum/maximum_corotating_ring_longitude`: 1,253 common images
  changed from 0.00/359.98 (1,179) or another pair (74) to the true limits
  with minimum greater than maximum; 1,314 images in the delivered bundle
  wrap.
- `Time_Coordinates`: start earlier by 1 s in 9,674 labels and stop later by
  1 s in 8,721 (start is the floor of the exposure start and stop the
  ceiling of the exposure end; the review copy rounded both to the nearest
  second, so the stated interval contained the exposure for only a third of
  the images).
- Cassini attributes: four added to every label (`ground_software_version_id`,
  `pds3_target_desc`, `valid_maximum_full_well`, `valid_maximum_DN_sat`);
  `mission_phase_name` "SOLSTICE MISSION" to "Solstice Mission" (and Equinox
  Mission, Tour, Tour Pre-Huygens); `image_observation_type` values lost
  their quotation marks (`'SCIENCE'` to `SCIENCE`), and 65 labels carry two
  such elements (SCIENCE and SUPPORT) where the review copy had one quoted
  string; `missing_lines` `N/A` to `-1` with an explanatory XML comment in
  1,072 labels.
- Display: `disp:vertical_display_direction` "Top to Bottom" to "Bottom to
  Top". Array `local_identifier` `image` to `reproj_image`; new
  `local_identifier` values `metadata_params` and `supplemental_info` on the
  tables.
- `Array_2D_Image` gained a description: "Line is the radial axis and Sample
  is the co-rotating longitude axis. Line 0 is the innermost row, at a delta
  radius of -1000 km ... Sample 0 is the minimum co-rotating longitude given
  by rings:minimum_corotating_ring_longitude ... wrapping through 360
  degrees when the minimum is greater than the maximum."
- Supplemental file area: file renamed (5.4); the `File/comment` gained the
  description of the matrix rows and the roll (see Notes for the reviewer);
  header length grew by 490 bytes; table description "This is a
  supplemental table containing C-matrix pointing information and relevant
  parameters" to "The three rows of the C-matrix. The parameters describing
  the pointing are in the header above this table."
- Params table area: header length 109 to 367 bytes; fields 6 to 16; record
  length 58 to 159; a description of record order added (records ascend
  from 0 degrees; for a wrapped image this differs from the column order;
  use the longitude field to locate a column).
- NAIF external reference reworded ("the Navigation and Ancillary
  Information Facility (NAIF) of NASA PDS provides...").
- Unchanged: LID, title, keywords, `Source_Product_Internal` (the calibrated
  image LIDVID), epoch, corotation rate, grid sampling, the 56 pre-existing
  Cassini attributes and their order, the `geom:` block, reference LIDs.

### 5.2 Arrays

- 18,351 of 19,913 common arrays byte-identical.
- 1,065 wrapped images: identical pixels, same width, the two longitude
  blocks swapped. The review copy stored the block from 0 degrees to the
  maximum first and the block from the minimum to 359.98 second; the
  delivered bundle stores the minimum-to-359.98 block first, as its label
  says.
- 495 images: identical valid pixels, but wider (by 1 to 2,259 columns,
  median 1,829) because longitudes without data inside the image's range
  are present as all-missing columns, so that column position maps to
  longitude. Most are in the SPOKEMOV observations
  (`iss_173ri_spokemov002_prime` 139, `iss_173ri_spokemov003_prime` 99,
  `iss_174ri_spokemov001_prime` 85). The review copy's arrays had no
  all-missing column at all. 112 of the 495 also changed from a
  0.00/359.98 label to a wrapped one.
- 2 images changed pixel values: `iss_111rf_fmovie002_prime/1622023115n`
  and `1622023389n` (re-navigated; maximum difference 0.07 and 0.045 I/F;
  the second is one column wider).
- Valid values range -0.0695 to 1.341 I/F; no array contains NaN or
  infinity in either build.

### 5.3 Metadata tables (`_reproj_img_metadata_params.tab`)

- Header: "Corotating Longitude, Inertial Longitude, Radial Resolution,
  Angular Resolution, Phase Angle, Emission Angle" to the sixteen names of
  Appendix C. Ten columns added (exposure mid-time ET, incidence angle,
  core radius, node, pericenter, true anomaly, and the corotating longitude
  and radius of Prometheus and Pandora); none removed. Record length 58 to
  159 bytes; header separator ", " to ",".
- Inertial longitude: changed in 17,562,077 of 19,299,562 common rows (91
  percent). Within one table the review-copy value was offset from the
  delivered one by a constant equal to the table's first corotating
  longitude (the review copy tabulated the longitude relative to the start
  of the image); differences range up to 180 degrees. The delivered values
  are consistent with the mid-time and co-rotating longitude of their own
  row.
- Longitudinal resolution: every row changed by the factor pi/180; the
  review copy's label gave the column the unit km/pixel and its values were
  in radians per pixel (for example 0.00064); the delivered values are in
  degrees per pixel (0.03686) with the unit deg/pixel, agreeing with the
  resolution statistics in the labels.
- Corotating longitude, phase and emission angle: unchanged in every common
  row except the two images whose arrays changed.
- Record counts unchanged except `1622023389n` (1,476 to 1,477).

### 5.4 Supplemental pointing files

- Renamed `<image>_reproj_suppl.txt` to `<image>_reproj_img_suppl.txt`.
- A seven-line preamble precedes the header, stating that the matrix rows
  are the camera X, Y and Z axes in J2000, that row 3 is the boresight and
  matches the right ascension and declination given, and how the roll is
  defined (see Notes for the reviewer). The 14 header lines, their order
  and their formats are unchanged.
- Values: RA changed in 19,903 of 19,913 common files (median 0.002 deg,
  maximum 0.011), Dec in 19,912 (median 0.0005 deg, maximum 0.019), roll in
  18,241 (median 0.002 deg; 162 files by more than 1 degree: 76 in
  `iss_096rf_fmovie004_prime` and 71 in `iss_093rf_fmovie001_prime` by 156
  to 208 degrees while the camera axes moved by less than 0.002 degrees,
  because the review copy switched roll reference direction for boresights
  beyond about 64 degrees of declination; 15 in `iss_180rf_hiresfrng001_prime`
  where the camera X and Y axes did rotate about an unchanged boresight).
  The C-matrix changed in all 19,913 files; the boresight (row 3) moved by
  a median of 0.0017 degrees (6 arcsec, about five Narrow Angle pixels or
  half a Wide Angle pixel), maximum 0.019 degrees. Navigation Type changed
  in 10 files (Manual to Stars 5, Manual to Ring and/or Satellite Models 3,
  Stars to Manual 2).
- Unchanged in all files: Source Data Product ID, the six start/mid/stop
  times, Trajectory Kernels Query Time, Stellar Aberration Correction, Light
  Travel Time Correction.

## 6. Mosaics

### 6.1 Labels

- Camera: all 302 review-copy labels named the Wide Angle Camera; 231 now
  name the Narrow Angle Camera and 74 the Wide Angle Camera, with matching
  instrument components.
- `rings:maximum_corotating_ring_longitude` 360.00 to 359.98 (all 302);
  `rings:description` says the minimum and maximum "always span the full
  extent of the mosaic, even if not all longitudes contain valid data";
  "observed_ring_elevation" to "emission angle" with the incidence sentence
  added; Albers citation corrected; resolution statistics reordered; the
  moon sentence added to 125 mosaic labels (section 3).
- Comment: the longitude sentence gives the true valid range ("valid data
  for a total of 199.86 degrees of co-rotating longitude spanning the
  (possibly discontinuous) 199.86 degrees from 186.60 to 26.44, measured to
  the outer edges of those two longitude bins") where the review copy said
  "360.00 degrees from 0.00 to 359.98" in every label; a navigation
  paragraph added ("Before reprojecting, the pointing specified by the
  available SPICE kernels was refined ... The subjective quality of the
  navigation for all of the images for this mosaic is "good""; good 224,
  fair 57, poor 24); "two distinct "movies" consisting of" to "covering"
  (M2 and M3 sentences); in the 21 R- and N-type labels the sentence "The
  source images were calibrated using CISSCAL 4.0..." appeared twice and
  now appears once; `iss_191ri_rcasoccb001_vims` gains the occultation
  Notes block the other occultation mosaics already carried.
- Time_Coordinates start changed in 158 labels (138 by -1 s, 20 by the
  image-set changes, for example `iss_000ri_satsrchap001_prime`
  2004-06-20T18:19:31Z to 19:15:31Z) and stop in 169 (134 by +1 s). SCLK
  start count changed in 20 labels and stop in 35. The "N source images"
  count changed in 4 labels; "covering X degrees of inertial longitude from
  A to B" in 249 to 281 labels (median 0.05 degrees); "for N seconds (H
  hours)" in 248 (median +1 s); the image-name range in the title in 248.
- Ring radii: `minimum_ring_radius` changed in 280 labels and
  `maximum_ring_radius` in 272 (median 0.15 km, maximum 214 km for
  `iss_134ri_spkmvdfhp001_prime`); mean phase in 4, mean emission in 3.
- Display direction "Top to Bottom" to "Bottom to Top"; `Array_2D_Image`
  description added ("Sample 0 is co-rotating longitude 0 degrees ... so
  that longitude = 0.02 * Sample degrees"); `Special_Constants` comment
  reworded ("No data are available for this pixel, either because no image
  covered this longitude, or because the data were missing or corrupted in
  the source image..."); mission phase capitalisation; Rings dictionary
  version.
- Source-image table area: header "Source Image Index, LIDVID" to
  `image_index,LIDVID` (header length 27 to 19); field description
  shortened. Params table area: header length 141 to 379; fields 8 to 17;
  record length 79 to 165; record-order description added ("for a mosaic
  the column is round(rings:corotating_ring_longitude / 0.02)").

### 6.2 Arrays

- 296 of 302 common arrays byte-identical. Six differ:
  `iosic_276rb_complitb4001_si` (7,200 columns of the four moved images now
  missing, 25 columns changed), `iss_007ri_hpmrdfmov001_prime` (1,366
  columns, 14 images added), `iss_007ri_lphrlfmov001_prime` (373 columns, 4
  images added), `iss_134ri_spkmvdfhp001_prime` (all 18,000 columns, rebuilt
  from 65 images instead of 15), `iss_111rf_fmovie002_prime` (90 columns,
  from the two re-navigated images) and `iss_172ri_spokemov002_prime` (672
  columns from one image whose reprojected array is unchanged; maximum
  difference 0.005 I/F).

### 6.3 Metadata tables

- Params header: "Corotating Longitude, Image Index, Mid-time SPICE ET,
  Inertial Longitude, Radial Resolution, Angular Resolution, Phase Angle,
  Emission Angle" to the seventeen names of Appendix C with `image_index`
  second. Nine columns added.
- Mid-time ET: changed in 2,843,874 of 2,843,952 common rows. The review
  copy's values were whole seconds, multiples of 16 s in every row; the
  delivered values carry milliseconds and equal the image mid-time within
  1 ms; they differ from the review copy by 1 to 20 s in 91 percent of rows.
- Inertial longitude: changed in 99.4 percent of rows (per-table median
  difference 10.7 degrees). Longitudinal resolution: every row, factor
  pi/180, unit corrected as in 5.3. Corotating longitude unchanged in every
  row; phase, emission, radial resolution and image index changed only in
  the tables whose source images changed.
- Source-image tables: record counts changed in 4; LIDVID sets changed in 36
  tables (437 LIDVIDs only in the review copy, 501 only in the delivered
  bundle) as described in section 4.2; index-to-LIDVID mapping identical in
  266 of 302.

## 7. Background-subtracted mosaics

### 7.1 Labels

- All the mosaic-label changes of 6.1 apply. In addition: "If insufficient
  data was available" to "were available"; a sentence added stating the
  background quality ("The subjective quality of the background modeling and
  subtraction process for this mosaic is "good""; good 214, fair 71, poor
  20); the `Special_Constants` comment gains "or because no valid
  background model could be fit at this longitude"; the moon sentence in 87
  labels. The stated background limits ("from A to 1000 km closer to Saturn
  and B to 1000 km further") are unchanged for every product.
- Mean phase changed in 201 labels (median 0.006 degrees), mean emission in
  190, mean radial resolution in 166 (median 0.004 km), ring radii in 281
  and 274, "valid data for a total of X degrees" in 263 (median -1.4
  degrees, maximum -145 degrees), all following the coverage change below.

### 7.2 Arrays

- No common array is byte-identical (0 of 302).
- Background model. For the 296 products whose plain mosaic is identical,
  the difference between the review-copy and delivered values in every
  retained column is exactly a linear function of the line index (residual
  below 1e-7 I/F, single-precision rounding). Its size: median absolute
  difference 1.7e-7 I/F, 0.08 percent of the pixel value (product maxima
  median 4.4e-5, largest 6.6e-3); the difference grows from 3.2e-7 at the
  inner edge through 5.9e-7 at the core to 9.4e-7 at the outer edge. The
  delivered fit uses lines 0-49 and 351-400 (delta radius -1000 to -755 km
  and +755 to +1000 km, 50 rows each side); the review copy's fit included
  line 350 (+750 km) in the outer window, 51 rows outside and 50 inside, so
  the row at +750 km was fitted as background while its mirror at -750 km
  was treated as part of the ring.
- Coverage. 263 products have fewer valid longitudes than in the review copy
  and 4 have more: 81,125 longitudes lost and 62 gained; valid longitudes
  over all 302 common products 2,803,916 to 2,722,853 (2.9 percent fewer).
  Per product the loss has median 42 and 75th percentile 229 longitudes.
  Largest: `iosic_276rb_complitb4001_si` 7,261 (the moved images),
  `iss_036rf_fmovie002_vims` 5,084 (32 percent of its longitudes),
  `iss_036rf_fmovie001_vims` 4,670 (26 percent), `iss_007ri_azscnloph001_prime`
  3,519 (20 percent), `iss_197rf_fmovie002_prime` 3,197 (30 percent),
  `iss_289rf_fmovie001_prime` 2,677, `iss_213rf_fmovie002_prime` 2,601,
  `iss_198ri_spokemov005_prime_1` 2,121, `iss_029rf_fmovie001_vims` 1,847,
  `iss_245ri_hiresafrg002_prime` 1,129 (38 percent). In 224 of the 263 the
  lost pixels are whole columns. 99.3 percent of the lost longitudes have
  every pixel of both background windows valid in the plain mosaic, and in
  the products with at least 20 lost longitudes the lost columns'
  background scatter is a median 1.21 times that of the kept columns (10th
  percentile 1.01). This loss is specific to the delivered build; the
  intermediate builds between the review copy and delivery kept the review
  copy's column set. The critique of 2026-09-27 (section 3.3) asks the
  author to accept or reverse it.
- The six products whose plain mosaic changed differ by up to 0.09 I/F.
- In every observation the valid longitudes of the background-subtracted
  product are a subset of the mosaic's.

### 7.3 Metadata tables

- As in 6.3, plus: 263 tables lost records and 4 gained, following 7.2 (for
  example `iss_000ri_satsrchap001_prime` 18,000 to 17,930,
  `iss_007ri_azscnloph001_prime` 17,986 to 14,467, `iss_197rf_fmovie002_prime`
  10,685 to 7,488).
- Source-image tables: the background-subtracted list contains only images
  that still contribute a longitude, so 12 products list fewer images than
  their plain mosaic (131 images in total; for example 105 of 169 for
  `iss_007ri_hpmrdfmov001_prime`) and number them from 0 independently. In
  the review copy the lists were identical to the mosaic's.

## 8. Browse products

- Sizes unchanged for every common product: mosaic types 18000x401,
  1800x400, 200x200, 100x100; reprojected images max(V, 800) by 401,
  max(V/10, 400) by 400, 200x200, 100x100, where V is the number of
  longitudes with data. One image is one column wider (`1622023389n`).
- Grey-level mapping changed: the stretch uses a blackpoint at the larger
  of the minimum value and zero (the review copy used the minimum, which is
  negative for most products), a whitepoint at the 99.8th percentile and
  gamma 0.5. Pixel-identical PNGs: 49 of 302 mosaics, 0 of 302
  background-subtracted, 463 of 19,913 reprojected images (full size). In
  the changed images the new grey level is a monotonic remap of the old
  one, never brighter; mean brightness fell from 22.9 to 20.7 (mosaic full
  size) and 41.9 to 40.2 (reprojected); low-contrast images that were bright
  overall now show a dark background with a visible ring (largest change
  `iss_007ri_azscnloph001_prime/1493725544w`, mean 226 to 177).
- Background-subtracted browse images show the dropped longitudes of 7.2 as
  black columns; their black fraction rose from 62 to 64 percent.
- Overlay text, its position, the column re-ordering of wrapped images and
  the vertical orientation are the same in both builds.
- Browse labels: `browse products` added as a fifth keyword to the 610
  mosaic-type browse labels. Description rewritten in all three types:
  stretch described as above with the sentence "Because the blackpoint is
  never negative, the negative values ... are all shown as black"; the
  small size corrected from "400x400" to "200x200" (mosaic types); the
  reprojected-image description gives the size rules in words (the review
  copy described the full image as "equal in size to the reprojected image"
  and the med image as "downsampled by 10 in longitude"), states which sizes
  are resampled, describes the overlay text, and states that wrapped images
  are shown in order of increasing longitude from 0 degrees. Titles of 36
  mosaic browse products changed image-name range.

## 9. Global index tables

- Moved from `document/supplemental/` to `miscellaneous/`; LIDs
  `...:document:mosaic_global_index`, `...:document:mosaic_bkg_sub_global_index`,
  `...:document:reproj_img_global_index` to `...:miscellaneous:global_mosaic_index`,
  `...:miscellaneous:global_mosaic_bkg_sub_index`, `...:miscellaneous:global_reproj_img_index`.
  Rows 302 to 305 and 20,303 to 20,584.
- Columns: mosaic and background-subtracted indexes 33 and 35 to 57 and 60;
  reprojected-image index 30 to 42. Record lengths 539, 550 and 510 bytes to
  762, 775 and 625.

  | Column | Change |
  |---|---|
  | `product_creation_date` (date only) | replaced by `pds:creation_date_time` (UTC date-time) in all three |
  | mean / minimum / maximum `core_radius` | added to all three |
  | minimum / maximum `true_anomaly` | added to all three |
  | `nav_quality` | added to all three, values G, F, P |
  | `bkgnd_quality` | added to the background-subtracted index, values G, F, P |
  | mean / minimum / maximum of `longitude_ascending_node`, `longitude_pericenter`, `corotating_longitude_prometheus`, `radius_prometheus`, `corotating_longitude_pandora`, `radius_pandora` | added to the two mosaic indexes, 18 columns |
  | `longitude_ascending_node`, `longitude_pericenter`, and the four Prometheus and Pandora columns | added to the reprojected-image index, 6 columns |
  | `cassini:spacecraft_clock_start_count` / `stop_count` | retyped from real to string, with a description of the 1/256 s fraction and the omitted partition |
  | `bkgnd_lower_limit` / `bkgnd_upper_limit` | retyped from real to integer, unit none to km |
  | `pds:start_date_time`, `pds:stop_date_time`, `percent_coverage`, `num_valid_longitudes`, `notes`, `num_images` | spurious unit "none" removed |

- Label text: the `Header/description` column lists equal the tables'
  header lines (the review copy's listed unprefixed short names that
  matched neither the header line nor the field names); the mosaic and
  background-subtracted index tables had `local_identifier`
  `global_reproj_img_index` and now have their own; the descriptions of
  `rings:minimum/maximum_corotating_ring_longitude` state that the index
  holds the valid-data range while the product label holds the full grid,
  and that a wrapped range has minimum greater than maximum; the citation
  descriptions add that angle ranges are computed on the circle.
- Values, mosaic indexes (302 common rows): minimum/maximum corotating
  longitude changed in 110 rows (the review copy wrote 0.00/359.98 for
  every full-grid mosaic; the delivered bundle gives the valid-data range,
  for example `iss_006ri_lphrlfmov001_prime` 79.24/68.18; 52 rows remain
  0.00/359.98 because their coverage includes both ends); minimum and
  maximum inertial longitude in 301 and 302 rows (median 0.06 degrees,
  following the corrected mid-times); start and stop times in 158 and 169
  rows (mostly by 1 s); ring radii in about 280 rows (median 0.15 km);
  `notes` in 1 row (`iss_191ri_rcasoccb001_vims` gains `O`); observation ID
  in 1 (the `iosic` split); `num_images` in 4; first and last image names
  in 20 and 35. The background-subtracted index additionally changed
  `num_valid_longitudes` and `percent_coverage` in 263 rows and the angle
  and resolution statistics in up to 201 rows, following 7.2.
- Values, reprojected index (19,913 common rows): start time -1 s in 9,674
  rows, stop time +1 s in 8,721; corotating limits in 1,253 rows (wrapped
  images); inertial limits in 177; `notes` in 9 rows (the
  `iss_191ri_rcasoccb001_vims` images gain `O`); `file_spec` in the 4 moved
  images. Unchanged in every row: observation ID, SCLK counts, phase,
  incidence and emission means.

## 10. Document collection and user guide

### 10.1 Files and labels

- `document/user_guide/` holds the PDF, its label, and five example
  programs: `mosaic_utils.py` (13 functions for reading mosaics,
  reprojected images and indexes with `pds4_tools`), `display_reproj_img.py`,
  `plot_ews_ma.py`, `plot_ews_df.py`, `find_prometheus_closest_approaches.py`.
  Each runs against the bundle and reproduces the guide's Figures 7 to 9.
- Document label: title "Cassini ISS F Ring Mosaics User's Guide" to "...
  User Guide"; `document_name` "F Ring Mosaics User's Guide" to "Cassini ISS
  F Ring Mosaics User Guide"; `publication_date` 2025-09-22 to 2026-09-28;
  `files` 1 to 6; the PDF's `document_standard_id` "PDF" to "PDF/A"; the
  descriptions add "Example Python programs are also included."; an unused
  `xmlns:pds` declaration removed; the NAC component listed before the WAC.
  `publication_year` 2025, DOI 10.17189/ajhh-aj88, edition and author lists
  unchanged.

### 10.2 The PDF

- 19 to 47 pages; PDF/A-2b to PDF/A-1b (the delivered file passes veraPDF
  PDF/A-1b with no failures; the review copy failed PDF/A-1b on 14 checks
  and passed only its own declared PDF/A-2b); title and author metadata
  filled in (were empty); bookmarks 0 to 56; link annotations 0 to 251;
  figures 6 to 9; numbered tables 0 to 5; references 8 to 15; text 5,535 to
  16,071 words; "User's Guide" to "User Guide" throughout; headings in
  sentence case.
- Title page: version line "V1.0" to "Version 1.0, 2025"; DOI as a full
  URL; added paragraphs "Citing this bundle", "Citing this User Guide" and
  "Versions and errata" (version recording, errata posted at the Node,
  error reports to the author).
- New sections: 2 "Quick start" (PDS4 primer, abbreviations OBSID / IMGID /
  LID / LIDVID / I/F / COISS, two tables listing every file of every
  product type); 3.1.6 "Reprojected images that were not used in a mosaic";
  5 "Reading labels and data product files" (15 headings: off-the-shelf
  tools for labels, tables, CSV and image files; SBN, RMS and PDR software;
  the example programs with three worked commands and their figures; the
  GitHub repository for further software); 7 "Metadata and global index
  file fields" (Table 3: the 16 per-longitude fields; Table 4: the 42
  reprojected-index fields; Table 5: the 57 mosaic-index fields plus the
  three background-subtracted-only fields, each with a definition, preceded
  by a statement of units).
- Restructured: the old section 5 "References" became 6.5; "Reprojection"
  split into "F ring orbit" and "Reprojection process"; "N: Non-inertial"
  renamed "N: Neither inertial nor corotating"; the global index material
  moved from 3.3.2 (document directory) to 4.4 (miscellaneous directory)
  and its field summary replaced by the complete list of note codes with
  the label prose each expands to.
- Corrected statements (old to new): Cassini arrival "June 30, 2004" to
  "July 1, 2004"; product counts 20,303 images / 302 mosaics to 20,584 / 305
  with 20,441 used and 143 unused; wrapped reprojected images "the full grid
  size is stored" to "stored in two sections, first from the minimum
  corotating longitude to 359.98° and then from 0° to the maximum"; array
  shape "18,000 x 401" to "401 x 18,000" with the storage order, line
  numbering and core line stated; background windows "-1000 to -750 km" and
  "+750 to +1000 km" to "-1000 to -755 km" and "+755 to +1000 km", with the
  index fields `bkgnd_lower_limit` and `bkgnd_upper_limit` defined as
  signed delta radii bounding the ring; "the chosen pixel limits are
  reported in the label" to the index fields plus the label's prose
  sentence; browse Med size "1800 x 401" to "1800 x 400" and the
  reprojected-image size rules (minimum widths 800 and 400) stated; the
  metadata table row order no longer described as one-to-one with array
  columns, with the column formulas given for both product types; "two
  additional columns" in mosaic tables to one (`image_index`); the R and N
  mosaic lists, the star names of the O observations, and the M3 ranges
  added; figure caption numbers updated to the delivered tables.
- Added material: observation-name field definitions (REV as a fixed
  three-character field, the TI codes, the PINST codes SI and PIE), 16
  campaign name patterns (was 8), the navigation and background quality
  grades G/F/P with their meaning, the node rate -2.68778°/day, the J2000
  basis of the orbit constants, the corotation epoch in ET seconds
  (220881665.1839181), the definition of inertial ring longitude, the
  incidence and emission angle convention, the Prometheus/Pandora
  provenance (sat393.bsp, de438.bsp, cpck15Dec2017.tpc) and the meaning of
  "visually confirmed", the negative-value caveat for background-subtracted
  data, the browse stretch parameters, footnotes defining I/F and
  equivalent width, and seven references (Attree 2012, 2014; Beurle 2010;
  Cooper 2013; Cuzzi 2024; Gehrels 1980; Murray 2005).
- Excerpts: the delivered guide's label and table excerpts come from one
  image (1622049830n) and match the bundle except a stale
  `creation_date_time` and an abbreviated supplemental-file preamble. The
  review copy's label excerpt combined values from two images and matched
  neither, its supplemental excerpt named a file pattern that did not
  exist in the review copy (`_reproj_img_suppl.txt` against the actual
  `_reproj_suppl.txt`) and gave pointing values that differed from the
  file, and its metadata rows differed from the review-copy tables in
  value and precision.

## 11. SPICE kernels

- `kernels.ker`: 1,154 to 1,153 entries; `cpck15Dec2017_saturn_only.tpc`
  (which exists only in the author's tree) replaced by `cpck15Dec2017.tpc`;
  CK `10024_10029ra.bc` removed; every listed kernel is a file NAIF
  distributes.
- `kernels.lblx`: an XML declaration added (the review copy began with the
  `xml-model` instruction); `Time_Coordinates` changed from nil
  ("inapplicable") to 2004-06-20T19:15:31Z / 2017-09-07T21:51:58Z;
  `Observing_System` name "... - Wide Angle Camera" to "Cassini Orbiter
  Imaging Science Subsystem" with an ISSNA component added; investigation
  reference type `collection_to_investigation` to `data_to_investigation`.

## 12. XML schema collection

- Inventory entries corrected (section 2); label title reworded; CRLF to LF.

## 13. Formatting and conventions

- Line endings: `bundle.lblx`, the context and xml_schema collection labels
  and the six data and browse collection labels had CRLF line endings in the
  review copy; every file in the delivered bundle is LF.
- Every product label and index label of the review copy contained tab
  characters inside the `xsi:schemaLocation` value; the delivered labels use
  spaces.
- `modification_date`: 2025-09-22 in every review-copy label; 2026-09-28 in
  37,904 labels and 2026-09-27 in 4,501 (the products of the 17
  alphabetically first observations) in the delivered bundle.
- `creation_date_time` values: 2025-09-22 to 2026-09-27/28; the one value
  without a `Z` (spice_kernels inventory) now has one. Index tables give
  creation as a UTC date-time instead of a date.
- Metadata table column names carry the `rings:`, `pds:` and `cassini:`
  prefixes of the dictionary attributes they correspond to.
- The members of `Reprojection_Grid_Parameters` are ordered mean, minimum,
  maximum.

## 14. Unchanged from the peer-review copy

- Bundle LID, title, version, DOI, publication year, keywords, authors,
  contributors; the PDS, DISP, GEOM and CASSINI dictionary versions;
  observation directory names and all product file names except the
  supplemental text file; LIDs of all common products; the 56 pre-existing
  Cassini attributes; the orbit model constants and corotation epoch; the
  reprojection grid of 5 km by 0.02 degrees over plus and minus 1000 km;
  the -999 sentinel and the little-endian single-precision array format;
  the calibration statement; the reference structure among mosaic,
  background-subtracted, reprojected and browse products; the
  calibrated-image source references and the spacecraft clock counts of
  every reprojected image; the supplemental files' time fields; browse
  image sizes, overlays and orientation; the `bkgnd_lower_limit` /
  `bkgnd_upper_limit` values of every product; the spice_kernels inventory;
  296 of 302 mosaic arrays and 18,351 of 19,913 reprojected arrays byte for
  byte, plus the pixel values of a further 1,560 reprojected arrays.

## 15. References resolved outside this bundle

- The five `system_bundle:xml_schema` products for information model 1.24
  carry the identifiers of the dictionary labels published at
  pds.nasa.gov; their presence in the registry was not confirmed.
- `urn:nasa:pds:cassini_iss_saturn:document:iss-data-user-guide::2.0` does
  not resolve; the registry's current version is `::1.1`.
- The 20,584 calibrated source products
  `urn:nasa:pds:cassini_iss_saturn:data_calibrated:<image>_calib::1.0` are
  forward references to the coordinated ISS delivery and do not yet resolve.
- The context product for L2 Puppis (`star.l02_pup`, version 1.0,
  2026-09-03) exists at PDS but is absent from the context list shipped
  with `validate` 4.2.0, which therefore reports it as not found.

## Appendix A. The 28 observations with corrected source images

The image count of each is the same in both copies; the images named and
archived differ. The images the review copy named were earlier images of
the same observation than the ones the mosaic was built from.

| Observation | Images | Replaced | Review copy named | Delivered |
|---|---|---|---|---|
| `iss_000ri_satsrchap001_prime` | 79 | 38 | 1466448221n–1466480861n | 1466486141n–1466504381n |
| `iss_036rf_fmovie001_vims` | 109 | 8 | 1545556618n–1545559736n | 1545610134n–1545613256n |
| `iss_039rf_fmovie001_vims` | 135 | 4 | 1551253524n–1551255505n | 1551308313n–1551310298n |
| `iss_041rf_fmovie002_vims` | 146 | 17 | 1552790437n–1552796197n | 1552844797n–1552850917n |
| `iss_043rf_fmovie001_vims` | 82 | 5 | 1555557017n–1555559413n | 1555611013n–1555613413n |
| `iss_044rf_fmovie001_vims` | 134 | 33 | 1557020880n–1557033524n | 1557073700n–1557086720n |
| `iss_075rf_fmovie002_vims` | 106 | 5 | 1593913221n–1593915277n | 1593967807n–1593969867n |
| `iss_091ri_apomosl109_vims` | 23 | 5 | 1604280041w–1604283940w | 1604292383w–1604296869w |
| `iss_098ri_tmapn30lp001_cirs` | 24 | 2 | 1608699231w–1608703375w | 1608704571w–1608705204w |
| `iss_100ri_subms20lp001_cirs` | 24 | 10 | 1610925548w–1610927708w | 1610933888w–1610943792w |
| `iss_105ri_tdifs20hp001_cirs` | 21 | 8 | 1615342663w–1615351843w | 1615352983w–1615360723w |
| `iss_105ri_tmapn45lp001_cirs_4` | 1 | 1 | 1614950500w | 1614951216w |
| `iss_109ri_tdifs20hp001_cirs` | 19 | 4 | 1619014450w–1619029570w | 1619033230w–1619036946w |
| `iss_173rf_hiresfrng001_prime` | 45 | 11 | 1729261283n–1729262583n | 1729265313n–1729266613n |
| `iss_173ri_spokemov003_prime` | 131 | 32 | 1728757643w–1728769330w | 1728809669w–1728822864w |
| `iss_174ri_spokemov001_prime` | 92 | 7 | 1730574595w–1730631715w | 1730684755w–1730691895w |
| `iss_174ri_spokemov002_prime` | 44 | 44 | 1730746588w–1730799478w | 1730806858w–1730859748w |
| `iss_179rf_fmovie001_prime` | 140 | 3 | 1736795325n–1736796085n | 1736848801n–1736849565n |
| `iss_181rf_fmovie001_prime` | 131 | 9 | 1739125110n–1739128386n | 1739178816n–1739182096n |
| `iss_196rf_fmovie003_prime` | 129 | 1 | 1755729895n | 1755783297n |
| `iss_211rf_fmovie001_prime` | 101 | 7 | 1798999446n–1799002613n | 1799052443n–1799056067n |
| `iss_213rf_fmovie001_prime` | 116 | 37 | 1804612783n–1804628011n | 1804664389n–1804681732n |
| `iss_246rf_fmovie002_prime` | 87 | 19 | 1856269247n–1856280367n | 1856323009n–1856334133n |
| `iss_253rf_fmovie001_prime_2` | 130 | 13 | 1860623305n–1860628221n | 1860676601n–1860681521n |
| `iss_253rf_fmovie001_prime_3` | 5 | 1 | 1860686001n | 1860686881n |
| `iss_253ri_hiresafrg001_pie` | 40 | 1 | 1860782762n | 1860785582n |
| `iss_260rf_fmovie001_prime` | 264 | 41 | 1864955784n–1864963820n | 1865008844n–1865016884n |
| `iss_289rf_fmovie001_prime` | 127 | 24 | 1881776562n–1881785942n | 1881830414n–1881839798n |
| **Total** | | **390** | | |

Eight further observations gained images without losing any:
`iss_007ri_hpmrdfmov001_prime` (+14, 1493887352n–1493892602n),
`iss_007ri_lphrlfmov001_prime` (+4, 1493661246n–1493662416n),
`iss_134ri_spkmvdfhp001_prime` (+50, 1656604376w–1656634560w),
`iss_199rf_fmovie002_prime` (+97), `iss_256ri_hiresafrg002_prime` (+16),
`iss_262rf_fmovie001_prime_12` (+27), `iss_268rf_fmovie001_prime_1` (+3),
and the four images that moved from `iosic_276rb_complitb4001_si` to
`iosic_276rb_complitb3001_si`. The 36 mosaics whose review-copy attribution
did not match their columns are these 28 plus `iss_007ri_hpmrdfmov001_prime`,
`iss_007ri_lphrlfmov001_prime`, `iss_134ri_spkmvdfhp001_prime`,
`iss_172ri_spokemov002_prime`, `iss_199rf_fmovie002_prime`,
`iss_256ri_hiresafrg002_prime`, `iss_262rf_fmovie001_prime_12` and
`iss_268rf_fmovie001_prime_1`.

## Appendix B. Mosaic-level satellite and star targets

Prometheus is named on 25 mosaics the review copy did not name it on:
`iss_029rf_fmovie001_vims`, `iss_029rf_fmovie002_vims`,
`iss_036rf_fmovie001_vims`, `iss_075rb_bmovie4001_vims`,
`iss_134ri_spkmvdfhp001_prime`, `iss_173ri_spokemov002_prime`,
`iss_174ri_spokemov001_prime`, `iss_174ri_spokemov002_prime`,
`iss_180rf_fmovie001_prime`, `iss_200ri_spokemov001_prime`,
`iss_200ri_spokemov004_prime_1`, `iss_200ri_spokemov004_prime_2`,
`iss_200ri_spokemov011_prime`, `iss_201ri_spokemov001_prime_1`,
`iss_201ri_spokemov001_prime_2`, `iss_201ri_spokemov009_prime_1`,
`iss_201ri_spokemov011_prime`, `iss_201ri_spokemov013_prime`,
`iss_201ri_spokemov015_prime`, `iss_239rf_fmovie002_prime_1`,
`iss_242rf_fmovie001_prime`, `iss_243rf_fmovie001_prime_1`,
`iss_245ri_hiresafrg002_prime`, `iss_274rf_fmovie002_prime`,
`iss_276ri_hiresafrg001_prime_2`; and on the three added observations.
It is no longer named on `iss_082ri_fmonitor003_prime` or
`iss_207rf_fmovie001_prime`, where it lies 159 to 277 km beyond the radial
limit.

Pandora is named on 9 mosaics the review copy did not name it on:
`iss_036rf_fmovie001_vims`, `iss_041rf_fmovie001_vims`,
`iss_041rf_fmovie002_vims`, `iss_105ri_tmapn45lp001_cirs_5`,
`iss_173rf_fmovie001_prime_1`, `iss_173ri_spokemov003_prime`,
`iss_196rf_fmovie003_prime`, `iss_241rf_fmovie001_prime`,
`iss_292rf_fmovie001_prime`. It is no longer named on
`iss_087rf_fmovie003_prime`, where it lies 73 km beyond the limit.

Background-subtracted mosaics: Prometheus gained on 21 and lost on 26
products, Pandora gained on 6 and lost on 8, the losses being products
whose valid longitudes no longer include the moon's position (for example
`iss_006ri_lphrlfmov001_prime`, `iss_007ri_azscnloph001_prime`,
`iss_057rf_fmovie001_vims`, `iss_260rf_fmovie001_prime`).

Stars are named on the 15 occultation observations:
`iss_172ri_betpegocc001_vims` (Scheat), `iss_172st_urgampeg001_uvis`
(Algenib), `iss_180ri_rcasocc001_vims` and `iss_191ri_rcasoccb001_vims`
(R Cassiopeiae), `iss_180ri_rlyrocc001_vims` and `iss_198ri_rlyrocc001_vims`
(R Lyrae), `iss_185ri_rhyaocc001_vims_1` and `_2` (R Hydrae),
`iss_194ri_mucepocc001_vims` (Herschel's Garnet Star),
`iss_196ri_betandocc001_vims` (Mirach), `iss_197ri_whyaocc001_vims`
(W Hydrae), `iss_201ri_l2pupocc001_vims_1`, `_2`, `iss_205ri_l2pupocc002_vims`
and `iss_206ri_l2pupocc002_vims` (L2 Puppis).

## Appendix C. Metadata table columns

The mosaic and background-subtracted mosaic tables carry the columns below
with `image_index` inserted after the co-rotating longitude, giving
seventeen. The review copy's mosaic table had eight columns and its
reprojected-image table six.

| Review copy (unit as labelled) | Delivered (unit) | Note |
|---|---|---|
| Corotating Longitude (deg) | `rings:corotating_ring_longitude` (deg) | renamed |
| Image Index (mosaic only) | `image_index` | renamed |
| Mid-time SPICE ET (mosaic only, s) | `rings:observed_event_tdb` (s) | renamed; values were quantized to whole seconds |
| (none; reprojected image) | `rings:observed_event_tdb` (s) | new; constant per image |
| Inertial Longitude (deg) | `rings:inertial_ring_longitude` (deg) | renamed; values corrected |
| Radial Resolution (km/pixel) | `rings:radial_resolution` (km/pixel) | renamed |
| Angular Resolution (mosaic: deg/pixel, reprojected image: km/pixel) | `rings:longitudinal_resolution` (deg/pixel) | renamed; values were radians per pixel in both |
| (none) | `rings:incidence_angle` (deg) | new; constant per image |
| Phase Angle (deg) | `rings:phase_angle` (deg) | renamed |
| Emission Angle (deg) | `rings:emission_angle` (deg) | renamed |
| (none) | `core_radius` (km) | new |
| (none) | `longitude_ascending_node` (deg) | new; constant per image |
| (none) | `longitude_pericenter` (deg) | new; constant per image |
| (none) | `true_anomaly` (deg) | new |
| (none) | `corotating_longitude_prometheus` (deg) | new; constant per image |
| (none) | `radius_prometheus` (km) | new; constant per image |
| (none) | `corotating_longitude_pandora` (deg) | new; constant per image |
| (none) | `radius_pandora` (km) | new; constant per image |

Source-image table: the first column is renamed from "Source Image Index"
to `image_index`; the LIDVID column is unchanged.
