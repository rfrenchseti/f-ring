# Change log for `urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025`

## How the delivered bundle differs from the peer-review copy

This document describes the delivered bundle against the copy distributed for
peer review on 2025-09-22. It describes contents only, for a reviewer who wants
to check each difference against the products themselves. Two appendices list
the affected observations by name.

Identifiers, versions and authorship are unchanged. The bundle logical
identifier, the eleven collection identifiers, every product identifier present
in both copies, all version identifiers, the information model version
1.24.0.0, the bundle DOI 10.17189/3tfh-th07, the bundle title, and the author
and contributor lists are the same. Nothing here changes how the bundle is
cited.

---

## 1. Summary

### 1.1 Corrections to product content or identity

1. **The source images of 28 mosaics are correctly identified, and the
   reprojected images archived for those observations are the ones the mosaic
   was built from.** In the review copy the source-image table of each of these
   observations named earlier images of the same observation, and the archived
   reprojected products were those wrong images. Each named image can be checked
   by testing whether its exposure window contains the mid-time recorded in the
   mosaic's own metadata table; every named image in all 305 mosaics passes.
   390 reprojected-image products are replaced. The bundle's earliest start time
   is 2004-06-20T19:15:31Z. Appendix A lists the observations.
2. **The 2010 SPKMVDFHP observations are archived as three mosaics.** The review
   copy carried a single mosaic labelled as the first of them but built from the
   third observation's 15 images. The delivered bundle has
   `iss_134ri_spkmvdfhp001_prime`, `_002_prime` and `_003_prime` with 65, 55 and
   15 images.
3. **The two IOSIC_276RB observations are separate.** The review copy combined
   9 images from both under one mosaic carrying the other observation's
   identifier. The delivered bundle has `iosic_276rb_complitb3001_si` with 4
   images and `iosic_276rb_complitb4001_si` with 5, each under its own
   identifier.
4. **Every reprojected-image array is stored as its label describes.** Each
   array begins at its stated minimum co-rotating longitude, continues
   contiguously through 360 degrees when the range wraps, keeps its no-data
   columns as the −999 sentinel, and has a sample count matching the stated
   range. In the review copy 1,693 labels stated a range that did not match the
   array width, 1,251 wrapping arrays were stored beginning at 0 degrees with
   the above-wrap segment appended, and 309 arrays had their no-data columns
   removed.
5. **The inertial longitude in every metadata table is consistent with the
   mid-time and co-rotating longitude in its own row.** In the review copy the
   reprojected-image tables were low by the longitude at which the table
   started, and 40% of mosaic rows disagreed with their own row by a median of
   25 degrees.
6. **Mid-times are exact.** The tabulated ephemeris time equals the image
   mid-time to within 2 ms. The review copy's values were quantized to 16, 32 or
   64 seconds depending on epoch.
7. **Longitudinal resolution is in degrees per pixel**, agreeing with the
   resolution statistics in the labels. The review copy's column held radians
   per pixel while its own label declared degrees.
8. **Every label names the correct camera.** 231 of 305 mosaics are Narrow Angle
   Camera and 74 are Wide Angle; 17,830 of 20,584 reprojected images are Narrow
   Angle. The review copy declared Wide Angle on every mosaic label and gave a
   generic instrument name with the Wide Angle component on every
   reprojected-image label.
9. **Satellite targets are listed per product.** A moon is named on a product
   only when its predicted position falls within 1050 km of the ring core at a
   co-rotating longitude that product covers: Prometheus on 772 reprojected
   images and 103 mosaics, Pandora on 74 and 22. The review copy listed
   Prometheus on 8,755 reprojected images and Pandora on 1,896. Appendix B lists
   the mosaic-level changes.
10. **Occultation stars are targets.** The occulted star is named, with its PDS
    context identifier, on every reprojected image, mosaic and
    background-subtracted mosaic of the 15 stellar occultation observations, on
    the bundle label, on the three data collection labels, and in the context
    inventory. Nine stars: Algenib, Herschel's Garnet Star, L2 Puppis, Mirach,
    R Cassiopeiae, R Hydrae, R Lyrae, Scheat, W Hydrae. The review copy named no
    star anywhere.
11. **Background-subtracted mosaics list only source images that contribute to
    them.** Seven observations list fewer images than their mosaic, for example
    105 of 169 for `iss_007ri_hpmrdfmov001_prime`.
12. **The background model is fitted to a symmetric region.** The interior
    background is the innermost 50 rows, delta radius −1000 to −755 km, and the
    exterior background is the outermost 50 rows, +755 to +1000 km. In the
    review copy the exterior region was 51 rows, reaching one row further in to
    +750 km, so the row at +750 km was fitted as background while its mirror at
    −750 km was treated as part of the ring. Every background-subtracted array
    differs from the review copy as a result; see section 6.1.
13. **The supplemental pointing files state their conventions and use one roll
    definition.** Each records that the three matrix rows are the camera X, Y and
    Z axes in J2000 coordinates, so the third row is the boresight, and that the
    roll is the angle of the camera X axis about the boresight measured from the
    direction of increasing declination. In the review copy 161 files, all with
    boresight declination beyond about 64 degrees, reported a roll measured
    against a different reference axis from the other 20,423, differing from
    them by 69 to 180 degrees.
14. **Time coordinates enclose the exposure.** Start times are floored and stop
    times ceiled. The review copy rounded both to the nearest second, so the
    stated interval contained the exposure for only a third of the images.

### 1.2 Additions

- Three observations, giving 305 mosaics.
- **Every reprojected image of the R and O observation classes is archived.**
  For most observations the archived reprojected images are exactly those that
  contributed to the mosaic. For the R observations, where Cassini follows one
  co-rotating longitude range across different inertial longitudes, and the O
  observations, which are stellar occultations, the mosaic is of limited use and
  the reprojected images are the product of interest, so every image of the
  observation is archived whether or not it supplied data to the mosaic. This
  adds 143 products in four of the nine R observations:
  `iss_199rf_fmovie002_prime` (97), `iss_262rf_fmovie001_prime_12` (27),
  `iss_256ri_hiresafrg002_prime` (16) and `iss_268rf_fmovie001_prime_1` (3). The
  other five R observations and all 15 O observations already archived every
  image, because in those every image contributes. A non-contributing image
  appears in no source-image table and is referenced by no image index; its own
  label says so.
- 20,584 reprojected images, 281 more than the review copy. Four reasons account
  for the difference exactly:

  | Reason | Images |
  |---|---|
  | The 2010 SPKMVDFHP observations archived in full as three observations | +120 |
  | R-class observations archiving every image | +143 |
  | Two observations gaining images that contribute to their mosaics: `iss_007ri_hpmrdfmov001_prime` (+14), `iss_007ri_lphrlfmov001_prime` (+4) | +18 |
  | The IOSIC_276RB separation: four images moved to a new observation | 0 |
  | The 28 observations with corrected source images: 390 replaced by 390 | 0 |
  | **Net** | **+281** |

- Nine new columns in every reprojected-image metadata table and eight in every
  mosaic metadata table: incidence angle, mid-time for reprojected images, F ring
  core radius, longitude of ascending node, longitude of pericenter, true
  anomaly, and the co-rotating longitude and radius of Prometheus and Pandora.
- A `miscellaneous` collection holding the three global index files, which carry
  12 to 25 more columns each: orbit and satellite statistics, navigation
  quality, background-fit quality, and a full creation timestamp.
- A `readme.txt` at the bundle root.
- Five example Python programs in the document collection, and a user guide of
  47 pages with three new sections, delivered as PDF/A-1b.
- Four Cassini mission-specific attributes in every reprojected-image label:
  ground software version, PDS3 target description, and the two valid-maximum
  values.
- A description of the array axes in every data label, giving which line is the
  inner edge and what each sample means.

### 1.3 Conventions

- Longitude ranges that wrap through 360 degrees are reported with the minimum
  greater than the maximum, in labels and in index files.
- A mosaic label's co-rotating longitude range is always the full grid, 0.00 to
  359.98. The range of longitudes containing valid data is given in the global
  index and in the label's comment.
- The declared vertical display direction is Bottom to Top in every data label,
  matching how the arrays are stored.
- Cassini mission phase names use the node's title-case vocabulary: Equinox
  Mission, Solstice Mission, Tour, Tour Pre-Huygens.
- The Rings dictionary is version 1.15.0.0, schema `PDS4_RINGS_1O00_1F00`, and
  the members of `Reprojection_Grid_Parameters` are ordered mean, minimum,
  maximum as that version requires.
- Metadata table column names carry the `rings:`, `pds:` and `cassini:` prefixes
  of the dictionary attributes they correspond to.
- Browse images are stretched with a black point never below zero and a white
  point at the 99.8th percentile of valid pixels.
- The F ring orbit reference reads "Albers et al. (2012), Table 3, fit #2".

---

## 2. Inventory

| Collection or file | Review copy | Delivered | Difference |
|---|---|---|---|
| `bundle.lblx` | 1 | 1 | content |
| `readme.txt` | none | 1 | new |
| `data_mosaic` | 302 products | 305 | +3 observations |
| `data_mosaic_bkg_sub` | 302 | 305 | +3 |
| `data_reproj_img` | 20,303 | 20,584 | 390 removed, 671 added, 4 moved |
| `browse_mosaic`, `browse_mosaic_bkg_sub` | 302 each | 305 each | +3 |
| `browse_reproj_img` | 20,303 | 20,584 | as `data_reproj_img` |
| `context` | 8 inventory rows | 17 | +9 star targets |
| `document` | guide + 3 global indexes | guide + 5 example programs | indexes moved out |
| `miscellaneous` | none | 3 global index products | new collection |
| `spice_kernels` | 1 product | 1 | kernel list differs by two entries |
| `xml_schema` | 5 inventory rows | 5 | all five identifiers restyled |

The supplemental pointing file of each reprojected image is named
`<image>_reproj_img_suppl.txt`; in the review copy it was
`<image>_reproj_suppl.txt`. The global index files keep their base names and
live in `miscellaneous/`.

---

## 3. Bundle level

- Time span 2004-06-20T19:15:31Z to 2017-09-07T21:51:58Z.
- The bundle description names the mosaics, reprojected images, metadata,
  documentation and global index files, and quotes that period.
- Eleven `Bundle_Member_Entry` elements, including the `miscellaneous`
  collection.
- Thirteen targets: the F ring, Saturn's rings, Prometheus, Pandora and the nine
  occultation stars, each referencing its PDS context product.
- A `File_Area_Text` block describing `readme.txt`, which gives the bundle and
  document identifiers, the recommended citation and contact addresses.

---

## 4. Observations and product membership

### 4.1 Observations added or separated

| Mosaic | Images | Note |
|---|---|---|
| `iosic_276rb_complitb3001_si` | 4, Wide Angle | new |
| `iosic_276rb_complitb4001_si` | 5 | 9,332 valid longitudes; starts 2017-05-27T02:19:39Z |
| `iss_134ri_spkmvdfhp001_prime` | 65 | 2010-06-30T12:33:28Z to 23:30:33Z |
| `iss_134ri_spkmvdfhp002_prime` | 55 | new |
| `iss_134ri_spkmvdfhp003_prime` | 15 | new |

### 4.2 Observations with corrected source images

Twenty-eight observations, 390 images, listed in Appendix A. The image count of
each is unchanged; the images named and archived differ. Mosaic start and stop
times, spacecraft clock counts and the "images used range from" sentence follow
the corrected lists, with start shifts of 393 to 60,269 seconds and stop shifts
of 60 to 60,270 seconds.

### 4.3 Observations that gained images

| Observation | Review copy | Delivered | Note |
|---|---|---|---|
| `iss_199rf_fmovie002_prime` (R) | 23 | 120 | 97 not used in the mosaic |
| `iss_256ri_hiresafrg002_prime` (R) | 51 | 67 | 16 not used |
| `iss_262rf_fmovie001_prime_12` (R) | 17 | 44 | 27 not used |
| `iss_268rf_fmovie001_prime_1` (R) | 16 | 19 | 3 not used |
| `iss_007ri_hpmrdfmov001_prime` | 155 | 169 | 14 added to the mosaic |
| `iss_007ri_lphrlfmov001_prime` | 229 | 233 | 4 added to the mosaic |

The first four are R-class observations archiving their complete image
sequences. The 143 non-contributing images say so in their own labels: "This
reprojected image is part of observation X but was not used to create mosaic Y,
which covers only part of the observation." The other 20,441 say "This
reprojected image was used to create mosaic Y."

---

## 5. Reprojected images

### 5.1 Arrays

Of the 19,913 images present in both copies, 18,351 arrays are byte-identical to
the review copy and 1,560 contain the same valid pixels in a different layout:
1,251 wrapping arrays begin at their minimum longitude and run through 360
degrees, and 309 have their no-data columns restored so that column position maps
to longitude. Two arrays in `iss_111rf_fmovie002_prime`, 1622023115n and
1622023389n, hold re-navigated pixel values differing by up to 0.07 in I/F.

### 5.2 Labels

Counts are over the 19,913 labels present in both copies unless stated.

| Element | Review copy | Delivered | Labels |
|---|---|---|---|
| `Observing_System/name` | "Cassini Orbiter Imaging Science Subsystem" | "Cassini Imaging Science Subsystem - Narrow \| Wide Angle Camera" | all |
| instrument component | `isswa.co` on every label | `issna.co` on 17,392, `isswa.co` on 2,521 | 17,392 changed |
| `Target_Identification` Prometheus | present on 8,755 | present on 772 | −8,100, +279 |
| `Target_Identification` Pandora | present on 1,896 | present on 74 | −1,819, +32 |
| `Target_Identification` star | none | 9 stars | +163 |
| `rings:minimum/maximum_corotating_ring_longitude` | 0.00 / 359.98 on 1,179 wrapping images | wrapped range, minimum greater than maximum | 1,253 |
| `Axis_Array[Sample]/elements` | inconsistent with the stated range on 1,693 | matches the stated range on all | 496 changed |
| `Array_2D_Image/description` | absent | line and sample definitions | all |
| `Array_2D_Image/local_identifier` | `image` | `reproj_image` | all |
| `start_date_time` / `stop_date_time` | rounded to the nearest second | floored / ceiled | 9,674 / 8,721 |
| `cassini:ground_software_version_id` | absent | four distinct values | all |
| `cassini:pds3_target_desc` | absent | five distinct values | all |
| `cassini:valid_maximum_full_well` | absent | 4095, or 9896 on 55 gain-mode-12 images | all |
| `cassini:valid_maximum_DN_sat` | absent | 4095 | all |
| `cassini:image_observation_type` | `'SCIENCE'`, `'SUPPORT'`, `'SCIENCE', 'SUPPORT'` | `SCIENCE`, `SUPPORT`, or two elements | 19,862 |
| `cassini:missing_lines` | `N/A` | `-1` | 1,068 |
| `cassini:mission_phase_name` | upper case | title case | all |
| `disp:vertical_display_direction` | Top to Bottom | Bottom to Top | all |
| supplemental `file_name` | `<image>_reproj_suppl.txt` | `<image>_reproj_img_suppl.txt` | all |

Prose. The citation description changes from "Reprojected version of Cassini ISS
calibrated image X from observation Y." to "... from Cassini observation Y. This
reprojected image was used to create mosaic Z." or, for the 143 non-contributing
images, "... is part of observation Y but was not used to create mosaic Z, which
covers only part of the observation."

The comment changes "this reprojected image is used in the mosaic named Z" to
"is associated with the mosaic named Z", updates the orbit reference to "Albers
et al. (2012), Table 3, fit #2", and adds the navigation quality of the mosaic.
The review copy's blanket sentence "Some reprojected F-ring images may include
Prometheus and/or Pandora, but their presence has not been visually confirmed.
&lt;Target_Identification&gt; lists only confirmed targets (the rings)." is replaced by
a per-product sentence giving the predicted-position test, present only where a
moon is actually named.

The `rings:description` replaces "phase angle and observed_ring_elevation" with
"phase angle and emission angle", adds that the minimum and maximum incidence
angle are set to the mean, and replaces "If the reprojection wraps around then
they will be 0 and 359.98" with "then the minimum will be greater than the
maximum".

### 5.3 Metadata tables

Six columns become sixteen. Record length goes from 58 to 159 bytes, and the
header separator from ", " to ",".

| Review copy (unit as labelled) | Delivered (unit) | Note |
|---|---|---|
| Corotating Longitude (deg) | `rings:corotating_ring_longitude` (deg) | renamed |
| — | `rings:observed_event_tdb` (s) | new; mid-time, constant per image |
| Inertial Longitude (deg) | `rings:inertial_ring_longitude` (deg) | renamed; values corrected |
| Radial Resolution (km/pixel) | `rings:radial_resolution` (km/pixel) | renamed |
| Angular Resolution (km/pixel) | `rings:longitudinal_resolution` (deg/pixel) | renamed; values were radians per pixel |
| — | `rings:incidence_angle` (deg) | new; constant per image |
| Phase Angle (deg) | `rings:phase_angle` (deg) | renamed |
| Emission Angle (deg) | `rings:emission_angle` (deg) | renamed |
| — | `core_radius` (km) | new |
| — | `longitude_ascending_node` (deg) | new; constant per image |
| — | `longitude_pericenter` (deg) | new; constant per image |
| — | `true_anomaly` (deg) | new |
| — | `corotating_longitude_prometheus` (deg), `radius_prometheus` (km) | new; constant per image |
| — | `corotating_longitude_pandora` (deg), `radius_pandora` (km) | new; constant per image |

The label states that records run in order of increasing co-rotating longitude
from 0 degrees, that this is not the array's column order when the range wraps,
that longitudes without valid data have no record, and that the longitude field
rather than the record number associates a record with a column.

### 5.4 Supplemental pointing files

Each file opens with a header stating that it holds a C-matrix describing the
rotation from J2000 to the camera pointing, that the three matrix rows are the
camera X, Y and Z axes in J2000 coordinates so the third row is the boresight
and matches the right ascension and declination given, and that the roll is the
angle of the camera X axis about the boresight measured from the direction of
increasing declination, positive towards increasing right ascension. The review
copy's header carried the first of these statements only.

Every file differs from the review copy. The pointing itself moved: the boresight
by a median of 6.4 arcseconds, about five Narrow Angle pixels or half a Wide
Angle pixel, with a 99th percentile of 10.6 arcseconds. The roll is computed
against a single reference for every boresight, so the 161 files with declination
beyond about 64 degrees no longer report a roll measured against a different
axis. The navigation type recorded for 10 images differs. The recorded times and
source product identifiers are unchanged.

### 5.5 Browse products

The images themselves are the same sizes as in the review copy: full is the
number of valid longitudes or 800 pixels, whichever is greater, by 401; med is
one tenth that width or 400, whichever is greater, by 400; small is 200 by 200;
thumb is 100 by 100. All omit longitudes with no data.

The label text is rewritten to describe what the images actually are. The review
copy described the full image as "equal in size to the reprojected image, with a
minimum width of 800 pixels" and the med image as "downsampled by 10 in
longitude"; the delivered text gives the rules above and says that any size other
than full is resampled and that one narrower than its minimum width is stretched
to reach it. The black point is described as the minimum image value or zero,
whichever is greater, so that negative calibration noise renders black. The med
image is stated to carry the observation name and the image name in the upper
left and the small and thumb images the image name alone. A new paragraph says
the browse columns run from 0 degrees, which is not the array's column order for
a wrapping image.

The mosaic browse labels carry the same black point and overlay corrections, gain
the keyword "browse products", and correct the stated small size from 400 by 400
to the actual 200 by 200.

---

## 6. Mosaics and background-subtracted mosaics

### 6.1 Arrays

All arrays are 401 by 18,000 single-precision floats with −999 for missing data.

**Mosaics.** Of the 302 observations present in both copies, 296 arrays are
byte-identical to the review copy. Six differ: `iosic_276rb_complitb4001_si`
loses the 7,200 longitude columns of the images that moved to the new
observation; `iss_007ri_hpmrdfmov001_prime` and `iss_007ri_lphrlfmov001_prime`
gain columns from added images; `iss_111rf_fmovie002_prime` changes over 90
columns from two re-navigated images; `iss_134ri_spkmvdfhp001_prime` is rebuilt
from 65 images instead of 15; and `iss_172ri_spokemov002_prime` changes over 672
columns.

**Background-subtracted mosaics.** Every array differs from the review copy,
because the background is fitted to a symmetric 50-row region on each side
rather than 50 inside and 51 outside. The change is small and systematic:
measured across full mosaics it moves every background-subtracted pixel, with a
maximum around 1.7e-4 in I/F, and shifts equivalent widths by a median of
0.0001 km against typical values of 5 to 20 km. Beyond that, the same six
observations listed above change for the reasons given, and seven others lose
whole longitude columns where no background model could be fitted.

In every observation the valid longitudes of the background-subtracted product
are a subset of the mosaic's.

### 6.2 Labels

Counts are over the 302 observations present in both copies, and apply to the
mosaic and background-subtracted label alike unless stated.

| Element | Review copy | Delivered | Labels |
|---|---|---|---|
| `Observing_System` and instrument component | Wide Angle on every label | Narrow Angle on 231, Wide Angle on 71 | 231 changed |
| `rings:maximum_corotating_ring_longitude` | 360.00 | 359.98 | all |
| `Target_Identification` Prometheus | 77 mosaics | 103 | +25, −2 |
| `Target_Identification` Pandora | 14 mosaics | 22 | +9, −1 |
| `Target_Identification` star | none | 9 stars on 15 mosaics | +15 |
| `start_date_time` | — | corrected source lists, floored | 158 |
| `stop_date_time` | — | corrected source lists, ceiled | 169 |
| `cassini:spacecraft_clock_start_count` / `stop_count` | — | follow the corrected lists | 20 / 35 |
| `cassini:observation_id` | `IOSIC_276RB_COMPLITB3001_SI` on the 4001 product | its own identifier | 1 |
| `rings:minimum_ring_radius` / `maximum_ring_radius` | — | median shift 0.15 km | 280 / 272 |
| `Array_2D_Image/description` | absent | line and sample definitions | all |
| title and citation description | — | corrected image names and times | 248 |

Prose. The comment updates the orbit reference, and adds four sentences on
navigation ending with the subjective quality of the mosaic, good on 223, fair
on 56, poor on 23. Where the review copy said a chunked observation "consists of
two distinct movies consisting of approximately the same co-rotating longitudes"
it now reads "covering". A duplicated CISSCAL sentence in 21 R and N labels
appears once. `iss_191ri_rcasoccb001_vims` gains the occultation Notes block the
other occultation mosaics already carried.

The `rings:description` carries the same three corrections as the
reprojected-image labels, plus a per-product satellite sentence giving the
predicted-position test and whether the presence was visually confirmed:
Prometheus confirmed on 75 and unconfirmed on 25, Pandora confirmed on 13 and
unconfirmed on 9.

Background-subtracted labels additionally state the background margins in words,
give the background quality, and extend the missing-data comment with "or
because no valid background model could be fit at this longitude".

### 6.3 Metadata tables

Seventeen columns, the sixteen of the reprojected-image table plus `image_index`
after the co-rotating longitude. Record length 165 bytes. The source-image table
has columns `image_index` and `LIDVID`.

---

## 7. Global index files

The three index files are products of the `miscellaneous` collection with
identifiers `global_mosaic_index`, `global_mosaic_bkg_sub_index` and
`global_reproj_img_index`. They hold 305, 305 and 20,584 rows and 57, 60 and 42
columns.

In the review copy they were products of the document collection, with
identifiers `mosaic_global_index`, `mosaic_bkg_sub_global_index` and
`reproj_img_global_index`, and all three declared the same table local
identifier, `global_reproj_img_index`. Field counts go from 33, 35 and 30.
Record lengths go from 539, 550 and 510 bytes to 762, 775 and 625.

| Column | Change |
|---|---|
| `product_creation_date` | removed, a date only |
| `pds:creation_date_time` | added, a full timestamp |
| mean / minimum / maximum `core_radius` | added to all three |
| minimum / maximum `true_anomaly` | added to all three |
| `nav_quality` | added to all three, values G, F, P |
| `bkgnd_quality` | added to the background-subtracted index, values G, F, P |
| mean / minimum / maximum of `longitude_ascending_node`, `longitude_pericenter`, `corotating_longitude_prometheus`, `radius_prometheus`, `corotating_longitude_pandora`, `radius_pandora` | added to the two mosaic indexes, 18 columns |
| `longitude_ascending_node`, `longitude_pericenter`, and the four Prometheus and Pandora columns | added to the reprojected-image index, 6 columns |
| `cassini:spacecraft_clock_start_count` / `stop_count` | retyped from real to string |
| `bkgnd_lower_limit` / `bkgnd_upper_limit` | retyped from real to integer, unit km |
| `pds:start_date_time`, `pds:stop_date_time`, `percent_coverage`, `num_valid_longitudes`, `notes`, `num_images` | spurious unit "none" removed |

The header line lists the prefixed field names, matching the field definitions;
in the review copy it listed unprefixed names.

Row values. In the two mosaic indexes the start time changes on 158 rows and the
stop time on 169, the clock counts on 20 and 35, the first and last image names
on 20 and 35, and the image count on 4. The co-rotating longitude limits change
on 110 rows, all from 0.00 / 359.98 to a wrapped range; 52 rows remain 0.00 /
359.98 because their coverage includes both ends. The inertial limits change on
essentially every row by a median of 0.06 degrees, following the corrected
mid-times. In the reprojected-image index the start and stop times change on
9,674 and 8,721 rows by one second, the co-rotating limits on 1,253 rows, and
the file path on the 4 moved images.

Spacecraft clock columns are typed as strings and their descriptions explain the
1/256-second fraction and the omitted partition. The background limit columns are
typed as integers in km and bound the ring, so the background is every row
outside them. The co-rotating longitude columns of the mosaic indexes note that
they give the range containing valid data, which differs from the product label's
full-grid range, and that a range wrapping through 360 degrees has its minimum
greater than its maximum.

---

## 8. Document collection

The user guide is 47 pages, delivered as PDF/A-1b with an sRGB output intent and
its `document_standard_id` recorded as PDF/A. It carries a title and author in
its metadata, 247 link annotations and a four-level bookmark tree.

Beyond the review copy's 19 pages it adds three sections: a quick start
orienting a first-time user to the files in each product directory; a section on
reading labels and data products with general-purpose software and with the
PDS4-specific tools, covering the five example programs now shipped alongside it;
and a field-by-field reference for the metadata tables and the three global
indexes, preceded by a statement of units. Two subsections are new, one on what
the R and O classes archive and one on the miscellaneous directory.

It states conventions the review copy did not: the orientation of the arrays, the
lit-side convention for incidence and emission angles, the wraparound longitude
convention with the formulas for mapping a longitude to a column, the meaning of
the background margin fields, and the definitions of the navigation and
background quality grades. It identifies the star behind each occultation and
warns against clipping the negative background pixels that subtraction leaves.
The front matter carries citation blocks for both the bundle and the guide and a
versions and errata policy.

The document label lists six files: the guide and the five example programs
`mosaic_utils.py`, `display_reproj_img.py`, `plot_ews_ma.py`, `plot_ews_df.py`
and `find_prometheus_closest_approaches.py`.

---

## 9. Other collections

**context.** Nine star rows added, 8 inventory rows becoming 17:
`star.13_lyr`, `star.bet_and`, `star.bet_peg`, `star.gam_peg`, `star.l02_pup`,
`star.mu._cep`, `star.r_cas`, `star.r_hya`, `star.w_hya`.

**document.** The three global index products are no longer members; they are in
the `miscellaneous` collection. In the review copy the inventory named them with
single-colon identifiers that matched no label. The ISS Data User's Guide moves
from version 1.0 to 2.0, and eight context products are added as secondary
members. The collection title loses a trailing period.

**miscellaneous.** New. Three primary rows for the global index products and the
same nine secondary rows as the document collection.

**spice_kernels.** The metakernel lists 1,153 kernels rather than 1,154:
`cpck15Dec2017_saturn_only.tpc` is replaced by `cpck15Dec2017.tpc` and
`10024_10029ra.bc` is removed, so every listed kernel is a file NAIF
distributes. The kernel label now carries the time span of the observations the
kernels were assembled to cover; in the review copy both time elements were nil
with the reason "inapplicable". Its instrument name changes from "Cassini
Orbiter Imaging Science Subsystem - Wide Angle Camera" to the subsystem name
with both camera components listed, and its investigation reference type from
`collection_to_investigation` to `data_to_investigation`. The collection label
loses a `Time_Coordinates` block that did not match the kernels and a leftover
curator comment.

**xml_schema.** All five inventory rows carry the identifiers the published
dictionary labels declare:

| Review copy | Delivered |
|---|---|
| `pds-xml_schema::1.24` | `pds-xml_schema_1.24.0.0::1.0` |
| `disp-xml_schema::1.15` | `disp-xml_schema_1.24.0.0_1.5.1.0::1.0` |
| `geom-xml_schema::1.19` | `geom-xml_schema_1.24.0.0_1.9.11.0::1.0` |
| `rings-xml_schema::1.14` | `rings-xml_schema_1.24.0.0_1.15.0::1.0` |
| `cassini-xml_schema::1.18` | `cassini-xml_schema_1.24.0.0_1.8.0.0::1.0` |

**Data and browse collections.** The three data collection labels gain the nine
star targets and carry the corrected bundle start time; all six carry the new
inventory counts.

---

## 10. Common to every label

| Change | Review copy | Delivered | Labels |
|---|---|---|---|
| Rings dictionary | `PDS4_RINGS_1O00_1E00`, version 1.14.0.0 | `PDS4_RINGS_1O00_1F00`, version 1.15.0.0 | 21,194 |
| `Reprojection_Grid_Parameters` order | minimum, maximum, mean | mean, minimum, maximum | 21,194 |
| `disp:vertical_display_direction` | Top to Bottom | Bottom to Top | 21,194 |
| `cassini:mission_phase_name` | `SOLSTICE MISSION`, `EQUINOX MISSION`, `TOUR`, `TOUR PRE-HUYGENS` | `Solstice Mission`, `Equinox Mission`, `Tour`, `Tour Pre-Huygens` | 21,194 |
| guide reference comment | "Detailed User's Guide" | "Detailed User Guide" | all that reference it |
| `person_orcid` | `http://orcid.org/...` | `https://orcid.org/...` | 16 |
| `Modification_Detail/description` | "Initial version" | "Initial version." | bundle, context, document |

The PDS, DISP, GEOM and Cassini dictionary versions are unchanged.

---

## 11. Unchanged from the peer-review copy

- All identifiers and versions; authors, contributors and affiliations; the
  investigation and spacecraft references; the DOI.
- The F ring orbit model, the co-rotation epoch and rate, the reprojection grid
  of 5 km by 0.02 degrees over plus and minus 1000 km, the −999 sentinel, the
  little-endian single-precision array format, and the calibration statement.
- 296 of 302 mosaic arrays and the valid pixels of 19,911 of 19,913 reprojected
  images.
- The Cassini mission-specific attributes other than the four added.
- The spacecraft clock counts of every reprojected image and the references to
  the calibrated source images.
- Browse image sizes and format.

---

## 12. References resolved outside this bundle

- The five `system_bundle:xml_schema` products for information model 1.24 are not
  yet in the PDS registry.
- `urn:nasa:pds:cassini_iss_saturn:document:iss-data-user-guide::2.0` and the
  20,584 calibrated source products are forward references to the coordinated ISS
  delivery.

---

## 13. Summary table

| Change | Product type | Count / compared | Review copy to delivered |
|---|---|---|---|
| Observations added | mosaic, bkg-sub, browse | 3 each | IOSIC_276RB_COMPLITB3001, SPKMVDFHP002, SPKMVDFHP003 |
| Source images corrected | mosaic | 28 observations | 390 images replaced |
| Products archived in full | reprojected image | 4 R observations | +143 non-contributing images |
| Products added, net | reprojected image, browse | 20,303 to 20,584 | +281 |
| Arrays re-laid out | reprojected image | 1,560 / 19,913 | wrapping arrays start at the minimum longitude; no-data columns restored |
| Arrays with changed values | reprojected image | 2 / 19,913 | re-navigated, up to 0.07 I/F |
| Arrays with changed values | mosaic | 6 / 302 | corrected source images |
| Arrays with changed values | background-subtracted mosaic | all | background region 50 rows each side, not 50 and 51 |
| Camera corrected | mosaic | 231 / 302 | Wide Angle to Narrow Angle |
| Camera corrected | reprojected image | 17,392 / 19,913 | generic name to the actual camera |
| Prometheus targets | reprojected image | 8,755 to 772 | listed per product, not per observation |
| Pandora targets | reprojected image | 1,896 to 74 | listed per product |
| Star targets | all data products | 0 to 193 | 9 occultation stars |
| Longitude convention | reprojected image label, indexes | 1,253 labels | 0.00 / 359.98 to a wrapped range |
| Longitude range | mosaic label | 302 | maximum 360.00 to 359.98 |
| Time coordinates | all data products | 9,674 start, 8,721 stop | rounded to floored and ceiled |
| Inertial longitude | all metadata tables | every table | corrected |
| Mid-time | all metadata tables | every table | quantized to exact |
| Longitudinal resolution | all metadata tables | every table | radians to degrees per pixel |
| Columns added | reprojected-image table | 6 to 16 | orbit and satellite quantities |
| Columns added | mosaic table | 8 to 17 | orbit and satellite quantities, image index |
| Columns added | global indexes | 33/35/30 to 57/60/42 | orbit, satellite, quality, timestamp |
| Supplemental header | reprojected image | all | conventions stated |
| Supplemental roll | reprojected image | 161 | measured against one reference for every boresight |
| Collection added | miscellaneous | 1 | global index files moved out of document |
| Schema identifiers | xml_schema inventory | 5 | restyled to the published form |
| Kernel list | metakernel | 1,154 to 1,153 | all kernels NAIF-distributed |
| Kernel time span | metakernel label | 1 | nil to the observation span |
| User guide | document | 19 to 47 pages | PDF to PDF/A-1b, three new sections |
| Example programs | document | 0 to 5 | Python readers |
| Readme | bundle root | 0 to 1 | new |
| Rings dictionary | all data labels | 21,194 | 1.14.0.0 to 1.15.0.0 |
| Display direction | all data labels | 21,194 | Top to Bottom to Bottom to Top |
| Mission phase names | all data labels | 21,194 | upper case to title case |

---

## Appendix A. The 28 observations with corrected source images

The image count of each is the same in both copies; the images named and archived
differ. The images the review copy named were earlier images of the same
observation than the ones the mosaic was built from.

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

Four further observations, listed in section 4.3, had partly incorrect lists in
the review copy and now archive their complete image sequences.

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
`iss_276ri_hiresafrg001_prime_2`.

Prometheus is not named on `iss_082ri_fmonitor003_prime` or
`iss_207rf_fmovie001_prime`, where it lies 159 to 277 km beyond the radial limit,
nor on three further background-subtracted products whose valid longitudes no
longer include it.

Pandora is named on 9 mosaics the review copy did not name it on:
`iss_036rf_fmovie001_vims`, `iss_041rf_fmovie001_vims`,
`iss_041rf_fmovie002_vims`, `iss_105ri_tmapn45lp001_cirs_5`,
`iss_173rf_fmovie001_prime_1`, `iss_173ri_spokemov003_prime`,
`iss_196rf_fmovie003_prime`, `iss_241rf_fmovie001_prime`,
`iss_292rf_fmovie001_prime`. It is not named on `iss_087rf_fmovie003_prime`,
where it lies 73 km beyond the limit, nor on two `iss_105ri_tmapn45lp001_cirs`
background-subtracted products.

Stars are named on the 15 occultation observations:
`iss_172ri_betpegocc001_vims` (Scheat), `iss_172st_urgampeg001_uvis` (Algenib),
`iss_180ri_rcasocc001_vims` and `iss_191ri_rcasoccb001_vims` (R Cassiopeiae),
`iss_180ri_rlyrocc001_vims` and `iss_198ri_rlyrocc001_vims` (R Lyrae),
`iss_185ri_rhyaocc001_vims_1` and `_2` (R Hydrae),
`iss_194ri_mucepocc001_vims` (Herschel's Garnet Star),
`iss_196ri_betandocc001_vims` (Mirach), `iss_197ri_whyaocc001_vims` (W Hydrae),
`iss_201ri_l2pupocc001_vims_1`, `_2`, `iss_205ri_l2pupocc002_vims` and
`iss_206ri_l2pupocc002_vims` (L2 Puppis).

## Appendix C. Metadata table columns

The mosaic and background-subtracted mosaic tables carry the same columns as
below with `image_index` inserted after the co-rotating longitude, giving
seventeen. The review copy's mosaic table had eight columns and its
reprojected-image table six.

| Review copy (unit as labelled) | Delivered (unit) | Note |
|---|---|---|
| Corotating Longitude (deg) | `rings:corotating_ring_longitude` (deg) | renamed |
| Image Index (mosaic only, none) | `image_index` (none) | renamed |
| Mid-time SPICE ET (mosaic only, s) | `rings:observed_event_tdb` (s) | renamed; values were quantized |
| — (reprojected image) | `rings:observed_event_tdb` (s) | new |
| Inertial Longitude (deg) | `rings:inertial_ring_longitude` (deg) | renamed; values corrected |
| Radial Resolution (km/pixel) | `rings:radial_resolution` (km/pixel) | renamed |
| Angular Resolution (mosaic: deg/pixel, reprojected image: km/pixel) | `rings:longitudinal_resolution` (deg/pixel) | renamed; values were radians per pixel in both |
| — | `rings:incidence_angle` (deg) | new; constant per image |
| Phase Angle (deg) | `rings:phase_angle` (deg) | renamed |
| Emission Angle (deg) | `rings:emission_angle` (deg) | renamed |
| — | `core_radius` (km) | new |
| — | `longitude_ascending_node` (deg) | new; constant per image |
| — | `longitude_pericenter` (deg) | new; constant per image |
| — | `true_anomaly` (deg) | new |
| — | `corotating_longitude_prometheus` (deg) | new; constant per image |
| — | `radius_prometheus` (km) | new; constant per image |
| — | `corotating_longitude_pandora` (deg) | new; constant per image |
| — | `radius_pandora` (km) | new; constant per image |

Source-image table: the first column is renamed from "Source Image Index" to
`image_index`; the LIDVID column is unchanged.
