# Critique of the Cassini ISS F Ring Mosaics bundle, 2026-09-27 build

Bundle reviewed: `/data/fring-bundles/pds4`, generated 2026-09-27 16:44 to
19:05 PDT from `main` at f3b0c97. Generation logs: `ERRORS.log` empty,
`WARNINGS.log` 102 lines, all of them moon-visibility notices.

This review was done without consulting the earlier critiques or the earlier
change log. A comparison with those documents is in section 8, which was
written after sections 1 to 7 were complete.

Sections 1 to 8 describe the 2026-09-27 build as reviewed. The author ruled
on every finding on 2026-09-28 and the bundle was rebuilt the same day;
section 9 records, finding by finding, what the rebuilt bundle does and does
not fix.

Contents: 305 observations (303 `iss_`, 2 `iosic_`), 20,584 reprojected
images, 42,405 labels, 190,785 files, 58 GB.

## 1. How the review was done

Six independent passes covered the whole bundle, each with scripts run over
every product:

| Pass | Coverage |
|---|---|
| Structure | XSD validation of all 42,405 labels against local copies of PDS 1O00, RINGS 1O00_1F00, DISP 1O00_1510, GEOM 1O00_19B0, CASSINI 1O00_1800; size and MD5 of all 148,380 referenced files; 11 inventories; 276,256 internal references; 42,391 table structures; 21,194 arrays |
| Label versus data | Every numeric and prose claim in the 21,194 observational labels recomputed from the arrays, tables and supplemental files; the three index tables checked row by row |
| Geometry | C-matrices, roll, RA/Dec, orbit model, corotation, moon positions, times, incidence trend, background fit, equivalent widths, array orientation |
| Browse | All 84,776 PNGs: validity, dimensions, stretch, ordering, overlays, checksums; visual sample of about 90 images |
| Documents | User guide against the bundle, sample programs executed, PDF/A, readme, context products against the PDS registry, kernels against NAIF, DOIs |
| Semantics | Every element name against its dictionary definition and permitted values; 57 comment sentence templates and 24 browse description templates checked for accuracy and consistency |

Tools: NASA PDS4 `validate` 4.2.0 (installed for this review and run on the
whole bundle, section 7), veraPDF 1.30.2, lxml XSD validation,
the PDS registry API and the NAIF kernel archive over the network.

## 2. Verdict

The bundle is structurally clean and its numbers are self-consistent. Three
findings are rated Major:

- The roll angle in every supplemental file is defined with its reference
  direction and sign reversed (section 3.1). A text correction.
- Two inventories cite version 2.0 of the Cassini ISS Data User's Guide, and
  the registry holds only version 1.1 (section 3.2). A text correction.
- The background-subtracted mosaics of this build contain 2.6 percent fewer
  valid longitudes than every earlier build, up to 38 percent in one
  product, and the dropped longitudes have complete background coverage
  (section 3.3). A data change the author has to accept or reverse.

Everything else is Minor or Cosmetic.

Outcome of these three in the 2026-09-28 rebuild: the roll sentence is
corrected, the coverage loss is reversed, and the `::2.0` reference is the
author's deliberate forward reference to the coordinated ISS delivery, so
the second Major rating is withdrawn. Section 9 has the measurements.

## 3. Major findings

### 3.1 The roll definition in the supplemental files and reprojected-image labels is reversed

All 20,584 `_reproj_img_suppl.txt` headers, and the `File/comment` of the
supplemental file area in all 20,584 reprojected-image labels, say:

> The roll is the angle of the camera X axis about the boresight, measured
> from the direction of increasing declination and positive towards
> increasing right ascension.

Computing that angle from the archived C-matrix (row 1 is the camera X axis,
row 3 the boresight) reproduces the printed roll in 2 of 20,584 files. The
printed value is the angle of the camera X axis measured from the direction
of increasing right ascension, positive towards increasing declination. That
definition reproduces every printed roll to within 0.0005 degrees, the
printing precision. It is also identical to the NAIF "twist" angle of the
RA/Dec/twist factorisation. The two definitions differ by 90 minus twice the
roll, which spans the full circle over the bundle.

Hand check on `data_reproj_img/iss_111rf_fmovie002_prime/1622022571n_reproj_img_suppl.txt`:
printed roll -44.3128; the stated definition gives +134.31; the corrected
definition gives -44.31.

The rest of the sentence is right: the rows are the camera axes in J2000,
row 3 is the boresight, and its RA/Dec match the header in all files. The
transpose hypothesis fails (the RA of column 3 differs from the header by up
to 180 degrees). Two review passes found this independently.

Consequence: a user who rebuilds the camera frame from RA, Dec and roll with
the stated definition gets a frame rotated about the boresight by 90 minus
twice the roll. The matrix is correct, so the fix is the sentence in the
suppl.txt template and the label template.

### 3.2 The document and miscellaneous inventories cite a version of the ISS Data User's Guide that does not exist

`document/collection_document.csv` and `miscellaneous/collection_miscellaneous.csv`
each contain

```
S,urn:nasa:pds:cassini_iss_saturn:document:iss-data-user-guide::2.0
```

The PDS registry (`/api/search/1/products/<lid>/all`) lists one version of
that product, `::1.1` (modification dates 2018-09-16 and 2023-07-20). The
RMS Node copy of the label at
`pds-rings.seti.org/pds4/bundles/cassini_iss/cassini_iss_saturn/document/iss-data-user-guide.xml`
carries `version_id` 1.1. The DOI 10.17189/1504135 quoted in the reference
comment of 42,392 labels resolves to the ds-view page for version 1.0.

The 42,392 `Internal_Reference` elements use the bare LID and are fine. Only
the two `S` inventory rows carry the version.

Consequence: a referential-integrity check of the two collections against
the registry fails on a member that does not exist.

### 3.3 The background-subtracted mosaics lost 81,125 valid longitudes, most of them fully covered

Across the 305 background-subtracted mosaics, 123,188 longitudes (4.3
percent) that are valid in the plain mosaic are absent. Of these, 75,708
(61 percent) have every pixel of both background windows valid in the
mosaic and 43,633 have one window empty. No retained longitude has an empty
window.

Compared with the review-1 build, and with every intermediate build from
2026-04-10 to 2026-09-09, which all carry the same background-subtracted
column set as review-1, this build drops 81,125 previously valid longitudes
in 263 products and gains 62 in 4. Restricting to the 296 products whose
plain mosaic is byte-identical between the two builds: 69,751 longitudes
lost (2.56 percent), 69,286 of them (99.3 percent) with complete 50-plus-50
background windows. The largest losses:

| Product | Valid in review-1 | Lost | Share |
|---|---|---|---|
| iss_245ri_hiresafrg002_prime | 2,958 | 1,129 | 38% |
| iss_036rf_fmovie002_vims | 16,045 | 5,084 | 32% |
| iss_197rf_fmovie002_prime | 10,685 | 3,197 | 30% |
| iss_036rf_fmovie001_vims | 18,000 | 4,670 | 26% |
| iss_007ri_azscnloph001_prime | 17,986 | 3,519 | 20% |
| iss_213rf_fmovie002_prime | 16,866 | 2,601 | 15% |
| iss_289rf_fmovie001_prime | 18,000 | 2,677 | 15% |

The dropped longitudes are only slightly noisier than the kept ones: over
the 174 products with at least 20 lost longitudes, the median ratio of the
standard deviation of the mosaic's background-window pixels in lost columns
to that in kept columns is 1.21 (10th percentile 1.01, 90th 1.80). In
`iss_197rf_fmovie002_prime` the ratio is 0.98: the 3,197 dropped
longitudes are indistinguishable from the 7,488 kept ones.

The labels say a longitude was removed "if insufficient data were available
to generate the model", after "statistically bad pixels (such as stars or
moons) were ignored". The pipeline's criterion is the count of pixels
remaining after its iterative masking, so the sentence is literally true,
but a reader concludes that only longitudes lacking background coverage are
missing. The rows that are present are computed correctly: the model is
exactly linear per column and fitted to lines 0-49 and 351-400 (section 6).

Consequence: any coverage, clump or longitude statistic drawn from the
background-subtracted products alone is biased against noisier longitudes,
and the equivalent-width profiles of the products above have gaps that the
review copy did not have. This is a decision for the author: accept the
reduced coverage and describe the criterion in the label and guide, or
restore the earlier coverage.

## 4. Minor findings

### 4.1 Dates, versions and citation metadata

**4.1.1 Publication year 2025 against 2026 dates.** All 42,405 labels have
`publication_year` 2025. In the same bundle every `modification_date` is
2026-09-27 or 2026-09-28, every `creation_date_time` is 2026-09-27/28, the
document label's `Document/publication_date` is 2026-09-28, the guide's
title page reads "Version 1.0, 2025", and both citations on that page and in
`readme.txt` say "(2025)". The LID embeds `rsfrench2025`. Both bundle DOIs
(10.17189/3tfh-th07 and 10.17189/ajhh-aj88) return 404. Whichever year the
DOI records carry, the labels and the guide will have to agree with it.

**4.1.2 Two modification dates for one "Initial version".** The generation
run crossed 00:00 UTC. 4,501 labels carry `modification_date` 2026-09-27
(the products of the 17 alphabetically first observations, in all six
data and browse collections) and 37,904 carry 2026-09-28. One label
(`data_reproj_img/iss_036rf_fmovie002_vims/1546712624n_reproj_img.lblx`)
carries 2026-09-27 while its files' `creation_date_time` and its browse
label say 2026-09-28. A single release date for all products would remove
the ambiguity.

**4.1.3 Two author lists for the user guide.** In
`document/user_guide/f-ring-mosaics-user-guide.lblx`,
`Citation_Information/List_Author` names French and Hedman;
`Document/List_Author` names French alone; the guide's title page and its
"Citing this User Guide" line name French alone. The DOI record for the
guide will carry one of these.

**4.1.4 Contributor sequence numbers 1, 2, 2, 2.** In all 16 labels with a
`List_Contributor` (bundle, 11 collections, document, 3 indexes) the four
DataCurators have sequence numbers 1, 2, 2, 2. The six ORCIDs pass the
ISO 7064 checksum and the three ROR identifiers resolve.

### 4.2 Background subtraction wording

**4.2.1 "From 750 to 1000 km" versus the rows used.** The fit uses lines
0-49 and 351-400, i.e. delta radius -1000 to -755 km and +755 to +1000 km:
for every candidate window, only that placement satisfies the least-squares
normal equations of the archived background, in 305 of 305 products, and any
window that includes the 750 km line satisfies them in none. The 305
background-subtracted labels say "the available data from 750 to 1000 km
closer to Saturn and 750 to 1000 km further from Saturn" (217 with the
default limits; the other 88 state their own limits the same way). The
guide, section 3.6, says "-1000 to -755 km" and "+755 to +1000 km". The
index label describes `bkgnd_lower_limit` as "radii between this and
core-1000 are used", which also reads as inclusive.

**4.2.2 "Insufficient data" as the reason for dropped longitudes.** See
section 3.3 for the data. Whatever the author decides there, the label
sentence "If insufficient data were available to generate the model" needs
a clause saying that the count is taken after the masking of statistically
bad pixels, so that a reader knows a fully covered longitude can be
removed. The guide's "statistically anomalous compared to nearby
longitudes" describes the mask better than the label's "statistically bad
pixels".

**4.2.3 The `B` note is applied to three products and not to six that were
thinned more.** `B` ("substantially fewer valid longitudes") is set for
`iss_007ri_hpmrdfmov001_prime`, `iss_007ri_lphrlfmov001_prime` and
`iss_083ri_fmonitor002_prime`, whose background-subtracted products keep 63,
68 and 69 percent of the mosaic's valid longitudes. Six products keep less
and carry no `B`: `iss_244ri_propretrg001_prime` (40%, 2,682 to 1,080
longitudes), `iss_178ri_egapmovmp001_prime` (43%), `iss_256ri_hiresafrg002_prime`
(52%), `iss_245ri_hiresafrg002_prime` (62%), `iss_199ri_egapmovmp001_prime`
(64%), `iss_036rf_fmovie002_vims` (68%). The three `B` codes also appear in
the plain-mosaic index and in 402 rows of the reprojected-image index, while
the `B` sentence exists only in the background-subtracted labels; every
other code reconciles with the prose.

**4.2.4 Source-image renumbering in 12 background-subtracted products.**
In 12 observations the background-subtracted `src_imgs.tab` lists fewer
images than the mosaic's (131 dropped in total; the list is always an
ordered subsequence) and both tables number from 0. From the first dropped
image onward the same `image_index` names different images in the two
products (for example `iss_105ri_tdifs20hp001_cirs`: index 0 is
`1615343803w` in the mosaic table and `1615344943w` in the
background-subtracted table; `iss_007ri_hpmrdfmov001_prime` drops 64 of
169). Each product is self-consistent, but nothing in the labels or the
guide says the two tables of one observation can differ. Joining the two
params tables on `image_index` silently mismatches source images.

### 4.3 Prometheus and Pandora statements

**4.3.1 Fifteen background-subtracted labels keep a moon whose longitude
was removed.** The moon sentence and `Target_Identification` are carried
from the mosaic into 87 background-subtracted labels. In 15 the column at
the moon's predicted longitude (plus or minus one column) is entirely -999
in the background-subtracted array; in 3 of those the exact column is
missing and the sentence says "its presence has been visually confirmed"
(`iss_072ri_spkhrlpdf001_prime`, `iss_080rf_fmovie005_prime`,
`iss_059rf_fmovie001_vims`). In 38 other cases where the moon's column was
removed, the sentence and target were dropped, which is the behaviour the
generation warnings describe. The visual confirmation was made on the plain
mosaic (guide section 3.4), and the background-subtracted label uses the
same wording "at a co-rotating longitude this mosaic covers".

**4.3.2 Four reprojected images omit the moon sentence for an edge
column.** In `iss_000ri_satsrchap001_prime/1466466941n`,
`iss_039rf_fmovie001_vims/1551267415n`, `iss_111rf_fmovie002_prime/1622035856n`
and `iss_093rf_fmovie003_prime/1605385758n` the predicted moon lies inside
the image's longitude range and within 1050 km of the core, in the first,
second or second-to-last column, and the label has neither sentence nor
target. The other 20,580 labels agree with the geometry (846 with, 19,734
without).

**4.3.3 Wording differs between product types.** Mosaic labels put the
sentence in `rings:description`; reprojected-image labels put it in the
`Observation_Area` comment and always say "has not been visually
confirmed", including images of mosaics whose label says "has been". The
guide explains the policy (confirmation is made on the mosaic); the labels
do not.

### 4.4 References and provenance

**4.4.1 Unused reprojected images reference the mosaic as their derived
product.** 143 reprojected images in `iss_199rf_fmovie002_prime` (97),
`iss_262rf_fmovie001_prime_12` (27), `iss_256ri_hiresafrg002_prime` (16)
and `iss_268rf_fmovie001_prime_1` (3) are absent from the mosaic's
`src_imgs.tab`. Their citation description correctly says "was not used to
create mosaic X, which covers only part of the observation", but the
`Observation_Area` comment says "associated with the mosaic named X" and the
`Reference_List` points at the mosaic and its background-subtracted twin
with `data_to_derived_product` and the comments "The mosaic without/with the
background subtracted". A further 131 images (the 12 observations of 4.2.4)
reference a background-subtracted product they do not contribute to.

**4.4.2 `data_to_derived_product` in both directions.** The 305
background-subtracted labels reference the plain mosaic, their source, with
`data_to_derived_product`. No mosaic or background-subtracted label carries
`Source_Product_Internal`; only the reprojected images do (pointing at the
external calibrated images). Provenance traversal by reference type
therefore runs backwards for these 305 links. The PDS reference-type list
for `Product_Observational` has no "data to source data product" value, so
this is a choice for the archivist.

**4.4.3 The metakernel is referenced by file name only.** All 20,584
reprojected-image labels give `geom:spice_kernel_file_name` `kernels.ker`
and no `Internal_Reference` to
`urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025:spice_kernels:kernels`,
although the GEOM dictionary says the LIDVID "should be given if one is
available". The 610 mosaic labels have no `geom:` class at all.

**4.4.4 Index products carry no context and no references.** The three
`Product_Ancillary` labels in `miscellaneous/` have no `Context_Area` (no
time range, mission, instrument, target) and no `Reference_List`, although
their `notes` field description says "See the User Guide for details".
`collection_miscellaneous.lblx` also has no `Context_Area`, yet its
inventory lists nine `S` members (the ISS user guide, two instruments, host,
mission, two rings, two moons) that none of its three products reference.
`collection_document.csv` lists the ISS user guide as `S` although the
guide product's `Reference_List` does not reference it. The
`spice_kernels` collection label has a `Context_Area` with only a
`Primary_Result_Summary` and two ring targets, while its single product has
the full set. Registry searches by time, target or mission will not find
the index products.

**4.4.5 Source products not yet in the registry.** All 20,584
`Source_Product_Internal` references point at
`urn:nasa:pds:cassini_iss_saturn:data_calibrated:<image>_calib::1.0`. The
registry returns 404 for the sampled LIDVID and no hits for the LID; the
`cassini_iss_saturn` bundle exists at `::1.1`. This depends on the
coordinated ISS delivery and is listed so the archivist can track it.

**4.4.6 The `validate` tool's context list does not contain L2 Puppis.**
`urn:nasa:pds:context:target:star.l02_pup` is referenced by 49 labels and
listed in the context inventory. The context product exists at
`pds.nasa.gov/data/pds4/context-pds4/target/star.l02_pup_1.0.xml`
(version 1.0, modification date 2026-09-03, "initial upon request from Mia
Mace"), but the `registered_context_products.json` shipped with
`validate` 4.2.0 (June 2026) has no entry for it, and the tool's `-u`
update option could not reach the registry from this machine. Every run of
the stock tool reports 49 `context_ref_not_found` errors that are not
defects of the bundle. The other 16 context LIDs are in the tool's list at
the versions the inventories cite, except that the tool lists
`mission.cassini-huygens` at 1.4 and the inventories cite 1.5, which exists
at PDS.

### 4.5 Browse products

**4.5.1 Vertical orientation is not stated.** Every browse PNG has line 400
(+1000 km) at the top and line 0 (-1000 km) at the bottom, matching the
data labels' `vertical_display_direction` "Bottom to Top" and flipped with
respect to array storage order. Row-profile correlation with the flipped
array is at least 0.99996 for every product; pixel-exact agreement for the
6,596 products without horizontal resampling. Neither the browse labels nor
guide sections 4.2.2 and 4.2.4 say which way radius runs in the PNG; a user
comparing the picture to the reshaped array sees it upside-down.

**4.5.2 The reprojected-image browse label misdescribes the small and
thumb overlays.** The label says "the small and thumb images carry the
image name alone". Every small and thumb image carries two lines: the image
name and, below it, "reproj img". The guide (4.2.2) describes this
correctly.

**4.5.3 Background-subtracted browse shows fewer longitudes with no
explanation.** For 302 of 305 observations the background-subtracted full
browse has more black columns than the mosaic browse (123,117 columns in
total; median 2.7 degrees per observation, maximum 106 degrees in
`iss_007ri_lphrlfmov001_prime`). The PNGs are faithful to the arrays, but
the browse label's description is identical to the mosaic browse label's
apart from the product word, so a reader flipping between the two images
sees parts of the ring vanish unexplained.

**4.5.4 "99.8% maximum value" for the whitepoint.** All 21,194 browse
labels describe the whitepoint as "the 99.8% maximum mosaic value" (or
"image value"). The value used is the 99.8th percentile of the valid pixels;
for `1874525875w` the maximum is 0.0483, 0.998 times the maximum is 0.0482,
and the 99.8th percentile is 0.0278, and only the percentile reproduces the
PNG. The guide says "99.8th-percentile value".

**4.5.5 Segment suffixes are absent from browse titles and
descriptions.** 118 of the 305 observation directories are segments of 42
split observations. Their browse titles name the observation without the
suffix, so `browse_mosaic/iss_253rf_fmovie001_prime_{1,2,3}` have titles
that differ only in the parenthesised image range, and 4,936
reprojected-image browse labels say "from observation
ISS_253RF_FMOVIE001_PRIME" with no segment. The segment appears in LIDs,
file names and the overlay text. The data labels follow the same
convention; the guide explains it, the browse labels do not.

### 4.6 Dictionary usage

**4.6.1 Wraparound on the corotating longitude pair.** 1,314
reprojected-image labels have `rings:minimum_corotating_ring_longitude`
greater than `rings:maximum_corotating_ring_longitude` (for example 356.46
and 26.44). The RINGS dictionary defines the *inertial* pair with the
clause "for ranges that cross the prime meridian, the minimum ring longitude
will have a value greater than the maximum"; the corotating pair's
definition has no such clause and reads "The minimum value in this product".
The labels' `rings:description` states the convention in prose, and the
index labels state that their columns of the same name hold a different
quantity (the valid-data range) for mosaics. Software that trusts the
dictionary would compute a negative span for these 1,314 products.

**4.6.2 Resolution attributes used as table fields.** The three index labels
use `rings:mean/minimum/maximum_radial_resolution` and
`rings:mean/minimum/maximum_longitudinal_resolution` as `Field_Character`
names (18 fields). The dictionary definition of each ends "Not intended to
be used as a table field". A field `name` is free text in PDS4, so nothing
binds the prefixed names to the dictionary, and the descriptions paraphrase
it accurately; the prefixes are decorative on all 78 index fields and 9
params-table fields per label.

**4.6.3 Undocumented sentinel and placeholder values.**
`cassini:filter_temperature` and `cassini:sensor_head_electronics_temperature`
are `-999.` with `unit="degC"` in two labels
(`data_reproj_img/iss_007ri_lphrlfmov001_prime/1493625366n` and
`1493638821n`); their definitions carry no sentinel clause (the
`optics_temperature_back` -999 in the 2,754 WAC labels is documented).
`cassini:calibration_lamp_state_flag` is `N/A` in 17,830 labels and
`cassini:telemetry_format_id` is `UNK` in 1,593; both validate, neither
value is described.

### 4.7 Supplemental file precision

RA and Dec are printed to six significant digits, so RA has three decimals
when RA is at least 100 degrees (15,351 files; 0.001 degree = 3.6 arcsec)
and Dec four decimals above 10 degrees (0.36 arcsec). The boresight from the
matrix differs from the degree values by up to 1.8 arcsec, more than one NAC
pixel (1.24 arcsec) in 2,413 files. The sexagesimal strings are given to
0.015 arcsec and the matrix to 1e-10, so the precision is there for a user
who reads those instead.

### 4.8 The SPICE metakernel

`kernels.ker` lists 1,153 bare file names (970 CK, 163 SPK, 12 binary PCK,
3 FK, 2 text PCK, 1 LSK, 1 SCLK, 1 IK) with no `PATH_VALUES`, so
`furnsh('kernels.ker')` fails unless all 1,153 files are in the working
directory. Every file exists at NAIF, but seven of the SPKs live under
`generic_kernels/spk/satellites/a_old_versions/` or `generic_kernels/spk/planets/`,
and neither `kernels.lblx` (whose `External_Reference` points at NAIF
documentation) nor guide section 4.5 says the kernels are not in the bundle
or where to fetch them. CK coverage 2004-06-19 to 2017-09-19 and SPK
coverage bracket every image; the three short CK gaps contain no image.

### 4.9 User guide

The guide is accurate against the bundle in every count, field name, file
name and constant checked (section 5). The remaining differences:

- **Stale excerpt.** Section 4.2.1 shows `creation_date_time`
  2026-09-09T22:09:20Z for `1622049830n_reproj_img.img`; the shipped label
  says 2026-09-28T00:33:52Z. Every other line of the two label excerpts
  matches.
- **Abbreviated excerpt without an ellipsis.** The suppl.txt excerpt shows a
  2-line preamble; the file has 7 lines (the five lines describing the
  matrix rows and the roll are omitted). Given 3.1, those lines will change
  anyway.
- **Mosaic table excerpt starts mid-file.** The `metadata_params` excerpt in
  4.2.3 starts at longitude 4.60 with no leading `[...]`; the file starts at
  0.00.
- **Quoted label wording differs slightly.** Section 3.3 quotes the
  navigation grade without the double quotes the labels use around the
  word; section 4.4.1 renders the note sentences with "corotating",
  "180° apart" and no quotes around "movies", where the labels write
  "co-rotating", "180 degrees apart" and `"movies"`; the M2 labels add "This
  mosaic consists of OBS chunk N. The other mosaic is available as obs_2",
  which the guide does not mention.
- **Dependencies of the sample programs are not stated.** Section 5.3.4 does
  not say the programs need `pds4_tools`, `numpy`, `matplotlib` and, for two
  of them, `pandas`; only the module docstrings do.

### 4.10 Small label inconsistencies

- The mosaic params-table `rings:incidence_angle` description says the value
  "will be the same for all corotating longitudes within a single mosaic",
  which is true, but the value is the last source image's (300 of 305; the
  other 5 differ by 0.001) while the source images span up to 0.016 degrees.
  The label does not say which image's value is used.
- 502 labels have `cassini:image_mid_time` 1 ms off the suppl.txt mid time
  because the two round a half-millisecond in opposite directions.
- 46,844 cells in the mosaic params tables differ by one unit in the last
  printed digit from the same longitude in the named source image's table
  (radial and longitudinal resolution, phase, emission, moon radii), against
  the description "taken from the given reprojected source image".
- The mosaic comment's "ring rotated under it for N seconds (H hours)" is
  the span of the rounded label times (floor of first start, ceil of last
  stop), 0.03 to 1.9 s longer than the actual first-start to last-stop span.
- Six background-subtracted products with tied largest coverage gaps report
  one of several equally valid "from A to B" endpoints (for example
  `iss_000ri_satsrchap001_prime`: "359.94 degrees from 166.90 to 166.82",
  while 12.58 to 12.50 is equally valid).
- Product labels give resolutions with `unit="km"` and `unit="deg"`; the
  index labels use `km/pixel` and `deg/pixel`; the guide uses km/pixel and
  °/pixel.

## 5. Cosmetic findings

- "F Ring" mid-sentence in 1,229 places (mosaic and browse citation
  descriptions, five collection descriptions, two document descriptions)
  against "F ring" in about 64,000.
- "co-rotating" in prose (212,664 uses) and "corotating" in element and field
  names and most field descriptions (593,494); every observational label
  contains both.
- "Data was missing" (20,584 reprojected-image special-constant comments)
  against "the data were missing" (610 mosaic comments).
- `core_radius` descriptions say "long_peri" while the field is
  `longitude_pericenter`; the reprojected-image `true_anomaly` description
  says "long_peri" and the mosaic one says "longitude_pericenter".
- "Albers et al. (2012), Table 3, fit #2" in labels; "fit 2" in the guide.
- "arbitrarily chosen to be a time near Cassini's arrival at Saturn" for an
  epoch (2007-01-01) two and a half years after orbit insertion.
- `<!--A value of -1 ...` in 1,072 labels lacks the space after `<!--`.
- The three `C-matrix ... column` fields have empty `description` elements.
- `spice_kernels/kernels.lblx` has the one-word title "Metakernel" and no
  author or contributor list, the only product label without them.
- `File/records` appears in 3 of 11 collection labels; `</Modification_History>`
  is indented 7 spaces in two collection labels.
- "dataset" in the index labels, "data set" in the readme and the guide;
  "indexes" and "indices" both in the guide.
- Thumbnail overlays occupy up to a quarter of the 100x100 image with no
  backing box, hiding the outer 20 to 25 percent of the radial range; the
  text is legible only after upscaling.
- PDF layout: page 15 holds nine lines, page 21 only Figure 6, pages 34 and
  35 have large blank areas because Figures 8 and 9 float to the next page.
- The readme's "(e.g. journal) citation" parenthesis.

## 6. Checked and sound

Every item below was verified over the whole bundle unless a count says
otherwise.

**Structure.** All 42,405 labels parse and validate against the five XSDs
(a deliberately corrupted label failed 7 of 8 corruption tests; the eighth,
an enumeration, is Schematron-only). Namespace declarations, `schemaLocation`
pairs and `xml-model` PIs agree with the elements used in every label. Every
one of the 190,785 files is a label or is named by exactly one label in its
directory; 148,372 `file_size` and 148,379 `md5_checksum` values match the
files. 42,405 distinct LIDs follow the bundle:collection:product pattern with
product IDs equal to the label basenames. The 11 inventories list exactly
the 42,393 products present, with matching `records`, sizes and checksums;
`bundle.lblx` lists the 11 collections with the right reference types. All
276,256 `Internal_Reference`, 20,584 `Source_Product_Internal` and 42,388
`Local_Internal_Reference` elements resolve, and every reference type is in
the Schematron list for its context. Mosaic, background-subtracted,
reprojected and browse labels cross-reference each other consistently in all
42,388 products. All 42,391 fixed-width tables have the declared header
length, record count, record length, field layout and data types, are pure
ASCII with LF endings, and their header lines equal the label field names.
All 21,194 arrays are exactly 401 by N by 4 bytes, contain no NaN or
infinity, none is entirely -999, and values span -0.07 to 1.34 I/F. The 32
labels run through `validate -R pds4.label` (bundle, 11 collections, every
product type) pass apart from the L2 Puppis registry entry.

**Label against data.** Every mosaic column is bit-identical to the column
of the source image named in its params table (2,888,402 columns, 1,314
wrapping images), which also proves the stated array ordering. Params-table
records equal the columns with any valid pixel in all 21,194 products; the
first and last columns of every reprojected image are valid. Label
mean/min/max phase, emission, incidence and resolutions equal the table
statistics; ring radii equal core plus and minus 1000 km; the "valid data
for X degrees spanning Y from A to B" and "covering X degrees of inertial
longitude" sentences reproduce from the tables in all 21,194 labels; source
image counts, title image names, times and SCLKs equal the `src_imgs` tables.
Start and stop times are floor and ceiling of the supplemental times;
`cassini:` DOY times, `exposure_duration`, `image_number` and SCLK counts
agree with the supplemental files; `rings:observed_event_tdb` equals the
UTC mid time converted with the correct leap seconds and TDB periodic term
to 1 ms. Camera (NAC or WAC) agrees with the image name suffix in all
20,584 images and no mosaic mixes cameras. Mission phases agree with the
dates. Navigation and background quality words equal the index letters
(good 224 / fair 57 / poor 24; good 214 / fair 71 / poor 20). The three index
tables agree with the labels in every row and column, including circular
means and ranges where angles wrap.

**Geometry.** C-matrices are orthonormal to 1.7e-10 with determinant +1.
The corotation relation holds to 0.0005 degrees over 26.3 million table
rows with the epoch 2007-01-01T00:00:00 UTC (ET 220881665.184, checked by
independent leap-second arithmetic). Core radius, true anomaly, pericenter
(24.2 + 2.70025 deg/day) and node (15.0 - 2.68778 deg/day) reproduce every
row from the J2000 epoch. The corotation rate 581.964 deg/day matches the
J2-J6 mean motion at a = 140,224 km, 2.3 km from the stated semimajor axis.
Prometheus and Pandora radii lie inside their known orbital ranges; their
drift rates in the corotating frame are +5.32 and -9.23 deg/day; a single
mean-motion fit gives 587.285 and 572.789 deg/day. Incidence follows the
season smoothly from 65.4 degrees in 2004 through 89.8 at equinox to 63.3 at
solstice with no outliers. 131 Prometheus detections are all at line below
200 and 10 Pandora detections above, confirming "Line 0 is the innermost
row". The background model is exactly linear per column; the fit window is
lines 0-49 and 351-400 in all 305 products; the per-column means of the
background windows are centred on zero. Equivalent widths over lines 150-250
have median 1.5 km and 1st to 99th percentiles 0.48 to 11.5 km, with no
negative per-observation median.

**Browse.** All 84,776 PNGs open, are 8-bit greyscale, and have the
dimensions the labels state (reprojected: max(V,800) by 401 and
max(floor(W/10),400) by 400, with V the number of valid columns, in all
20,584). The stretch is blackpoint max(min,0), whitepoint 99.8th percentile,
gamma 0.5, quantised as floor(256x): regenerating it reproduces the full
PNG to within one grey level for all 6,596 products without horizontal
resampling. Wrapped images are re-ordered to start at 0 degrees; no-data
longitudes are omitted; every no-data pixel is black; the full mosaic
browse always shows all 18,000 columns. Overlays are present on every
med/small/thumb and absent from every full image; mosaic overlay text
matches the label and guide. All file sizes, MD5s and `browse_to_data`
references are correct. A visual sample of about 90 images showed no gross
navigation error, nothing blank or saturated, and no seam at the 360/0
boundary.

**Documents.** The guide's directory tree, file names, Table 3/4/5 field
names and order (42, 57, 60), the mosaic lists for every note code, the
counts 20,584 / 305 / 20,441 / 143 / 42 SPOKEMOV, the observation-name
codes SI and PIE, the orbit constants (which reproduce a metadata row to
the printed precision), the browse size rules, the note-code list, and the
figure captions all match the bundle. The TOC page numbers match all 55
headings. All 26 URLs in the guide and the 17 cited in labels return HTTP
200. The four sample programs run against the bundle with `pds4_tools` 1.4
and reproduce Figures 7 to 9 (same ten images, same separations); all 13
functions listed for `mosaic_utils.py` exist. The PDF passes veraPDF
PDF/A-1b (129 rules, 113,938 checks, 0 failures), embeds all 15 fonts, has
56 bookmarks and 251 link annotations with no broken destinations, and its
`document_standard_id` values for the six files match their bytes.
`readme.txt` is 7-bit ASCII with LF endings and agrees with the bundle
label. The context inventory equals the set of 17 context LIDs referenced
anywhere in the bundle; their versions are the latest at PDS; the names,
types and alternate designations of all 13 targets match the context
products. The xml_schema LIDVIDs equal those in the dictionary labels at
PDS. All 1,153 kernels exist at NAIF; `sat393.bsp`, `de438.bsp` and
`cpck15Dec2017.tpc` named in prose are listed.

**Semantics.** Every enumerated value in use is in its Schematron list;
every `unit` attribute is permitted for its attribute; every numeric
attribute lies within its dictionary facets; `disp:` and `rings:`
local references match the array identifiers; `rings:radial_resolution`
and `longitudinal_resolution` follow the dictionary rule (statistics in the
label, values in the table); the 57 comment templates and 24 browse
templates reproduce every number from the data.

## 7. PDS4 validate results

`validate` 4.2.0 (release of 2026-06-15) was installed for this review and
run on the whole bundle with `-e lblx -R pds4.bundle`, label validation
(XSD and Schematron), data-content validation and referential integrity
all on. The run took 2 h 35 min once the machine was otherwise idle.

Result (report header: Version 4.2.0, 2026-09-28T02:58:18Z):

| Check | Result |
|---|---|
| Products validated | 42,405 |
| Products passed | 42,356 |
| Products failed | 49 |
| Errors | 49, all `error.label.context_ref_not_found` |
| Warnings | 0 |
| Referential integrity checks | 42,405 passed, 0 failed |

The 49 failures are the labels that reference
`urn:nasa:pds:context:target:star.l02_pup`: `bundle.lblx`, the three data
collection labels, and the 45 observational labels of the four L2 Puppis
occultation observations (`iss_201ri_l2pupocc001_vims_1` and `_2`,
`iss_205ri_l2pupocc002_vims`, `iss_206ri_l2pupocc002_vims`: 8 mosaic-type
and 37 reprojected-image labels). The context product exists at PDS
(version 1.0, 2026-09-03); the tool's bundled registry snapshot predates
it (section 4.4.6). No other Schematron, content or referential message
was produced.

The tool's referential-integrity pass covers references inside the bundle
only. It did not test the `iss-data-user-guide::2.0` inventory rows
(section 3.2), the 20,584 calibrated-image source references (4.4.5) or
the five `xml_schema` members against the registry, so their status rests
on the registry queries reported in those sections.

The full 22 MB report is in the review session's scratch directory
(`validate-out/full_bundle_report.txt`); it is not committed.

## 8. Comparison with the earlier critiques and change log

Written after sections 1 to 7 were complete, from the 2026-09-03 and
2026-09-10 critiques and the change log dated 2026-09-10.

### 8.1 Findings in this review that no earlier review reported

- **3.1, roll definition reversed.** The 2026-09-10 review (its B1) asked
  for one roll convention and a stated definition. The regenerated bundle
  has one convention in all 20,584 files, so that finding is closed, but the
  definition written into the files and labels describes the other angle.
  This defect did not exist before the fix.
- **3.3, background-subtracted coverage loss.** Specific to this build. The
  2026-09-10 change log predicted that "seven others lose whole longitude
  columns" and that the background change would move pixels by "a maximum
  around 1.7e-4"; the regenerated products lose columns in 263 of 302 and
  the per-product maxima reach 6.6e-3. Neither earlier critique could see
  this because the background stage had not been re-run.
- **3.2, the `::2.0` reference.** Both earlier critiques listed the ISS Data
  User's Guide `::2.0` as an expected forward reference to the coordinated
  ISS delivery. This review checked the registry: it holds `::1.1`, dated
  2023-07-20, as the current version. Whether a 2.0 is coming is a question
  for the Node; until it exists the reference is dangling.
- 4.1.2 two modification dates; 4.2.3 the `B` note; 4.2.4 the renumbered
  source lists (a consequence of the 2026-09-03 B9 fix, which the 2026-09-10
  review saw as correct); 4.3.1 moons kept in 15 background-subtracted
  labels whose column was removed; 4.4.2 `data_to_derived_product` in both
  directions; 4.4.3 no reference to the metakernel product; the
  `Context_Area` half of 4.4.4; 4.5.3 unexplained coverage difference
  between the two browse products of an observation; 4.5.4 the "99.8%
  maximum" wording; 4.5.5 segment suffixes absent from browse titles; 4.6.1
  wraparound outside the dictionary definition of the corotating pair;
  4.6.2 resolution attributes used as table fields; 4.7 six-digit RA/Dec;
  4.2.2 the "insufficient data" wording; the `image_mid_time` rounding and
  duration rounding in 4.10; and the cosmetic items on "Data was",
  "long_peri", "near Cassini's arrival", the `<!--A` comment, the empty
  C-matrix field descriptions, `File/records`, "dataset", thumbnail
  overlays and the readme parenthesis.

### 8.2 Findings reported earlier and still present

- **Browse vertical orientation unstated** (2026-09-03 B20 "no browse label
  says which way is up"): still unstated; 4.5.1.
- **Reprojected-image small and thumb overlay text** (2026-09-10 B3): the
  mosaic labels were corrected; the reprojected-image label still says
  "the image name alone" and the images carry a second line; 4.5.2.
- **Moon sentence omitted on edge columns** (2026-09-10 B5 named three of
  the four images in 4.3.2): unchanged.
- **Publication year 2025 against 2026 dates** (2026-09-10 B7): unchanged,
  pending the DOI record; 4.1.1.
- **Background limits "from 750 to 1000 km" in the labels** (2026-09-10 G2
  noted the label wording; only the guide was corrected): 4.2.1.
- **Index products without a `Reference_List`** (2026-09-10 B10 note): 4.4.4.
- **Context products listed as secondary members three ways** and the
  miscellaneous inventory naming products nothing references (2026-09-10
  B10 note; the document-collection rows were a PDS reviewer's request):
  4.4.4 and the structure pass.
- **Bundle description omits the background-subtracted mosaics and browse
  products** (2026-09-10 B10 note): unchanged.
- **Metakernel has no path values and cannot be loaded as shipped**
  (2026-09-03 B25): 4.8. The absence of a header block was ruled acceptable;
  the path question was not ruled on.
- **Tied-gap endpoints** (2026-09-03 B11, related item): 4.10.
- **Mosaic metadata differing from the source image in the last digit**
  (2026-09-10 B10 note, 12 values): now measured at 46,844 cells; 4.10.
- **Guide excerpts stale** (2026-09-03 G3, 2026-09-10 G3): one creation date
  and one abbreviated preamble remain; 4.9.
- **"F Ring" versus "F ring" and "co-rotating" versus "corotating"**
  (2026-09-10 B9): partly corrected (the index labels' "F-ring" is gone);
  section 5.
- **readme line of 86 characters** (2026-09-10 B9): unchanged, and
  previously judged acceptable.
- **Cassini DOY times without `Z`** (2026-09-03 B28): unchanged, valid.
- **`rings:longitudinal_resolution` printed with three significant digits
  for small values** (2026-09-03 B28): unchanged, not re-examined here.

### 8.3 Findings reported earlier and closed in this build

Confirmed closed by this review: the two-convention roll (as a convention),
the wraparound record-order description (2026-09-10 B2), the 0.02-degree
span discrepancy (B4), the "within the valid data range" wording (B5), the
nil metakernel time range (B6), the reprojected-index caveat (B8), the B9
wording list except the items in 8.2, the mosaic-index field description
(G1), the three background-row ranges in the guide (G2), the G3 claims, and
the G4 presentation items that the guide comparison could check (bookmarks
for M1 to M4, captioned and numbered tables, landscape footers, defined
IMGID and OBSID, wraparound formulas in the section the text cites). From
the 2026-09-03 list: B1, B2, B3 (matrix), B5, B9, B11 (full-circle span),
B12, B13, B15, B17, B18, B19, B21, B22, B23, B24, B25 (kernel names), B26
(stale PDF, PDF/A), B27, B28 (line endings, tabs, "Initial version.",
component order, `xmlns:pds`, readme creation time, sexagesimal padding),
G1 to G15 and the editorial list as far as the guide comparison covers it.

### 8.4 Items re-found here that the author has already decided

Recorded so they are not raised again: the two author lists on the document
product (4.1.3); the contributor sequence numbers (4.1.4); the mosaic
incidence value being the last image's (4.10); the -999 temperatures copied
from the source labels (4.6.3); the per-image "has not been visually
confirmed" wording (4.3.3); no header block in `kernels.ker`; CISSCAL 4.0;
the per-product labels carrying no author list.

### 8.5 The two change logs

The change log dated 2026-09-10 was written before the background stage
was re-run and before this regeneration. Compared with the change log
written for this review, from measurements on the regenerated bundle:

- **Background-subtracted arrays.** The earlier log says the change is
  "small and systematic ... a maximum around 1.7e-4 in I/F" and that
  "seven others lose whole longitude columns". Measured: median pixel
  difference 1.7e-7, per-product maxima median 4.4e-5 and largest 6.6e-3,
  and 263 products lose 81,125 longitudes. The earlier statement is wrong
  on the column loss and should be replaced.
- **Roll definition.** The earlier log restates the header text as the
  definition of the roll. That text is reversed (3.1). The final log
  describes the header as shipped and points to this finding.
- **ISS Data User's Guide `::2.0`.** The earlier log calls it a forward
  reference to the coordinated ISS delivery. The final log records that the
  registry holds `::1.1` as current and that `::2.0` does not resolve.
- **Reprojected-image arrays.** The earlier log partitions the 1,560
  re-laid-out arrays as 1,251 wrapping arrays and 309 with sentinel columns
  restored; the final log partitions the same 1,560 by what changed in the
  array (1,065 with the two longitude blocks swapped, 495 widened by
  inserted empty columns, 112 of which also wrap). Both sums agree; the
  final log's partition is the one a reader can verify from the arrays.
- **Source-image attribution.** The earlier log describes 28 observations
  with replaced images and four with partly incorrect lists, tested by
  exposure windows. The final log adds the direct column-content
  measurement: 420,401 misattributed longitudes in 36 mosaics, 108,065 of
  them from images the review copy did not archive.
- **Browse pixel values.** The earlier log says the images are unchanged
  apart from the labels; the final log records that most PNGs changed grey
  levels under the corrected stretch and that the background-subtracted
  browse images show the dropped longitudes.
- **Omitted from the earlier log and added:** the split modification dates,
  the CRLF and tab formatting defects of the review copy, the missing XML
  declaration in the kernel label, the curator comment removed from the
  SPICE collection label, the boresight and roll changes in the supplemental
  files with their sizes, the review copy's excerpts that never matched the
  review copy, and the PDF/A status of the review copy's guide (PDF/A-2b,
  failing 1b).
- Everything else in the earlier log agrees with the measurements here
  within the difference between the 2026-09-09 and 2026-09-27 builds (the
  three added observations and the quality-word counts).

The earlier log's appendices (the 28 observations with their image ranges,
the mosaic-level moon changes, the column table) were re-derived from the
two bundles and are correct; they are carried into the final log.

## 9. Status after the 2026-09-28 rebuild

The author ruled on every finding above on 2026-09-28, the background stage
was re-run with the column-rejection rule of the earlier builds restored,
and the bundle was regenerated (`/data/fring-bundles/pds4`, 2026-09-28
13:56 to 16:17 PDT, from `critique_2026_09_27_fixes`; `ERRORS.log` empty,
`WARNINGS.log` 81 lines, all moon-visibility notices; 305 observations,
20,584 reprojected images, 42,405 labels, 190,785 files, 59 GB). Every
statement below was measured on that bundle.

### 9.1 The three Major findings

| Finding | Status |
|---|---|
| 3.1 Roll definition reversed | Fixed |
| 3.2 `iss-data-user-guide::2.0` | Not a defect (author ruling) |
| 3.3 Background-subtracted coverage loss | Fixed |

**3.1 Fixed.** All 20,584 supplemental headers and all 20,584
reprojected-image labels now read "measured from the direction of
increasing right ascension and positive towards increasing declination",
which is the definition the printed numbers follow. No file carries the old
sentence. The matrices are unchanged.

**3.2 Not a defect.** The author ruled that `::2.0` is a deliberate forward
reference to the version of the Cassini ISS Data User's Guide that the
coordinated ISS delivery will publish, in the same way as the
`data_calibrated` source references of 4.4.5. The Major rating of section
3.2 is withdrawn; the two inventory rows are listed with the other
references that resolve outside the bundle.

**3.3 Fixed.** The in-loop "too few pixels" recheck in
`mosaics/ring/ring_model_bkgnd.py` counts the original image mask again
instead of the accumulated mask that commit 52cb449 introduced, and the
background stage was re-run. Column by column, across the 302 products
common to the review copy, the valid-longitude set now matches the review
copy: 9 longitudes dropped in 5 products and 66 added in 6, plus
`iosic_276rb_complitb4001_si`, whose plain mosaic itself changed (7,202
longitudes; section 6.2 of the change log). The background-subtracted
index's `num_valid_longitudes` differs from the review copy in 11 of 302
rows, against 263 before.

Background subtraction now drops 47,216 longitudes of the 2,888,402 valid
in the plain mosaics (1.6 percent, in 272 products, median 52 per product),
against 123,188 (4.3 percent) in the 2026-09-27 build. Of those 47,216,
303 (0.6 percent) have all 50 rows of both background windows valid in the
plain mosaic, against 61 percent in the 2026-09-27 build: what is dropped is
now almost entirely longitudes whose background windows really are
incomplete.

### 9.2 Minor findings

| Finding | Status | Evidence in the rebuilt bundle |
|---|---|---|
| 4.1.1 Publication year | Fixed | `publication_year` 2026 in all 42,405 labels; guide title page "Version 1.0, 2026"; both citations "(2026)"; document `publication_date` 2026-09-28. The LID keeps `rsfrench2025` by decision. Both bundle DOIs still return 404 (external) |
| 4.1.2 Two modification dates | Fixed | `modification_date` 2026-09-28 in all 42,405 labels; the run stayed inside one UTC day |
| 4.1.3 Two author lists for the guide | Fixed in the template | Hedman was removed from `Citation_Information/List_Author` on the author's instruction (2026-09-30), so both lists, the guide's title page and its "Citing this User Guide" line name French alone. He remains an author of the bundle itself, in the bundle and collection labels, the index labels, the readme's citation and the guide's "Citing this bundle" line. Reaches the bundle at the next regeneration |
| 4.1.4 Contributor sequence 1, 2, 2, 2 | Author decision, intentional | unchanged |
| 4.2.1 "750 to 1000 km" | Fixed | "755 to 1000 km" appears in 269 of the 305 background-subtracted labels, on one side or both; the products with adjusted margins state their own limits the same way; no label says "750 to 1000"; the index labels now define `bkgnd_lower_limit`/`bkgnd_upper_limit` as the ring bounds with the background running from -1000 km "up to but not including" the lower limit and from "beyond" the upper limit through +1000 km |
| 4.2.2 "Insufficient data" | Fixed | all 305 labels now say pixels statistically anomalous compared with nearby longitudes were masked before the fit and that the minimum counts are applied "after that masking" |
| 4.2.3 The `B` note | Fixed | `B` is now on the 6 products with the lowest retention (43, 64, 65, 66, 68, 72 percent of the mosaic's valid longitudes); the next lowest is 73 percent. The note is also mirrored into the plain-mosaic label of each |
| 4.2.4 Source-image renumbering | Fixed | the field description now reads "The index number specific to this mosaic for the source image"; guide section 4.2.3 states that the two tables of one observation can number differently (7 observations, 102 images, in this build) and says to match on LIDVID or longitude |
| 4.3.1 Moons on removed columns | Fixed | with the coverage restored, one product remains (`iss_243rf_fmovie001_prime_2`, whose predicted Prometheus longitude falls about one column outside the retained set while the label says "has been visually confirmed"), against 15 products and 3 confirmations before |
| 4.3.2 Four edge-column images | Fixed | all four labels now carry the moon sentence and `Target_Identification`; the edge-of-data rejection was removed from `_image_has_satellite` |
| 4.3.3 Wording differs between product types | Fixed | the reprojected-image comment now states that visual confirmation is made on the mosaic, is recorded only in the mosaic label, and applies to whichever image supplied the pixels |
| 4.4.1 Unused images reference the mosaic | Fixed | the 143 unused images carry no `data_to_derived_product` reference to either mosaic product and say so in their description; the 102 images whose longitudes the subtraction removed reference the mosaic only, with a sentence saying why; guide section 3.1.6 explains both cases. Residue: the `Observation_Area` comment of the 143 still says "is associated with the mosaic named X" |
| 4.4.2 `data_to_derived_product` direction | Open, archivist decision | unchanged |
| 4.4.3 Metakernel referenced by file name | Fixed | `Internal_Reference` to `...:spice_kernels:kernels` with `reference_type` `geometry_to_SPICE_kernel` in all 20,584 reprojected-image labels and in all 610 mosaic-type labels, which gained the `geom:` class |
| 4.4.4 Index products without context | Fixed | the three index labels have a `Context_Area` (time range, investigation, observing system, targets) and a `Reference_List`; `collection_miscellaneous.lblx` and `collection_spice_kernels.lblx` gained a `Context_Area` |
| 4.4.5 Source products not in the registry | Open, external | depends on the coordinated ISS delivery |
| 4.4.6 L2 Puppis absent from the tool's list | Open, tool side | see 9.4 |
| 4.5.1 Vertical orientation unstated | Fixed | all 21,194 browse labels state that the rows run with delta radius increasing upward, top row +1000 km, which reverses the array storage order; guide sections 4.2.2 and 4.2.4 say the same |
| 4.5.2 Small and thumb overlays | Fixed | all 20,584 reprojected-image browse labels describe both lines of the overlay |
| 4.5.3 Background-subtracted browse | Fixed | all 305 labels say the images can show fewer longitudes, as additional black columns, than the browse images of the mosaic |
| 4.5.4 "99.8% maximum value" | Fixed | "99.8th percentile" in all 21,194 browse labels |
| 4.5.5 Segment suffixes in browse | Fixed as ruled | "(segment N)" added to the title and descriptions of the 4,936 reprojected-image products of split observations and of their 4,936 browse products. Mosaic browse titles keep the observation name without the segment: an observation is the full Cassini observation, and the segment is carried by the LID and file name |
| 4.6.1 Corotating-longitude wraparound | Deferred | the RINGS dictionary definition will be changed; no bundle change |
| 4.6.2 Resolution attributes as field names | Open | unchanged; the names remain decorative |
| 4.6.3 Sentinels and placeholders | Open in part | the -999 temperatures are carried from the source labels by decision; `N/A` and `UNK` remain undescribed |
| 4.7 Supplemental precision | Fixed | right ascension, declination and roll are printed to six decimals (0.0036 arcsec) |
| 4.8 The metakernel | Open | `kernels.ker` still has no `PATH_VALUES`, and neither its label nor the guide says the kernels are not in the bundle or where NAIF distributes them |
| 4.9 User guide | Fixed in part; see 9.3 | |
| 4.10 Small label inconsistencies | See 9.3 | |

### 9.3 The 4.9 and 4.10 lists item by item

Fixed in 4.10:

- **Exposure mid-times.** `cassini:image_mid_time` now equals the
  supplemental file's mid time in all 20,584 labels (0 mismatches, against
  502): both are taken from the PDS3 `IMAGE_MID_TIME`.
- **Mosaic table cells against the source image.** Radial resolution,
  longitudinal resolution, phase and emission now agree exactly with the
  named source image's table in every row of all 610 mosaic-type tables
  (0 differing cells, against 46,844): the generator copies the source
  image's float64 values instead of rounding the mosaic's float32 copy.
  Two residues: `rings:incidence_angle` differs from the source images by
  design, which the field description states, and the moon radii of
  `iosic_276rb_complitb3001_si` still differ in the last digit (11,396
  cells, up to 0.001 km) because they are not among the copied fields.
- **The rotation time in the mosaic comment.** Computed from the
  millisecond-precision start and stop times instead of the rounded label
  times (`iss_000ri_satsrchap001_prime` now says 52,800 seconds where the
  label times give 52,801).
- **Resolution units.** Both product-label descriptions now say the radial
  and longitudinal resolutions are per source-image pixel. The `rings:`
  attributes keep `unit="km"` and `unit="deg"`: the RINGS 1.15 Schematron
  permits only length units on the radial attributes and angle units on
  the longitudinal ones, so `km/pixel` cannot be used there.

Open in 4.10:

- The `rings:incidence_angle` field description still does not say which
  source image's value the mosaic carries.
- The six background-subtracted products with tied largest gaps still
  report one of several equally valid endpoint pairs.
- The index labels still use `km/pixel` and `deg/pixel` as field units
  while the product labels use `km` and `deg` (see above).

In 4.9, fixed: the year on the title page and in both citations; the
background windows as 755 km with the worked 910 km example; a paragraph on
the source-image renumbering; a paragraph (3.1.6) on the images that did
not contribute to a mosaic and the references their labels carry; the
browse orientation in 4.2.2 and 4.2.4; the masking clause and the per-mosaic
minimums in 3.6; "corotating" throughout, matching the labels; "indexes"
for index files and "indices" only for array indices.

The remaining 4.9 items were fixed in the guide sources on 2026-09-28 and
the PDF was rebuilt (48 pages, veraPDF PDF/A-1b PASS) and installed in
`pds4_bundle_gen/templates/`. They reach the bundle at the next
regeneration; the shipped 2026-09-28 PDF still carries the old text.

- Section 4.2.1 shows the shipped `creation_date_time` (2026-09-28T21:45:52Z),
  and the surrounding paragraph now says that a data file's creation time
  records when that copy was written and so differs between builds, which
  keeps the excerpt true of any build.
- The supplemental-file excerpt carries the whole seven-line preamble,
  including the roll definition, and the six-decimal right ascension,
  declination and roll of the shipped files.
- The `metadata_params` excerpt in 4.2.3 has a leading `[...]`.
- Section 5.3.4 lists the dependencies: Python 3 with `pds4_tools`, `numpy`
  and `matplotlib`, plus `pandas` for the DataFrame functions, which import
  it on use.
- The two quoted label strings that differed are now quoted as the labels
  write them: the navigation grade in double quotes, and the M1 to M4 and B
  note sentences in full, with `"movies"` quoted, "180 degrees apart" spelled
  out, the chunk sentences included, and the mosaic-label form of the B note
  given alongside the background-subtracted one.

### 9.4 Cosmetic findings

Fixed: "F Ring" mid-sentence (none remain outside title case); "co-rotating"
(0 uses, "corotating" everywhere); "Data was missing" ("the data were
missing" in all 21,194 labels that carry the sentence); the `long_peri`
references in the `core_radius` and `true_anomaly` descriptions; "fit #2" in
the guide as well as the labels; the missing space in `<!-- A value of -1`;
the three empty `C-matrix ... column` descriptions; the one-word title of
`kernels.lblx`; `File/records` in the three collection labels that carried
it and the `</Modification_History>` indent (all 11 collection labels are
now identical in both respects); "dataset" in the index labels, the readme
and the guide.

Open: the epoch sentence ("arbitrarily chosen to be a time near Cassini's
arrival at Saturn" for 2007-01-01); the thumbnail overlays covering a
quarter of the 100x100 image; the readme's "(e.g. journal) citation"
parenthesis; the PDF's page-break gaps. `kernels.lblx` still carries no
author list, which matches the other per-product labels by decision.

### 9.5 PDS4 validate on the rebuilt bundle

`validate` 4.2.0 was run again on the rebuilt bundle with the same options
(`-e lblx -R pds4.bundle`, label, content and referential validation), on
2026-09-28 from 18:54 to 21:03 PDT (2 h 8 min).

| Check | Result |
|---|---|
| Products validated | 42,405 |
| Products passed | 42,356 |
| Products failed | 49 |
| Errors | 49, all `error.label.context_ref_not_found` |
| Warnings | 0 |
| Referential integrity checks | 42,405 passed, 0 failed |

The result is identical to section 7: the same 49 labels that reference
`urn:nasa:pds:context:target:star.l02_pup`, which the tool's bundled context
list does not contain (4.4.6). Nothing the rebuild changed introduced a
Schematron, content or referential message, and no message of section 7 was
resolved by it, because there were none to resolve. The report is in the
session's scratch directory (`validate-out/full_bundle2_report.txt`) and is
not committed.
