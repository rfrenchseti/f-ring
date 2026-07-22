##########################################################################################
# Compare the old .IMG files (CISSCAL 3.3 or 3.6) vs the new .IMG files (CISSCAL 4.0)
# by looking at the pixels in the general vicinity of the F ring and printing statistics
# about them.
##########################################################################################

import os

import matplotlib.pyplot as plt
import numpy as np

import oops
import oops.hosts.cassini.iss as iss

image_versions = ( # These are either the first or last images in the movie sequence
    ('N1466448701_1_CALIB-3.3.IMG', 'N1466448701_1_CALIB-4.0.IMG'), # ISS_000RI_SATSRCHAP001_PRIME
    ('N1479201492_1_CALIB-3.3.IMG', 'N1479201492_1_CALIB-4.0.IMG'), # ISS_00ARI_SPKMOVPER001_PRIME
    ('N1492052646_1_CALIB-3.3.IMG', 'N1492052646_1_CALIB-4.0.IMG'), # ISS_006RI_LPHRLFMOV001_PRIME
    ('N1493613276_1_CALIB-3.3.IMG', 'N1493613276_1_CALIB-4.0.IMG'), # ISS_007RI_LPHRLFMOV001_PRIME
    ('N1538168640_1_CALIB-3.3.IMG', 'N1538168640_1_CALIB-4.0.IMG'), # ISS_029RF_FMOVIE001_VIMS
    ('N1541012989_1_CALIB-3.3.IMG', 'N1541012989_1_CALIB-4.0.IMG'), # ISS_031RF_FMOVIE001_VIMS
    ('N1542047155_1_CALIB-3.3.IMG', 'N1542047155_1_CALIB-4.0.IMG'), # ISS_032RF_FMOVIE001_VIMS
    ('N1543166702_1_CALIB-3.3.IMG', 'N1543166702_1_CALIB-4.0.IMG'), # ISS_033RF_FMOVIE001_VIMS
    ('N1545556618_1_CALIB-3.3.IMG', 'N1545556618_1_CALIB-4.0.IMG'), # ISS_036RF_FMOVIE001_VIMS
    ('N1546748805_1_CALIB-3.3.IMG', 'N1546748805_1_CALIB-4.0.IMG'), # ISS_036RF_FMOVIE002_VIMS
    ('N1549801218_1_CALIB-3.3.IMG', 'N1549801218_1_CALIB-4.0.IMG'), # ISS_039RF_FMOVIE002_VIMS
    ('N1551253524_1_CALIB-3.3.IMG', 'N1551253524_1_CALIB-4.0.IMG'), # ISS_039RF_FMOVIE001_VIMS
    ('N1552790437_1_CALIB-3.3.IMG', 'N1552790437_1_CALIB-4.0.IMG'), # ISS_041RF_FMOVIE002_VIMS
    ('N1554026927_1_CALIB-3.3.IMG', 'N1554026927_1_CALIB-4.0.IMG'), # ISS_041RF_FMOVIE001_VIMS

    ('N1557020880_1_CALIB-3.6.IMG', 'N1557020880_1_CALIB-4.0.IMG'), # ISS_044RF_FMOVIE001_VIMS
    ('N1571435192_1_CALIB-3.6.IMG', 'N1571435192_1_CALIB-4.0.IMG'), # ISS_051RI_LPMRDFMOV001_PRIME
    ('N1577809417_1_CALIB-3.6.IMG', 'N1577809417_1_CALIB-4.0.IMG'), # ISS_055RF_FMOVIE001_VIMS
    ('N1578386361_1_CALIB-3.6.IMG', 'N1578386361_1_CALIB-4.0.IMG'), # ISS_055RI_LPMRDFMOV001_PRIME
    ('N1579790806_1_CALIB-3.6.IMG', 'N1579790806_1_CALIB-4.0.IMG'), # ISS_057RF_FMOVIE001_VIMS
    ('N1589589182_1_CALIB-3.6.IMG', 'N1589589182_1_CALIB-4.0.IMG'), # ISS_068RF_FMOVIE001_VIMS
    ('N1593913221_1_CALIB-3.6.IMG', 'N1593913221_1_CALIB-4.0.IMG'), # ISS_075RF_FMOVIE002_VIMS
    ('N1598806665_1_CALIB-3.6.IMG', 'N1598806665_1_CALIB-4.0.IMG'), # ISS_083RI_FMOVIE109_VIMS
    ('N1601485634_1_CALIB-3.6.IMG', 'N1601485634_1_CALIB-4.0.IMG'), # ISS_087RF_FMOVIE003_PRIME
    ('N1602717403_1_CALIB-3.6.IMG', 'N1602717403_1_CALIB-4.0.IMG'), # ISS_089RF_FMOVIE003_PRIME
    ('N1610364098_1_CALIB-3.6.IMG', 'N1610364098_1_CALIB-4.0.IMG'), # ISS_100RF_FMOVIE003_PRIME
)

print('FILENAME                              MIN         MAX   MEDIAN   MEAN   MED RATIO')
for img1, img2 in image_versions:
    fimg1 = os.path.join('/home/rfrench/DS/f-ring/compare_cisscal_versions', img1)
    fimg2 = os.path.join('/home/rfrench/DS/f-ring/compare_cisscal_versions', img2)
    obs1 = iss.from_file(fimg1, fast_distortion=True)
    obs2 = iss.from_file(fimg2, fast_distortion=True)
    bp = oops.Backplane.Backplane(obs1)
    bp_radii = bp.ring_radius('saturn:ring').mvals.filled(0)
    good_mask = (bp_radii < 140620) & (bp_radii > 139820)
    # plt.imshow(good_mask)
    # plt.show()

    o1d = obs1.data
    o2d = obs2.data
    o1d = o1d[good_mask]
    o2d = o2d[good_mask]
    ratio = o2d / o1d
    print(('%-30s' % img1),
          ('%11.3f' % np.min(ratio)), ('%11.3f' % np.max(ratio)),
          ('%7.3f' % np.median(ratio)), ('%7.3f' % np.mean(ratio)),
          ('%7.3f' % (np.median(o2d) / np.median(o1d))))
    # plt.imshow(ratio)
    # plt.show()
