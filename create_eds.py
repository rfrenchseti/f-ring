##########################################################################################
# Create various types of equivalent depth for the occultations. Also correlate with
# FMOVIE mosaics.
#
# By default we create:
#   Occultation Name
#   Ring Event Start Time
#   Ring Event End Time
#   Ring Event Mean Time
#   RevNo
#   Observation Name
#   Star Name
#   Direction
#   Minimum Wavelength
#   Maximum Wavelength
#   Data Quality Score
#   Minimum Longitude
#   Maximum Longitude
#   Mean Longitude
#   Ring Elevation
#   Lowest Detectable Opacity
#   Highest Detectable Opacity
#   Prometheus Distance
#   Equivalent Tau
#   Core30 ED
#   Core50 ED
#   Full ED
#
# If --compare-ew is specified, we also create:
#   Before ISS ObsName
#   Before ISS Delta
#   After ISS ObsName
#   After ISS Delta
##########################################################################################

import argparse
from collections import namedtuple
import csv
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd

import pdsparser

import f_ring_util
import prometheus_util


OCC_RADIAL_RESOLUTION = 1 # Radial resolution of occultations
OCC_SLUSH = 1000          # Range around 140220 to look for peak tau
ED_CORE_HW30 = 30         # Half-width around core to compare to Albers 2012 core
ED_CORE_HW50 = 50         # Half-width around core to compare to
ED_FULL_HW = 500          # Half-width around core to compare to Albers 2012 full F ring


cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
   cmd_line = []

parser = argparse.ArgumentParser()

parser.add_argument('--instrument', type=str, default='UVIS',
                    help='Limit to one instrument (UVIS or VIMS)')

parser.add_argument('--occ-use-orbit', action='store_true', default=False,
                     help='Use F ring orbit instead of peak tau for core location')

parser.add_argument('--compare-ew', action='store_true', default=False,
                     help='Compare occultations with closest mosaic')
parser.add_argument('--ew-inner-radius', type=int, default=139470,
                    help='The inner radius of the range')
parser.add_argument('--ew-outer-radius', type=int, default=140965,
                    help='The outer radius of the range')

parser.add_argument('--output-csv-filename', type=str,
                    help='Name of output CSV file')
parser.add_argument('--plot-results', action='store_true', default=False,
                    help='Plot the core occultation for each observation')
parser.add_argument('--plot-aggregate', action='store_true', default=False,
                    help='Plot the combined core occultations for all observations')
parser.add_argument('--save-plots', action='store_true', default=False,
                    help='Same as --plot-results but save plots to disk instead')

f_ring_util.add_parser_arguments(parser)

arguments = parser.parse_args(cmd_line)


# Where to find the calibrated occultation files
OCC_DIR = {'UVIS': '/seti/fring_occultations_1km_uvis',
           'VIMS': '/seti/fring_occultations_1km_vims'}

# Occultations that need to be ignored
BAD_OCC = ['UVIS_HSP_2004_280_XI2CET_E', # Completely bad data
           'UVIS_HSP_2006_206_ZETOPH_E', # Completely missing data
           'UVIS_HSP_2006_252_ALPTAU_I', # Completely bad data
           'UVIS_HSP_2006_268_DELVIR_E', # Completely missing data
           'UVIS_HSP_2006_268_DELVIR_I', # Completely missing data
           'UVIS_HSP_2006_269_ALPSCO_I', # Full < Core
           'UVIS_HSP_2006_285_ALPVIR_I', # Completely bad data
           'UVIS_HSP_2006_286_GAMLUP_E', # Insufficient F ring data
           'UVIS_HSP_2006_306_MUPSA_I',  # Completely missing data
           'UVIS_HSP_2006_318_MUPSA_I',  # Completely missing data
           'UVIS_HSP_2006_364_DELPER_E', # Insufficient F ring data
           'UVIS_HSP_2007_063_BETPSA_I', # Completely missing data
           'UVIS_HSP_2007_079_BETSGR_E', # Completely missing data
           'UVIS_HSP_2007_082_ALPAUR_I', # Completely bad data
           'UVIS_HSP_2007_129_LAMSCO_I', # Insufficient F ring data
           'UVIS_HSP_2008_014_IOTCEN_E', # Insufficient F ring data
           'UVIS_HSP_2008_026_KAPCEN_I', # Insufficient F ring data
           'UVIS_HSP_2008_026_IOTCEN_E', # Completely missing data
           'UVIS_HSP_2008_026_SAO205839_I', # Completely missing data
           'UVIS_HSP_2008_038_BETLUP_I', # Completely missing data
           'UVIS_HSP_2008_040_GAMCNC_E', # Completely missing data
           'UVIS_HSP_2008_058_BETHYA_I', # Completely missing data
           'UVIS_HSP_2008_095_ALPSEX_E', # Completely bad/missing data
           'UVIS_HSP_2008_095_ALPSEX_I', # Completely bad/missing data
           'UVIS_HSP_2008_156_THEHYA_E', # Completely missing data
           'UVIS_HSP_2008_156_THEHYA_I', # Completely missing data
           'UVIS_HSP_2008_163_THEHYA_E', # Completely missing data
           'UVIS_HSP_2008_163_THEHYA_I', # Completely missing data
           'UVIS_HSP_2008_238_GAMCRU_I', # Completely bad data
           'UVIS_HSP_2008_268_ALPARA_I', # Lots of negative data
           'UVIS_HSP_2008_327_GAMCRU_I', # Completely bad/missing data
           'UVIS_HSP_2008_332_THEHYA_E', # Completely bad/missing data
           'UVIS_HSP_2008_332_THEHYA_I', # Completely bad/missing data
           'UVIS_HSP_2009_012_ALPCRU_I', # Big dropout next to core
           'UVIS_HSP_2009_012_GAMCRU_I', # Completely bad/missing data
           'UVIS_HSP_2009_058_EPSCAS_E', # Full < Core
           'UVIS_HSP_2009_062_THEHYA_E', # Completely bad/missing data
           'UVIS_HSP_2009_062_THEHYA_I', # Completely bad/missing data
           'UVIS_HSP_2010_148_LAMAQL_I', # Completely missing data
           'UVIS_HSP_2012_204_ZETCMA_E', # Completely missing data
           'UVIS_HSP_2012_204_ZETCMA_I', # Completely missing data
           'UVIS_HSP_2012_292_GAMCOL_E', # Completely bad data
           'UVIS_HSP_2012_293_ALPVIR_I', # Many dropouts
           'UVIS_HSP_2012_324_ALPLYR_I', # Full < Core
           'UVIS_HSP_2013_114_BETLIB_E', # Completely missing data
           'UVIS_HSP_2013_114_BETLIB_I', # Completely missing data
           'UVIS_HSP_2013_121_LAMTAU_E', # Completely missing data
           'UVIS_HSP_2013_121_LAMTAU_I', # Completely missing data
           'UVIS_HSP_2013_189_DELCEN_I', # Full < Core
           'UVIS_HSP_2015_049_KAPORI_I', # Dropout in inside skirt
           'UVIS_HSP_2016_153_DELSCO_I', # Full < Core
           'UVIS_HSP_2016_177_ALPSCO_E', # Completely missing data
           'UVIS_HSP_2016_177_ALPSCO_I', # Completely missing data
           'UVIS_HSP_2016_218_ALPSCO_I', # Completely bad data
           'UVIS_HSP_2016_243_ALPSCO_E', # Completely bad/missing data
           'UVIS_HSP_2016_243_ALPSCO_I', # Completely bad/missing data
           'UVIS_HSP_2016_267_ALPSCO_E', # Completely missing data
           'UVIS_HSP_2016_277_SIGSGR_I', # Very negative data
           'UVIS_HSP_2016_287_ALPSCO_E', # Completely missing data
           'UVIS_HSP_2016_287_ALPSCO_I', # Completely missing data
           'UVIS_HSP_2016_306_EPSSGR_E', # Completely missing data
           'UVIS_HSP_2016_306_EPSSGR_I', # Completely missing data
           'UVIS_HSP_2016_306_KAPSCO_I', # Full < Core
           'UVIS_HSP_2016_353_BETCRU_I', # Calibration too negative
           'UVIS_HSP_2016_360_ALPPAV_I', # Completely missing data
           'UVIS_HSP_2017_228_BETORI_E', # Insufficient inside data

           'VIMS_2005_217_OMICET_E', # Insufficient data
           'VIMS_2005_217_OMICET_I', # Insufficient data
           'VIMS_2005_232_ALPSCO_E', # Full < Core
           'VIMS_2006_252_ALPTAU_I', # Bad calibration base
           'VIMS_2006_268_DELVIR_E', # Insufficient data
           'VIMS_2006_268_DELVIR_I', # Insufficient data
           'VIMS_2006_285_RLEO_I',   # Bad oscillations in skirts
           'VIMS_2006_301_CWLEO_I',  # Full < Core
           'VIMS_2007_082_ALPAUR_I', # Bad calibration
           'VIMS_2007_105_RHYA_I',   # Insufficient data
           'VIMS_2007_163_ALPORI_I', # Bad oscillations and bad calibration
           'VIMS_2008_140_RLEO_E',   # Full < Core
           'VIMS_2008_155_CWLEO_E',  # Insufficient data
           'VIMS_2008_160_GAMCRU_I', # Bad calibration
           'VIMS_2008_162_CWLEO_I',  # Bad oscillations
           'VIMS_2008_174_GAMCRU_I', # Bad calibration base
           'VIMS_2008_202_GAMCRU_I', # Bad calibration base
           'VIMS_2008_209_GAMCRU_I', # Bad calibration base
           'VIMS_2008_210_BETGRU_I', # Bad calibration base
           'VIMS_2008_320_GAMCRU_I', # Bad calibration base
           'VIMS_2008_328_EPSMUS_E', # Insufficient data
           'VIMS_2008_328_GAMCRU_I', # Bad calibration base
           'VIMS_2008_343_GAMCRU_I', # Bad calibration
           'VIMS_2009_013_ALPTRA_E', # Extremely noisy
           'VIMS_2009_013_ALPTRA_I', # Extremely noisy
           'VIMS_2009_035_TXCAM_I',  # Full < Core
           'VIMS_2009_053_GAMCRU_I', # Oscillations
           'VIMS_2009_077_GAMCRU_E', # Bad calibration base
           'VIMS_2009_077_GAMCRU_I', # Bad calibration base
           'VIMS_2009_129_ALPAUR_E', # Bad calibration base
           'VIMS_2009_129_ALPAUR_I', # Bad calibration base
           'VIMS_2009_239_ALPORI_I', # Bad calibration base
           'VIMS_2010_154_OMICET_E', # Noisy
           'VIMS_2010_154_OMICET_I', # Insufficient data
           'VIMS_2010_205_OMICET_I', # Full < Core
           'VIMS_2012_315_ALPCET_I', # Bad calibration base
           'VIMS_2012_324_ALPLYR_I', # Full < Core
           'VIMS_2013_030_RCAS_I',   # Bad calibration base
           'VIMS_2013_033_WHYA_I',   # Full < Core
           'VIMS_2013_091_RCAS_I',   # NaN
           'VIMS_2013_149_RCAS_I',   # Full < Core
           'VIMS_2013_161_RCAS_I',   # Bad calibration base
           'VIMS_2013_229_WHYA_E',   # Insufficient data
           'VIMS_2013_241_BETAND_I', # Bad calibration base
           'VIMS_2013_327_L2PUP_E',  # Bad calibration base
           'VIMS_2014_022_L2PUP_E',  # Bad calibration base
           'VIMS_2014_022_L2PUP_I',  # Bad calibration base
           'VIMS_2014_067_ALPLYR_E', # Noisy
           'VIMS_2014_067_ALPLYR_I', # Noisy
           'VIMS_2014_067_RLYR_E',   # Bad calibration base
           'VIMS_2014_067_RLYR_I',   # Bad calibration base
           'VIMS_2014_198_ALPLYR_I', # Noisy
           'VIMS_2015_073_XOPH_I',   # Clipped
           'VIMS_2015_273_30PSC_E',  # Bad data
           'VIMS_2015_273_30PSC_I',  # Bad data
           'VIMS_2016_069_RAQL_E',   # Bad data & clipped
           'VIMS_2016_069_RAQL_I',   # Clipped
           'VIMS_2016_173_2CEN_I',   # Bad calibration base
           'VIMS_2016_201_ALPSCO_I', # Bad calibration base
           'VIMS_2016_244_XOPH_E',   # Clipped
           'VIMS_2016_244_XOPH_I',   # Clipped
           'VIMS_2016_268_XOPH_E',   # Clipped
           'VIMS_2016_268_XOPH_I',   # Clipped
           'VIMS_2016_284_LAMVEL_E', # Bad calibration base
           'VIMS_2016_284_LAMVEL_I', # Bad calibration base
           'VIMS_2016_294_LAMVEL_I', # Negative skirt
           'VIMS_2016_331_ETACAR_E', # Extreme noise
           'VIMS_2016_331_ETACAR_I', # Extreme noise
           'VIMS_2017_001_GAMCRU_I', # Bad calibration for inner skirt Full >> Core
           'VIMS_2017_007_VYCMA_E',  # Negative ED
           'VIMS_2017_007_VYCMA_I',  # Negative ED
           'VIMS_2017_050_VYCMA_E',  # Full < Core
           'VIMS_2017_050_VYCMA_I',  # Full < Core
           'VIMS_2017_073_LAMVEL_I', # Totally bad data
           'VIMS_2017_094_LAMVEL_I', # Bad calibration for inner skirt Full >> Core
           'VIMS_2017_104_ALPORI_E', # Full < Core
           'VIMS_2017_120_ALPCMA_I', # Full < Core
           'VIMS_2017_245_GAMCRU_I', # Full < 0
           'VIMS_2017_251_GAMCRU_I', # Full < 0
           ]


ew_ring_lower_limit = int((arguments.ew_inner_radius -
                          arguments.radius_inner_delta -
                          arguments.ring_radius) / arguments.radius_resolution)
ew_ring_upper_limit = int((arguments.ew_outer_radius -
                          arguments.radius_inner_delta -
                          arguments.ring_radius) / arguments.radius_resolution)

##########################################################################################

def plot(title=None,
         title_before=None, mosaic_img_before=None, ew_slice_before=None,
         title_after=None, mosaic_img_after=None, ew_slice_after=None):
    if not arguments.plot_results and not arguments.save_plots:
        return

    fig = plt.figure(figsize=(12, 8))
    if mosaic_img_before is not None or mosaic_img_after is not None:
        ax1 = fig.add_subplot(311)
    else:
        ax1 = fig.add_subplot(111)
    l1, = plt.plot(radii_slush-max_radius, occ_slush, color='black', label='Occultation')
    plt.ylabel('Optical depth')
    # plt.ylim(-0.01, 0.5)
    ax1.tick_params(axis='y', labelcolor='black')

    if ew_slice_before is not None or ew_slice_after is not None:
        ax2 = ax1.twinx()
        ax2.tick_params(axis='y', labelcolor='red')
        xrange = np.arange(-ED_FULL_HW, ED_FULL_HW+1, arguments.radius_resolution)
        if ew_slice_before is not None:
            l2, = ax2.plot(xrange, ew_slice_before/np.max(ew_slice_before),
                           color='orange', lw=1, label='ISS Before')
        if ew_slice_after is not None:
            l2, = ax2.plot(xrange, ew_slice_after/np.max(ew_slice_after),
                           color='red', lw=1, label='ISS After')
        plt.ylabel('Relative I/F', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        plt.legend()
    plt.xlabel('Core offset (km)')
    plt.xlim(-ED_FULL_HW, ED_FULL_HW)

    if mosaic_img_before is not None:
        ax = fig.add_subplot(312)
        plt.imshow(mosaic_img_before[::-1, :],
                   extent=(0, 360,
                           arguments.ew_inner_radius - arguments.ring_radius,
                           arguments.ew_outer_radius - arguments.ring_radius),
                   aspect='auto',
                   cmap='gray', vmin=0, vmax=255)
        plt.title(title_before)
    if mosaic_img_after is not None:
        ax = fig.add_subplot(313)
        plt.imshow(mosaic_img_after[::-1, :],
                   extent=(0, 360,
                           arguments.ew_inner_radius - arguments.ring_radius,
                           arguments.ew_outer_radius - arguments.ring_radius),
                   aspect='auto',
                   cmap='gray', vmin=0, vmax=255)
        plt.title(title_after)

    if title is None:
        plt.suptitle(root)
    else:
        plt.suptitle(f'{root} ({title})')
    plt.tight_layout()
    if arguments.save_plots:
        if not os.path.exists('plots_occs'):
            os.mkdir('plots_occs')
        plt.savefig(f'plots_occs/{root}.png')
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()
    else:
        plt.show()


if arguments.compare_ew:
    # When --compare-ew is specified, we read in and cache the relevant metadata
    # for all of the mosaics
    print('Reading mosaic metadata')
    MosaicData = namedtuple('MosaicData', 'midtime obsid longitudes')
    mosaic_data_list = []
    for obs_id in f_ring_util.enumerate_obsids(arguments):
        if '166RI' in obs_id or '237RI' in obs_id:
            print(f'{obs_id:30s} SKIPPING')
            continue

        (bkgnd_sub_mosaic_filename,
         bkgnd_sub_mosaic_metadata_filename) = f_ring_util.bkgnd_sub_mosaic_paths(
            arguments, obs_id)

        if (not os.path.exists(bkgnd_sub_mosaic_filename) or
            not os.path.exists(bkgnd_sub_mosaic_metadata_filename)):
            print('NO FILE', bkgnd_sub_mosaic_filename,
                  'OR', bkgnd_sub_mosaic_metadata_filename)
            continue

        with open(bkgnd_sub_mosaic_metadata_filename, 'rb') as bkgnd_metadata_fp:
            metadata = pickle.load(bkgnd_metadata_fp, encoding='latin1')

        # Co-rotating WRT F ring core
        longitudes = np.degrees(metadata['longitudes'])
        good_long = longitudes >= 0
        midtime = np.mean(metadata['ETs'][good_long])
        mosaic_data_list.append(MosaicData(midtime=midtime, obsid=obs_id,
                                           longitudes=longitudes))

if arguments.output_csv_filename:
    csv_fp = open(arguments.output_csv_filename, 'w')
    writer = csv.writer(csv_fp)
    hdr = ['Occultation Name',
           'Ring Event Start Time',
           'Ring Event End Time',
           'Ring Event Mean Time',
           'RevNo',
           'Observation Name',
           'Star Name',
           'Direction',
           'Minimum Wavelength',
           'Maximum Wavelength',
           'Data Quality Score',
           'Minimum Inertial Longitude',
           'Maximum Inertial Longitude',
           'Mean Inertial Longitude',
           'Minimum Co-Rot Longitude',
           'Maximum Co-Rot Longitude',
           'Mean Co-Rot Longitude',
           'Ring Elevation',
           'Lowest Detectable Opacity',
           'Highest Detectable Opacity',
           'Prometheus Distance',
           'Equivalent Tau',
           'Core30 ED',
           'Core50 ED',
           'Full ED']
    if arguments.compare_ew:
        hdr += ['Before ISS ObsName',
                'Before ISS Delta',
                'After ISS ObsName',
                'After ISS Delta']
    writer.writerow(hdr)

# Read in the summary of occultations and extract the .TAB files for the given instrument
data_pd = pd.read_csv(os.path.join(OCC_DIR[arguments.instrument], 'data.csv'),
                      header=0, index_col='OPUS ID')
manifest_pd = pd.read_csv(os.path.join(OCC_DIR[arguments.instrument], 'manifest.csv'),
                          header=0, index_col='OPUS ID')
manifest_pd = manifest_pd.loc[manifest_pd['File Path'].apply(lambda x: x[-4:]) == '.TAB']
data_pd = data_pd.join(manifest_pd, how='inner')

ed_core30_hw_pix = int(ED_CORE_HW30 / OCC_RADIAL_RESOLUTION)
ed_core50_hw_pix = int(ED_CORE_HW50 / OCC_RADIAL_RESOLUTION)
ed_full_hw_pix   = int(ED_FULL_HW   / OCC_RADIAL_RESOLUTION)

tau_list = []
ed_core30_list = []
ed_core50_list = []
ed_full_list = []
core_occ_list = []
core_radius_list = []
by_star_list = []

for pd_index, row in data_pd.iterrows():
    filename = row['File Path']
    dq = row['Data Quality Score']
    _, base = os.path.split(filename)
    root = base.replace(f'_TAU{OCC_RADIAL_RESOLUTION:02d}KM.TAB', '')
    root = root.replace(f'_TAU_{OCC_RADIAL_RESOLUTION:02d}KM.TAB', '')
    if root in BAD_OCC:
        continue
    # if base > 'UVIS_HSP_2008_343': # Limit to Albers 2012 data set
    #     continue
    full_filename = os.path.join(OCC_DIR[arguments.instrument], base)
    label_filename = full_filename.replace('.TAB', '.LBL')

    segments = base.split('_')
    inst = segments[0]
    if (arguments.instrument is not None and arguments.instrument != inst and
        arguments.instruments != 'BOTH'):
        continue

    label = pdsparser.PdsLabel.from_file(label_filename).as_dict()

    star = f'{segments[4]}({segments[5]})'
    date = f'{segments[2]}-{segments[3]}'
    by_star_list.append(f'{star} {date}')
    col_num = 3 if inst == 'VIMS' else 4
    date_col_num = 9 if inst == 'VIMS' else 7
    occ_pd = pd.read_csv(full_filename, header=None)
    # Extract a large region around the nominal semimajor axis
    occ_pd_slush = occ_pd[(occ_pd[0] >= 140220-OCC_SLUSH) & (occ_pd[0] <= 140220+OCC_SLUSH)]
    radii_slush = np.array(occ_pd_slush[0])
    long_slush = np.array(occ_pd_slush[1]) # Inertial WRT ascending node
    occ_slush = np.array(occ_pd_slush[col_num])
    date_slush = np.array(occ_pd_slush[date_col_num])
    if radii_slush[0] != 140220-OCC_SLUSH or radii_slush[-1] != 140220+OCC_SLUSH:
        print(f'{root}: Insufficient F ring data')
        continue
    # Find the index of the highest tau or the computed F ring core
    if arguments.occ_use_orbit:
        # Use the nominal semimajor axis to figure out the date and longitude to
        # use to compute the actual F ring radius
        sma_idx = np.argmin(np.abs(radii_slush-140221))
        et_at_sma = date_slush[sma_idx]
        long_at_sma = long_slush[sma_idx]
        core_radius = f_ring_util.fring_radius_at_longitude(et_at_sma,
                                                            np.radians(long_at_sma))
        # Find the radius in the data closest to the known radius of the core
        core_idx = np.argmin(np.abs(radii_slush-core_radius))
        print(core_radius, core_idx)
    else:
        # If not using the orbit, just find the location of the highest tau
        core_idx = np.argmax(occ_slush)
    long_at_core = long_slush[core_idx] # Inertial
    max_radius = radii_slush[core_idx]
    if core_idx-ed_full_hw_pix < 0 or core_idx+ed_full_hw_pix+1 >= len(occ_slush):
        print(f'{root}: Core max too close to edge')
        plot('Bad edge')
        continue

    # Extract +/- 50 km and compute the derived tau
    core_occ   =   occ_slush[core_idx-ed_core50_hw_pix:core_idx+ed_core50_hw_pix+1]
    core_radii = radii_slush[core_idx-ed_core50_hw_pix:core_idx+ed_core50_hw_pix+1]
    core_occ_list.append(core_occ)                 # For aggregate plotting
    core_radius_list.append(core_radii-max_radius) # For aggregate plotting
    avg_tau = -np.log(np.mean(np.exp(-core_occ)))
    if avg_tau < 0:
        print(f'{root}: {dq} Avg Tau < 0')
        plot('Avg Tau < 0')
        continue

    # Compute equivalent depth for +/- 30 km and +/- 50 km
    ed_core30_occ = occ_slush[core_idx-ed_core30_hw_pix:core_idx+ed_core30_hw_pix+1]
    ed_core30 = np.sum(ed_core30_occ) * OCC_RADIAL_RESOLUTION
    ed_core50_occ = occ_slush[core_idx-ed_core50_hw_pix:core_idx+ed_core50_hw_pix+1]
    ed_core50 = np.sum(ed_core50_occ) * OCC_RADIAL_RESOLUTION
    ed_full_occ = occ_slush[core_idx-ed_full_hw_pix:core_idx+ed_full_hw_pix+1]
    ed_full = np.sum(ed_full_occ) * OCC_RADIAL_RESOLUTION
    if ed_full < ed_core30:
        print(f'{root} {dq}: Full ED {ed_full:6.3f} < Core ED {ed_core30:6.3f}')
        continue
    full_long_data = long_slush[core_idx-ed_full_hw_pix:core_idx+ed_full_hw_pix+1]
    full_date_data = date_slush[core_idx-ed_full_hw_pix:core_idx+ed_full_hw_pix+1]
    print(f'{root:22s} {dq}: Derived tau {avg_tau:6.3f} / Core30 ED {ed_core30:6.3f} / '
          f'Core50 ED {ed_core50:6.3f} / Full ED {ed_full:6.3f}')
    tau_list.append(avg_tau)
    ed_core30_list.append(ed_core30)
    ed_core50_list.append(ed_core50)
    ed_full_list.append(ed_full)

    min_inertial_long = np.min(full_long_data)
    max_inertial_long = np.max(full_long_data)
    mean_inertial_long = np.mean(full_long_data)
    if min_inertial_long < 90 and max_inertial_long > 270:
        print(f'{root}: WARNING co-inertial long range crosses 0')

    # Compute co-rotating longitude
    full_corot_data = np.degrees(f_ring_util.fring_inertial_to_corotating(
        np.radians(full_long_data), full_date_data))
    min_corot_long = np.min(full_corot_data)
    max_corot_long = np.max(full_corot_data)
    mean_corot_long = np.mean(full_corot_data)
    if min_corot_long < 90 and max_corot_long > 270:
        print(f'{root}: WARNING co-rot long range crosses 0')
    ed_mid_time = np.mean(full_date_data)

    # Find the closest mosaic before and after the occultation
    greyscale_img = {'before': None, 'after': None}
    ew_slice = {'before': None, 'after': None}
    if arguments.compare_ew:
        min_corot_long_int  =  int(min_corot_long / arguments.longitude_resolution)
        max_corot_long_int  =  int(max_corot_long / arguments.longitude_resolution)
        mean_corot_long_int = int(mean_corot_long / arguments.longitude_resolution)
        # Number of pixels on each side of core for full F ring
        r500_pix = int(ED_FULL_HW / arguments.radius_resolution)
        best_obsid = {'before': None, 'after': None}
        best_midtime_diff = {'before': 1e38, 'after': 1e38}
        for midtime, obsid, longitudes in mosaic_data_list:
            if (longitudes[int(min_corot_long_int)] >= 0 and
                longitudes[int(max_corot_long_int)] >= 0):
                if (midtime > ed_mid_time and
                        midtime-ed_mid_time < best_midtime_diff['after']):
                    best_midtime_diff['after'] = midtime - ed_mid_time
                    best_obsid['after'] = obsid
                if (midtime < ed_mid_time and
                        ed_mid_time-midtime < best_midtime_diff['before']):
                    best_midtime_diff['before'] = ed_mid_time - midtime
                    best_obsid['before'] = obsid
        print(f'{root}: Closest '
              f'{best_obsid["before"]} {best_midtime_diff["before"]/86400:.2f} days / '
              f'{best_obsid["after"]} {best_midtime_diff["after"]/86400:.2f} days')

        long_diff = {'before': None, 'after': None}
        for suffix in ('before', 'after'):
            days = best_midtime_diff[suffix] / 86400

            if days > 30:
                best_obsid[suffix] = None
                continue

            # Read in the mosaic and metadata
            (bkgnd_sub_mosaic_filename,
             bkgnd_sub_mosaic_metadata_filename) = f_ring_util.bkgnd_sub_mosaic_paths(
                arguments, best_obsid[suffix])
            with open(bkgnd_sub_mosaic_metadata_filename, 'rb') as bkgnd_metadata_fp:
                metadata = pickle.load(bkgnd_metadata_fp, encoding='latin1')
            longitudes = metadata['longitudes']
            mosaic_et = metadata['ETs'][min_corot_long_int]
            print('Mosaic ET', mosaic_et)
            print('Phase', np.degrees(metadata['phase_angles'][min_corot_long_int]))
            mosaic_inertial_long = np.degrees(f_ring_util.fring_corotating_to_inertial(
                longitudes[min_corot_long_int], mosaic_et))
            long_diff[suffix] = min_corot_long - mosaic_inertial_long
            with np.load(bkgnd_sub_mosaic_filename) as npz:
                bsm_img = ma.MaskedArray(**npz)
                bsm_img = bsm_img.filled(0)
            good_long = longitudes >= 0
            restr_bsm_img = bsm_img[ew_ring_lower_limit:ew_ring_upper_limit+1,:]
            ring_midpoint = int(restr_bsm_img.shape[0]/2)
            ew_slice[suffix] = restr_bsm_img[ring_midpoint-r500_pix:
                                             ring_midpoint+r500_pix+1,
                                             mean_corot_long_int]
            # Contrast-stretch the mosaic and add the lines for occultation longitude
            gamma = 0.5
            blackpoint = max(np.min(restr_bsm_img[:, good_long]), 0)
            whitepoint_ignore_frac = 0.995
            img_sorted = sorted(list(restr_bsm_img[:, good_long].flatten()))
            whitepoint = img_sorted[np.clip(int(len(img_sorted)*
                                                whitepoint_ignore_frac),
                                            0, len(img_sorted)-1)]
            grey = np.floor((np.maximum(restr_bsm_img-blackpoint, 0)/
                             (whitepoint-blackpoint))**gamma*256)
            grey = np.clip(grey, 0, 255)
            # Draw thick lines for the min and max occultation longitudes
            grey[:, min_corot_long_int:min_corot_long_int+8] = 255
            grey[:, max_corot_long_int:max_corot_long_int+8] = 255
            greyscale_img[suffix] = grey

    if arguments.output_csv_filename:
        prometheus_dist = prometheus_util.prometheus_close_approach(ed_mid_time, 0)[0]
        row = [root,
               f_ring_util.et2utc(np.min(full_date_data)),
               f_ring_util.et2utc(np.max(full_date_data)),
               f_ring_util.et2utc(ed_mid_time),
               int(label['ORBIT_NUMBER']),
               label.get('OBSERVATION_ID', 'N/A').strip('"\n '),
               label['STAR_NAME'],
               label['RING_OCCULTATION_DIRECTION'],
               label['MINIMUM_WAVELENGTH'],
               label['MAXIMUM_WAVELENGTH'],
               label['DATA_QUALITY_SCORE'],
               np.round(min_inertial_long, 3),
               np.round(max_inertial_long, 3),
               np.round(mean_inertial_long, 3),
               np.round(min_corot_long, 3),
               np.round(max_corot_long, 3),
               np.round(mean_corot_long, 3),
               label['OBSERVED_RING_ELEVATION'],
               label['LOWEST_DETECTABLE_OPACITY'],
               label['HIGHEST_DETECTABLE_OPACITY'],
               np.round(prometheus_dist, 3),
               np.round(avg_tau, 3),
               np.round(ed_core30, 3),
               np.round(ed_core50, 3),
               np.round(ed_full, 3)]
        if arguments.compare_ew:
            row += [best_obsid['before'],
                    np.round(best_midtime_diff['before']/86400, 2),
                    best_obsid['after'],
                    np.round(best_midtime_diff['after']/86400, 2)]
        writer.writerow(row)

    if arguments.compare_ew:
        before_str = None
        if best_obsid['before'] is not None:
            before_str = (f'{best_obsid["before"]} '
                          f'({best_midtime_diff["before"]/86400:.2f} days before) '
                          f'[Occ-Mosaic Long={long_diff["before"]:.2f}]')
        after_str = None
        if best_obsid['after'] is not None:
            after_str = (f'{best_obsid["after"]} '
                         f'({best_midtime_diff["after"]/86400:.2f} days after) '
                         f'[Occ-Mosaic Long={long_diff["after"]:.2f}]')
        plot(f'Core30 ED {ed_core30:6.3f} / Full ED {ed_full:6.3f}',
             title_before=before_str, mosaic_img_before=greyscale_img['before'],
             ew_slice_before=ew_slice['before'],
             title_after=after_str, mosaic_img_after=greyscale_img['after'],
             ew_slice_after=ew_slice['after'])
    else:
        plot(f'Core30 ED {ed_core30:6.3f} / Full ED {ed_full:6.3f}')


##########################################################################################

# Print final statistics

min_tau = np.min(tau_list)
max_tau = np.max(tau_list)
mean_tau = np.mean(tau_list)
median_tau = np.median(tau_list)
num_tau = len(tau_list)
print(f'Tau Min {min_tau:6.3f} / Max {max_tau:6.3f} / '
      f'Mean {mean_tau:6.3f} / Median {median_tau:6.3f} / {num_tau} obs')
min_ed_core30 = np.min(ed_core30_list)
max_ed_core30 = np.max(ed_core30_list)
mean_ed_core30 = np.mean(ed_core30_list)
std_ed_core30 = np.std(ed_core30_list)
median_ed_core30 = np.median(ed_core30_list)
print(f'Core30 ED Min {min_ed_core30:6.3f} / Max {max_ed_core30:6.3f} / '
      f'Mean {mean_ed_core30:6.3f} +/- {std_ed_core30:6.3f} / Median {median_ed_core30:6.3f}')
min_ed_core50 = np.min(ed_core50_list)
max_ed_core50 = np.max(ed_core50_list)
mean_ed_core50 = np.mean(ed_core50_list)
std_ed_core50 = np.std(ed_core50_list)
median_ed_core50 = np.median(ed_core50_list)
print(f'Core50 ED Min {min_ed_core50:6.3f} / Max {max_ed_core50:6.3f} / '
      f'Mean {mean_ed_core50:6.3f} +/- {std_ed_core50:6.3f} / Median {median_ed_core50:6.3f}')
min_ed_full = np.min(ed_full_list)
max_ed_full = np.max(ed_full_list)
mean_ed_full = np.mean(ed_full_list)
std_ed_full = np.std(ed_full_list)
median_ed_full = np.median(ed_full_list)
print(f'Full   ED Min {min_ed_full:6.3f} / Max {max_ed_full:6.3f} / '
      f'Mean {mean_ed_full:6.3f} +/- {std_ed_core50:6.3f} / Median {median_ed_full:6.3f}')

if arguments.plot_aggregate or arguments.save_plots:
    fig = plt.figure(figsize=(12, 8))
    for core_radius, core_occ in zip(core_radius_list, core_occ_list):
        plt.plot(core_radius, core_occ, alpha=0.3)
    plt.xlabel('Core offset (km)')
    plt.ylabel('Optical depth')
    plt.xlim(-ED_CORE_HW50, ED_CORE_HW50)
    plt.title(f'{arguments.instrument} - Mean tau {mean_tau:6.3f} ({num_tau} obs)')
    plt.tight_layout()
    if arguments.save_plots:
        if not os.path.exists('plots_occs'):
            os.mkdir('plots_occs')
        plt.savefig(f'plots_occs/{arguments.instrument}_agg.png')
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()
    else:
        plt.show()

if arguments.output_csv_filename:
    csv_fp.close()

# by_star_list.sort()
# for entry in by_star_list:
#     print(entry)
