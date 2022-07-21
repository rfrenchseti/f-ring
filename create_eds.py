##########################################################################################

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f_ring_util import et2utc
import prometheus_util

import pdsparser

RADIAL_RESOLUTION = 1
SLUSH = 1000
ED_CORE_HW30 = 30  # +/-
ED_CORE_HW50 = 50
ED_FULL_HW = 500 # +/-
# SKIRT1 = 700
# SKIRT2 = 800

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
   cmd_line = []

parser = argparse.ArgumentParser()

parser.add_argument('--core-half-width', type=int, default=50,
                    help='The half width of the core in km')
parser.add_argument('--instrument', type=str, default='UVIS',
                    help='Limit to one instrument (UVIS or VIMS)')
parser.add_argument('--output-csv-filename', type=str,
                    help='Name of output CSV file')
parser.add_argument('--plot-results', action='store_true', default=False,
                    help='Plot the core occultation for each observation')
parser.add_argument('--plot-aggregate', action='store_true', default=False,
                    help='Plot the combined core occultations for all observations')
parser.add_argument('--save-plots', action='store_true', default=False,
                    help='Same as --plot-results but save plots to disk instead')

arguments = parser.parse_args(cmd_line)


OCC_DIR = {'UVIS': '/seti/fring_occultations_1km_uvis',
           'VIMS': '/seti/fring_occultations_1km_vims'}

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
           'VIMS_2017_007_VYCMA_E',  # Negative ED
           'VIMS_2017_007_VYCMA_I',  # Negative ED
           'VIMS_2017_050_VYCMA_E',  # Full < Core
           'VIMS_2017_050_VYCMA_I',  # Full < Core
           'VIMS_2017_073_LAMVEL_I', # Totally bad data
           'VIMS_2017_104_ALPORI_E', # Full < Core
           'VIMS_2017_120_ALPCMA_I', # Full < Core
           'VIMS_2017_245_GAMCRU_I', # Full < 0
           'VIMS_2017_251_GAMCRU_I', # Full < 0
           ]


def plot(title=None):
    if arguments.plot_results or arguments.save_plots:
        fig = plt.figure(figsize=(12, 8))
        plt.plot(radii-max_radius, occ_data)
        plt.xlabel('Core offset (km)')
        plt.ylabel('Optical depth')
        plt.xlim(-500,500)
        plt.ylim(-0.01, 0.5)
        if title is None:
            plt.title(root)
        else:
            plt.title(f'{root} ({title})')
        plt.tight_layout()
        if arguments.save_plots:
            if not os.path.exists('plots'):
                os.mkdir('plots')
            plt.savefig(f'plots/{root}.png')
            fig.clear()
            plt.close()
            plt.cla()
            plt.clf()
        else:
            plt.show()


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
           'Minimum Longitude',
           'Maximum Longitude',
           'Mean Longitude',
           'Ring Elevation',
           'Lowest Detectable Opacity',
           'Highest Detectable Opacity',
           'Prometheus Distance',
           'Equivalent Tau',
           'Core30 ED',
           'Core50 ED',
           'Full ED']
    writer.writerow(hdr)

data_pd = pd.read_csv(os.path.join(OCC_DIR[arguments.instrument], 'data.csv'),
                      header=0, index_col='OPUS ID')
manifest_pd = pd.read_csv(os.path.join(OCC_DIR[arguments.instrument], 'manifest.csv'),
                          header=0, index_col='OPUS ID')
manifest_pd = manifest_pd.loc[manifest_pd['File Path'].apply(lambda x: x[-4:]) == '.TAB']
data_pd = data_pd.join(manifest_pd, how='inner')

core_hw_pix = int(arguments.core_half_width / RADIAL_RESOLUTION)
ed_core30_hw_pix = int(ED_CORE_HW30 / RADIAL_RESOLUTION)
ed_core50_hw_pix = int(ED_CORE_HW50 / RADIAL_RESOLUTION)
ed_full_hw_pix = int(ED_FULL_HW / RADIAL_RESOLUTION)

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
    root = base.replace('_TAU01KM.TAB', '').replace('_TAU_01KM.TAB', '')
    if root in BAD_OCC:
        continue
    # if base > 'UVIS_HSP_2008_343': # Limit to Albers 2012 data set
    #     continue
    full_filename = os.path.join(OCC_DIR[arguments.instrument], base)
    label_filename = full_filename.replace('.TAB', '.LBL')

    segments = base.split('_')
    inst = segments[0]
    if arguments.instrument is not None and arguments.instrument != inst:
        continue

    label = pdsparser.PdsLabel.from_file(label_filename).as_dict()

    star = f'{segments[4]}({segments[5]})'
    date = f'{segments[2]}-{segments[3]}'
    by_star_list.append(f'{star} {date}')
    col_num = 3 if inst == 'VIMS' else 4
    date_col_num = 9 if inst == 'VIMS' else 7
    occ_pd = pd.read_csv(full_filename, header=None)
    occ = occ_pd[(occ_pd[0] >= 140220-SLUSH) & (occ_pd[0] <= 140220+SLUSH)]
    radii = np.array(occ[0])
    long_data = np.array(occ[1])
    occ_data = np.array(occ[col_num])
    date_data = np.array(occ[date_col_num])
    if radii[0] != 140220-SLUSH or radii[-1] != 140220+SLUSH:
        print(f'{root}: Insufficient F ring data')
        continue
    max_idx = np.argmax(occ_data)
    max_radius = radii[max_idx]
    # if occ_data[max_idx] < 0.02:
    #     print(f'{root}: No peak')
    #     continue
    # if occ_data[max_idx] > 0.5:
    #     print(f'{root}: Peak too high')
    #     continue
    if max_idx-ed_full_hw_pix < 0 or max_idx+ed_full_hw_pix+1 >= len(occ):
        print(f'{root}: Core max too close to edge')
        plot('Bad edge')
        continue
    # skirt1 = occ_pd[(occ_pd[0] >= 140220-SKIRT2) & (occ_pd[0] <= 140220-SKIRT1)]
    # skirt1_radii, skirt1_data = np.array(skirt1[0]), np.array(skirt1[col_num])
    # skirt2 = occ_pd[(occ_pd[0] >= 140220+SKIRT1) & (occ_pd[0] <= 140220+SKIRT2)]
    # skirt2_radii, skirt2_data = np.array(skirt2[0]), np.array(skirt2[col_num])
    # if len(skirt1) != SKIRT2-SKIRT1+1 or len(skirt2) != SKIRT2-SKIRT1+1:
    #     print(f'{root}: Insufficient skirt')
    #     plot('Bad skirt')
    #     continue
    # mean1 = np.mean(skirt1_data)
    # std1 = np.std(skirt1_data)
    # mean2 = np.mean(skirt2_data)
    # std2 = np.std(skirt2_data)
    # print(f'{root} {mean1:8.5f} {std1:.5f} {mean2:8.5f} {std2:.5f} ', end='')
    # if mean1 < -.1 or mean2 < -.1:
    #     print('Negative skirt')
    #     continue
    # if (std1 > 0.006 or std2 > 0.006):
    #     print('Too noisy')
    #     continue
    core_occ = occ_data[max_idx-core_hw_pix:max_idx+core_hw_pix+1]
    core_radii = radii[max_idx-core_hw_pix:max_idx+core_hw_pix+1    ]
    core_occ_list.append(core_occ)
    core_radius_list.append(core_radii-max_radius)
    avg_tau = -np.log(np.mean(np.exp(-core_occ)))
    if avg_tau < 0:
        print(f'{root}: {dq} Avg Tau < 0')
        plot('Avg Tau < 0')
        continue
    ed_core30_occ = occ_data[max_idx-ed_core30_hw_pix:max_idx+ed_core30_hw_pix+1]
    ed_core30 = np.sum(ed_core30_occ) * RADIAL_RESOLUTION
    ed_core50_occ = occ_data[max_idx-ed_core50_hw_pix:max_idx+ed_core50_hw_pix+1]
    ed_core50 = np.sum(ed_core50_occ) * RADIAL_RESOLUTION
    ed_full_occ = occ_data[max_idx-ed_full_hw_pix:max_idx+ed_full_hw_pix+1]
    ed_full = np.sum(ed_full_occ) * RADIAL_RESOLUTION
    if ed_full < ed_core30:
        print(f'{root} {dq}: Full ED {ed_full:6.3f} < Core ED {ed_core30:6.3f}')
        continue
    full_long_data = long_data[max_idx-ed_full_hw_pix:max_idx+ed_full_hw_pix+1]
    full_date_data = date_data[max_idx-ed_full_hw_pix:max_idx+ed_full_hw_pix+1]
    print(f'{root:22s} {dq}: Derived tau {avg_tau:6.3f} / Core30 ED {ed_core30:6.3f} / '
          f'Core50 ED {ed_core50:6.3f} / Full ED {ed_full:6.3f}')
    tau_list.append(avg_tau)
    ed_core30_list.append(ed_core30)
    ed_core50_list.append(ed_core50)
    ed_full_list.append(ed_full)

    if arguments.output_csv_filename:
        mid_time = np.mean(full_date_data)
        prometheus_dist = prometheus_util.prometheus_close_approach(mid_time, 0)[0]
        row = [root,
               et2utc(np.min(full_date_data)),
               et2utc(np.max(full_date_data)),
               et2utc(mid_time),
               int(label['ORBIT_NUMBER']),
               label.get('OBSERVATION_ID', 'N/A').strip('"\n '),
               label['STAR_NAME'],
               label['RING_OCCULTATION_DIRECTION'],
               label['MINIMUM_WAVELENGTH'],
               label['MAXIMUM_WAVELENGTH'],
               label['DATA_QUALITY_SCORE'],
               np.round(np.min(full_long_data), 3),
               np.round(np.max(full_long_data), 3),
               np.round(np.mean(full_long_data), 3),
               label['OBSERVED_RING_ELEVATION'],
               label['LOWEST_DETECTABLE_OPACITY'],
               label['HIGHEST_DETECTABLE_OPACITY'],
               np.round(prometheus_dist, 3),
               np.round(avg_tau, 3),
               np.round(ed_core30, 3),
               np.round(ed_core50, 3),
               np.round(ed_full, 3)]
        writer.writerow(row)

    plot(f'Core30 ED {ed_core30:6.3f} / Full ED {ed_full:6.3f}')

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
    plt.xlim(-arguments.core_half_width, arguments.core_half_width)
    plt.title(f'{arguments.instrument} - Mean tau {mean_tau:6.3f} ({num_tau} obs)')
    plt.tight_layout()
    if arguments.save_plots:
        if not os.path.exists('plots'):
            os.mkdir('plots')
        plt.savefig(f'plots/{arguments.instrument}_agg.png')
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
