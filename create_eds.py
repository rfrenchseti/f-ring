##########################################################################################

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

radial_resolution = 1

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

if arguments.output_csv_filename:
    csv_fp = open(arguments.output_csv_filename, 'w')
    writer = csv.writer(csv_fp)
    hdr = ['XXX']
    writer.writerow(hdr)

OCC_DIR = {'UVIS': '/seti/fring_occultations_1km_uvis',
           'VIMS': '/seti/fring_occultations_1km_vims'}

BAD_OCC = ['UVIS_HSP_2009_012_ALPCRU_I_TAU01KM.TAB',
           'UVIS_HSP_2012_004_IOTORI_E_TAU01KM.TAB',
           'VIMS_2008_160_GAMCRU_I_TAU_01KM.TAB',
           'VIMS_2008_209_GAMCRU_I_TAU_01KM.TAB',
           'VIMS_2008_231_GAMCRU_I_TAU_01KM.TAB',
           'VIMS_2008_271_RLEO_I_TAU_01KM.TAB',
           'VIMS_2008_278_RLEO_I_TAU_01KM.TAB',
           'VIMS_2008_320_GAMCRU_I_TAU_01KM.TAB',
           'VIMS_2008_328_GAMCRU_I_TAU_01KM.TAB',
           'VIMS_2009_077_GAMCRU_I_TAU_01KM.TAB',
           'VIMS_2012_315_ALPCET_I_TAU_01KM.TAB',
           'VIMS_2013_241_BETAND_I_TAU_01KM.TAB',
           'VIMS_2014_022_L2PUP_E_TAU_01KM.TAB',
           'VIMS_2014_067_RLYR_I_TAU_01KM.TAB',
           'VIMS_2016_173_2CEN_I_TAU_01KM.TAB',
           'VIMS_2016_201_ALPSCO_I_TAU_01KM.TAB',
           'VIMS_2016_244_XOPH_I_TAU_01KM.TAB',
           'VIMS_2016_244_XOPH_E_TAU_01KM.TAB',
           'VIMS_2016_294_LAMVEL_I_TAU_01KM.TAB',
           'VIMS_2017_001_GAMCRU_I_TAU_01KM.TAB',
           'VIMS_2017_245_GAMCRU_I_TAU_01KM.TAB',
           'VIMS_2017_251_GAMCRU_I_TAU_01KM.TAB']

data_pd = pd.read_csv(os.path.join(OCC_DIR[arguments.instrument], 'data.csv'),
                      header=0, index_col='OPUS ID')
manifest_pd = pd.read_csv(os.path.join(OCC_DIR[arguments.instrument], 'manifest.csv'),
                          header=0, index_col='OPUS ID')
manifest_pd = manifest_pd.loc[manifest_pd['File Path'].apply(lambda x: x[-4:]) == '.TAB']
data_pd = data_pd.join(manifest_pd, how='inner')
data_pd = data_pd.loc[(data_pd['Data Quality Score'] == 'Good') |
                      (data_pd['Data Quality Score'] == 'Fair')]

core_hw_pix = int(arguments.core_half_width / radial_resolution)

tau_list = []
core_occ_list = []
core_radius_list = []

for pd_index, row in data_pd.iterrows():
    filename = row['File Path']
    dq = row['Data Quality Score']
    _, base = os.path.split(filename)
    if base in BAD_OCC:
        continue
    print(filename)
    inst = base[:4]
    if arguments.instrument is not None and arguments.instrument != inst:
        continue
    col_num = 3 if inst == 'VIMS' else 4
    full_filename = os.path.join(OCC_DIR[arguments.instrument], base)
    occ_pd = pd.read_csv(full_filename, header=None)
    occ = occ_pd[(occ_pd[0] >= 139400) & (occ_pd[0] <= 141000)]
    radii, occ_data = np.array(occ[0]), np.array(occ[col_num])
    if len(occ) < 160:
        print(f'{base}: Insufficient F ring data')
        continue
    skirt_pix = int(100 / radial_resolution)
    mean1 = np.mean(occ_data[:skirt_pix])
    std1 = np.std(occ_data[:skirt_pix])
    mean2 = np.mean(occ_data[-skirt_pix:])
    std2 = np.std(occ_data[-skirt_pix:])
    print(f'{base} {mean1:8.5f} {std1:.5f} {mean2:8.5f} {std2:.5f}')
    if mean1 < -.1 or mean2 < -.1:
        print(f'{base}: Negative skirt')
        continue
    if (std1 > 0.006 or std2 > 0.006):
        print(f'{base}: Too noisy')
        continue
    max_idx = np.argmax(occ_data)
    max_radius = radii[max_idx]
    # if occ_data[max_idx] < 0.02:
    #     print(f'{base}: No peak')
    #     continue
    # if occ_data[max_idx] > 0.5:
    #     print(f'{base}: Peak too high')
    #     continue
    if max_idx-core_hw_pix < 0 or max_idx+core_hw_pix+1 >= len(occ):
        print(f'{base}: Core max too close to edge')
        if False:
            plt.plot(radii-max_radius, occ_data)
            plt.xlabel('Core offset (km)')
            plt.ylabel('Optical depth')
            plt.tight_layout()
            plt.show()
        continue
    core_occ = occ_data[max_idx-core_hw_pix:max_idx+core_hw_pix+1]
    core_radii = radii[max_idx-core_hw_pix:max_idx+core_hw_pix+1    ]
    core_occ_list.append(core_occ)
    core_radius_list.append(core_radii-max_radius)
    avg_tau = -np.log(np.mean(np.exp(-np.clip(core_occ, 0, 100))))
    if avg_tau < 0:
        print(f'{base} {dq}: Avg Tau < 0')
        if False:
            plt.plot(radii-max_radius, occ_data)
            plt.xlabel('Core offset (km)')
            plt.ylabel('Optical depth')
            plt.tight_layout()
            plt.show()
        continue
    print(f'{base} {dq}: Avg tau {avg_tau:.3f}')
    tau_list.append(avg_tau)

    if arguments.output_csv_filename:
            row = [obs_id, slice_num, np.sum(~slice_bad_long), slice_et_date,
                   np.round(slice_min_long, 2),
                   np.round(slice_max_long, 2),
                   np.round(slice_min_res, 3),
                   np.round(slice_max_res, 3),
                   np.round(slice_mean_res, 3),
                   np.round(np.degrees(slice_min_ph), 3),
                   np.round(np.degrees(slice_max_ph), 3),
                   np.round(np.degrees(slice_mean_ph), 3),
                   np.round(np.degrees(slice_min_em), 3),
                   np.round(np.degrees(slice_max_em), 3),
                   np.round(np.degrees(slice_mean_em), 3),
                   np.round(np.degrees(incidence_angle), 3),
                   percentage_ew_ok,
                   np.round(slice_ew_mean, 5),
                   np.round(slice_ew_std, 5),
                   np.round(slice_ew_mean_mu, 5),
                   np.round(slice_ew_std_mu, 5)]
            if arguments.compute_widths:
                row += [
                        np.round(np.mean(slice_w1), 3),
                        np.round(np.std(slice_w1), 3),
                        np.round(np.mean(slice_w2), 3),
                        np.round(np.std(slice_w2), 3),
                        np.round(np.mean(slice_w3), 3),
                        np.round(np.std(slice_w3), 3)]

            writer.writerow(row)

    if arguments.plot_results or arguments.save_plots:
        fig = plt.figure(figsize=(12, 8))
        plt.plot(radii-max_radius, occ_data)
        plt.xlabel('Core offset (km)')
        plt.ylabel('Optical depth')
        plt.ylim(-0.01, 0.5)
        plt.tight_layout()
        if arguments.save_plots:
            if not os.path.exists('plots'):
                os.mkdir('plots')
            plt.savefig(f'plots/{base}.png')
            fig.clear()
            plt.close()
            plt.cla()
            plt.clf()
        if arguments.plot_results:
            plt.show()

mean_tau = np.mean(tau_list)
num_tau = len(tau_list)
print(f'Average tau {mean_tau:.3f} with {num_tau} obs')

if arguments.plot_aggregate or arguments.save_plots:
    fig = plt.figure(figsize=(12, 8))
    for core_radius, core_occ in zip(core_radius_list, core_occ_list):
        plt.plot(core_radius, core_occ, alpha=0.3)
    plt.xlabel('Core offset (km)')
    plt.ylabel('Optical depth')
    plt.xlim(-arguments.core_half_width, arguments.core_half_width)
    plt.title(f'{arguments.instrument} - Mean tau {mean_tau:.3f} ({num_tau} obs)')
    plt.tight_layout()
    if arguments.save_plots:
        if not os.path.exists('plots'):
            os.mkdir('plots')
        plt.savefig(f'plots/{arguments.instrument}_agg.png')
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()
    if arguments.plot_aggregate:
        plt.show()

if arguments.output_csv_filename:
    csv_fp.close()
