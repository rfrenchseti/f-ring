import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
if '..' not in sys.path: sys.path.append('..')
from f_ring_util.f_ring import (compute_corrected_ew,
                         fit_hg_phase_function,
                         hg_func,
                         print_hg_params)
import argparse
import os

#Finding the tau values of the regions (right now, various core widths)
#Storing as a textfile in the appropriate dump_ew_csv directory

#Define variables
region = 'core'
size = '0' #slice/step size of the csv file, currently only have 0

#define the inner/outer radii being used for the core values currently
rins = 140220 - np.arange(50, 1001, 50)
routs = 140220 + np.arange(50, 1001, 50)

#removing outliers
eqx_cutoff = 1.5

#set location of data where csvs for each radius dumped
dump_ew_dir = '~/REU_2022/data/dump_ew_csv/'+region

#Write value of all the best taus, along with current radii used, to ongoing file
tau_filepath = '/Users/mlessard/REU_2022/data/dump_ew_csv/'+region+'/slice'+size+'_'+region+'_tau_values.csv'

#open the file, write the header (overwrites if it exists)
taufile = open(tau_filepath, 'w+')
taufile.write('best_tau,rin,rout,unique_obs')
taufile.write('\n')

#calculate the tau value for each core width, and append to textfile
for i in range(len(rins)):
    rin = str(rins[i])
    rout = str(routs[i])

    print(rin)
    print(rout)

    #check to see if rin < rout, if not, append "none" to data
    if rin >= rout:
        best_tau = 'NaN'
        print()
        print(f'** Best Tau: NaN')

    elif rin < rout:
        #location of ew data
        filepath = dump_ew_dir+'/rin'+rin+'_rout'+rout+'/slice'+size+'_ew_stats.csv'

        orig_obsdata = pd.read_csv(filepath, parse_dates=['Date']); ms=20; alpha=0.7
        # orig_obsdata = pd.read_csv(filepath, parse_dates=['Date']); ms=4; alpha=0.1
        #orig_obsdata = pd.read_csv('../data_files/good_qual_full.csv', parse_dates=['Date']); ms=20; alpha=0.7
        # orig_obsdata = pd.read_csv('../data_files/good_qual_1deg.csv', parse_dates=['Date']); ms=4; alpha=0.1
        print('** SUMMARY STATISTICS **')
        print('Unique observation names:', len(orig_obsdata.groupby('Observation')))
        print('Total slices:', len(orig_obsdata))
        print('Starting date:', orig_obsdata['Date'].min())
        print('Ending date:', orig_obsdata['Date'].max())
        print('Time span:', orig_obsdata['Date'].max()-orig_obsdata['Date'].min())

        good_i = np.abs(orig_obsdata['Incidence']-90) > 0.5
        obsdata = orig_obsdata[good_i]
        print('Removed OBSIDs:', set(orig_obsdata[~good_i].groupby('Observation').indices))
        print('Final unique observation names:', len(obsdata))


        # # Optimize Tau for Low-Phase Observations

        # Find "small" (<6) e or i

        low_phase_mask = obsdata['Mean Phase'] <= 60
        low_phase_obsdata = obsdata[low_phase_mask]
        lp_low_e_mask = np.abs(low_phase_obsdata['Mean Emission']-90) < 6
        lp_low_i_mask = np.abs(low_phase_obsdata['Incidence']-90) < 6
        lp_low_ei_mask = lp_low_e_mask | lp_low_i_mask

        low_e_mask = np.abs(obsdata['Mean Emission']-90) < 6
        low_i_mask = np.abs(obsdata['Incidence']-90) < 6
        low_ei_mask = low_e_mask | low_i_mask

        # Find the optimal tau to minimize scatter (means method)
        best_tau = None
        best_ratio = 1e38
        for tau in np.arange(0.000, 1.001, 0.001):
            corrected_ew = compute_corrected_ew(low_phase_obsdata['Normal EW'],
                                                low_phase_obsdata['Mean Emission'],
                                                low_phase_obsdata['Incidence'],
                                                tau=tau)
            mean_low = np.mean(corrected_ew[lp_low_ei_mask])
            mean_notlow = np.mean(corrected_ew[~lp_low_ei_mask])
            if abs(mean_notlow/mean_low-1) < best_ratio:
                best_ratio = abs(mean_notlow/mean_low-1)
                best_tau = tau
            if abs(mean_low/mean_notlow-1) < best_ratio:
                best_ratio = abs(mean_low/mean_notlow-1)
                best_tau = tau
            #print(f'Tau {tau:.3f} - Mean Normal EW Low E/I: {mean_low:8.5f} '
            #      f'EW Other: {mean_notlow:8.5f} '
            #      f'Ratio: {mean_notlow/mean_low:5.3f}')

        print()
        print(f'** Best Tau: {best_tau:.3f}')
        best_tau='{:.3f}'.format(best_tau)

    #Write the value of the best-fit tau, current radii to the taufile
    taufile.write(str(best_tau)+','+rin+','+rout+','+str(len(obsdata)))
    taufile.write('\n')

    print()
    print(region.capitalize()+' region, inner radius '+rin+', outer radius '+rout+' completed')

#close the file
taufile.close()
