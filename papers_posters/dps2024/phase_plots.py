import cspyce
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
if '..' not in sys.path: sys.path.append('../..'); sys.path.append('../../external')

from f_ring_util.f_ring import (compute_corrected_ew,
                                fit_hg_phase_function,
                                hg_func,
                                limit_by_quant,
                                print_hg_params,
                                read_ew_stats,
                                scale_hg_phase_function)
from f_ring_util.moons import prometheus_close_approach

plt.rcParams.update({'font.size': 24})


def plot_various_quants(obsdata, include_phase=True):
    """Choose various quantiles of NormalEW and plot w/phase curves."""
    print('Params: All points / Mean of each Observation group')
    fig, axs = plt.subplots(4, 1, figsize=(13, 13.75))
    for plot_num, (perc1, perc2, color, name) in enumerate(((100, None, 'black', 'All obs'),
                                                            ( 75, None, 'black', '3rd quartile'),
                                                            ( 50, None, 'black', '2nd quartile'),
                                                            ( 25, None, 'black', '1st quartile'))):
        ax = axs[plot_num]
        quant_obsdata = limit_by_quant(obsdata, perc1, perc2)
        ax.scatter(quant_obsdata['Mean Phase'], quant_obsdata['Normal EW Mean'], marker='o',
                   s=5, color=color, alpha=1)
        title = name
        if include_phase:
            params, _, _ = fit_hg_phase_function(2, None, quant_obsdata)
            xrange = np.arange(quant_obsdata['Mean Phase'].min(), quant_obsdata['Mean Phase'].max()+1)
            full_phase_model = hg_func(params, xrange)
            lcolor = 'black' if color != 'black' else '#ff6060'
            total_scale = params[1] + params[3]
            w1 = params[1] / total_scale
            w2 = params[3] / total_scale
            if params[1] < params[3]:
                title += f' ($g_1$={params[2]:.3f}, $w_1$={w2:.3f}; $g_2$={params[0]:.3f}, $w_2$={1-w2:.3f})'
            else:
                title += f' ($g_1$={params[0]:.3f}, $w_1$={w1:.3f}; $g_2$={params[2]:.3f}, $w_2$={1-w1:.3f})'
            ax.plot(xrange, full_phase_model, '-', color=lcolor, lw=4, label=title)
            print(f'*** {perc1} / {perc2}: {color}')
            print_hg_params(params)
        ax.set_yscale('log')
        ax.set_xlim(-2, 182)
        ax.set_xticks([0,45,90,135,180])
        if plot_num != 3:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Phase Angle (°)')
        # if plot_num == 0:
        #     ax.set_title('Phase Curves for Data Subsets')
        ax.tick_params('both', length=10, which='major')
        ax.tick_params('both', length=5, which='minor')
        ax.set_yticks([1,10], labels=['1', '10'])
        ax.legend(prop={'size': 16})
    plt.suptitle('Phase Curves for Data Subsets')
    fig.supylabel('Normal Equivalent Width')
    plt.tight_layout()


def plot_points_phase_time(obsdata, params, title=None, time_fit=3, col='Normal EW Mean', ax=None, **kwargs):
    """Plot scattered EW points by time with fit time curve colored by phase."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))


def _standard_alpha(obsdata):
    """Return alpha based on number of points to plot."""
    if len(obsdata) < 1000:
        return 1
    elif len(obsdata) < 10000:
        return 0.6
    else:
        return 0.3

################################################################################

obsdata_0_1 = read_ew_stats('../../data_files/cass_ew_0_1.csv',
                            obslist_filename='CASSINI_OBSERVATION_LIST',
                            obslist_column='For Photometry')
obsdata_60_0 = read_ew_stats('../../data_files/cass_ew_60_0.csv',
                            obslist_filename='CASSINI_OBSERVATION_LIST',
                            obslist_column='For Photometry')

plot_various_quants(obsdata_0_1)

plt.savefig('phase-curves.png', dpi=300)

################################################################################

cutoff1 = 100
cutoff2 = None
obsdata_limited = limit_by_quant(obsdata_0_1, cutoff1, cutoff2)
params_limited, _, _ = fit_hg_phase_function(2, None, obsdata_limited)
print(f'Cutoff {cutoff1} / {cutoff2}, 1 Degree Slices')
print_hg_params(params_limited)

color1 = 'black'
color2 = '#00a000'

for color_col in ['Mean Phase', 'Mu0']:
    for obsdata in [obsdata_limited, obsdata_60_0]:
        obsdata['Phase Model'] = hg_func(params_limited, obsdata['Mean Phase'])
        obsdata['Mu0'] = np.abs(np.cos(np.radians(obsdata['Mean Emission'])))

        fig, ax1 = plt.subplots(figsize=(13, 7))
        time0 = np.datetime64('1970-01-01T00:00:00') # epoch
        obsdata['Date_secs'] = (obsdata['Date']-time0).dt.total_seconds()/86400
        obsdata['Phase Curve Ratio'] = obsdata['Normal EW Mean'] / obsdata['Phase Model']
        alpha = 1

        if obsdata is obsdata_limited:
            s = 5
        else:
            s = 25

        p = ax1.scatter(obsdata['Date'], obsdata['Phase Curve Ratio'], marker='o', s=s,
                        c=obsdata[color_col], cmap=cm.jet, alpha=alpha)

        timecoeff = np.polyfit(obsdata['Date_secs'], obsdata['Phase Curve Ratio'], 2)
        timerange = np.arange(obsdata['Date_secs'].min(), obsdata['Date_secs'].max(), 100)
        timefit = np.polyval(timecoeff, timerange)
        ax1.plot(timerange, timefit, '-', lw=4, color=color1, label='Quadratic fit to brightness')
        ax1.set_xlim((2003.8-1970)*365, (2018.2-1970)*365)
        ax1.set_yscale('log')
        ax1.set_ylabel(f'Phase-Normalized EW', color=color1)
        plt.tick_params('both', length=10, which='major')
        plt.tick_params('both', length=5, which='minor')

        if obsdata is obsdata_limited:
            plt.yticks([1, 5, 10], labels=['1', '5', '10'])
        else:
            plt.yticks([1, 2], labels=['  1', '  2'])

        ax2 = ax1.twinx()
        et1 = cspyce.utc2et('2004-01-01')
        et2 = cspyce.utc2et('2018-01-01')
        mpl_time0 = cspyce.utc2et('1970-01-01T00:00:00')
        ets = np.linspace(et1, et2, 100)
        dists = np.zeros(ets.shape)
        for i, et in enumerate(ets):
            dists[i], _, _, _ = prometheus_close_approach(et)

        ax1.plot([], [], '--', lw=2, color=color2, label='Prometheus distance')
        ax2.plot((ets-mpl_time0)/86400, dists, '--', lw=2, color=color2, label='Prometheus dist')
        # ax2.legend(loc='upper center', prop={'size': 13})

        ax2.set_ylabel(f'Distance (km)', color=color2)
        ax1.set_xlabel('Date of Observation')
        ax2.set_xlabel('Date of Observation')

        ax1.legend(loc='upper center', prop={'size': 13})

        if color_col == 'Mean Phase':
            name = 'phase'
        else:
            name = 'mu0'
        if obsdata is obsdata_limited:
            name += '-all'
            plt.title('Brightness vs. Time (All Measurements)', pad=12)
        else:
            name += '-60'
            plt.title('Brightness vs. Time (Each Mosaic)', pad=12)

        plt.tight_layout()
        plt.savefig(f'time-variance-{name}.png', dpi=300)
