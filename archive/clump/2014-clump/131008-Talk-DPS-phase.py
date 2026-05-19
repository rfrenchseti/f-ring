'''
Created on Nov 22, 2011

@author: rfrench
'''

from optparse import OptionParser
import numpy as np
import numpy.ma as ma
import pickle
import ringutil
import sys
import os.path
import matplotlib.pyplot as plt
import colorsys
import cspice
import scipy.optimize as sciopt
import scipy.stats.distributions as scipydist
import matplotlib

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['-a']
    
parser = OptionParser()

ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

paper_root = os.path.join(ringutil.PAPER_ROOT, 'Talk-DPS2013')

color_voyager1 = '#98FF8A'
color_voyager2 = '#6AB361'
color_cassini = '#D1A264'
color_wavelet = '#F74F4F'
color_wavelets = ('#F74F4F', '#e08e24')
color_clump = (0xf7/256., 0x4f/256., 0x4f/256.)
color_profile = '#ffffff'
color_axis = '#d0d0d0'
color_text = '#d0d0d0'
color_poisson = '#00a0a0'
lw_poisson = 4
lw_profile = 2
lw_clump = 3
lw_wavelet = 3
ms_scatter = 20
fullsize_figure = (8,5)

color_background = (0,0,0)
color_foreground = (1,1,1)
color_dark_grey = (0.5, 0.5, 0.5)
color_grey = (0.625, 0.625, 0.625)
color_bright_grey = (0.75, 0.75, 0.75)
markersize = 8.5
markersize_voyager = 3.5

blackpoint = 0.
whitepoint = 0.522407851536
gamma = 0.5
radius_res = options.radius_resolution
radius_start = options.radius_start
matplotlib.rc('figure', facecolor=color_background)
matplotlib.rc('axes', facecolor=color_background, edgecolor=color_axis, labelcolor=color_text)
matplotlib.rc('xtick', color=color_axis, labelsize=20)
matplotlib.rc('xtick.major', size=8)
matplotlib.rc('xtick.minor', size=6)
matplotlib.rc('ytick', color=color_axis, labelsize=20)
matplotlib.rc('ytick.major', size=8)
matplotlib.rc('ytick.minor', size=6)
matplotlib.rc('font', size=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('text', color=color_text)

def fix_graph_colors(fig, ax, ax2, legend):
    for line in ax.xaxis.get_ticklines() + ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines() + ax.yaxis.get_ticklines(minor=True):
        line.set_color(color_foreground)
    if legend != None:
        legend.get_frame().set_facecolor(color_background)
        legend.get_frame().set_edgecolor(color_background)
        for text in legend.get_texts():
            text.set_color(color_text) 

def save_fig(fig, ax, fn,ax2 = None, legend=None):
    fix_graph_colors(fig, ax, ax2, legend)
    fn = os.path.join(paper_root,fn)
    print 'Saving', fn
    plt.savefig(fn, bbox_inches='tight', facecolor=color_background)   
    plt.close()

#===============================================================================
# 
#===============================================================================

def calculate_clump_ratio(ets, clumpspec):
    t0, t1, h, b = clumpspec
    a = h / (t1-t0)
    ret_ratio = []
    for t in ets:
        if t < t0:
            ret_ratio.append(1.)
        elif t < t1:
            ret_ratio.append(a*(t-t0)+1.)
        else:
            ret_ratio.append(1.+h*np.exp(-b*(t-t1)))
    ret_ratio = np.array(ret_ratio)
    ret_ratio[np.where(ret_ratio<1e-10)[0]] = 1e-10
    return ret_ratio

def compute_z(mu, mu0, tau, transmission):
    transmission_list = tau*(mu-mu0)/(mu*mu0*(np.exp(-tau/mu)-np.exp(-tau/mu0)))
    reflection_list = tau*(mu+mu0)/(mu*mu0*(1-np.exp(-tau*(1/mu+1/mu0))))
    ret = np.where(transmission, transmission_list, reflection_list)
    return ret

def compute_corrected_ew(ew, emission, incidence, tau): # In degrees
    transmission = emission > 90.
    mu = np.abs(np.cos(emission*np.pi/180))
    mu0 = np.abs(np.cos(incidence*np.pi/180))
    ret = ew * mu * compute_z(mu, mu0, tau, transmission)
    return ret

def compute_corrected_ew_clumpspec(ew, emission, incidence, tau, clumpspec, ets): # In degrees
    transmission = emission > 90.
    mu = np.abs(np.cos(emission*np.pi/180))
    mu0 = np.abs(np.cos(incidence*np.pi/180))
    ret = ew * mu * compute_z(mu, mu0, tau, transmission)
    ret /= calculate_clump_ratio(ets, clumpspec)
    return ret

def optimize_tau(ew_list, phase_list, emission_list, incidence_list):
    best_resid = 1e38
    best_tau = None
    for tau in np.arange(0.0001,0.2,0.0001):
        new_ew = compute_corrected_ew(ew_list, emission_list, incidence_list, tau)
        log_new_ew = np.log10(new_ew)
        coeffs = np.polyfit(phase_list, log_new_ew, 3)
        resid = np.sqrt(np.sum((log_new_ew - np.polyval(coeffs, phase_list))**2))
        if resid < best_resid:
            best_resid = resid
            best_tau = tau
    print best_tau, best_resid
    return best_tau
            
def optimize_clumpspec(ew_list, phase_list, emission_list, incidence_list, et_list, tau):
        def clumpspec_func(params, ew_list, phase_list, emission_list, incidence_list, et_list, tau):
            t0, t1, h, b = params
            if t0 > t1:
                return 1e20*(t0-t1)
            if t0 < np.min(et_list):
                return -1e20*(t0-np.min(et_list))
            if t1 > np.max(et_list):
                return 1e20*(t1-np.max(et_list)) 
            if h <= 0:
                return -1e20*h
            if b <= 0:
                return -1e30*b
            if b >= 1e-4:
                return 1e30*b
            new_ew = compute_corrected_ew_clumpspec(ew_list, emission_list, incidence_list, tau, params, et_list)
            log_new_ew = np.log10(new_ew)
            coeffs = np.polyfit(phase_list, log_new_ew, 3)
            resid = np.sqrt(np.sum((log_new_ew - np.polyval(coeffs, phase_list))**2))
            return resid

        ret = sciopt.fmin_powell(clumpspec_func, (np.min(et_list), np.max(et_list), 1, .000001),
                                 args=(ew_list, phase_list, emission_list, incidence_list, et_list, tau),
                                 ftol = 1e-20, xtol = 1e-20, disp=False, full_output=False)
        return ret
        
def optimize_tau_clumpspec(ew_list, phase_list, emission_list, incidence_list, et_list):
    best_resid = 1e38
    best_tau = None
    for tau in np.arange(0.001,0.050,0.001):
        clumpspec = (217771265.18297502, 230029221.32973513, 0.84288311887621625, 8.7843910005748475e-08)
#        clumpspec = optimize_clumpspec(ew_list, phase_list, emission_list, incidence_list, et_list, tau)
        new_ew = compute_corrected_ew_clumpspec(ew_list, emission_list, incidence_list, tau, clumpspec, et_list)
        log_new_ew = np.log10(new_ew)
        coeffs = np.polyfit(phase_list, log_new_ew, 3)
        resid = np.sqrt(np.sum((log_new_ew - np.polyval(coeffs, phase_list))**2))
        print tau, resid, clumpspec
        if resid < best_resid:
            best_resid = resid
            best_tau = tau
    return best_tau, clumpspec

#===============================================================================
# 
#===============================================================================

def get_color(val, min_val, max_val):
    color_frac = (max_val-val) / (max_val-min_val) * .75 - .08
    while color_frac > 1:
        color_frac -= 1
    while color_frac < 0:
        color_frac += 1
    color = colorsys.hsv_to_rgb(color_frac, 1, 1)
    return color

def plot_color_by(mean_ew_list, std_ew_list, phase_angle_list, color_by_list, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    min_color_by = np.min(color_by_list)
    max_color_by = np.max(color_by_list)
    for idx in range(len(mean_ew_list)):
        color = get_color(color_by_list[idx], min_color_by, max_color_by)
        plt.errorbar(phase_angle_list[idx], mean_ew_list[idx], yerr=std_ew_list[idx], 
                     fmt='o', mfc=color, mec=color, ms=8)
    for frac_num in range(7):
        frac_val = frac_num / 6.
        val = min_color_by + frac_val*(max_color_by-min_color_by)
        color = get_color(val, min_color_by, max_color_by)
        plt.plot(10, 1e-8, 'o', mfc=color, mec=color, ms=8, label='%.2f' % val)
    ax.set_yscale('log')
    plt.xlabel(r'Phase angle ($^\circ$)')
    plt.ylabel(r'Equivalent width (km) $\times$ $\mu$')
    plt.title(title)
    ax.set_xlim(0,180)
    ax.set_ylim(.2, 20)
    plt.legend(numpoints=1, loc='upper left')

def poly_scale_func(params, coeffs, phase_list, log_ew_list):
    scale = params[0]
    return np.sqrt(np.sum((log_ew_list - (np.polyval(coeffs, phase_list) + scale))**2))
    
def plot_phase_curve(phase_list, ew_list, std_list, coeffs, fmt, ms, ls, mec, mfc, label):
    log_ew_list = np.log10(ew_list)

    if coeffs is None:
        coeffs = np.polyfit(phase_list, log_ew_list, 3)
        scale = 0.
    else:
        scale = sciopt.fmin_powell(poly_scale_func, (1.,), args=(coeffs, phase_list, log_ew_list),
                                   ftol = 1e-8, xtol = 1e-8, disp=False, full_output=False)

    std_list = None
#    plt.errorbar(phase_list, ew_list, yerr=std_list, ecolor=mec, fmt=fmt, ms=ms, mec=mec, mfc=mfc, mew=2, label=label)
    plt.plot(np.arange(0,180), 10**scale*10**np.polyval(coeffs, np.arange(0,180)), ls, lw=3, color=mec,
             label=label)

    return coeffs, 10.**scale




#===============================================================================
# 
#===============================================================================

phase_list_db = {}
emission_list_db = {}
incidence_list_db = {}
ew_list_db = {}
std_list_db = {}
base_ew_list_db = {}
peak_ew_list_db = {}
et_list_db = {}
spacecraft_list = ['V1', 'V2', 'C', 'V1I', 'V1O', 'V2I', 'V2O']

for key in spacecraft_list:
    phase_list_db[key] = np.empty(0)
    emission_list_db[key] = np.empty(0)
    incidence_list_db[key] = np.empty(0)
    ew_list_db[key] = np.empty(0)
    std_list_db[key] = np.empty(0)
    base_ew_list_db[key] = np.empty(0)
    peak_ew_list_db[key] = np.empty(0)
    et_list_db[key] = np.empty(0)
    
voyager_photometry_filename = 'T:/fring-brightening/voyager/New_Vgr_Photometry.tab'
v_file = open(voyager_photometry_filename, 'r')
v_file.readline() # skip header
while True:
    v_fields = v_file.readline().strip().split()
    if len(v_fields) < 5:
        break
    filename = v_fields[0]
    phase_angle = float(v_fields[1])
    emission_angle = float(v_fields[2])
    incidence_angle = float(v_fields[3])
    ew = float(v_fields[4])
    if filename[0] == '1':
        spacecraft = 'V1'
    else:
        spacecraft = 'V2'
    phase_list_db[spacecraft] = np.append(phase_list_db[spacecraft], phase_angle)
    emission_list_db[spacecraft] = np.append(emission_list_db[spacecraft], emission_angle)
    incidence_list_db[spacecraft] = np.append(incidence_list_db[spacecraft], incidence_angle)
    ew_list_db[spacecraft] = np.append(ew_list_db[spacecraft], ew)
    std_list_db[spacecraft] = np.append(std_list_db[spacecraft], 0.)
v_file.close()


for obs_id, image_name, full_path in ringutil.enumerate_files(options, args, obsid_only=True):
    if (obs_id == 'ISS_036RF_FMOVIE001_VIMS' or
        obs_id == 'ISS_036RF_FMOVIE002_VIMS' or
        obs_id == 'ISS_039RF_FMOVIE002_VIMS' or
        obs_id == 'ISS_039RF_FMOVIE001_VIMS' or
        obs_id == 'ISS_041RF_FMOVIE002_VIMS' or
        obs_id == 'ISS_041RF_FMOVIE001_VIMS'):
        continue
    
#    if (obs_id != 'ISS_000RI_SATSRCHAP001_PRIME' and
#        obs_id != 'ISS_00ARI_SPKMOVPER001_PRIME' and
#        obs_id != 'ISS_006RI_LPHRLFMOV001_PRIME' and
#        obs_id != 'ISS_007RI_LPHRLFMOV001_PRIME' and
#        obs_id != 'ISS_029RF_FMOVIE001_VIMS' and
#        obs_id != 'ISS_031RF_FMOVIE001_VIMS' and
#        obs_id != 'ISS_032RF_FMOVIE001_VIMS' and
#        obs_id != 'ISS_033RF_FMOVIE001_VIMS' and
#        obs_id != 'ISS_036RF_FMOVIE001_VIMS' and
#        obs_id != 'ISS_036RF_FMOVIE002_VIMS' and
#        obs_id != 'ISS_039RF_FMOVIE001_VIMS' and
#        obs_id != 'ISS_039RF_FMOVIE002_VIMS' and
#        obs_id != 'ISS_041RF_FMOVIE002_VIMS' and
#        obs_id != 'ISS_041RF_FMOVIE001_VIMS' and
#        obs_id != 'ISS_044RF_FMOVIE001_VIMS' and
#        obs_id != 'ISS_051RI_LPMRDFMOV001_PRIME' and
#        obs_id != 'ISS_055RF_FMOVIE001_VIMS' and
#        obs_id != 'ISS_055RI_LPMRDFMOV001_PRIME' and
#        obs_id != 'ISS_057RF_FMOVIE001_VIMS' and
#        obs_id != 'ISS_068RF_FMOVIE001_VIMS' and
#        obs_id != 'ISS_075RF_FMOVIE002_VIMS' and
#        obs_id != 'ISS_083RI_FMOVIE109_VIMS' and
#        obs_id != 'ISS_087RF_FMOVIE003_PRIME' and
#        obs_id != 'ISS_089RF_FMOVIE003_PRIME' and
#        obs_id != 'ISS_100RF_FMOVIE003_PRIME' and
#        obs_id != 'V1I' and obs_id != 'V1O' and obs_id != 'V2I' and obs_id != 'V2O'):
#        continue

    (reduced_mosaic_data_filename, reduced_mosaic_metadata_filename,
    bkgnd_mask_filename, bkgnd_model_filename,
    bkgnd_metadata_filename) = ringutil.bkgnd_paths(options, obs_id)

    (ew_data_filename, ew_mask_filename) = ringutil.ew_paths(options, obs_id)

    if (not os.path.exists(ew_data_filename+'.npy')) or (not os.path.exists(reduced_mosaic_metadata_filename)):
        continue
    
    reduced_metadata_fp = open(reduced_mosaic_metadata_filename, 'rb')
    mosaic_data = pickle.load(reduced_metadata_fp)
    obsid_list = pickle.load(reduced_metadata_fp)
    image_name_list = pickle.load(reduced_metadata_fp)
    full_filename_list = pickle.load(reduced_metadata_fp)
    reduced_metadata_fp.close()

    (longitudes, resolutions, image_numbers,
     ETs, emission_angles, incidence_angles,
     phase_angles) = mosaic_data

    ew_data = np.load(ew_data_filename+'.npy')
    ew_data = ew_data.view(ma.MaskedArray)
    ew_data.mask = np.load(ew_mask_filename+'.npy')
    phase_angles = phase_angles.view(ma.MaskedArray)
    phase_angles.mask = ew_data.mask
    emission_angles = emission_angles.view(ma.MaskedArray)
    emission_angles.mask = ew_data.mask
    incidence_angles = incidence_angles.view(ma.MaskedArray)
    incidence_angles.mask = ew_data.mask
    ETs = ETs.view(ma.MaskedArray)
    ETs.mask = ew_data.mask

    spacecraft = 'C'
    if obs_id[0] == 'V':
        spacecraft = obs_id
        ew_data[np.where(ew_data == 0.)] = ma.masked
        ew_data /= np.abs(np.cos(emission_angles*np.pi/180.)) # Voyager data is already "normal" EW

    mean_phase = ma.mean(phase_angles)
    mean_emission = ma.mean(emission_angles)
    mean_et = ma.mean(ETs)

    if obs_id == 'V1I' or obs_id == 'V1O':
        mean_incidence = 86.0
    elif obs_id == 'V2I' or obs_id == 'V2O':
        mean_incidence = 82.0
    else:
        mean_incidence = ma.mean(incidence_angles)

    if mean_incidence > 87:
        continue
    
    sorted_ew_data = np.sort(ew_data)
    num_valid = np.count_nonzero(np.logical_not(ew_data.mask))
    perc_idx = int(num_valid * 0.15)
    baseline = sorted_ew_data[perc_idx]
    perc_idx = int(num_valid * 0.95)
    peak = sorted_ew_data[perc_idx]
    
    mean_ew = ma.mean(ew_data)
    std_ew = np.std(ew_data)

    phase_list_db[spacecraft] = np.append(phase_list_db[spacecraft], mean_phase)
    emission_list_db[spacecraft] = np.append(emission_list_db[spacecraft], mean_emission)
    incidence_list_db[spacecraft] = np.append(incidence_list_db[spacecraft], mean_incidence)
    ew_list_db[spacecraft] = np.append(ew_list_db[spacecraft], mean_ew)
    std_list_db[spacecraft] = np.append(std_list_db[spacecraft], std_ew)
    base_ew_list_db[spacecraft] = np.append(base_ew_list_db[spacecraft], baseline)
    peak_ew_list_db[spacecraft] = np.append(peak_ew_list_db[spacecraft], peak)
    et_list_db[spacecraft] = np.append(et_list_db[spacecraft], mean_et)
        
    percentage_ok = float(len(np.where(longitudes >= 0)[0])) / len(longitudes) * 100

    print '%-30s %3d%% P %7.3f E %7.3f I %7.3f %-15s EW %8.5f +/- %8.5f' % (obs_id, percentage_ok,
        mean_phase, mean_emission, mean_incidence, cspice.et2utc(mean_et, 'C', 0)[:12], mean_ew, std_ew)
        
for key in spacecraft_list:
    std_list_db[key] = np.where(ew_list_db[key] < std_list_db[key], ew_list_db[key]*.999, std_list_db[key])



v1_log_ew_list = np.log10(ew_list_db['V1'])
v2_log_ew_list = np.log10(ew_list_db['V2'])
coeffs = np.polyfit(phase_list_db['V1'], v1_log_ew_list, 3)
v1_v2_scale = sciopt.fmin_powell(poly_scale_func, (1.,), args=(coeffs, phase_list_db['V2'], v2_log_ew_list),
                                 ftol = 1e-8, xtol = 1e-8, disp=False, full_output=False)
v1_v2_scale = 10**v1_v2_scale
print 'V2/V1 SCALE', v1_v2_scale, 1/v1_v2_scale
v1_v2_scale = .5

#c_tau_mean, clumpspec_mean = optimize_tau_clumpspec(ew_list_db['C'], phase_list_db['C'], emission_list_db['C'], incidence_list_db['C'], et_list_db['C'])

c_tau_mean = optimize_tau(ew_list_db['C'], phase_list_db['C'], emission_list_db['C'], incidence_list_db['C'])
c_tau_peak = optimize_tau(peak_ew_list_db['C'], phase_list_db['C'], emission_list_db['C'], incidence_list_db['C'])
c_tau_base = optimize_tau(base_ew_list_db['C'], phase_list_db['C'], emission_list_db['C'], incidence_list_db['C'])
v_tau = optimize_tau(np.append(ew_list_db['V1'], ew_list_db['V2']/v1_v2_scale),
                     np.append(phase_list_db['V1'], phase_list_db['V2']),
                     np.append(emission_list_db['V1'], emission_list_db['V2']),
                     np.append(incidence_list_db['V1'], incidence_list_db['V2']))

#v_tau = c_tau
#c_tau = 0.000001
#v_tau = 0.000001

#c_tau_base = 0.036
#v_tau = 0.036

print 'C TAU PEAK', c_tau_peak
print 'C TAU MEAN', c_tau_mean
print 'C TAU BASE', c_tau_base
print 'V TAU', v_tau

v_tau_mean = c_tau_mean
v_tau_base = c_tau_base
v_tau_peak = c_tau_peak

c_ew_list = compute_corrected_ew(ew_list_db['C'], emission_list_db['C'], incidence_list_db['C'], c_tau_mean)
c_std_list = compute_corrected_ew(std_list_db['C'], emission_list_db['C'], incidence_list_db['C'], c_tau_mean)
c_base_ew_list = compute_corrected_ew(base_ew_list_db['C'], emission_list_db['C'], incidence_list_db['C'], c_tau_base)
c_peak_ew_list = compute_corrected_ew(peak_ew_list_db['C'], emission_list_db['C'], incidence_list_db['C'], c_tau_peak)
v1_ew_list = compute_corrected_ew(ew_list_db['V1'], emission_list_db['V1'], incidence_list_db['V1'], v_tau)
v2_ew_list = compute_corrected_ew(ew_list_db['V2'], emission_list_db['V2'], incidence_list_db['V2'], v_tau)


v1io_phase = []
v1io_ews = []
v1io_base_ews = []
v1io_peak_ews = []
v2io_phase = []
v2io_ews = []
v2io_base_ews = []
v2io_peak_ews = []

base_str = ''
peak_str = ''
peak_base_str = ''

for v_prof_key in ['V1I', 'V1O', 'V2I', 'V2O']:
    phase_angle = phase_list_db[v_prof_key]
    ew = compute_corrected_ew(ew_list_db[v_prof_key], emission_list_db[v_prof_key], 
                              incidence_list_db[v_prof_key], v_tau_mean)
    base_ew = compute_corrected_ew(base_ew_list_db[v_prof_key], emission_list_db[v_prof_key], 
                                   incidence_list_db[v_prof_key], v_tau_base)
    peak_ew = compute_corrected_ew(peak_ew_list_db[v_prof_key], emission_list_db[v_prof_key], 
                                   incidence_list_db[v_prof_key], v_tau_peak)
    if v_prof_key[:2] == 'V1':
        color = 'red'
        v1io_phase.append(phase_angle[0])
        v1io_ews.append(ew[0])
        v1io_base_ews.append(base_ew[0])
        v1io_peak_ews.append(peak_ew[0])
    else:
        color = 'green'
        v2io_phase.append(phase_angle[0])
        v2io_ews.append(ew[0])
        v2io_base_ews.append(base_ew[0])
        v2io_peak_ews.append(peak_ew[0])
    base_str += v_prof_key+'  Mean / '+v_prof_key+'  Base   '+str(ew[0] / base_ew[0])+'\n'
    peak_str += v_prof_key+'  Peak / '+v_prof_key+'  Mean   '+str(peak_ew[0] / ew[0])+'\n'
    peak_base_str += v_prof_key+'  Peak / '+v_prof_key+'  Base   '+str(peak_ew[0] / base_ew[0])+'\n'

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)

coeffs = None
coeffs, _ = plot_phase_curve(phase_list_db['C'], c_base_ew_list, None, None, 'v', 10, '-', 'black', 'black', r'$\frac{\mathrm{Baseline}}{}$')
coeffs, _ = plot_phase_curve(phase_list_db['C'], c_base_ew_list, None, None, 'v', 10, '-', color_cassini, color_cassini, 'Cassini')
_, v1iob_scale = plot_phase_curve(v1io_phase, v1io_base_ews, None, coeffs, 'v', 10, '-', color_voyager1, color_voyager1, 'Voyager 1')
_, v2iob_scale = plot_phase_curve(v2io_phase, v2io_base_ews, None, coeffs, 'v', 10, '-', color_voyager2, color_voyager2, 'Voyager 2')
#_, c_mean_scale = plot_phase_curve(phase_list_db['C'], c_ew_list, c_std_list, coeffs, 'o', 8, '-', 'black', 'black', 'Cassini')
#_, v1io_scale = plot_phase_curve(v1io_phase, v1io_ews, None, coeffs, 'o', 12, '-', 'red', 'red', 'V1IO')
#_, v2io_scale = plot_phase_curve(v2io_phase, v2io_ews, None, coeffs, 'o', 12, '-', 'green', 'green', 'V2IO')
#_, c_peak_scale = plot_phase_curve(phase_list_db['C'], c_peak_ew_list, None, coeffs, '^', 10, '-.', 'black', 'black', 'Cassini Peak')
#_, v1iop_scale = plot_phase_curve(v1io_phase, v1io_peak_ews, None, coeffs, '^', 10, '-.', 'red', 'red', 'V1IO Peak')
#_, v2iop_scale = plot_phase_curve(v2io_phase, v2io_peak_ews, None, coeffs, '^', 10, '-.', 'green', 'green', 'V2IO Peak')
#_, v1_scale = plot_phase_curve(phase_list_db['V1'], v1_ew_list, None, coeffs, 'd', 8, ':', 'red', 'none', 'V1')
#_, v2_scale = plot_phase_curve(phase_list_db['V2'], v2_ew_list, None, coeffs, 'd', 8, ':', 'green', 'none', 'V2')

ax.set_yscale('log')
plt.xlabel(r'Phase angle ($^\circ$)')
plt.ylabel(r'$\tau$-Adjusted Equivalent Width (km)')
plt.title('')
ax.set_xlim(0,180)
ax.set_ylim(.08, 50)
ax.get_xaxis().set_ticks([0,90,180])
leg = plt.legend(loc='upper left')

save_fig(fig, ax, 'phase_curve_baseline.png', legend=leg)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)

coeffs = None
coeffs, _ = plot_phase_curve(phase_list_db['C'], c_base_ew_list, None, None, 'v', 10, '-', 'black', 'black', r'$\frac{\mathrm{Baseline}}{}$')
coeffs, _ = plot_phase_curve(phase_list_db['C'], c_base_ew_list, None, None, 'v', 10, '-', color_cassini, color_cassini, 'Cassini')
_, v1iob_scale = plot_phase_curve(v1io_phase, v1io_base_ews, None, coeffs, 'v', 10, '-', color_voyager1, color_voyager1, 'Voyager 1')
_, v2iob_scale = plot_phase_curve(v2io_phase, v2io_base_ews, None, coeffs, 'v', 10, '-', color_voyager2, color_voyager2, 'Voyager 2')
_, _ = plot_phase_curve(phase_list_db['C'], c_peak_ew_list, None, coeffs, '^', 10, '--', 'black', 'black', r'$\frac{\mathrm{Peak}}{}$')
_, c_peak_scale = plot_phase_curve(phase_list_db['C'], c_peak_ew_list, None, coeffs, '^', 10, '--', color_cassini, color_cassini, 'Cassini')
_, v1iop_scale = plot_phase_curve(v1io_phase, v1io_peak_ews, None, coeffs, '^', 10, '--', color_voyager1, color_voyager1, 'Voyager 1')
_, v2iop_scale = plot_phase_curve(v2io_phase, v2io_peak_ews, None, coeffs, '^', 10, '--', color_voyager2, color_voyager2, 'Voyager 2')

ax.set_yscale('log')
plt.xlabel(r'Phase angle ($^\circ$)')
plt.ylabel(r'$\tau$-Adjusted Equivalent Width (km)')
plt.title('')
ax.set_xlim(0,180)
ax.set_ylim(.08, 50)
ax.get_xaxis().set_ticks([0,90,180])
leg = plt.legend(loc='upper left', ncol=2)

save_fig(fig, ax, 'phase_curve_peak.png', legend=leg)
