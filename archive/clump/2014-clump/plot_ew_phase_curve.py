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

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['-a']
    
parser = OptionParser()

ringutil.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

polyfit_order = 3
default_coeffs = [6.16620051e-07,  -9.01877510e-05,   5.71805708e-03,  -3.37683617e-01]

#===============================================================================
# 
#===============================================================================

def weighted(x_list, w):
    if w is None:
        return x_list
    assert len(x_list) == len(w)
    new_list = []
    for i in range(len(x_list)):
        for j in range(w[i]):
            new_list.append(x_list[i])
    return new_list

def compute_mu(e):
    return np.abs(np.cos(e*np.pi/180.))

#def calculate_clump_ratio(ets, clumpspec):
#    t0, t1, h, b = clumpspec
#    a = h / (t1-t0)
#    ret_ratio = []
#    for t in ets:
#        if t < t0:
#            ret_ratio.append(1.)
#        elif t < t1:
#            ret_ratio.append(a*(t-t0)+1.)
#        else:
#            ret_ratio.append(1.+h*np.exp(-b*(t-t1)))
#    ret_ratio = np.array(ret_ratio)
#    ret_ratio[np.where(ret_ratio<1e-10)[0]] = 1e-10
#    return ret_ratio

def compute_z(mu, mu0, tau, is_transmission):
    transmission_list = tau*(mu-mu0)/(mu*mu0*(np.exp(-tau/mu)-np.exp(-tau/mu0)))
    reflection_list = tau*(mu+mu0)/(mu*mu0*(1-np.exp(-tau*(1/mu+1/mu0))))
    ret = np.where(is_transmission, transmission_list, reflection_list)
    return ret

def compute_corrected_ew_one(ew, emission, incidence, tau):
    is_transmission = emission > 90.
    mu = compute_mu(emission)
    mu0 = np.abs(np.cos(incidence*np.pi/180))
    ret = ew * compute_z(mu, mu0, tau, is_transmission)
    return ret

def compute_corrected_ew(ew, emission, incidence, tau): # In degrees
    if type(ew) == type([]):
        ret = []
        for i in range(len(ew)):
            ret.append(np.mean(compute_corrected_ew_one(ew[i], emission[i], incidence[i], tau)))
    else:
        ret = compute_corrected_ew_one(np.array(ew), np.array(emission), incidence, tau)
 
    return ret
 
#def compute_corrected_ew_clumpspec(ew, emission, incidence, tau, clumpspec, ets): # In degrees
#    transmission = emission > 90.
#    mu = np.abs(np.cos(emission*np.pi/180))
#    mu0 = np.abs(np.cos(incidence*np.pi/180))
#    ret = ew * compute_z(mu, mu0, tau, transmission)
#    ret /= calculate_clump_ratio(ets, clumpspec)
#    return ret

def optimize_tau(ew_list, phase_list, emission_list, incidence_list, weights=None):
    best_resid = 1e38
    best_tau = None
    for tau in np.arange(0.001,0.5,0.001):
        new_ew = compute_corrected_ew(ew_list, emission_list, incidence_list, tau)
        log_new_ew = np.log10(new_ew)
        coeffs = np.polyfit(weighted(phase_list, weights), weighted(log_new_ew, weights), polyfit_order)
        resid = np.sqrt(np.sum((log_new_ew - np.polyval(coeffs, phase_list))**2))
        if resid < best_resid:
            best_resid = resid
            best_tau = tau
    return best_tau
            
#def optimize_clumpspec(ew_list, phase_list, emission_list, incidence_list, et_list, tau):
#        def clumpspec_func(params, ew_list, phase_list, emission_list, incidence_list, et_list, tau):
#            t0, t1, h, b = params
#            if t0 > t1:
#                return 1e20*(t0-t1)
#            if t0 < np.min(et_list):
#                return -1e20*(t0-np.min(et_list))
#            if t1 > np.max(et_list):
#                return 1e20*(t1-np.max(et_list)) 
#            if h <= 0:
#                return -1e20*h
#            if b <= 0:
#                return -1e30*b
#            if b >= 1e-4:
#                return 1e30*b
#            new_ew = compute_corrected_ew_clumpspec(ew_list, emission_list, incidence_list, tau, params, et_list)
#            log_new_ew = np.log10(new_ew)
#            coeffs = np.polyfit(phase_list, log_new_ew, 3)
#            resid = np.sqrt(np.sum((log_new_ew - np.polyval(coeffs, phase_list))**2))
#            return resid
#
#        ret = sciopt.fmin_powell(clumpspec_func, (np.min(et_list), np.max(et_list), 1, .000001),
#                                 args=(ew_list, phase_list, emission_list, incidence_list, et_list, tau),
#                                 ftol = 1e-20, xtol = 1e-20, disp=False, full_output=False)
#        return ret
#        
#def optimize_tau_clumpspec(ew_list, phase_list, emission_list, incidence_list, et_list):
#    best_resid = 1e38
#    best_tau = None
#    for tau in np.arange(0.001,0.050,0.001):
#        clumpspec = (217771265.18297502, 230029221.32973513, 0.84288311887621625, 8.7843910005748475e-08)
##        clumpspec = optimize_clumpspec(ew_list, phase_list, emission_list, incidence_list, et_list, tau)
#        new_ew = compute_corrected_ew_clumpspec(ew_list, emission_list, incidence_list, tau, clumpspec, et_list)
#        log_new_ew = np.log10(new_ew)
#        coeffs = np.polyfit(phase_list, log_new_ew, polyfit_order)
#        resid = np.sqrt(np.sum((log_new_ew - np.polyval(coeffs, phase_list))**2))
#        print tau, resid, clumpspec
#        if resid < best_resid:
#            best_resid = resid
#            best_tau = tau
#    return best_tau, clumpspec

#===============================================================================
# 
#===============================================================================

def poly_scale_func(params, coeffs, phase_list, log_ew_list):
    scale = params[0]
    return np.sqrt(np.sum((log_ew_list - (np.polyval(coeffs, phase_list) + scale))**2))
    
def plot_phase_curve(phase_list, ew_list, std_list, coeffs, fmt, ms, ls, mec, mfc, label, weights=None):
    log_ew_list = np.log10(ew_list)

    if coeffs is None:
        coeffs = np.polyfit(weighted(phase_list, weights), weighted(log_ew_list, weights), polyfit_order)
        scale = 0.
    else:
        scale = sciopt.fmin_powell(poly_scale_func, (1.,), args=(coeffs, phase_list, log_ew_list),
                                   ftol = 1e-8, xtol = 1e-8, disp=False, full_output=False)

    std_list = None
    plt.errorbar(phase_list, ew_list, yerr=std_list, ecolor=mec, fmt=fmt, ms=ms, mec=mec, mfc=mfc, mew=2, label=label)
    plt.plot(np.arange(0,180), 10**scale*10**np.polyval(coeffs, np.arange(0,180)), ls, lw=2, color=mec,
             label=label)

    return coeffs, 10.**scale

#===============================================================================
# 
#===============================================================================

class ObsData(object):
    pass

obsdata_db = {}

spacecraft_list = ['V1', 'V2', 'C', 'V1I', 'V1O', 'V2I', 'V2O']

for key in spacecraft_list:
    obsdata_db[key] = []
    
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
    obsdata = ObsData()
    obsdata.mean_phase_angle = phase_angle
    obsdata.phase_angles = np.array([phase_angle])
    obsdata.mean_emission_angle = emission_angle
    obsdata.emission_angles = np.array([emission_angle])
    obsdata.mean_incidence_angle = incidence_angle
    obsdata.mean_ew = ew * compute_mu(emission_angle)
    obsdata.ews = np.array([obsdata.mean_ew])
    obsdata_db[spacecraft].append(obsdata)
v_file.close()

total_cassini_obs = 0
used_cassini_obs = 0

for obs_id, image_name, full_path in ringutil.enumerate_files(options, args, obsid_only=True):
#    if (obs_id == 'ISS_036RF_FMOVIE001_VIMS' or
#        obs_id == 'ISS_036RF_FMOVIE002_VIMS' or
#        obs_id == 'ISS_039RF_FMOVIE002_VIMS' or
#        obs_id == 'ISS_039RF_FMOVIE001_VIMS' or
#        obs_id == 'ISS_041RF_FMOVIE002_VIMS' or
#        obs_id == 'ISS_041RF_FMOVIE001_VIMS'):
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

    orig_ew_data = ew_data
    orig_longitudes = longitudes
    orig_emission_angles = emission_angles
    
    ew_data = ma.compressed(ew_data)
    phase_angles = ma.compressed(phase_angles)
    emission_angles = ma.compressed(emission_angles)
    incidence_angles = ma.compressed(incidence_angles)
    ETs = ma.compressed(ETs)
    
    spacecraft = 'C'
    if obs_id[0] == 'V':
        spacecraft = obs_id

    if spacecraft == 'C':
        total_cassini_obs += 1
    
    if np.mean(incidence_angles) > 87:
        print 'SKIPPING EQUINOX', obs_id
        continue

    if spacecraft == 'C':
        used_cassini_obs += 1
    
    mu_ratio = compute_mu(np.min(emission_angles)) / compute_mu(np.max(emission_angles))
    if mu_ratio < 1:
        mu_ratio = 1/mu_ratio
    phase_ratio = 10**np.polyval(default_coeffs, np.max(phase_angles)) / 10**np.polyval(default_coeffs, np.min(phase_angles))
    if mu_ratio > phase_ratio:
        # Prefer emission
        zipped = zip(emission_angles, phase_angles, incidence_angles, ETs, ew_data)
        zipped.sort()
        emission_angles = [a for a,b,c,d,e in zipped]
        phase_angles = [b for a,b,c,d,e in zipped]
        incidence_angles = [c for a,b,c,d,e in zipped]
        ETs = [d for a,b,c,d,e in zipped]
        ew_data = [e for a,b,c,d,e in zipped]
    else:
        # Prefer phase
        zipped = zip(phase_angles, emission_angles, incidence_angles, ETs, ew_data)
        zipped.sort()
        phase_angles = [a for a,b,c,d,e in zipped]
        emission_angles = [b for a,b,c,d,e in zipped]
        incidence_angles = [c for a,b,c,d,e in zipped]
        ETs = [d for a,b,c,d,e in zipped]
        ew_data = [e for a,b,c,d,e in zipped]
    
    phase_angles = np.array(phase_angles)
    emission_angles = np.array(emission_angles)
    incidence_angles = np.array(incidence_angles)
    ETs = np.array(ETs)
    ew_data = np.array(ew_data)
    
    for num_splits in range(1,31):
        split_size = len(ew_data) // num_splits
        is_bad = False
        for split in range(num_splits):
            s_ea = emission_angles[split_size*split:split_size*(split+1)]
            s_pa = phase_angles[split_size*split:split_size*(split+1)]
            mu_ratio = compute_mu(np.min(s_ea)) / compute_mu(np.max(s_ea))
            if mu_ratio < 1:
                mu_ratio = 1/mu_ratio
            phase_ratio = 10**np.polyval(default_coeffs, np.max(s_pa)) / 10**np.polyval(default_coeffs, np.min(s_pa))
#            print obs_id, num_splits, np.min(s_ea), np.max(s_ea), mu_ratio, np.min(s_pa), np.max(s_pa), phase_ratio
            if mu_ratio > 1.2 or phase_ratio > 1.2:
                is_bad = True
            if is_bad:
                break
        if not is_bad:
            break

    split_size = len(ew_data) // num_splits
    
    for split in range(num_splits):
        s_ew_data = ew_data[split_size*split:split_size*(split+1)]
        s_phase_angles = phase_angles[split_size*split:split_size*(split+1)]
        s_emission_angles = emission_angles[split_size*split:split_size*(split+1)]
        s_incidence_angles = incidence_angles[split_size*split:split_size*(split+1)]
        s_ETs = ETs[split_size*split:split_size*(split+1)]
        
        mean_phase = ma.mean(s_phase_angles)
        mean_emission = ma.mean(s_emission_angles)
        mean_et = ma.mean(s_ETs)    
        mean_incidence = ma.mean(s_incidence_angles)
            
        s_ew_data *= compute_mu(s_emission_angles)
    
        sorted_ew_data = np.sort(s_ew_data)
        num_valid = len(s_ew_data)
        perc_idx = int(num_valid * 0.15)
        baseline = sorted_ew_data[perc_idx]
        perc_idx = int(num_valid * 0.95)
        peak = sorted_ew_data[perc_idx]
        
        mean_ew = ma.mean(s_ew_data)
    
        obsdata = ObsData()
        obsdata.obs_id = obs_id
        obsdata.split = split
        obsdata.mean_phase_angle = mean_phase
        obsdata.phase_angles = s_phase_angles
        obsdata.mean_emission_angle = mean_emission
        obsdata.emission_angles = s_emission_angles
        obsdata.mean_incidence_angle = mean_incidence
        obsdata.mean_ew = mean_ew
        obsdata.ews = s_ew_data
        obsdata.baseline_ew = baseline
        obsdata.peak_ew = peak
        obsdata.mean_et = mean_et
    
        percentage_ok = float(len(np.where(longitudes >= 0)[0])) / len(longitudes) * 100
        
        obsdata.percentage_ok = percentage_ok

        obsdata_db[spacecraft].append(obsdata)
        
        print '%-30s/%d %3d%% P %7.3f %7.3f-%7.3f E %7.3f %7.3f-%7.3f I %7.3f %-15s EW %8.5f +/- %8.5f' % (obs_id, split, percentage_ok,
            mean_phase, np.min(s_phase_angles), np.max(s_phase_angles), 
            mean_emission, np.min(s_emission_angles), np.max(s_emission_angles), 
            mean_incidence, cspice.et2utc(mean_et, 'C', 0)[:12], mean_ew, np.std(s_ew_data))
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    plt.plot(orig_longitudes, orig_ew_data*np.abs(np.cos(orig_emission_angles*np.pi/180.)))
#    plt.show()
        
print 'TOTAL CASSINI OBS', total_cassini_obs
print 'USED CASSINI OBS', used_cassini_obs
    
#for key in spacecraft_list:
#    std_list_db[key] = np.where(mean_ew_list_db[key] < std_list_db[key], mean_ew_list_db[key]*.999, std_list_db[key])

c_weights = [len(x.ews)//100 for x in obsdata_db['C']]

v1_log_ew_list = np.log10([x.mean_ew for x in obsdata_db['V1']])
v2_log_ew_list = np.log10([x.mean_ew for x in obsdata_db['V2']])
coeffs = np.polyfit([x.mean_phase_angle for x in obsdata_db['V1']], v1_log_ew_list, polyfit_order)
v1_v2_scale = sciopt.fmin_powell(poly_scale_func, (1.,), args=(coeffs, [x.mean_phase_angle for x in obsdata_db['V2']], v2_log_ew_list),
                                 ftol = 1e-8, xtol = 1e-8, disp=False, full_output=False)
v1_v2_scale = 10**v1_v2_scale
print 'ACTUAL V2/V1 SCALE', v1_v2_scale, 1/v1_v2_scale
v1_v2_scale = .5
print 'ASSUMED V2/V1 SCALE', v1_v2_scale, 1/v1_v2_scale

#c_tau_mean, clumpspec_mean = optimize_tau_clumpspec(ew_list_db['C'], mean_phase_list_db['C'], emission_list_db['C'], mean_incidence_list_db['C'], et_list_db['C'])

c_tau_mean = optimize_tau([x.ews for x in obsdata_db['C']],
                          [x.mean_phase_angle for x in obsdata_db['C']],
                          [x.emission_angles for x in obsdata_db['C']],
                          [x.mean_incidence_angle for x in obsdata_db['C']],
                          weights=c_weights)
#c_tau_peak = optimize_tau([x.peak_ew for x in obsdata_db['C']],
#                          [x.mean_phase_angle for x in obsdata_db['C']],
#                          [x.mean_emission_angle for x in obsdata_db['C']],
#                          [x.mean_incidence_angle for x in obsdata_db['C']])
c_tau_base = optimize_tau([x.baseline_ew for x in obsdata_db['C']],
                          [x.mean_phase_angle for x in obsdata_db['C']],
                          [x.mean_emission_angle for x in obsdata_db['C']],
                          [x.mean_incidence_angle for x in obsdata_db['C']],
                          weights=c_weights)
v_tau = optimize_tau(np.append([x.mean_ew for x in obsdata_db['V1']], np.array([x.mean_ew for x in obsdata_db['V2']])/v1_v2_scale),
                     np.append([x.mean_phase_angle for x in obsdata_db['V1']], [x.mean_phase_angle for x in obsdata_db['V2']]),
                     np.append([x.mean_emission_angle for x in obsdata_db['V1']], [x.mean_emission_angle for x in obsdata_db['V2']]),
                     np.append([x.mean_incidence_angle for x in obsdata_db['V1']], [x.mean_incidence_angle for x in obsdata_db['V2']]))

#v_tau = c_tau
#c_tau = 0.000001
#v_tau = 0.000001

#c_tau_base = 0.034
#c_tau_mean = 0.034
#v_tau = 0.034

#print 'C TAU PEAK', c_tau_peak
print 'ACTUAL C TAU MEAN', c_tau_mean
print 'ACTUAL C TAU BASE', c_tau_base
print 'ACTUAL V TAU', v_tau

v_tau_mean = c_tau_base
v_tau_base = c_tau_base
v_tau_peak = c_tau_base
del v_tau

c_tau_mean = c_tau_base
c_tau_peak = c_tau_base

print 'ASSUMED C TAU PEAK', c_tau_peak
print 'ASSUMED C TAU MEAN', c_tau_mean
print 'ASSUMED C TAU BASE', c_tau_base
print 'ASSUMED V TAU PEAK', v_tau_peak
print 'ASSUMED V TAU MEAN', v_tau_mean
print 'ASSUMED V TAU BASE', v_tau_base

c_mean_phase_list = [x.mean_phase_angle for x in obsdata_db['C']]
v1_mean_phase_list = [x.mean_phase_angle for x in obsdata_db['V1']]
v2_mean_phase_list = [x.mean_phase_angle for x in obsdata_db['V2']]

c_ew_list = compute_corrected_ew([x.ews for x in obsdata_db['C']],
                                 [x.emission_angles for x in obsdata_db['C']],
                                 [x.mean_incidence_angle for x in obsdata_db['C']],
                                 c_tau_mean)
c_mean_ew_list = [np.mean(x) for x in c_ew_list]
c_std_list = [np.std(x) for x in c_ew_list]
del c_ew_list

c_base_ew_list = compute_corrected_ew([x.baseline_ew for x in obsdata_db['C']],
                                      [x.mean_emission_angle for x in obsdata_db['C']],
                                      [x.mean_incidence_angle for x in obsdata_db['C']],
                                      c_tau_base)
c_peak_ew_list = compute_corrected_ew([x.peak_ew for x in obsdata_db['C']],
                                      [x.mean_emission_angle for x in obsdata_db['C']],
                                      [x.mean_incidence_angle for x in obsdata_db['C']],
                                      c_tau_peak)
v1_ew_list = compute_corrected_ew([x.mean_ew for x in obsdata_db['V1']],
                                  [x.mean_emission_angle for x in obsdata_db['V1']],
                                  [x.mean_incidence_angle for x in obsdata_db['V1']],
                                  v_tau_mean)
v2_ew_list = compute_corrected_ew([x.mean_ew for x in obsdata_db['V2']],
                                  [x.mean_emission_angle for x in obsdata_db['V2']],
                                  [x.mean_incidence_angle for x in obsdata_db['V2']],
                                  v_tau_mean)

print
print 'BASELINE DATA'
for i in xrange(len(obsdata_db['C'])):
    obsdata = obsdata_db['C'][i]
    print '%-30s/%d %3d%% P %7.3f %7.3f-%7.3f E %7.3f %7.3f-%7.3f I %7.3f %-15s BASEEW %8.5f' % (obsdata.obs_id, obsdata.split, obsdata.percentage_ok,
            obsdata.mean_phase_angle, np.min(obsdata.phase_angles), np.max(obsdata.phase_angles), 
            obsdata.mean_emission_angle, np.min(obsdata.emission_angles), np.max(obsdata.emission_angles), 
            obsdata.mean_incidence_angle, cspice.et2utc(obsdata.mean_et, 'C', 0)[:12], c_base_ew_list[i])

print

fig = plt.figure()
ax = fig.add_subplot(111)

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
    phase_angle = [x.mean_phase_angle for x in obsdata_db[v_prof_key]]
    ew_list = compute_corrected_ew([x.ews for x in obsdata_db[v_prof_key]],
                                   [x.emission_angles for x in obsdata_db[v_prof_key]],
                                   [x.mean_incidence_angle for x in obsdata_db[v_prof_key]],
                                   v_tau_mean)
    ew_list = [np.mean(x) for x in ew_list]
    base_ew_list = compute_corrected_ew([x.baseline_ew for x in obsdata_db[v_prof_key]],
                                        [x.mean_emission_angle for x in obsdata_db[v_prof_key]],
                                        [x.mean_incidence_angle for x in obsdata_db[v_prof_key]],
                                        v_tau_base)
    peak_ew_list = compute_corrected_ew([x.peak_ew for x in obsdata_db[v_prof_key]],
                                        [x.mean_emission_angle for x in obsdata_db[v_prof_key]],
                                        [x.mean_incidence_angle for x in obsdata_db[v_prof_key]],
                                        v_tau_peak)
    if v_prof_key[:2] == 'V1':
        color = 'red'
        v1io_phase.append(phase_angle[0])
        v1io_ews.append(ew_list[0])
        v1io_base_ews.append(base_ew_list[0])
        v1io_peak_ews.append(peak_ew_list[0])
    else:
        color = 'green'
        v2io_phase.append(phase_angle[0])
        v2io_ews.append(ew_list[0])
        v2io_base_ews.append(base_ew_list[0])
        v2io_peak_ews.append(peak_ew_list[0])
    base_str += v_prof_key+'  Mean / '+v_prof_key+'  Base   '+str(ew_list[0] / base_ew_list[0])+'\n'
    peak_str += v_prof_key+'  Peak / '+v_prof_key+'  Mean   '+str(peak_ew_list[0] / ew_list[0])+'\n'
    peak_base_str += v_prof_key+'  Peak / '+v_prof_key+'  Base   '+str(peak_ew_list[0] / base_ew_list[0])+'\n'

coeffs = None
coeffs, _ = plot_phase_curve(c_mean_phase_list, c_base_ew_list, None, None, 'v', 10, '--', 'black', 'black', 'Cassini Baseline', weights=c_weights)
print coeffs
_, v1iob_scale = plot_phase_curve(v1io_phase, v1io_base_ews, None, coeffs, 'v', 10, '--', 'red', 'red', 'V1IO Baseline')
_, v2iob_scale = plot_phase_curve(v2io_phase, v2io_base_ews, None, coeffs, 'v', 10, '--', 'green', 'green', 'V2IO Baseline')
_, c_mean_scale = plot_phase_curve(c_mean_phase_list, c_mean_ew_list, c_std_list, coeffs, 'o', 8, '-', 'black', 'black', 'Cassini')
_, v1io_scale = plot_phase_curve(v1io_phase, v1io_ews, None, coeffs, 'o', 12, '-', 'red', 'red', 'V1IO')
_, v2io_scale = plot_phase_curve(v2io_phase, v2io_ews, None, coeffs, 'o', 12, '-', 'green', 'green', 'V2IO')
_, c_peak_scale = plot_phase_curve(c_mean_phase_list, c_peak_ew_list, None, coeffs, '^', 10, '-.', 'black', 'black', 'Cassini Peak')
_, v1iop_scale = plot_phase_curve(v1io_phase, v1io_peak_ews, None, coeffs, '^', 10, '-.', 'red', 'red', 'V1IO Peak')
_, v2iop_scale = plot_phase_curve(v2io_phase, v2io_peak_ews, None, coeffs, '^', 10, '-.', 'green', 'green', 'V2IO Peak')
_, v1_scale = plot_phase_curve(v1_mean_phase_list, v1_ew_list, None, coeffs, 'd', 8, ':', 'red', 'none', 'V1')
_, v2_scale = plot_phase_curve(v2_mean_phase_list, v2_ew_list, None, coeffs, 'd', 8, ':', 'green', 'none', 'V2')
#
###_, c_mean_scale = plot_phase_curve(mean_phase_list_db['C'], c_ew_list/calculate_clump_ratio(et_list_db['C'], clumpspec_mean), c_std_list, coeffs, 's', 8, '-', 'red', 'red', 'Cassini')
#
print 'Cass Peak / Cass Base  ', c_peak_scale
print 'V1IO Peak / V1IO Base  ', v1iop_scale / v1iob_scale
print 'V2IO Peak / V12O Base  ', v2iop_scale / v2iob_scale
print peak_base_str
print 'Cass Peak / Cass Mean  ', c_peak_scale / c_mean_scale
print 'V1IO Peak / V1IO Mean  ', v1iop_scale / v1io_scale
print 'V2IO Peak / V12O Mean  ', v2iop_scale / v2io_scale
print peak_str
print 'Cass Mean / Cass Base  ', c_mean_scale
print 'V1IO Mean / V1IO Base  ', v1io_scale / v1iob_scale
print 'V2IO Mean / V12O Base  ', v2io_scale / v2iob_scale
print base_str
print 'Cass Peak / V1IO Peak  ', c_peak_scale / v1iop_scale
print 'Cass Peak / V2IO Peak  ', c_peak_scale / v2iop_scale
print 'V1IO Peak / V2IO Peak  ', v1iop_scale / v2iop_scale 
print
print 'Cass Mean / V1IO Mean  ', c_mean_scale / v1io_scale
print 'Cass Mean / V2IO Mean  ', c_mean_scale / v2io_scale
print 'V1IO Mean / V2IO Mean  ', v1io_scale / v2io_scale
print 'V1   Mean / V1IO Mean  ', v1_scale / v1io_scale
print 'V2   Mean / V2IO Mean  ', v2_scale / v2io_scale
print
print 'Cass Base / V1IO Base  ', 1/v1iob_scale
print 'Cass Base / V2IO Base  ', 1/v2iob_scale
print 'V1IO Base / V2IO Base  ', v1iob_scale / v2iob_scale
print
print 'Cass Mean / V1   Mean  ', c_mean_scale / v1_scale
print 'Cass Mean / V2   Mean  ', c_mean_scale / v2_scale
print 'V1   Mean / V2   Mean  ', v1_scale / v2_scale

#for obsdata in obsdata_db['C']:
#    mu_ratio = compute_mu(np.min(obsdata.emission_angles)) / compute_mu(np.max(obsdata.emission_angles))
#    if mu_ratio < 1:
#        mu_ratio = 1/mu_ratio
#    phase_ratio = 10**np.polyval(coeffs, np.max(obsdata.phase_angles)) / 10**np.polyval(coeffs, np.min(obsdata.phase_angles))
#    print '%-30s MU %7.3f PHASE %7.3f' % (obsdata.obs_id, mu_ratio, phase_ratio)
    
ax.set_yscale('log')
plt.xlabel(r'Phase angle ($^\circ$)')
plt.ylabel(r'Tau-adjusted Mean Equivalent width (km)')
plt.title('')
ax.set_xlim(0,180)
ax.set_ylim(.08, 30)
plt.legend(loc='upper left', ncol=2)

plt.show()
