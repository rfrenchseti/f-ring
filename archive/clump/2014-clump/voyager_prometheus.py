import numpy as np
import numpy.ma as ma
import ringutil
import cspice
import matplotlib.pyplot as plt
import clumputil
import ringimage

def compute_r(a, e, arg): # Takes argument of pericenter
    return a*(1-e**2.) / (1+e*np.cos(arg))
def compute_r_fring(arg):
    return compute_r(fring_a, fring_e, arg)
def compute_r_prom(arg):
    return compute_r(prom_a, prom_e, arg)
def compute_xy_fring(truelong, trueperi):
    r = compute_r_fring(truelong - trueperi)
    return np.array([r*np.cos(truelong), r*np.sin(truelong)])
def compute_xy_prom(truelong, trueperi, offset=0):
    r = compute_r_prom(truelong - trueperi)+offset
    return np.array([r*np.cos(truelong), r*np.sin(truelong)])


n0 = 582.27 # deg/day for S04
v1_epoch = cspice.utc2et('12 NOV 1980 23:46:33')
v2_epoch = cspice.utc2et('26 AUG 1981 03:24:08')

clump_2c_split_et = v2_epoch - 28*86400
dist, corot_long_prometheus = ringutil.prometheus_close_approach(clump_2c_split_et, 0)

print 'OUR COROT', corot_long_prometheus

prom_inertial = ringutil.CorotatingToInertial(corot_long_prometheus, clump_2c_split_et)

print 'INERTIAL', prom_inertial

long_shift = - (n0 * ((clump_2c_split_et - v2_epoch) / 86400.)) % 360.

prom_corot = (prom_inertial + long_shift) % 360.

print 'OLD COROT LONG', prom_corot

start_time = clump_2c_split_et

bosh2002_epoch_et = cspice.utc2et('JD 2451545.0') # J2000
print 'Bosh 2002 epoch', cspice.et2utc(bosh2002_epoch_et, 'C', 0)
bosh2002_fring_a = 140223.7
bosh2002_fring_e = 0.00254
bosh2002_fring_curly = 24.1 * np.pi/180
bosh2002_fring_curly_dot = 2.7001 * np.pi/180 / 86400 # rad/sec

murray97_epoch_et = cspice.utc2et('JD 2444556.490')
murray97_fring_a = 140219.
murray97_fring_e = 0.00279
murray97_fring_curly = 235 * np.pi/180
murray97_fring_curly_dot = 2.7001 * np.pi/180 / 86400 # rad/sec

french2003_epoch_et = cspice.utc2et('JD 2444839.6682')
print 'French 2003 Voyager epoch', cspice.et2utc(french2003_epoch_et, 'C', 0)
french2003_prom_a = 139377.33
french2003_prom_e = .00192
french2003_prom_curly = 228 * np.pi/180

fring_epoch = bosh2002_epoch_et
fring_a = bosh2002_fring_a
fring_e = bosh2002_fring_e
fring_curly0 = bosh2002_fring_curly
fring_curly_dot = bosh2002_fring_curly_dot

#fring_epoch = murray97_epoch_et
#fring_a = murray97_fring_a
#fring_e = murray97_fring_e
#fring_curly0 = murray97_fring_curly
#fring_curly_dot = murray97_fring_curly_dot

#a_prom = 139380  # Jacobson et al. 2008
#e_prom = 0.0022
#precess_prom = 2.7577 * np.pi/180. / 86400. # rad/sec

prom_epoch = french2003_epoch_et
prom_a = french2003_prom_a
prom_e = french2003_prom_e
prom_curly0 = french2003_prom_curly
prom_curly_dot = 2.7577 * np.pi/180. / 86400. # rad/sec Jacobson et al. 2008

#alignment_et = cspice.utc2et('1983 DEC 27 12:00:00') # Aligned - min dist  
alignment_et = cspice.utc2et('1975 MAY 04 12:00:00') # Anti-aligned
#sample_et = cspice.utc2et('2008 DEC 23 01:21:00') # ISS_098RI_TMAP Prom = 262
sample_et = cspice.utc2et('2009 MAY 26 13:17:00') # ISS_111RF_FMOVIE002 Prom = 277
#start_time = alignment_et
#start_time = sample_et
alignment_fring_curly = fring_curly0 + fring_curly_dot * (alignment_et-fring_epoch)
alignment_prom_curly = prom_curly0 + prom_curly_dot * (alignment_et-prom_epoch)
print (alignment_fring_curly*180/np.pi)%360, (alignment_prom_curly*180/np.pi)%360

true_long = np.arange(0., 361., 1.) * np.pi/180.

long_peri_fring = (start_time-fring_epoch) * fring_curly_dot + fring_curly0
long_peri_prom = (start_time-prom_epoch) * prom_curly_dot + prom_curly0

#fring_x, fring_y = compute_xy_fring(true_long, long_peri_fring)
#prom_x, prom_y = compute_xy_prom(true_long, long_peri_prom)
#dist_array = np.sqrt((fring_x-prom_x)**2.0 + (fring_y-prom_y)**2.0)
fring_r = compute_r_fring(true_long-long_peri_fring)
prom_r = compute_r_prom(true_long-long_peri_prom)
dist_array = abs(fring_r-prom_r)
min_idx = np.argmin(dist_array)
min_dist_long = true_long[min_idx]
min_dist = dist_array[min_idx]

prom_inertial = min_dist_long * 180/np.pi

print 'MIN DIST LONG', min_dist_long * 180/np.pi
print 'MIN DIST', min_dist

ru_min_dist, ru_long = ringutil.prometheus_close_approach(start_time, 0)
ru_long = ringutil.CorotatingToInertial(ru_long, start_time)
print 'RU MIN DIST LONG', ru_long
print 'RU MIN DIST', ru_min_dist

long_shift = - (n0 * ((clump_2c_split_et - v2_epoch) / 86400.)) % 360.

prom_corot = (prom_inertial + long_shift) % 360.

print 'OLD COROT LONG', prom_corot

spice_dist_list = []
our_dist_list = []

for et in np.arange(clump_2c_split_et, clump_2c_split_et+10*86400/2, 86400/100):
    spice_dist, spice_long = ringimage.saturn_to_prometheus(et)
    spice_dist_list.append(spice_dist)
    
    spice_long = ringutil.CorotatingToInertial(spice_long, et)
    
#    print 'SPICE DIST', spice_dist, 'LONG', spice_long,
    long_peri_prom = (et-prom_epoch) * prom_curly_dot + prom_curly0

    our_dist = compute_r_prom(spice_long*np.pi/180-long_peri_prom)
    our_dist_list.append(our_dist)
    
#    print 'OUR DIST', our_dist

plt.plot(spice_dist_list, '-', color='red')
plt.plot(our_dist_list, '-', color='green')
plt.show()
