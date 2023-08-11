import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import rebound


EPS = 1e-30

FRING_MEAN_MOTION = 581.964 # deg/day
FRING_A = 140221.3
FRING_E = 0.00235
FRING_W0 = 0 #24.2 * oops.RPD # rad
FRING_DW = 0 #2.70025 * oops.RPD # rad/day

PROM_MASS = 1.5972e17
PROM_RAD = 85.6/2
PROM_A = 139380.
PROM_E = 0.0022
PROM_W0 = 180.
PROM_LON = 180.
PROM_SNAPSHOT_FUDGE = 20.

DELTA_A_INNER = 500.
DELTA_A_OUTER = 500.

STEP_A = 10.
MIN_LONG = 180.
MAX_LONG = 195.
STEP_LONG = 0.1

MIN_A = FRING_A - DELTA_A_INNER
MAX_A = FRING_A + DELTA_A_OUTER


ID_SATURN = 0
ID_PROMETHEUS = 1
ID_BACKGROUND_SHEET = 2
ID_COLLISION = 3


LAST_LINES = []
PROM_NEXT_PERI = None
SNAPSHOT_NUM = 0
PLOT_MAX_A = 0
PLOT_MIN_A = 1e38

plt.figure(figsize=(12,8))

def plot(t):
    global LAST_LINES
    for line in LAST_LINES:
        line.remove()
    LAST_LINES = []
    corot_long = t * 581.964 / 86400 # deg/sec
    # plt.figure()

    xdata_by_id = {}
    ydata_by_id = {}
    for p in sim.particles:
        hash = p.hash.value
        if hash == ID_SATURN:
            continue
        r = np.sqrt(p.x**2+p.y**2+p.z**2)
        inertial_long_rad = np.arctan2(p.y, p.x)
        corot_radius = radius_at_longitude(inertial_long_rad, t)
        corot_long_deg = (np.degrees(inertial_long_rad) - corot_long) % 360
        r -= corot_radius
        if hash not in xdata_by_id:
            xdata_by_id[hash] = []
            ydata_by_id[hash] = []
        xdata_by_id[hash].append(corot_long_deg)
        ydata_by_id[hash].append(r)
    for hash in xdata_by_id.keys():
        xdata = xdata_by_id[hash]
        ydata = ydata_by_id[hash]
        if hash == ID_PROMETHEUS:
            pc = plt.scatter(xdata, ydata, s=15, color='green')
        elif hash == ID_BACKGROUND_SHEET:
            pc = plt.scatter(xdata, ydata, s=[1]*len(xdata), color='black')
        elif hash == ID_COLLISION:
            pc = plt.scatter(xdata, ydata, s=2, color='red')
        else:
            assert False, hash
        LAST_LINES.append(pc)
    plt.xlim(MIN_LONG-10,MAX_LONG+10)
    plt.ylim(PLOT_MIN_A-FRING_A, PLOT_MAX_A-FRING_A)
    plt.pause(0.0001)

    print(t)

def radius_at_longitude(longitude, et):
    curly_w = FRING_W0 + FRING_DW*et/86400.

    radius = (FRING_A * (1-FRING_E**2) /
              (1 + FRING_E * np.cos(longitude-curly_w)))

    return radius

def apply_gradient(gradients, a):
    for a_min, a_max, val_min, val_max in gradients:
        if a_min <= a <= a_max:
            return (a-a_min)/(a_max-a_min) * (val_max-val_min) + val_min
    return 0


sim = rebound.Simulation()
sim.units = ('km', 's', 'kg')
sim.integrator = "whfast"
sim.dt = 1000

# Saturn
sim.add(m=5.683e26, hash=ID_SATURN)
# Prometheus
sim.add(m=PROM_MASS, a=PROM_A, e=PROM_E, omega=np.radians(PROM_W0),
        theta=np.radians(PROM_LON), r=PROM_RAD, hash=ID_PROMETHEUS)

for a in np.arange(MIN_A, MAX_A+EPS, STEP_A):
    e = FRING_E
    w0 = FRING_W0
    PLOT_MIN_A = min(PLOT_MIN_A, a*(1-e))
    PLOT_MAX_A = max(PLOT_MAX_A, a*(1+e))
    for long in np.arange(MIN_LONG, MAX_LONG+EPS, STEP_LONG):
        sim.add(m=0, a=a, theta=np.radians(long), e=e, omega=np.radians(w0),
                hash=ID_BACKGROUND_SHEET)

# for n in range(100):
#     long = np.radians((MIN_LONG+MAX_LONG)/2)
#     p = rebound.Particle(sim,
#                          m=0,
#                          a=(MAX_A+MIN_A)/2,
#                          theta=long,
#                          e=FRING_E, omega=np.radians(FRING_W0),
#                          hash=ID_COLLISION)
#     long += np.pi/2
#     dx = -np.sin(long)
#     dy = -np.cos(long)
#     p.vx += dx * n / 5000
#     p.vy += dy * n / 5000
#     sim.add(p)

T = 360 / 581.964 * 86400
n_peri = 3

plot(0)
next_t = 0
while next_t < T * n_peri:
    next_t += 1000
    sim.integrate(next_t)
    plot(next_t)

    for p in sim.particles:
        if p.hash.value == ID_COLLISION:
            print(p.e-FRING_E, p.a-FRING_A)

plt.pause(1000)

# for o in sim.calculate_orbits():
#     print(o)

# op = rebound.OrbitPlot(sim, orbit_style=None)
