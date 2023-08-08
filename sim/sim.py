import os
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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


gradient_type = sys.argv[1]

GRADIENTS = {
    'no_gradient': {
        'e': [],
        'omega': [],
    },
    'e_outside_const_0.0015': {
        'e': [
                 [FRING_A, MAX_A, 0.0015, 0.0015]
             ],
        'omega': []
    },
    'e_outside_const_-0.0015': {
        'e': [
                 [FRING_A, MAX_A, -0.0015, -0.0015]
             ],
        'omega': []
    },
    'e_outside_grad_0.0015': {
        'e': [
                 [FRING_A, MAX_A, 0, 0.0015]
             ],
        'omega': []
    },
    'e_outside_grad_-0.0015': {
        'e': [
                 [FRING_A, MAX_A, 0, -0.0015]
             ],
        'omega': []
    },
    'e_inside_const_0.0015': {
        'e': [
                 [MIN_A, FRING_A, 0.0015, 0.0015]
             ],
        'omega': []
    },
    'e_inside_const_-0.0015': {
        'e': [
                 [MIN_A, FRING_A, -0.0015, -0.0015]
             ],
        'omega': []
    },
    'e_inside_grad_0.0015': {
        'e': [
                 [MIN_A, FRING_A, 0.0015, 0]
             ],
        'omega': []
    },
    'e_inside_grad_-0.0015': {
        'e': [
                 [MIN_A, FRING_A, -0.0015, 0]
             ],
        'omega': []
    },
    'om_outside_const_30': {
        'e': [],
        'omega': [
                    [FRING_A, MAX_A, 30, 30]
                 ]
    },
    'om_outside_const_-30': {
        'e': [],
        'omega': [
                    [FRING_A, MAX_A, -30, -30]
                 ]
    },
    'om_outside_grad_30': {
        'e': [],
        'omega': [
                    [FRING_A, MAX_A, 0, 30]
                 ]
    },
    'om_outside_grad_-30': {
        'e': [],
        'omega': [
                    [FRING_A, MAX_A, 0, -30]
                 ]
    },
    'om_inside_const_30': {
        'e': [],
        'omega': [
                    [MIN_A, FRING_A, 30, 30]
                 ]
    },
    'om_inside_const_-30': {
        'e': [],
        'omega': [
                    [MIN_A, FRING_A, -30, -30]
                 ]
    },
    'om_inside_grad_30': {
        'e': [],
        'omega': [
                    [MIN_A, FRING_A, 30, 0]
                 ]
    },
    'om_inside_grad_-30': {
        'e': [],
        'omega': [
                    [MIN_A, FRING_A, -30, 0]
                 ]
    },

}



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

    first_particle = True
    xdata = []
    ydata = []
    for p in sim.particles[1:]:
        r = np.sqrt(p.x**2+p.y**2+p.z**2)
        inertial_long_rad = np.arctan2(p.y, p.x)
        corot_radius = radius_at_longitude(inertial_long_rad, t)
        corot_long_deg = (np.degrees(inertial_long_rad) - corot_long) % 360
        r -= corot_radius
        if first_particle:
            # Prometheus
            LAST_LINES.extend(plt.plot(corot_long_deg, r, '.', ms=15, color='red'))
            first_particle = False
        else:
            xdata.append(corot_long_deg)
            ydata.append(r)
    LAST_LINES.extend(plt.plot(xdata, ydata, '.', ms=1, color='black'))
    plt.xlim(MIN_LONG-10,MAX_LONG+10)
    plt.ylim(PLOT_MIN_A-FRING_A, PLOT_MAX_A-FRING_A)
    plt.pause(0.0001)

    # Snap plots at Prometheus closest and furthest approach
    global PROM_NEXT_PERI, SNAPSHOT_NUM
    p = sim.particles[1]
    r = np.sqrt(p.x**2+p.y**2+p.z**2)
    if r > PROM_A * (1+PROM_E) - PROM_SNAPSHOT_FUDGE:
        # Apoapse
        if not PROM_NEXT_PERI:
            # We took a periapse, so taking an apoapse is OK
            PROM_NEXT_PERI = True
            print('Snap apoapse')
            dir = f'plots/{gradient_type}'
            os.makedirs(dir, exist_ok=True)
            plt.savefig(f'{dir}/{gradient_type}_{SNAPSHOT_NUM:03d}_apo.png')
            SNAPSHOT_NUM += 1
    elif r < PROM_A * (1-PROM_E) + PROM_SNAPSHOT_FUDGE:
        # Periapse
        if PROM_NEXT_PERI:
            # We took an apoapse, so taking a periapse is OK
            PROM_NEXT_PERI = False
            print('Snap periapse')
            dir = f'plots/{gradient_type}'
            os.makedirs(dir, exist_ok=True)
            plt.savefig(f'{dir}/{gradient_type}_{SNAPSHOT_NUM:03d}_peri.png')
            SNAPSHOT_NUM += 1

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
sim.add(m=5.683e26)
# Prometheus
sim.add(m=PROM_MASS, a=PROM_A, e=PROM_E, omega=np.radians(PROM_W0),
        theta=np.radians(PROM_LON), r=PROM_RAD)

for a in np.arange(MIN_A, MAX_A+EPS, STEP_A):
    e = FRING_E + apply_gradient(GRADIENTS[gradient_type]['e'], a)
    w0 = FRING_W0 + apply_gradient(GRADIENTS[gradient_type]['omega'], a)
    PLOT_MIN_A = min(PLOT_MIN_A, a*(1-e))
    PLOT_MAX_A = max(PLOT_MAX_A, a*(1+e))
    for long in np.arange(MIN_LONG, MAX_LONG+EPS, STEP_LONG):
        sim.add(m=0, a=a, theta=np.radians(long), e=e, omega=np.radians(w0))

plot(0)
next_t = 0
while next_t < 320000:
    next_t += 1000
    sim.integrate(next_t)
    plot(next_t)


# for o in sim.calculate_orbits():
#     print(o)

# op = rebound.OrbitPlot(sim, orbit_style=None)
