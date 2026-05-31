import os

import matplotlib.pyplot as plt
import numpy as np
import rebound


EPS = 1e-30

FRING_MEAN_MOTION = 581.964 # deg/day
FRING_A = 140221.3
FRING_E = 0.00235
FRING_W0_DEG = 0  #24.2
FRING_DW_DEGDAY = 0  #2.70025

PROM_MASS = 1.5972e17
PROM_RADIUS = 85.6/2
PROM_A = 139378.
PROM_E = 0.00223
PROM_W0 = 180.
PROM_LON = 180.
PROM_SNAPSHOT_FUDGE = 20.

# Radial extent of F ring sheet
DELTA_A_INNER = 250.
DELTA_A_OUTER = 250.
MIN_A = FRING_A - DELTA_A_INNER
MAX_A = FRING_A + DELTA_A_OUTER

# Longitudinal extent of F ring sheet
MIN_LONG_DEG = 180.
MAX_LONG_DEG = 205.

# Radial and longitudinal spacing between F ring sheet particles
STEP_A = 20.
STEP_LONG_DEG = 0.2


def radius_at_longitude(longitude_rad, et):
    curly_w = FRING_W0_RAD + FRING_DW_RADDAY * et / 86400.

    radius = (FRING_A * (1-FRING_E**2) /
              (1 + FRING_E * np.cos(longitude_rad-curly_w)))

    return radius


FRING_W0_RAD = np.radians(FRING_W0_DEG)
FRING_DW_RADDAY = np.radians(FRING_DW_DEGDAY)
PROM_LON_RAD = np.radians(PROM_LON)
PROM_W0_RAD = np.radians(PROM_W0)

LAST_LINES = []
PROM_NEXT_PERI = None
SNAPSHOT_NUM = 0
PLOT_MAX_A = 0
PLOT_MIN_A = PROM_A * (1-PROM_E) + (FRING_A - radius_at_longitude(PROM_LON_RAD, 0) - PROM_RADIUS)

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
    plt.xlim(MIN_LONG_DEG-10,MAX_LONG_DEG+10)
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
            dir = f'plots/example'
            os.makedirs(dir, exist_ok=True)
            # plt.savefig(f'{dir}/example_{SNAPSHOT_NUM:03d}_apo.png')
            SNAPSHOT_NUM += 1
    elif r < PROM_A * (1-PROM_E) + PROM_SNAPSHOT_FUDGE:
        # Periapse
        if PROM_NEXT_PERI:
            # We took an apoapse, so taking a periapse is OK
            PROM_NEXT_PERI = False
            print('Snap periapse')
            dir = f'plots/example'
            os.makedirs(dir, exist_ok=True)
            # plt.savefig(f'{dir}/example_{SNAPSHOT_NUM:03d}_peri.png')
            SNAPSHOT_NUM += 1

    print(t)


sim = rebound.Simulation()
sim.units = ('km', 's', 'kg')
sim.integrator = "whfast"
sim.dt = 1000  # sec

# Saturn
sim.add(m=5.683e26)
# Prometheus
sim.add(m=PROM_MASS, a=PROM_A, e=PROM_E, omega=PROM_W0_RAD,
        theta=PROM_LON_RAD, r=PROM_RADIUS)

for a in np.arange(MIN_A, MAX_A+EPS, STEP_A):
    e = FRING_E
    w0 = FRING_W0_RAD
    PLOT_MIN_A = min(PLOT_MIN_A, a*(1-e))
    PLOT_MAX_A = max(PLOT_MAX_A, a*(1+e))
    for long_deg in np.arange(MIN_LONG_DEG, MAX_LONG_DEG+EPS, STEP_LONG_DEG):
        sim.add(m=0, a=a, theta=np.radians(long_deg), e=e, omega=w0)

T = 360 / 581.964 * 86400
n_peri = 6
TIME_STEP = 1000 # sec

plot(0)
next_t = 0
while next_t < T * n_peri:
    next_t += TIME_STEP
    sim.integrate(next_t)
    plot(next_t)

plt.pause(5)

# for o in sim.calculate_orbits():
#     print(o)

# op = rebound.OrbitPlot(sim, orbit_style=None)
