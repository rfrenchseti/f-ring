"""
Minimal demo of the F-ring "streamer-channel" simulation.

Saturn's F ring sits just outside the orbit of the "shepherd" moon Prometheus.
On every Prometheus periapse the moon reaches out toward the ring, gravitationally
kicks the nearby particles, and a few orbits later that kick has sheared into the
diagonal "streamer" and "channel" features visible in Cassini ISS mosaics.

Units throughout: km, seconds, kg, radians for math / degrees for human-facing inputs.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import rebound


# Small offset used to make np.arange include the right endpoint.
EPS = 1e-30

# F-ring orbit (Albers et al. 2012 / our adopted reference orbit).
# Mean motion is only used here to define the co-rotating frame for plotting;
# REBOUND derives its own mean motion from Saturn's mass + the particle's a.
FRING_MEAN_MOTION = 581.964  # deg/day
FRING_A = 140221.3
FRING_E = 0.00235
# Pericenter and pericenter precession rate of the F ring. Set to zero in this
# demo so the ring is a fixed ellipse aligned with the x-axis -- makes the
# co-rotating plot trivially easy to read. The "real" values are in the comments.
FRING_W0_DEG = 0    # 24.2
FRING_DW_DEGDAY = 0  # 2.70025

# Prometheus. Mass and radius from Jacobson et al.; orbit from the same source.
# We put Prometheus at apoapse pointing toward longitude 180 so its periapse
# passages happen at longitude 0 (i.e. *away* from the sheet of test particles
# we lay down between 180 and 205 deg). That way the test particles see
# Prometheus approach from one side and the streamer/channel develops cleanly
# in the plotting window without wrap-around.
PROM_MASS = 1.5972e17
PROM_RADIUS = 85.6 / 2
PROM_A = 139378.
PROM_E = 0.00223
PROM_W0 = 180.
PROM_LON = 180.
# Tolerance (km) used when deciding "Prometheus is near apoapse/periapse" for
# snapshotting. Has to be larger than the per-step radial motion or we'd miss
# the event entirely.
PROM_SNAPSHOT_FUDGE = 20.

# Radial half-width of the test-particle sheet around the F ring. 250 km on each
# side covers the visible streamer/channel features without wasting particles
# on regions too far from the ring to matter.
DELTA_A_INNER = 250.
DELTA_A_OUTER = 250.
MIN_A = FRING_A - DELTA_A_INNER
MAX_A = FRING_A + DELTA_A_OUTER

# Longitudinal extent of the sheet. We only need a narrow longitude window
# because (a) the encounter geometry repeats every Prometheus period and
# (b) integrating a full ring of particles is wasteful for a demo.
MIN_LONG_DEG = 180.
MAX_LONG_DEG = 205.

# Particle spacing. STEP_A=20 km gives ~25 radial lines; STEP_LONG=0.2 deg gives
# ~125 particles per line. ~3000 particles total -- enough to see the streamer
# structure, few enough that the live matplotlib animation stays interactive.
STEP_A = 20.
STEP_LONG_DEG = 0.2


def radius_at_longitude(longitude_rad, et):
    """Radius of the (precessing) F-ring ellipse at a given inertial longitude.

    We subtract this from each particle's radius before plotting so the eccentric
    F ring shows up as a flat horizontal line at r=0. Without this subtraction
    every plot would be dominated by the +/-330 km eccentricity oscillation and
    the much smaller (~tens of km) streamer/channel features would be invisible.
    """
    curly_w = FRING_W0_RAD + FRING_DW_RADDAY * et / 86400.

    radius = (FRING_A * (1 - FRING_E**2) /
              (1 + FRING_E * np.cos(longitude_rad - curly_w)))

    return radius


FRING_W0_RAD = np.radians(FRING_W0_DEG)
FRING_DW_RADDAY = np.radians(FRING_DW_DEGDAY)
PROM_LON_RAD = np.radians(PROM_LON)
PROM_W0_RAD = np.radians(PROM_W0)

# Mutable state shared between plot() calls. Kept as module globals because
# the function is called inside a tight integration loop and we want to avoid
# threading state through arguments.
LAST_LINES = []
PROM_NEXT_PERI = None  # None=haven't snapshotted yet, True=expecting peri next, False=expecting apo
SNAPSHOT_NUM = 0
PLOT_MAX_A = 0
# Lower y-limit of the plot. We pick the lowest point Prometheus reaches
# (its periapse) minus its physical radius, expressed relative to the F ring at
# Prometheus's longitude. This guarantees Prometheus's body is always inside
# the plot box even when it's at closest approach to the ring.
PLOT_MIN_A = PROM_A * (1 - PROM_E) + (FRING_A - radius_at_longitude(PROM_LON_RAD, 0) - PROM_RADIUS)

plt.figure(figsize=(12, 8))


def plot(t):
    """Redraw the co-rotating snapshot at simulation time `t` (seconds).

    Two transformations make the plot legible:
      1. We rotate the longitude axis by the F-ring mean motion so the ring
         stays roughly stationary on screen (otherwise everything would whip
         past at ~580 deg/day).
      2. We subtract `radius_at_longitude` from each particle's r so the ring's
         own eccentricity is removed and only the perturbation is visible.
    """
    global LAST_LINES
    # Remove the previous frame's artists -- much cheaper than calling clf() and
    # rebuilding the figure, and keeps the axes/limits stable for the animation.
    for line in LAST_LINES:
        line.remove()
    LAST_LINES = []
    corot_long = t * 581.964 / 86400  # deg/sec -- subtracted below to enter the F-ring frame

    first_particle = True
    xdata = []
    ydata = []
    # sim.particles[0] is Saturn, [1] is Prometheus, [2:] are the test particles.
    for p in sim.particles[1:]:
        r = np.sqrt(p.x**2 + p.y**2 + p.z**2)
        inertial_long_rad = np.arctan2(p.y, p.x)
        corot_radius = radius_at_longitude(inertial_long_rad, t)
        corot_long_deg = (np.degrees(inertial_long_rad) - corot_long) % 360
        r -= corot_radius
        if first_particle:
            # Prometheus is drawn big and red so it's easy to spot.
            LAST_LINES.extend(plt.plot(corot_long_deg, r, '.', ms=15, color='red'))
            first_particle = False
        else:
            # Batch all test particles into a single plot() call -- one Line2D
            # is far cheaper to redraw every frame than thousands of them.
            xdata.append(corot_long_deg)
            ydata.append(r)
    LAST_LINES.extend(plt.plot(xdata, ydata, '.', ms=1, color='black'))
    plt.xlim(MIN_LONG_DEG - 10, MAX_LONG_DEG + 10)
    plt.ylim(PLOT_MIN_A - FRING_A, PLOT_MAX_A - FRING_A)
    plt.pause(0.0001)

    # Auto-snapshot logic. The streamer/channel pattern reaches its visually
    # cleanest "diagonal stripes" appearance at certain phases of Prometheus's
    # orbit (notably near peri/apoapse), so we save a PNG whenever Prometheus
    # passes through one of those phases. PROM_NEXT_PERI flips state so we only
    # save once per crossing instead of every frame in the FUDGE-wide window.
    global PROM_NEXT_PERI, SNAPSHOT_NUM
    p = sim.particles[1]
    r = np.sqrt(p.x**2 + p.y**2 + p.z**2)
    if r > PROM_A * (1 + PROM_E) - PROM_SNAPSHOT_FUDGE:
        # Apoapse
        if not PROM_NEXT_PERI:
            # We last snapshotted at periapse (or never), so it's OK to snap apoapse.
            PROM_NEXT_PERI = True
            print('Snap apoapse')
            dir = f'plots/example'
            os.makedirs(dir, exist_ok=True)
            # Uncomment when you want to actually save the frames to disk:
            # plt.savefig(f'{dir}/example_{SNAPSHOT_NUM:03d}_apo.png')
            SNAPSHOT_NUM += 1
    elif r < PROM_A * (1 - PROM_E) + PROM_SNAPSHOT_FUDGE:
        # Periapse
        if PROM_NEXT_PERI:
            PROM_NEXT_PERI = False
            print('Snap periapse')
            dir = f'plots/example'
            os.makedirs(dir, exist_ok=True)
            # plt.savefig(f'{dir}/example_{SNAPSHOT_NUM:03d}_peri.png')
            SNAPSHOT_NUM += 1

    print(t)


sim = rebound.Simulation()
sim.units = ('km', 's', 'kg')
# WHFast is the Wisdom-Holman symplectic integrator: very fast and conserves
# energy well for nearly-Keplerian systems like this (one dominant central
# mass plus small perturbations). The price is a fixed timestep and reduced
# accuracy during close encounters -- fine here because Prometheus never
# actually collides with the test particles in this geometry.
sim.integrator = "whfast"
# 1000 s is well under 1% of the orbital period (~14 hr), which is the rule of
# thumb for WHFast accuracy. Smaller dt would be more accurate but not needed
# for this demo.
sim.dt = 1000  # sec

# Saturn (index 0). Mass only -- position defaults to origin.
sim.add(m=5.683e26)
# Prometheus (index 1). Given a physical radius `r` so REBOUND could in principle
# detect collisions, though we don't enable a collision module here.
sim.add(m=PROM_MASS, a=PROM_A, e=PROM_E, omega=PROM_W0_RAD,
        theta=PROM_LON_RAD, r=PROM_RADIUS)

# Lay down the initial sheet of massless test particles. They start on
# perfectly Keplerian F-ring-like orbits (same e and pericenter as the ring);
# any deviation from that in the plots is entirely due to Prometheus.
for a in np.arange(MIN_A, MAX_A + EPS, STEP_A):
    e = FRING_E
    w0 = FRING_W0_RAD
    # Track the apo/peri extremes of the initial sheet so the plot y-limits
    # are wide enough to contain everything we just added.
    PLOT_MIN_A = min(PLOT_MIN_A, a * (1 - e))
    PLOT_MAX_A = max(PLOT_MAX_A, a * (1 + e))
    for long_deg in np.arange(MIN_LONG_DEG, MAX_LONG_DEG + EPS, STEP_LONG_DEG):
        # m=0 makes these test particles: they feel Saturn and Prometheus but
        # don't perturb anything themselves. This is what lets us cheaply run
        # thousands of them.
        sim.add(m=0, a=a, theta=np.radians(long_deg), e=e, omega=w0)

# T = one F-ring rotation period in seconds. We integrate for `n_peri` of them;
# 6 is enough to see several streamer/channel cycles develop without taking
# forever. Increase if you want to watch the channels widen and dissipate.
T = 360 / 581.964 * 86400
n_peri = 6
TIME_STEP = 1000  # sec -- one plot/integration chunk

plot(0)
next_t = 0
while next_t < T * n_peri:
    next_t += TIME_STEP
    # We integrate in small chunks and call plot() each time so we get a live
    # animation. REBOUND's internal step is still sim.dt; this just controls
    # how often we stop to redraw.
    sim.integrate(next_t)
    plot(next_t)

# Keep the window open briefly so the final frame is visible before exit.
plt.pause(5)

# Useful when debugging orbital elements -- prints semi-major axis, e, etc.
# for every particle at the final time.
# for o in sim.calculate_orbits():
#     print(o)

# Alternative visualization built into REBOUND -- shows full orbital ellipses
# instead of point positions. Handy for sanity-checking the initial setup.
# op = rebound.OrbitPlot(sim, orbit_style=None)
