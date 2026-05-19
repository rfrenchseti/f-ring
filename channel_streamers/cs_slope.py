import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'external'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gravity


TWOPI = 2*np.pi
EPS = 1e-20

SATURN = gravity.SATURN

MAX_ORBITS = 9




def r_from_f(f, e):
    return a*(1-e**2) / (1 + e*np.cos(f))

def x_from_r_f(r, f):
    return r * np.cos(f)

def y_from_r_f(r, f):
    return r * np.sin(f)

def M_from_t(t, n, tperi=0):
    return n * (t - tperi)

def t_from_M(M, n, tperi=0):
    return M / n + tperi

def f_from_E(E, e):
    x = np.sqrt((1+e)/(1-e))
    f = np.arctan(x * np.tan(E/2)) * 2
    f = f % TWOPI
    return f

def E_from_f(f, e):
    x = np.sqrt((1+e)/(1-e))
    E = np.arctan(np.tan(f/2) / x) * 2
    E = E % TWOPI
    return E

def r_from_E(E, a, e):
    return a * (1 - e * np.cos(E))

def M_from_E(E, e):
    return E - e * np.sin(E)

def E_from_M(M, e):
    return (M +
            e * np.sin(M) +
            e*e * 0.5 * np.sin(2*M) +
            e*e*e * (0.375 * np.sin(3*M) - 0.125 * np.sin(M)) +
            e*e*e*e * (1/3 * np.sin(4*M) - 1/6 * np.sin(2*M)))


def compute_slopes_at_f(a0, e0, w0, delta_a, ge, gw, prom, f, n_orbits=3, verbose=False):
    gw = np.radians(gw)
    prom = np.radians(prom)
    f = np.radians(f)

    a1 = a0 + delta_a
    e1 = e0 + (ge/a0) * delta_a
    w1 = w0 + gw * delta_a

    n0 = SATURN.n(a0, e0) # rad/sec
    n1 = SATURN.n(a1, e1)

    T0 = TWOPI / n0 # sec
    T1 = TWOPI / n1

    Tperi0 = T0 * (w0 / TWOPI) # Fraction of 2pi times period = fraction of period
    Tperi1 = T1 * (w1 / TWOPI)

    # The channel-streamer was "vertical" (meaning f0=f1) at the longitude
    # "prom". We work backwards to find the time offset such that at T=0,
    # f0=f1.
    E0back = E_from_f(prom, e0)
    E1back = E_from_f(prom, e1)
    M0back = M_from_E(E0back, e0)
    M1back = M_from_E(E1back, e1)
    Toff0 = t_from_M(M0back, n0) - T0
    Toff1 = t_from_M(M1back, n1) - T1

    # We only care about the measurement at one true anomaly
    # First compute the time we're looking at
    E0 = E_from_f(f, e0)
    M0 = M_from_E(E0, e0)
    t = t_from_M(M0, n0, Tperi0) - Toff0 - T0

    slopes = []

    for orbit in range(n_orbits):
        # Now compute the slope at that time
        M0 = M_from_t(t+Toff0, n0, Tperi0)
        E0 = E_from_M(M0, e0)
        f0 = f_from_E(E0, e0)
        r0 = r_from_E(E0, a0, e0)

        M1 = M_from_t(t+Toff1, n1, Tperi1)
        E1 = E_from_M(M1, e1)
        f1 = f_from_E(E1, e1)
        r1 = r_from_E(E1, a1, e1)

        # Bring f1 into the reference frame of f0
        f1 = (f1 + w1-w0) % TWOPI

        slope = (np.degrees(f0-f1) % 360) / (r1-r0)
        slopes.append(slope)

        t += T0

    return slopes


def run_scenario(a0, e0, w0, delta_a, ge, gw, prom, n_orbits, verbose=False):
    gw = np.radians(gw)
    prom = np.radians(prom)

    a1 = a0 + delta_a
    e1 = e0 + (ge/a0) * delta_a
    w1 = w0 + gw * delta_a

    n0 = SATURN.n(a0, e0) # rad/sec
    n1 = SATURN.n(a1, e1)

    T0 = TWOPI / n0 # sec
    T1 = TWOPI / n1

    Tperi0 = T0 * (w0 / TWOPI) # Fraction of 2pi times period = fraction of period
    Tperi1 = T1 * (w1 / TWOPI)

    # The channel-streamer was "vertical" (meaning f0=f1) at the longitude
    # "prom". We work backwards to find the time offset such that at T=0,
    # f0=f1.
    E0back = E_from_f(prom, e0)
    E1back = E_from_f(prom, e1)
    M0back = M_from_E(E0back, e0)
    M1back = M_from_E(E1back, e1)
    Toff0 = t_from_M(M0back, n0) - T0
    Toff1 = t_from_M(M1back, n1) - T1

    times = []
    M0s = []
    M1s = []
    E0s = []
    E1s = []
    r0s = []
    r1s = []
    rds = []
    f0s = []
    f1s = []
    fds = []
    slopes = []

    for t in np.arange(0, n_orbits * T0, T0 / 3600):
        M0 = M_from_t(t+Toff0, n0, Tperi0)
        E0 = E_from_M(M0, e0)
        f0 = f_from_E(E0, e0)
        r0 = r_from_E(E0, a0, e0)

        M1 = M_from_t(t+Toff1, n1, Tperi1)
        E1 = E_from_M(M1, e1)
        f1 = f_from_E(E1, e1)
        r1 = r_from_E(E1, a1, e1)

        # Bring f1 into the reference frame of f0
        f1 = (f1 + w1-w0) % TWOPI

        slope = (np.degrees(f0-f1) % 360) / (r1-r0)

        if abs(slope) > 0.06:
            continue

        times.append(t)
        M0s.append(M0)
        M1s.append(M1)
        E0s.append(E0)
        E1s.append(E1)
        r0s.append(r0)
        r1s.append(r1)
        rds.append(r1-r0)
        f0s.append(np.degrees(f0))
        f1s.append(np.degrees(f1))
        fds.append(np.degrees(f0-f1) % 360)
        slopes.append(slope)

        if verbose:
            print(f'{t/T0:7.3f}  '
                  f'{r0:10.3f} {np.degrees(f0):7.3f}  {r1:10.3f} {np.degrees(f1):7.3f}  '
                  f'{slope:10.5f}')

    ret = {
        'M0': np.array(M0s),
        'M1': np.array(M1s),
        'E0': np.array(E0s),
        'E1': np.array(E1s),
        'r0': np.array(r0s),
        'r1': np.array(r1s),
        'rd': np.array(rds),
        'f0': np.array(f0s),
        'f1': np.array(f1s),
        'fd': np.array(fds),
        'slope': np.array(slopes)
    }

    return ret


def plot_om_gradient_vs_a(prom):
    prom = np.radians(prom)

    plt.figure()

    a0 = 140221.3
    delta_a = 20
    e0 = 0.00235
    ge = 0
    w0 = 0
    gw = 3 / 500

    for delta_a in np.arange(0, 500+EPS, 20):
        ret = run_scenario(a0, e0, w0, delta_a, ge, gw, prom=prom, n_orbits=3)
        plt.plot(ret['f0'], ret['slope'], '.', ms=1)

    plt.xlabel('True anomaly (deg)')
    plt.ylabel('Slope (deg/km)')
    plt.xlim(0, 360)
    plt.show()


def plot_e_gradient_vs_a(prom):
    prom = np.radians(prom)

    plt.figure()

    a0 = 140221.3
    delta_a = 200
    e0 = 0.00235
    ge = 0.0015 / 500
    w0 = 0
    gw = 0

    for delta_a in np.arange(0, 500+EPS, 20):
        ret = run_scenario(a0, e0, w0, delta_a, ge, gw, prom=prom, n_orbits=3)
        plt.plot(ret['f0'], ret['slope'], '.', ms=1)

    plt.xlabel('True anomaly (deg)')
    plt.ylabel('Slope (deg/km)')
    plt.xlim(0, 360)
    plt.show()


def plot_e_gradients_by_f(ges, prom):
    plt.figure()

    a0 = 140221.3
    delta_a = 500
    e0 = 0.00235
    w0 = 0
    gw = 0

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    min_y = 1e38
    max_y = -1e38

    for i, ge in enumerate(ges):
        ret = run_scenario(a0, e0, w0, delta_a, ge, gw, prom, 5)
        plt.plot(ret['f0'], ret['slope'], '.', ms=1, color=colors[i % len(colors)])
        min_y = min(min_y, np.min(ret['slope']))
        max_y = max(max_y, np.max(ret['slope']))
        # All of this hack is to make the legend more readable with a big marker
        plt.plot(-1000, -1000, '.', ms=15, color=colors[i % len(colors)],
                 label=f'{ge:.5f}')

    plt.legend()
    plt.xlabel('True anomaly (deg)')
    plt.ylabel('Slope (deg/km)')
    plt.xlim(0, 360)
    plt.ylim(min_y, max_y)
    plt.title(f'E gradients (a*de/da) [Prom f={prom:.3f}]')
    plt.show()


def plot_e_gradients(f, prom, ge_min=-1, ge_max=1,
                     n_orbits=7, expected_slopes=[],
                     plot_dir=None, gradient_dir=None,
                     obs_id=None, obs_id_root=None, suffix=None):
    plt.figure()

    a0 = 140221.3
    delta_a = 500
    e0 = 0.00235
    w0 = 0
    gw = 0

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    ge_vals = []
    slopes_per_ge = []
    for _ in range(n_orbits):
        slopes_per_ge.append([])

    for ge in np.linspace(ge_min, ge_max, num=1000):
        ge_vals.append(ge)
        slopes = compute_slopes_at_f(a0, e0, w0, delta_a, ge, gw, prom, f,
                                     n_orbits=n_orbits)

        for i, slope in enumerate(slopes):
            if abs(slope) > .06 or slope < 0:
                slope = np.nan
            slopes_per_ge[i].append(slope)

    ge_vals = np.array(ge_vals)

    for i, slopes in enumerate(slopes_per_ge):
        plt.plot(ge_vals, slopes, '.', ms=5,
                 color=colors[i % len(colors)])

    if obs_id_root is not None:
        print(obs_id_root)
    best_mean = None
    for throw_away_first in [False, True]:
        best_std = 1e38
        for phase in range(n_orbits):
            ge_intercepts = []
            slope_intercepts = []
            bad = False
            used_slopes = expected_slopes
            if throw_away_first:
                used_slopes = used_slopes[1:]
            for i, expected_slope in enumerate(used_slopes):
                if i+phase >= len(slopes_per_ge):
                    break
                mask = ~np.isnan(slopes_per_ge[i+phase])
                dist = np.abs(np.array(slopes_per_ge[i+phase])[mask] - expected_slope)
                if len(dist) == 0:
                    bad = True
                    break
                min_x = np.argmin(dist)
                if min_x == 0 or min_x == len(dist)-1:
                    bad = True
                    break
                ge_intercepts.append(ge_vals[mask][min_x])
                slope_intercepts.append(expected_slope)
            if bad:
                continue
            if len(ge_intercepts) == 0:
                continue
            plt.plot(ge_intercepts, slope_intercepts, '-', color='red')
            mean = np.mean(ge_intercepts)
            std = np.std(ge_intercepts)
            if std < best_std:
                best_mean = mean
                best_std = std
            print(f'  Phase {phase:2d} Mean {mean:9.6f} Std {std:9.6f}')
        if best_mean is not None:
            break
        if not throw_away_first:
            print('!!! No result - throwing away first slope !!!')
    if best_mean is None:
        print('!!! No result found !!!')
    elif gradient_dir is not None:
        with open(f'{gradient_dir}/{obs_id_root}_grad.csv', 'w') as fp:
            fp.write(f'{obs_id},{suffix},{best_mean},{best_std}\n')

    for slope in expected_slopes:
        plt.plot([ge_min, ge_max], [slope, slope], '-', color='black')

    plt.xlabel('E gradient (a*de/da)')
    plt.ylabel('Slope (deg/km)')
    plt.title(f'Slopes for E gradients [f={f:.3f} Prom={prom:.3f}]')

    if plot_dir is None:
        plt.show()
    else:
        plt.savefig(f'{plot_dir}/{obs_id_root}.png')
    plt.close()


def interactive_cs(argv):
    # f = float(argv[1])
    # prom = float(argv[2])
    # expected_slopes = [float(x) for x in argv[3:]]
    slope_df = pd.read_csv(argv[1], header=None)
    expected_slopes = slope_df[4].to_numpy()
    metadata_df = pd.read_csv(f'../data_files/cass_ew_0_0_moons.csv',
                              parse_dates=['Date'],
                              index_col='Observation')
    f = metadata_df.loc[argv[0]]['Mean True Anomaly']
    prom = metadata_df.loc[argv[0]]['Prometheus Closest True Anomaly']
    if len(argv) == 3:
        skip_num = int(argv[2])
        expected_slopes = expected_slopes[skip_num:]
    plot_e_gradients(f=f, prom=prom, expected_slopes=expected_slopes[:MAX_ORBITS-1],
                     n_orbits=MAX_ORBITS)

def batch_cs(argv):
    metadata_df = pd.read_csv(f'../data_files/cass_ew_0_0_moons.csv',
                              parse_dates=['Date'],
                              index_col='Observation')

    csv_dir = argv[0]
    plot_dir = argv[1]
    grad_dir = argv[2]

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(grad_dir, exist_ok=True)

    for csv_filename in sorted(os.listdir(csv_dir)):
        if not csv_filename.upper().endswith('.CSV'):
            continue
        print(f'Processing {csv_filename}')

        csv_root = csv_filename.replace('.CSV', '').replace('.csv', '')
        obs_id = obs_id_root = csv_root
        for suffix in ['inner', 'outer']:
            if obs_id.endswith((suffix, suffix.upper())):
                obs_id = obs_id.replace(f'_{suffix}', '')
                break
        else:
            suffix = ''

        try:
            slope_df = pd.read_csv(os.path.join(csv_dir, csv_filename), header=None)
        except:
            print(f'** ERROR READING {csv_filename}')
            continue

        expected_slopes = slope_df[4].to_numpy()
        f = metadata_df.loc[obs_id]['Mean True Anomaly']
        prom = metadata_df.loc[obs_id]['Prometheus Closest True Anomaly']
        plot_e_gradients(f=f, prom=prom, expected_slopes=expected_slopes[:MAX_ORBITS-1],
                         n_orbits=MAX_ORBITS,
                         plot_dir=plot_dir, gradient_dir=grad_dir,
                         obs_id=obs_id, obs_id_root=obs_id_root, suffix=suffix)


# plot_e_gradient_vs_a(0)
# plot_om_gradient_vs_a(0)

# plot_e_gradients_by_f([0.0015/500, 0.003/500], prom=33)
# plot_e_gradients_by_f([-3e-6, -2e-6, -1e-6, 0, 1e-6, 2e-6, 3e-6], prom=328)
# plot_e_gradients_by_f([6e-6, 8e-6, 10e-6, 12e-6], prom=328)
# plot_e_gradients_by_f([2e-6, 3e-6], prom=49)
# plot_e_gradients(f=246, prom=329)
# plot_e_gradients(f=180, prom=270)

if sys.argv[1] == 'batch':
    batch_cs(sys.argv[2:])
else:
    interactive_cs(sys.argv[1:])
