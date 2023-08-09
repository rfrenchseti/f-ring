import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'external'))

import matplotlib.pyplot as plt
import numpy as np

import gravity


TWOPI = 2*np.pi
EPS = 1e-20


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
    return f

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

def run_scenario(a0, e0, w0, delta_a, ge, gw, prom, n_orbits, verbose=False):
    gw = np.radians(gw)
    prom = np.radians(prom)

    a1 = a0 + delta_a
    e1 = e0 + ge * delta_a
    w1 = w0 + gw * delta_a

    n0 = SATURN.n(a0, e0)
    n1 = SATURN.n(a1, e1)

    T0 = TWOPI / n0
    T1 = TWOPI / n1

    Tperi0 = T0 * (w0 / TWOPI)
    Tperi1 = T1 * (w1 / TWOPI)

    # The channel-streamer was "vertical" (meaning f0=f1) at the longitude
    # "prom". We work backwards to find the time offset such that at T=0,
    # f0=f1.
    E0back = E_from_f(prom, e0)
    E1back = E_from_f(prom, e1)
    M0back = M_from_E(E0back, e0)
    M1back = M_from_E(E1back, e1)
    Toff0 = t_from_M(M0back, n0)
    Toff1 = t_from_M(M1back, n1)
    print(Toff0, Toff1)

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

        if abs(slope) > 0.1:
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


def plot_om_gradient_vs_a():
    plt.figure()

    a0 = 140221.3
    delta_a = 20
    e0 = 0.00235
    ge = 0
    w0 = 0
    gw = 30 / 500

    for delta_a in np.arange(0, 500+EPS, 20):
        ret = run_scenario(a0, e0, w0, delta_a, ge, gw, 3)
        plt.plot(ret['f0'], ret['slope'], '.', ms=1)

    plt.xlabel('True anomaly (deg)')
    plt.ylabel('Slope (deg/km)')
    plt.xlim(0, 360)
    plt.show()


def plot_e_gradient_vs_a():
    plt.figure()

    a0 = 140221.3
    delta_a = 20
    e0 = 0.00235
    ge = 0.0015 / 500
    w0 = 0
    gw = 0

    for delta_a in np.arange(0, 500+EPS, 20):
        ret = run_scenario(a0, e0, w0, delta_a, ge, gw, 3)
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
                 label=f'{ge:.8f}')

    plt.legend()
    plt.xlabel('True anomaly (deg)')
    plt.ylabel('Slope (deg/km)')
    plt.xlim(0, 360)
    plt.ylim(min_y, max_y)
    plt.title(f'E gradients (deg/km) [Prom f={prom:.3f}]')
    plt.show()


def plot_e_gradients(f, prom, ge_min=-6e-6, ge_max=6e-6):
    plt.figure()

    a0 = 140221.3
    delta_a = 500
    e0 = 0.00235
    w0 = 0
    gw = 0

    for ge in np.arange(ge_min, ge_max, num=10):
        ret = run_scenario(a0, e0, w0, delta_a, ge, gw, prom, 5)
        plt.plot(ge, ret['slope'], '.', ms=1,
                 label=f'{ge:.8f}')

    lgnd = plt.legend()
    for handle in lgnd.legendHandles:
        handle._sizes = [130]
    plt.xlabel('True anomaly (deg)')
    plt.ylabel('Slope (deg/km)')
    plt.xlim(0, 360)
    plt.title(f'E gradients (deg/km)')
    plt.show()



SATURN = gravity.SATURN

# plot_e_gradient_vs_a()
# plot_om_gradient_vs_a()

plot_e_gradients_by_f([0.0015/500], prom=90)
# plot_e_gradients_by_f([-3e-6, -2e-6, -1e-6, 0, 1e-6, 2e-6, 3e-6], prom=328)
# plot_e_gradients_by_f([6e-6, 8e-6, 10e-6, 12e-6], prom=328)
# plot_e_gradients_by_f([2e-6, 3e-6], prom=49)
# plot_e_gradients(f=215, prom=49)
