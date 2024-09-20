import matplotlib.pyplot as plt
import numpy as np
import sys
if '..' not in sys.path: sys.path.append('../..'); sys.path.append('../../external')

from f_ring_util.f_ring import hg_func

plt.rcParams.update({'font.size': 24})

PARAMS = [
    ('G Ring (Hedman 2005)', ':', 'k', 0.995, 0.643, 0.665, 0.176, 0.035, 0.181),
    ('D68 Ringlet (Hedman 2019)', '--', 'k', 0.995, 0.754, 0.585, 0.151, 0.005, 0.095),
    ('Charming Ringlet (Hedman 2020)', '-.', 'k', 0.643, 0.89, -0.247, 0.11, 0, 0),
    ('F Ring (all obs)', '-', 'r', 0.676, 0.613, 0.048, 1-0.613, 0, 0),
    ('F Ring (3rd quartile)', '-', 'r', 0.676, 0.633, 0.029, 0.367, 0, 0),
    ('F Ring (2nd quartile)', '-', 'g', 0.677, 0.645, 0.021, 0.355, 0, 0),
    ('F Ring (1st quartile)', '-', 'b', 0.679, 0.652, 0.019, 0.348, 0,0 )
]

fig = plt.figure(figsize=(13,7))
xrange = np.linspace(0., 170., 170)

for name, ls, color, g1, w1, g2, w2, g3, w3 in PARAMS:
    hg = hg_func((g1, w1, g2, w2, g3, w3), xrange)
    if w3 != 0:
        ext_name = f'{name:32s} $g_1$={g1:5.3f}, $w_1$={w1:5.3f}; $g_2$={g2:6.3f}, $w_2$={w2:5.3f}; $g_3$={g3:5.3f}'
    else:
        ext_name = f'{name:32s} $g_1$={g1:5.3f}, $w_1$={w1:5.3f}; $g_2$={g2:6.3f}'
    # if w3 != 0:
    #     ext_name = f'{name:32s} {g1:6.3f} @ {w1:.3f}; {g2:6.3f} @ {w2:.3f}; {g3:6.3f} @ {w3:.3f}'
    # else:
    #     ext_name = f'{name:32s} {g1:6.3f} @ {w1:.3f}; {g2:6.3f} @ {w2:.3f}'
    plt.plot(xrange, hg/hg[0], lw=3, ls=ls, color=color, label=ext_name)
plt.legend(loc='upper left', prop={'family': 'Ubuntu Mono', 'size': 13})
plt.xlim(-2,182)
plt.xlabel('Phase Angle (°)')
plt.xticks([0,45,90,135,180])
plt.yscale('log')
plt.yticks([1, 5, 10], labels=['1', '5', '10'])
plt.ylabel('Relative Phase Function')
plt.tick_params('both', length=10, which='major')
plt.tick_params('both', length=5, which='minor')
plt.title('Phase Functions for Dusty Rings', pad=12)
plt.tight_layout()

plt.savefig('h-g.png', dpi=1200)
