import matplotlib.pyplot as plt
import numpy as np
import sys
if '..' not in sys.path: sys.path.append('../..'); sys.path.append('../../external')

from f_ring_util.f_ring import read_ew_stats

plt.rcParams.update({'font.size': 24})

obsdata_0_1 = read_ew_stats('../../data_files/cass_ew_0_1.csv',
                            obslist_filename='CASSINI_OBSERVATION_LIST',
                            obslist_column='For Photometry')

def set_params(ax):
    ax.set_xlim((2003.8-1970)*365, (2018.2-1970)*365)
    ax.tick_params('both', length=10, which='major')

fig, axs = plt.subplots(2, 1, figsize=(13, 9))

ax1, ax2 = axs
ax1.plot(obsdata_0_1['Date'], obsdata_0_1['Mean Phase'], '.', ms=10, color='blue')
ax1.set_ylabel(r'Phase Angle $\alpha$ (°)')
ax1.set_ylim(-5,185)
ax1.set_yticks([0,45,90,135,180])
ax1.tick_params(labelbottom=False)
ax1.set_title('Distribution of Observations vs.\nPhase and Emission Angle', pad=12)
set_params(ax1)

ax2.plot(obsdata_0_1['Date'], np.abs(np.cos(np.radians(obsdata_0_1['Mean Emission']))),  '.', ms=10, color='green')
ax2.set_ylabel(r'$\mu=|\cos(e)|$')
ax2.set_ylim(-.02,1.02)
set_params(ax2)

plt.tight_layout()

plt.savefig('coverage.png', dpi=300)
