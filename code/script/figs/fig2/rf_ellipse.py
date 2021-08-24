import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import h5py


sns.set_context('paper')
sns.set_style('ticks')


# Parameters
LAYER = '0.4.0'
UNIT = 10
COL0 = '#DFE8EC'
COL1 = [0., 0., 0., 0.5]


# Load input data
grads = h5py.File(Paths.data('runs/270420/rfs_base.h5'), 'r+')
ells = pd.read_csv(Paths.data('runs/270420/summ_base_ell.csv'))


# -----------------------------------------------------------  Preprocess  ----
gmap = grads[f'grads_{LAYER}_0.0'][UNIT, ...].sum(axis = 0)
ell = ells.loc[ells['unit'].map(lambda s: s.startswith(LAYER))].iloc[UNIT]


# -----------------------------------------------------------------  Plot  ----

fig, ax = plt.subplots(figsize = (2*2.1415, 2*2.1415))
plt.imshow(gmap, cmap = 'YlGnBu_r')
plt.xlim(0, 112)
plt.ylim(112, 0)

axpts = lambda d, a: [
    ell[f'com_{d}'],
    ell[f'com_{d}'] + ell[f'{a}_{d}'] * np.sqrt(ell[f'{a}_sigma'])] 

# Major axis
plt.plot(axpts('x', 'major'), axpts('y', 'major'),
    ls = '-', lw = 6, marker = '', color = COL1,
    zorder = 2)
plt.plot(axpts('x', 'major'), axpts('y', 'major'),
    ls = '-', lw = 4, marker = '', color = COL0,
    zorder = 3)
# Minor axis
plt.plot(axpts('x', 'minor'), axpts('y', 'minor'),
    ls = '-', lw = 6, marker = '', color = COL1,
    zorder = 2)
plt.plot(axpts('x', 'minor'), axpts('y', 'minor'),
    ls = '-', lw = 4, marker = '', color = COL0,
    zorder = 3)
# center of mass
plt.plot([ell['com_x']], [ell['com_y']],
    ls = '', marker = 'o', ms = 11,
    color = COL1, zorder = 2)
plt.plot([ell['com_x']], [ell['com_y']],
    ls = '', marker = 'o', ms = 8,
    color = COL0, zorder = 3)
# Ellipse
gl = np.arctan2(
    axpts('y', 'major')[1] - ell['com_y'],
    axpts('x', 'major')[1] - ell['com_x'])
for kws in (
        {'ec': COL1, 'lw': 6, 'zorder': 2},
        {'ec': COL0, 'lw': 4, 'zorder': 3}):
    plt_ell = Ellipse(
        xy = (ell[f'com_x'], ell[f'com_y']),
        width = 2*np.sqrt(ell[f'major_sigma']),
        height = 2*np.sqrt(ell[f'minor_sigma']),
        angle = (180 / np.pi) * gl,
        fill = False, **kws)
    ax.add_patch(plt_ell)

plt.axis('off')
plt.tight_layout()
plt.savefig(Paths.plots('figures/fig2/rf_model.pdf'))





