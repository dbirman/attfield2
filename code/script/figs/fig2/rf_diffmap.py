import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import pandas as pd
import numpy as np
import h5py


sns.set_context('paper')
sns.set_style('ticks')


# Parameters
LAYER = '0.4.0'
UNITS = [10, 180, 251, 280]
XLIM = (None, None)
YLIM = (150, 0)
FIGSIZE = (2.1415 * 3, 2.1415 * 2)
FOCL_COLOR = '#d55c00'
DIST_COLOR = '#000000'

# Load input data
pre_grads = h5py.File(Paths.data('runs/270420/rfs_base.h5'), 'r+')
pre_ells = pd.read_csv(Paths.data('runs/270420/summ_base_ell.csv'))
post_grads = h5py.File(Paths.data('runs/270420/rfs_cts_gauss_beta_11.0.h5'), 'r+')
post_ells = pd.read_csv(Paths.data('runs/270420/summ_cts_gauss_b11.0_ell.csv'))




# -----------------------------------------------------------  Action to plot an RF  ----


# pre_gmap = pre_grads[f'grads_{LAYER}_0.0'][UNIT, ...].sum(axis = 0)
# post_gmap = post_grads[f'grads_{LAYER}_0.0'][UNIT, ...].sum(axis = 0)
# pre_ell = pre_ells.loc[pre_ells['unit'].map(lambda s: s.startswith(LAYER))
#     ].iloc[UNIT]
# post_ell = post_ells.loc[post_ells['unit'].map(lambda s: s.startswith(LAYER))
#     ].iloc[UNIT]

def plot_rf_heat(grads, LAYER, UNIT, color, alpha_mult):
    gmap = grads[f'grads_{LAYER}_0.0'][UNIT, ...].sum(axis = 0)

    # gradient heatmap
    dat = gmap / gmap.max()
    img_color = np.array(cm.colors.to_rgb(color) + (1.,))
    img = np.ones(dat.shape + (4,)) * img_color[None, None, :]  
    img[..., 3] = dat * alpha_mult  # Apply alpha to flat-color image
    plt.imshow(img)


def plot_rf_ell(ells, LAYER, UNIT, color):
    ell = ells.loc[ells['unit'].map(lambda s: s.startswith(LAYER))
        ].iloc[UNIT]

    # center of mass
    plt.plot([ell['com_x']], [ell['com_y']],
        ls = '', marker = 'o', ms = 9,
        color = 'w', zorder = 2)
    plt.plot([ell['com_x']], [ell['com_y']],
        ls = '', marker = 'o', ms = 6,
        color = color, zorder = 3)

    # ellipse
    ell_args = (
        [ell['com_x'], ell['com_y']],
        3 * np.sqrt(ell.minor_sigma), 3 * np.sqrt(ell.major_sigma),
        np.arctan2(ell.major_x, ell.major_y) * 180/np.pi)
    dist_rf_highlight = Ellipse(*ell_args,
        fc = (1,1,1,0), ec = (1,1,1,0.5), lw = 4)
    dist_rf = Ellipse(*ell_args,
        fc = (1,1,1,0), ec = color, lw = 2)
    ax.add_artist(dist_rf_highlight)
    ax.add_artist(dist_rf)




# -----------------------------------------------------------------  Plot  ----


fig, ax = plt.subplots(figsize = FIGSIZE)
for unit in UNITS:
    plot_rf_heat(pre_grads, LAYER, unit, DIST_COLOR, 0.5)
for unit in UNITS:
    plot_rf_heat(post_grads, LAYER, unit, FOCL_COLOR, 1.)
for unit in UNITS:
    plot_rf_ell(pre_ells, LAYER, unit, DIST_COLOR)
for unit in UNITS:
    plot_rf_ell(post_ells, LAYER, unit, FOCL_COLOR)

plt.plot([56], [56],
    marker = 'X', color = '#263238', ms = 10,
    mew = 1.5, mec = 'w')

plt.xlim(*XLIM)
plt.ylim(*YLIM)

sns.despine()
plt.tight_layout()
plt.savefig(Paths.plots('figures/fig2/rf_movemap.pdf'), transparent = True)





