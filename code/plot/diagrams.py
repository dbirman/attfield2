
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import numpy as np
import h5py
from plot import kwargs as pkws


def rf_grad_data(path):
    return h5py.File(path, 'r+')


def plot_rf_heat(ax, grads, LAYER, UNIT, color, alpha_mult):
    gmap = grads[f'grads_{LAYER}_0.0'][UNIT, ...].sum(axis = 0)

    # gradient heatmap
    dat = gmap / gmap.max()
    img_color = np.array(cm.colors.to_rgb(color) + (1.,))
    img = np.ones(dat.shape + (4,)) * img_color[None, None, :]  
    img[..., 3] = dat * alpha_mult  # Apply alpha to flat-color image
    ax.imshow(img)


def plot_rf_ell(ax, ells, LAYER, UNIT, color, pkws):
    ell = ells.loc[ells.index.map(lambda s: s.startswith(LAYER))
        ].iloc[UNIT]

    # center of mass
    ax.plot([ell['com_x']], [ell['com_y']],
        zorder = 2, **pkws.rf_point_outer)
    ax.plot([ell['com_x']], [ell['com_y']],
        color = color, zorder = 3,
        **pkws.rf_point_main)

    # ellipse
    fwhm = 2.355
    ell_args = (
        [ell['com_x'], ell['com_y']],
        fwhm * np.sqrt(ell.minor_sigma), fwhm * np.sqrt(ell.major_sigma),
        np.arctan2(ell.major_x, ell.major_y) * 180/np.pi)
    # dist_rf_highlight = Ellipse(*ell_args,
    #     fc = (1,1,1,0), ec = (1,1,1,0.5), lw = 4)
    dist_rf = Ellipse(*ell_args,
        fc = (1,1,1,0), ec = color, **pkws.rf_ellipse)
    # ax.add_artist(dist_rf_highlight)
    ax.add_artist(dist_rf)


def rf_ellipses(ax, rfs_dist, rfs_focl, grads_dist, grads_focl, layer, units,
                loc, color_dist, color_focl, pkws = pkws):
    lstr = '.'.join(str(i) for i in layer)
    for unit in units:
        plot_rf_heat(ax, grads_dist, lstr, unit, color_dist, 0.9)
    for unit in units:
        plot_rf_heat(ax, grads_focl, lstr, unit, color_focl, 0.9)
    for unit in units:
        plot_rf_ell(ax, rfs_dist, lstr, unit, color_dist, pkws)
    for unit in units:
        plot_rf_ell(ax, rfs_focl, lstr, unit, color_focl, pkws)

    ax.plot([loc[0]], [loc[1]],
        **pkws.rf_locus)

    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    # sns.despine()



