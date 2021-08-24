import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from plot import lineplots
from plot import quivers
from plot import behavior
from plot import poster_kwargs as pkws
from plot import util

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

class params:
    # parameters: size
    total_size = (10, 8.3) #cm

    # parameters: IO
    minigrids = True
    output = 'plots/figures/fig5/fig-mim_gauss-quiv-poster.pdf'
    dist_ells = Paths.data('runs/270420/summ_base_ell.csv')
    focl_ells = [
        Paths.data('runs/fig5/summ_mim_gauss_b1.1_ell.csv'),
        Paths.data('runs/fig5/summ_mim_gauss_b2.0_ell.csv'),
        Paths.data('runs/fig5/summ_mim_gauss_b4.0_ell.csv'),
        Paths.data('runs/fig5/summ_mim_gauss_b11.0_ell.csv'),
    ]
    comp_ells = [
        Paths.data('runs/270420/summ_cts_gauss_b1.1_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b2.0_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b4.0_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b11.0_ell.csv'),
    ]
    bhv_focl = {
        ('1.1', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_1.1.h5'),
        ('2.0', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_2.0.h5'),
        ('4.0', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_4.0.h5'),
        ('11.0', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_11.0.h5'),
        ('1.1', 'Flat'): Paths.data('runs/val_rst/bhv_mim_gauss_beta_1.1.h5'),
        ('2.0', 'Flat'): Paths.data('runs/val_rst/bhv_mim_gauss_beta_2.0.h5'),
        ('4.0', 'Flat'): Paths.data('runs/val_rst/bhv_mim_gauss_beta_4.0.h5'),
        ('11.0', 'Flat'): Paths.data('runs/val_rst/bhv_mim_gauss_beta_11.0.h5'),
    }
    bhv_labels = ['Shifted RFs']
    bhv_dist = Paths.data('runs/val_rst/bhv_base.h5')
    sgain_focl = [
        Paths.data('runs/fig5/lenc_mg_n100_b1.1.h5.sgain.npz'),
        Paths.data('runs/fig5/lenc_mg_n100_b2.0.h5.sgain.npz'),
        Paths.data('runs/fig5/lenc_mg_n100_b4.0.h5.sgain.npz'),
        Paths.data('runs/fig5/lenc_mg_n100_b11.0.h5.sgain.npz'),
    ]
    sgain_comp = [
        Paths.data('runs/fig2/lenc_task_gauss_b1.1.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b2.0.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b4.0.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b11.0.h5.sgain.npz'),
    ]

    # axis limits
    size_lim = (0.5, 1.2)
    shift_lim = (-7, 15)
    gain_lim = (0, 10)

# load lineplot data
lp_pre_ells, lp_att_ells, lp_dists, lp_dists_px = lineplots.rf_data(
    params.dist_ells, params.focl_ells,
    loc = (56, 56), rad = 1)
_, lp_comp_ells, _, _ = lineplots.rf_data(
    params.dist_ells, params.comp_ells,
    loc = (56, 56), rad = 1)

# load behavior data
bhv_focl_data, bhv_dist_data = behavior.bhv_data(
    params.bhv_focl, params.bhv_dist, "Dist.")

# process sizemap/quiver data
qv_dist_ell, qv_focl_ell, qv_smooth_samp = quivers.quiver_data(
    lp_pre_ells, lp_att_ells[3], (0, 4, 0), 200)

# load gain data
sgain_focl = lineplots.gain_data(
    lp_pre_ells, params.sgain_focl, loc = (56, 56))
sgain_comp = lineplots.gain_data(
    lp_pre_ells, params.sgain_comp, loc = (56, 56))


import matplotlib
sns.set('notebook')
sns.set_style('ticks')
matplotlib.rcParams.update(pkws.rc)

# make figure
cm = 1/2.54
fig = plt.figure(
    constrained_layout = False,
    figsize = [s*cm for s in params.total_size])

# make gridspec
gs = gridspec.GridSpec(
    nrows = 1, ncols = 1, figure = fig,
    wspace = 0.1, hspace = 0.1)


# panel b
ax_d = fig.add_subplot(gs[0, 0])
size_map = quivers.quiverplot(
    qv_dist_ell, qv_focl_ell, qv_smooth_samp,
    ax_d, cmap = 'coolwarm', vrng = params.size_lim,
    pkws = pkws)
util.axis_expand(ax_d, L = 0.15, B = 0.1, R = -0.1, T = 0.15)
util.labels(ax_d,
    pkws.labels.image_position.format('Horizontal'),
    pkws.labels.image_position.format('Vertical'),
    pkws = pkws)
util.colorbar(
    fig, ax_d, size_map, ticks = params.size_lim + (1,),
    label = pkws.labels.rf_size, label_vofs = -0.09, label_margin = 0.04,
    margin = 0.03, width = 0.03,
    pkws = pkws)


# save
plt.savefig(params.output, transparent = True)











