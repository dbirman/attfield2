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
    total_size = (22.6, 11.2) #cm

    # parameters: IO
    output = 'plots/figures/fig4/fig-stitch-poster.pdf'
    dist_ells = Paths.data('runs/270420/summ_base_ell.csv')
    focl_ells = [
        Paths.data('runs/fig4/summ_stitch_b1.1_ell.csv'),
        Paths.data('runs/fig4/summ_stitch_b2.0_ell.csv'),
        Paths.data('runs/fig4/summ_stitch_b4.0_ell.csv'),
        Paths.data('runs/fig4/summ_stitch_b11.0_ell.csv'),
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
        ('1.1', 'Flat'): Paths.data('runs/val_rst/bhv_stitch_beta_1.1.h5'),
        ('2.0', 'Flat'): Paths.data('runs/val_rst/bhv_stitch_beta_2.0.h5'),
        ('4.0', 'Flat'): Paths.data('runs/val_rst/bhv_stitch_beta_4.0.h5'),
        ('11.0', 'Flat'): Paths.data('runs/val_rst/bhv_stitch_beta_11.0.h5'),
    }
    bhv_labels = ["Flat Gain"]
    bhv_dist = Paths.data('runs/val_rst/bhv_base.h5')
    sgain_focl = [
        Paths.data('runs/fig4/enc_stitch_b1.1.h5.sgain.npz'),
        Paths.data('runs/fig4/enc_stitch_b2.0.h5.sgain.npz'),
        Paths.data('runs/fig4/enc_stitch_b4.0.h5.sgain.npz'),
        Paths.data('runs/fig4/enc_stitch_b11.0.h5.sgain.npz'),
    ]
    sgain_comp = [
        Paths.data('runs/fig2/lenc_task_gauss_b1.1.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b2.0.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b4.0.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b11.0.h5.sgain.npz'),
    ]

    # axis limits
    size_lim = (0.3, 1.1)
    shift_lim = (-20, 15)
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
    nrows = 1, ncols = 1, figure = fig,)


"""
# panel b
ax_d = fig.add_subplot(gs[1, 0])
size_map = quivers.quiverplot(
    qv_dist_ell, qv_focl_ell, qv_smooth_samp,
    ax_d, cmap = 'coolwarm', vrng = params.size_lim)
util.axis_expand(ax_d, L = 0.2, B = 0.2, R = -0.1, T = 0.05)
util.labels(ax_d,
    pkws.labels.image_position.format('Horizontal'),
    pkws.labels.image_position.format('Vertical'))
util.colorbar(
    fig, ax_d, size_map, ticks = params.size_lim + (1,),
    label = pkws.labels.rf_size, label_vofs = -0.01)

# panel c
gs_c = gs[1,1].subgridspec(4, 2, **pkws.mini_gridspec,
    width_ratios = [2, 1])
# c: breakout
ax_c = np.array([fig.add_subplot(gs_c[i, 1]) for i in range(4)])
lineplots.mini_lineplot(
    lineplots.rf_file_iterator(
        'shift', lp_dists, lp_att_ells, (0,4,0),
        comp_ells = lp_comp_ells),
    ax_c.ravel(),
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_b,
    xlim = pkws.lineplot_xlim, ylim = params.shift_lim)
# c: main panel
ax_c = fig.add_subplot(gs_c[:, 0])
lineplots.lineplot(
    lineplots.rf_file_iterator(
        'shift', lp_dists, lp_att_ells, (0,4,0),
        comp_ells = lp_comp_ells),
    ax_c,
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_b,
    xlim = pkws.lineplot_xlim, ylim = params.shift_lim)
util.labels(ax_c, pkws.labels.unit_distance, pkws.labels.rf_shift)



# panel d
gs_d = gs[1,2].subgridspec(4, 2, **pkws.mini_gridspec,
    width_ratios = [2, 1])
# d: breakout
ax_d = np.array([fig.add_subplot(gs_d[i, 1]) for i in range(4)])
lineplots.mini_lineplot(
    lineplots.rf_file_iterator(
        'size', lp_dists, lp_att_ells, (0,4,0),
        comp_ells = lp_comp_ells),
    ax_d.ravel(),
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_b,
    xlim = pkws.lineplot_xlim, ylim = params.size_lim,
    yticks = params.size_lim)
# d: main panel
ax_d = fig.add_subplot(gs_d[:, 0])
lineplots.lineplot(
    lineplots.rf_file_iterator(
        'size', lp_dists, lp_att_ells, (0,4,0),
        comp_ells = lp_comp_ells),
    ax_d,
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_b,
    xlim = pkws.lineplot_xlim, ylim = params.size_lim)
util.labels(ax_d, pkws.labels.unit_distance, pkws.labels.rf_size)


# panel e
ax_e = fig.add_subplot(gs[2, 0])
lineplots.lineplot(
    lineplots.gain_file_iterator(
        lp_dists, sgain_focl, (0,4,0),
        gain_comp = sgain_comp),
    ax_e,
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_b,
    xlim = (0, 180), ylim = params.gain_lim)
util.labels(ax_e, pkws.labels.unit_distance, pkws.labels.effective_gain)
util.legend(
    fig, ax_e, pkws.labels.beta, pkws.pal_b,
    inset = pkws.legend_inset)
"""

# panel f
ax_f = fig.add_subplot(gs[0, 0])
behavior.bhv_plot(
    bhv_focl_data, bhv_dist_data,
    bar1 = 0.69, bar2 = 0.87,
    ax = ax_f, yrng = pkws.bhv_yrng, pal = pkws.pal_bhv,
    jitter = 0.05, bootstrap_n = 1000, pkws = pkws,
    trim_axes = True, offset_axes = 5, yticks = (0.55, 0.95))
util.legend(fig, ax_f,
    [pkws.labels.gaussian_model] + params.bhv_labels,
    pkws.pal_bhv,
    inset = pkws.legend_inset, inset_y = 0,
    left = True, pkws = pkws)
util.axis_expand(ax_f, L = -0.06, B = -0.11, R = 0.01, T = 0.1)

# save
plt.savefig(params.output, transparent = True)











