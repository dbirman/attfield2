import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from plot import lineplots
from plot import quivers
from plot import behavior
from plot import diagrams
from plot import readouts
from plot import kwargs as pkws
from plot import util

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

class params:
    # parameters: size
    total_size = pkws.twocol_size #cm

    # parameters: IO
    output = 'plots/figures/fig2/fig-gauss.pdf'
    dist_ells = Paths.data('runs/270420/summ_base_ell.csv')
    focl_ells = [
        Paths.data('runs/270420/summ_cts_gauss_b1.1_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b2.0_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b4.0_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b11.0_ell.csv'),
    ]
    sgain_focl = [
        Paths.data('runs/fig2/lenc_task_gauss_b1.1.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b2.0.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b4.0.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b11.0.h5.sgain.npz'),
    ]
    grads_dist = Paths.data('runs/270420/rfs_base.h5')
    grads_focl = Paths.data('runs/270420/rfs_cts_gauss_beta_11.0.h5')
    acts_dist = Paths.data('runs/fig2/fnenc_task_base.h5')
    acts_focl = Paths.data('runs/fig2/enc_task_gauss_b4.0.h5')
    # parameters: what to plot
    layer_plot_file = 3
    rf_diagram_units = [10, 156, 180, 251, 280]

    # axis limits
    size_lim = (0.65, 1.2)
    shift_lim = (-0.9, 15)
    gain_lim = (0, 12.2)


# ----------------  load data  ----

# load rf data
lp_pre_ells, lp_att_ells, lp_dists, lp_dists_px = lineplots.rf_data(
    params.dist_ells, params.focl_ells,
    loc = (56, 56), rad = 1)
# load gain data
sgain_focl = lineplots.gain_data(
    lp_pre_ells, params.sgain_focl, loc = (56, 56))
# process sizemap/quiver data
qv_dist_ell, qv_focl_ell, qv_smooth_samp = quivers.quiver_data(
    lp_pre_ells, lp_att_ells[3], (0, 4, 0), 200)
# load gradient data
grads_dist = diagrams.rf_grad_data(params.grads_dist)
grads_focl = diagrams.rf_grad_data(params.grads_focl)
# load encoding/readout data
readout_data = readouts.readout_data(
    params.acts_dist, params.acts_focl, (0, 4, 3))


# ----------------  make structure  ----

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
    nrows = 3, ncols = 3, figure = fig,
    **pkws.twocol_gridspec)
base_a = util.panel_label(fig, gs[0, 0], "a")
base_b = util.panel_label(fig, gs[0, 1], "b")
base_c = util.panel_label(fig, gs[0, 2], "c")
base_d = util.panel_label(fig, gs[1, 0], "d")
base_e = util.panel_label(fig, gs[1, 1], "e")
base_f = util.panel_label(fig, gs[1, 2], "f")
base_g = util.panel_label(fig, gs[2, 0], "g")
base_h = util.panel_label(fig, gs[2, 1], "h")
base_h = util.panel_label(fig, gs[2, 2], "i")

# ----------------  top row  ----

# panel a
ax_a = fig.add_subplot(gs[0, 0])
diagrams.rf_ellipses(
    ax_a, lp_pre_ells, lp_att_ells[3],
    grads_dist, grads_focl, (0,4,0),
    params.rf_diagram_units, loc = (56, 56),
    color_dist = '#000000', color_focl = '#d55c00')
util.axis_expand(ax_a, L = 0.05, B = 0.1, R = 0.05, T = 0)
util.labels(ax_a,
    pkws.labels.image_position.format('Horizontal'),
    pkws.labels.image_position.format('Vertical'))

# panel b
ax_b = fig.add_subplot(gs[0, 1])
quiver_mappable = quivers.quiverplot(
    qv_dist_ell, qv_focl_ell, qv_smooth_samp,
    ax_b, cmap = 'coolwarm', vrng = params.size_lim)
util.axis_expand(ax_b, L = 0.1, B = 0.1, R = 0, T = 0)
util.labels(ax_b,
    pkws.labels.image_position.format('Horizontal'),
    pkws.labels.image_position.format('Vertical'))
util.colorbar(
    fig, ax_b, quiver_mappable, ticks = params.size_lim + (1,),
    label = pkws.labels.rf_size, label_vofs = -0.04)

# panel c
ax_c = fig.add_subplot(gs[0, 2])
r2_mappable = readouts.r2_map(ax_c, readout_data, vrng = (0.6, 1))
util.axis_expand(ax_c, L = 0.1, B = 0.1, R = 0, T = 0)
util.labels(ax_c,
    pkws.labels.feat_map_position.format('Horizontal'),
    pkws.labels.feat_map_position.format('Vertical'))
util.colorbar(
    fig, ax_c, r2_mappable, ticks = [0.6, 1.],
    label = pkws.labels.feature_r2,)


# ----------------  middle row  ----

# panel d
ax_d = fig.add_subplot(gs[1, 0])
lineplots.lineplot(
    lineplots.rf_layer_iterator(
        'shift', lp_dists, lp_att_ells[params.layer_plot_file]),
    ax_d,
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_l,
    xlim = pkws.layerplot_xlim, ylim = params.shift_lim)
util.labels(ax_d, pkws.labels.unit_distance, pkws.labels.rf_shift)
util.legend(fig, ax_d, pkws.labels.layer, pkws.pal_l, inset = 0.08)

# panel e
ax_e = fig.add_subplot(gs[1, 1])
lineplots.lineplot(
    lineplots.rf_layer_iterator(
        'size', lp_dists, lp_att_ells[params.layer_plot_file]),
    ax_e,
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_l,
    xlim = pkws.layerplot_xlim, ylim = params.size_lim) #ylim = pkws.small_size_ylim
util.labels(ax_e, pkws.labels.unit_distance, pkws.labels.rf_size)

# panel f
gs_f = gs[1,2].subgridspec(4, 2, **pkws.mini_gridspec,
    width_ratios = [2, 1])
ax_f = np.array([fig.add_subplot(gs_f[i, 1]) for i in range(4)])
lineplots.mini_lineplot(
    lineplots.gain_layer_iterator(
        lp_dists, sgain_focl[params.layer_plot_file]),
    ax_f.ravel(),
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_l,
    xlim = pkws.lineplot_xlim, ylim = params.gain_lim)
ax_f = fig.add_subplot(gs_f[:, 0])
lineplots.lineplot(
    lineplots.gain_layer_iterator(
        lp_dists, sgain_focl[params.layer_plot_file]),
    ax_f,
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_l,
    xlim = pkws.lineplot_xlim, ylim = params.gain_lim)
util.labels(ax_f, pkws.labels.unit_distance, pkws.labels.effective_gain)


# ----------------  bottom row  ----

# panel g
ax_g = fig.add_subplot(gs[2, 0])
lineplots.lineplot(
    lineplots.rf_file_iterator(
        'shift', lp_dists, lp_att_ells, (0,4,0)),
    ax_g,
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_b,
    xlim = pkws.lineplot_xlim, ylim = params.shift_lim)
util.labels(ax_g, pkws.labels.unit_distance, pkws.labels.rf_shift)
util.legend(
    fig, ax_g, pkws.labels.beta, pkws.pal_b,
    inset = pkws.legend_inset)

# panel h
ax_h = fig.add_subplot(gs[2, 1])
lineplots.lineplot(
    lineplots.rf_file_iterator(
        'size', lp_dists, lp_att_ells, (0,4,0)),
    ax_h,
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_b,
    xlim = pkws.lineplot_xlim, ylim = params.size_lim)
util.labels(ax_h, pkws.labels.unit_distance, pkws.labels.rf_size)

# panel i
ax_f = fig.add_subplot(gs[2, 2])
lineplots.lineplot(
    lineplots.gain_file_iterator(
        lp_dists, sgain_focl, (0,4,0)),
    ax_f,
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_b,
    xlim = pkws.lineplot_xlim, ylim = params.gain_lim)
util.labels(ax_f, pkws.labels.unit_distance, pkws.labels.effective_gain)

# save
plt.savefig(params.output, transparent = True)












