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
    total_size = (8., 7.0) #cm

    # parameters: IO
    output = 'plots/figures/fig2/fig-gauss-gain-poster.pdf'
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
    gain_lim = (0, 13)


lp_pre_ells, lp_att_ells, lp_dists, lp_dists_px = lineplots.rf_data(
    params.dist_ells, params.focl_ells,
    loc = (56, 56), rad = 1)
# load gain data
sgain_focl = lineplots.gain_data(
    lp_pre_ells, params.sgain_focl, loc = (56, 56))
# # process sizemap/quiver data
# qv_dist_ell, qv_focl_ell, qv_smooth_samp = quivers.quiver_data(
#     lp_pre_ells, lp_att_ells[3], (0, 4, 0), 200)
# load gradient data



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


ax_e = fig.add_subplot(gs[0, 0])
lineplots.lineplot(
    lineplots.gain_file_iterator(
        lp_dists, sgain_focl, (0,4,0)),
    ax_e,
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_b,
    xlim = (0, 180), ylim = params.gain_lim,
    pkws = pkws)
util.labels(ax_e, pkws.labels.unit_distance, pkws.labels.effective_gain, pkws = pkws)
util.legend(
    fig, ax_e, pkws.labels.beta, pkws.pal_b,
    inset = pkws.legend_inset, inset_y = 1.7,
    pkws = pkws)
ax_e.set_yticks([1, params.gain_lim[1]])
ax_e.set_xticks([0, 150])
sns.despine(ax = ax_e, trim = True, offset = 5)
util.axis_expand(ax_e, L = -0.21, B = -0.08, R = 0.1, T = 0.05)

# save
plt.savefig(params.output, transparent = True)











