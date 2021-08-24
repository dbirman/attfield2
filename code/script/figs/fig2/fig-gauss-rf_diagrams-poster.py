import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from plot import lineplots
from plot import diagrams
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
    total_size = (8, 6) #cm

    # parameters: IO
    output = 'plots/figures/fig2/fig-gauss-rf_diagrams-poster.pdf'
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
    rf_diagram_units = [10, 180, 251, 280]

    # axis limits
    size_lim = (0.65, 1.2)
    shift_lim = (-0.9, 15)
    gain_lim = (0, 12.2)



# load rf data
lp_pre_ells, lp_att_ells, lp_dists, lp_dists_px = lineplots.rf_data(
    params.dist_ells, params.focl_ells,
    loc = (56, 56), rad = 1)
# load gain data
sgain_focl = lineplots.gain_data(
    lp_pre_ells, params.sgain_focl, loc = (56, 56))

grads_dist = diagrams.rf_grad_data(params.grads_dist)
grads_focl = diagrams.rf_grad_data(params.grads_focl)
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


ax_a = fig.add_subplot(gs[0, 0])
diagrams.rf_ellipses(
    ax_a, lp_pre_ells, lp_att_ells[3],
    grads_dist, grads_focl, (0,4,0),
    params.rf_diagram_units, loc = (56, 56),
    color_dist = '#000000', color_focl = '#d55c00', pkws = pkws)
ax_a.set_ylim([150, 0])
util.axis_expand(ax_a, L = 0.0, B = 0.05, R = 0.05, T = 0.1)
util.labels(ax_a,
    "Image space",
    "Image space", pkws = pkws)

# save
plt.savefig(params.output, transparent = True)











