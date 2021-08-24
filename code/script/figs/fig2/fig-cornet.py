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
    output = 'plots/figures/fig2/fig-cornet.pdf'
    bhv_dist = Paths.data('runs/val_rst/bhv_base.h5')
    bhv_focl = {
        ('1.1', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_1.1.h5'),
        ('2.0', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_2.0.h5'),
        ('4.0', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_4.0.h5'),
        ('11.0', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_11.0.h5'),
    }
    bhv_stats_output = Paths.data('runs/fig2/fig-cornet-bhv_stats.csv')


# ----------------  load data  ----

# load behavior data
bhv_focl_data, bhv_dist_data = behavior.bhv_data(
    params.bhv_focl, params.bhv_dist, "Dist.")


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
    nrows = 3, ncols = 2, figure = fig,
    **pkws.twocol_gridspec)
base_a = util.panel_label(fig, gs[0, 0], "a")
base_b = util.panel_label(fig, gs[0, 1], "b")
base_c = util.panel_label(fig, gs[1, 0], "c")
base_d = util.panel_label(fig, gs[2, 0], "d")

# ----------------  top row  ----

# panel a
ax_a = fig.add_subplot(gs[0, 0])
bhv_cis = behavior.bhv_plot(
    bhv_focl_data, bhv_dist_data,
    bar1 = 0.69, bar2 = 0.87,
    ax = ax_a, yrng = pkws.bhv_yrng, pal = pkws.pal_bhv,
    jitter = 0.03, bootstrap_n = 1000)
util.axis_expand(ax_a, L = -0.1, B = 0, R = 0, T = 0)
bhv_cis.to_csv(params.bhv_stats_output)

# save
plt.savefig(params.output, transparent = True)











