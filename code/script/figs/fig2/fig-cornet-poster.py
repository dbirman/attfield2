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
from plot import poster_kwargs as postkws
from plot import util

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

class params:
    # parameters: size
    total_size = (17, 11) #cm

    # parameters: IO
    output = 'plots/figures/fig2/fig-cornet-poster.pdf'
    bhv_dist = Paths.data('runs/val_rst/bhv_base.h5')
    bhv_focl = {
        ('1.1', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_1.1.h5'),
        ('2.0', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_2.0.h5'),
        ('4.0', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_4.0.h5'),
        ('11.0', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_11.0.h5'),
    }


# ----------------  load data  ----

# load behavior data
bhv_focl_data, bhv_dist_data = behavior.bhv_data(
    params.bhv_focl, params.bhv_dist, "Dist.")


# ----------------  make structure  ----

import matplotlib
sns.set('notebook')
sns.set_style('ticks')
matplotlib.rcParams.update(postkws.rc)

# make figure
cm = 1/2.54
fig = plt.figure(
    constrained_layout = False,
    figsize = [s*cm for s in params.total_size])

# make gridspec
gs = gridspec.GridSpec(nrows = 1, ncols = 1, figure = fig)

# ----------------  top row  ----

# panel a
ax_a = fig.add_subplot(gs[0, 0])
behavior.bhv_plot(
    bhv_focl_data, bhv_dist_data,
    bar1 = 0.69, bar2 = 0.87,
    ax = ax_a, yrng = pkws.bhv_yrng, pal = pkws.pal_bhv,
    jitter = 0.08, bootstrap_n = 1000, pkws = postkws,
    trim_axes = True, offset_axes = 5, yticks = (0.55, 0.95))
util.axis_expand(ax_a, L = -0.13, B = -0.11, R = -0.05, T = 0.1)

# save
plt.savefig(params.output, transparent = True)











