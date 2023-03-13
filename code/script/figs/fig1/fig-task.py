"""
Post-library TODO:
- label MODEL_DIST_D in parameters
"""

import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from plot import behavior
from plot import kwargs as pkws
from plot import util

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

class params:
    # --- Aesthetics
    total_size = (pkws.onecol, pkws.onecol / .36) #cm
    total_aspect = .44
    pal = [pd.read_csv(Paths.data('cfg/pal_beta.csv'))['color'].values[3], '.3']

    # --- Figure and data outputs
    output = 'plots/figures/fig1/fig-task.pdf'
    cis_file = Paths.data("runs/ci_cmd.txt")

    # --- Data inputs
    human_data_csv = Paths.data('human_dps.csv') # fig-task.md

    # --- General parameters
    MODEL_DIST_D = 0.75


# ----------------  load data  ----

duration_data, curve_data = behavior.human_bhv_data(params.human_data_csv)


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
    nrows = 2, ncols = 1, figure = fig,
    height_ratios = [1/.44, 1/.36 - 1/.44],
    **pkws.twocol_gridspec)
base_a = util.panel_label(fig, gs[0, 0], "a", xoffset = 0.02)

base_b = util.panel_label(fig, gs[1, 0], "b", xoffset = 0.02)

# ----------------  behavior data  ----

# panel b : human behavior
ax_b = fig.add_subplot(gs[1, 0])
match_duration = behavior.human_bhv_plot(ax_b,
    duration_data, curve_data, params.pal,
    yticks = [0, 0.5, 1.0, 1.5, 2.0],
    MODEL_DIST_D = params.MODEL_DIST_D)
util.legend(
    fig, ax_b, ['Focal', 'Distributed'], params.pal,
    inset = pkws.legend_inset, inset_y = 2.6)
util.axis_expand(ax_b, L = -0.16, R = 0.02, B = 0.1, T = -0.05)
behavior.update_ci_text(params.cis_file,
    MatchDuration = '{:.3g}'.format(match_duration))

# save
plt.savefig(params.output, transparent = True)












