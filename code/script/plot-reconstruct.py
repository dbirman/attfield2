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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("output")
parser.add_argument("acts_dist")
parser.add_argument("acts_focl")
parser.add_argument("regs")
parser.add_argument("--total_size", nargs = 2, type = float,
    default = (pkws.onecol, 2*pkws.onecol) )
params = parser.parse_args()


# class params:
#     # parameters: size
#     total_size = (pkws.onecol, 2*pkws.onecol) #cm

#     # parameters: IO
#     output = 'plots/runs/acuity/reconstruct_mccw.pdf'
#     acts_dist = Paths.data('runs/acuity/fnenc_task_base.h5')
#     acts_focl = Paths.data('runs/acuity/enc_task_gauss_b4.0.h5')
#     regs = Paths.data('models/logregs_mccw_iso224_t50.npz')


# ----------------  load data  ----

# load encoding/readout data
regs = readouts.load_logregs(params.regs)
readout_data = readouts.readout_data(
    params.acts_dist, params.acts_focl,
    (0, 4, 3))



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
    height_ratios = [3, 2],
    **pkws.onecol_gridspec)
fine_gc =  gridspec.GridSpec(
    nrows = 6, ncols = 3, figure = fig,
    wspace = 0.3, hspace = 0.3,
    left = 0.1, right = 0.9,
    bottom = 0.06, top = 0.94)
base_a = util.panel_label(fig, fine_gc[0, 0], "a")
base_b = util.panel_label(fig, fine_gc[0, 2], "b")
base_c = util.panel_label(fig, gs[1, 0], "c")

# ----------------  readout panel  ----

# panel a
ax_b = fig.add_subplot(fine_gc[0, 2])
gain_mappable = readouts.gain_map(ax_b, readout_data)
util.labels(ax_b, "Layer 4 Feature Map", "")
util.colorbar(
    fig, ax_b, gain_mappable,
    label = pkws.labels.effective_gain,
    shrink = 0,)

# panel b
ax_c = fig.add_subplot(gs[1, 0])
readouts.reconstructed_bhv(ax_c, readout_data, regs)
util.labels(ax_c, None, pkws.labels.bhv_performance)

# save
plt.savefig(params.output, transparent = True)

