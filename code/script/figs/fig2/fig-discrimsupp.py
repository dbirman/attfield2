import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np


from plot import readouts
from plot import kwargs as pkws
from plot import behavior
from plot import util
from plot import lineplots


class params:

    # output files
    output = 'plots/figures/fig2/fig-discrimsupp.pdf'
    cis_file = Paths.data("runs/ci_review.txt")

    # fig con fig
    total_size = (pkws.twocol_size[0],  0.45 * pkws.twocol_size[0])

    # input files - discrimination task
    cls_regs = 'data/models/regs_ign112_pair.npz'
    cls_dist_file = 'ssddata/apool/enc_ign_cls_tifc.h5'
    cls_focl_file = 'ssddata/apool/enc_ign_cls_tifc_b4.0.h5'
    cls_scores = "data/runs/flip/clstask_scores.npz"





# ----------------  load data : discrimination  ----



# cls_regs = np.load(params.cls_regs)
# cls_regs = {k: v for k,v in cls_regs.items() if not k.endswith('_auc')}

# cls_readout_data = readouts.readout_data(
#     params.cls_dist_file, params.cls_focl_file,
#     (0, 4, 3))
# cls_scores = readouts.reconstructed_bhv_auc(
#     cls_readout_data, cls_regs,
#     readouts.diff_pct_correct)
# np.savez(params.cls_scores, **cls_scores)
# print("Saved scores to", params.cls_scores)
# exit()

cls_scores = np.load(params.cls_scores)






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
    nrows = 1, ncols = 2, figure = fig,
    width_ratios = [5, 4],
    **{**pkws.onecol_gridspec, 'left': 0.05, 'top': 0.9, 'bottom': 0.1, 'right': 0.95})
base_a = util.panel_label(fig, gs[0, 0], "a", xoffset = 0.05)
base_b = util.panel_label(fig, gs[0, 1], "b", xoffset = 0)






# ----------------  discrimination task  ----

# panel b: cls task
ax_b = fig.add_subplot(gs[0, 1])
cls_bhv_cis = readouts.reconstructed_bhv_plot(ax_b, cls_scores)
ax_b.set_ylim(0.68, 1.0)
import matplotlib.ticker as mtick
ax_b.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1., decimals = 0))
sns.despine(ax = ax_b)
util.labels(ax_b, None, pkws.labels.discrim_performance.format("category discrimination"))
behavior.update_ci_text(params.cis_file,
    ClsMultPerformance = behavior.group_ci_text(cls_bhv_cis, 'dist_fake', '', col = 'cond', col_suffix = '_b'),
    ClsMultFX = behavior.group_ci_text(cls_bhv_cis, 'dist_fake', 'fx_', col = 'cond'),
    ClsFakeDrop = behavior.group_ci_text(cls_bhv_cis, 'focl_fake', 'fx_', col = 'cond'),
    ClsUndoPerformance = behavior.group_ci_text(cls_bhv_cis, 'dist_undo', '', col = 'cond', col_suffix = '_b'),
    ClsUndoFX = behavior.group_ci_text(cls_bhv_cis, 'dist_undo', 'fx_', col = 'cond'),)


# save
plt.savefig(params.output, transparent = True)

