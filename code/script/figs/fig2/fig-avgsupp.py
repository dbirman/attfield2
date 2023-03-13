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
    output = 'plots/figures/fig2/fig-avgsupp.pdf'
    cis_file = Paths.data("runs/ci_review.txt")

    scores_4x4 = Paths.data("runs/apool/4x4_neccsuff_scores.npz")
    coef_file = Paths.data("runs/apool/ign_iso224_coefs.npz")
    scores_7x7 = Paths.data("runs/apool/7x7_neccsuff_scores.npz")

    # fig con fig
    total_size = (pkws.onecol, 0.9 * pkws.onecol)


# ---------------------  load data  ----

scores_4x4 = np.load(params.scores_4x4)
scores_7x7 = np.load(params.scores_7x7)
coef_data = np.load(params.coef_file)


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
    nrows = 1, ncols = 1, figure = fig,
    **{**pkws.onecol_gridspec, 'bottom': 0.1, 'top': 0.92, 'left': 0.15}) # , 'left': 0.15
base_a = util.panel_label(fig, gs[0, 0], "a", xoffset = 0)



# ------------------  4x4 neccsuff  ----

# panel a
ax_a = fig.add_subplot(gs[0, 0])
bhv_cis = readouts.reconstructed_bhv_plot(ax_a, scores_4x4)
util.labels(ax_a, None, "Masked readout performance [AUC (d')]")
behavior.update_ci_text(params.cis_file,
    MaskMultPerformance = behavior.group_ci_text(bhv_cis, 'dist_fake', '', col = 'cond', col_suffix = '_b'),
    MaskMultFX = behavior.group_ci_text(bhv_cis, 'dist_fake', 'fx_', col = 'cond'),
    MaskFakeDrop = behavior.group_ci_text(bhv_cis, 'focl_fake', 'fx_', col = 'cond'),
    MaskUndoPerformance = behavior.group_ci_text(bhv_cis, 'dist_undo', '', col = 'cond', col_suffix = '_b'),
    MaskUndoFX = behavior.group_ci_text(bhv_cis, 'dist_undo', 'fx_', col = 'cond'),)


# --------------------  more stats  ----

dist_performance_headroom = coef_data['apool_auc'] - scores_4x4['score_dist']
median_agg = lambda arr: np.median(arr, axis = 1)
headroom_ci = util.mean_ci(dist_performance_headroom, 1000, median_agg)
print(headroom_ci)
print(np.median(dist_performance_headroom))
behavior.update_ci_text(params.cis_file,
    MaskHeadroom = (headroom_ci[0], np.median(dist_performance_headroom), headroom_ci[1]))

# ------------------  stats for 7x7 analysis  ----

fullmap_cis = readouts.reconstructed_bhv_plot(None, scores_7x7)

aucdrop_7x7 = coef_data['fullmap_auc'] - coef_data['apool_auc']
behavior.update_ci_text(params.cis_file,
    FullmapDistPerformance = behavior.group_ci_text(fullmap_cis, 'dist_fake', '', col = 'cond', col_suffix = '_a'),
    FullmapFoclPerformance = behavior.group_ci_text(fullmap_cis, 'focl_fake', '', col = 'cond', col_suffix = '_a'),
    FullmapMultPerformance = behavior.group_ci_text(fullmap_cis, 'dist_fake', '', col = 'cond', col_suffix = '_b'),
    FullmapMultFX = behavior.group_ci_text(fullmap_cis, 'dist_fake', 'fx_', col = 'cond'),
    FullmapFakeDrop = behavior.group_ci_text(fullmap_cis, 'focl_fake', 'fx_', col = 'cond'),
    FullmapUndoPerformance = behavior.group_ci_text(fullmap_cis, 'dist_undo', '', col = 'cond', col_suffix = '_b'),
    FullmapUndoFX = behavior.group_ci_text(fullmap_cis, 'dist_undo', 'fx_', col = 'cond'),
    # FullmapFoclUndoFX = behavior.group_ci_text(fullmap_cis, 'focl_undo', 'fx_', col = 'cond'),
    CIFullmapAUCDrop = (
        aucdrop_7x7.min(),
        np.median(aucdrop_7x7),
        aucdrop_7x7.max()))


# save
plt.savefig(params.output, transparent = True)

















