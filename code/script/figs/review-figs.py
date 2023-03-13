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
    output = 'plots/runs/apool/fig-review.pdf'
    cis_file = Paths.data("runs/ci_review.txt")

    # fig con fig
    total_size = (pkws.onecol, 2.5 * pkws.onecol)

    # input files - discrimination task
    cls_regs = 'data/models/regs_ign112_pair.npz'
    cls_dist_file = 'ssddata/apool/enc_ign_cls_tifc.h5'
    cls_focl_file = 'ssddata/apool/enc_ign_cls_tifc_b4.0.h5'

    src_regs = 'data/models/opposed_regs_ign112_flip.npz'
    src_dist_file = 'ssddata/apool/enc_ign_manysrc_tifc.h5'
    src_focl_file = 'ssddata/apool/enc_ign_manysrc_tifc_b4.0.h5'

    # input files - rf shrinkage
    dist_ells = Paths.data('runs/270420/summ_base_300L4_ell.csv')
    shk_focl_ells = [
        Paths.data('runs/shrink/rfs_shrink_b0.3_ell.csv'),
    ]
    shk_comp_ells = [
        Paths.data('runs/270420/summ_cts_gauss_300L4_b11.0_ell.csv'),
    ]
    shk_bhv_focl = {
        ('0.1', 'Shrink'): Paths.data('runs/shrink/bhv_shrink_beta_0.1.h5'),
        ('0.3', 'Shrink'): Paths.data('runs/shrink/bhv_shrink_beta_0.3.h5'),
    }
    bhv_labels = ['Squeezed RFs']
    bhv_dist = Paths.data('runs/val_rst/bhv_base.h5')
    # bhv_stats_output = Paths.data('runs/fig5/fig-mg-bhv_stats.csv')

    # axis limits
    size_lim = (0.65, 1.2)
    


# ----------------  load data : discrimination  ----

cls_regs = np.load(params.cls_regs)
cls_regs = {k: v for k,v in cls_regs.items() if not k.endswith('_auc')}
# cls_readout_data = readouts.readout_data(
#     params.cls_dist_file, params.cls_focl_file,
#     (0, 4, 3))

src_regs = np.load(params.src_regs)
cats = [l[:-1] for l in open('data/imagenet/cats.txt', 'r').readlines()]
src_regs = {c: src_regs[f'{c}:uprt'] for c in cats}
# src_readout_data = readouts.readout_data(
#     params.src_dist_file, params.src_focl_file,
#     (0, 4, 3))


# cls_scores = readouts.reconstructed_bhv_auc(
#     cls_readout_data, cls_regs,
#     readouts.diff_pct_correct)
# src_scores = readouts.reconstructed_bhv_auc(
#     src_readout_data, src_regs,
#     readouts.diff_pct_correct)

# import pickle
# pickle.dump(
#     [cls_scores, src_scores],
#     open('/tmp/tifc_scores.pkl', 'wb'))
import pickle
cls_scores, src_scores = pickle.load(open('/tmp/tifc_scores.pkl', 'rb'))



# ----------------  load data : rf shrink  ----


lp_pre_ells, lp_att_ells, lp_dists, lp_dists_px = lineplots.rf_data(
    params.dist_ells, params.shk_focl_ells,
    loc = (56, 56), rad = 1)
_, lp_comp_ells, _, _ = lineplots.rf_data(
    params.dist_ells, params.shk_comp_ells,
    loc = (56, 56), rad = 1)

shk_bhv_focl_data, bhv_dist_data = behavior.bhv_data(
    params.shk_bhv_focl, params.bhv_dist, "Dist.")







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
    nrows = 5, ncols = 2, figure = fig,
    **{**pkws.onecol_gridspec, 'left': 0.15, 'hspace': 0.4})
base_a = util.panel_label(fig, gs[0, :], "a", xoffset = 0)
base_b = util.panel_label(fig, gs[1, :], "b", xoffset = 0)
base_c = util.panel_label(fig, gs[2, :], "c", xoffset = 0)
base_d = util.panel_label(fig, gs[3, :], "d", xoffset = 0)






# ----------------  discrimination task  ----

# panel a: manysrc task
ax_a = fig.add_subplot(gs[0, :])
src_bhv_cis = readouts.reconstructed_bhv_plot(ax_a, src_scores)
util.labels(ax_a, None, pkws.labels.discrim_performance.format("flip discrimination"))
behavior.update_ci_text(params.cis_file,
    FlipMultPerformance = behavior.group_ci_text(src_bhv_cis, 'dist_fake', '', col = 'cond', col_suffix = '_b'),
    FlipMultFX = behavior.group_ci_text(src_bhv_cis, 'dist_fake', 'fx_', col = 'cond'),
    FlipFakeDrop = behavior.group_ci_text(src_bhv_cis, 'focl_fake', 'fx_', col = 'cond'),
    FlipUndoPerformance = behavior.group_ci_text(src_bhv_cis, 'dist_undo', '', col = 'cond', col_suffix = '_b'),
    FlipUndoFX = behavior.group_ci_text(src_bhv_cis, 'dist_undo', 'fx_', col = 'cond'),)


# panel b: cls task
ax_b = fig.add_subplot(gs[1, :])
cls_bhv_cis = readouts.reconstructed_bhv_plot(ax_b, cls_scores)
util.labels(ax_b, None, pkws.labels.discrim_performance.format("class discrimination"))
behavior.update_ci_text(params.cis_file,
    ClsMultPerformance = behavior.group_ci_text(cls_bhv_cis, 'dist_fake', '', col = 'cond', col_suffix = '_b'),
    ClsMultFX = behavior.group_ci_text(cls_bhv_cis, 'dist_fake', 'fx_', col = 'cond'),
    ClsFakeDrop = behavior.group_ci_text(cls_bhv_cis, 'focl_fake', 'fx_', col = 'cond'),
    ClsUndoPerformance = behavior.group_ci_text(cls_bhv_cis, 'dist_undo', '', col = 'cond', col_suffix = '_b'),
    ClsUndoFX = behavior.group_ci_text(cls_bhv_cis, 'dist_undo', 'fx_', col = 'cond'),)



# ----------------  rf shrink attention  ----


# panel c: cls task
ax_c = fig.add_subplot(gs[2, :])
lineplots.lineplot(
    lineplots.rf_file_iterator(
        'size', lp_dists, lp_att_ells, (0,4,0),
        comp_ells = lp_comp_ells),
    ax_c,
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_b.values[-1:],
    xlim = pkws.lineplot_xlim, ylim = params.size_lim)
util.labels(ax_c, pkws.labels.unit_distance, pkws.labels.rf_size)
util.legend(
    fig, ax_c,
    ['Max shift'] + [0.3],
    np.concatenate([[pkws.legend_header_color], pkws.pal_b.values[-1:]]),
    inset = pkws.legend_inset,
    inset_y = 2.2)


ax_d = fig.add_subplot(gs[3, :])
bhv_cis = behavior.bhv_plot(
    shk_bhv_focl_data, bhv_dist_data,
    bar1 = behavior.d2auc(0.75), bar2 = behavior.d2auc(1.28),
    ax = ax_d, yrng = pkws.bhv_yrng, pal = pkws.pal_bhv[1:],
    jitter = 0.03, bootstrap_n = 1000)
ax_d.set_ylim(pkws.bhv_yrng) # not properly set in bhv_plot for some reason
util.legend(fig, ax_d,
    params.bhv_labels,
    pkws.pal_bhv[1:],
    inset = pkws.legend_inset, inset_y = pkws.legend_inset / 2,
    left = True)
util.axis_expand(ax_d, L = -0.1, B = 0, R = -0.1, T = 0)
behavior.update_ci_text(params.cis_file,
    ShrinkPerformance = behavior.ci_text(bhv_cis, "Shrink", "0.3", ''),
    ShrinkFX = behavior.ci_text(bhv_cis, "Shrink", "0.3", 'fx_'))


# save
plt.savefig(params.output, transparent = True)




# ----------------  average pool  ----






