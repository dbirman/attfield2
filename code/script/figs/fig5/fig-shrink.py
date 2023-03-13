import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from plot import lineplots
from plot import quivers
from plot import behavior
from plot import kwargs as pkws
from plot import util

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

class params:
    # parameters: size
    total_size = (18.9, 16.3) #cm

    # parameters: IO
    minigrids = True
    output = 'plots/figures/fig5/fig-shrink.pdf'
    dist_ells = Paths.data('runs/270420/summ_base_300L4_ell.csv')
    focl_ells = [
        Paths.data('runs/shrink/rfs_shrink_b0.1_ell.csv'),
        # Paths.data('runs/shrink/rfs_shrink_b0.2_ell.csv'),
        Paths.data('runs/shrink/rfs_shrink_b0.2_ell.csv'),
        Paths.data('runs/shrink/rfs_shrink_b0.3_ell.csv'),
        Paths.data('runs/shrink/rfs_shrink_b0.4_ell.csv'),
    ]
    comp_ells = [
        Paths.data('runs/270420/summ_cts_gauss_b1.1_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b2.0_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b4.0_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b11.0_ell.csv'),
    ]
    gaussian_bhv_betas = ('1.1', '2.0', '4.0', '11.0')
    shrink_bhv_betas = ('  0.1', '  0.2', '  0.3', '   0.4')
    shrink_lineplot_betas = ('0.1', '0.2', '0.3', '0.4')
    bhv_focl = {
        ('0.1', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_1.1.h5'),
        # ('0.1', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_2.0.h5'),
        ('0.2', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_2.0.h5'),
        ('0.3', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_4.0.h5'),
        ('0.4', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_11.0.h5'),
        ('0.1', 'Shrink'): Paths.data('runs/shrink/bhv_shrink_beta_0.1.h5'),
        # ('0.1', 'Shrink'): Paths.data('runs/shrink/bhv_shrink_beta_0.2.h5'),
        ('0.2', 'Shrink'): Paths.data('runs/shrink/bhv_shrink_beta_0.2.h5'),
        ('0.3', 'Shrink'): Paths.data('runs/shrink/bhv_shrink_beta_0.3.h5'),
        ('0.4', 'Shrink'): Paths.data('runs/shrink/bhv_shrink_beta_0.4.h5'),
    }
    bhv_labels = ['Shrunken RFs']
    bhv_dist = Paths.data('runs/val_rst/bhv_base.h5')
    bhv_stats_output = Paths.data('runs/fig5/fig-ms-bhv_stats.csv')

    sgain_focl = [
        Paths.data('runs/shrink/lenc_ms_n100_b0.1.h5.sgain.npz'),
        # Paths.data('runs/shrink/lenc_ms_n100_b0.2.h5.sgain.npz'),
        Paths.data('runs/shrink/lenc_ms_n100_b0.2.h5.sgain.npz'),
        Paths.data('runs/shrink/lenc_ms_n100_b0.3.h5.sgain.npz'),
        Paths.data('runs/shrink/lenc_ms_n100_b0.4.h5.sgain.npz'),
    ]
    sgain_comp = [
        Paths.data('runs/fig2/lenc_task_gauss_b1.1.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b2.0.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b4.0.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b11.0.h5.sgain.npz'),
    ]
    sgain_stats_output = Paths.data('runs/fig5/fig-ms-gain_stats.csv')
    cis_file = Paths.data('runs/ci_review.txt')

    # axis limits
    size_lim = (0.6, 1.2)
    shift_lim = (-5, 17)
    gain_lim = (0, 10)

# load lineplot data
lp_pre_ells, lp_att_ells, lp_dists, lp_dists_px = lineplots.rf_data(
    params.dist_ells, params.focl_ells,
    loc = (56, 56), rad = 1)
_, lp_comp_ells, _, _ = lineplots.rf_data(
    params.dist_ells, params.comp_ells,
    loc = (56, 56), rad = 1, layer_mask = '0.4.0')

# load behavior data
bhv_focl_data, bhv_dist_data = behavior.bhv_data(
    params.bhv_focl, params.bhv_dist, "Dist.")

# process sizemap/quiver data
qv_dist_ell, qv_focl_ell, qv_smooth_samp = quivers.quiver_data(
    lp_pre_ells, lp_att_ells[-1], (0, 4, 0), 200)

# load gain data
sgain_focl = lineplots.gain_data(
    lp_pre_ells, params.sgain_focl, loc = (56, 56))
sgain_comp = lineplots.gain_data(
    lp_pre_ells, params.sgain_comp, loc = (56, 56))


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
    wspace = 0.4, hspace = 0.45,
    left = 0.06, top = 0.96, right = 0.96, bottom = 0.06)
base_a = util.panel_label(fig, gs[0, 0], "a", xoffset = 0.1)
base_b = util.panel_label(fig, gs[1, 0], "b", xoffset = 0.05)
base_c = util.panel_label(fig, gs[1, 1], "c", xoffset = 0.0)
base_d = util.panel_label(fig, gs[1, 2], "d", xoffset = 0.0)
base_e = util.panel_label(fig, gs[2, 0], "e", xoffset = 0.0)
base_f = util.panel_label(fig, gs[2, 1], "f", xoffset = 0.0)

# panel b
ax_d = fig.add_subplot(gs[1, 0])
size_map = quivers.quiverplot(
    qv_dist_ell, qv_focl_ell, qv_smooth_samp,
    ax_d, cmap = 'coolwarm', vrng = params.size_lim)
util.axis_expand(ax_d, L = 0.2, B = 0.2, R = -0.1, T = 0.05)
util.labels(ax_d,
    pkws.labels.image_position.format('Horizontal'),
    pkws.labels.image_position.format('Vertical'))
util.colorbar(
    fig, ax_d, size_map, ticks = params.size_lim + (1,),
    label = pkws.labels.rf_size, label_vofs = -0.03)


# panel c : single axis
ax_c = fig.add_subplot(gs[1,1])
lineplots.lineplot(
    lineplots.rf_file_iterator(
        'shift', lp_dists, lp_att_ells, (0,4,0),
        comp_ells = lp_comp_ells
        ),
    ax_c,
    line_span = 30, rad = 30, pal = pkws.pal_b,
    xlim = (0, 180), ylim = params.shift_lim)
util.labels(ax_c, pkws.labels.unit_distance, pkws.labels.rf_shift)
util.legend(
    fig, ax_c, 
    ['Attn. strength'] + [
        f'{a} / {b.strip()}'
        for b, a in zip(pkws.labels.beta, params.shrink_lineplot_betas)
    ],
    np.concatenate([[pkws.legend_header_color], pkws.pal_b.values]),
    inset = pkws.legend_inset / 2, inset_y = 0)
util.line_legend(
    fig, ax_c,
    ["Gaussian gain", "Shrunken RFs"],
    [dict(ls="--", lw = 1), dict(ls="-", lw = 1)],
    pkws.legend_header_color,
    inset = pkws.legend_inset, inset_y = 0)


# panel d : single axis
# ax_d = fig.add_subplot(gs[1,2])
# lineplots.lineplot(
#     lineplots.rf_file_iterator(
#         'size', lp_dists, lp_att_ells, (0,4,0),
#         comp_ells = lp_comp_ells
#         ),
#     ax_d,
#     line_span = 30, rad = 30, pal = pkws.pal_b,
#     xlim = (0, 180), ylim = params.size_lim)
# util.labels(ax_d, pkws.labels.unit_distance, pkws.labels.rf_size)
# panel d: breakout
gs_d = gs[1, 2].subgridspec(4, 2, **pkws.mini_gridspec,
    width_ratios = [2, 1])
ax_d = np.array([fig.add_subplot(gs_d[i, 1]) for i in range(4)])
d_data_iter = list(lineplots.rf_file_iterator(
        'size', lp_dists, lp_att_ells, (0,4,0),
        comp_ells = lp_comp_ells
        ),)
lineplots.mini_lineplot(
    d_data_iter,
    ax_d.ravel(),
    line_span = 30, rad = 30, pal = pkws.pal_b,
    xlim = (0, 180), ylim = params.size_lim, yticks = params.size_lim)
# panel d: main
ax_d = fig.add_subplot(gs_d[:, 0])
lineplots.lineplot(
    d_data_iter,
    ax_d,
    line_span = 30, rad = 30, pal = pkws.pal_b,
    xlim = (0, 180), ylim = params.size_lim)
util.labels(ax_d, pkws.labels.unit_distance, pkws.labels.rf_size)

# panel e
# ax_e = fig.add_subplot(gs[2, 0])
# lineplots.lineplot(
#     lineplots.gain_file_iterator(
#         lp_dists, sgain_focl, (0,4,0),
#         gain_comp = sgain_comp),
#     ax_e,
#     line_span = 30, rad = 30, pal = pkws.pal_b,
#     xlim = (0, 180), ylim = params.gain_lim)
# util.labels(ax_e, pkws.labels.unit_distance,
#     "TODO: mim_gauss gain")

gs_e = gs[2,0].subgridspec(4, 2, **pkws.mini_gridspec,
    width_ratios = [2, 1])
# e: breakout
ax_e = np.array([fig.add_subplot(gs_e[i, 1]) for i in range(4)])
e_data_iter = list(lineplots.gain_file_iterator(
    lp_dists, sgain_focl, (0,4,0),
    gain_comp = sgain_comp
    ))
lineplots.mini_lineplot(
    e_data_iter,
    ax_e.ravel(),
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_b,
    xlim = pkws.lineplot_xlim, ylim = params.gain_lim)
# e: main panel
ax_e = fig.add_subplot(gs_e[:, 0])
lineplots.lineplot(
    e_data_iter,
    ax_e,
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_b,
    xlim = pkws.lineplot_xlim, ylim = params.gain_lim)
util.labels(ax_e, pkws.labels.unit_distance, pkws.labels.effective_gain)
gain_cis = util.mean_ci_table(
    [os.path.basename(f) for f in params.sgain_focl],
    [focl for _, _, focl, _ in e_data_iter],
    1000)
gain_cis.to_csv(params.sgain_stats_output, index = False)
behavior.update_ci_text(params.cis_file,
    ShrinkGainMean = behavior.group_ci_text(gain_cis, 'lenc_ms_n100_b0.4.h5.sgain.npz', ''),
    ShrinkGainSD = behavior.group_ci_text(gain_cis, 'lenc_ms_n100_b0.4.h5.sgain.npz', ''))


# panel f
ax_f = fig.add_subplot(gs[2, 1:])
bhv_cis = behavior.bhv_plot(
    bhv_focl_data, bhv_dist_data,
    bar1 = behavior.d2auc(0.75), bar2 = behavior.d2auc(1.28),
    ax = ax_f, yrng = pkws.bhv_yrng, pal = pkws.pal_bhv,
    jitter = 0.03, bootstrap_n = 1000)
util.legend(fig, ax_f,
    [pkws.labels.gaussian_model] +
    params.bhv_labels, pkws.pal_bhv,
    inset = pkws.legend_inset, inset_y = pkws.legend_inset / 2,
    left = True)
labs = ax_f.set_xticklabels(['Dist.',] +
    [b+" / "+(" "*4) for b in params.gaussian_bhv_betas])[1:]
for ltext, side_text in zip(labs, params.shrink_bhv_betas):
    ax_f.text(
        *ltext.get_position(),
        side_text,
        ha = 'left', va = 'top',
        transform = ltext.get_transform(),
        color = pkws.pal_bhv[1],
        fontproperties = ltext.get_fontproperties())
# bhv_cis.to_csv(params.bhv_stats_output)
behavior.update_ci_text(params.cis_file,
    ShrinkPerformancePointOne = behavior.ci_text(bhv_cis, 'Shrink', '0.1', ''),
    ShrinkPerformanceTwo = behavior.ci_text(bhv_cis, 'Shrink', '0.2', ''),
    ShrinkPerformanceFour = behavior.ci_text(bhv_cis, 'Shrink', '0.3', ''),
    ShrinkPerformanceEleven = behavior.ci_text(bhv_cis, 'Shrink', '0.4', ''),
    ShrinkFXPointOne = behavior.ci_text(bhv_cis, 'Shrink', '0.1', 'fx_'),
    ShrinkFXTwo = behavior.ci_text(bhv_cis, 'Shrink', '0.2', 'fx_'),
    ShrinkFXFour = behavior.ci_text(bhv_cis, 'Shrink', '0.3', 'fx_'),
    ShrinkFXEleven = behavior.ci_text(bhv_cis, 'Shrink', '0.4', 'fx_'))

# save
plt.savefig(params.output, transparent = True)


# import numpy as np

# fig, ax = plt.subplots(ncols = 2, figsize = (7, 4))
# ax = ax[1]
# plt.plot(np.arange(5), np.arange(5))
# ax.set_xticks([0, 1, 2, 3, 4])
# labs = ax.set_xticklabels([b + ' / ' + (' ' * 3) for b in ['0', '1', '2', '3', '4']])
# for ltext in labs:
#     t = ax.text(
#         *ltext.get_position(),
#         ' a',
#         ha = 'left', va = 'top',
#         transform = ltext.get_transform(),
#         fontproperties = ltext.get_fontproperties())
# plt.show()








