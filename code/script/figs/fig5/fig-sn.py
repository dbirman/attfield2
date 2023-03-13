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
    total_size = pkws.twocol_size #cm

    # parameters: IO
    minigrids = False
    output = 'plots/figures/fig5/fig-sn.pdf'
    dist_ells = Paths.data('runs/270420/summ_base_ell.csv')
    focl_ells = [
        Paths.data('runs/fig5/ell_sn4_n100_b1.1_ell.csv'),
        Paths.data('runs/fig5/ell_sn4_n100_b2.0_ell.csv'),
        Paths.data('runs/fig5/ell_sn4_n100_b4.0_ell.csv'),
        Paths.data('runs/fig5/ell_sn4_n100_b11.0_ell.csv'),
    ]
    comp_ells = [
        Paths.data('runs/270420/summ_cts_gauss_b1.1_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b2.0_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b4.0_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b11.0_ell.csv'),
    ]
    bhv_focl = {
        ('1.1', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_1.1.h5'),
        ('2.0', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_2.0.h5'),
        ('4.0', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_4.0.h5'),
        ('11.0', 'Gauss'): Paths.data('runs/fig2/bhv_gauss_n600_beta_11.0.h5'),
        ('1.1', 'al'): Paths.data('runs/fig5/sna_bhv_n300_b1.1.h5'),
        ('2.0', 'al'): Paths.data('runs/fig5/sna_bhv_n300_b2.0.h5'),
        ('4.0', 'al'): Paths.data('runs/fig5/sna_bhv_n300_b4.0.h5'),
        ('11.0', 'al'): Paths.data('runs/fig5/sna_bhv_n300_b11.0.h5'),
        ('1.1', 'l1'): Paths.data('runs/fig5/sn1_bhv_n300_b1.1.h5'),
        ('2.0', 'l1'): Paths.data('runs/fig5/sn1_bhv_n300_b2.0.h5'),
        ('4.0', 'l1'): Paths.data('runs/fig5/sn1_bhv_n300_b4.0.h5'),
        ('11.0', 'l1'): Paths.data('runs/fig5/sn1_bhv_n300_b11.0.h5'),
        ('1.1', 'l2'): Paths.data('runs/fig5/sn2_bhv_n300_b1.1.h5'),
        ('2.0', 'l2'): Paths.data('runs/fig5/sn2_bhv_n300_b2.0.h5'),
        ('4.0', 'l2'): Paths.data('runs/fig5/sn2_bhv_n300_b4.0.h5'),
        ('11.0', 'l2'): Paths.data('runs/fig5/sn2_bhv_n300_b11.0.h5'),
        ('1.1', 'l3'): Paths.data('runs/fig5/sn3_bhv_n300_b1.1.h5'),
        ('2.0', 'l3'): Paths.data('runs/fig5/sn3_bhv_n300_b2.0.h5'),
        ('4.0', 'l3'): Paths.data('runs/fig5/sn3_bhv_n300_b4.0.h5'),
        ('11.0', 'l3'): Paths.data('runs/fig5/sn3_bhv_n300_b11.0.h5'),
        ('1.1', 'l4'): Paths.data('runs/fig5/sn4_bhv_n300_b1.1.h5'),
        ('2.0', 'l4'): Paths.data('runs/fig5/sn4_bhv_n300_b2.0.h5'),
        ('4.0', 'l4'): Paths.data('runs/fig5/sn4_bhv_n300_b4.0.h5'),
        ('11.0', 'l4'): Paths.data('runs/fig5/sn4_bhv_n300_b11.0.h5'),
    }
    bhv_labels = ['All Layers', "Layer 1", "Layer 2", "Layer 3", "Layer 4"]
    bhv_dist = Paths.data('runs/val_rst/bhv_base.h5')
    bhv_stats_output = Paths.data('runs/fig5/fig-sn-bhv_stats.csv')

    sgain_focl = [
        Paths.data('runs/fig5/lenc_sna_n100_b1.1.h5.sgain.npz'),
        Paths.data('runs/fig5/lenc_sna_n100_b2.0.h5.sgain.npz'),
        Paths.data('runs/fig5/lenc_sna_n100_b4.0.h5.sgain.npz'),
        Paths.data('runs/fig5/lenc_sna_n100_b11.0.h5.sgain.npz'),
    ]
    sgain_comp = [
        Paths.data('runs/fig2/lenc_task_gauss_b1.1.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b2.0.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b4.0.h5.sgain.npz'),
        Paths.data('runs/fig2/lenc_task_gauss_b11.0.h5.sgain.npz'),
    ]
    sgain_stats_output = Paths.data('runs/fig5/fig-sn-gain_stats')
    sgain_comp_output = Paths.data('runs/fig2/fig-gauss-gain_stats.csv')
    cis_file = Paths.data("runs/ci_cmd.txt")
    rawscores_df = Paths.data('runs/rawscores.csv')

    # axis limits
    size_lim = (0.85, 1.1)
    shift_lim = (-2.5, 15)
    gain_lim = (0, 10)

# load lineplot data
lp_pre_ells, lp_att_ells, lp_dists, lp_dists_px = lineplots.rf_data(
    params.dist_ells, params.focl_ells,
    loc = (56, 56), rad = 1)
_, lp_comp_ells, _, _ = lineplots.rf_data(
    params.dist_ells, params.comp_ells,
    loc = (56, 56), rad = 1)

# load behavior data
bhv_focl_data, bhv_dist_data = behavior.bhv_data(
    params.bhv_focl, params.bhv_dist, "Dist.")

# process sizemap/quiver data
qv_dist_ell, qv_focl_ell, qv_smooth_samp = quivers.quiver_data(
    lp_pre_ells, lp_att_ells[3], (0, 4, 0), 200)

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
base_a = util.panel_label(fig, gs[0, 0], "a")
base_b = util.panel_label(fig, gs[1, 0], "b")
base_c = util.panel_label(fig, gs[1, 1], "c")
base_d = util.panel_label(fig, gs[1, 2], "d")
base_e = util.panel_label(fig, gs[2, 0], "e")
base_f = util.panel_label(fig, gs[2, 1], "f")

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
    label = pkws.labels.rf_size, label_vofs = -0.04)


# panel c : single axis
ax_c = fig.add_subplot(gs[1,1])
lineplots.lineplot(
    lineplots.rf_file_iterator(
        'shift', lp_dists, lp_att_ells, (0,4,0),
        comp_ells = lp_comp_ells),
    ax_c,
    line_span = 30, rad = 30, pal = pkws.pal_b,
    xlim = (0, 180), ylim = params.shift_lim)
util.labels(ax_c, pkws.labels.unit_distance, pkws.labels.rf_shift)
util.legend(
    fig, ax_c, 
    ['Gain strength'] + pkws.labels.beta,
    np.concatenate([[pkws.legend_header_color], pkws.pal_b.values]),
    inset = pkws.legend_inset)


# panel d : single axis
ax_d = fig.add_subplot(gs[1,2])
lineplots.lineplot(
    lineplots.rf_file_iterator(
        'size', lp_dists, lp_att_ells, (0,4,0),
        comp_ells = lp_comp_ells),
    ax_d,
    line_span = 30, rad = 30, pal = pkws.pal_b,
    xlim = (0, 180), ylim = params.size_lim)
util.labels(ax_d, pkws.labels.unit_distance, pkws.labels.rf_size)


# -----------  panel e
gs_e = gs[2, 0].subgridspec(4, 2, **pkws.mini_gridspec,
    width_ratios = [2, 1])
# e: breakout
ax_e = np.array([fig.add_subplot(gs_e[i, 1]) for i in range(4)])
e_data_iter = list(lineplots.gain_file_iterator(
    lp_dists, sgain_focl, (0,4,0),
    gain_comp = sgain_comp))
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
# output confidence intervals on resulting gain
gain_mean_ci_table = util.mean_ci_table(
    [os.path.basename(f) for f in params.sgain_focl],
    [focl for _, _, focl, _ in e_data_iter],
    1000)
gain_mean_ci_table.to_csv(params.sgain_stats_output + '.csv', index = False)
gain_sd_ci_table = util.mean_ci_table(
    [os.path.basename(f) for f in params.sgain_focl],
    [focl for _, _, focl, _ in e_data_iter],
    1000, aggfunc = lambda x: x.std(axis = 1))
gain_sd_ci_table.to_csv(params.sgain_stats_output + '_sd.csv', index = False)
behavior.update_ci_text(params.cis_file,
    SensGainMean = behavior.group_ci_text(gain_mean_ci_table, 'lenc_sna_n100_b4.0.h5.sgain.npz', ''),
    SensGainSD = behavior.group_ci_text(gain_sd_ci_table, 'lenc_sna_n100_b4.0.h5.sgain.npz', ''))

# ----------- panel f
ax_f = fig.add_subplot(gs[2, 1:])
bhv_pal = np.concatenate([pkws.pal_bhv, pkws.pal_l.values])
bhv_cis = behavior.bhv_plot(
    bhv_focl_data, bhv_dist_data,
    bar1 = behavior.d2auc(0.75), bar2 = behavior.d2auc(1.28), dodge = 0.11,
    ax = ax_f, yrng = pkws.bhv_yrng, pal = bhv_pal,
    jitter = 0.02, bootstrap_n = 1000,
    rawscores_df = params.rawscores_df)
util.legend(fig, ax_f,
    [pkws.labels.gaussian_model, params.bhv_labels[0]],
    pkws.pal_bhv,
    inset = pkws.legend_inset, inset_y = pkws.legend_inset / 4,
    left = True)
util.legend(fig, ax_f,
    params.bhv_labels[1:3], pkws.pal_l.values[0:2],
    inset = 2.2, inset_y = pkws.legend_inset / 4,
    left = True)
util.legend(fig, ax_f,
    params.bhv_labels[3:], pkws.pal_l.values[2:4],
    inset = 3.5, inset_y = pkws.legend_inset / 4,
    left = True)
bhv_cis.to_csv(params.bhv_stats_output)
behavior.update_ci_text(params.cis_file,
    GaussPerformancePointOne = behavior.ci_text(bhv_cis, 'Gauss', '1.1', ''),
    GaussPerformanceTwo = behavior.ci_text(bhv_cis, 'Gauss', '2.0', ''),
    GaussPerformanceFour = behavior.ci_text(bhv_cis, 'Gauss', '4.0', ''),
    GaussPerformanceEleven = behavior.ci_text(bhv_cis, 'Gauss', '11.0', ''),
    SensPerformancePointOne = behavior.ci_text(bhv_cis, 'al', '1.1', ''),
    SensPerformanceTwo = behavior.ci_text(bhv_cis, 'al', '2.0', ''),
    SensPerformanceFour = behavior.ci_text(bhv_cis, 'al', '4.0', ''),
    SensPerformanceEleven = behavior.ci_text(bhv_cis, 'al', '11.0', ''),
    SensFXPointOne = behavior.ci_text(bhv_cis, 'al', '1.1', 'fx_'),
    SensFXTwo = behavior.ci_text(bhv_cis, 'al', '2.0', 'fx_'),
    SensFXFour = behavior.ci_text(bhv_cis, 'al', '4.0', 'fx_'),
    SensFXEleven = behavior.ci_text(bhv_cis, 'al', '11.0', 'fx_'))


# save
plt.savefig(params.output, transparent = True)











