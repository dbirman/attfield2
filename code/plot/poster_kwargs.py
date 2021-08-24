rc = {
    'figure.dpi': 100,
    'font.sans-serif': 'Arial',
    'font.family': 'sans-serif',
    'axes.labelsize': 20.,
    'axes.linewidth': 1.5,
    'xtick.major.size': 100. * 2.54 * .04,  #dots/inch * inch/cm * len
    'ytick.major.size': 100. * 2.54 * .04,
    'xtick.major.width': 1.5, 
    'ytick.major.width': 1.5,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
}

mini_gridspec = dict(
    wspace = 0.3, hspace = 0.3)

axis_label = dict(
    fontsize = rc['axes.labelsize'],
    fontstyle = 'italic')
colorbar_label = dict(
    fontsize = 18.,
    fontstyle = 'italic')
axis_label_margin = 8 # points
legend_text = dict(
    fontsize = 20,
    fontstyle = 'italic')
legend_line_height = 1.8
legend_inset = 0.3 #cm


import pandas as pd
pal_l = pd.read_csv(Paths.data('cfg/pal_layer.csv'))['color']
pal_b = pd.read_csv(Paths.data('cfg/pal_beta.csv'))['color']
pal_bhv = ['#020202', '#d55c00']
pal_old = ['#0288D1', '#C62828', '#FFB300', '#5E35B1', '#43A047', '#FF4400', '#33FF00']
# pal_old = ['#333333', '#FF4400', '#6A1B9A', '#512DA8', '#2196F3', '#26C6DA']


# ----- axis limits ----

lineplot_xlim = (0, 180)
layerplot_xlim = (-3, 180)
gain_ylim = (0, 12)
shift_ylim = (-20, 15)
pos_shift_ylim = (-5, 15)
size_ylim = (-0.5, 0.1)
small_size_ylim = (-0.2, 0.1)
bhv_yrng = (0.55, 0.95)


# ----- line plots ----

lineplot_point = dict(
    lw = 3)
avg_line = dict(
    lw = 2)
avg_line_secondary = dict(
    lw = 2, ls = '--')
mini_avg_line = dict(
    lw = 1)
mini_avg_line_secondary = dict(
    lw = 1, ls = '--')
lineplot_span = 20


# ----- behavior plots ----

bhv_cat = dict(
    marker = 'o', s = 50,
    lw = 1, edgecolor = 'w',
    alpha = 1.
    )
bhv_ci = dict(
    lw = 3., marker = '',
    solid_capstyle = 'butt')
bhv_mean = dict(
    marker = 's', s = 190,
    edgecolor = 'w', lw = 0.5)
bhv_connector = dict(
    lw = 0.5, color = '.9',)
bhv_bar_text_kws = dict(
    alpha = 1.,
    fontsize = 18.0,
    fontstyle = 'italic')
bhv_bar1 = dict(
    ls = '--', lw = 2,
    color = '#37474F', alpha = 0.3,)
bhv_bar2 = dict(
    ls = '-', lw = 2,
    color = '#263238', alpha = 0.3)


# ----- quiver plots ----

size_cmap = 'coolwarm'
quiver_point = dict(
    ms = 3.5, color = '#111111')
quiver_line = dict(
    lw = 1, color = '#111111')



# diagram

rf_point_outer  = dict(
    ls = '', marker = 'o', ms = 8,
    color = 'w')
rf_point_main = dict(ls = '', marker = 'o', ms = 6)
rf_ellipse = dict(lw = 2)
rf_locus = dict(marker = 'X', color = '#263238', ms = 10,
        mew = 1, mec = 'w')


# ----- axis labels ----

class labels:
    rf_shift = 'RF shift [px] '
    rf_size = "RF Size\nRatio"
    effective_gain = "Output Layer Gain   "
    unit_distance = 'Unit distance from center [px]'
    bhv_performance = "Performance\n[AUC (d')]"
    bhv_beta = "Attention Strength"
    feat_map_position = "{} position in feature map"
    image_position = "{} image space"
    feature_r2 = "Avg. Focal-Dist. Corr."
    layer = [f"Layer {i+1}" for i in range(4)]
    beta = [r"1.1x", "2x", "4x", "11x"]
    gaussian_model = "Gaussian Gain"
    reconst_models = [
        'Dist.',
        'Multiplied',
        'Propagated',
        'Divided']







