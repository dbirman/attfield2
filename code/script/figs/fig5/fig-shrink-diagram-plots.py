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
    total_size = (5, 5) #cm

    # parameters: IO
    output = 'plots/figures/fig5/fig-shrink-diagram-plots.pdf'
    dist_ells = Paths.data('runs/270420/summ_base_300L4_ell.csv')
    focl_ells = [
        Paths.data('runs/shrink/rfs_shrink_b0.4_ell.csv'),
    ]
    comp_ells = [
        Paths.data('runs/270420/summ_cts_gauss_b11.0_ell.csv'),
    ]
    

    # axis limits
    size_lim = (0.6, 1.2)


lp_pre_ells, lp_att_ells, lp_dists, lp_dists_px = lineplots.rf_data(
    params.dist_ells, params.focl_ells,
    loc = (56, 56), rad = 1)
_, lp_comp_ells, _, _ = lineplots.rf_data(
    params.dist_ells, params.comp_ells,
    loc = (56, 56), rad = 1, layer_mask = '0.4.0')


import matplotlib
sns.set('notebook')
sns.set_style('ticks')
matplotlib.rcParams.update(pkws.rc)

# make figure
cm = 1/2.54
fig = plt.figure(
    constrained_layout = False,
    figsize = [s*cm for s in params.total_size])

ax_a = fig.add_subplot(111)
f = lambda r: 1 - 0.3 * np.exp(-2.44 * r ** 2 / 112 ** 2) * np.cos(2.89 * r ** 2 / 112 ** 2)
r = np.linspace(0, 180, 100)
lineplots.lineplot(
    lineplots.rf_file_iterator(
        'size', lp_dists, lp_att_ells, (0,4,0),
        comp_ells = lp_comp_ells
        ),
    ax_a,
    line_span = 30, rad = 30, pal = [pkws.pal_b[3]],
    xlim = (0, 180), ylim = params.size_lim,
    comp_only = True)
ax_a.plot(r, f(r), color = '.3',)
util.axis_expand(ax_a, T = 0, R = 0, L = -0.12, B = -0.2)
util.labels(ax_a, pkws.labels.unit_distance, pkws.labels.rf_size)



plt.savefig(params.output, transparent = True)










