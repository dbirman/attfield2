import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from plot import lineplots
from plot import quivers
from plot import behavior
from plot import diagrams
from plot import readouts
from plot import normalization
from plot import kwargs as pkws
from plot import util

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

class params:


    # parameters: IO
    output = 'plots/figures/fig2/fig-quivermap.pdf'
    dist_ells = Paths.data('runs/270420/summ_base_ell.csv')
    focl_ells = [
        Paths.data('runs/270420/summ_cts_gauss_b1.1_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b2.0_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b4.0_ell.csv'),
        Paths.data('runs/270420/summ_cts_gauss_b11.0_ell.csv'),
    ]
    size_lim = (0.65, 1.2)
    


# ----------------  load data  ----

# load rf data
lp_pre_ells, lp_att_ells, lp_dists, lp_dists_px = lineplots.rf_data(
    params.dist_ells, params.focl_ells,
    loc = (56, 56), rad = 1)

# process sizemap/quiver data
qv_dist_ell, qv_focl_ell, qv_smooth_samp = quivers.quiver_data(
    lp_pre_ells, lp_att_ells[3], (0, 4, 0), 200)



# ----------------  make structure  ----

import matplotlib
sns.set('notebook')
sns.set_style('ticks')
matplotlib.rcParams.update(pkws.rc)

# make figure
cm = 1/2.54
fig = plt.figure(figsize = (3, 3))


# ----------------  top row  ----




ax_b = fig.add_subplot(1, 1, 1)
quiver_mappable = quivers.quiverplot(
    qv_dist_ell, qv_focl_ell, qv_smooth_samp,
    ax_b, cmap = 'coolwarm', vrng = params.size_lim,
    arrows = False)
ax_b.set_axis_off()
# util.axis_expand(ax_b, L = 0.1, B = 0.1, R = 0, T = 0)
# util.labels(ax_b,
#     pkws.labels.image_position.format('Horizontal'),
#     pkws.labels.image_position.format('Vertical'))
# util.colorbar(
#     fig, ax_b, quiver_mappable, ticks = params.size_lim + (1,),
#     label = pkws.labels.rf_size, label_vofs = -0.04)



# save
plt.savefig(params.output, transparent = True)












