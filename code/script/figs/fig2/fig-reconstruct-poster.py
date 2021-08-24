import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from plot import lineplots
from plot import quivers
from plot import behavior
from plot import diagrams
from plot import readouts
from plot import poster_kwargs as pkws
from plot import util

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

from plot import util

from matplotlib.collections import LineCollection
import seaborn as sns

from sklearn import metrics as skmtr
from scipy import stats
import numpy as np

import matplotlib.colors as mc
import h5py

class params:
    # parameters: size
    total_size = (9.3, 12.2) #cm

    # parameters: IO
    output = 'plots/figures/fig2/fig-reconstruct-poster.pdf'
    acts_dist = Paths.data('runs/fig2/fnenc_task_base.h5')
    acts_focl = Paths.data('runs/fig2/enc_task_gauss_b4.0.h5')
    regs = Paths.data('models/logregs_iso224_t100.npz')


# ----------------  load data  ----

# load encoding/readout data
readout_data = readouts.readout_data(
    params.acts_dist, params.acts_focl,
    (0, 4, 3))
regs = readouts.load_logregs(params.regs)


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
    left = 0.3, top = 0.95, right = 0.9, bottom = 0.3)


# ----------------  readout panel  ----


def reconstructed_bhv(
        ax, readout_data, regs,
        jitter = 0.15, bootstrap_n = 1000, pkws = pkws, yticks = None):

    # measure auc given positive and negative encodings
    compose_auc = lambda pos, neg: skmtr.roc_auc_score(
        np.concatenate([
            np.ones(pos.shape[0]),
            np.zeros(neg.shape[0])]),
        np.concatenate([
            pos.mean(axis = (-2, -1)).sum(axis = -1),
            neg.mean(axis = (-2, -1)).sum(axis = -1)]))

    # encode according to regression weights and measure auc
    roc_dist = []; roc_focl = []
    roc_fake = []; roc_undo = []
    for i_cat, c in enumerate(regs):
        weights = regs[c].w.detach().numpy()[..., None, None]

        w_pos_dist = readout_data.pos_dist[i_cat] * weights
        w_neg_dist = readout_data.neg_dist[i_cat] * weights 
        roc_dist.append(compose_auc(w_pos_dist, w_neg_dist))

        w_pos_focl = readout_data.pos_focl[i_cat] * weights
        w_neg_focl = readout_data.neg_focl[i_cat] * weights 
        roc_focl.append(compose_auc(w_pos_focl, w_neg_focl))

        w_pos_fake = readout_data.pos_fake[i_cat] * weights
        w_neg_fake = readout_data.neg_fake[i_cat] * weights 
        roc_fake.append(compose_auc(w_pos_fake, w_neg_fake))

        w_pos_undo = readout_data.pos_undo[i_cat] * weights
        w_neg_undo = readout_data.neg_undo[i_cat] * weights 
        roc_undo.append(compose_auc(w_pos_undo, w_neg_undo))

        # focl_dot =   cued[LAYER][i_cat] * regs[c].w.detach().numpy()[..., None, None]
        # roc_focl.append(compose_auc(pos_focl[-1], neg_focl[-1]))

        # fake_dot = fake_cued * regs[c].w.detach().numpy()[..., None, None]
        # roc_fake.append(compose_auc(pos_fake[-1], neg_fake[-1]))

        # undo_dot = fake_undo * regs[c].w.detach().numpy()[..., None, None]
        # roc_undo.append(compose_auc(pos_undo[-1], neg_undo[-1]))
    roc_dist = np.stack(roc_dist); roc_focl = np.stack(roc_focl)
    roc_fake = np.stack(roc_fake); roc_undo = np.stack(roc_undo)

    # generate jitter array shaped like each condition
    jtr_dist = readouts.jtr(roc_dist, jitter); jtr_focl = readouts.jtr(roc_focl, jitter)
    jtr_fake = readouts.jtr(roc_fake, jitter); jtr_undo = readouts.jtr(roc_undo, jitter)

    # measure mean confidence intervals
    ci_dist = util.mean_ci(roc_dist, bootstrap_n)
    ci_fake = util.mean_ci(roc_fake, bootstrap_n)
    ci_focl = util.mean_ci(roc_focl, bootstrap_n)
    ci_undo = util.mean_ci(roc_undo, bootstrap_n)

    # raw data / by category
    ax.scatter(0 + jtr_dist, roc_dist, color = '.6',
        **pkws.bhv_cat)
    ax.scatter(1 + jtr_fake, roc_fake, color = 0.45 * mc.to_rgba_array(pkws.pal_bhv[1]) + 0.5 * np.ones(4),
        **pkws.bhv_cat)
    ax.scatter(2 + jtr_focl, roc_focl, color =  '.6',
        **pkws.bhv_cat)
    ax.scatter(3 + jtr_undo, roc_undo, color = '.6',
        **pkws.bhv_cat)

    # connecting lines
    ax.add_collection(LineCollection(
        np.stack([
            np.stack([0 + jtr_dist, 1 + jtr_fake]).T,
            np.stack([roc_dist, roc_fake]).T
        ], axis = -1),
        zorder = -1, **pkws.bhv_connector
    ))

    # mean and ci
    ax.plot([0, 0], ci_dist, color = '.3', zorder = 2, **pkws.bhv_ci)
    ax.plot([1, 1], ci_fake, color = '.3', zorder = 2, **pkws.bhv_ci)
    ax.plot([2, 2], ci_focl, color = '.3', zorder = 2, **pkws.bhv_ci)
    ax.plot([3, 3], ci_undo, color = '.3', zorder = 2, **pkws.bhv_ci)
    ax.scatter([0], [roc_dist.mean()], color = '.2', zorder = 3, **pkws.bhv_mean)
    ax.scatter([1], [roc_fake.mean()], color = pkws.pal_bhv[1], zorder = 3, **pkws.bhv_mean)
    ax.scatter([2], [roc_focl.mean()], color = '.2', zorder = 3, **pkws.bhv_mean)
    ax.scatter([3], [roc_undo.mean()], color = '.2', zorder = 3, **pkws.bhv_mean)

    ax.set_xlim(-0.5, 3.5)
    # ax.axhline(0.5, lw = 1, color = '.7', zorder = -1, ls = '--')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(pkws.labels.reconst_models, )
    ax.set_ylim(pkws.bhv_yrng)
    if yticks is None:
        ax.set_yticks(ax.get_yticks())
    else:
        ax.set_yticks(yticks)


# panel b
ax_c = fig.add_subplot(gs[0, 0])
reconstructed_bhv(ax_c, readout_data, regs, pkws = pkws, yticks = pkws.bhv_yrng)
ax_c.set_xticks([0, 1, 2])
sns.despine(ax = ax_c, trim = True, offset = 5)
ax_c.set_xticklabels(pkws.labels.reconst_models[:3], rotation = 45, ha = 'center')
ax_c.set_xlim(-0.5, 2.5)
util.labels(ax_c, None, "Performance [AUC (d')]", pkws = pkws)

# save
plt.savefig(params.output, transparent = True)











