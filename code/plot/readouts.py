
from plot import kwargs as pkws
from plot import util

from matplotlib.collections import LineCollection
import seaborn as sns

from sklearn import metrics as skmtr
from scipy import stats
import numpy as np
import h5py


class readout_data:

    def __init__(self, acts_dist, acts_focl, layer):
        uncued = h5py.File(acts_dist, 'r+')
        cued = h5py.File(acts_focl, 'r+')
        layer = '.'.join(str(i) for i in layer)
        n_cats = len(uncued[layer])

        uncued_sd = np.stack([
            np.mean(uncued[layer][i_cat] ** 2, axis = (0, 1), keepdims = True) ** 0.5
            for i_cat in range(n_cats)])
        cued_sd   = np.stack([
            np.mean(  cued[layer][i_cat] ** 2, axis = (0, 1), keepdims = True) ** 0.5
            for i_cat in range(n_cats)])
        eff_gain = cued_sd / uncued_sd
        self.eff_gain = eff_gain

        self.pos_dist = []; self.pos_focl = []
        self.pos_fake = []; self.pos_undo = []
        self.neg_dist = []; self.neg_focl = []
        self.neg_fake = []; self.neg_undo = []

        for i_cat in range(n_cats):
            pos_ixs = uncued['y'][i_cat].astype('bool')
            neg_ixs = ~pos_ixs
            dist_dot = uncued[layer][i_cat] #* regs[c].w.detach().numpy()[..., None, None]
            self.pos_dist.append(dist_dot[pos_ixs, :, :, :])
            self.neg_dist.append(dist_dot[neg_ixs,  :, :, :])
            focl_dot =   cued[layer][i_cat] #* regs[c].w.detach().numpy()[..., None, None]
            self.pos_focl.append(focl_dot[pos_ixs, :, :, :])
            self.neg_focl.append(focl_dot[neg_ixs,  :, :, :])
            fake_cued = uncued[layer][i_cat] * eff_gain[i_cat]
            fake_dot = fake_cued #* regs[c].w.detach().numpy()[..., None, None]
            self.pos_fake.append(fake_dot[pos_ixs, :, :, :])
            self.neg_fake.append(fake_dot[neg_ixs,  :, :, :])
            fake_undo = cued[layer][i_cat] / eff_gain[i_cat]
            undo_dot = fake_undo #* regs[c].w.detach().numpy()[..., None, None]
            self.pos_undo.append(undo_dot[pos_ixs, :, :, :])
            self.neg_undo.append(undo_dot[neg_ixs,  :, :, :])
        self.pos_dist = np.stack(self.pos_dist); self.pos_focl = np.stack(self.pos_focl)
        self.pos_fake = np.stack(self.pos_fake); self.pos_undo = np.stack(self.pos_undo)
        self.neg_dist = np.stack(self.neg_dist); self.neg_focl = np.stack(self.neg_focl)
        self.neg_fake = np.stack(self.neg_fake); self.neg_undo = np.stack(self.neg_undo)


def load_logregs(fname):
    from proc import detection_task as det
    return det.load_logregs(fname)


def r2_map(ax, readout_data, vrng = (None, None)):
    r2_map = []
    for i_row in range(7):
        row = []
        for i_col in range(7):
            row.append(stats.pearsonr(
                np.concatenate([
                    readout_data.pos_dist[:, :, :, i_row, i_col].ravel(),
                    readout_data.neg_dist[:, :, :, i_row, i_col].ravel()]),
                np.concatenate([
                    readout_data.pos_focl[:, :, :, i_row, i_col].ravel(),
                    readout_data.neg_focl[:, :, :, i_row, i_col].ravel()]))[0])
        r2_map.append(row)
    r2_map = np.stack(r2_map)

    img = ax.imshow(
        r2_map, cmap = 'bone',
        vmin = vrng[0], vmax = vrng[1])
    ax.set_xticks([])
    ax.set_yticks([])
    return img


def gain_map(ax, data):
    gains = data.eff_gain.mean(axis = 0)[0, 0]
    vlim = np.round([gains.min(), gains.max()], 0)
    im = ax.imshow(gains, cmap = 'bone', vmin = vlim[0], vmax = vlim[1])
    ax.set_xticks([])
    ax.set_yticks([])
    return im





def auc2d(auc): return np.sqrt(2) * stats.norm.ppf(auc)
def d2auc(d): return stats.norm.cdf(d / np.sqrt(2))

def jtr(arr, JTR):
    return np.random.uniform(-JTR, JTR, arr.shape)


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
    jtr_dist = jtr(roc_dist, jitter); jtr_focl = jtr(roc_focl, jitter)
    jtr_fake = jtr(roc_fake, jitter); jtr_undo = jtr(roc_undo, jitter)

    # measure mean confidence intervals
    ci_dist = util.mean_ci(roc_dist, bootstrap_n)
    ci_fake = util.mean_ci(roc_fake, bootstrap_n)
    ci_focl = util.mean_ci(roc_focl, bootstrap_n)
    ci_undo = util.mean_ci(roc_undo, bootstrap_n)

    # raw data / by category
    ax.scatter(0 + jtr_dist, roc_dist, color = '.6',
        **pkws.bhv_cat)
    ax.scatter(1 + jtr_fake, roc_fake, color = '.6',
        **pkws.bhv_cat)
    ax.scatter(2 + jtr_focl, roc_focl, color = '.6',
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
    ax.add_collection(LineCollection(
        np.stack([
            np.stack([2 + jtr_focl, 3 + jtr_undo]).T,
            np.stack([roc_focl, roc_undo]).T
        ], axis = -1),
        zorder = -1, **pkws.bhv_connector
    ))

    # mean and ci
    ax.plot([0, 0], ci_dist, color = '.3', zorder = 2, **pkws.bhv_ci)
    ax.plot([1, 1], ci_fake, color = '.3', zorder = 2, **pkws.bhv_ci)
    ax.plot([2, 2], ci_focl, color = '.3', zorder = 2, **pkws.bhv_ci)
    ax.plot([3, 3], ci_undo, color = '.3', zorder = 2, **pkws.bhv_ci)
    ax.scatter([0], [roc_dist.mean()], color = '.2', zorder = 3, **pkws.bhv_mean)
    ax.scatter([1], [roc_fake.mean()], color = '.2', zorder = 3, **pkws.bhv_mean)
    ax.scatter([2], [roc_focl.mean()], color = '.2', zorder = 3, **pkws.bhv_mean)
    ax.scatter([3], [roc_undo.mean()], color = '.2', zorder = 3, **pkws.bhv_mean)

    ax.set_xlim(-0.5, 3.5)
    # ax.axhline(0.5, lw = 1, color = '.7', zorder = -1, ls = '--')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(pkws.labels.reconst_models, )
    # ax.set_ylim(pkws.bhv_yrng)
    if yticks is None:
        ax.set_yticks(ax.get_yticks())
    else:
        ax.set_yticks(yticks)
    # ax.set_yticklabels([
    #     '{:.2f} ({:.1f})'.format(y, auc2d(y))
    #     for y in ax.get_yticks()])


