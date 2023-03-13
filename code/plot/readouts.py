
from plot import kwargs as pkws
from plot import util

from matplotlib.collections import LineCollection
import seaborn as sns

from sklearn import metrics as skmtr
from scipy import stats
import pandas as pd
import numpy as np
import torch
import h5py


class readout_data:

    def __init__(self, acts_dist, acts_focl, layer):
        uncued = h5py.File(acts_dist, 'r')
        cued = h5py.File(acts_focl, 'r')
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

# measure auc given positive and negative encodings
compose_auc = lambda pos, neg: skmtr.roc_auc_score(
    np.concatenate([
        np.ones(pos.shape[0]),
        np.zeros(neg.shape[0])]),
    np.concatenate([
        pos.mean(axis = (-2, -1)).sum(axis = -1),
        neg.mean(axis = (-2, -1)).sum(axis = -1)]))


diff_pct_correct = lambda fn_pos, fn_neg: (
    (fn_pos.mean(axis = (2, 3)).sum(axis = 1) -
     fn_neg.mean(axis = (2, 3)).sum(axis = 1)
     ) > 0).mean()

def reconstructed_bhv_auc(readout_data, regs, score_func = compose_auc):
    roc_dist = []; roc_focl = []
    roc_fake = []; roc_undo = []
    for i_cat, c in enumerate(regs):
        if isinstance(regs, dict):
            if hasattr(regs[c], 'w'):
                weights = regs[c].w.detach().numpy()[..., None, None]
            else:
                weights = regs[c][..., None, None]
        elif isinstance(regs, np.ndarray):
            weights = regs[i_cat]
        else:
            raise ValueError(f"Don't know how to deal with regs as {type(regs)}")

        w_pos_dist = readout_data.pos_dist[i_cat] * weights
        w_neg_dist = readout_data.neg_dist[i_cat] * weights
        roc_dist.append(score_func(w_pos_dist, w_neg_dist))

        w_pos_focl = readout_data.pos_focl[i_cat] * weights
        w_neg_focl = readout_data.neg_focl[i_cat] * weights 
        roc_focl.append(score_func(w_pos_focl, w_neg_focl))

        w_pos_fake = readout_data.pos_fake[i_cat] * weights
        w_neg_fake = readout_data.neg_fake[i_cat] * weights 
        roc_fake.append(score_func(w_pos_fake, w_neg_fake))

        w_pos_undo = readout_data.pos_undo[i_cat] * weights
        w_neg_undo = readout_data.neg_undo[i_cat] * weights 
        roc_undo.append(score_func(w_pos_undo, w_neg_undo))

    roc_dist = np.stack(roc_dist); roc_focl = np.stack(roc_focl)
    roc_fake = np.stack(roc_fake); roc_undo = np.stack(roc_undo)

    return dict(score_dist = roc_dist, score_focl = roc_focl,
                score_fake = roc_fake, score_undo = roc_undo)


def reconstructed_bhv(ax, readout_data, regs, score_func = compose_auc, **kws):
    scores_dict = reconstructed_bhv_auc(readout_data, regs, score_func)
    return reconstructed_bhv_plot(ax, scores_dict, **kws)


def reconstructed_bhv_plot(
        ax, scores_dict,
        jitter = 0.15, bootstrap_n = 1000, pkws = pkws, yticks = None,
        rawscores_df = None):
    
    # load in the raw scores dataframe to save outputs to
    rawscores_df_path = rawscores_df
    try:
        if rawscores_df is None: raise ValueError
        rawscores_df = pd.read_csv(rawscores_df, index_col = 0)
    except (pd.errors.EmptyDataError, ValueError) as e:
        rawscores_df = pd.DataFrame()

    
    # encode according to regression weights and measure auc
    # doesn't actually have to be roc, just an outdated var name
    roc_dist = scores_dict['score_dist']
    roc_focl = scores_dict['score_focl']
    roc_fake = scores_dict['score_fake']
    roc_undo = scores_dict['score_undo']

    # generate jitter array shaped like each condition
    jtr_dist = jtr(roc_dist, jitter); jtr_focl = jtr(roc_focl, jitter)
    jtr_fake = jtr(roc_fake, jitter); jtr_undo = jtr(roc_undo, jitter)

    # measure mean confidence intervals
    median_agg = lambda arr: np.median(arr, axis = 1)
    ci_dist = util.mean_ci(roc_dist, bootstrap_n, aggfunc = median_agg)
    ci_fake = util.mean_ci(roc_fake, bootstrap_n, aggfunc = median_agg)
    ci_focl = util.mean_ci(roc_focl, bootstrap_n, aggfunc = median_agg)
    ci_undo = util.mean_ci(roc_undo, bootstrap_n, aggfunc = median_agg)

    if ax is not None:
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
        ax.scatter([0], [np.median(roc_dist)], color = '.2', zorder = 3, **pkws.bhv_mean)
        ax.scatter([1], [np.median(roc_fake)], color = '.2', zorder = 3, **pkws.bhv_mean)
        ax.scatter([2], [np.median(roc_focl)], color = '.2', zorder = 3, **pkws.bhv_mean)
        ax.scatter([3], [np.median(roc_undo)], color = '.2', zorder = 3, **pkws.bhv_mean)

        ax.set_xlim(-0.5, 3.5)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(pkws.labels.reconst_models, )

        if yticks is None:
            ax.set_yticks(ax.get_yticks())
        else:
            ax.set_yticks(yticks)


    # format confidence inervals and return
    cis_ret = {
        'cond': ['dist_fake', 'dist_undo', 'focl_fake', 'focl_undo'],
        'lo_a': [], 'center_a': [], 'hi_a': [],
        'lo_b': [], 'center_b': [], 'hi_b': [],
        'fx_lo': [], 'fx_center': [], 'fx_hi': []}
    median_agg = lambda arr: np.median(arr, axis = 1)
    for i_pair, (aucs_a, aucs_b) in  enumerate([
            (roc_dist, roc_fake),
            (roc_dist, roc_undo),
            (roc_focl, roc_fake),
            (roc_focl, roc_undo)]):
        ci_a = util.mean_ci(aucs_a, bootstrap_n, aggfunc = median_agg)
        ci_b = util.mean_ci(aucs_b, bootstrap_n, aggfunc = median_agg)
        cis_ret['lo_a'].append( ci_a[0] )
        cis_ret['hi_a'].append( ci_a[1] )
        cis_ret['lo_b'].append( ci_b[0] )
        cis_ret['hi_b'].append( ci_b[1] )
        cis_ret['center_a'].append( np.median(aucs_a) )
        cis_ret['center_b'].append( np.median(aucs_b) )
        fx_ci = util.mean_ci(aucs_b - aucs_a, bootstrap_n, aggfunc = median_agg)
        cis_ret['fx_lo'].append( fx_ci[0] )
        cis_ret['fx_hi'].append( fx_ci[1] )
        cis_ret['fx_center'].append( np.median(aucs_b - aucs_a) )
        
    # write scores into rawscores dataframe
    rawscores_df['Reconstruct_dist'] = roc_dist
    rawscores_df['Reconstruct_fake'] = roc_fake
    rawscores_df['Reconstruct_undo'] = roc_undo

    # save raw scores if a dataframe was given
    if rawscores_df_path is not None:
        np.round(rawscores_df, 5).to_csv(rawscores_df_path, index = True)
    return pd.DataFrame(cis_ret)


