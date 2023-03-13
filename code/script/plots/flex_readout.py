
import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from argparse import ArgumentParser
import itertools as iit
import sklearn.metrics as skmtr
import pandas as pd
import numpy as np
np.cat = np.concatenate
import h5py

from plot.util import mean_ci
import plot.kwargs

from proc import detection_task as det

sns.set_context('paper')
sns.set_style('ticks')


parser = ArgumentParser(
    description = 
        "Plot qualitative receptive field location and size change summary.")
parser.add_argument('output_path',
    help = 'Path to PDF file where the plots should go, (format string)')
parser.add_argument("regs",
    help = 'Regressions to compare encodings do.')
parser.add_argument("uncued",
    help = 'Uncued / before-mod ellipse RF summaries.')
parser.add_argument("cued",
    help = 'Cued / after-mod ellipse RF summaries.')
parser.add_argument("layer",
    help = 'Layer from which encodings were taken.')
parser.add_argument('--jtr', type = float, default = 0.15,
    help =  'X-axis jitter to apply to points')
parser.add_argument('--em', type = float, default = 2.1415,
    help = 'Size constant.')
parser.add_argument('--bootstrap_n', type = int, default = 1000)
args = parser.parse_args()
args.layer = '.'.join([str(l) for l in eval('tuple('+args.layer+')')])

# Parameters
''' Debug settings
LAYER = '0.4.3'
JTR = 0.15
MODEL = 'gauss'
FOLDER = 'fig2'
TGT = 3
DST0 = 4
DST1 = 5
EM = 2.1415
class args:
    output_path = Paths.plots('figures/fig2/flex_{}_gauss_b4.0.pdf')
    uncued = Paths.data('runs/fig2/fnenc_task_base.h5')
    cued = Paths.data('runs/fig2/enc_task_gauss_b4.0.h5')
    regs = Paths.data('models/logregs_iso224_t100.npz')
'''
LAYER = args.layer
JTR = args.jtr
TGT = 3
DST0 = 4
DST1 = 5
EM = args.em

# Load input data
uncued = h5py.File(args.uncued, 'r+')
cued = h5py.File(args.cued, 'r+')
regs = det.load_logregs(args.regs)

pos_dist = []; pos_focl = []
neg_dist = []; neg_focl = []
for i_cat, c in enumerate(regs):
    dist_dot = uncued[LAYER][i_cat] * regs[c].w.detach().numpy()[..., None, None]
    pos_dist.append(dist_dot[1::2, :, :, :].reshape([-1, 7, 7]) )
    neg_dist.append(dist_dot[::2,  :, :, :].reshape([-1, 7, 7]) )
    focl_dot =   cued[LAYER][i_cat] * regs[c].w.detach().numpy()[..., None, None]
    pos_focl.append(focl_dot[1::2, :, :, :].reshape([-1, 7, 7]) )
    neg_focl.append(focl_dot[::2,  :, :, :].reshape([-1, 7, 7]) )
pos_dist = np.cat(pos_dist); pos_focl = np.cat(pos_focl)
neg_dist = np.cat(neg_dist); neg_focl = np.cat(neg_focl)


def tgt_dct(img):
    tgt = img[:, :TGT, :TGT].reshape(len(img), -1)
    dct = np.concatenate([
        img[:, DST0:DST1,  :DST1].reshape(len(img), -1),
        img[:, :DST0,  DST0:DST1].reshape(len(img), -1)
    ], axis = 1)
    return tgt, dct

def jtr(arr):
    return np.random.uniform(-JTR, JTR, arr.shape)

pos_dist_tgt, pos_dist_dct = tgt_dct(pos_dist)
pos_focl_tgt, pos_focl_dct = tgt_dct(pos_focl)
neg_dist_tgt, neg_dist_dct = tgt_dct(neg_dist)
neg_focl_tgt, neg_focl_dct = tgt_dct(neg_focl)

C0 = '#E64A19'
C1 = '#E64A19'
C2 = '#1976D2' #Lighter: '#03A9F4'
C3 = '#1976D2' #Ligher: '#FFCA28'

aggs = {
    'mean': lambda a: a.mean(axis = 0),
    'take': lambda a: [print(a.shape), a[::1711].ravel()][-1]
}
AGG_NAME = 'mean'
AGG = aggs[AGG_NAME]
S = 12

if True:
    fig, ax = plt.subplots(figsize = (3 * EM, EM))

    plt.scatter(
        0 + jtr(AGG(pos_dist_tgt)), AGG(pos_dist_tgt),
        lw = 0.25, c = C0, edgecolor = (1, 1, 1, 1), s = S,
        label = 'Uncued, Target')
    plt.scatter(
        1 + jtr(AGG(neg_dist_tgt)), AGG(neg_dist_tgt),
        lw = 0.25, c = C1, edgecolor = (1, 1, 1, 1), s = S)

    plt.scatter(
        2 + jtr(AGG(pos_focl_tgt)), AGG(pos_focl_tgt),
        lw = 0.25, c = C2, edgecolor = (1, 1, 1, 1), s = S,
        label = 'Cued, Target')
    plt.scatter(
        3 + jtr(AGG(neg_focl_tgt)), AGG(neg_focl_tgt),
        lw = 0.25, c = C3, edgecolor = (1, 1, 1, 1), s = S)


    plt.scatter(
        4 + jtr(AGG(pos_dist_dct)), AGG(pos_dist_dct),
        lw = 0.5, edgecolor = C0, color = (1, 1, 1, 1), s = 0.6*S,
        label = 'Uncued, Edge')
    plt.scatter(
        5 + jtr(AGG(neg_dist_dct)), AGG(neg_dist_dct),
        lw = 0.5, edgecolor = C1, color = (1, 1, 1, 1), s = 0.6*S)

    plt.scatter(
        6 + jtr(AGG(pos_focl_dct)), AGG(pos_focl_dct),
        lw = 0.5, edgecolor = C2, color = (1, 1, 1, 1), s = 0.6*S,
        label = 'Cued, Edge')
    plt.scatter(
        7 + jtr(AGG(neg_focl_dct)), AGG(neg_focl_dct),
        lw = 0.5, edgecolor = C3, color = (1, 1, 1, 1), s = 0.6*S)

    plt.axhline(lw = 1, color = '.7', zorder = -1, ls = '--')
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(['Pos', 'Neg'] * 4)
    plt.legend(frameon = True, ncol = 2)
    plt.tight_layout()

    plt.savefig(args.output_path.format("axis"))
    plt.close()



# -----------------------------------------------------------------------------
# --------- Readout update plot -----------------------------------------------
# -----------------------------------------------------------------------------


# True tandard deviation:
# uncued_sd = np.stack([
#     np.std(uncued[LAYER][i_cat], axis = (0, 1), keepdims = True)
#     for i_cat in range(len(regs))])
# cued_sd   = np.stack([
#     np.std(  cued[LAYER][i_cat], axis = (0, 1), keepdims = True)
#     for i_cat in range(len(regs))])

# Assuming zero mean:
uncued_sd = np.stack([
    np.mean(uncued[LAYER][i_cat] ** 2, axis = (0, 1), keepdims = True) ** 0.5
    for i_cat in range(len(regs))])
cued_sd   = np.stack([
    np.mean(  cued[LAYER][i_cat] ** 2, axis = (0, 1), keepdims = True) ** 0.5
    for i_cat in range(len(regs))])

eff_gain = cued_sd / uncued_sd

fig, ax = plt.subplots(figsize = (2 * EM, 2*EM))
plt.pcolormesh(
    eff_gain.mean(axis = 0)[0, 0], cmap = 'viridis',
    linewidth = 0, rasterized = True).set_edgecolor('face')
plt.ylim(plt.gca().get_ylim()[::-1])
plt.gca().set_aspect(1.)
plt.colorbar()
plt.title("Effective Gain")
plt.tight_layout()
plt.savefig(args.output_path.format("gain"))
plt.close()

compose_auc = lambda pos, neg: skmtr.roc_auc_score(
    np.cat([
        np.ones(pos.shape[0]),
        np.zeros(neg.shape[0])]),
    np.cat([
        pos.mean(axis = (-2, -1)).sum(axis = -1),
        neg.mean(axis = (-2, -1)).sum(axis = -1)]))

pos_dist = []; pos_focl = []; pos_fake = []; pos_undo = []
neg_dist = []; neg_focl = []; neg_fake = []; neg_undo = []
roc_dist = []; roc_focl = []; roc_fake = []; roc_undo = []
for i_cat, c in enumerate(regs):
    dist_dot = uncued[LAYER][i_cat] * regs[c].w.detach().numpy()[..., None, None]
    pos_dist.append(dist_dot[1::2, :, :, :])
    neg_dist.append(dist_dot[::2,  :, :, :])
    roc_dist.append(compose_auc(pos_dist[-1], neg_dist[-1]))
    focl_dot =   cued[LAYER][i_cat] * regs[c].w.detach().numpy()[..., None, None]
    pos_focl.append(focl_dot[1::2, :, :, :])
    neg_focl.append(focl_dot[::2,  :, :, :])
    roc_focl.append(compose_auc(pos_focl[-1], neg_focl[-1]))
    fake_cued = uncued[LAYER][i_cat] * eff_gain[i_cat]
    fake_dot = fake_cued * regs[c].w.detach().numpy()[..., None, None]
    pos_fake.append(fake_dot[1::2, :, :, :])
    neg_fake.append(fake_dot[::2,  :, :, :])
    roc_fake.append(compose_auc(pos_fake[-1], neg_fake[-1]))
    fake_undo = cued[LAYER][i_cat] / eff_gain[i_cat]
    undo_dot = fake_undo * regs[c].w.detach().numpy()[..., None, None]
    pos_undo.append(undo_dot[1::2, :, :, :])
    neg_undo.append(undo_dot[::2,  :, :, :])
    roc_undo.append(compose_auc(pos_undo[-1], neg_undo[-1]))
pos_dist = np.stack(pos_dist); pos_focl = np.stack(pos_focl)
pos_fake = np.stack(pos_fake); pos_undo = np.stack(pos_undo)
neg_dist = np.stack(neg_dist); neg_focl = np.stack(neg_focl)
neg_fake = np.stack(neg_fake); neg_undo = np.stack(neg_undo)
roc_dist = np.stack(roc_dist); roc_focl = np.stack(roc_focl)
roc_fake = np.stack(roc_fake); roc_undo = np.stack(roc_undo)

ci_dist = mean_ci(roc_dist, args.bootstrap_n)
ci_fake = mean_ci(roc_fake, args.bootstrap_n)
ci_focl = mean_ci(roc_focl, args.bootstrap_n)
ci_undo = mean_ci(roc_undo, args.bootstrap_n)

jtr_dist = jtr(roc_dist); jtr_focl = jtr(roc_focl)
jtr_fake = jtr(roc_fake); jtr_undo = jtr(roc_undo)

if True:
    fig, ax = plt.subplots(figsize = (2 * EM, 2 * EM))

    # raw data / by category
    plt.scatter(0 + jtr_dist, roc_dist, color = '.4',
        **plot.kwargs.bhv_cat)
    plt.scatter(1 + jtr_fake, roc_fake, color = 'C4',
        **plot.kwargs.bhv_cat)
    plt.scatter(2 + jtr_focl, roc_focl, color = 'C0',
        **plot.kwargs.bhv_cat)
    plt.scatter(3 + jtr_undo, roc_undo, color = 'C2',
        **plot.kwargs.bhv_cat)

    # connecting lines
    ax.add_collection(LineCollection(
        np.stack([
            np.stack([0 + jtr_dist, 1 + jtr_fake]).T,
            np.stack([roc_dist, roc_fake]).T
        ], axis = -1),
        zorder = -1, **plot.kwargs.bhv_connector
    ))
    ax.add_collection(LineCollection(
        np.stack([
            np.stack([2 + jtr_focl, 3 + jtr_undo]).T,
            np.stack([roc_focl, roc_undo]).T
        ], axis = -1),
        zorder = -1, **plot.kwargs.bhv_connector
    ))

    # mean and ci
    ax.plot([0, 0], ci_dist, color = '.2', zorder = 2, **plot.kwargs.bhv_ci)
    ax.plot([1, 1], ci_fake, color = '.2', zorder = 2, **plot.kwargs.bhv_ci)
    ax.plot([2, 2], ci_focl, color = '.2', zorder = 2, **plot.kwargs.bhv_ci)
    ax.plot([3, 3], ci_undo, color = '.2', zorder = 2, **plot.kwargs.bhv_ci)
    ax.scatter([0], [roc_dist.mean()], color = '.2', zorder = 3, **plot.kwargs.bhv_mean)
    ax.scatter([1], [roc_fake.mean()], color = '.2', zorder = 3, **plot.kwargs.bhv_mean)
    ax.scatter([2], [roc_focl.mean()], color = '.2', zorder = 3, **plot.kwargs.bhv_mean)
    ax.scatter([3], [roc_undo.mean()], color = '.2', zorder = 3, **plot.kwargs.bhv_mean)

    plt.xlim(-0.5, 3.5)
    plt.axhline(0.5, lw = 1, color = '.7', zorder = -1, ls = '--')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Dist.', 'Fake', 'Focal', 'Undo'])
    plt.ylim(None, 1)
    plt.tight_layout()
    plt.title("Reconstructed Performance")

    plt.savefig(args.output_path.format('bhv'))
    plt.close()




# -------------------- Explained variance
from sklearn import metrics as skmtr
from scipy import stats


r2_map = []
for i_row in range(7):
    row = []
    for i_col in range(7):

        row.append(stats.pearsonr(
            np.concatenate([
                pos_dist[:, :, :, i_row, i_col].ravel(),
                neg_dist[:, :, :, i_row, i_col].ravel()]),
            np.concatenate([
                pos_focl[:, :, :, i_row, i_col].ravel(),
                neg_focl[:, :, :, i_row, i_col].ravel()]))[0])
    r2_map.append(row)
r2_map = np.stack(r2_map)

plt.imshow(r2_map, cmap = 'Spectral')
plt.colorbar()
plt.savefig(args.output_path.format('r2map'))
plt.close()


if True:
    fake_feat_r2 = []
    undo_feat_r2 = []
    for i_feat in range(512):
        fake_feat_r2.append(stats.pearsonr(
            np.concatenate([
                pos_focl[:, :, i_feat].mean(axis=(-2, -1)).ravel(),
                neg_focl[:, :, i_feat].mean(axis=(-2, -1)).ravel()]),
            np.concatenate([
                pos_fake[:, :, i_feat].mean(axis=(-2, -1)).ravel(),
                neg_fake[:, :, i_feat].mean(axis=(-2, -1)).ravel()]))[0])

        undo_feat_r2.append(stats.pearsonr(
            np.concatenate([
                pos_dist[:, :, i_feat].mean(axis=(-2, -1)).ravel(),
                neg_dist[:, :, i_feat].mean(axis=(-2, -1)).ravel()]),
            np.concatenate([
                pos_undo[:, :, i_feat].mean(axis=(-2, -1)).ravel(),
                neg_undo[:, :, i_feat].mean(axis=(-2, -1)).ravel()]))[0])

    fig, ax = plt.subplots(nrows = 2, sharex = True)
    bins = np.linspace(0, 1, 20)
    ax[0].hist(fake_feat_r2, color = (1,1,1,0), edgecolor = 'C0', linewidth = 1,
        label = 'Focl v.s. Fake', bins = bins)
    ax[1].hist(undo_feat_r2, color = (1,1,1,0), edgecolor = 'C1', linewidth = 1,
        label = 'Dist v.s. Undo', bins = bins)
    ax[0].legend(loc = 'upper left', frameon = False)
    ax[1].legend(loc = 'upper left', frameon = False)
    ax[1].set_xlabel('Feature $R^2$')
    plt.savefig(args.output_path.format('r2feat'))



