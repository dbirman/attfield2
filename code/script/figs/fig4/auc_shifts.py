import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import sklearn.metrics as skmtr
import pandas as pd
import numpy as np
import h5py


sns.set_context('paper')
sns.set_style('ticks')


# Parameters
LAYER = '0.4.3'
JTR = 0.3
MODEL = 'sens_l3'
FOLDER = 'fig5'
OUT_FOLDER = 'fig5'
TGT = 3
DST0 = 4
DST1 = 5
EM = 2.1415

# Load input data
uncued = h5py.File(Paths.data('runs/fig2/fnenc_task_base.h5'), 'r+')
cued = h5py.File(Paths.data(f'runs/{FOLDER}/fn_enc_task_{MODEL}_b4.0.h5'), 'r+')
# cued = h5py.File(Paths.data('runs/fig2/fn_enc_task_gauss_b4.0.h5'), 'r+')


# -----------------------------------------------------------  Preprocess  ----

img_focl = [  cued[LAYER][i_cat, :, i_cat, :, :] for i_cat in range(20)]
img_dist = [uncued[LAYER][i_cat, :len(img_focl[i_cat]), i_cat, :, :] for i_cat in range(20)]
ys = np.concatenate([np.tile([0, 1], len(i) // 2) for i in img_dist])
img_dist = np.reshape(img_dist, [-1, 7, 7])
img_focl = np.reshape(img_focl, [-1, 7, 7])

auc_dist = np.array([[
    skmtr.roc_auc_score(ys, img_dist[:, r, c])
    for c in range(7)] for r in range(7)])[None]
auc_focl = np.array([[
    skmtr.roc_auc_score(ys, img_focl[:, r, c])
    for c in range(7)] for r in range(7)])[None]

var_dist = np.array([[
    np.std(img_dist[:, r, c])
    for c in range(7)] for r in range(7)])[None]
var_focl = np.array([[
    np.std(img_focl[:, r, c])
    for c in range(7)] for r in range(7)])[None]

mag_dist = np.array([[
    np.sqrt(np.mean(img_dist[:, r, c] ** 2))
    for c in range(7)] for r in range(7)])[None]
mag_focl = np.array([[
    np.sqrt(np.mean(img_focl[:, r, c] ** 2))
    for c in range(7)] for r in range(7)])[None]




# corr_map = np.array([[
#     np.corrcoef(img_dist[:, r, c], img_focl[:, r, c])[1,0]
#     for c in range(7)] for r in range(7)])
# plt.imshow(corr_map, cmap = 'cividis_r'); plt.colorbar()


# vmin = min(var_dist.min(), var_focl.min())
# vmax = max(var_dist.max(), var_focl.max())
# plt.imshow(var_dist[0], cmap = "YlGnBu_r", vmin = vmin, vmax = vmax); plt.show()
# plt.imshow(var_focl[0], cmap = "YlGnBu_r", vmin = vmin, vmax = vmax); plt.show()
# plt.imshow(var_focl[0] / var_dist[0], cmap = 'viridis'); plt.colorbar()




# -----------------------------------------------------------------  Plot  ----

for ((img_dist, img_focl,), type_ttl) in [
            [(auc_dist, auc_focl,), "AUC"],
            [(var_dist, var_focl,), "SD"],
            # [(mag_dist, mag_focl,), "Magnitude"],
        ]:

    # -----------------------000---------------  Heatmaps  ----


    vmin = min(img_dist.min(), img_focl.min())
    vmax = max(img_dist.max(), img_focl.max())
    fig, ax = plt.subplots(
        figsize = (4 * EM, EM), ncols = 5,
        gridspec_kw = {'width_ratios': [1, 1, 0.05, 1, 0.05]})
    ax[0].imshow(img_dist[0], cmap = "YlGnBu_r", vmin = vmin, vmax = vmax)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title('Distributed')
    im = ax[1].imshow(img_focl[0], cmap = "YlGnBu_r", vmin = vmin, vmax = vmax)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title('Cued')
    plt.colorbar(im, cax = ax[2]).ax.set_title(type_ttl)

    im = ax[3].imshow(img_focl[0] / auc_dist[0], cmap = 'viridis')
    ax[3].set_xticks([]); ax[3].set_yticks([])
    ax[3].set_title('Cued / Distributed')
    plt.colorbar(im, cax = ax[4]).ax.set_title("Ratio")
    plt.tight_layout()
    plt.savefig(Paths.plots(
        f'figures/{OUT_FOLDER}/info_heat_{type_ttl}_{MODEL}.pdf'))


    # ----------------------  Target vs distractor points  ----

    img_dist_tgt = img_dist[:, :TGT, :TGT].reshape(len(img_dist), -1).ravel()
    img_dist_dct = np.concatenate([
        img_dist[:, DST0:DST1,  :DST1].reshape(len(img_dist), -1),
        img_dist[:, :DST0,  DST0:DST1].reshape(len(img_focl), -1)
    ], axis = 1).ravel()
    img_focl_tgt = img_focl[:, :TGT, :TGT].reshape(len(img_focl), -1).ravel()
    img_focl_dct = np.concatenate([
        img_focl[:, DST0:DST1,  :DST1].reshape(len(img_dist), -1),
        img_focl[:, :DST0,  DST0:DST1].reshape(len(img_focl), -1)
    ], axis = 1).ravel()
 
    jtr_dtgt = np.random.uniform(-JTR, JTR, img_dist_tgt.size)
    jtr_ftgt = np.random.uniform(-JTR, JTR, img_focl_tgt.size)
    jtr_ddct = np.random.uniform(-JTR, JTR, img_dist_dct.size)
    jtr_fdct = np.random.uniform(-JTR, JTR, img_focl_dct.size)


    fig, ax = plt.subplots(figsize = (EM, 1.5 * EM))

    # Distributed many dots
    plt.scatter(
        jtr_dtgt + 0, img_dist_tgt,
        s = 20, color = '#D84315', alpha = 0.95,
        lw = 0.75, ec = 'w')
    plt.scatter(
        jtr_ddct + 2, img_dist_dct,
        s = 20, color = '#D84315', alpha = 0.95,
        lw = 0.75, ec = 'w')

    # Focal many dots
    plt.scatter(
        jtr_ftgt + 1, img_focl_tgt,
        s = 20, color = '#1565C0', alpha = 0.95,
        lw = 0.75, ec = 'w')
    plt.scatter(
        jtr_fdct + 3, img_focl_dct,
        s = 20, color = '#1565C0', alpha = 0.95,
        lw = 0.75, ec = 'w')


    # Target traces
    tgt_traces = LineCollection(np.stack([
            np.stack([jtr_dtgt + 0, img_dist_tgt], axis = -1),
            np.stack([jtr_ftgt + 1, img_focl_tgt], axis = -1)
        ], axis = 1),
        lw = 0.5, color = '.8', alpha = .8, zorder = 0)
    plt.gca().add_collection(tgt_traces)

    # Edge traces
    dct_traces = LineCollection(np.stack([
            np.stack([jtr_ddct + 2, img_dist_dct], axis = -1),
            np.stack([jtr_fdct + 3, img_focl_dct], axis = -1)
        ], axis = 1),
        lw = 0.5, color = '.8', alpha = .8, zorder = 0)
    plt.gca().add_collection(dct_traces)


    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['F', 'D', 'F', 'D'])
    ax.set_xlim(-3*JTR, 3 + 3*JTR)
    if type_ttl == 'AUC':
        plt.axhline(0.5, lw = 1, ls = '--', color = '#263238', zorder = 0)
    sns.despine()
    plt.xlabel("Target        Edge")
    plt.ylabel(f"Unit {type_ttl}")
    plt.tight_layout()

    plt.savefig(Paths.plots(
        f'figures/{OUT_FOLDER}/info_point_{type_ttl}_{MODEL}.pdf'))



# Distribued mean
'''plt.plot(
    [0.], img_dist_tgt.mean(),
    marker = 's', color = '#263238',
    ms = 7, mec = (1,1,1,1), mew = 1)
plt.plot(
    [2.], img_dist_dct.mean(),
    marker = 's', color = '#263238',
    ms = 7, mec = (1,1,1,1), mew = 1)

# Focal mean
plt.plot(
    [1.], img_focl_tgt.mean(),
    marker = 's', color = '#263238',
    ms = 7, mec = (1,1,1,1), mew = 1)
plt.plot(
    [3.], img_focl_dct.mean(),
    marker = 's', color = '#263238',
    ms = 7, mec = (1,1,1,1), mew = 1)'''


