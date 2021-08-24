import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import pandas as pd
import numpy as np
import h5py

from proc import detection_task as det


sns.set_context('paper')
sns.set_style('ticks')


# Parameters
LAYER = '0.4.3'
JTR = 0.3
MODEL = 'gauss'
TGT = 3
DST0 = 4
DST1 = 5
ZOOM = True

# Load input data
uncued = h5py.File(Paths.data('runs/fig2/fn_fnenc_task_base.h5'), 'r+')
# cued = h5py.File(Paths.data(f'runs/fig4/fn_enc_task_{MODEL}_b4.0.h5'), 'r+')
cued = h5py.File(Paths.data('runs/fig2/fn_enc_task_gauss_b4.0.h5'), 'r+')


# -----------------------------------------------------------  Preprocess  ----

img_dist = [uncued[LAYER][i_cat, :, i_cat, :, :] for i_cat in range(20)]
img_focl = [  cued[LAYER][i_cat, :, i_cat, :, :] for i_cat in range(20)]
img_dist = np.reshape(img_dist, [-1, 7, 7])
img_focl = np.reshape(img_focl, [-1, 7, 7])
ys = np.repeat([0, 1], len(img_dist))

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

jtr_tgt = np.random.uniform(-JTR, JTR, img_focl_tgt.size)
jtr_dct = np.random.uniform(-JTR, JTR, img_focl_dct.size)


# -----------------------------------------------------------------  Plot  ----


fig, ax = plt.subplots(figsize = (2.1415, 1.5 * 2.1415))

plt.scatter(
    jtr_tgt + 0, img_dist_tgt,
    s = 10, color = '#C62828', alpha = 0.7,
    lw = 0.25, ec = 'w')
plt.scatter(
    jtr_dct + 2, img_dist_dct,
    s = 10, color = '#9a5c5c', alpha = 0.7,
    lw = 0.25, ec = 'w')

plt.plot(
    [0.], img_dist_tgt.mean(),
    marker = 's', color = '#263238',
    ms = 6, mec = (1,1,1,0.7), mew = 2)
plt.plot(
    [2.], img_dist_dct.mean(),
    marker = 's', color = '#263238',
    ms = 6, mec = (1,1,1,0.7), mew = 2)

plt.scatter(
    jtr_tgt + 1, img_focl_tgt,
    s = 10, color = '#283593', alpha = 0.7,
    lw = 0.25, ec = 'w')
plt.scatter(
    jtr_dct + 3, img_focl_dct,
    s = 10, color = '#65677f', alpha = 0.7,
    lw = 0.25, ec = 'w')

plt.plot(
    [1.], img_focl_tgt.mean(),
    marker = 's', color = '#263238',
    ms = 6, mec = (1,1,1,0.7), mew = 2)
plt.plot(
    [3.], img_focl_dct.mean(),
    marker = 's', color = '#263238',
    ms = 6, mec = (1,1,1,0.7), mew = 2)

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['F', 'D', 'F', 'D'])
plt.axhline(0, lw = 1, color = '#263238', zorder = 0)
if ZOOM: plt.ylim(-100, 100)
sns.despine()
plt.xlabel("Target        Edge")
plt.ylabel("Decision Axis")
plt.tight_layout()

zstr = "_zoom" if ZOOM else ""
plt.savefig(Paths.plots(
    f'figures/fig4/enc_{MODEL}{zstr}_t{TGT}d{DST0}-{DST1}.pdf'))




