
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

import itertools as iit
import sklearn.metrics as skmtr
import pandas as pd
import numpy as np
np.cat = np.concatenate
import h5py

from proc import detection_task as det

sns.set_context('paper')
sns.set_style('ticks')


# Parameters
LAYER = '0.4.3'
JTR = 0.15
MODEL = 'gauss'
FOLDER = 'fig2'
TGT = 3
DST0 = 4
DST1 = 5
EM = 2.1415

# Load input data
uncued = h5py.File(Paths.data('runs/fig2/fnenc_task_base.h5'), 'r+')
cued = h5py.File(Paths.data(f'runs/{FOLDER}/enc_task_{MODEL}_b4.0.h5'), 'r+')
fn_uncued = h5py.File(Paths.data('runs/fig2/fn_fnenc_task_base.h5'), 'r+')
fn_cued = h5py.File(Paths.data(f'runs/{FOLDER}/fn_enc_task_{MODEL}_b4.0.h5'), 'r+')
regs = det.load_logregs(Paths.data('models/logregs_iso224_t100.npz'))

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

plt.savefig(Paths.plots(f'runs/feat_motion/feat_prod_{AGG_NAME}.pdf'))
plt.close()



"""
---------------------------------------------------------------------

Flexing of feature space: Average response to "Banana" increases even
for "Bathtub" units when attn preferentially applied to "Banana".

"""

nonlins = {
    'lin': lambda x: x,
    'sig': lambda x: 1 / (1 + np.exp(-x)),
    'rlu': lambda x: np.maximum(0, x),
}

units = np.arange(20)

with PdfPages(Paths.plots('runs/feat_motion/outgroup_tuning.pdf')) as pdf:
    for unit, (nl_name, nl) in iit.product(units, nonlins.items()):

        # 0th decoder/classifier unit
        act_dist = fn_uncued[LAYER][:, :, unit, :, :].mean(axis = (-2, -1))
        act_focl =   fn_cued[LAYER][:, :, unit, :, :].mean(axis = (-2, -1))

        fig, ax = plt.subplots(figsize = (2 * EM, EM))

        other = np.cat([np.arange(unit), np.arange(unit+1, 20)])
        # Bathtub unit, images with a banana and 3 distractors
        (lambda Y: (plt.scatter(
                0 + jtr(Y), Y,
                color = '#E64A19', s = 25),
            plt.scatter(
                0, Y.mean(),
                color = '.3', s = 37, lw = 1, edgecolor = '1.'
            )))(
            nl(act_dist[other, 1::2]).mean(axis = -1))
        # Bathtub unit, images with an attended banana and 3 distractors
        (lambda Y: (plt.scatter(
                1 + jtr(Y), Y,
                color = '#1976D2', s = 25),
            plt.scatter(
                1, Y.mean(),
                color = '.3', s = 37, lw = 1, edgecolor = '1.'
            )))(
            nl(act_focl[other, 1::2]).mean(axis = -1))

        cat = list(regs.keys())[unit]
        plt.title(f"Nonlin: {nl_name} | Unit: {cat}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Uncued', 'Cued'])
        plt.xlim(-0.5, 1.5)
        plt.tight_layout()
        pdf.savefig()
        plt.close()



plt.scatter(
    SG(act_dist[0, 1::2]), SG(act_dist[1, 1::2]))
plt.scatter(
    SG(act_dist[0, 1::2]).mean(), SG(act_dist[1, 1::2]).mean(),
    color = '.3')
plt.scatter(
    SG(act_focl[0, 1::2]), SG(act_focl[1, 1::2]))
plt.scatter(
    SG(act_focl[0, 1::2]).mean(), SG(act_focl[1, 1::2]).mean(),
    color = '.5')
plt.gca().set_aspect(1.)





