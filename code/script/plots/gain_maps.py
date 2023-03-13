
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
parser.add_argument('--r2vrng', type = float, default = [None, None], nargs = 2,
    help = 'R2 color axis range.')
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
n_cat = len(uncued[LAYER])

pos_dist = []; pos_focl = []
neg_dist = []; neg_focl = []
for i_cat in range(n_cat):
    dist_dot = uncued[LAYER][i_cat]
    pos_dist.append(dist_dot[1::2, :, :, :])
    neg_dist.append(dist_dot[::2,  :, :, :])
    focl_dot =   cued[LAYER][i_cat]
    pos_focl.append(focl_dot[1::2, :, :, :])
    neg_focl.append(focl_dot[::2,  :, :, :])
pos_dist = np.stack(pos_dist); pos_focl = np.stack(pos_focl)
neg_dist = np.stack(neg_dist); neg_focl = np.stack(neg_focl)


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
    for i_cat in range(n_cat)])
cued_sd   = np.stack([
    np.mean(  cued[LAYER][i_cat] ** 2, axis = (0, 1), keepdims = True) ** 0.5
    for i_cat in range(n_cat)])

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




# -------------------- Explained variance
from sklearn import metrics as skmtr
from scipy import stats


r2_map = []
for i_row in range(pos_dist.shape[3]):
    row = []
    for i_col in range(pos_dist.shape[4]):

        row.append(stats.pearsonr(
            np.concatenate([
                pos_dist[:, :, :, i_row, i_col].ravel(),
                neg_dist[:, :, :, i_row, i_col].ravel()]),
            np.concatenate([
                pos_focl[:, :, :, i_row, i_col].ravel(),
                neg_focl[:, :, :, i_row, i_col].ravel()]))[0])
    r2_map.append(row)
r2_map = np.stack(r2_map)

fig, ax = plt.subplots(figsize = (2 * EM, 2*EM))
plt.imshow(r2_map, cmap = 'Spectral', vmin = args.r2vrng[0], vmax = args.r2vrng[1])
plt.colorbar()
plt.savefig(args.output_path.format('r2map'))
plt.close()
