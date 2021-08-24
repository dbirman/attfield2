import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

from argparse import ArgumentParser
import itertools as iit
import pickle as pkl
import numpy as np
import h5py
import tqdm
import os


parser = ArgumentParser(
    description = 
        "Train logistic regressions on isolated object detection task.")
parser.add_argument('output_path',
    help = 'Path to PDF file for plots; format string with key `lstr`.' + 
           ' Ex: data/enc_l{lstr}.pdf')
parser.add_argument("image_meta",
    help = 'Metadata output from image generator used to create encodings.')
parser.add_argument("enc", nargs = '+',
    help = 'Path to one or more encodings HDF5 archive.')
parser.add_argument("--disp", nargs = '+', default = None,
    help = 'Display names of the input encoding files.')
parser.add_argument("--meta", default = None,
    help = 'Lambda for metadata in suptitle. Passed img_i and the ' + 
           'current metadata dictionary.')
parser.add_argument("--max_img", type = int, default = float('inf'),
    help = 'Max number of images to display per group.')
parser.add_argument("--max_feat", type = int, default = float('inf'),
    help = 'Max number of feat to display per group.')
parser.add_argument("--cmap", default = 'diverge',
    help = 'Colormap version: diverge, center, or sequential.')
args = parser.parse_args()
if args.meta is not None: args.meta = eval(args.meta)

"""
Test args:
class args:
    output_path = '/tmp/tmp.pdf'
    enc = ['data/runs/050520/enc_edge_gauss_b2.0.h5']
    disp = ['2.0']
    image_meta = 'data/runs/050520/bars_meta.pkl'
"""

# -------------------------------------- Load inputs ----

# Encodings archive
enc = [h5py.File(f, 'r') for f in args.enc]
if args.disp is None:
    disp = [os.path.basename(f) for f in args.enc]
else: disp = args.disp

# Image metadata
meta_pkl = pkl.load(open(args.image_meta, 'rb'))



# ------------------------------------------- Plot ----

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


layers = list(enc[0].keys())

# Plot RF movement in a separate plot for each layer
sns.set('notebook')
sns.set_style('ticks')
with PdfPages(args.output_path) as pdf:

    for lstr in layers:
        print("Layer:", lstr) 
        n_feat = min(enc[0][lstr].shape[1], args.max_feat)
        for feat_i in range(n_feat):
            for grp_i in range(len(meta_pkl)):
                print("Feature:", feat_i, "Group:", meta_pkl[grp_i][0])
                n_img = min(meta_pkl[grp_i][1][0], args.max_img)
                for img_i in range(n_img):
            
                    fig, ax = plt.subplots(
                        figsize = (4 * len(enc), 4),
                        ncols = len(enc))
                    if len(enc) == 1: ax = [ax]
                    for f_i, dat in enumerate(enc):
                        arr = dat[lstr][grp_i, img_i, feat_i]
                        if args.cmap == 'unipolar':
                            color_args = {'cmap': 'viridis'}
                        elif args.cmap == 'center':
                            color_args = dict(
                                norm = MidpointNormalize(
                                    vmin = arr.min(), vmax = arr.max(),
                                    midpoint = 0),
                                cmap = 'coolwarm')
                        else:
                            vrng = abs(arr).max()
                            color_args = dict(
                                vmin = -vrng, vmax = vrng,
                                cmap = 'coolwarm')
                        plt_img = ax[f_i].imshow(
                            arr, **color_args)
                        ax[f_i].set_title(disp[f_i])
                        ax[f_i].set_aspect(1.)
                        plt.colorbar(plt_img, ax = ax[f_i])
                    plt.tight_layout(rect = [0.05, 0.05, 0.95, 0.93])
                    lyr_str = f"Layer: {lstr}"
                    grp_str = f"Group: {meta_pkl[grp_i][0]}"
                    img_str = f"Image: {img_i}"
                    fet_str = f"Feature: {feat_i}"
                    if args.meta is not None:
                        meta_str = " | " + args.meta(
                            img_i = img_i, **meta_pkl[grp_i][2])
                    else: meta_str = ""
                    fig.suptitle(f"{lyr_str} | {fet_str} | {grp_str} " + 
                                 f"| {img_str}{meta_str}")
                    pdf.savefig()
                    plt.close('all')






