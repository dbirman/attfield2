"""
Plot heatmap of receptive fields, as output by backprop.py, possibly
overlayed with others. As a diagnostic and positive sanity check.
"""

import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from argparse import ArgumentParser
import numpy as np
import h5py
import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

parser = ArgumentParser(
    description = "Plot tests of  logistic regressions on isolated"+
                  "object detection task.")
parser.add_argument('output_path',
    help = 'Desired path for output PDF.')
parser.add_argument("data_paths", nargs = '+',
    help = 'One or more HDF5 gradient archives with overlapping keys.')
parser.add_argument('--verbose', action = "store_true",
    help = 'Run with extra progress output.')
parser.add_argument('--nodata', action = 'store_true',
    help = 'Run on fake data, for when data files aren\'t accessible.')
args = parser.parse_args()

if args.nodata:
    grads = [{} for i in range(len(args.data_paths))]
    keys = ['grads_0.2.0_0.0.0', 'grads_0.4.0_0.0.0']
    size = 40
    n_unit = 10
    for k in keys:
        for i in range(len(grads)):
            grads[i][k] = np.zeros([n_unit, size, size, 1])
            for j in range(n_unit):
                x = np.linspace(-1, 3, size)[None, None, :]
                y = np.linspace(-1, 3, size)[None, :, None]
                x_part = (x - 0.005*j**2 + i - 0.3)**2
                y_part = (y + 0.05*j - i - 0.6)**2
                g = np.maximum(0, 1 - x_part - y_part)
                grads[i][k][j] = g
else:
    grads = [h5py.File(p, 'r') for p in args.data_paths]

# Overlapping keys
keys = [k for k in grads[0].keys() if all([(k in g) for g in grads])]
if len(keys) == 0:
    raise ValueError("No overlapping keys.")

# Set up color maps
pal = [
    ( 21, 101, 192), # Blue
    (198,  40,  40), # Red
    (104, 159,  56), # Green
    (106,  27, 154), # Violet
]
if len(grads) > len(pal):
    raise NotImplementedError
cmaps = []
for col in pal:
    cm_kernel = np.full((256, 4), 0.)
    cm_kernel[:, 0] = col[0] / 255
    cm_kernel[:, 1] = col[1] / 255
    cm_kernel[:, 2] = col[2] / 255
    cm_kernel[:, 3] = np.linspace(0, 1/np.sqrt(len(grads)), 256)
    cm = ListedColormap(cm_kernel)
    cmaps.append(cm)

sns.set_style('dark')
with PdfPages(args.output_path) as pdf:
    for key in keys:
        print("Gradients:", key)
        for i_unit in tqdm.trange(len(grads[0][key])):
            
            fig, ax = plt.subplots(figsize = (6, 5))
            for i_dat in range(len(grads)):

                rf = abs(grads[i_dat][key][i_unit]).sum(axis = 0)
                im = ax.imshow(rf, cmap = cmaps[i_dat])
                plt.colorbar(im).ax.set_title(i_dat)

            plt.title(f"{key} | Unit {i_unit}")
            plt.tight_layout(rect = [0.05, 0.05, 0.95, 0.9])
            pdf.savefig()
            plt.close()



