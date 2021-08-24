import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import voxel_selection as vx
from proc import attention_models as att

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

from argparse import ArgumentParser
import pandas as pd
import numpy as np


parser = ArgumentParser(
    description = 
        "Plot summaries of receptive field statistics.")
parser.add_argument('output_path',
    help = 'Path to PDF file where the plots should go.')
parser.add_argument("rf_summary",
    help = 'RF summary CSV file containing parameters to be plotted.')
parser.add_argument('--bins', type = int, default = 20,
    help = 'Number of histogram bins. Default: 20')
parser.add_argument('--norm', action = 'store_true',
    help = 'Normalize axis range across layers.')
parser.add_argument('--custom', nargs='+',
    help = 'Custom data columns, format: "2w=lambda row: 2*row.width"')
args = parser.parse_args()
if args.custom is not None:
    args.custom = {
        col.split('=')[0]: eval('='.join(col.split('=')[1:]))
        for col in args.custom}
else: args.custom = {}


# -------------------------------------- Load inputs ----

# Receptive fields
params = pd.read_csv(args.rf_summary)
units = vx.VoxelIndex.from_serial(params['unit'])
params = params.set_index('unit')

for k, f in args.custom.items():
    params[k] = params.apply(f, axis = 1)

# -------------------------------------- Plot ----

sns.set_style('ticks')
with PdfPages(args.output_path) as pdf:
    for layer in units:
        # Select rows of `params` with unit in this layer.
        lstr = '.'.join(str(i) for i in layer)
        row_mask = params.index.map(lambda u: u.startswith(lstr))
        rfs = params.loc[row_mask]

        for col in params.columns:
        
            fig, ax = plt.subplots(figsize = (7, 4))
            
            sns.distplot(rfs[col],
                kde = True, hist = True,
                bins = args.bins, kde_kws = dict(cut = 0),
                color = '#4a5e69')
            plt.axvline(rfs[col].mean(), lw = 1, ls = '--',
                color = '#4a5e69')
            yrng = plt.gca().get_ylim()
            plt.plot(
                [np.quantile(rfs[col], 0.25),
                 np.quantile(rfs[col], 0.75)],
                [.9 * yrng[1] - .1 * yrng[0]] * 2,
                color = '#4a5e69', lw = 3)

            if args.norm:
                plt.xlim(params[col].min(), params[col].max())

            plt.title(f"Layer: {layer}  |  Param: {col}")
            sns.despine(ax = plt.gca())
            plt.tight_layout()
            pdf.savefig()
            plt.close()

