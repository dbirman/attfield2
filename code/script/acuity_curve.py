import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from plot import util
import plot.kwargs

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns

from argparse import ArgumentParser
import sklearn.metrics as skmtr
from scipy import stats
import pandas as pd
import numpy as np
import h5py

parser = ArgumentParser()
parser.add_argument('output_pdf')
parser.add_argument('scores_file')
parser.add_argument('--pal')
parser.add_argument('--figsize', nargs = 2, type = int, default = (8, 4))
args = parser.parse_args()
# class args:
#     output_pdf = 'plots/runs/acuity/acuity_spacing_gauss_mccw.pdf'
#     scores_file = 'data/runs/acuity/bhv_gauss_mccw_byspacing_scores.csv'
#     pal = 'data/cfg/pal_beta_withcomp.csv'
#     figsize = (8, 4)
scores = pd.read_csv(args.scores_file)
pal = pd.read_csv(args.pal)['color']

with PdfPages(args.output_pdf) as pdf:
    for cat, cat_scores in scores.groupby('cat'):
        fig, ax = plt.subplots(figsize = args.figsize)
        unique_f = cat_scores['disp'].unique()
        f_xs = dict(zip(unique_f, np.arange(len(unique_f))))
        for i_cond, (cond, cond_scores) in enumerate(cat_scores.groupby('cond')):
            # print('cond:', cond)
            xs = cond_scores['disp'].map(f_xs).values
            x_sort = np.argsort(xs)
            ax.plot(
                xs[x_sort], cond_scores['score'].values[x_sort],
                color = pal[i_cond], label = cond)
        ax.set_title(cat)
        ax.legend(frameon = False)
        plt.tight_layout()
        sns.despine(ax = ax)
        pdf.savefig()
        plt.close()
