"""
Plot standard machine learning metrics for trained logregs as 
sanity that they were trained effectively.
"""

import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, roc_auc_score
from argparse import ArgumentParser
import numpy as np
import h5py
import os

import matplotlib.pyplot as plt
import seaborn as sns

parser = ArgumentParser(
    description = "Plot tests of  logistic regressions on isolated"+
                  "object detection task.")
parser.add_argument('output_path',
    help = 'Path to a folder where PDFs should be stored.')
parser.add_argument("data_path",
    help = 'Path to the HDF5 archive containing the behavioral data.')
parser.add_argument('--verbose', action = "store_true",
    help = 'Run with extra progress output.')
parser.add_argument('--nodata', action = 'store_true',
    help = 'Run without access to true data files. They\'re often top '+
           'large to live on the same machine as scripting is done on.')
parser.add_argument("--ci_file", nargs = 2, default = None)
args = parser.parse_args()



# Set up / load inputs
if args.nodata:
    data = {}
    cats = ['banana', 'bathtub']
    cats = np.array(cats).astype('S')
    attrs_type = type('attrarr', (np.ndarray,), {'attrs': None})
    for cond, N in (('train', 40), ('test', 20)):
        data[f'ys_{cond}'] = np.random.randint(
            2, size = N * len(cats)).astype('bool')
        data[f'fn_{cond}'] = np.random.normal(
            2, size = N * len(cats)) + 1 * data[f'ys_{cond}']
        data[f'preds_{cond}'] = data[f'fn_{cond}'] > 0
        data[f'cat_{cond}'] = attrs_type((len(cats) * N,))
        data[f'cat_{cond}'][...] = np.tile(np.arange(len(cats)), N)
        data[f'cat_{cond}'] = data[f'cat_{cond}'].astype('int')
        data[f'cat_{cond}'].attrs = {}
        data[f'cat_{cond}'].attrs['cat_names'] = cats
else:
    data = h5py.File(args.data_path, 'r')

# Run for each category
with PdfPages(args.output_path) as pdf:
    sns.set_style('ticks')
    auc_scores = {}
    for i_cat in np.unique(data[f'cat_train']):

        mask_train = data['cat_train'][...] == i_cat
        mask_test = data['cat_test'][...] == i_cat
        ys_train = data['ys_train'][mask_train]
        ys_test = data['ys_test'][mask_test]
        fn_train = data['fn_train'][mask_train]
        fn_test = data['fn_test'][mask_test]

        train_test = np.array(['Train', "Test"])[np.concatenate(
            [np.zeros(len(ys_train), dtype = int),
             np.ones(len(ys_test), dtype = int)])]
        hue = np.array(['Neg', 'Pos'])[np.concatenate(
            [(ys_train == 1).astype('int8'),
             (ys_test == 1).astype('int8')])]
        fn = np.concatenate([fn_train, fn_test])

        fpr_tr, tpr_tr, _ = roc_curve(ys_train, fn_train)
        fpr_te, tpr_te, _ = roc_curve(ys_test, fn_test)

        fig, ax = plt.subplots(figsize = (9, 5), ncols = 2,
             gridspec_kw={'width_ratios': [1, 1.5]})
        sns.boxplot(
            x = train_test, y = fn, hue = hue,
            hue_order = ['Pos', 'Neg'],
            palette = ['#F5F5F5', '#F5F5F5'], linewidth = 1.,
            whis = float('inf'), dodge = True, ax = ax[0])
        sns.stripplot(
            x = train_test, y = fn, hue = hue,
            hue_order = ['Pos', 'Neg'],
            size = 4, palette = ['#388e3c', '#d32f2f'], alpha = 0.7,
            dodge = True, jitter = 0.3, ax = ax[0])
        ax[0].set_xlim(-0.8, 1.8)
        ax[0].get_legend().remove()
        handles, labels = ax[0].get_legend_handles_labels()
        ax[0].legend(handles[2:], labels[2:])
        ax[0].set_ylabel("Score")

        ax[1].plot(fpr_tr, tpr_tr, label = "Train",
            lw = 1., color = "#0288d1")
        ax[1].plot(fpr_te, tpr_te, label = "Test",
            lw = 1., color = "#FFB300")
        ax[1].plot([0, 1], [0, 1],  ls = '--', lw = 1, color = '.8')
        ax[1].legend()
        ax[1].set_ylabel("Sensitivity (TP / P)")
        ax[1].set_ylabel("Specificity (TN / N)")

        cat_name = data['cat_train'].attrs['cat_names'][i_cat]
        plt.suptitle(f"\nCategory: {cat_name}")
        sns.despine(ax = ax[0])
        sns.despine(ax = ax[1])
        plt.tight_layout(rect = (0.05, 0.05, 0.95, 0.9))

        pdf.savefig()
        plt.close()

        cat_name = data['cat_train'].attrs['cat_names'][i_cat].decode()
        auc_scores[cat_name] = roc_auc_score(ys_test, fn_test)

    fig, ax = plt.subplots(figsize = (9, 5))
    K = np.array(list(auc_scores.keys()))
    V = np.array(list(auc_scores.values()))
    srt = np.argsort(V)
    markerline, stemlines, baseline = ax.stem(
        np.arange(len(srt)), V[srt],
        markerfmt = 'o', linefmt = '-', basefmt = '',
        use_line_collection = True)
    plt.setp(stemlines, 'linewidth', 1)
    plt.setp(stemlines, 'color', '#1976D2')
    plt.setp(markerline, 'color', '#1976D2')
    plt.setp(baseline, 'visible', False)
    ax.set_xticks(np.arange(len(srt)))
    ax.set_xticklabels(K[srt], rotation = -90,)
    ax.set_ylabel("Test AUC")
    sns.despine(ax = ax)
    plt.tight_layout(rect = (0.05, 0.05, 0.95, 0.9))
    pdf.savefig()
    plt.close()

if args.ci_file is not None:
    from plot import behavior
    from plot import util

    center = np.median(list(auc_scores.values()))
    lo = np.min(list(auc_scores.values()))
    hi = np.max(list(auc_scores.values()))
    # median_agg = lambda arr: np.median(arr, axis = 1)
    # ci = util.mean_ci(list(auc_scores.values()), 1000, aggfunc = median_agg)

    behavior.update_ci_text(args.ci_file[0],
        **{args.ci_file[1]: (lo, center, hi)})







