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



parser = ArgumentParser(
    description = 
        "Plot summaries of receptive field statistics.")
parser.add_argument('output_path',
    help = 'Path to PDF file where the plots should go.')
parser.add_argument("bhv_files", nargs = '+',
    help = 'HDF5 behavior archives')
parser.add_argument('--disp', nargs = '+', default = None, type = str,
    help = 'Display names for the given files. These may overlap ' + 
           'if --cond also specified.')
parser.add_argument('--cond', nargs = '+', default = None,
    help = 'Experimental condition for given files that will be ' + 
           'mapped onto hue.')
parser.add_argument('--cmp',
    help =  'HDF5 behavior arvhive containing a separate control condition.')
parser.add_argument('--cmp_disp',
    help =  'Name for the separate control condition.')
parser.add_argument('--y_rng', type = float, nargs = 2, default = None,
    help =  'Y-axis range, two floats: (min, max)')
parser.add_argument('--bar1', type = float, default = None,
    help = 'Weak comparison bar')
parser.add_argument('--bar2', type = float, default = None,
    help = 'Stronger comparison bar')
parser.add_argument('--metric', type = str, default = 'auc',
    help = 'Performance metric: auc or acc')
parser.add_argument('--sns_context', type = str, default = None)
parser.add_argument('--scores_out', type = str, default = None)
parser.add_argument('--figsize', type = float, nargs = 2, default = (6, 4))
parser.add_argument('--jitter', type = float, default = 0.03)
parser.add_argument('--bootstrap_n', type = int, default = 1000)
args = parser.parse_args()

"""
Test args:
class args:
    output_path = '/Users/kaifox/tmp/tmp.pdf'
    bhv_files = ['data/runs/fig2/bhv_gauss_n300_beta_1.1.h5',
                 'data/runs/fig2/bhv_gauss_n300_beta_2.0.h5',
                 'data/runs/fig2/bhv_gauss_n300_beta_4.0.h5',
                 'data/runs/fig2/bhv_gauss_n300_beta_11.0.h5']
    disp = ['1.1', '2.0', '4.0', '11.0']
    cond = None
    cmp = 'data/runs/fig2/bhv_base.h5'
    cmp_disp = 'Dist'
    figsize = (6,4)
    jitter = 0.15
    bar1 = None
    bar2 = None
    metric = 'auc'
    yrng = (0, 1)
"""

# -------------------------------------- Load inputs ----

# Behavioral archive files
data = []
cats = None
for i_f, fname in enumerate(args.bhv_files):
    print("file:", fname)
    f = h5py.File(fname, 'r+')
    if args.cond is None:
        f_disp = fname if args.disp is None else args.disp[i_f]
    else:
        f_disp = fname if args.disp is None else args.disp[i_f]
    # Get categories and make sure they match
    catkeys = [k for k in f.keys() if k.endswith('_y')]
    # New format
    if len(catkeys) > 0:
        f_cats = np.array([n.decode() for n in f[catkeys[0]].attrs['cat_names']])
    # Old format
    else:
        f_cats = np.array([n.decode() for n in f['cat_ids'].attrs['cat_names']])
    if cats is None:
        cats = f_cats
    if not all(f_cats == cats):
        raise ValueError("Categories in {fname} don't match other files.")
    # Organize the data into pandas for grouping
    # Old format
    if 'cat_ids' in f.keys():
        df = pd.DataFrame(dict(
            cat = f_cats[f['cat_ids'][...]],
            fn = np.concatenate([
                f[cat][...]
                for cat in f_cats]),
            y = np.concatenate([
                f['true_ys'][...][f['cat_ids'][...] == i_cat]
                for i_cat, cat in enumerate(f_cats)])
        ))
    # New format
    else:
        df = pd.DataFrame(dict(
            cat = np.concatenate([
                [cat] * len(f[f'{cat}_fn'])
                for cat in f_cats]),
            fn = np.concatenate([
                f[f'{cat}_fn'][...]
                for cat in f_cats]),
            y = np.concatenate([
                f[f'{cat}_y'][...]
                for i_cat, cat in enumerate(f_cats)])
        ))
    df['i_f'] = i_f
    df['disp'] = f_disp
    if args.cond is not None:
        df['cond'] = args.cond[i_f]
    data.append(df)
    f.close()
data = pd.concat(data)

if args.cmp is not None:
    f = h5py.File(args.cmp, 'r')
    catkeys = [k for k in f.keys() if k.endswith('_y')]
    # New format
    if len(catkeys) > 0:
        f_cats = np.array([n.decode() for n in f[catkeys[0]].attrs['cat_names']])
    # Old format
    else:
        f_cats = np.array([n.decode() for n in f['cat_ids'].attrs['cat_names']])
    if not all(f_cats == cats):
        raise ValueError("Categories in {fname} don't match other files.")
    # Old format
    if 'cat_ids' in f.keys():
        cmp_data = pd.DataFrame(dict(
            cat = f_cats[f['cat_ids'][...]],
            fn = np.concatenate([
                f[cat][...]
                for cat in f_cats]),
            y = np.concatenate([
                f['true_ys'][...][f['cat_ids'][...] == i_cat]
                for i_cat, cat in enumerate(f_cats)])
        ))
    # New format
    else:
        cmp_data = pd.DataFrame(dict(
            cat = np.concatenate([
                [cat] * len(f[f'{cat}_fn'])
                for cat in f_cats]),
            fn = np.concatenate([
                f[f'{cat}_fn'][...]
                for cat in f_cats]),
            y = np.concatenate([
                f[f'{cat}_y'][...]
                for i_cat, cat in enumerate(f_cats)])
        ))
    cmp_data['disp'] = args.cmp if args.cmp_disp is None else args.cmp_disp



# -------------------------------------- Plot ----

if args.metric == 'auc':
    score_func = lambda y, fn: skmtr.roc_auc_score(y, fn)
elif args.metric == 'acc':
    score_func = lambda y, fn: (y == (fn > 0)).mean()

if args.sns_context is not None:
    print("set context to:", args.sns_context)
    sns.set_context(args.sns_context)

pal = mc.to_rgba_array([
    '#0288D1', '#C62828', '#FFB300', '#5E35B1', '#43A047'
])

sns.set('notebook')
sns.set_style('ticks')
cond_xs = {}
cond_xs_full = {}
cond_cat_scores = {}
cat_order = None
f_order = None
with PdfPages(args.output_path) as pdf:
    
    fig, ax = plt.subplots(figsize = args.figsize)

    # Compute where each file should go on x axis based on its display name
    uniq_disp = data['disp'].unique()
    disp_xs = dict(zip(uniq_disp, range(len(uniq_disp))))

    # Process separately by condition if requested
    if args.cond is not None:
        by_cond = data.groupby('cond', sort = False)
        ofs = ( np.arange(len(by_cond)) - (len(by_cond) - 1) / 2 ) * 0.2
        cond_iter = zip(by_cond, ofs)
    else:
        cond_iter = [((None, data), 0)]

    for i_cond, ((cond, cond_data), cond_ofs) in enumerate(cond_iter):

        # Process each file's data into points / error bars
        groupby_f = cond_data.groupby('i_f')
        xs = np.array([
            disp_xs[f_data['disp'].iloc[0]]
            for i_f, f_data in groupby_f])
        xs_full = np.array([
            [disp_xs[f_data['disp'].iloc[0]]] * len(f_data['cat'].unique())
            for i_f, f_data in groupby_f])
        jtr = np.array([
            np.random.uniform(
                low = -args.jitter, high = args.jitter,
                size = len(f_data['cat'].unique()))
            for i_f, f_data in groupby_f])
        cat_scores = [
            [ score_func(cat_data['y'], cat_data['fn'])
            for cat, cat_data in f_data.groupby('cat')]
            for i_f, f_data in groupby_f
        ]
        means = np.array([np.mean(l) for l in cat_scores])
        cis = np.array([util.mean_ci(l, args.bootstrap_n) for l in cat_scores])

        # Save arranged data for significance tests
        cond_xs[cond] = xs
        cond_xs_full[cond] = xs_full
        cond_cat_scores[cond] = cat_scores
        if cat_order is None:
            cat_order = [
                [ cat for cat, cat_data in f_data.groupby('cat')]
                for i_f, f_data in groupby_f
            ]
            f_order = [
                [ f_data['disp'].iloc[0] for cat, cat_data in f_data.groupby('cat')]
                for i_f, f_data in groupby_f
            ]

        # Tie in to legend if plotting multiple conditions
        if args.cond is not None:
            lab = {'label': cond}
        else: lab = {}

        # Plot raw data / by category
        ax.scatter(
            (xs_full + jtr + cond_ofs).ravel(),
            np.array(cat_scores).ravel(),
            color = 0.45 * pal[i_cond] + 0.5 * np.ones(4),
            zorder = -2, **plot.kwargs.bhv_cat)
        # Plot means
        ax.scatter(
            xs + cond_ofs, means, color = pal[i_cond], zorder = 2,
            **plot.kwargs.bhv_mean, **lab)
        # CIs
        for x, ci in zip(xs + cond_ofs, cis):
            ax.plot([x, x], ci, zorder = 1, color = '.2',
                    **plot.kwargs.bhv_ci)

        # ax.errorbar(
        #     x = xs + cond_ofs, y = means,
        #     yerr = np.stack([means - mins, maxs - means]),
        #     marker = 's', lw = 1, color = pal[i_cond],
        #     ls = '-' if args.cond is None else '',
        #     **lab)

    if args.bar1 is not None:
        ax.axhline(args.bar1, ls = '--', lw = 1,
            color = '#37474F', alpha = 0.3, zorder = -3)
    if args.bar2 is not None:
        ax.axhline(args.bar2, ls = '-', lw = 1,
            color = '#263238', alpha = 0.3, zorder = -3)

    # Plot the comparison file
    if args.cmp is not None:
        cat_scores = [
            score_func(cat_data['y'], cat_data['fn'])
            for cat, cat_data in cmp_data.groupby('cat')]
        jtr = np.random.uniform(
            low = -0.03, high = 0.03,
            size = len(cat_scores))
        means = np.mean(cat_scores)
        cis = util.mean_ci(cat_scores, args.bootstrap_n)


        # Plot raw data / by category
        ax.scatter(
            np.full(len(cat_scores), -1) + jtr,
            cat_scores,
            color = [0.45 * .2 + 0.5] * 3,
            zorder = -2, **plot.kwargs.bhv_cat)
        # Plot mean
        ax.scatter(
            [-1], [means], color = '.2', zorder = 2,
            **plot.kwargs.bhv_mean, **lab)
        # CI
        ax.plot([-1, -1], cis, zorder = 1, color = '.2',
                **plot.kwargs.bhv_ci)

        # save comparison file scores for significance tests
        comp_cat_scores = cat_scores
        # ax.errorbar(
        #     x = [-1], y = [means],
        #     yerr = [[means - mins], [maxs - means]],
        #     marker = 's', color = '#37474F', lw = 1)

    # Label the x axis according to the display
    if args.cmp is not None:
        ax.set_xticks(np.arange(-1, len(uniq_disp)))
        ax.set_xticklabels([args.cmp_disp] + [k for k in uniq_disp])
    else:
        ax.set_xticks(np.arange(len(uniq_disp)))
        ax.set_xticklabels([k for k in uniq_disp])

    if args.y_rng is not None:
        ax.set_ylim(*args.y_rng)

    ax.set_xlabel("Attention Strength, Beta")
    if args.metric == 'auc':
        ax.set_ylabel("AUC")
    elif args.metric == 'acc':
        ax.set_ylabel("Percent Correct")
    sns.despine(ax = ax)
    plt.tight_layout()
    pdf.savefig()

if args.scores_out is not None:
    score_df = dict(disp = [], cond = [], cat = [], score = [])
    for cond in cond_cat_scores:
        score_df['score'] += [a for b in cond_cat_scores[cond] for a in b]
        score_df['cat'] += [a for b in cat_order for a in b]
        disps = [a for b in f_order for a in b]
        score_df['disp'] += disps
        score_df['cond'] += [cond] * len(disps) 
    pd.DataFrame(score_df).to_csv(args.scores_out, index = False)



# ------------- Statistical significance tests -----

exit()

# Test 1:
# Check for significant monotonic relationship between attention strength
# and AUC within each condition


cond_monotonic_stats = {}
for i_cond, ((cond, cond_data), cond_ofs) in enumerate(cond_iter):
    xs = cond_xs_full[cond]
    scores = cond_cat_scores[cond]
    cond_monotonic_stats[cond] = stats.pearsonr(
        np.array(xs).ravel(), stats.rankdata(scores).ravel())


# Test 2:
# Check for significant difference between each attention strength and the
# comparison file

cond_pairwise_stats = {}
for i_cond, ((cond, cond_data), cond_ofs) in enumerate(cond_iter):
    scores = cond_cat_scores[cond]
    cond_pairwise_stats[cond] = [
        (lambda r: (r.statistic, r.pvalue))(
            stats.ttest_rel(comp_cat_scores, scores[i]))
        for i in range(len(scores))
    ]
