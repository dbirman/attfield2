
from plot import util
import plot.kwargs as default_pkws

import matplotlib.pyplot as plt
import matplotlib.colors as mc
import sklearn.metrics as skmtr
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np
import h5py


def auc2d(auc): return np.sqrt(2) * stats.norm.ppf(auc)
def d2auc(d): return stats.norm.cdf(d / np.sqrt(2))


def bhv_data(input_files, comp, comp_disp):
    bhv_files = list(input_files.values())
    disp, cond = list(zip(*input_files.keys()))

    data = []
    cats = None
    for i_f, fname in enumerate(bhv_files):
        f = h5py.File(fname, 'r+')
        f_disp = disp[i_f]
        # Get categories and make sure they match
        catkeys = [k for k in f.keys() if k.endswith('_y')]
        # New format
        f_cats = np.array([n.decode() for n in f[catkeys[0]].attrs['cat_names']])
        if cats is None:
            cats = f_cats
        if not all(f_cats == cats):
            raise ValueError("Categories in {fname} don't match other files.")
        # Organize the data into pandas for grouping
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
        df['cond'] = cond[i_f]
        data.append(df)
        f.close()
    data = pd.concat(data)

    if comp is not None:
        f = h5py.File(comp, 'r')
        catkeys = [k for k in f.keys() if k.endswith('_y')]
        f_cats = np.array([n.decode() for n in f[catkeys[0]].attrs['cat_names']])
        if not all(f_cats == cats):
            raise ValueError("Categories in {fname} don't match other files.")
        comp_data = pd.DataFrame(dict(
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
        comp_data['disp'] = comp_disp
    return data, comp_data





def bhv_plot(data, comp_data, bar1, bar2, pal,
             ax, yrng, jitter, bootstrap_n, dodge = 0.2,
             pkws = default_pkws, trim_axes = False,
             offset_axes = 0, yticks = None,):
    score_func = lambda y, fn: skmtr.roc_auc_score(y, fn)

    # Compute where each file should go on x axis based on its display name
    uniq_disp = data['disp'].unique()
    disp_xs = dict(zip(uniq_disp, range(len(uniq_disp))))

    # Process separately by condition if requested
    by_cond = data.groupby('cond', sort = False)
    ofs = ( np.arange(len(by_cond)) - (len(by_cond) - 1) / 2 ) * dodge
    cond_iter = zip(by_cond, ofs)

    pal = mc.to_rgba_array(pal)


    if comp_data is not None:
        cis_ret = {'cond': [], 'disp': [], 'lo': [], 'hi': [], 'fx_lo': [], 'fx_hi': []}
        comp_cat_scores = [
            score_func(cat_data['y'], cat_data['fn'])
            for cat, cat_data in comp_data.groupby('cat')]
    else:
        cis_ret = {'cond': [], 'disp': [], 'lo': [], 'hi': []}

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
                low = -jitter, high = jitter,
                size = len(f_data['cat'].unique()))
            for i_f, f_data in groupby_f])
        cat_scores = [
            [ score_func(cat_data['y'], cat_data['fn'])
            for cat, cat_data in f_data.groupby('cat')]
            for i_f, f_data in groupby_f
        ]
        means = np.array([np.mean(l) for l in cat_scores])
        cis = np.array([util.mean_ci(l, bootstrap_n) for l in cat_scores])
        cis_ret['cond'] += [cond] * len(cis)
        cis_ret['disp'] += [f_data['disp'].iloc[0] for i_f, f_data in groupby_f]
        cis_ret['lo'] += [ci[0] for ci in cis]
        cis_ret['hi'] += [ci[1] for ci in cis]
        if comp_data is not None:
            fx_cis = np.array([
                util.mean_ci(focl - dist, bootstrap_n) for focl, dist in
                zip(cat_scores, comp_cat_scores)])
            cis_ret['fx_lo'] += [ci[0] for ci in fx_cis]
            cis_ret['fx_hi'] += [ci[1] for ci in fx_cis]

        # Tie in to legend if plotting multiple conditions
        if cond is not None:
            lab = {'label': cond}
        else: lab = {}

        # Plot raw data / by category
        ax.scatter(
            (xs_full + jtr + cond_ofs).ravel(),
            np.array(cat_scores).ravel(),
            color = 0.45 * pal[i_cond] + 0.5 * np.ones(4),
            zorder = -2, **pkws.bhv_cat)
        # Plot means
        ax.scatter(
            xs + cond_ofs, means, color = pal[i_cond], zorder = 2,
            **pkws.bhv_mean, **lab)
        # CIs
        for x, ci in zip(xs + cond_ofs, cis):
            ax.plot([x, x], ci, zorder = 1, color = '.2',
                    **pkws.bhv_ci)

    bar_text_kws = dict(
        ha = 'left', va = 'bottom',
        zorder = -3,)
    data_to_ax = lambda y: ax.transAxes.inverted().transform(
        (0, ax.transData.transform(
            (1, y))[1]))[1]
    if bar1 is not None:
        ax.axhline(bar1, zorder = -3, **pkws.bhv_bar1)
        ax.text(
            len(uniq_disp) - 1 + ofs[-1] + dodge, bar1 + 0.01,
            'Human\nDist.', color = '#37474F',
            **bar_text_kws, **pkws.bhv_bar_text_kws)
    if bar2 is not None:
        ax.axhline(bar2, zorder = -3, **pkws.bhv_bar2)
        ax.text(
            len(uniq_disp) - 1 + ofs[-1] + dodge, bar2 + 0.01,
            'Human\nFocal', color = '#263238',
            **bar_text_kws, **pkws.bhv_bar_text_kws)

    # Plot the comparison file
    if comp_data is not None:
        cat_scores = [
            score_func(cat_data['y'], cat_data['fn'])
            for cat, cat_data in comp_data.groupby('cat')]
        jtr = np.random.uniform(
            low = -jitter, high = jitter,
            size = len(cat_scores))
        means = np.mean(cat_scores)
        cis = util.mean_ci(cat_scores, bootstrap_n)
        cis_ret['cond'] += [comp_data['disp'].iloc[0]]
        cis_ret['disp'] += [comp_data['disp'].iloc[0]]
        cis_ret['lo'] += [cis[0]]
        cis_ret['hi'] += [cis[1]]
        cis_ret['fx_lo'] += [np.nan]
        cis_ret['fx_hi'] += [np.nan]


        # Plot raw data / by category
        ax.scatter(
            np.full(len(cat_scores), -1) + jtr,
            cat_scores,
            color = [0.45 * .2 + 0.5] * 3,
            zorder = -2, **pkws.bhv_cat)
        # Plot mean
        ax.scatter(
            [-1], [means], color = '.2', zorder = 2,
            **pkws.bhv_mean, **lab)
        # CI
        ax.plot([-1, -1], cis, zorder = 1, color = '.2',
                **pkws.bhv_ci)

    # Label the x axis according to the display
    if comp_data is not None:
        ax.set_xticks(np.arange(-1, len(uniq_disp)))
        ax.set_xticklabels([comp_data['disp'].iloc[0]] + [k for k in uniq_disp])
    else:
        ax.set_xticks(np.arange(len(uniq_disp)))
        ax.set_xticklabels([k for k in uniq_disp])
    ax.set_ylim(yrng)
    if yticks is None:
        ax.set_yticks(ax.get_yticks())
    else:
        ax.set_yticks(yticks)
    ax.set_yticklabels([
        '{:.2f} ({:.1f})'.format(y, auc2d(y))
        for y in ax.get_yticks()])

    ax.set_xlabel(pkws.labels.bhv_beta, **pkws.axis_label)
    ax.set_ylabel(pkws.labels.bhv_performance, **pkws.axis_label)
    sns.despine(ax = ax, trim = trim_axes, offset = offset_axes)

    return pd.DataFrame(cis_ret)





