
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
from scipy import optimize
from scipy import stats



def auc2d(auc): return np.sqrt(2) * stats.norm.ppf(auc)
def d2auc(d): return stats.norm.cdf(d / np.sqrt(2))


def log_psyc(d, rate, scale, bias_outer):
    inner = (d+1) * rate 
    inner = np.clip(inner, 1e-10, None)
    return scale * np.log(inner) + bias_outer

def log_psyc_inv(acc, rate, scale, bias_outer):
    return np.exp((acc - bias_outer) / scale) / rate - 1


def human_bhv_data(data_csv): 
    dps = pd.read_csv(data_csv)

    curve_xrange = (0, .3)
    curves = [[0.4897, 167.4127],
              [0.3147, 154.4935]]
    cond_names = ["Focal", "Distributed"]
    cond_keys = [2, 1]
    aggregated_data = []

    for cond_key in zip(cond_keys):
        dp_mask = (dps['Attend'] == cond_key)
        aggregated_data.append(np.array([
            [duration] +
            # mean within duration
            [duration_data['dprime'].mean()] +
            # 95% confidence interval around mean
            util.mean_ci(duration_data['dprime'], n = 1000).tolist()
            for duration, duration_data in dps.loc[dp_mask].groupby('Duration')
        ]))

    return aggregated_data, (curve_xrange, curves, cond_names)



def get_dprime(responsePresent, targetPresent):
    tpr = (responsePresent[targetPresent == 1] == 1).mean()
    fpr = (responsePresent[targetPresent == 0] == 1).mean()
    return stats.norm.ppf(tpr) - stats.norm.ppf(fpr)


def human_bhv_plot(ax, aggregated_data, curve_info, pal, MODEL_DIST_D, pkws = default_pkws, yticks = None):
    (curve_xrange, curves, cond_names) = curve_info
    curve_eval = np.linspace(*curve_xrange, 300)

    for cond_data, cond_curve, cond_color in zip(aggregated_data, curves, pal):
        # psychometric curve
        ax.plot(
            curve_eval, cond_curve[0] * np.log(cond_curve[1] * curve_eval + 1),
            '-', lw = 2, color = cond_color)
        # mean dprime at each duration with CI
        for i in range(cond_data.shape[0]):
            ax.plot(
                [cond_data[i, 0], cond_data[i, 0]],
                cond_data[i, 2:],
                '-', color = cond_color, zorder = 2,
                **pkws.bhv_ci)
        ax.scatter(
            cond_data[:, 0], cond_data[:, 1],
            color = cond_color, zorder = 3, **pkws.bhv_mean)

    dist_curve = curves[1]
    match_duration = 1/dist_curve[1] * (np.exp(MODEL_DIST_D / dist_curve[0]) - 1)
    match_dist_d = dist_curve[1] * np.log(dist_curve[0] * match_duration + 1)
    ax.axvline(match_duration, lw = 1, color = 'k')
    # ax.scatter([match_duration], [match_dist_d], marker = 'X', 
    #     s = pkws.bhv_mean['s'], color = 'k',
    #     edgecolor = 'w', linewidth = 1, zorder = 2)

    ax.set_ylabel("Sensitivity [ d' (AUC) ]", **pkws.axis_label)
    ax.set_yticks(yticks if yticks is not None else ax.get_yticks()[::2])
    ax.set_yticklabels([
        '{:.1f} ({:.2f})'.format(y, d2auc(y))
        for y in ax.get_yticks()])
    plt.xlabel("Stimulus Duration [ ms ]", **pkws.axis_label)
    sns.despine()

    return match_duration





def bhv_data(input_files, comp, comp_disp, blacklist = ()):
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
                for cat in f_cats if cat not in blacklist]),
            fn = np.concatenate([
                f[f'{cat}_fn'][...]
                for cat in f_cats if cat not in blacklist]),
            y = np.concatenate([
                f[f'{cat}_y'][...]
                for i_cat, cat in enumerate(f_cats) if cat not in blacklist])
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
                for cat in f_cats if cat not in blacklist]),
            fn = np.concatenate([
                f[f'{cat}_fn'][...]
                for cat in f_cats if cat not in blacklist]),
            y = np.concatenate([
                f[f'{cat}_y'][...]
                for i_cat, cat in enumerate(f_cats) if cat not in blacklist])
        ))
        comp_data['disp'] = comp_disp
    return data, comp_data





def bhv_plot(data, comp_data, bar1, bar2, pal,
             ax, yrng, jitter, bootstrap_n, dodge = 0.2,
             pkws = default_pkws, trim_axes = False,
             offset_axes = 0, yticks = None, rawscores_df = None):
    score_func = lambda y, fn: skmtr.roc_auc_score(y, fn)
    rawscores_df_path = rawscores_df
    try:
        if rawscores_df is None: raise ValueError
        rawscores_df = pd.read_csv(rawscores_df, index_col = 0)
    except (pd.errors.EmptyDataError, ValueError) as e:
        rawscores_df = pd.DataFrame()

    # Compute where each file should go on x axis based on its display name
    uniq_disp = data['disp'].unique()
    disp_xs = dict(zip(uniq_disp, range(len(uniq_disp))))

    # Process separately by condition if requested
    by_cond = data.groupby('cond', sort = False)
    ofs = ( np.arange(len(by_cond)) - (len(by_cond) - 1) / 2 ) * dodge
    cond_iter = zip(by_cond, ofs)

    pal = mc.to_rgba_array(pal)


    if comp_data is not None:
        cis_ret = {
            'cond': [], 'disp': [],
            'lo': [], 'center': [], 'hi': [],
            'fx_lo': [], 'fx_center': [], 'fx_hi': []}
        comp_cat_scores = [
            score_func(cat_data['y'], cat_data['fn'])
            for cat, cat_data in comp_data.groupby('cat')]
    else:
        cis_ret = {
            'cond': [], 'disp': [],
            'lo': [], 'center': [], 'hi': []}

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
            for cat, cat_data in f_data.groupby('cat') ]
            for i_f, f_data in groupby_f
        ]
        rawscores_df = rawscores_df.iloc[np.argsort(rawscores_df.index)]
        for ii_f, (i_f, f_data) in enumerate(groupby_f):
            cats = [ cat for cat, _ in f_data.groupby('cat') ]
            cat_scores_sort = np.argsort(cats)
            rawscores_df[cond + '_' + f_data['disp'].iloc[0]] = np.array(cat_scores[ii_f])[cat_scores_sort]
            rawscores_df.index = np.array(cats)[cat_scores_sort]
        median_agg = lambda arr: np.median(arr, axis = 1)
        means = np.array([np.median(l) for l in cat_scores])
        cis = np.array([util.mean_ci(l, bootstrap_n, aggfunc = median_agg) for l in cat_scores])
        cis_ret['cond'] += [cond] * len(cis)
        cis_ret['disp'] += [f_data['disp'].iloc[0] for i_f, f_data in groupby_f]
        cis_ret['lo'] += [file_ci[0] for file_ci in cis]
        cis_ret['center'] += [m for m in means]
        cis_ret['hi'] += [file_ci[1] for file_ci in cis]
        if comp_data is not None:
            fx_means = np.array([
                np.median(np.array(focl) - comp_cat_scores) for focl in
                cat_scores])
            fx_cis = np.array([
                util.mean_ci(np.array(focl) - comp_cat_scores, bootstrap_n, aggfunc = median_agg) for focl in
                cat_scores])
            cis_ret['fx_lo'] += [ci[0] for ci in fx_cis]
            cis_ret['fx_center'] += [m for m in fx_means]
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
        cats = [ cat for cat, _ in comp_data.groupby('cat') ]
        cat_scores_sort = np.argsort(cats)
        rawscores_df[comp_data['disp'].iloc[0]] = np.array(cat_scores)[cat_scores_sort]
        means = np.median(cat_scores)
        cis = util.mean_ci(cat_scores, bootstrap_n, aggfunc = median_agg)
        cis_ret['cond'] += [comp_data['disp'].iloc[0]]
        cis_ret['disp'] += [comp_data['disp'].iloc[0]]
        cis_ret['lo'] += [cis[0]]
        cis_ret['center'] += [means]
        cis_ret['hi'] += [cis[1]]
        cis_ret['fx_lo'] += [np.nan]
        cis_ret['fx_center'] += [np.nan]
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

    if rawscores_df_path is not None: np.round(rawscores_df, 5).to_csv(rawscores_df_path, index = True)
    return pd.DataFrame(cis_ret)


def update_ci_text(ci_text_file, **kwargs):
    newtext = []
    with open(ci_text_file, 'r') as f:
        for line in f.readlines():
            # line structure \newcommand{\CINameOfCi}{0.75 [0.69, 0.78]}
            ci_name_start = len("\\newcommand{\\CI")
            ci_name_end = line.index("}")
            ci_name = line[ci_name_start:ci_name_end]
            if ci_name in kwargs:
                if isinstance(kwargs[ci_name], tuple) or isinstance(kwargs[ci_name], list) or isinstance(kwargs[ci_name], np.ndarray):
                    newtext.append("\\newcommand{\\CI" + ci_name + '}[1]{' +
                        f"{kwargs[ci_name][1]:.2f}#1 " + 
                        f"[{kwargs[ci_name][0]:.2f}, {kwargs[ci_name][2]:.2f}]" +
                        '}')
                else:
                    newtext.append("\\newcommand{\\CI" + ci_name + '}[1]{' +
                        kwargs[ci_name] +
                        '#1}')
                kwargs.pop(ci_name)
            elif len(line.strip()) > 0:
                newtext.append(line.strip())
    if len(kwargs) != 0:
        print("Did not find CI commands:", ', '.join(list(kwargs.keys())))
    with open(ci_text_file, 'w') as f:
        f.write('\n'.join(newtext))

def ci_text(df, cond, disp, col_prefix):
    row = df.index[(df['cond'] == cond) & (df['disp'] == disp)][0]
    return (
        df[col_prefix + 'lo'].iloc[row],
        df[col_prefix + 'center'].iloc[row],
        df[col_prefix + 'hi'].iloc[row])

def group_ci_text(df, group, col_prefix, col = 'group', col_suffix = ''):
    row = df.index[df[col] == group][0]
    return (
        df[col_prefix + 'lo' + col_suffix].iloc[row],
        df[col_prefix + 'center' + col_suffix].iloc[row],
        df[col_prefix + 'hi' + col_suffix].iloc[row])


