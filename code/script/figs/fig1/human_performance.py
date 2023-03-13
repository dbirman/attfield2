import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import optimize
import pandas as pd
import numpy as np


sns.set_context('paper')
sns.set_style('ticks')

# Parameters
DODGE = 0.
F_RESOLUTION = 100

trials = pd.read_csv(Paths.data('human_final.csv'))
pal = pd.read_csv(Paths.data('cfg/pal_categ_1.csv'))['color'].values

# Convert duration column to milliseconds
trials['ms'] = 4 * trials['duration']
# Split into focal and distributed trial
cued = trials.loc[trials['focal'] == 1]
uncued = trials.loc[trials['focal'] == 0]




# ------------------------------ Psychometric Function Fit


def log_psyc(d, rate, scale, bias_outer):
    inner = (d+1) * rate 
    inner = np.clip(inner, 1e-10, None)
    return scale * np.log(inner) + bias_outer

def log_psyc_inv(acc, rate, scale, bias_outer):
    return np.exp((acc - bias_outer) / scale) / rate - 1

funcs = {}
feval_min =  np.inf
feval_max = -np.inf
for (df, key) in [(cued,'focl'),(uncued,'dist')]:
    group = df.groupby('duration')
    xs = [i + DODGE
          for i in range(len(group))]
    ys = [(s['targetPresent'] == s['responsePresent']).mean()
          for _, s in group]
    feval_min = min(feval_min, np.min(xs))
    feval_max = max(feval_max, np.max(xs))

    popt, _ = optimize.curve_fit(log_psyc, xs, ys)
    funcs[key] = popt


# ------------------------------ Plot

fig, ax = plt.subplots(figsize = (4.5, 3))

for (f_key, color) in [
        ('dist',pal[1]),
        ('focl',pal[0])]:
    eval_x = np.linspace(feval_min, feval_max, F_RESOLUTION)
    eval_y = log_psyc(eval_x, *funcs[f_key])
    plt.plot(eval_x, eval_y, color = color)

for (df, label, color, dodge) in [
        (cued, 'Focal', pal[0], -DODGE),
        (uncued, 'Distributed', pal[1], DODGE)]:

    group = df.groupby('duration')
    xs_err = [i + dodge
              for i in range(len(group))]
    ys_err = [(s['targetPresent'] == s['responsePresent']).mean()
              for _, s in group]

    plt.plot(xs_err, ys_err,
        ls = '', lw = 0, color = color,
        marker = 'o', ms = 6, mew = '1', mec = 'w',
        label = label)


plt.ylim(0.3, 1.0)
plt.axhline(0.5, lw = 1, ls = '--', color = '#37474F', alpha = 0.3)
ax.set_xticks(np.arange(len(np.unique(trials['ms'].values))))
ax.set_xticklabels(np.unique(trials['ms'].values))
plt.legend(frameon = False, loc = 4)
plt.ylabel("Pecent Correct")
plt.xlabel("Stimulus Duration [ms]")
sns.despine()
plt.tight_layout()

plt.savefig(Paths.plots('figures/fig1/human_performance.pdf'))

    """
    Code that might be useful once we have multiple subjects / full data

    # Each point will represent one category (*eventually subject*)
    point_groups = [f.groupby('category') for _, f in group]

    # Individual points: by category
    xs = [i + dodge + np.random.uniform(low = -JTR, high = JTR, size = len(pgrp))
          for (i, pgrp) in enumerate(point_groups)]
    ys = [[(s['targetPresent'] == s['responsePresent']).mean()
           for _, s in pgrp] for pgrp in point_groups]

    plt.scatter(
        x = np.ravel(xs), y = np.ravel(ys),
        color = color)

    # xs_err = [i + dodge for i in range(len(group))]
    # ys_err = [s.mean() for _, s in group]
    # q25_err = [np.quantile(s, .25) - s.mean() for _, s in group]
    # q75_err = [np.quantile(s, .75) - s.mean() for _, s in group]

    plt.errorbar(
        x = xs_err, y = ys_err,
        yerr = np.stack([q25_err, q75_err]),
        label = label, color = color)
