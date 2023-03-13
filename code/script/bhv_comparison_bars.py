import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import optimize
import pandas as pd
import numpy as np

from scipy import optimize
from scipy import stats


def auc2d(auc): return np.sqrt(2) * stats.norm.ppf(auc)
def d2auc(d): return stats.norm.cdf(d / np.sqrt(2))

sns.set_context('paper')
sns.set_style('ticks')


def log_psyc(d, rate, scale, bias_outer):
    inner = (d+1) * rate 
    inner = np.clip(inner, 1e-10, None)
    return scale * np.log(inner) + bias_outer

def log_psyc_inv(acc, rate, scale, bias_outer):
    return np.exp((acc - bias_outer) / scale) / rate - 1

# parameters
MODEL_DIST_D = 0.75
F_RESOLUTION = 100


trials = pd.read_csv(Paths.data('human_final.csv'))
pal = pd.read_csv(Paths.data('cfg/pal_beta.csv'))['color'].values[[3, 0]]

# Convert duration column to milliseconds
trials['ms'] = 4 * trials['duration']
# Split into focal and distributed trial
cued = trials.loc[trials['focal'] == 1]
uncued = trials.loc[trials['focal'] == 0]

funcs = {}
feval_min =  np.inf
feval_max = -np.inf
for (df, key) in [(cued,'focl'),(uncued,'dist')]:
    group = df.groupby('duration')
    xs = [i for i in range(len(group))]
    tpr = [(lambda v: v['responsePresent'] == 1)(s.loc[s['targetPresent'] == 1]).mean()
           for _, s in group]
    fpr = [(lambda v: v['responsePresent'] == 1)(s.loc[s['targetPresent'] == 0]).mean()
           for _, s in group]
    ys = [stats.norm.ppf(t) - stats.norm.ppf(f) for t, f in zip(tpr, fpr)]
    feval_min = min(feval_min, np.min(xs))
    feval_max = max(feval_max, np.max(xs))

    popt, _ = optimize.curve_fit(log_psyc, xs, ys)
    funcs[key] = popt



fig, ax = plt.subplots(figsize = (4.5, 3))

for (f_key, color) in [
        ('dist',pal[1]),
        ('focl',pal[0])]:
    eval_x = np.linspace(feval_min, feval_max, F_RESOLUTION)
    eval_y = log_psyc(eval_x, *funcs[f_key])
    plt.plot(eval_x, eval_y, color = color)


plt.axhline(MODEL_DIST_D, lw = 1, ls = '--', color = '#37474F', alpha = 0.3)
print("Distributed d':", MODEL_DIST_D)
match_duration = log_psyc_inv(MODEL_DIST_D, *funcs['dist'])
focl_match = log_psyc(match_duration, *funcs['focl'])
print("Focal match: d' = ", focl_match, )
plt.axvline(match_duration, lw = 1, color = 'k')
plt.axhline(focl_match, lw = 1, color = '.7')
ax.set_xticks(np.arange(len(np.unique(trials['ms'].values))))
ax.set_xticklabels(np.unique(trials['ms'].values))
plt.legend(frameon = False, loc = 4)
plt.ylabel("D'")
plt.xlabel("Stimulus Duration [ms]")
sns.despine()
plt.tight_layout()
plt.show()


