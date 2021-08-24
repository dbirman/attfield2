import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics as skmtr
from scipy import optimize
import pandas as pd
import numpy as np
import h5py



sns.set_context('paper')
sns.set_style('ticks')


# Parameters
F_RESOLUTION = 100
MATCH_BETA = '2.0'
MODEL_FOCL_DISP = 'Model: Gaussian 2x Gain'

# Load data
trials = pd.read_csv(Paths.data('human_pilot.csv'))
trials['ms'] = 4 * trials['duration']
cued = trials.loc[trials['focal'] == 1]
uncued = trials.loc[trials['focal'] == 0]
model_dist = h5py.File(Paths.data(
    'runs/fig2/bhv_base_n600.h5'), 'r+')
model_focl = h5py.File(Paths.data(
    f'runs/fig2/bhv_gauss_n600_beta_{MATCH_BETA}.h5'), 'r+')




# -----------------------------------------------------------  Preprocess  ----



# Determine human psychometric curves

# def log_psyc(d, rate, scale, bias_outer):
#     inner = (d+1) * rate 
#     inner = np.clip(inner, 1e-10, None)
#     return scale * np.log(inner) + bias_outer

# def log_psyc_inv(acc, rate, scale, bias_outer):
#     return np.exp((acc - bias_outer) / scale) / rate - 1

funcs = {}
feval_min =  np.inf
feval_max = -np.inf
for (df, key) in [(cued,'focl'),(uncued,'dist')]:
    group = df.groupby('duration')
    xs = [i + dodge
          for i in range(len(group))]
    ys = [(s['targetPresent'] == s['responsePresent']).mean()
          for _, s in group]
    feval_min = min(feval_min, np.min(xs))
    feval_max = max(feval_max, np.max(xs))

    popt, _ = optimize.curve_fit(log_psyc, xs, ys)
    funcs[key] = popt


def log_psyc(t, scale, rate, ofs):
    return scale * np.log(rate * t + ofs)

def log_psyc_inv(d, scale, rate, ofs):
    return (np.exp(d / scale) - ofs) / rate

funcs = {
    'dist': {'scale': 0.195, 'rate': 667.375, 'ofs': 1.},
    'focl': {'scale': 0.300, 'rate': 784.698, 'ofs': 1.}
}


def auc2d(auc): return np.sqrt(2) * stats.norm.ppf(auc)
def d2auc(d): return stats.norm.cdf(d / np.sqrt(2))


# Determine model distributed condition performance

catkeys = [k for k in model_dist.keys() if k.endswith('_y')]
f_cats = model_dist[catkeys[0]].attrs['cat_names']
f_cats = np.array([n.decode() for n in f_cats])
dist_responses = pd.DataFrame(dict(
    fn = np.concatenate([
        model_dist[f'{cat}_fn'][...]
        for cat in f_cats]),
    y = np.concatenate([
        model_dist[f'{cat}_y'][...]
        for i_cat, cat in enumerate(f_cats)])
))
dist_responses['response'] = dist_responses['fn'] > 0
# acc_dist = (dist_responses['y'] == dist_responses['response']).mean()
auc_dist = skmtr.roc_auc_score(dist_responses['y'], dist_responses['fn'])


# Determine model focal condition performance

catkeys = [k for k in model_focl.keys() if k.endswith('_y')]
f_cats = model_focl[catkeys[0]].attrs['cat_names']
f_cats = np.array([n.decode() for n in f_cats])
focl_responses = pd.DataFrame(dict(
    fn = np.concatenate([
        model_focl[f'{cat}_fn'][...]
        for cat in f_cats]),
    y = np.concatenate([
        model_focl[f'{cat}_y'][...]
        for i_cat, cat in enumerate(f_cats)])
))
focl_responses['response'] = focl_responses['fn'] > 0
# acc_focl = (focl_responses['y'] == focl_responses['response']).mean()
auc_focl = skmtr.roc_auc_score(focl_responses['y'], focl_responses['fn'])


# Find matching time between model and human

t_match_dist = log_psyc_inv(auc2d(auc_dist), **funcs['dist'])
# t_match_focl = log_psyc_inv(acc_focl, **funcs['focl'])
model_auc_match = d2auc(log_psyc(t_match_dist, **funcs['focl']))

# -----------------------------------------------------------------  Plot  ----

pal = sns.color_palette()

# Intialize plot
fig, ax = plt.subplots(figsize = (4.5, 3))


# Plot model performance matches
plt.axhline(auc2d(auc_dist),
    ls = '--', lw = 1,
    color = pal[1], alpha = 0.7,
    label = 'Model: Distribued')
# plt.axhline(acc_focl,
#     ls = '--', lw = 1,
#     color = pal[0], alpha = 0.7,
#     label = MODEL_FOCL_DISP)

# Plot psychometric function

for (f_key, color, label) in [
        ('dist',pal[1], 'Human: Distibuted'),
        ('focl',pal[0], 'Human: Focal')]:
    eval_x = np.linspace(0.01, 0.3, F_RESOLUTION)
    eval_y = log_psyc(eval_x, **funcs[f_key])
    plt.plot(eval_x, eval_y, color = color, label = label)
# plt.ylim(0, 3)

# Plot match points
plt.plot([t_match_dist], [acc_dist],
    ls = '', marker = 'o', ms = 6, mew = 1, mec = 'w',
    color = pal[1])
plt.plot([t_match_focl], [acc_focl],
    ls = '', marker = 'o', ms = 6, mew = 1, mec = 'w',
    color = pal[0])


# Figure aesthetics
plt.legend(loc = 8, ncol = 2, frameon = False)
plt.ylim(0.3, 1.0)
ax.set_xticks(np.arange(len(np.unique(trials['ms'].values))))
ax.set_xticklabels(np.unique(trials['ms'].values))
plt.ylabel("Pecent Correct")
plt.xlabel("Stimulus Duration [ms]")
sns.despine()
plt.tight_layout()

plt.savefig(Paths.plots('figures/fig1/model_match.pdf'))



