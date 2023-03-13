import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import optimize
import pandas as pd
import numpy as np

from plot import util
from plot import kwargs


sns.set_context('paper')
sns.set_style('ticks')

pal = pd.read_csv(Paths.data('cfg/pal_categ_1.csv'))['color'].values
dps = pd.read_csv(Paths.data('human_dps.csv'))

curve_xrange = (0, .3)
duration_to_ms_factor = 1
curves = [[0.3147, 154.4935],
          [0.4897, 167.4127]]
cond_names = ["Distributed", "Focal"]
cond_keys = [1, 2]
MODEL_DIST_D = 0.75


plt.figure()

for cond_key, cond_curve, cond_color in zip(cond_keys, curves, pal):
    # psychometric curve
    curve_eval = np.linspace(*curve_xrange, 300)
    plt.plot(
        curve_eval, cond_curve[0] * np.log(cond_curve[1] * curve_eval + 1),
        '-', lw = 2, color = cond_color)
    # aggregate data across participants
    dp_mask = (dps['Attend'] == cond_key)
    aggregated_data = np.array([
        [duration * duration_to_ms_factor] +
        # mean within duration
        [duration_data['dprime'].mean()] +
        # 95% confidence interval around mean
        util.mean_ci(duration_data['dprime'], n = 1000).tolist()
        for duration, duration_data in dps.loc[dp_mask].groupby('Duration')
    ])
    # plot mean dprime at each duration with CI
    for i in range(aggregated_data.shape[0]):
        plt.plot(
            [aggregated_data[i, 0], aggregated_data[i, 0]],
            aggregated_data[i, 2:],
            '-', color = cond_color, zorder = 2,
            **kwargs.bhv_ci)
    plt.plot(
        aggregated_data[:, 0], aggregated_data[:, 1],
        color = cond_color, zorder = 3, **kwargs.bhv_mean)

dist_curve = curves[1]
match_duration = 1/dist_curve[1] * (np.exp(MODEL_DIST_D / dist_curve[0]) - 1)
match_dist_d = dist_curve[1] * np.log(dist_curve[0] * match_duration + 1)
plt.axvline(match_duration, lw = 1, color = 'k')
plt.plot(match_duration, match_dist_d, 'X', ms = 6, color = 'k', mec = 'w', mew = 1, zorder = 2)





