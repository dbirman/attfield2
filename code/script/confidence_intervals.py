import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from plot import util

import pandas as pd
import numpy as np
data = pd.read_csv('data/runs/rawscores.csv', index_col = 0)

from scipy import stats
intervals = {}
for col in data.columns:
    ci = stats.bootstrap((data[col],), np.median).confidence_interval
    diffs = data[col] - data['Dist.']

    median_agg = lambda arr: np.median(arr, axis = 1)
    custom_ci = util.mean_ci(data[col].values, 1000, aggfunc = median_agg)

    if col != 'Dist.' and col != 'Reconstruct_dist':
        fx_ci = stats.bootstrap((diffs,), np.median).confidence_interval
        intervals[col] = [np.median(data[col]), ci.low, ci.high, np.median(diffs), fx_ci.low, fx_ci.high, custom_ci[0], custom_ci[1]]
    else:
        intervals[col] = [np.median(data[col]), ci.low, ci.high, "", "", "", custom_ci[0], custom_ci[1]]
out_df = pd.DataFrame(intervals)
out_df.index = ['center', 'lo', 'hi', 'fx_center', 'fx_lo', 'fx_hi', 'lo2', 'hi2']
np.round(out_df, 5).to_csv('data/runs/rawscores_ci.csv', index = True)