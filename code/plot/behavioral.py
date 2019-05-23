from pprint import pprint
import numpy as np
import skvideo.io

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib import colors
import matplotlib.patches as patches
import seaborn as sns
sns.set(color_codes = True)
sns.set_style('ticks')


def task_performance(filename, fs, ys, data_groups):
    '''
    Show performance of various models on a binary classification task
    ### Arguments
    - `filename` --- A path to a pdf file where plots will be saved.
    - `fs` --- A list of dictionaries, each of which maps a category
        to the decision function evaluated on a set of inputs.
    - `ys` --- A list of dictionaries, with order correpsonding to
        that of `fs`, where each dictionary maps categories to a
        boolean array giving the true binary class of that input
    - `data_groups` --- The data groups that each item in the `fs`
        and `ys` lists comes from. For example, this might be
        `['Train', 'Validation']`. Or could give the names of different
        models being compared
    '''
    categories = ys[0].keys()
    n_grp = len(data_groups)
    with PdfPages(filename) as pdf:
        for c in categories:
            fs_flat = np.concatenate([fs_i[c] for fs_i in fs])
            ys_flat = np.concatenate([ys_i[c] for ys_i in ys])
            group_flat = np.concatenate([
                np.repeat(data_groups[i], len(fs[i][c]))
                for i in range(n_grp)])

            ax = sns.stripplot(y = fs_flat,
                          x = ys_flat,
                          hue = group_flat,
                          dodge=True,
                          size=3)
            ax.axhline(0, 0, 1, ls = '--', color = "#aaaaaa", lw = 1)
            sns.despine(ax = ax)
            plt.title("Task Performance | Category: " + c)
            plt.ylabel("Decision Function")
            pdf.savefig()
            plt.close()


