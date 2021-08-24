import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import voxel_selection as vx

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import seaborn as sns

from argparse import ArgumentParser
import pandas as pd
import numpy as np


parser = ArgumentParser(
    description = 
        "Train logistic regressions on isolated object detection task.")
parser.add_argument('output_path',
    help = 'Path to PDF file where the plot should go.')
parser.add_argument("pre_coms",
    help = 'RF summary with center of mass before attention applied.')
parser.add_argument("post_coms",
    help = 'RF summary with center of mass after attention applied.')
args = parser.parse_args()

"""
Test args:
class args:
    output_path = '/tmp/tmp.pdf'
    pre_coms = 'data/runs/210420/base_com-u100.csv'
    post_coms = 'data/runs/210420/beta_2.0_com.csv'
"""

# -------------------------------------- Load inputs ----

# Receptive fields
pre_coms = pd.read_csv(args.pre_coms)
att_coms = pd.read_csv(args.post_coms)
units = vx.VoxelIndex.from_serial(att_coms['unit'])
pre_coms = pre_coms.set_index('unit')
att_coms = att_coms.set_index('unit')


# -------------------------------------- Summarize ----

# Match receptive fields according to unit
coms = att_coms.join(pre_coms, on = 'unit',
    lsuffix = '_att', rsuffix = '_pre')

# Plot RF movement in a separate plot for each layer
sns.set('notebook')
sns.set_style('dark')
with PdfPages(args.output_path) as pdf:
    for layer in units:
        lstr = '.'.join(str(i) for i in layer)
        row_mask = coms.index.map(lambda u: u.startswith(lstr))
        rfs = coms.loc[row_mask]

        lines = np.stack([
            rfs[['com_x_pre', 'com_y_pre']].values,
            rfs[['com_x_att', 'com_y_att']].values
        ], axis = 1)

        fig, ax = plt.subplots(figsize = (6, 6))
        line_segments = LineCollection(lines,
            linestyle='solid', linewidth = 1)
        ax.add_collection(line_segments)
        ax.scatter(
            x = rfs['com_x_att'], y = rfs['com_y_att'],
            marker = 's', s = 4)
        ax.set_ylim(*ax.get_ylim()[::-1])
        ax.set_title(f"Layer: {layer}")
        pdf.savefig()
        plt.close()

        # dist = np.sqrt(
        #     (rfs['com_x_pre'] - rfs['com_x_att']) ** 2 + 
        #     (rfs['com_y_pre'] - rfs['com_y_att']) ** 2)



