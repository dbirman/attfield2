import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import voxel_selection as vx
from plot import util
import plot.kwargs as pkws

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import seaborn as sns

from argparse import ArgumentParser
from loess.loess_1d import loess_1d
import pandas as pd
import numpy as np


parser = ArgumentParser(
    description = 
        "Train logistic regressions on isolated object detection task.")
parser.add_argument('output_path',
    help = 'Path to PDF file where the plot should go.')
parser.add_argument("pre_ells",
    help = 'RF summary with center of mass before attention applied.')
parser.add_argument("post_ells", nargs = '+',
    help = 'One or more RF summary with center of mass after attention.')
parser.add_argument("--compare", nargs = '+', default = None,
    help = 'One or more RF summary with center of mass after comparison model.')
parser.add_argument("--loc", nargs = 2, type = float, required = True,
    help = 'Position of the center of the attention field (x, y)')
parser.add_argument("--rad", type = float, default = 1.,
    help = 'Radius of the attention field. Default 1, units = px.')
parser.add_argument('--disp', nargs = '+', default = None, type = str,
    help = 'Display names for the post_CoM summaries.')
parser.add_argument('--px_per_degree', type = float, default = None,
    help = 'Ratio of pixels to degrees of visual angle. If not given ' +
           'then shifts displayed in pixels.')
parser.add_argument('--pal_f', default = None, type = str,
    help = 'Color palette to use when color spread across the different' + 
           'post-attention activation files.')
parser.add_argument('--pal_l', default = None, type = str,
    help = 'Color palette to use when color spread across the different' + 
           'layers of the network.')
parser.add_argument('--ylim', default = (None, None), nargs = 2, type = float)
parser.add_argument('--xlim', default = (None, None), nargs = 2, type = float)
parser.add_argument('--figsize', default = (6,6), nargs = 2, type = float)
parser.add_argument('--mini_figsize', default = (2.5, 2.5), nargs = 2, type = float)
parser.add_argument('--line_span', default = 7, type = int)
args = parser.parse_args()

"""
Test args:
class args:
    output_path = '/tmp/tmp.pdf'
    pre_ells = 'data/runs/270420/summ_base_ell.csv'
    post_ells = [
        'data/runs/270420/summ_cts_gauss_b1.1_ell.csv',
        'data/runs/270420/summ_cts_gauss_b2.0_ell.csv']
    loc = [56, 56]
    rad = 1.
    disp = ["1.1", "2.0"]
    px_per_degree = 22.
"""

# -------------------------------------- Load inputs ----

# Receptive fields
pre_ells = pd.read_csv(args.pre_ells)
units = vx.VoxelIndex.from_serial(pre_ells['unit'])
pre_ells.set_index('unit', inplace = True)
att_ells = []
for fname in args.post_ells:
    att_ells.append(pd.read_csv(fname))
    att_ells[-1].set_index('unit', inplace = True)
    # Match receptive fields according to unit
    if not all(att_ells[-1].index == pre_ells.index):
        raise ValueError(
            f"Unit sets differ in {args.pre_ells} and {fname}")

comp_ells = []
if args.compare is not None:
    for fname in args.compare:
        comp_ells.append(pd.read_csv(fname))
        comp_ells[-1].set_index('unit', inplace = True)
        # Match receptive fields according to unit
        if not all(comp_ells[-1].index == pre_ells.index):
            raise ValueError(
                f"Unit sets differ in {args.pre_ells} and {fname}")


if args.pal_f is not None:
    pal_f = pd.read_csv(args.pal_f)['color']
else:
    pal_f = ['#0288D1', '#C62828', '#FFB300', '#5E35B1', '#43A047']

if args.pal_l is not None:
    pal_l = pd.read_csv(args.pal_l)['color']
else:
    pal_l = ['#0288D1', '#C62828', '#FFB300', '#5E35B1', '#43A047']



# -------------------------- Precompute location & shift ----

# Calculate distance from center in units of radii
dists = np.sqrt(
    ((pre_ells['com_x'] - args.loc[0]) / args.rad) ** 2 +
    ((pre_ells['com_y'] - args.loc[1]) / args.rad) ** 2)

pre_sizes = pre_ells.major_sigma * pre_ells.minor_sigma * np.pi / args.rad**2
pre_sizes_px = pre_ells.major_sigma * pre_ells.minor_sigma * np.pi

# For each RF group provided, calculate shift toward center
for rfs in att_ells:
    curr_sizes_px = rfs.major_sigma * rfs.minor_sigma * np.pi
    rfs['shift'] = np.log10(curr_sizes_px / pre_sizes_px)

for rfs in comp_ells:
    curr_sizes_px = rfs.major_sigma * rfs.minor_sigma * np.pi
    rfs['shift'] = np.log10(curr_sizes_px / pre_sizes_px)




# -------------------------------------- Plot ----

sns.set('notebook')
sns.set_style('ticks')
import matplotlib
matplotlib.rcParams.update(pkws.rc)

pal = ['#0288D1', '#C62828', '#FFB300', '#5E35B1', '#43A047']

# Plot RF movement in a separate plot for each layer
with PdfPages(args.output_path) as pdf:
    # Plot axis = layer, color = beta
    for layer in units:
        lstr = '.'.join(str(i) for i in layer)
        mask = pre_ells.index.map(lambda u: u.startswith(lstr))

        fig, ax = plt.subplots(figsize = args.figsize)
        for i_f, rfs in enumerate(att_ells):
            ax.scatter(
                dists[mask], rfs.loc[mask, 'shift'],
                color = pal_f[i_f],
                alpha = 0.7, zorder = 1,
                label = args.disp[i_f] if args.disp is not None else None,
                **pkws.lineplot_point)

            line_xs, line_ys = util.running_mean_line(
                dists[mask], rfs.loc[mask, 'shift'].values,
                args.line_span)
            ax.plot(
                line_xs, line_ys,
                color = pal_f[i_f],
                **pkws.avg_line)

            if args.compare is not None:
                comp_line_xs, comp_line_ys = util.running_mean_line(
                    dists[mask], comp_ells[i_f].loc[mask, 'shift'].values,
                    args.line_span)
                ax.plot(
                    comp_line_xs, comp_line_ys,
                    color = pal_f[i_f],
                    **pkws.avg_line_secondary)

        ax.set_ylim(args.ylim)

        if args.disp is not None:
            ax.legend(frameon = False)

        sns.despine(ax = ax)
        plt.tight_layout()

        ax.set_title(f'Layer: {layer}')
        if abs(args.rad - 1.) < 1e-13:
            ax.set_xlabel("Unit distance from center [px]", **pkws.axis_label)
        else:
            ax.set_xlabel("Unit distance from center [att-sigma]", **pkws.axis_label)
        ax.set_ylabel(r"RF Size Ratio, Cued / Uncued [ log10 ]", **pkws.axis_label)

        pdf.savefig(transparent = True)
        plt.close()
    
    # Plot axis = beta, color = layer
    for i_f, rfs in enumerate(att_ells):

        fig, ax = plt.subplots(figsize = args.figsize)
        for i_l, layer in enumerate(units):
            lstr = '.'.join(str(i) for i in layer)
            mask = pre_ells.index.map(lambda u: u.startswith(lstr))
            ax.scatter(
                dists[mask], rfs.loc[mask, 'shift'],
                color = pal_l[i_l],
                alpha = 0.7, zorder = 1,
                label = f'Layer: {lstr}',
                **pkws.lineplot_point)

            line_xs, line_ys = util.running_mean_line(
                dists[mask], rfs.loc[mask, 'shift'].values,
                args.line_span)
            ax.plot(
                line_xs, line_ys,
                color = pal_l[i_l],
                **pkws.avg_line)

            if args.compare is not None:
                comp_line_xs, comp_line_ys = util.running_mean_line(
                    dists[mask], comp_ells[i_f].loc[mask, 'shift'].values,
                    args.line_span)
                ax.plot(
                    comp_line_xs, comp_line_ys,
                    color = pal_l[i_l],
                    **pkws.avg_line_secondary)

        ax.set_ylim(args.ylim)

        if args.disp is not None:
            ax.legend(frameon = False)

        sns.despine(ax = ax)
        plt.tight_layout()

        ax.set_title(args.disp[i_f] if args.disp is not None else None)
        if abs(args.rad - 1.) < 1e-13:
            ax.set_xlabel("Unit distance from center [px]")
        else:
            ax.set_xlabel("Unit distance from center [att-sigma]")
        ax.set_ylabel(r"RF Size Ratio, Cued / Uncued [ log10 ]")

        pdf.savefig(transparent = True)
        plt.close()

    # Plot axis = beta, layer
    for clr_func in ( # plot with both layer & file coloring
        lambda i_f, i_l: pal_f[i_f],
        lambda i_f, i_l: pal_l[i_l]):

        for i_f, rfs in enumerate(att_ells):
            for i_l, layer in enumerate(units):
                lstr = '.'.join(str(i) for i in layer)
                mask = pre_ells.index.map(lambda u: u.startswith(lstr))

                fig, ax = plt.subplots(figsize = args.mini_figsize)
                ax.scatter(
                    dists[mask], rfs.loc[mask, 'shift'],
                    color = clr_func(i_f, i_l),
                    alpha = 0.7, zorder = 1,
                    **pkws.lineplot_point)

                line_xs, line_ys = util.running_mean_line(
                    dists[mask], rfs.loc[mask, 'shift'].values,
                    args.line_span)
                ax.plot(
                    line_xs, line_ys, zorder = 3,
                    color = clr_func(i_f, i_l),
                    **pkws.mini_avg_line)

                if args.compare is not None:
                    comp_line_xs, comp_line_ys = util.running_mean_line(
                        dists[mask], comp_ells[i_f].loc[mask, 'shift'].values,
                        args.line_span)
                    ax.plot(
                        comp_line_xs, comp_line_ys,
                        color = '.2', zorder = 1,
                        **pkws.mini_avg_line_secondary)

                ax.set_ylim(args.ylim)
                ax.set_yticks([ax.get_ylim()[0], 0, ax.get_ylim()[1]])
                ax.set_xlim(args.xlim)
                ax.set_xticks(ax.get_xlim())

                sns.despine(ax = ax)
                plt.tight_layout()

                ax.set_title(
                    f'Layer: {lstr} | ' +
                    args.disp[i_f] if args.disp is not None else None)
                if abs(args.rad - 1.) < 1e-13:
                    ax.set_xlabel("Unit distance from center [px]", **pkws.axis_label)
                else:
                    ax.set_xlabel("Unit distance from center [att-sigma]", **pkws.axis_label)
                ax.set_ylabel(r"RF Size Ratio, Cued / Uncued [ log10 ]", **pkws.axis_label)

                pdf.savefig(transparent = True)
                plt.close()







