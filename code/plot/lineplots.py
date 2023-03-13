from proc import voxel_selection as vx
from plot import util
import plot.kwargs as pkws

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def rf_data(pre_ells, post_ells, loc, rad, px_per_degree = None, layer_mask = None):

    pre_ells_fname = pre_ells
    pre_ells = pd.read_csv(pre_ells)
    get_layer_mask = lambda df: np.char.startswith(df['unit'].values.astype('str'), layer_mask)
    if layer_mask is not None:
        pre_ells = pre_ells.loc[get_layer_mask(pre_ells)]
    units = vx.VoxelIndex.from_serial(pre_ells['unit'])
    pre_ells.set_index('unit', inplace = True)
    att_ells = []
    for fname in post_ells:
        if fname is None:
            att_ells.append(None)
            continue
        att_ells.append(pd.read_csv(fname))
        if layer_mask is not None:
            att_ells[-1] = att_ells[-1].loc[get_layer_mask(att_ells[-1])]
        att_ells[-1].set_index('unit', inplace = True)
        # Match receptive fields according to unit
        # print(att_ells[-1].shape, pre_ells.shape)
        if not all(att_ells[-1].index == pre_ells.index):
            raise ValueError(
                f"Unit sets differ in {pre_ells_fname} and {fname}")

    dists = np.sqrt(
        ((pre_ells['com_x'] - loc[0]) / rad) ** 2 +
        ((pre_ells['com_y'] - loc[1]) / rad) ** 2)

    dists_px = np.sqrt(
        ((pre_ells['com_x'] - loc[0])) ** 2 +
        ((pre_ells['com_y'] - loc[1])) ** 2)

    # For each RF group provided, calculate shift toward center
    for rfs in att_ells:
        if rfs is None: continue
        curr_dists = np.sqrt(
            (rfs['com_x'] - loc[0]) ** 2 +
            (rfs['com_y'] - loc[1]) ** 2)
        rfs['shift'] = (dists_px - curr_dists)
        if px_per_degree is not None:
            rfs['shift'] /= px_per_degree


    # for each RF group provided calculate size change log10
    pre_sizes = np.sqrt(pre_ells.major_sigma * pre_ells.minor_sigma) * np.pi / rad**2
    pre_sizes_px = np.sqrt(pre_ells.major_sigma * pre_ells.minor_sigma) * np.pi
    for rfs in att_ells:
        if rfs is None: continue
        curr_sizes_px = np.sqrt(rfs.major_sigma * rfs.minor_sigma) * np.pi
        rfs['size'] = curr_sizes_px / pre_sizes_px

    return pre_ells, att_ells, dists, dists_px


def rf_file_iterator(field, dists, att_ells, layer, comp_ells = None):
    lstr = '.'.join(str(i) for i in layer)
    mask = att_ells[0].index.map(lambda u: u.startswith(lstr))
    for i_f, rfs in enumerate(att_ells):
        if comp_ells is not None and comp_ells[i_f] is not None:
                comp_masked = comp_ells[i_f].loc[mask, field].values
        else: comp_masked = None
        yield i_f, dists[mask], rfs.loc[mask, field].values, comp_masked

def rf_layer_iterator(field, dists, rfs, comp_rfs = None):
    units = vx.VoxelIndex.from_serial(rfs.index)
    for i_l, layer in enumerate(units):
        lstr = '.'.join(str(i) for i in layer)
        mask = rfs.index.map(lambda u: u.startswith(lstr))
        if comp_rfs is not None:
            comp_masked = comp_rfs.loc[mask, field].values
        else: comp_masked = None
        yield i_l, dists[mask], rfs.loc[mask, field].values, comp_masked


def gain_data(ells, sgain_files, loc):
    full_sd_gains = []
    for f in sgain_files:
        full_sd_gains.append({k: v for k, v in np.load(f).items()})
    # Calculate distance from center in units of radii
    units = vx.VoxelIndex.from_serial(ells.index)
    sd_gains = [{} for _ in full_sd_gains]
    for layer in units.keys():
        lstr = '.'.join(str(i) for i in layer)
        for i in range(len(sd_gains)):
            sd_gains[i][layer] = full_sd_gains[i][lstr][units[layer]._idx]
    return sd_gains


def gain_file_iterator(dists, gain_focl, layer, gain_comp = None):
    lstr = '.'.join(str(i) for i in layer)
    mask = dists.index.map(lambda u: u.startswith(lstr))
    for i_f in range(len(gain_focl)):
        if gain_comp is not None and gain_comp[i_f] is not None:
            comp = gain_comp[i_f][layer]
        else: comp = None
        focl = gain_focl[i_f][layer]
        yield i_f, dists[mask], focl, comp


def gain_layer_iterator(dists, gain_focl, gain_comp = None):
    for i_l, layer in enumerate(gain_focl.keys()):
        lstr = '.'.join(str(i) for i in layer)
        mask = dists.index.map(lambda u: u.startswith(lstr))
        if gain_comp is not None:
            comp = gain_comp[layer]
        else: comp = None
        yield i_l, dists[mask], gain_focl[layer], comp



def lineplot(
        rf_iterator, ax, line_span, rad, pal, px_per_degree = None,
        xlim = (None, None), ylim = (None, None),
        pkws = pkws, desctext = None, comp_only = False,
        peak_label_scales = (), peak_label_x = 150, include_peak_labels = ()):

    for i, dists, rfs, comp_rfs in rf_iterator:
        if not comp_only:
            ax.plot(
                *util.expand(dists, rfs),
                color = pal[i], alpha = 0.7,
                zorder = 2, 
                **pkws.lineplot_point)

            line_xs, line_ys = util.running_mean_line(
                dists.values, rfs,
                line_span, res = 400)
            ax.plot(
                line_xs, line_ys,
                color = pal[i], zorder = 3,
                **pkws.avg_line)

            # average line peak label
            if i in include_peak_labels:
                peak_y = np.max(line_ys)
                peak_x = line_xs[np.argmax(line_ys)]
                ax.annotate(
                    '{:.1f}%'.format(peak_y * peak_label_scales[i] * 100),
                    (peak_x, peak_y),
                    xytext = (peak_label_x, peak_y),
                    arrowprops = dict(lw = 0.5, arrowstyle = '-', color = '.4'),
                    color = pal[i],
                    va = 'center', ha = 'center',
                    fontsize = 7.0, fontstyle = 'italic',
                    zorder = 0)

        if comp_rfs is not None:
            comp_line_xs, comp_line_ys = util.running_mean_line(
                dists.values, comp_rfs,
                line_span)
            ax.plot(
                comp_line_xs, comp_line_ys,
                color = pal[i], zorder = 1,
                **pkws.avg_line_secondary)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    sns.despine(ax = ax)

    if desctext is not None:
        ax.text(0.95, 1.03, desctext, ha = 'right',
            color = 'k', fontsize = 6.0, fontstyle = 'italic',
            transform = ax.transAxes)



def mini_lineplot(
        rf_iterator, ax, line_span, rad, pal, px_per_degree = None,
        xlim = (None, None), ylim = (None, None), xticks = None, yticks = None,
        pkws = pkws):

    for i, dists, rfs, comp_rfs in rf_iterator:
        ax[i].plot(
            *util.expand(dists, rfs),
            color = pal[i],
            alpha = 0.7, zorder = 2,
            **pkws.lineplot_point)

        line_xs, line_ys = util.running_mean_line(
            dists, rfs,
            line_span, res = 400)
        ax[i].plot(
            line_xs, line_ys,
            color = pal[i], zorder = 3,
            **pkws.avg_line)

        if comp_rfs is not None:
            comp_line_xs, comp_line_ys = util.running_mean_line(
                dists, comp_rfs,
                line_span)
            ax[i].plot(
                comp_line_xs, comp_line_ys,
                color = '.2', zorder = 1,
                **pkws.mini_avg_line_secondary)

        ax[i].set_ylim(ylim)
        if yticks is None:
            ax[i].set_yticks([
                np.ceil(ax[i].get_ylim()[0]), 0, np.floor(ax[i].get_ylim()[1])])
        else:
            ax[i].set_yticks(yticks)
        if i != 0:
            ax[i].set_yticklabels([""] * len(ax[i].get_yticks()))
        ax[i].set_xlim(xlim)
        if i != len(ax) - 1:
            ax[i].set_xticks([])
        else:
            ax[i].set_xticks(xticks if xticks is not None else ax[i].get_xlim())
        sns.despine(ax = ax[i], trim = False, offset = 5,
            bottom = (i != len(ax) - 1))





