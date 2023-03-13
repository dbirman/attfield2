import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot import kwargs as default_pkws


# -------------------------------------- Panel / plot structure ----

def panel_label(fig, gs, panel, xoffset = -0.09, off = True):
    ax = fig.add_subplot(gs, label = panel)
    ax.text(
        -xoffset, 1.05, panel,
        ha = 'right', transform = ax.transAxes,
        fontsize = 8, fontweight = '700', 
    )
    if off: ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def labels(ax, xlab, ylab, pkws = default_pkws):
    if xlab is not None:
        ax.set_xlabel(xlab, **pkws.axis_label)
    if ylab is not None:
        ax.set_ylabel(ylab, **pkws.axis_label)

def minigrid_labels(fig, axes, xlab, ylab, pkws = default_pkws):
    for r in range(axes.shape[0]):
        for c in range(axes.shape[1]):
            if c > 0:
                axes[r, c].set_yticklabels([""]*len(axes[r,c].get_yticks()))
            if r < axes.shape[0] - 1:
                axes[r, c].set_xticklabels([""]*len(axes[r,c].get_xticks()))
    # axes[0, 0].set_ylabel(ylab, **pkws.axis_label)
    # axes[-1, 0].set_xlabel(xlab, **pkws.axis_label)
    tl_bb = axes[0,0].get_tightbbox(fig.canvas.get_renderer())
    bl_bb = axes[-1,0].get_tightbbox(fig.canvas.get_renderer())
    tl_l_bb = axes[0,0].spines['left'].get_tightbbox(fig.canvas.get_renderer())
    bl_l_bb = axes[-1,0].spines['left'].get_tightbbox(fig.canvas.get_renderer()) 
    bl_b_bb = axes[-1,0].spines['bottom'].get_tightbbox(fig.canvas.get_renderer()) 
    br_b_bb = axes[-1,-1].spines['bottom'].get_tightbbox(fig.canvas.get_renderer())
    fig.text(
        (tl_bb.x0 - pkws.axis_label_margin) / pkws.rc['figure.dpi'],
        (tl_l_bb.y1 + bl_l_bb.y0) / 2 / pkws.rc['figure.dpi'],
        ylab, ha = 'center',
        transform = fig.dpi_scale_trans,
        rotation = 90, rotation_mode = 'anchor',
        **pkws.axis_label)
    fig.text(
        (bl_b_bb.x0 + br_b_bb.x1) / 2 / pkws.rc['figure.dpi'],
        (bl_bb.y0 - pkws.axis_label_margin) / pkws.rc['figure.dpi'],
        xlab, ha = 'center', va = 'top',
        transform = fig.dpi_scale_trans,
        **pkws.axis_label)


def axis_expand(ax, L, B, R, T):
    pos = ax.get_position()
    ax.set_position(
        [pos.x0 - L * pos.width, pos.y0 - B * pos.height,
         (1 + L + R) * pos.width, (1 + B + T) * pos.height])

def colorbar(
        fig, ax, mappable, label, ticks = None,
        margin = 0.01, shrink = 0.2, width = 0.01,
        label_margin = 0.012, label_vofs = 0,
        pkws = default_pkws):
    pos = ax.get_position()
    cax = fig.add_subplot(111, label = str(colorbar.unique_id))
    colorbar.unique_id += 1
    cax.set_position([
        pos.x0 + pos.width + margin,
        pos.y0 + pos.height * shrink  / 2,
        width,
        pos.height - pos.height * shrink])
    if ticks is None:
        ticks = [mappable.norm.vmin, mappable.norm.vmax]
    plt.colorbar(mappable, cax = cax, ticks = ticks)
    fig.text(
        (pos.x0 + pos.width + margin + width) + label_margin,
        pos.y0 + pos.height / 2 + label_vofs,
        label, ha = 'center', va = 'top',
        rotation = 90, rotation_mode = 'anchor',
        **pkws.colorbar_label)
colorbar.unique_id = 0


def legend_transform_data(fig, ax, inset, inset_y, pkws):
    text_height = ax.transAxes.inverted().transform((0, 
        fig.dpi_scale_trans.transform((0, pkws.legend_text['fontsize']))[1] /
        pkws.rc['figure.dpi']))[1] - ax.transAxes.inverted().transform((0,0))[1]
    if inset_y is None: inset_y = inset
    inset_y = ax.transAxes.inverted().transform((0,
        inset_y / 2.54 * pkws.rc['figure.dpi']))[1
        ] - ax.transAxes.inverted().transform((0,0))[1]
    inset_x = ax.transAxes.inverted().transform((
        inset / 2.54 * pkws.rc['figure.dpi'], 0))[0
        ] - ax.transAxes.inverted().transform((0, 0))[0]
    return text_height, inset_x, inset_y

def legend(fig, ax, labels, pal, inset = 0, left = False, inset_y = None, pkws = default_pkws):
    text_height, inset_x, inset_y = legend_transform_data(fig, ax, inset, inset_y, pkws)
    for i_l, lab in enumerate(labels):
        if left:
            ax.text(
                inset_x, 1 - i_l * pkws.legend_line_height * text_height - inset_y,
                lab, color = pal[i_l],
                ha = 'left', va = 'top',
                transform = ax.transAxes,
                **pkws.legend_text)
        else:
            ax.text(
                1 - inset_x, 1 - i_l * pkws.legend_line_height * text_height - inset_y,
                lab, color = pal[i_l],
                ha = 'right', va = 'top',
                transform = ax.transAxes,
                **pkws.legend_text)


def line_legend(
    fig, ax, labels, line_kw, color,
    line_length = 0.07, line_pad = 0.03,
    inset = 0, left = False, inset_y = None, pkws = default_pkws):

    text_height, inset_x, inset_y = legend_transform_data(fig, ax, inset, inset_y, pkws)
    for i_l, (lab, lkw) in enumerate(zip(labels, line_kw)):
        y = 1 - i_l * pkws.legend_line_height * text_height - inset_y
        ax.plot(
            [inset_x, inset_x + line_length],
            [y - text_height / 2] * 2,
            color = color,
            zorder = -1,
            transform = ax.transAxes,
            **lkw)
        ax.text(
            inset_x + line_length + line_pad, y,
            lab, color = color,
            ha = 'left', va = 'top',
            transform = ax.transAxes,
            **pkws.legend_text)



# -------------------------------------- Binned-mean average line and CIs ----

def mean_ci(arr, n, aggfunc = lambda x: x.mean(axis = 1)):
    arr = np.array(arr)
    if arr.size > 1e3:
        print("Measuring ci for large array:", arr.size)
    return np.quantile(
        aggfunc(np.random.choice(arr.ravel(), (n, arr.size), replace = True)),
        [0.025, 0.975])

def mean_ci_table(labels, arrs, n, aggfunc = lambda x: x.mean(axis = 1)):
    cis = [mean_ci(arr, n, aggfunc = aggfunc) for arr in arrs]
    centers = [aggfunc(np.array([arr]))[0] for arr in arrs]
    return pd.DataFrame({
        'group': list(labels),
        'lo': list(ci[0] for ci in cis),
        'center': list(c for c in centers),
        'hi': list(ci[1] for ci in cis),
        })

def binned_mean_line(xs, ys, n_bins, boostrap_n):
    # bin_edges = np.quantile(xs, np.linspace(0, 1, n_bins + 1))
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    if isinstance(xs, pd.Series): xs = xs.values
    ixs = np.argsort(xs)
    binned_ixs = np.array_split(ixs, n_bins)
    bin_centers = [xs[bin_assign].mean() for bin_assign in binned_ixs]
    bin_means = np.array([
        ys[bin_].mean()
        for bin_ in binned_ixs])
    bin_cis = np.array([
        mean_ci(ys[bin_], boostrap_n)
        for bin_ in binned_ixs])
    return (
        bin_centers, bin_means,
        bin_cis[:, 0], bin_cis[:, 1]
    )


def running_mean_line(xs, ys, span, res = 400):
    x_eval = np.linspace(xs.min(), xs.max(), res)
    yout = loess(xs, ys, x_eval, span)
    return x_eval, yout
    # if isinstance(xs, pd.Series): xs = xs.values
    # if isinstance(ys, pd.Series): ys = ys.values
    # sort_ix = np.argsort(xs)
    # means = pd.Series(ys[sort_ix]).rolling(
    #     window = span, win_type = 'cosine', min_periods = span).mean()
    # line_xs = np.roll(xs[sort_ix], span//2)[span:]
    # # line_xs = xs[sort_ix]
    # return line_xs, means.values[span:]


def loess(xs, ys, x_eval, span = 30):
    if isinstance(xs, pd.Series): xs = xs.values
    dists = abs(x_eval[None, :] - xs[:, None])
    normed_dists = np.clip(dists / span, -1, 1)
    w = (1 - (normed_dists)**3)**3
    yhat = []
    for i in range(len(x_eval)):
        X = np.stack([np.ones(len(xs)), xs]).T
        W = np.diag(w[:, i])
        beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ ys
        X_eval = np.stack([np.ones(len(x_eval)), x_eval]).T
        yhat.append(X_eval @ beta[:, None])
    yhat = np.stack(yhat)[np.diag_indices(len(yhat))]
    return yhat

# -------------------------------------- Group opacity scatterplot ----

def expand(x, y, gap=1e-4):
    add = np.tile([0, gap, np.nan], len(x))
    x1 = np.repeat(x, 3) + add
    y1 = np.repeat(y, 3) + add
    return x1, y1




