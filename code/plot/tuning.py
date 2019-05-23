from proc import video_gen

import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib import colors
import matplotlib.patches as patches
import seaborn as sns
sns.set(color_codes = True)
sns.set_style('ticks')


def colorbar_x(ax, thetas):
    rgb = video_gen.lab_colors(thetas)
    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    X, Y = np.meshgrid(
        thetas,
        np.linspace(-yrange/40, -3*yrange/40, 2))
    cmap = colors.ListedColormap(rgb)
    ax.pcolormesh(X, Y, X/max(thetas),
        cmap = cmap, vmin = 0, vmax = np.pi)


def cts_image_samples_x(video, n):
    video = (video-video.min())/(video.max()-video.min())
    def make_custom(ax, groups):
        sort_idx = np.argsort(groups)
        select_idx = np.linspace(0, len(video)-1, n).astype(int)
        images = video[sort_idx][select_idx]
        images = np.transpose(images, [0, 2, 3, 1])
        bbox = ax.get_position().bounds
        ax.set_position([
            bbox[0],
            bbox[1] + bbox[3]/6,
            bbox[2],
            5*bbox[3]/6,
            ])
        
        gs = gridspec.GridSpec(2, n, height_ratios=[5,1])
        for i, img in enumerate(images):
            sub_ax = ax.figure.add_subplot(gs[1, i])
            sub_ax.set_axis_off()
            sub_ax.imshow(img, interpolation = 'bilinear')
    return make_custom



def curve1d(filename, voxels, acts, grouping,
    custom_x = None, mode = 'line'):
    '''
    ### Arguments
    - `mode` --- `'line'` for sns.lineplot or `'cat'` for
        sns.catplot
    '''
    if mode == 'line': plot_fn = sns.lineplot
    elif mode == 'cat': plot_fn = sns.pointplot
    n_repeat = len(acts)
    repeat_group = np.tile(grouping, n_repeat)
    with PdfPages(filename) as pdf:
        for layer, layer_vox in voxels.items():
            for i_vox in range(layer_vox.nvox()):
                fig, ax  = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 3))
                plt.title("Tuning Curve |" + 
                          " Layer " + str(layer) + 
                          " Voxel " + str(i_vox))
                ys = np.concatenate([
                    acts[i][layer][:, i_vox]
                    for i in range(n_repeat)])
                ax = plot_fn(x = repeat_group, y = ys,
                    err_style = 'band',
                    ax = ax)
                #sns.despine(ax = ax)
                if custom_x is not None:
                    ax.set_xticklabels([])
                    custom_x(ax, grouping)
                pdf.savefig()
                plt.close()


def pca_cts(filename, voxels, acts, embedded, pca, dims = [0, 2, 4]):
    with PdfPages(filename) as pdf, sns.axes_style('darkgrid'):
        for layer, layer_vox in voxels.items():

            # Voxels in PCA space after having activations
            # averaged over input repetitions

            saved_lims = None
            _, ax = plt.subplots(nrows = 1, ncols = len(dims),
                figsize = (13, 4))
            for i, d in enumerate(dims):
                ax[i].scatter(
                    x = embedded[layer][:, d],
                    y = embedded[layer][:, d+1])
                ax[i].set_title("Dims [{},{}]".format(d, d+1))
                sns.despine(ax = ax[i])
                if d == 0:
                    saved_lims = (ax[i].get_xlim(), ax[0].get_ylim())
            plt.subplots_adjust(top = 0.85)
            plt.suptitle("Distribution of Tunings | " + 
                         "Layer " + str(layer))
            pdf.savefig()
            plt.close()

            for i_vox in range(layer_vox.nvox()):
                vox_embed = pca[layer].transform([
                    acts[i][layer][:, i_vox]
                    for i in range(len(acts))])
                _, ax = plt.subplots(nrows = 1, ncols = 2,
                    figsize = (8, 3))
                for i in range(2):
                    ax[i].scatter(
                        x = vox_embed[:, 0],
                        y = vox_embed[:, 1])
                    sns.despine(ax = ax[i])
                # Only set original limits on the left version of
                # the plot, allow right to be zoomed in
                if saved_lims is not None:
                    ax[0].set_xlim(*saved_lims[0])
                    ax[0].set_ylim(*saved_lims[1])
                plt.suptitle("Distribution of Activity | " + 
                             "Layer " + str(layer) + " | " +
                             "Voxel " + str(i_vox) + " | " +
                             "Dims [0,1]".format(d, d+1))
                pdf.savefig()
                plt.close()


def pca_fit_quality(filename, voxels, pca):
    '''
    ### Arguments
    - `filename` --- Path to a PDF file to plot to.
    - 'voxels' --- Dictionary giving a VoxelIndex object
        for each layer.
    - `pca` --- Dictionary giving a scikit-learn PCA object
        for each layer.
    '''
    with PdfPages(filename) as pdf:
        for layer, layer_vox in voxels.items():
            var = pca[layer].explained_variance_ratio_
            _, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 3),
                gridspec_kw = {'width_ratios': [4, 9]})
            ax[0].plot(var[:10], 'b-o')
            ax[0].set_yscale('log')
            ax[1].plot(var, 'b-')
            ax[1].set_yscale('log')
            plt.title("PCA Explained Variance | " + 
                          "Layer " + str(layer))
            sns.despine()
            pdf.savefig()
            plt.close()





