from proc import lsq_fields
import utils

import itertools as iit
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.patches as patches
import seaborn as sns
sns.set(color_codes = True)
sns.set_style('white')


def plot_multichannel(data, axes, vrange = None, cmap = cm.bone):
    if vrange is None: vmin, vmax = data.min(), data.max()
    else: vmin, vmax = vrange
    
    if vmin < 0 and vmax > 0:
        mid = 1 - (vmax/(abs(vmin - vmax)))
        cmap = utils.shiftedColorMap(cmap, midpoint = mid)
    norm = colors.Normalize(vmin = vmin, vmax = vmax)
        
    for i, ax in enumerate(axes):
        img = ax.imshow(data[i], norm = norm, cmap = cmap)
        plt.colorbar(img, ax = ax)



def single_grad_plot(grads, layers, i_layer, i_vox,
                     show = True, bbox = None):
    '''
    Plot a diagnostic of the gradients calculated

    ### Arguments
    - `grads` --- Output of `gradients_raw()`
    - `layers` --- A list of `LayerIndex`es giving the layers of the
        model that voxels were pulled from
    - `i_layer` --- Index in the `layers` list giving layer we should
        plot for voxels from
    - `i_vox` --- Index in the voxels list passed to `gradients_raw()`
        giving the voxel whose gradients are to be plotted.
    - `show` --- Either `True`, in which case the plots will be shown
        in a GUI window, or a function taking an optional parameter `i`
        giving the batch number that is called as a replacement for
        matplotlib's `show` method.
    - `bbox` --- A tuple of the form (left, top, width, height) giving
        the coordinates and size of a black-stroked rectangle to be 
        drawn over the gradient heatmap or `None` for no rectangle.
    '''
    
    if len(grads[layers[i_layer]].shape) == 5:
        full_grad = np.mean(grads[layers[i_layer]][:, i_vox], axis = 0)
    else:
        full_grad = grads[layers[i_layer]][i_vox]
    fig, axes = plt.subplots(ncols = 3, figsize = (14, 4))
    plot_multichannel(full_grad, axes, cmap = cm.PuOr_r)
    if bbox is not None:
        for ax in axes:
            rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],
                linewidth=1,edgecolor='k',facecolor='none')
            ax.add_patch(rect)
    plt.suptitle('Full Gradient | ' + 
                 'Layer: {} | '.format(layers[i_layer]) +
                 'Voxel = {}'.format(i_vox))
    if show == True: plt.show()
    else: show()


def grad_heatmap(filename, layers, voxels, grads, bboxes = None):
    with PdfPages(filename) as pdf:
        showfunc = lambda *_: (pdf.savefig(), plt.close())
        for i_layer, l in enumerate(layers):
            for i_vox in range(voxels[l].nvox()):
                if bboxes:
                    bb = tuple(bboxes[l][k][i_vox] for k in range(4))
                    single_grad_plot(grads, layers, i_layer, i_vox,
                                 show = showfunc, bbox = bb)
                else:
                    single_grad_plot(grads, layers, i_layer, i_vox,
                                 show = showfunc)




def single_gaussian_rfs_refinement(layers, image_size, estimates,
                                   params, i_layer, i_vox,
                                   show = True):
    w, h = image_size
    gt_params = estimates[layers[i_layer]]
    fitted_params = params[layers[i_layer]][i_vox:i_vox+1].T
    gt = lsq_fields.gauss_with_params_torch(w, h, *gt_params)
    gt = gt.numpy()[i_vox]
    fitted = lsq_fields.gauss_with_params_torch(w, h, *fitted_params)
    fitted = fitted.numpy()[0]

    fig, axes = plt.subplots(ncols = 2, figsize = (14, 4))
    plot_multichannel(np.array([gt, fitted]), axes)
    axes[0].set_title("Initial Estimate")
    axes[1].set_title("Fitted")
    plt.suptitle("Layer: {} | ".format(layers[i_layer]) + 
                 "Voxel: {}".format(i_vox))

    if show == True: plt.show()
    else: show(i_layer, i_vox)


def gaussian_rfs_refinement(filename, layers, voxels, image_size,
                            estimates, params):
    plot_pdf = PdfPages(filename)
    with plot_pdf as pdf:
        showfunc = lambda *_: (pdf.savefig(), plt.close())
        for i_layer, l in enumerate(layers):
            for i_vox in range(voxels[l].nvox()):
                single_gaussian_rfs_refinement(
                    layers, image_size, estimates, params,
                    i_layer, i_vox, show = showfunc)



def motion_vectors(filename, input_shape, mod,
                   voxels, raw_rfs, mod_rfs,
                   pdf = None, modes = [2,1]):
    '''
    ### Arguments
    - `input_shape` --- A tuple of ints (height, width)
    - `mod` --- A GaussianSpatialGain layer mod
    - `modes` --- Voxel plotting modes: 1 means individual
        voxels, while 2 means all voxels. A list can be passed
        and both modes will be performed in the given order.
    '''

    # Set up some initial variables before iteratino
    if mod is not None:
        mod = mod.scale_array((1, 3,) + tuple(input_shape))
    cmap = utils.shiftedColorMap(cm.Greens, midpoint = 0.35, stop = 0.7)

    with PdfPages(filename) as pdf:
        for layer in voxels.keys():
            for mode in modes:

                # Pull out positions and shifts from parameter arrays
                curr_raw = np.array(raw_rfs[layer]).T
                curr_mod = np.array(mod_rfs[layer]).T
                xs = curr_raw[0]
                ys = curr_raw[1]
                nans = np.isnan(curr_mod[3] * curr_raw[3])
                dxs = curr_mod[0] - xs
                dys = curr_mod[1] - ys
                
                to_zip = xs, ys, dxs, dys, nans, curr_mod[3], curr_raw[3]
                for i, (x,y,dx,dy,is_nan,J,K) in enumerate(zip(*to_zip)):
                    if mod is not None:
                        plt.imshow(np.squeeze(mod[0]), cmap = cmap)
                    else:
                        plt.xlim(0, input_shape[0])
                        plt.ylim(input_shape[1], 0)
                        plt.gca().set_aspect(1)
                
                    if not is_nan:
                        plt.plot([x, x+dx], [y, y+dy], 'k-')
                        plt.plot([x+dx], [y+dy], 'ks')

                    if mode == 1:
                        plt.title("RF Center Shifts" + 
                            " | Layer: " + str(layer) + 
                            " | Voxel: " + str(i))
                        pdf.savefig()
                        plt.close()
                if mode == 2:
                    plt.title("RF Center Shifts | Layer: " + str(layer))
                    pdf.savefig()
                    plt.close()
    return pdf
                


def activation_trials(filename, voxels, true_acts, pred_acts):
    with PdfPages(filename) as pdf:
        for layer, layer_vox in voxels.items():
            for i_vox in range(layer_vox.nvox()):
                plt.figure(figsize = (15, 3))
                plt.plot(pred_acts[layer][:, i_vox],
                         color = 'orange', ls = '-',
                         label = "Predicted", lw = 1)
                plt.plot(true_acts[layer][:, i_vox], 'b-',
                         label = "True", lw = 1)
                plt.legend()
                plt.title("Activation Timecourses | " + 
                          "Layer " + str(layer) +
                          " | Voxel " + str(i_vox))
                pdf.savefig()
                plt.close()


def grad_mass_diff(grads_1, grads_2):
    return {l: abs(grads_1[l]) - abs(grads_2[l])
            for l in grads_1.keys()}


def grad_diff_heatmap_by_pos(filename, mod_results, raw_grads,
                             positions, sigmas, betas):
    with PdfPages(filename) as pdf:
        for pos in positions:
            # all_diffs.shape: (sigma * beta * vox, channel, row, col)
            all_diffs = np.concatenate([
                grad_mass_diff(mod_results[(pos, sigma, beta)][0], raw_grads)
                for sigma, beta in iit.product(sigmas, betas)])
            diffs = all_diffs.mean(axis = 0)
            
            fig, axes = plt.subplots(ncols = 3, figsize = (14, 4))
            plot_multichannel(diffs, axes, cmap = cm.PuOr_r)
            plt.title("Gradient Shift | Attention at X={},Y={}".format(*pos))
            pdf.savefig()
            plt.close()

def grad_diff_heatmaps(filename, mod_results, raw_grads,
                             positions, sigmas, betas):
    with PdfPages(filename) as pdf:
        for layer in raw_grads.keys():
            for pos, sigma, beta in iit.product(positions, sigmas, betas):
                curr_grad = mod_results[(pos, sigma, beta)][0]
                diffs = grad_mass_diff(curr_grad, raw_grads)
                curr_diffs = diffs[layer].mean(axis = 0)
                
                fig, axes = plt.subplots(ncols = 3, figsize = (14, 4))
                plot_multichannel(curr_diffs, axes, cmap = cm.PuOr_r)
                plt.suptitle("Gradient Shift | Attention: "+
                          "X={},Y={},S={},B={}".format(*pos, sigma, beta))
                pdf.savefig()
                plt.close()


def shift_by_amp(filename, amps, shifts):
    '''
    ### Arguments
    - `amps` --- A dictionary mapping layers to  arrays of attentional
        amplitudes (usually the parameter beta).
    - `shifts` --- A dictionary mapping layers to arrays of attentional
        shift magnitudes.
    '''
    with PdfPages(filename) as pdf:
        for layer in amps.keys():
            ax = sns.violinplot(x = amps[layer], y = shifts[layer],
                inner = 'point')
            plt.title("Shift by Attenion Amplitude " + 
                      " | Layer: " + str(layer))
            plt.ylabel("Shift (Pixels)")
            plt.xlabel("Attenional Amplitude")
            ax.axhline(0, 0, 1, ls = '--', color = "#aaaaaa", lw = 1)
            pdf.savefig()
            plt.close()








