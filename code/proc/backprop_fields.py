from proc import voxel_selection as vx
from proc import network_manager as nm
from proc import video_gen
from proc import cornet
import utils

from torchvision import transforms
from pprint import pprint
import numpy as np
import torch
import tqdm
import time
import sys

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import seaborn as sns
sns.set_style('white')


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

def batch_transform(batch, transform):
    newbatch = [transform(batch[i]) for i in range(len(batch))]
    return torch.stack(newbatch)

def noise_video(frames, channels, width, height, numpy = True):
    inputs = np.random.randint(0, 255,
        size = (frames, channels, width, height))
    inputs = torch.from_numpy(inputs).float()
    inputs = batch_transform(inputs, normalize)
    if numpy: inputs = inputs.numpy()
    return inputs



def gradients_raw_fullbatch(model, inputs, voxels, approx_n = 20, mods = {}):

    # Even if we're running the approximation and computing subsets
    # of the frames, we need to run this so we can return the computed
    # as expected
    inputs = torch.tensor(inputs.detach().numpy(), requires_grad = True)
    manager = nm.NetworkManager.assemble(model, inputs, mods = mods)

    grads = {}
    for layer, layer_vox in voxels.items():
        grads[layer] = np.zeros(
            (layer_vox.nvox(),) + inputs.shape[1:],
            dtype = np.float32)
        mod = manager.modules[layer]
        tens = manager.computed[layer]
        print("Gradients for layer:", layer)

        masks_iter = layer_vox.backward_masks_fullbatch(manager)
        for i, (i_vox, vox_idx, mask) in enumerate(masks_iter):
            sys.stdout.write("\rVoxel: " + str(i_vox+1) + 
                             " / " + str(layer_vox.nvox()))
            sys.stdout.flush()

            if approx_n is not None:
                # Approximate region for this voxel based on the row/col
                # position of the voxel in the layer. Uses the ratio of
                # voxel index (vox_idx) to this layer's size (mask.size)
                # and then applies that ratio to the original image
                # (inputs.size()) selecting a 3-voxel window 
                col_idx = int(inputs.size()[-1] * 
                              (vox_idx[-1] / mask.size()[-1]))
                col_slice = slice(max(0, col_idx - 3), col_idx + 3)
                row_idx = int(inputs.size()[-2] *
                              (vox_idx[-2] / mask.size()[-2]))
                row_slice = slice(max(0, row_idx - 1), row_idx + 1)

                # Find a good range of frames from the input video.
                # Sorts by average pixel intensity around the approximate
                # receptive field center and selects evenly spaced frames
                # fram that sorted list
                approx_voxel = inputs.detach().numpy()
                approx_voxel = approx_voxel[:, :, col_slice, row_slice]
                approx_voxel = np.mean(approx_voxel, axis = (1,2,3))
                sorted_idxs = np.argsort(np.random.permutation(approx_voxel))
                frame_idxs = np.linspace(0, len(sorted_idxs)-1, approx_n)
                frame_idxs = np.floor(np.unique(frame_idxs)).astype('uint32')
                frames = sorted_idxs[frame_idxs]
                
                mgr_subset = nm.NetworkManager.assemble(model, inputs[frames],
                    mods = mods)
                curr_tens = mgr_subset.computed[layer]

            else:
                curr_tens = tens

            model.zero_grad()
            inputs.grad = None
            curr_tens.backward(mask[frames], retain_graph = True)
            grads[layer][i_vox, ...] = torch.sum(inputs.grad, dim = (0,))
        print()

    return manager, grads


def random_voxels_for_model(model, layers, percent,
                            channels, width, height):
    inputs = inputs = noise_video(11, channels, height, width, numpy = False)
    mgr = nm.NetworkManager.assemble(model, inputs)
    return vx.random_voxels(mgr, layers, percent)


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


def grad_plot(grads, layers, i_layer, i_vox, mode = 'avg', show = True):
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
    - `mode` --- Can be `'avg'`, in which case the gradients will be
        averaged across the batch dimension, or `batch`, in which
        case a different plot will be given for each batch.
    - `show` --- Either `True`, in which case the plots will be shown
        in a GUI window, or a function taking an optional parameter `i`
        giving the batch number that is called as a replacement for
        matplotlib's `show` method.
    
    ### Raises

    - `ValueError` --- If an invalid `mode` was passed.
    '''
    
    if mode == 'avg':
        if len(grads[layers[i_layer]].shape) == 5:
            full_grad = np.mean(grads[layers[i_layer]][:, i_vox], axis = 0)
        else:
            full_grad = grads[layers[i_layer]][i_vox]
        fig, axes = plt.subplots(ncols = 3, figsize = (14, 4))
        plot_multichannel(full_grad, axes, cmap = cm.PuOr_r)
        plt.suptitle('Full Gradient | ' + 
                     'Layer: {} | '.format(layers[i_layer]) +
                     'Voxel = {}'.format(i_vox))
        if show == True: plt.show()
        else: show()
    elif mode == 'batch':
        for ib in range(len(grads[layers[i_layer]])):
            fig, axes = plt.subplots(ncols = 3, figsize = (10, 3))
            batch_grads = grads[layers[i_layer]][ib, i_vox]
            plot_multichannel(batch_grads, axes, cmap = cm.PuOr_r)
            plt.suptitle('Layer: {}'.format(layers[i_layer]) + 
                      ' | Voxel: {}'.format(i_vox) + 
                      ' | Batch: {}'.format(ib))
            if show == True: plt.show()
            else: show(ib)
    else:
        raise ValueError("Unknown mode " + str(mode) +
            ". Use either 'avg' or 'batch.")



if __name__ == '__main__':
    model, ckpt = cornet.load_cornet("Z")
    all_idxs = vx.LayerIndex.all_model_idxs(model)
    #pprint(all_idxs)
    layers = [vx.LayerIndex(1, 1)]
    voxels = random_voxels_for_model(
        model, layers, 0.00001, 3, 246, 246)
    noise_vid, computed, grads = gradients_raw(model, voxels)
    grad_plot(grads, layers, 0, 1)

    if False:
        import pickle as pkl
        pkl.dump({
            'grads': grads,
            'voxels': voxels,
            'layers': layers,
            'noise_video': noise_vid
        }, open('grads.pkl', 'wb'))


