from proc import voxel_selection as vx
from proc import network_manager as nm
from proc import video_gen
from proc import cornet

from pprint import pprint
import numpy as np
import torch
import tqdm
import time
import sys



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
                sorted_idxs = np.argsort(approx_voxel)
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


def weighted_modes(X, weights):
    weights_sum = np.sum(weights, axis = 1)
    if np.any(weights_sum == 0):
        for i in np.where(weights_sum == 0)[0]:
            weights[i] = 1./weights.shape[1]
        weights_sum = np.sum(weights, axis = 1)
    expectation = np.sum(X[np.nax, :] * weights, axis = 1)
    mean = expectation / weights_sum
    residuals = (X[np.nax, :] - mean[:, np.nax])**2
    expectation_sq = np.sum(residuals * weights, axis = 1)
    var = expectation_sq / weights_sum
    return mean, var


def parameter_estimates_grads(grads, voxels):
    """
    Estimate parameters of gaussian receptive field based
    on backpropagated gradients. Grads should be output
    of `backprop_fields.gradients_raw_fullbatch`.
    """

    params = {}
    for layer, layer_vox in voxels.items():
        # Get gradients averaged across color dimension
        curr_grads = np.mean(grads[layer], axis = 1)

        # Perform weighted averages to calculate 1st and 2nd modes
        # These give us the mean and varance in X and Y
        # (treating X and Y as independent)
        mean_X, var_X = weighted_modes(
            np.arange(curr_grads.shape[1]),
            abs(np.mean(curr_grads, axis = 1)))
        mean_Y, var_Y = weighted_modes(
            np.arange(curr_grads.shape[2]),
            abs(np.mean(curr_grads, axis = 2)))

        var = (var_X+var_Y)/2
        var[var < 1e-2] = 1 
        beta = np.max(curr_grads, axis = (1,2))
        beta[abs(beta) < 1e-2] = np.nan

        params[layer] = (mean_X, mean_Y, var, beta)

    return params




if __name__ == '__main__':
    model, ckpt = cornet.load_cornet("Z")
    all_idxs = vx.LayerIndex.all_model_idxs(model)
    #pprint(all_idxs)
    layers = [vx.LayerIndex(1, 1)]
    voxels = vx.random_voxels_for_model(
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


