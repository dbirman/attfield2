from proc import voxel_selection as vx
from proc import network_manager as nm
from proc import backprop_fields as bf
from proc import possible_fields
from proc import tuning_curves
from proc import lsq_fields
from proc import video_gen
from proc import cornet
import plot.rfs
import plot.tuning

import scipy.ndimage.interpolation as ndint
from scipy.stats import truncnorm
import numpy as np
import skvideo.io
import torch

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

def brownian_limited(n, limit):
    ret = [0]
    for i in range(n-1):
        rand = np.random.randint(3)-1
        ret.append(max(-limit, min(limit, ret[-1] + rand)))
    return np.array(ret)


def modulate_inputs(inp, max_spatial = 5, noise_sd = 1):
    '''Apply permutations to the input to make tuning curve
    measurements sensitive to invariances.
    
    ### Returns
    - `arr` --- The framewise-permuted version of `inp`. This
    will have a padding of `2*max_spatial` pixels removed in 
    both spatial dimensions.'''

    x_shifts = brownian_limited(len(inp), max_spatial)
    y_shifts = brownian_limited(len(inp), max_spatial)
    shifted = torch.tensor([
        ndint.shift(img, (0, x_shifts[i], y_shifts[i]))
        for i, img in enumerate(inp)])
    noise = truncnorm.rvs(0, 2, loc = 1, scale = noise_sd,
        size = (len(shifted), 1, *shifted.shape[2:]))
    noisy = shifted * torch.tensor(noise, dtype = torch.float)
    spatial_slice = slice(max_spatial, -max_spatial)
    cropped = noisy[:, :, spatial_slice, spatial_slice]
    return cropped




if __name__ == '__main__':

    # ---- Model & Inputs Setup ----

    model, ckpt = cornet.load_cornet("Z")
    layers = [(0, 0, 0), (0, 2, 0)]
    FRAME = 64
    REPEAT = 10
    MAX_SPACE = 10
    voxels = vx.random_voxels_for_model(
        model, layers, 100, 3, FRAME, FRAME)
    print("Generating video")
    inputs = {
        'obj': video_gen.mcgill_images(FRAME+2*MAX_SPACE, 32),
    }
    groups = {
        'obj': video_gen.mcgill_groups(32),
    }



    # ---- Compute Activation & Receptive fields ----

    acts = {}
    grads = {}
    bboxes = {}
    pca = {}
    embedded = {}
    for k, inp in inputs.items():

        print("Activations: {}".format(k))
        acts[k] = []
        for i in range(REPEAT):
            inp_mod = modulate_inputs(inp)
            manager = nm.NetworkManager.assemble(model, inp_mod)
            true_acts, _ = lsq_fields.voxel_activity(
                voxels, manager, inp_mod.numpy(), None)
            acts[k].append(true_acts)

        print("PCA: {}".format(k))
        pca[k], embedded[k] = tuning_curves.tuning_pca(
            voxels, acts[k])

        print("Gradients: {}".format(k))
        manager = nm.NetworkManager.assemble(model, inp)
        _, curr_grads = bf.gradients_raw_fullbatch(
            model, inp, voxels,
            approx_n = 30)
        grads[k] = curr_grads

        bboxes[k] = possible_fields.rf_bboxes(voxels, manager)



    # ---- Saving and Plotting for Each Input Type ----

    for k, result in acts.items():

        print("Saving and Plotting: {}".format(k))


        # ---- Classic Tuning Curve Plots ----

        inp = np.transpose(inputs[k], [0, 2, 3, 1])
        vrange = inp.max()-inp.min()
        skvideo.io.vwrite(
            'data/tune_inputs_{}.mp4'.format(k),
            255*(inp-inp.min())/vrange)

        #custom_x = plot.tuning.cts_image_samples_x(inputs[k], 8)
        plot.tuning.curve1d(
            'plots/tune_{}.pdf'.format(k),
            voxels, acts[k], groups[k],
            mode = 'cat')



        # ---- PCA Tuning Curve Plots ----

        plot.tuning.pca_fit_quality(
            'plots/tune_fitquality_{}.pdf'.format(k),
            voxels, pca[k])
        plot.tuning.pca_cts(
            'plots/tune_pca_{}.pdf'.format(k),
            voxels, acts[k], embedded[k], pca[k])



        # ---- Diagnostic & Routine Plots ----

        plot.rfs.grad_heatmap('plots/tune_{}_grads.pdf'.format(k),
           layers, voxels, grads[k], bboxes = bboxes[k])






