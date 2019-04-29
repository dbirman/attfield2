from proc import attention_models as att
from proc import network_manager as nm
from proc import voxel_selection as vx
from proc import possible_fields
from proc import backprop_fields
from proc import lsq_fields
from proc import video_gen
from proc import cornet
import plot.rfs

import itertools as iit
import pandas as pd
import numpy as np
import skvideo.io

from pprint import pprint



def get_rfs(inputs, voxels, mods, approx_n = 30):
    manager, grads = backprop_fields.gradients_raw_fullbatch(
        model, inputs, voxels, mods = mods,
        approx_n = approx_n)
    estimated = backprop_fields.parameter_estimates_grads(grads, voxels)
    params = possible_fields.estimates_to_params(estimated)
    return manager, grads, params





def gaussian_shift(pos, original, rf_params):
    '''
    Amount that rf_params is linearly shifted from original 
    towards `pos`
    '''
    ret = {}
    for layer, params in rf_params.items():
        # Distance from `pos` before attention
        dx_pre  = np.array(original[layer]).T[0]-pos[0]
        dy_pre  = np.array(original[layer]).T[1]-pos[1]
        dist_pre = np.sqrt(dx_pre**2 + dy_pre**2)
        # Distance from `pos` after attention
        dx_post = np.array(params).T[0]-pos[0]
        dy_post = np.array(params).T[1]-pos[1]
        dist_post = np.sqrt(dx_post**2 + dy_post**2)
        # Return difference in distances
        ret[layer] = dist_post - dist_pre
    return ret


def linear_shift(theta, original, rf_params):
    '''
    Amount that rf_params is linearly shifted from original in
    the direction theta (radians).
    '''
    ret = {}
    for layer, params in rf_params.items():
        # RF Shift vectors
        dx  = np.array(original[layer]).T[0]-np.array(params).T[0]
        dy  = np.array(original[layer]).T[1]-np.array(params).T[1]
        # Project shift vectors onto theta direction
        proj = np.cos(theta)*dx + np.sin(theta)*dx
        ret[layer] = proj
    return ret


def aggregate_condition(all_results, idx):
    ret = {}
    for desc in all_results.keys():
        if desc[idx] not in ret:
            ret[desc[idx]] = []
        ret[desc[idx]].append(all_results[desc])
    ret = {k: {l: np.concatenate(param_list[l], axis = 0)
               for l in param_list[0].keys()}
           for k, param_list in ret.items()}

def grad_vox_mean(grads):
    print("vox_mean():  grads.shape", grads.shape)
    return {l: np.mean(grads, axis = 0)
            for l in grads.keys()}





if __name__ == '__main__':


    # ---- Model & Inputs Setup ----

    model, ckpt = cornet.load_cornet("Z")
    layers = [(0, 0, 0), (0, 2, 0)]
    '''frame = 64
    voxels = vx.random_voxels_for_model(
        model, layers, 50, 3, frame, frame)
    positions = [(10, 30), (50, 50)]
    sigmas = [300, 1500]
    betas =  [1.1, 2.0, 3.0, 10.0]
    thetas = [0, np.pi/4]'''
    frame = 64
    voxels = vx.random_voxels_for_model(
        model, layers, 10, 3, frame, frame)
    positions = [(10, 30)]
    sigmas = [300]
    betas =  [1.1]
    thetas = [np.pi/4]
    print("Generating video")
    inputs = video_gen.rf_slider_check(frame,
        check_widths = [5, 10],
        speed = 2)


    # ---- Compute Receptive fields, etc. ----

    print("[ Unmodified ]")
    manager, raw_grads, raw_rfs = get_rfs(inputs, voxels, {})

    # Forget voxels with no gradient found
    new_voxels = {}
    for layer, layer_vox in voxels.items():
        nans = np.array(raw_rfs[layer]).T[3]
        new_voxels[layer] = vx.VoxelIndex(layer,
            [idxs[~np.isnan(nans)] for idxs in layer_vox._idx])
        raw_grads[layer] = raw_grads[layer][~np.isnan(nans)]
        raw_rfs[layer] = np.array(raw_rfs[layer])[~np.isnan(nans)]
    voxels = new_voxels

    # Compute gaussian spatial gains
    all_mods = iit.product(
        positions, sigmas, betas)
    mod_results = {}
    for pos, sigma, beta in all_mods:
        print("\n[ Modified:", pos, ",", sigma, ",", beta, "]")
        mod = att.GaussianSpatialGain(pos, sigma, beta)
        mgr, mod_grads, mod_rfs = get_rfs(inputs, voxels, {(0,): mod})
        shift = gaussian_shift(pos, raw_rfs, mod_rfs)
        mod_results[(pos, sigma, beta)] = mod_grads, mod_rfs, shift

    # Computer linear gains
    linear_results = {}
    for theta in thetas:
        for beta in betas:
            print("\n[ Modified:", theta, ",", beta, "]")
            mod = att.LinearSpatialGain(theta, beta)
            mgr, mod_grads, mod_rfs = get_rfs(inputs, voxels, {(0,): mod})
            shift = linear_shift(theta, raw_rfs, mod_rfs)
            linear_results[(theta, beta)] = mod_grads, mod_rfs, shift


    # ---- Diagnostic Data Dumps & Plots ----

    video = inputs.float()-inputs.min()
    video = video/video.max()
    video *= 255
    skvideo.io.vwrite("data/sgatt_inputs.mp4", video)







    # ---- Saving & Plotting Attention Comparisons ----

    print("\n[ Plotting ]")

    plot.rfs.grad_diff_heatmaps('plots/sgatt_grad_mass.pdf',
        mod_results, raw_grads,
        positions, sigmas, betas)

    plot.rfs.grad_heatmap(
            'plots/detail/sgatt_raw_grads.pdf',
            layers, voxels, raw_grads)


    # Gaussian Attention Effect Plotting
    for pos, sigma, beta in iit.product(positions, sigmas, betas):
        plot.rfs.motion_vectors(
            'plots/detail/sgatt_center_shifts_{}_{}_{}_{}.pdf'.format(
                *pos, sigma, beta),
            (frame, frame), att.GaussianSpatialGain(pos, sigma, beta),
            voxels, raw_rfs, mod_results[(pos, sigma, beta)][1],
            modes = [2, 1])

        plot.rfs.grad_heatmap(
            'plots/detail/sgatt_{}_{}_{}_{}_grads.pdf'.format(
                *pos, sigma, beta),
            layers, voxels, mod_results[(pos, sigma, beta)][0])

    plot.rfs.shift_by_amp(
        'plots/sgatt_linear_amps.pdf',
        amps = {l: np.concatenate([
            b * np.ones_like(mod_results[(p, s, b)][2][l])
            for p,s,b in iit.product(positions, sigmas, betas)])
         for l in layers},
        shifts = {l: np.concatenate([
            mod_results[(p, s, b)][2][l]
            for p,s,b in iit.product(positions, sigmas, betas)])
         for l in layers})




    # Linear Attention Effect Plotting
    for theta, beta in iit.product(thetas, betas):
        plot.rfs.motion_vectors(
            'plots/detail/sgatt_center_shifts_linear_{}_{}.pdf'.format(
                thetas, beta),
            (frame, frame), att.LinearSpatialGain(theta, beta),
            voxels, raw_rfs, linear_results[(theta, beta)][1],
            modes = [2, 1])

        plot.rfs.grad_heatmap(
            'plots/detail/sgatt_linear_{}_{}_grads.pdf'.format(
                thetas, beta),
            layers, voxels, linear_results[(theta, beta)][0])

    plot.rfs.shift_by_amp(
        'plots/sgatt_linear_amps.pdf',
        amps = {l: np.concatenate([
            b * np.ones_like(linear_results[(t, b)][2][l])
            for t,b in iit.product(thetas, betas)])
         for l in layers},
        shifts = {l: np.concatenate([
            linear_results[(t, b)][2][l]
            for t,b in iit.product(thetas, betas)])
         for l in layers})















