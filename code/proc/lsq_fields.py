"""
Methods for finding receptive fields of CNN voxels via least-suqares
approximation of model parameters. (The only receptive-field model
currently implemented is a gaussian). While this is far from being the
most accurate or fastest method of finding CNN voxels, it matches the
processes performed in animal/human psysiological experiments.
"""

from proc import voxel_selection as vx
from proc import backprop_fields as bf
from proc import possible_fields
from proc import video_gen
from proc import cornet

from scipy import optimize
import pandas as pd
import numpy as np
import scipy.stats
import contextlib
import torch
import tqdm
import sys
import os

np.nax = np.newaxis


def gauss_with_params_numpy(w, h, mean_X, mean_Y, var, beta):
    """
    ### Arguments
    - `w` --- Width of the image (number of columns)
    - `h` --- Height of the image (number of rows)
    - `mean_X` --- 1D array, where the i-th entry gives the X (column)
        value of the mean vector for the gaussian in the i-th image
    - `mean_Y` --- 1D array, where the i-th entry gives the Y (row)
        value of the mean vector for the gaussian in the i-th image
    - `var` --- 1D array, where the i-th entry gives the variance
        for the gaussian in the i-th image
    - `beta` --- 1D array, where the i-th entry gives the magnitude
        for the gaussian in the i-th image

    ### Returns
    - `arr` --- Numpy array containing gaussian, shape: (nimg, h, w)
    """

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    var_reshaped = 2*np.sqrt(var)[np.nax, np.nax, :]
    X_normed = (X[:, :, np.nax]-mean_X[np.nax, np.nax, :])
    Y_normed = (Y[:, :, np.nax]-mean_Y[np.nax, np.nax, :])
    z = np.exp(-X_normed**2/var_reshaped - Y_normed**2/var_reshaped)
    return beta[:, np.nax, np.nax] * z.transpose(2, 0, 1)


def gauss_with_params_torch(w, h, mean_X, mean_Y, var, beta):
    imgs = []
    x_cord = torch.arange(w).float()
    x_grid = x_cord.repeat(h).view(h, w)
    y_cord = torch.arange(h).float()
    y_grid = y_cord.repeat(w).view(w, h).t()
    for i in range(len(mean_X)):
        xy_grid = torch.stack([x_grid-mean_X[i], y_grid-mean_Y[i]], dim=-1)
        imgs.append(beta[i] * torch.exp(
              -torch.sum(xy_grid**2., dim=-1) / (2*var[i])))
    return torch.stack(imgs)



def apply_gaussian_rf(inputs, params):
    '''
    inputs.shape: (n_frames, h, w)
    params shape: (n_params, n_vox)
    '''
    w = inputs.shape[2]
    h = inputs.shape[1]
    pred_rfs = gauss_with_params_torch(w, h, *params)
    # pred_rfs.shape:  (n_vox, h, w)
    # pred_acts.shape: (n_vox, n_frames)
    pred_acts = pred_rfs[np.nax, ...] * inputs[:, np.nax, ...]
    pred_acts = torch.sum(pred_acts, dim = (2, 3))

    mean_const = pred_acts.detach().numpy()
    mean_const = mean_const.mean(axis = 0, keepdims = True)
    mean_const = torch.tensor(mean_const)
    return pred_acts - mean_const



def rf_obj(inputs, targets):
    w = inputs.shape[3]
    h = inputs.shape[2]
    inputs_bw = torch.tensor(np.mean(inputs, axis = 1)).float()

    def f_obj(x, bypass_ret = []):
        # x should be a flattened version of a matrix with
        # shape (n_vox, n_params)
        # params.shape : (n_params, nvox)
        params = x.reshape(targets.shape[1], -1).T
        params = torch.tensor(params, requires_grad = True)
        pred_acts = apply_gaussian_rf(inputs_bw, params.float())
        
        # targets.size(): (n_vox, n_frames)
        residuals = ((pred_acts - targets.detach())**2).mean()
        # store memoized computation for backprop
        rf_obj.cache[tuple(x)] = (params, residuals)
        bypass_ret.append((params, residuals))
        return residuals.detach().numpy()
    return f_obj

rf_obj.cache = {}


def rf_jac(inputs, targets):
    f_obj = rf_obj(inputs, targets)
    def f_jac(x):
        # x should be a flattened version of a matrix with
        # shape (n_vox, n_param)
        if tuple(x) in rf_obj.cache:
            params, residuals = rf_obj.cache[tuple(x)]
            rf_obj.cache.pop(tuple(x))
        else:
            obj_ret = []
            f_obj(x, bypass_ret = obj_ret)
            params, residuals = obj_ret[0]

        residuals.backward(retain_graph = True)
        ret = params.grad.numpy()
        del params, residuals
        return ret.ravel()
    return f_jac

def minimize_callback(hist_dict, f_obj):
    def cb(x, *_):
        hist_dict['x'].append(x)
        hist_dict['f'].append(f_obj(x))
        hist_dict['i'].append(len(hist_dict['x'])+1)

        if len(hist_dict['f']) > 1:
            print("\r{:>5d} | {:>10.2e} | {:>10.2e}".format(
                hist_dict['i'][-1], hist_dict['f'][-1],
                hist_dict['f'][-1]-hist_dict['f'][-2]),
                end = "")
        else:
            print("\r{:>5d} | {:>10.2e} | ".format(
                hist_dict['i'][-1], hist_dict['f'][-1]),
                end = "")
    return cb


def fit_rfs(inputs, manager, voxels, estimates, verbose = 0):

    w = inputs.shape[3]
    h = inputs.shape[2]
    params_hist = {}
    params = {}
    for i_layer, (layer, layer_vox) in enumerate(voxels.items()):
        print("Fitting for layer:", layer,
              " (", i_layer+1, "/", len(voxels), ")")

        params_hist[layer] = {'x': [], 'f': [], 'i': []}
        targets = layer_vox.index_into_computed(manager)
        f_obj = rf_obj(inputs, targets)
        f_jac = rf_jac(inputs, targets)
        callback = minimize_callback(params_hist[layer], f_obj)
        # estimates[layer].shape: (n_param, n_vox)
        # so transpose and flatten arranges in format
        # [meanX_1, meanY_1, var_1, beta_1, meanX_2, ...]
        p0 = np.array(estimates[layer]).T.ravel()
        # Tile out min and max vectors to match
        mins = np.tile([0, 0, 1e-16,   -np.inf], [layer_vox.nvox()])
        maxs = np.tile([w, h, max(w,h), np.inf], [layer_vox.nvox()])
        result = optimize.minimize(f_obj, p0,
            method = "L-BFGS-B",
            bounds = tuple(zip(mins, maxs)),
            jac = f_jac,
            callback = callback,
            options = {'maxfun': 400})
        params[layer] = result.x.reshape(layer_vox.nvox(), -1)

    return params, params_hist


def voxel_activity(voxels, manager, inputs, params):
    '''Pull voxel activity from a model and apply an RF to inputs.
    ### Arguments
    - `inputs` --- Network inputs (numpy array, not torch tensor)
        that should be of shape: (batch, channel, row, col)
    - `params` --- Params should be in their standard form of
        shape (n_vox, n_param)
    ### Returns
    - `true_acts` --- Activations as pulled from the model.
        Shape is (batch, n_vox)
    - `pred_acts` --- Activations as predicted by the gaussian RF.
        Also of shape (batch, n_vox)
    '''
    inputs_bw = torch.tensor(np.mean(inputs, axis = 1)).float()
    true_acts_ret = {}
    pred_acts_ret = {}
    for layer, layer_vox in voxels.items():
        true_acts = layer_vox.index_into_computed(manager)
        true_acts_ret[layer] = true_acts.detach().numpy()
        # apply_gaussian_rf expects params shape: (n_param, n_vox)
        if params is not None:
            params_T = np.array(params[layer]).T
            pred_acts = apply_gaussian_rf(inputs_bw, params_T)
            pred_acts_ret[layer] = pred_acts.detach().numpy()
        else:
            pred_acts_ret[layer] = np.full(true_acts.shape, np.nan)
    return true_acts_ret, pred_acts_ret


def save_activity_csv(filename, voxels, true_acts, pred_acts):
    # true_acts.shape, pred_acts.shape: (batch, n_vox)
    dfs = []
    voxel_strs = vx.VoxelIndex.serialize(layer_vox)
    for layer, layer_vox in voxels.items():

        nframe = len(true_acts[layer])
        voxel_strs_layer = np.tile(voxel_strs[layer][np.nax, :], [nframe, 1])
        frame_idxs = np.tile(np.arange(nframe)[:, np.nax],
                             [1, layer_vox.nvox()])

        dfs.append(pd.DataFrame({
            'unit': voxel_strs_layer.ravel(),
            'frame': frame_idxs.ravel(),
            'true': true_acts[layer].ravel(),
            'pred': pred_acts[layer].ravel(), 
        }))

    full = pd.concat(dfs)
    full.to_csv(filename)


def load_activity_csv(filename):
    df = pd.read_csv(filename)
    voxels = vx.VoxelIndex.from_serial(df['unit'].unique())

    true_acts = {}
    pred_acts = {}
    layer_strs = df['unit'].apply(lambda s: s.split(';')[0])
    for l_str, group in df.groupby(layer_strs):
        layer = tuple(int(i) for i in l_str.split('.'))
        to_shape = [-1, voxels[layer].nvox()]
        true_acts[layer] = group['true'].values.reshape(to_shape)
        pred_acts[layer] = group['pred'].values.reshape(to_shape)
    return voxels, true_acts, pred_acts



def save_rf_csv(filename, voxels, params):
    dfs = []
    voxel_strs = vx.VoxelIndex.serialize(voxels)
    for layer, layer_vox in voxels.items():
        params_array = np.array(params[layer]).T
        dfs.append(pd.DataFrame({
            'unit':     voxel_strs[layer],
            'x':   params_array[0],
            'y':   params_array[1],
            'var': params_array[2],
            'amp': params_array[3],
        }))
    full = pd.concat(dfs)
    if filename is not None:
        full.to_csv(filename)
    return full


def load_rf_csv(filename):
    df = pd.read_csv(filename)
    voxels = vx.VoxelIndex.from_serial(df['unit'])

    params = {}
    layer_strs = df['unit'].apply(lambda s: s.split(';')[0])
    for l_str, group in df.groupby(layer_strs):
        layer = tuple(int(i) for i in l_str.split('.'))
        # Params[layer] is expected to have shape (n_vox, 4)
        params[layer] = group[['x', 'y', 'var', 'amp']].values

    return voxels, params


if __name__ == "__main__":

    model, ckpt = cornet.load_cornet("Z")
    layers = [(0, 3, 3)]
    lsq_rfs(model, layers, verbose = 1,
        bar_speed = 4., percent = 30,
        grad_n = 20, grad_mode = 'check')




