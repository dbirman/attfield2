"""
Methods for finding receptive fields of CNN voxels via least-suqares
approximation of model parameters. (The only receptive-field model
currently implemented is a gaussian). While this is far from being the
most accurate or fastest method of finding CNN voxels, it matches the
processes performed in animal/human psysiological experiments.
"""

from proc import voxel_selection as vx
from proc import backprop_fields as bf
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

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

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


def parameter_estimates(grads, voxels):
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
        pred_rfs = gauss_with_params_torch(w, h, *params.float())
        # inputs_bw.shape: (n_frames, h, w)
        # pred_rfs.shape:  (n_vox, h, w)
        # pred_acts.shape: (n_vox, n_frames)
        pred_acts = pred_rfs[np.nax, ...] * inputs_bw[:, np.nax, ...]
        pred_acts = torch.sum(pred_acts, dim = (2, 3))
        
        # targets.size(): (n_vox, n_frames)
        residuals = ((pred_acts - targets.detach())**2).mean()
        # store memoized computation for backprop
        rf_obj.cache[tuple(x)] = (params, residuals)
        bypass_ret.append((params, residuals))
        return residuals
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




def plot_rfs(inputs, estimates, params, i_layer, i_vox, show = True):
    w = inputs.shape[3]
    h = inputs.shape[2]
    gt_params = estimates[layers[i_layer]]
    fitted_params = params[layers[i_layer]][i_vox:i_vox+1].T
    gt = gauss_with_params_torch(w, h, *gt_params).numpy()[i_vox]
    fitted = gauss_with_params_torch(w, h, *fitted_params).numpy()[0]

    fig, axes = plt.subplots(ncols = 2, figsize = (14, 4))
    bf.plot_multichannel(np.array([gt, fitted]), axes)
    axes[0].set_title("Initial Estimate")
    axes[1].set_title("Fitted")
    plt.suptitle("Layer: {} | ".format(layers[i_layer]) + 
                 "Voxel: {}".format(i_vox))

    if show == True: plt.show()
    else: show(i_layer, i_vox)


def lsq_rfs(model, layers, channels = 3, framesize = 246,
            percent = 0.00001, grad_n = 10, bar_speed = 1.,
            output_dir = '.', verbose = 0,
            grad_mode = 'both', voxels = None, mods = {}):
    if voxels is None:
        print("Random voxels")
        voxels = bf.random_voxels_for_model(
            model, layers, percent, channels, framesize, framesize)

    print("Video gen")
    if grad_mode == 'slider':
        inputs = video_gen.rf_slider(framesize, speed = bar_speed)*255
    elif grad_mode == 'check':
        inputs = video_gen.rf_slider_check(framesize, speed = bar_speed)
    elif os.path.isfile(grad_mode):
        inputs = skvideo.io.vread(grad_mode)
    else:
        inputs = [video_gen.rf_slider(framesize, speed = bar_speed)*255,
                  video_gen.rf_slider_check(framesize, speed = bar_speed)]
        inputs = np.concatenate(inputs)
    inputs = np.tile(inputs[:, np.nax, :, :], [1, channels, 1, 1])
    inputs = torch.from_numpy(inputs).float()
    inputs = bf.batch_transform(inputs, bf.normalize)

    print("Gradients")
    manager, grads = bf.gradients_raw_fullbatch(
        model, inputs, voxels,
        approx_n = grad_n, mods = mods)
    estimates = parameter_estimates(grads, voxels)

    '''with PdfPages(os.path.join(output_dir, "grads.pdf")) as pdf:
        showfunc = lambda *_: (pdf.savefig(), plt.close())
        for i_layer, l in enumerate(layers):
            for i_vox in range(voxels[l].nvox()):
                bf.grad_plot(grads, layers, i_layer, i_vox,
                             show = showfunc)'''

    # Copmute RFs based on previous estimates and least squares
    print("Fitting parameters")
    params, hist = fit_rfs(inputs.numpy(), manager, voxels, estimates)

    '''plot_pdf = (PdfPages(os.path.join(output_dir, "rfs.pdf"))
                if output_dir else contextlib.nullcontext()) 
    with plot_pdf as pdf:
        showfunc = (lambda *_: (pdf.savefig(), plt.close()) 
                    if output_dir else lambda: None)
        for i_layer, l in enumerate(layers):
            for i_vox in range(voxels[l].nvox()):
                plot_rfs(inputs, estimates, params, i_layer, i_vox,
                         show = showfunc)'''

    return voxels, estimates, params


def save_rf_csv(filename, voxels, params):
    dfs = []
    for layer, layer_vox in voxels.items():
        layer_str = '.'.join([str(i) for i in layer])
        voxel_strs = [layer_str + ';' + v for v in layer_vox.serialize()]

        params_array = np.array(params[layer]).T
        dfs.append(pd.DataFrame({
            'unit':     voxel_strs,
            'orig_x':   params_array[0],
            'orig_y':   params_array[1],
            'orig_var': params_array[2],
            'orig_amp': params_array[3],
        }))
    full = pd.concat(dfs)
    full.to_csv(filename)



if __name__ == "__main__":

    model, ckpt = cornet.load_cornet("Z")
    layers = [(0, 1)]
    lsq_rfs(model, layers, verbose = 1,
        bar_speed = 12., percent = 4,
        grad_n = 10, grad_mode = 'slider')



    if False:
        # Not updated after code-breaking commit ec72c1f
        import pickle as pkl
        loaded_data = pkl.load(open('grads.pkl', 'rb'))
        grads = loaded_data['grads']
        voxels = loaded_data['voxels']
        layers = loaded_data['layers']
        noise_vid = loaded_data['noise_video']

        model, ckpt = cornet.load_cornet("Z")
        computed = model(torch.tensor(noise_vid))

        estimates = parameter_estimates(grads, voxels)
        params, errs = fit_rfs(noise_vid, computed, voxels, estimates)

        plot_rfs(noise_vid, estimates, params, 0, 5)
        if False:
            pkl.dump({
                'grads': grads,
                'voxels': voxels,
                'layers': layers,
                'noise_video': noise_vid,
                'estimates': estimates,
                'params': params,
                'errs': errs
            }, open('params.pkl', 'wb'))




