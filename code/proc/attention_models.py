from proc import network_manager as nm
from proc import detection_task as det
from proc import deformed_conv as dfc
from proc import lsq_fields

import importlib.util
from torch import nn
import numpy as np
import torch
import json
import os



def load_model(filename, **kwargs):
    """
    Load the global variable `model` from filename, the implication being that
    the object is a LayerMod.
    """
    # Run the model file
    fname_hash = hash(filename)
    spec = importlib.util.spec_from_file_location(
        f"model_file_{fname_hash}", filename)
    model_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_file)
    return model_file.attn_model(**kwargs)


def load_cfg(string):
    """
    Either load the JSON file pointed to by `string` or parse it as configs
    """
    if string is None:
        return {}
    elif string.endswith('.json'):
        with open(string) as f:
            return json.load(f)
    else:
        def try_eval(s):
            try: return eval(s)
            except: return s
        return {
                try_eval(s.split('=')[0].strip()): # LHS = key
                try_eval(s.split('=')[1].strip())  # RHS = val
            for s in string.split(':')
        }


def pct_to_shape(pcts, shape):
    return tuple(p * s for p, s in zip(pcts, shape[-len(pcts):]))




class GaussianSpatialGain(nm.LayerMod):

    def __init__(self, loc, scale, amp):
        super(GaussianSpatialGain, self).__init__()
        self.loc = loc
        self.scale = scale
        self.amp = amp

    def pre_layer(self, inp, *args, **kwargs):
        """
        ### Arguments
        - `inp` --- Main layer input, of shape (batch, channel, row, col)
        """
        scale = self.scale_array(inp.size())
        scaled = inp * scale.to(device = inp.get_device())
        return (scaled,) + args, kwargs, None

    def scale_array(self, shape):
        h = shape[2]
        w = shape[3]
        gain = lsq_fields.gauss_with_params_torch(
            w, h, [self.loc[0]], [self.loc[1]], [self.scale], [1])
        gain *= (self.amp-1) / gain.mean()
        return gain[np.newaxis, ...] + 1


class GaussianLocatedGain(nm.LayerMod):
    def __init__(self, center, r, beta):
        '''
        ### Arguments
        - `center` --- Center location of the gaussian field in the input,
            a tuple of the form (row_center, col_center)
        - `r` --- Approximate radius of influence of the gaussian field,
            a tuple of the form (row_r, col_r)
        - `beta` --- Multiplicative strength factor
        '''
        super(GaussianLocatedGain, self).__init__()
        self.center = center
        self.r = r
        self.beta = beta

    def pre_layer(self, inp, *args, **kwargs):
        """
        ### Arguments
        - `inp` --- Main layer input, of shape (batch, channel, row, col)
        """
        scaled = inp * self.scale_array(inp)
        return (scaled,) + args, kwargs, None

    def scale_array(self, match):
        shape = match.shape
        # interpolate percentage-unit params up to layer scale
        loc = pct_to_shape(self.center, shape)
        rad = pct_to_shape(self.r, shape)
        # Create grid
        r = np.broadcast_to(np.arange(shape[-2])[:, None], shape[-2:])
        c = np.broadcast_to(np.arange(shape[-1])[None, :], shape[-2:])
        # Gaussian field
        local_r = (r - loc[0]) / rad[0]
        local_c = (c - loc[1]) / rad[1]
        G = np.exp( - local_r**2 / 2 ) * np.exp( - local_c**2 / 2 )
        G = (self.beta - 1) * G + 1
        # Match characteristic of input tensor
        return torch.tensor(G, dtype = match.dtype, device = match.device)




class GaussianLocatedAdd(nm.LayerMod):
    def __init__(self, center, r, beta):
        '''
        ### Arguments
        - `center` --- Center location of the gaussian field in the input,
            a tuple of the form (row_center, col_center)
        - `r` --- Approximate radius of influence of the gaussian field,
            a tuple of the form (row_r, col_r)
        - `beta` --- Multiplicative strength factor
        '''
        super(GaussianLocatedAdd, self).__init__()
        self.center = center
        self.r = r
        self.beta = beta

    def post_layer(self, output, cache):
        """
        ### Arguments
        - `inp` --- Main layer input, of shape (batch, channel, row, col)
        """
        output = output + self.scale_array(output)
        return output

    def scale_array(self, match):
        shape = match.shape
        # interpolate percentage-unit params up to layer scale
        loc = pct_to_shape(self.center, shape)
        rad = pct_to_shape(self.r, shape)
        # Create grid
        r = np.broadcast_to(np.arange(shape[-2])[:, None], shape[-2:])
        c = np.broadcast_to(np.arange(shape[-1])[None, :], shape[-2:])
        # Gaussian field
        local_r = (r - loc[0]) / rad[0]
        local_c = (c - loc[1]) / rad[1]
        G = np.exp( - local_r**2 / 2 ) * np.exp( - local_c**2 / 2 )
        G = (self.beta - 1) * G
        # Match characteristic of input tensor
        return torch.tensor(G, dtype = match.dtype, device = match.device)




class LinearSpatialGain(nm.LayerMod):

    def __init__(self, theta, amp):
        super(LinearSpatialGain, self).__init__()
        self.theta = theta
        self.amp = amp

    def pre_layer(self, inp, *args, **kwargs):
        """
        ### Arguments
        - `inp` --- Main layer input, of shape (batch, channel, row, col)
        """
        scaled = inp * self.scale_array(inp.size())
        return (scaled,) + args, kwargs, None

    def scale_array(self, shape):
        Bs = np.cos(self.theta)
        As = np.sin(self.theta)
        basic_x = np.arange(shape[3])/shape[3]
        basic_y = np.arange(shape[2])/shape[2]
        X, Y = np.meshgrid(basic_x, basic_y)
        grid = As * X[np.newaxis, ...] + Bs * Y[np.newaxis, ...]
        grid = (grid * (self.amp-1) + 1)
        return torch.tensor(grid).float()



class QuadPostAttention(det.QuadAttention):
    pre_layer = nm.LayerMod.pre_layer
    def post_layer(self, outputs, cache):
        scale = self.scale_array(outputs.size())
        if outputs.is_cuda:
            scaled = outputs * scale.to(device = outputs.get_device())
        else:
            scaled = outputs * scale
        return scaled


class GaussianManualShiftAttention(nm.LayerMod):

    def __init__(self, center, r, beta):
        '''
        ### Arguments
        - `center` --- Center location of the gaussian field in the input,
            a tuple of the form (row_center, col_center)
        - `r` --- Approximate radius of influence of the gaussian field,
            a tuple of the form (row_r, col_r)
        - `beta` --- Multiplicative strength factor
        '''
        super(GaussianManualShiftAttention, self).__init__()
        self.center = center
        self.r = r
        self.beta = beta

    def pre_layer(self, inp, *args, **kwargs):
        '''
        ### Arguments
        - `inp` --- As would be passed to nn.Conv2d, a pytorch tensor of
            shape (N, H, C, W).
        - `args`, `kwargs` --- Allows for extra information to be passed,
            maintaining compatibility with torch's call structure. Importantly,
            this function makes use of kwargs['__layer'], as is appended
            by the NetworkManager object.
        '''

        conv = kwargs['__layer']
        if not isinstance(conv, nn.Conv2d):
            raise NotImplementedError("GaussianManualShiftAttention only" + 
                " implemented for wapping torch 2d convolutions. Was asked" + 
                " to wrap {}".format(type(conv)))

        # Set up mimicry of the layer we're wrapping
        pad = dfc.conv_pad(conv)
        flt = dfc.broadcast_filter(conv)
        sten = dfc.filter_stencil(conv)
        grid = dfc.conv_grid(conv, inp.shape[2], inp.shape[2])

        # Shift receptive fields
        # The factor 2 * (27 / 112) matches det.QuadAttention and scales
        # the gaussian field to have approximately sd. = radius
        field = dfc.make_gaussian_shift_field(self.beta, *self.center,
            2 * self.r[0] * (27 / 112), 2 * self.r[1] * (27 / 112))
        grid = dfc.shift_grid_by_field(grid, field)

        # Perform convolution and set up layer bypass
        ix = dfc.merge_grid_and_stencil(grid, sten)
        convolved = dfc.deformed_conv(inp, ix, flt, pad, bias = conv.bias)
        return (inp,) + args, kwargs, convolved

    def post_layer(self, outputs, cache):
        '''Implement layer bypass, replacing the layer's computation
        with the deformed convolution'''
        return cache




class GaussianConvShiftAttention(nm.LayerMod):

    def __init__(self, center, r, beta):
        '''
        ### Arguments
        - `center` --- Center location of the gaussian field in the input,
            a tuple of the form (row_center, col_center)
        - `r` --- Approximate radius of influence of the gaussian field,
            a tuple of the form (row_r, col_r)
        - `beta` --- Multiplicative strength factor
        '''
        super(GaussianConvShiftAttention, self).__init__()
        self.center = center
        self.r = r
        self.beta = beta

    def pre_layer(self, inp, *args, **kwargs):
        '''
        ### Arguments
        - `inp` --- As would be passed to nn.Conv2d, a pytorch tensor of
            shape (N, H, C, W).
        - `args`, `kwargs` --- Allows for extra information to be passed,
            maintaining compatibility with torch's call structure. Importantly,
            this function makes use of kwargs['__layer'], as is appended
            by the NetworkManager object.
        '''
        conv = kwargs['__layer']
        if not isinstance(conv, nn.Conv2d):
            raise NotImplementedError("GaussianConvShiftAttention only" + 
                " implemented for wapping torch 2d convolutions. Was asked" + 
                " to wrap {}".format(type(conv)))
        return (inp,) + args, kwargs, kwargs['__layer']

    def post_layer(self, output, cache):
        '''
        ### Arguments
        - `outputs` --- As would returned by a conv2d, a pytorch tensor of
            shape (N, H, C, W).
        - `cache` --- Allows for extra information to be from pre_layer to
            post_layer. Not used.
        '''

        # Identify the locations mapped to by the standard convolution
        grid_shape = output.shape[-2:]
        rs = np.broadcast_to(np.arange(output.shape[-2])[:, None], grid_shape)
        cs = np.broadcast_to(np.arange(output.shape[-1])[None, :], grid_shape)
        grid = np.stack((rs, cs), axis = -1)

        loc = pct_to_shape(self.center, output.shape)
        rad = pct_to_shape(self.r, output.shape)

        # Shift receptive fields
        # The factor 2 * (27 / 112) matches det.QuadAttention and scales
        # the gaussian field to have approximately sd. = radius
        field = dfc.make_gaussian_shift_field(self.beta, *loc,
            2 * rad[0] * (27 / 112), 2 * rad[1] * (27 / 112))
        grid = dfc.shift_grid_by_field(grid, field)

        shifted = dfc.rigid_shift(output, grid)

        return shifted



class GaussianSensitivityGradAttention(nm.LayerMod):

    def __init__(self, center, r, beta, neg_mode = True):
        '''
        ### Arguments
        - `center` --- Center location of the gaussian field in the input,
            a tuple of the form (row_center, col_center)
        - `r` --- Approximate radius of influence of the gaussian field,
            a tuple of the form (row_r, col_r)
        - `beta` --- Multiplicative strength factor
        - `neg_mode` --- Negative multiplicative factor handling: True for
            warning, `'raise'` for exception, `'fix'` to offset feild locations
            with a negative to be 0 or positive.
        '''
        super(GaussianSensitivityGradAttention, self).__init__()
        self.center = center
        self.r = r
        self.beta = beta
        self.neg_mode = neg_mode

    def pre_layer(self, inp, *args, **kwargs):
        '''
        ### Arguments
        - `inp` --- As would be passed to nn.Conv2d, a pytorch tensor of
            shape (N, H, C, W).
        - `args`, `kwargs` --- Allows for extra information to be passed,
            maintaining compatibility with torch's call structure. Importantly,
            this function makes use of kwargs['__layer'], as is appended
            by the NetworkManager object.
        '''

        conv = kwargs['__layer']
        if not isinstance(conv, nn.Conv2d):
            raise NotImplementedError("GaussianSensitivityGradAttention only" + 
                " implemented for wapping torch 2d convolutions. Was asked" + 
                " to wrap {}".format(type(conv)))

        # Set up mimicry of the layer we're wrapping
        pad = dfc.conv_pad(conv)
        flt = dfc.broadcast_filter(conv)
        sten = dfc.filter_stencil(conv)
        grid = dfc.conv_grid(conv, inp.shape[2], inp.shape[2])
        ix = dfc.merge_grid_and_stencil(grid, sten)

        loc = pct_to_shape(self.center, inp.shape)
        rad = pct_to_shape(self.r, inp.shape)

        # Shift receptive fields
        # The factor 2 * (27 / 112) matches det.QuadAttention and scales
        # the gaussian field to have approximate radius sd.
        field = dfc.make_gaussian_sensitivity_field(*loc,
            2 * rad[0] * (27 / 112), 2 * rad[1] * (27 / 112))
        flt = dfc.apply_filter_field(flt, ix, field, amp = self.beta,
            negative_warn = self.neg_mode)

        # Perform convolution and set up layer bypass
        convolved = dfc.deformed_conv(inp, ix, flt, pad, bias = conv.bias)
        return (inp,) + args, kwargs, convolved

    def post_layer(self, outputs, cache):
        '''Implement layer bypass, replacing the layer's computation
        with the deformed convolution'''
        return cache







