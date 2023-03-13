from proc import network_manager as nm
from proc import deformed_conv as dfc

from torch import nn
import torch
import numpy as np
import copy


def pct_to_shape(pcts, shape):
    return tuple(p * s for p, s in zip(pcts, shape[-len(pcts):]))


class ManualShrinkAttention(nm.LayerMod):

    def __init__(self, center, r, beta, diagnostic = None):
        '''
        ### Arguments
        - `center` --- Center location of the gaussian field in the input,
            a tuple of the form (row_center, col_center)
        - `r` --- Approximate radius of influence of the gaussian field,
            a tuple of the form (row_r, col_r)
        - `beta` --- Multiplicative strength factor
        '''
        super(ManualShrinkAttention, self).__init__()
        self.center = center
        self.r = r
        self.beta = beta
        self.filter_cache = {}


    def squeeze_array(self, shape, beta, center, r):
        # interpolate percentage-unit params up to layer scale
        loc = pct_to_shape(center, shape)
        rad = pct_to_shape(r, shape)
        # Create grid
        r = np.broadcast_to(np.arange(shape[-2])[:, None], shape[-2:])
        c = np.broadcast_to(np.arange(shape[-1])[None, :], shape[-2:])
        # squared Mahalanobis distance from center point
        local_r = (r - loc[0]) / rad[0]
        local_c = (c - loc[1]) / rad[1]
        local_sqrad = local_r**2 + local_c**2
        # functional approximation of beta=11 gaussian RF size change curve
        G = np.exp( - (4.88 * local_sqrad) / 2 )
        H = np.cos(2.89 * local_sqrad)
        squeeze = 1 - beta * H * G
        import matplotlib.pyplot as plt
        print(rad)
        plt.plot(local_sqrad.ravel(), G.ravel())
        plt.show()
        exit()
        return squeeze

    def pre_layer(self, inp, *args, **kwargs):
        """
        ### Arguments
        - `inp` --- Main layer input, of shape (batch, channel, row, col)
        """

        conv = kwargs['__layer']
        if not isinstance(conv, nn.Conv2d):
            raise NotImplementedError("ManualShrinkAttention only" + 
                " implemented for wapping torch 2d convolutions. Was asked" + 
                " to wrap {}".format(type(conv)))

        # Set up mimicry of the layer we're wrapping
        if (conv, inp.shape) not in self.filter_cache:
            self.filter_cache = {}

            pad = dfc.conv_pad(conv)
            flt = dfc.broadcast_filter(conv)
            grid = dfc.conv_grid(conv, inp.shape[2], inp.shape[3])
            factor = self.squeeze_array((grid.shape[0], grid.shape[1]), self.beta, self.center, self.r)
            factor_broad = factor[..., None, None, None]
            sten = dfc.filter_stencil(conv)[None, None]
            sten_center = sten.mean(axis = (2, 3), keepdims = True)
            sten = factor_broad * sten + (1-factor_broad) * sten_center
            ix = dfc.merge_grid_and_nonuniform_stencil(grid, sten)

            dfc_params = (ix, flt, pad)
            self.filter_cache[(conv, inp.shape)] = dfc_params
        else:
            dfc_params = self.filter_cache[(conv, inp.shape)]

        convolved = dfc.deformed_conv(inp, *dfc_params, bias = conv.bias)

        return (inp,) + args, kwargs, convolved

    def post_layer(self, outputs, cache):
        '''Implement layer bypass, replacing the layer's computation
        with the deformed convolution'''
        return cache



def attn_model(layer, beta, r = 0.5):
    '''
    - `neg_mode` --- True for warning, `'raise'` for exception, `'fix'` to offset
        feild locations with a negative to be 0 or positive.
    '''
    # One layer
    if isinstance(layer[0], int):
        return {
            tuple(layer): ManualShrinkAttention((0.25, 0.25), (r, r), beta)
        }
    # Multiple layers
    else:
        return {
            tuple(L): ManualShrinkAttention((0.25, 0.25), (r, r), beta)
            for L in layer
        }








