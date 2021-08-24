from proc import network_manager as nm
from proc import deformed_conv as dfc

from torch import nn
import numpy as np




# def make_unnormalized_gaussian_sens_field(mu_r, mu_c, sigma_r, sigma_c):
#     def field(r, c):
#         local_r = (r - mu_r) / sigma_r
#         local_c = (c - mu_c) / sigma_c
#         G = np.exp( - local_r**2 / 2 ) * np.exp( - local_c**2 / 2 )
#         return G
#     return field


def pct_to_shape(pcts, shape):
    return tuple(p * s for p, s in zip(pcts, shape[-len(pcts):]))


class NormalizedSensitivityGradAttention(nm.LayerMod):

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
        super(NormalizedSensitivityGradAttention, self).__init__()
        self.center = center
        self.r = r
        self.beta = beta
        self.neg_mode = neg_mode
        self.filter_cache = {}

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
            raise NotImplementedError("NormalizedSensitivityGradAttention only" + 
                " implemented for wapping torch 2d convolutions. Was asked" + 
                " to wrap {}".format(type(conv)))

        # Set up mimicry of the layer we're wrapping
        if (conv, inp.shape) not in self.filter_cache:
            self.filter_cache = {}

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
            field =  dfc.make_gaussian_sensitivity_field(*loc,
                4 * rad[0] * (27 / 112), 4 * rad[1] * (27 / 112))
            flt, _ = dfc.apply_magnitude_field(flt, ix, field, pad, amp = self.beta)

            self.filter_cache[(conv, inp.shape)] = ix, flt, pad
        else:
            ix, flt, pad = self.filter_cache[(conv, inp.shape)]

        # Perform convolution and set up layer bypass
        convolved = dfc.deformed_conv(inp, ix, flt, pad, bias = conv.bias)

        return (inp,) + args, kwargs, convolved

    def post_layer(self, outputs, cache):
        '''Implement layer bypass, replacing the layer's computation
        with the deformed convolution'''
        return cache


def attn_model(layer, beta, r = 0.25, neg_mode = True, **kws):
    '''
    - `neg_mode` --- True for warning, `'raise'` for exception, `'fix'` to offset
        feild locations with a negative to be 0 or positive.
    '''
    # One layer
    if isinstance(layer[0], int):
        return {
            tuple(layer): NormalizedSensitivityGradAttention((0.25, 0.25), (r, r), beta, neg_mode)
        }
    # Multiple layers
    else:
        return {
            tuple(L): NormalizedSensitivityGradAttention((0.25, 0.25), (r, r), beta, neg_mode)
            for L in layer
        }








