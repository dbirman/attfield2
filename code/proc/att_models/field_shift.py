from proc import network_manager as nm
from proc import spatial_fields as fld
from proc import deformed_conv as dfc

import pickle as pkl
from torch import nn
import numpy as np
import torch


def attn_model(layer, beta, field_file):
    field = fld.LinearField.load(field_file)
    return  {
        tuple(layer): FieldShiftAttention(beta, field)
    }


class FieldShiftAttention(nm.LayerMod):

    def __init__(self, beta, field):
        '''
        ### Arguments
        - `field` --- Field object with a query function that yields shift
            row and column shift amounts at given locations of input space. 
        - `beta` --- Multiplicative strength factor
        '''
        super(FieldShiftAttention, self).__init__()
        self.beta = beta
        self.field = field
        self.grid_cache = {}

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
            raise NotImplementedError("FieldShiftAttention only" + 
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

        if (self.beta, output.shape[-2:]) not in self.grid_cache:

            # Identify the locations mapped to by the standard convolution
            grid_shape = output.shape[-2:]
            rs = np.linspace(0, 1, output.shape[-2] + 1)[:-1]
            cs = np.linspace(0, 1, output.shape[-1] + 1)[:-1]
            rs = np.broadcast_to(rs[:, None], grid_shape)
            cs = np.broadcast_to(cs[None, :], grid_shape)
            grid = np.stack((rs, cs), axis = -1)

            # Shift receptive fields according to the given field object
            field_r, field_c = self.field.query(rs, cs)
            field_r *= grid_shape[0] * self.beta
            field_c *= grid_shape[1] * self.beta
            grid[..., 0] *= grid_shape[0]
            grid[..., 1] *= grid_shape[1]
            shifts = np.stack([field_r, field_c], axis = -1)
            grid += shifts
            # import matplotlib.pyplot as plt
            # plt.imshow(shifts[:, :, 0]); plt.colorbar(); plt.show(); exit()

            self.grid_cache[self.beta, output.shape[-2:]] = grid

        else:
            grid = self.grid_cache[self.beta, output.shape[-2:]]

        shifted = dfc.rigid_shift(output, grid)
        return shifted










