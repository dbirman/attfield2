from proc import network_manager as nm
import numpy as np
import torch


def pct_to_shape(pcts, shape):
    return tuple(p * s for p, s in zip(pcts, shape[-len(pcts):]))

class RecurrentGaussianGainAttention(nm.LayerMod):

    def __init__(self, center, r, inp_beta, state_beta):
        '''
        ### Arguments
        - `center` --- Center location of the gaussian field in the input,
            a tuple of the form (row_center, col_center)
        - `r` --- Approximate radius of influence of the gaussian field,
            a tuple of the form (row_r, col_r)
        - `inp_beta`, `state_beta` --- Multiplicative strength factor to be
            appliedto the inputs and states, respectively.
        '''
        super(nm.LayerMod, self).__init__()
        self.center = center
        self.r = r
        self.inp_beta = inp_beta
        self.state_beta = state_beta

    def pre_layer(self, inp = None, state = None, batch_size = None, **kws):
        """
        ### Arguments
        - `x` --- 
        """
        print("Applying gain to", kws['__layer'])
        if inp is not None:
            scaled_inp = inp * self.scale_array(inp, self.inp_beta)
        else:
            scaled_inp = None

        if state is not None:
            scaled_state = state * self.scale_array(state, self.state_beta)
        else:
            scaled_state = None
        
        return (scaled_inp, scaled_state, batch_size), {}, None


    def scale_array(self, match, beta):
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
        G = (beta - 1) * G + 1
        # Match characteristic of input tensor
        return torch.tensor(G, dtype = match.dtype, device = match.device)



def attn_model(layer, beta, loc = (0.25, 0.25), r = (0.125, 0.125)):
    '''
    - `beta` --- Tuple of floats (inp_beta, state_beta)
    '''
    # One layer
    if isinstance(layer[0], int):
        return {
            tuple(layer): RecurrentGaussianGainAttention(loc, r, *beta)
        }
    # Multiple layers
    else:
        return {
            tuple(L): RecurrentGaussianGainAttention(loc, r, *beta)
            for L in layer
        }
