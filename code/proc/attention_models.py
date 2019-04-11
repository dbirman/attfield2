from proc import network_manager as nm
from proc import lsq_fields

import numpy as np



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
        h = inp.size()[2]
        w = inp.size()[3]
        gain = lsq_fields.gauss_with_params_torch(
            w, h, [self.loc[0]], [self.loc[1]], [self.scale], [self.amp])
        scaled = inp * gain[np.newaxis, ...]
        return (scaled,) + args, kwargs