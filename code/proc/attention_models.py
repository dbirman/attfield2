from proc import network_manager as nm
from proc import lsq_fields

import numpy as np
import torch


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.patches as patches
import seaborn as sns
sns.set(color_codes = True)
sns.set_style('white')


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

