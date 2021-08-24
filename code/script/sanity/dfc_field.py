import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)


import matplotlib.pyplot as plt
import numpy as np

from torch import nn
import torch

import proc.deformed_conv as dfc


def pct_to_shape(pcts, shape):
    return tuple(p * s for p, s in zip(pcts, shape[-len(pcts):]))

# set up a situation to mimic
conv = nn.Conv2d(1, 1, 7, bias = False, padding = 3)
c, r = np.meshgrid(np.arange(14), np.arange(14))
inp = torch.tensor((c % 4 == 1) + (r % 5 == 1) + (r % 6 == 1))[None, None, :, :].float()


# set up dfc mimicry
pad = dfc.conv_pad(conv)
flt = dfc.broadcast_filter(conv)
sten = dfc.filter_stencil(conv)
grid = dfc.conv_grid(conv, inp.shape[2], inp.shape[2])
ix = dfc.merge_grid_and_stencil(grid, sten)

center = (0.25, 0.25)
r = (0.25, 0.25)
beta = 11.0
loc = pct_to_shape(center, inp.shape)
rad = pct_to_shape(r, inp.shape)

# Had to introduce + pad[i] terms here to fix dfc bug,
# but eventually moved this behavior to deformed_conv.py itself
# in the apply_magnitude_field function
# field =  dfc.make_gaussian_sensitivity_field(
#   loc[0] + pad[0], loc[1] + pad[1],
#     4 * rad[0] * (27 / 112), 4 * rad[1] * (27 / 112))
unshifted_field =  dfc.make_gaussian_sensitivity_field(
    loc[0], loc[1],
    4 * rad[0] * (27 / 112), 4 * rad[1] * (27 / 112))

# Get the gained version of the convolution for dfc
# and measure the effective gain field
gained_flt, normalizer = dfc.apply_magnitude_field(flt, ix, unshifted_field, pad, amp = beta)
c, r = np.meshgrid(np.arange(inp.shape[-2]), np.arange(inp.shape[-1]))
gain_field = torch.tensor((unshifted_field(c, r) * (beta-1)) + 1).float()

# calculate via torch and via dfc
raw_conved = conv(inp * gain_field)
dfc_conved = dfc.deformed_conv(inp, ix, gained_flt, pad)

plt.imshow((raw_conved - dfc_conved).detach()[0,0], cmap = 'RdBu')
plt.colorbar()

rng = max(abs(raw_conved.min()), abs(raw_conved.max()),
          abs(dfc_conved.min()), abs(dfc_conved.max()))
fig, ax = plt.subplots(figsize = (8, 4), ncols = 2)
im = ax[0].imshow(raw_conved[0,0].detach(), vmin = -rng, vmax = rng, cmap = 'RdBu')
plt.colorbar(im, ax = ax[0]); ax[0].set_title("raw")
im = ax[1].imshow(dfc_conved[0,0].detach(), vmin = -rng, vmax = rng, cmap = 'RdBu')
plt.colorbar(im, ax = ax[1]); ax[1].set_title("dfc")
plt.show()



















