from proc import network_manager as nm
from proc import deformed_conv as dfc

from torch import nn
import torch
import numpy as np
import copy


def pct_to_shape(pcts, shape):
    return tuple(p * s for p, s in zip(pcts, shape[-len(pcts):]))


class NormalizedSensitivityGradAttention(nm.LayerMod):

    def __init__(self, center, r, beta, diagnostic = None):
        '''
        ### Arguments
        - `center` --- Center location of the gaussian field in the input,
            a tuple of the form (row_center, col_center)
        - `r` --- Approximate radius of influence of the gaussian field,
            a tuple of the form (row_r, col_r)
        - `beta` --- Multiplicative strength factor
        '''
        super(NormalizedSensitivityGradAttention, self).__init__()
        self.center = center
        self.r = r
        self.beta = beta
        self.filter_cache = {}
        self.diagnostic = diagnostic #"plots/detail/sn_filters.pdf"

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

    def pre_layer(self, inp, *args, **kwargs):
        """
        ### Arguments
        - `inp` --- Main layer input, of shape (batch, channel, row, col)
        """
        scale_array = self.scale_array(inp)

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
            gained_flt, normalizer = dfc.apply_magnitude_field(
                flt, ix, field, pad, amp = self.beta)

            # c, r = np.meshgrid(np.arange(inp.shape[-2]), np.arange(inp.shape[-1]))
            # gain_field = torch.tensor((field(c, r) * (self.beta-1)) + 1).float()
            # print("WHY WERE THE GAIN FIELDS DIFFERENT?")
            # import matplotlib.pyplot as plt
            # rng = max(abs(gain_field.min()), abs(gain_field.max()),
            #           abs(scale_array.min()), abs(scale_array.max()))
            # fig, ax = plt.subplots(figsize = (8, 4), ncols = 2)
            # im = ax[0].imshow(gain_field.detach(), vmin = 0, vmax = rng, cmap = 'RdBu')
            # plt.colorbar(im, ax = ax[0]); ax[0].set_title("gain_field")
            # im = ax[1].imshow(scale_array.detach(), vmin = 0, vmax = rng, cmap = 'RdBu')
            # plt.colorbar(im, ax = ax[1]); ax[1].set_title("scale_array")
            # plt.show()

            # plt.imshow((gain_field - gain_field).detach(), )
            # global n
            # n = n + 1
            # if n > 3: exit()

            if self.diagnostic is not None:
                from matplotlib.backends.backend_pdf import PdfPages
                import matplotlib.pyplot as plt
                with PdfPages(self.diagnostic) as pdf:
                    to_plot = [(0,0), (0,1), (0,2)]
                    n_filts = 30
                    rs = np.arange(gained_flt.shape[2])[
                        ::max(1, gained_flt.shape[2]//n_filts)]
                    cs = np.arange(gained_flt.shape[3])[
                        ::max(1, gained_flt.shape[3]//n_filts)]
                    # c, r = np.meshgrid(r, c)
                    for ix in to_plot:
                        for plot_arr, plot_name in [
                            (gained_flt / normalizer[:, None, :, :, None, None], 'gained'),
                            (gained_flt, "normalized")]:
                            sten_rows, sten_cols = gained_flt.shape[4:]
                            sten_rows += 1
                            sten_cols += 1
                            img = np.zeros([
                                len(rs) * sten_rows,
                                len(cs) * sten_cols])
                            for i_r, r in enumerate(rs):
                                for i_c, c in enumerate(cs):
                                    img[i_r * sten_rows : (i_r + 1) * sten_rows - 1,
                                        i_c * sten_cols : (i_c + 1) * sten_cols - 1
                                    ] = plot_arr[ix[0], ix[1], r, c, :, :].detach()
                            rng = max(abs(img.min()), abs(img.max()))
                            plt.imshow(img, vmin = -rng, vmax = rng, cmap = 'RdBu')
                            plt.colorbar()
                            plt.title(f"Filter {ix[1]}->{ix[0]} : {plot_name}")
                            pdf.savefig()
                            plt.close()

                        plt.imshow(1/(normalizer[ix[0]].detach()), cmap = 'viridis')
                        plt.colorbar()
                        plt.title(f"-log10(normalizer)")
                        pdf.savefig()
                        plt.close()

            self.filter_cache[(conv, inp.shape)] = normalizer
        else:
            normalizer = self.filter_cache[(conv, inp.shape)]

        if conv.bias is not None:
            bias_correction = conv.bias[:, None, None] * (normalizer - 1)
        # convolved = dfc.deformed_conv(inp, ix, gained_flt, pad, bias = conv.bias)

        return (inp * scale_array,) + args, kwargs, (normalizer, bias_correction)
        # return (inp,) + args, kwargs, normalizer

    def post_layer(self, outputs, cache):
        '''Implement layer bypass, replacing the layer's computation
        with the deformed convolution'''
        # shape: (C_out, rows, cols)
        normalizer, bias_correction = cache
        norm_ret = outputs * normalizer[None] - bias_correction[None]
        # mult_ret = convolved * normalizer[None] - bias_correction[None]
        # import matplotlib.pyplot as plt
        # rng = max(abs(norm_ret[0,0].min()), abs(norm_ret[0,0].max()),
                  # abs(mult_ret[0,0].min()), abs(mult_ret[0,0].max()))
        # fig, ax = plt.subplots(figsize = (8, 4), ncols = 2)
        # im = ax[0].imshow(norm_ret[0,0].detach(), vmin = -rng, vmax = rng, cmap = 'RdBu')
        # plt.colorbar(im, ax = ax[0]); ax[0].set_title("norm")
        # im = ax[1].imshow(mult_ret[0,0].detach(), vmin = -rng, vmax = rng, cmap = 'RdBu')
        # plt.colorbar(im, ax = ax[1]); ax[1].set_title("mult")
        # plt.show()

        # plt.imshow((outputs - convolved)[0,0].detach(), cmap = 'RdBu')
        # plt.colorbar(); plt.title('diff'); plt.show()
        # global n
        # n = n + 1
        # if n > 3: exit()
        # shape: (batch, C_out, rows, cols)
        return norm_ret

# n = 0


def attn_model(layer, beta, r = 0.25, **kws):
    '''
    - `neg_mode` --- True for warning, `'raise'` for exception, `'fix'` to offset
        feild locations with a negative to be 0 or positive.
    '''
    # One layer
    if isinstance(layer[0], int):
        return {
            tuple(layer): NormalizedSensitivityGradAttention((0.25, 0.25), (r, r), beta)
        }
    # Multiple layers
    else:
        return {
            tuple(L): NormalizedSensitivityGradAttention((0.25, 0.25), (r, r), beta)
            for L in layer
        }








