from torch import nn
import numpy as np
import torch
import gc





# =======================================================================
# -  Manipulating grids, stencils, filters, etc                         -
# =======================================================================


def conv_grid(conv, in_rows, in_cols):
    '''
    For a given torch Conv2D and input shape, return the convolution
    grid before deformation.

    ### Returns
    - `grid` --- `np.ndarray` of shape (out_row, out_col, 2) giving
        the index in the input image corresponding to the top-left-most
        component of the filter
    '''

    if conv.groups != 1:
        raise NotImplementedError("conv_grid not implemented for " + 
                                  "grouped convolutions.")
    if conv.dilation != (1, 1):
        raise NotImplementedError("conv_grid not implemented for " + 
                                  "dilated convolutions.")

    # Note: https://github.com/vdumoulin/conv_arithmetic/
    # is a great resource for understanding these computations

    max_r_ix = (in_rows + 2 * conv.padding[0] 
                - (conv.kernel_size[0] - 1))
    r_ix = np.arange(0, max_r_ix, conv.stride[0])

    max_c_ix = (in_cols + 2 * conv.padding[1] 
                - (conv.kernel_size[1] - 1))
    c_ix = np.arange(0, max_c_ix, conv.stride[1])

    return np.stack(np.meshgrid(c_ix, r_ix)[::-1], axis = -1)




def filter_stencil(conv):
    '''
    Create a stencil to match the operation performed by the given Conv2d

    ### Arguments
    - `conv` --- `nn.Conv2d` layer to match.

    ### Returns
    - `stencil` --- `np.ndarray` shape
        (sten_row, sten_col, 2)
    '''

    if conv.groups != 1:
        raise NotImplementedError("conv_grid not implemented for " + 
                                  "grouped convolutions.")

    # Note: https://github.com/vdumoulin/conv_arithmetic/
    # is a great resource for understanding these computations

    r_ix = np.arange(conv.kernel_size[0]) * conv.dilation[0]
    c_ix = np.arange(conv.kernel_size[1]) * conv.dilation[1]
    return np.stack(np.meshgrid(c_ix, r_ix)[::-1], axis = -1)


def conv_pad(conv):
    '''
    Extract padding information from a Conv2D to use for `take_conv_inputs`
    '''
    return (conv.padding[1], conv.padding[1], conv.padding[0], conv.padding[0])


def broadcast_filter(conv):
    '''
    Create a filter tensor applying the same operatios as `conv` would.

    ### Arguments
    - `conv` --- `nn.Conv2d` layer to match.

    ### Returns
    - `flt` --- Torch parameter.
        Shape (C_out, C_in, 1, 1, sten_rows, sten_cols), notably broadcasts
        with (C_out, C_in, out_row, out_col, sten_rows, sten_cols) and
        so is compatible with `deformed_conv`
    '''
    return conv.weight[:, :, None, None, :, :]



def shift_grid_by_field(grid, field_fn):
    '''
    ### Arguments
    - `field_fn` --- A function taking two arguments: (row, col)
        and which returns two values: (row_shift, col_shift)
    '''
    r_shift, c_shift = field_fn(grid[..., 0], grid[..., 1])
    shifts = np.stack((rshift, c_shift), axis = -1)
    return grid + shifts



# Not sure exactly how this used to be used, so keeping it around
def __make_gaussian_shift_field_old(amp, mu_r, mu_c, sigma_r, sigma_c):
    '''
    ### Arguments
    - `amp` --- Multiplicative factor in shift magnitude.
    '''
    def field(r, c):
        local_r = (r - mu_r) / sigma_r
        local_c = (c - mu_c) / sigma_c
        G = np.exp( - local_r**2 / 2 ) * np.exp( - local_c**2 / 2 )
        # Shift values are by default *very* small so scale up by 1000
        Dr = 1000 * amp * local_r/sigma_r**2 * G
        Dc = 1000 * amp * local_c/sigma_c**2 * G
        return Dr, Dc
    return field


def make_gaussian_shift_field(amp, mu_r, mu_c, sigma_r, sigma_c):
    '''
    ### Arguments
    - `amp` --- Multiplicative factor in shift magnitude.
    '''
    def field(r, c):
        local_r = (r - mu_r) / sigma_r
        local_c = (c - mu_c) / sigma_c
        G = - np.exp( - local_r**2 / 2 ) * np.exp( - local_c**2 / 2 )
        Dr = (amp - 1) * 2 * local_r * G
        Dc = (amp - 1) * 2 * local_c * G
        return Dr, Dc
    return field


def make_gaussian_sensitivity_field(mu_r, mu_c, sigma_r, sigma_c):
    def field(r, c):
        local_r = (r - mu_r) / sigma_r
        local_c = (c - mu_c) / sigma_c
        G = np.exp( - local_r**2 / 2 ) * np.exp( - local_c**2 / 2 )
        return G
    return field


def apply_magnitude_field(flt, ix, field_fn, pad, amp):
    """
    ### Arguments
    - `flt` --- Torch tensor, elongated filter, shape
        shape: (C_out, C_in, out_row, out_col, sten_rows, sten_cols)
        Note that these dimensions can also be `1` in which case
        torch/numpy will broadcast appropriately.
    - `ix` --- Numpy array of locations which the filter applies to,
        shape: (out_row, out_col, sten_row, sten_col, 2)
    - `field_fn` --- A function taking two arguments: (row, col)
        and which returns the field value at the point. Should
        permit vectorization by passing row, col as vectors.
    - `amp` --- Multiplicative strength factor of the field adjustment
    """

    # Shape: (out_row * out_col * sten_row * sten_col)
    field = field_fn(ix[..., 0].ravel() - pad[2], ix[..., 1].ravel() - pad[0])
    # Shape: (1, 1, out_row, out_col, sten_row, sten_col)
    field = field.reshape(ix.shape[:-1])
    field = (field * (amp-1)) + 1
    

    # Apply multiplicative factors and return
    field = torch.tensor(field[None, None, ...], dtype = flt.dtype)
    if flt.is_cuda:
        field = field.cuda()
    # shape: (C_out, C_in, out_row, out_col, sten_rows, sten_cols)
    scaled_flt = field * flt

    # calculate new magnitudes to normalize by
    representative_flt_new = scaled_flt
    # shape: (C_out, out_row, out_col)
    flt_magnitudes_new = torch.sqrt((representative_flt_new ** 2).sum(dim = (1, -2, -1)))
    # assume filter magnitudes were the same per-channel before applying field
    # shape: (C_out, C_in, 1, 1, sten_rows, sten_cols)
    representative_flts_old = flt[:, :, 0, 0, :, :][:, :, None, None, :, :]
    # shape: (C_out, 1, 1)
    flt_magnitudes_old = torch.sqrt((representative_flts_old ** 2).sum(dim = (1, -2, -1)))
    # shape: (C_out, out_row, out_col)
    normalizer = flt_magnitudes_old / flt_magnitudes_new
    renormed = normalizer[:, None, :, :, None, None] * scaled_flt

    return renormed, normalizer
    



def apply_filter_field(flt, ix, field_fn, amp, negative_warn = True):
    """
    ### Arguments
    - `flt` --- Torch tensor, elongated filter, shape
        shape: (C_out, C_in, out_row, out_col, sten_rows, sten_cols)
        Note that these dimensions can also be `1` in which case
        torch/numpy will broadcast appropriately.
    - `ix` --- Numpy array of locations which the filter applies to,
        shape: (out_row, out_col, sten_row, sten_col, 2)
    - `field_fn` --- A function taking two arguments: (row, col)
        and which returns the field value at the point. Should
        permit vectorization by passing row, col as vectors.
    - `amp` --- Multiplicative strength factor of the field adjustment
    """
    # Shape: (out_row * out_col * sten_row * sten_col)
    field = field_fn(ix[..., 0].ravel(), ix[..., 1].ravel())
    # Shape: (out_row, out_col, sten_row, sten_col)
    field = field.reshape(ix.shape[:-1])
    # normalize the field adjustments so as not to increase the
    # overall signal level
    # Shape: (out_row, out_col, sten_row, sten_col)
    field -= field.mean(axis = (-2, -1), keepdims = True)

    # Convert to multiplicative factor and apply
    field = ((field * amp) + 1)

    # Warn if the amplitude is large enough to cause the filter
    # to flip at any point
    if negative_warn is True or negative_warn == 'raise':
        if np.any(field < 0):
            if negative_warn is True:
                import sys
                sys.stderr.write(f"WARNING: multiplicative filter " + 
                    f"field < 0 for amplitude {amp}.")
                sys.stderr.flush()
            else:
                raise ValueError(f"Multiplicative filter " + 
                    f"field < 0 for amplitude {amp}.")
    # Or optionally offset those field location with negative amplitude
    # to have zero or positive
    elif negative_warn == 'fix':
        if np.any(field < 0):
            add = field.min(axis = (-2, -1), keepdims = True)
            add[add >= 0] = 0.
            print("[dfc] fixing", np.count_nonzero(add), "filters")
            field += add

    # Apply and return
    field = torch.tensor(field[None, None, ...], dtype = flt.dtype)
    if flt.is_cuda:
        field = field.cuda()
    return field * flt



# =======================================================================
# -  Perform operations using grids, stencils, etc                      -
# =======================================================================


def merge_grid_and_stencil(conv_grid, filter_stencil):
    '''
    Combine a grid and stencil to get convolution indices
    ### Returns
    - `ix` --- `np.ndarray` shape (out_row, out_col, sten_row, sten_col, 2)
    '''

    # Shape: (out_row, out_col, 2, 1, 1)
    grid_broadcast = conv_grid[..., None, None]
    # Shape: (1, 1, 2, filter_rows, filter_cols)
    filter_stencil = np.moveaxis(filter_stencil, -1, 0)
    sten_broadcast = filter_stencil[None, None, ...]

    ix = grid_broadcast + sten_broadcast
    ix = np.moveaxis(ix, 2, -1)

    return ix


def take_conv_inputs(inp, ix, pad):
    '''
    Get a view of the input tensor (N, C, H, W) as indexed by
    applying the given stencil at every location of the given grid.
    
    ### Returns
    - `sliced` --- `torch.tensor`
        shape: (N, C, out_row, out_col, sten_row, sten_col)
    '''

    # To understand what this is doing:
    #inp = torch.arange(50).view(2, 5, 5)
    #inp[..., ((1, 2), (1, 2)), ((3, 4), (3, 4))].shape
    #r_ix = np.array(((1, 2), (1, 2), (1, 2)))
    #c_ix = np.array(((1, 2), (1, 2), (1, 2)))
    #torch.arange(50).view(2, 5, 5)[..., r_ix, c_ix]


    # Correct for indices outside the input image
    r_tp_pad = -min(0, ix[..., 0].min())
    r_bt_pad = max(0, ix[..., 0].max() - (inp.shape[2]-1))
    c_lf_pad = -min(0, ix[..., 1].min())
    c_ri_pad = max(0, ix[..., 1].max() - (inp.shape[3]-1))
    pad = (pad[0] + c_lf_pad, pad[1] + c_ri_pad,
           pad[2] + r_tp_pad, pad[3] + r_bt_pad)
    r_ix = ix[..., 0] + r_tp_pad
    c_ix = ix[..., 1] + c_lf_pad

    if not all(p == 0 for p in pad):
        inp = nn.functional.pad(inp, pad = pad, mode = 'constant', value = 0.)
    ret = inp[..., r_ix, c_ix]
    del r_ix, c_ix;
    gc.collect()
    return ret




def deformed_conv(inp, ix, flt, pad, bias = None):
    '''
    ### Arguments
    - `flt` --- Elongated filter to apply, for example, the
        output of `broadcast_filter`. Shape must be
        (C_out, C_in, out_row, out_col, sten_rows, sten_cols)
        Note that these dimensions can also be `1` in which case
        torch will broadcast appropriately.
    '''


    # Perform the convolution at 4 (integer) reference locations to 
    # interpolate between

    op_keys = {0: np.floor, 1: np.ceil}
    ref_convs = {}

    # Add batch dimension to filter
    # shape: (1, C_out, C_in, out_row, out_col, sten_rows, sten_cols)
    flt = flt[None, ...]

    for row_op in range(2):
        for col_op in range(2):

            # Apply floor or ceiling to get real indexes from floats
            rounded_ix = np.stack((
                op_keys[row_op](ix[..., 0]),
                op_keys[col_op](ix[..., 1])
            ), axis = -1).astype('long')

            # Elongate input
            # shape: (N, C_out, C_in, out_row, out_col, sten_rows, sten_cols)
            inp_view = take_conv_inputs(inp, rounded_ix, pad)
            inp_view = inp_view[:, None, ...]

            # Perform convolution
            # shape: (N, C_out, C_in, out_row, out_col, sten_rows, sten_cols)
            torch.cuda.empty_cache()
            conved = inp_view * flt
            

            # shape: (N, C_out, out_row, out_col, sten_rows, sten_cols)
            conved = torch.sum(conved, dim = 2)
            ref_convs[(row_op, col_op)] = conved

    # Perform bilinear interpolation weighted by the float indices

    # Get fractional part of indices (interpolation weights)
    c = np.remainder(ix[..., 0].astype('float'), 1)
    r = np.remainder(ix[..., 1].astype('float'), 1)

    # Shape: (1, 1, out_row, out_col, sten_row, sten_col)
    c = torch.tensor(c, dtype = inp.dtype, device = ref_convs[0,0].device)[None, None, ...]
    r = torch.tensor(r, dtype = inp.dtype, device = ref_convs[0,0].device)[None, None, ...]

    # shape: (N, C_out, out_row, out_col, sten_row, sten_col)
    conved = (
        (1 - c) * (1 - r) * ref_convs[0, 0] +
             c  * (1 - r) * ref_convs[1, 0] + 
        (1 - c) *      r  * ref_convs[0, 1] + 
             c  *      r  * ref_convs[1, 1])


    # Perform final sum of convolution and add bias
    # Shape: (N, C_out, out_row, out_col, sten_row, sten_col)
    conved = torch.sum(conved, dim = (-2, -1))
    if bias is not None:
        print("used bias")
        conved = conved + bias[None, :, None, None]

    return conved



def rigid_shift(inputs, grid):
    """
    Index into the last two dimensions of `inputs` by `grid`
    with bilinear interpolation.
    """

    # don't leave image bounds
    grid[..., 0] = np.clip(grid[..., 0], 0, inputs.shape[-2] - 1)
    grid[..., 1] = np.clip(grid[..., 1], 0, inputs.shape[-1] - 1)

    # weights
    c = np.remainder(grid[..., 0].astype('float'), 1)
    r = np.remainder(grid[..., 1].astype('float'), 1)
    c = torch.tensor(c, dtype = inputs.dtype, device = inputs.device)
    r = torch.tensor(r, dtype = inputs.dtype, device = inputs.device)

    op_keys = {0: np.floor, 1: np.ceil}
    ref_convs = {}
    weights = {
        ('r', 0): (1 - c),
        ('r', 1): c,
        ('c', 0): (1 - r),
        ('c', 1): r
    }

    shifted = None

    for row_op in range(2):
        for col_op in range(2):

            # Apply floor or ceiling to get real indexes from floats
            rounded_ix = np.stack((
                op_keys[row_op](grid[..., 0]),
                op_keys[col_op](grid[..., 1])
            ), axis = -1).astype('long')

            contrib = inputs[..., rounded_ix[..., 0], rounded_ix[..., 1]]
            contrib *= weights['r', row_op] * weights['c', col_op]

            if shifted is None:
                shifted = contrib
            else:
                shifted += contrib

    return shifted



    


