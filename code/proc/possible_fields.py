from proc import voxel_selection as vx
from proc import network_manager as nm
from proc import backprop_fields
from proc import cornet

from torch import nn
import pandas as pd
import numpy as np


def as_tuple(param):
    if not isinstance(param, tuple):
        param = (param, param)
    else:
        param = param
    return param


def trace_rf(layers, manager):
    '''
    Trace a bounding box on the spatial receptive field as it will
    be altered by sequential computation of several layers. At the
    moment only 2D max-pooling and convolution are understood as having
    and affect on the receptive field.
    ### Arguments
    - `layers` --- An iterable giving the indexes of layers within the
        passed NetworkManager.
    - `manager` --- A NetworkManager to inspect.
    ### Returns
    - `center_adjusts` --- A tuple of ints `(row, col)` giving the
        multiplicative factor to convert from the center location of
        a receptive field after these layers' computation to the center
        location in an input image.
    - `span` --- A tuple of ints `(rows, cols)` giving the number of
        rows and columns of an input image that the receptive field of
        a voxel in the last layer of `layers` could span.
    '''
    row_adjust = col_adjust = 1
    row_span = col_span = 1

    for layer in layers[::-1]:
        module = manager.modules[layer]

        if isinstance(module, nn.modules.pooling.MaxPool2d):

            stride = as_tuple(module.stride)
            row_adjust *= stride[0]
            col_adjust *= stride[1]

            kernel_size = as_tuple(module.kernel_size)
            dilation = as_tuple(module.dilation)
            row_span *= stride[0]
            col_span *= stride[1]
            row_span += (kernel_size[0]) * dilation[0]
            col_span += (kernel_size[1]) * dilation[1]

        elif isinstance(module, nn.modules.conv.Conv2d):

            stride = as_tuple(module.stride)
            row_adjust *= stride[0]
            col_adjust *= stride[1]

            kernel_size = as_tuple(module.kernel_size)
            dilation = as_tuple(module.dilation)
            row_span += (kernel_size[0]) * dilation[0]
            col_span += (kernel_size[1]) * dilation[1]

    return (row_adjust, col_adjust), (row_span, col_span)


def sequential_leaves(manager, stop = False):
    '''Find the leaf nodes of a network model, assuming that it is
    arranged sequentially. These are the layers with no children
    in the manager, where a child is defined as a layer having the
    same index but with an added suffix. The `stop` argument allows
    one to stop the search after a certain index is reached if you
    only need the list up to a certain layer.'''
    leaves = []
    max_len = 0
    for layer_i in sorted(manager.computed.keys()):
        is_leaf = True
        # Make sure layer_i has no children
        for layer_j in manager.computed.keys():
            # Don't bother comparing if the potential child has a
            # shorter or equal-length layer index
            if len(layer_i) < len(layer_j):
                # If layer_i's index is the same as the start of
                # layer_j's index then layer_j is a child of layer_i
                if all([layer_i[k] == layer_j[k]
                        for k in range(len(layer_i))]):
                    is_leaf = False
        if is_leaf:
            leaves.append(layer_i)
            max_len = max(max_len, len(layer_i))
        if stop is not False:
            if len(layer_i) == len(stop):
                if all([layer_i[k] == stop[k]
                        for k in range(len(layer_i))]):
                    return leaves
    return leaves


def rf_bboxes(voxels, manager, layer_order = 'sequential'):
    '''
    Compute a bounding box on the possible receptive fields for given
    `voxels` in the network described by `manager`.
    ### Arguments
    - `voxels` --- A dictionary mapping layers to VoxelIndex objects.
    - `manager` --- A NetworkManager object
    - `layer_order` --- Can be `'sequential'`, in which case the
        `sequential_leaves` function will be used to determine which
        layers contribute to the computation of a voxel. Otherwise
        `layer_order` should be a list giving the order of layer
        computation and the model is still assumed to be sequential
        but that function will not be used.
    ### Returns
    - `bboxes` --- A dictionary mapping layers to collections of four
        arrays: left, top, width, height. The i-th index of each array
        describes the bounding box of the i-th voxel.
    '''
    bbs = {}
    for layer, layer_vox in voxels.items():
        # Get a list of the layers that compute the voxels in question
        if layer_order == 'sequential':
            curr_layers = sequential_leaves(manager, stop = layer)
        else:
            curr_layers = layer_order[:layer_order.index(layer)+1]
        # Trace back a bounding box on the receptive fields
        rf = trace_rf(curr_layers, manager)
        vox_rows = layer_vox._idx[-2]
        vox_cols = layer_vox._idx[-1]

        # Arrange the RF data from `trace_rf` into four arrays:
        # left, top, width, height
        bbs[layer] = (
            vox_cols * rf[0][1] - rf[1][1]//2,
            vox_rows * rf[0][0] - rf[1][0]//2,
            np.ones_like(vox_cols) * rf[1][1],
            np.ones_like(vox_rows) * rf[1][0]
        )
    return bbs


def parameter_estimates_bbox(bboxes, voxels):
    """
    Estimate parameters of gaussian receptive field based
    on possible bounding boxes.
    """

    params = {}
    for layer, layer_vox in voxels.items():
        col, row, w, h = bboxes[layer]
        mean_X = col + w/2
        mean_Y = row + h/2
        var = (w+h)/4
        beta = np.ones_like(var)
        params[layer] = (mean_X, mean_Y, var, beta)
    return params


def estimates_to_params(estimates):
    return {l: np.array(est).T.tolist() for l, est in estimates.items()}



def save_bbox_csv(filename, voxels, bboxes):
    dfs = []
    for layer, layer_vox in voxels.items():
        layer_str = '.'.join([str(i) for i in layer])
        voxel_strs = [layer_str + ';' + v for v in layer_vox.serialize()]

        dfs.append(pd.DataFrame({
            'unit': voxel_strs,
            'x':    bboxes[layer][0],
            'y':    bboxes[layer][1],
            'w':    bboxes[layer][2],
            'h':    bboxes[layer][3],
        }))
    full = pd.concat(dfs)
    full.to_csv(filename)


def load_bbox_csv(filename):
    df = pd.read_csv(filename)
    voxels = vx.VoxelIndex.from_serial(df['unit'])

    bboxes = {}
    layer_strs = map(df['unit'], lambda s: s.split(';')[1])
    for l_str, group in df.groupby(layer_strs):
        layer = tuple(int(i) for i in l_str.split('.'))
        # bboxes[layer] is expected to have shape ([left, top, w, h], n_vox)
        bboxes[layer] = group[['x', 'y', 'var', 'amp']].as_matrix()

    return bboxes



if __name__ == '__main__':

    model, ckpt = cornet.load_cornet("Z")
    inputs = backprop_fields.noise_video(11, 3, 256, 256, numpy = False)
    manager = nm.NetworkManager.assemble(model, inputs)

    voxel = {(0, 1, 0) : vx.VoxelIndex((0, 1, 0), ([1], [50], [30]))}
    _, grads = backprop_fields.gradients_raw_fullbatch(model, inputs, voxel)
    rfbb = trace_rf(sequential_leaves(manager, stop = (0, 1, 0)), manager)
    
    rfbb = (30 * rfbb[0][1] - rfbb[1][1], 50 * rfbb[0][0] - rfbb[1][0],
            rfbb[1][1] * 2 + 1, rfbb[1][0] * 2 + 1)
    backprop_fields.grad_plot(grads, [(0, 1, 0)], 0, 0, bbox = rfbb)



