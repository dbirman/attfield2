from proc import network_manager as nm

import torch
from torch import nn
from pprint import pprint
import numpy as np

def deep_repr(computed):
    if type(computed) == tuple or type(computed) == list:
        return type(computed)(deep_repr(x) for x in computed)
    elif issubclass(type(computed), torch.Tensor):
        return computed.size()
    elif issubclass(type(computed), nn.Module):
        return type(computed).__name__
    else:
        return computed

def deep_cat(computed_list):
    """
    [((conv, ~~), (block, [(pool, ~~), (relu, ~~)])),
     ((conv, ~~), (block, [(pool, ~~), (relu, ~~)]))]
    Each ~~ is a tensor and ~~'s that line up vertically should
    have the same size. This function recurses over the horizontal
    nested structure, at each recursive step passing a new list
    to the child call.
    """
    cltype = type(computed_list[0])
    if issubclass(cltype, nn.Module):
        # base case 1, modules flatten into just the module itself
        return computed_list[0]
    elif issubclass(cltype, torch.Tensor):
        # base case 2, tensors flatten by concatenation
        return torch.cat(computed_list)
    else:
        # recursive case
        # need to flatten over the first dimension of computed_list
        return cltype(deep_cat([x[i] for x in computed_list])
               for i in range(len(computed_list[0]))) 



class VoxelIndex():
    def __init__(self, layer, idx):
        self._layer = layer
        self._idx = tuple(idx)

    def __hash__(self):
        return hash(self._layer._idxs) + hash(self._idx)

    def index_into_computed(self, manager):
        tensor = manager.computed[self._layer]
        # Idx a tuple of arrays giving indexes for each dimension
        # Idx shape as ndarray: (ndim, nvox)
        # For this type of indexing, batch must be the last tensor dimension
        # `tensor` shape : (batch, ...)
        tensor_idxed = tensor.permute(*range(1, len(tensor.size())), 0)[self._idx]
        # `tensor_idxed` shape : (nvox, batch)
        tensor = tensor_idxed.permute(1, 0)
        # return `tensor` shape : (batch, nvox)
        return tensor

    def backward_masks(self, computed):
        masks = []
        tensor = self._layer.index_into_model(computed)[1]
        template = torch.zeros(tensor.size())
        idxs = np.array(self._idx).T
        for i in range(self.nvox()):
            for j in range(tensor.size()[0]):
                curr_idx = tuple(idxs[i].tolist())
                template.__setitem__((j,) + curr_idx, 1)
                yield (j, i), curr_idx, template
                template.__setitem__((j,) + curr_idx, 0)

    def backward_masks_fullbatch(self, manager):
        masks = []
        tensor = manager.computed[self._layer]
        template = torch.zeros(tensor.size())
        idxs = np.array(self._idx).T
        batch_slice = (slice(0, tensor.size()[0]),)
        for i in range(self.nvox()):
            curr_idx = tuple(idxs[i].tolist())
            template.__setitem__(batch_slice + curr_idx, 1)
            yield (i), curr_idx, template
            template.__setitem__(batch_slice + curr_idx, 0)

    def backward_mask(self, computed, i_vox):
        """Return a backward mask for the i_vox-th voxel"""
        tensor = self._layer.index_into_model(computed)[1]
        template = torch.zeros(tensor.size()[1:])
        idx = tuple(np.array(self._idx).T[i_vox].tolist())
        template[idx] = 1
        return idx, template

    def index_into_batch(self, batch, i_vox):
        """
        Return values of i-th voxel in this VoxelIndex from the given
        batch's data. Extract a timeseries of the voxel's response
        across the batch dimension.
        """
        ret = batch
        for dim in self._idx:
            ret = ret[:, dim[i_vox], ...]
        return ret

    def nvox(self):
        return len(self._idx[0])

    def serialize(self):
        ret = []
        idxs = np.array(self._idx).T
        for vox_idx in idxs:
            ret.append('.'.join([str(i) for i in vox_idx]))
        return ret

    @staticmethod
    def from_serial(voxels):
        by_layer = {}
        for v in voxels:
            layer = tuple(int(i) for i in v.split(';')[0].split('.'))
            if layer not in by_layer:
                by_layer[layer] = []
            idx = tuple(int(i) for i in v.split(';')[1].split('.'))
            by_layer[layer].append(idx)
        return {l: VoxelIndex(layer, np.array(i).T.tolist())
                for l,i in by_layer.items()}


def random_voxels(manager, layers, percent):

    use_nvox = (percent > 1) or (type(percent) == int)
    layer_voxels = {}
    for layer_idx in layers:
        module = manager.modules[layer_idx]
        tensor = manager.computed[layer_idx]
        nvox = (int(percent) if use_nvox else \
                int(np.prod(tensor.size()[1:]) * percent))
        # Select random indices for all dimensions except batch
        voxels = [np.random.randint(0, dim, size = nvox)
                  if nvox >= dim else
                  np.random.choice(dim, size = nvox, replace = False)
                  for dim in tensor.size()[1:]]
        layer_voxels[layer_idx] = VoxelIndex(layer_idx, tuple(voxels))
    return layer_voxels


def random_voxels_for_model(model, layers, percent,
                            channels, width, height):
    inputs = torch.ones(1, channels, height, width)
    mgr = nm.NetworkManager.assemble(model, inputs)
    return random_voxels(mgr, layers, percent)


