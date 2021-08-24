import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import network_manager as nm
from proc import voxel_selection as vx

from argparse import ArgumentParser
import numpy as np
import torch


parser = ArgumentParser(
    description = 
        "Generate radially distributed sets of units.")
parser.add_argument('output_path',
    help = 'Path to an npz file where output should be stored.')
parser.add_argument("n_unit", type = int,
    help = 'Number of units to pull per layer.')
parser.add_argument('layers', nargs = '+',
    help = 'List of layers to pull units from, eg "(0,0,0)" "(0,2,0)"')
parser.add_argument('--loc', required = True, nargs = 2, type = float,
    help = 'Percentage-unit location of the center of the attention field.')
parser.add_argument("--channels", type = int,
    help = 'Number of channels in the input image. Default: 3')
parser.add_argument("--size", type = int,
    help = 'Size of the of input images (square). Default: 224')
parser.add_argument('--model_file', default = None,
    help = 'Optional python file with a function `get_model` returns a '+
           'pytorch model object.')
args = parser.parse_args()
args.layers = [eval('tuple('+l+')') for l in args.layers]




# ----------------------------- Functional breakdown ----

def local_coords(N, loc):
    rs_final, cs_final = 10 * np.ones(N), 10 * np.ones(N)
    gen_ix = np.arange(N)
    max_dist = np.array([loc[0], 1 - loc[0], loc[1], 1 - loc[1]]).max()
    n_iter = 1
    while len(gen_ix):
        print(f"Refinement 1; {len(gen_ix)} units invalid.")
        th = np.random.uniform(low = 0, high = 2 * np.pi, size = len(gen_ix))
        r = np.random.uniform(low = 0, high = max_dist, size = len(gen_ix))
        rs, cs = loc[0] + r * np.sin(th), loc[1] + r * np.cos(th)
        rs_final[gen_ix] = rs
        cs_final[gen_ix] = cs
        invalid = (rs < 0) | (rs > 1) | (cs < 0) | (cs > 1)
        gen_ix = gen_ix[invalid]
    return rs_final, cs_final


def quantize_to_shape(rs, cs, shape):
    return (
        np.digitize(rs, bins = np.linspace(0, 1 + 1e-7, shape[-2])),
        np.digitize(cs, bins = np.linspace(0, 1 + 1e-7, shape[-1])),
    )


def distribute_channels(rs, cs, n_channels):
    ixs = np.lexsort(np.stack([rs, cs]))
    chan = np.concatenate([
        np.random.permutation(n_channels)
        for _ in range(len(ixs) // n_channels + 1)
    ])[:len(ixs)]
    return np.stack([chan, rs[ixs], cs[ixs]])



# -------------------------------------- Load inputs ----

if args.model_file is None:
    from proc import cornet
    model, _ = cornet.load_cornet("Z")
else:
    exec(open(args.model_file, 'r').read())
    model = get_model()

# -------------------------------------- Generate indices ----

inputs = torch.zeros(1, args.channels, args.size, args.size)
mgr = nm.NetworkManager.assemble(model, inputs)

units = {}
for layer in args.layers:
    print(f"Generating voxels for layer: {layer}")
    shape = mgr.computed[layer].shape
    rs, cs = local_coords(args.n_unit, args.loc)
    rs, cs = quantize_to_shape(rs, cs, shape)
    idxs = distribute_channels(rs, cs, shape[-3])
    units[layer] = vx.VoxelIndex(layer, idxs)

# -------------------------------------- Write outputs ----

unit_strs = vx.VoxelIndex.serialize(units)
with open(args.output_path, 'w') as f:
    f.write('unit\n')
    for l in unit_strs:
        for s in unit_strs[l]:
            f.write(s); f.write('\n')




