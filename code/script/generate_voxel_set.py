import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import voxel_selection as vx
from argparse import ArgumentParser

parser = ArgumentParser(
    description = 
        "Train logistic regressions on isolated object detection task.")
parser.add_argument('output_path',
    help = 'Path to an npz file where output should be stored.')
parser.add_argument("n_unit", type = int,
    help = 'Number of units to pull per layer.')
parser.add_argument('layers', nargs = '+',
    help = 'List of layers to pull units from, eg "(0,0,0)" "(0,2,0)"')
parser.add_argument("--channels", type = int,
    help = 'Number of channels in the input image. Default: 3')
parser.add_argument("--size", type = int,
    help = 'Size of the of input images (square). Default: 224')
parser.add_argument('--model_file', default = None,
    help = 'Optional python file with a function `get_model` returns a '+
           'pytorch model object.')
args = parser.parse_args()
args.layers = [eval('tuple('+l+')') for l in args.layers]

if args.model_file is None:
    from proc import cornet
    model, _ = cornet.load_cornet("Z")
else:
    exec(open(args.model_file, 'r').read())
    model = get_model()

units = vx.random_voxels_for_model(
    model, args.layers, args.n_unit, args.channels, args.size, args.size)
unit_strs = vx.VoxelIndex.serialize(units)
with open(args.output_path, 'w') as f:
    f.write('unit\n')
    for l in unit_strs:
        for s in unit_strs[l]:
            f.write(s); f.write('\n')



