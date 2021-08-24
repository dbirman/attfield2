"""
Run a model on a set of inputs and save its activations.
This should usually be done on small sets of images, or the output
files will become prohibitively large.
"""

import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import detection_task as det
from proc import attention_models as atts
from proc import network_manager as nm
from proc import cornet

from argparse import ArgumentParser
import numpy as np
import h5py
import sys
import os
import gc

parser = ArgumentParser(
    description = 
        "Extract full layer activations.")
parser.add_argument('output_path',
    help = 'Path to an HDF5 file where activations should be stored.')
parser.add_argument("image_gen",
    help = 'Path to a python script generating inputs.')
parser.add_argument('model', type = str, default = None,
    help = 'Python file with a function `get_model` that returns a PyTorch'+
           'model for the script to run backprop on. If not provided, the '+
           'script will use CorNet-Z.')
parser.add_argument("layers", nargs = "+",
    help = 'List of layers to pull encodings of.')
parser.add_argument("--attn", default = None,
    help = 'Path to a python file defining attention to apply. The '+
           '`attn_model` function defined in the file will be called '+
           'to instantiate a LayerMod implementing attention.')
parser.add_argument('--attn_cfg', default = None,
    help = 'Path to a JSON file to pass as kwargs to `attn_model()` '+
           'from the attention model file.')
parser.add_argument('--gen_cfg', default = None,
    help = 'Path to a JSON file to pass as kwargs to `generate_set()` '+
           'from the image gen file.')
parser.add_argument('--regs', default = None,
    help = 'Regression objects to run on each spatial location.')
parser.add_argument('--batch_size', type = int, default = -1,
    help = 'If given data will be run in batches.')
parser.add_argument('--max_feat', type = int, default = float('inf'),
    help = 'Max number of feautres to output from each layer.')
parser.add_argument('--cuda', action = 'store_true',
    help = 'Force data and weight tensors to reside on GPU.')
args = parser.parse_args()
args.layers = [eval('tuple('+l+')') for l in args.layers]


# -------------------------------------- Load inputs ----

# Load inputs
# Model:
if args.model is None:
    from proc import cornet
    model, _ = cornet.load_cornet("Z")
else:
    spec = importlib.util.spec_from_file_location(
        "model", args.model)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model = model_module.get_model()
# Image gen:
gen_kws = atts.load_cfg(args.gen_cfg)
# Image generator
spec = importlib.util.spec_from_file_location(
    "image_gen", args.image_gen)
image_gen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_gen_module)
generate_set = image_gen_module.generate_set
# Attention:
if args.attn is not None:
    kws = atts.load_cfg(args.attn_cfg)
    att_mods = atts.load_model(args.attn, **kws)
else:
    att_mods = {}
# Regressions
if args.regs is not None:
    regs = det.load_logregs(args.regs, bias = False)

# And prep output object
outputs = None
if args.regs is not None:
    fn_outputs = None


# ------------------------------------------------- Run model ----

meta, imgs = generate_set(**gen_kws)

for i_grp, ((grp_key, grp_meta), grp_imgs) in enumerate(zip(meta, imgs())):
    print("Group:", grp_key)

    batches = (np.arange(len(grp_imgs)), )
    if args.batch_size > 0:
        if len(grp_imgs) > args.batch_size:
            n_batches = len(grp_imgs)//args.batch_size
            batches = np.array_split(batches[0], n_batches)


    for batch_n, batch_ix in enumerate(batches):

        batch_imgs = grp_imgs[batch_ix]
        mgr = nm.NetworkManager.assemble(model, batch_imgs,
            mods = att_mods, with_grad = False, cuda = args.cuda)

        if outputs is None:
            outputs = h5py.File(args.output_path, 'w')
            for layer in args.layers:
                lstr = '.'.join(str(l) for l in layer)
                n_feat = min(mgr.computed[layer].shape[1], args.max_feat)
                outputs.create_dataset(lstr,
                    (len(meta), len(grp_imgs), n_feat) +
                     mgr.computed[layer].shape[2:],
                    np.float32)
            outputs.create_dataset('y', (len(meta), len(grp_imgs)), np.uint8)
            # Set up outputs of regressions
            if args.regs is not None:
                fn_outputs = h5py.File(
                    os.path.dirname(args.output_path) + 
                    '/fn_' + os.path.basename(args.output_path), 'w')
                for layer in args.layers:
                    lstr = '.'.join(str(l) for l in layer)
                    fn_outputs.create_dataset(lstr,
                        (len(meta), len(grp_imgs), len(regs)) +
                         mgr.computed[layer].shape[2:],
                        np.float32)

        outputs['y'][i_grp] = grp_meta['ys']
        for layer in args.layers:
            lstr = '.'.join(str(l) for l in layer)
            enc = mgr.computed[layer].detach().cpu()
            n_feat = min(enc.shape[1], args.max_feat)
            import matplotlib.pyplot as plt
            # if lstr == '0.1.3':
            #     plt.imshow(enc[0, 0]); plt.colorbar(); plt.title(i_grp); plt.show()
            outputs[lstr][i_grp, batch_ix.tolist()] = enc[:, :n_feat]

            if args.regs is not None:
                for i_c, c in enumerate(regs):
                    fns = np.apply_along_axis(
                        regs[c].decision_function,
                        1, enc
                    ).squeeze()
                    fn_outputs[lstr][i_grp, batch_ix.tolist(), i_c] = fns

outputs.close()












