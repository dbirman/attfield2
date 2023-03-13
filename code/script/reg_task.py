import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import attention_models as atts
from proc import network_manager as nm
from proc import detection_task as det
from proc import lsq_fields
from proc import cornet
from argparse import ArgumentParser
from functools import reduce
import operator as op
import traceback
import numpy as np
import torch
import h5py
import json
import sys
import os
import gc


parser = ArgumentParser(
    description = 
        "Run backpropagation from one set of layers to another.")
parser.add_argument('output_path',
    help = 'Path to an HDF5 file where output should be stored.')
parser.add_argument("comp_images_path",
    help = 'Path to the HDF5 archive containing the input images.')
parser.add_argument("test_n", type = int,
    help = 'Number of images to train regresions on.')
parser.add_argument('regs', default = None,
    help = 'Optional file giving regressions to apply at the output of ' +
           'the network. If given, additional output will be generated ' +
           'giving the value of the decision function for each input.')
parser.add_argument("--attn", default = None,
    help = 'Path to a python file defining attention to apply. The '+
           '`attn_model` function defined in the file will be called '+
           'to instantiate a LayerMod implementing attention.')
parser.add_argument('--attn_cfg', default = None,
    help = 'Path to a JSON file to pass as kwargs to `attn_model()` '+
           'from the attention model file.')
parser.add_argument("--decoders", nargs = "*",
    help = 'Optional layer indices that perform decoding and should be '+
           'skipped when generating encodings. Given as '+
           'list of tuples, e.g. "(0, 4, 2)" for cornet-Z')
parser.add_argument('--batch_size', type = int, default = -1,
    help = 'If given data will be run in batches.')
parser.add_argument("--cats", nargs = '*', default = [],
    help = 'If given, a whitelist for categories to train regressors for.')
parser.add_argument('--model', type = str, default = None,
    help = 'Python file with a function `get_model` that returns a PyTorch'+
           'model for the script to run backprop on. If not provided, the '+
           'script will use CorNet-Z.')
parser.add_argument('--profile', action = 'store_true',
    help = 'Run the script via line_profiler for debugging, and '+
           'output stats to the given path.')
parser.add_argument('--cuda', action = 'store_true',
    help = 'Force data and weight tensors to reside on GPU.')
parser.add_argument('--nodata', action = "store_true",
    help = 'Run on fake data, for when data files aren\'t accessible.')
args = parser.parse_args()
args.decoders = [eval('tuple('+l+')') for l in args.decoders]
if len(args.cats) == 0: args.cats = None



def cleanup():
    # Sometimes circular references are caught by
    # a double `collect` call
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()





# Enable profiling if requested

if args.profile:
    from line_profiler import LineProfiler
    profile = LineProfiler()
else:
    profile = lambda f: f

@profile
def main():
    # Load input files 
    if args.model is None:
        from proc import cornet
        model, _ = cornet.load_cornet("Z")
    else:
        spec = importlib.util.spec_from_file_location(
            "model", args.model)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        model = model_module.get_model()
    if args.regs is not None:
        regs = det.load_logregs(args.regs)
    if args.nodata:
        four_task = det.FakeDetectionTask(args.cats, 224, 3)
    else:
        four_task = det.DistilledDetectionTask(
            args.comp_images_path, whitelist = args.cats)
    imgs, ys, _, img_ixs = four_task.val_set(None, args.test_n)
    if args.attn is not None:
        kws = atts.load_cfg(args.attn_cfg)
        att_mods = atts.load_model(args.attn, **kws)
        print("loading attention", args.attn)
    else:
        att_mods = {}
        print("no attention", args.attn)
    

    # Ensure model will be un on GPU if requested
    if args.cuda:
        model.cuda()

    # Don't treat categories separately

    # Setup outputs
    fn_outputs = h5py.File(args.output_path, 'w')

    # ======   Run behavior task   ======

    for c in four_task.cats:

        print("Category:", c)

        # Save true outputs so we don't have to load the dataset later
        ys_ds = fn_outputs.create_dataset(f'{c}_y', (len(imgs[c]),), np.int8)
        ys_ds[...] = ys[c]
        ys_ds.attrs['cat_names'] = np.array(list(regs.keys())).astype('S')
        # record index of image in the dataset
        ix_ds = fn_outputs.create_dataset(f'{c}_ix', (len(imgs[c]),), np.uint32)
        ix_ds[...] = img_ixs[c]

        # Create a place to store the eventual scores
        fn_ds = fn_outputs.create_dataset(f'{c}_fn', (len(imgs[c]),), np.float32)

        batches = (np.arange(len(imgs[c])), )
        if args.batch_size > 0:
            if len(imgs[c]) > args.batch_size:
                n_batch = np.ceil(len(imgs[c])/args.batch_size)
                batches = np.array_split(batches[0], n_batch)

        # NOTE: code assumes we'll still see images in expected order
        for batch_n, batch_ix in enumerate(batches):
            print(f"Batch {batch_n+1} / {len(batches)}")

            run_batch(
                model, imgs[c], batch_ix, att_mods,
                fn_ds, regs[c])
            cleanup()

    fn_outputs.close()

@profile
def run_batch(
    model, imgs, batch_ix, att_mods,
    fn_ds, reg):
    """Run gradients calculations for a batch and and them
    to the running totals."""
    
    # Run encodings with voxel gradients tracked
    decoder_bypass = {l: det.LayerBypass() for l in args.decoders}
    batch_imgs = imgs[batch_ix]

    # Ensure model will be un on GPU if requested
    if args.cuda:
        batch_imgs = batch_imgs.cuda()

    # import matplotlib.pyplot as plt
    # print(batch_imgs[0].min(), batch_imgs[1].max())
    # plt.imshow(np.transpose(batch_imgs[0], [1,2,0]) / 255)
    # plt.show()

    # Run the model
    mgr = nm.NetworkManager.assemble(model, batch_imgs,
        mods = nm.mod_merge(att_mods, decoder_bypass),
        with_grad = True,
        cuda = args.cuda)

    enc = mgr.computed[(0,)].cpu().detach().numpy()
    fn = reg.decision_function(enc)
    fn_ds[list(batch_ix)] = fn


    # np.savez('mgr.npz',
    #    **{str(k): t.detach() for k, t in mgr.computed.items()})
    # exit()

    mgr.close_hooks()
    del mgr; cleanup()

main()


if args.profile:
    profile.print_stats()

print("Success. Exiting.")






