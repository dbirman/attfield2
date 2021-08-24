import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

import tracemalloc
tracemalloc.start(25)

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
parser.add_argument('unit_path',
    help = 'Path to a CSV file defining the units to measure gradients of.')
parser.add_argument("wrt_layers", nargs = "+",
    help = 'List of layers to measure gradients with respect to.')
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
parser.add_argument('--regs', default = None,
    help = 'Optional file giving regressions to apply at the output of ' +
           'the network. If given, additional output will be generated ' +
           'giving the value of the decision function for each input.')
parser.add_argument("--cats", nargs = '*', default = [],
    help = 'If given, a whitelist for categories to train regressors for.')
parser.add_argument('--batch_size', type = float, default = -1,
    help = 'If given data will be run in batches.')
parser.add_argument('--model', type = str, default = None,
    help = 'Python file with a function `get_model` that returns a PyTorch'+
           'model for the script to run backprop on. If not provided, the '+
           'script will use CorNet-Z.')
parser.add_argument('--abs', action = 'store_true',
    help = 'Calculate the absolute rather than signed value of the gradient.')
parser.add_argument('--no_back', action = 'store_true',
    help = 'Don\'t run or save backprop data. Only auxillary features.')
parser.add_argument('--profile', action = 'store_true',
    help = 'Run the script via line_profiler for debugging, and '+
           'output stats to the given path.')
parser.add_argument('--cuda_profile', type = str, default = None,
    help = 'Run the script via torch.autograd.profile for debugging, and '+
           'output stats to the given path.')
parser.add_argument('--cuda', action = 'store_true',
    help = 'Force data and weight tensors to reside on GPU.')
parser.add_argument('--compress', action = 'store_true',
    help = 'Compress saved data with LZF. Takes much longer.')
parser.add_argument('--verbose', action = "store_true",
    help = 'Run with extra progress output.')
parser.add_argument('--nodata', action = "store_true",
    help = 'Run on fake data, for when data files aren\'t accessible.')
args = parser.parse_args()
args.decoders = [eval('tuple('+l+')') for l in args.decoders]
args.wrt_layers = [eval('tuple('+l+')') for l in args.wrt_layers]
if len(args.cats) == 0: args.cats = None




def cleanup():
    # Sometimes circular references are caught by
    # a double `collect` call
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()




# Enable profiling if requested
do_prof = False
if args.cuda_profile is not None: do_prof = True
th_prof = torch.autograd.profiler.profile(
    enabled = do_prof, use_cuda = args.cuda)
th_prof.__enter__()

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
    units = lsq_fields.load_units_from_csv(args.unit_path)
    if args.regs is not None:
        regs = det.load_logregs(args.regs)
    else:
        regs = None
    if args.nodata:
        four_task = det.FakeDetectionTask(args.cats, 224, 3)
    else:
        four_task = det.DistilledDetectionTask(
            args.comp_images_path, whitelist = args.cats)
    imgs, ys, _ = four_task.val_set(None, args.test_n)
    if args.attn is not None:
        kws = atts.load_cfg(args.attn_cfg)
        att_mods = atts.load_model(args.attn, **kws)
        print("loading attention", args.attn)
    else:
        att_mods = {}
        print("no attention", args.attn)
    print(att_mods)
    

    # Ensure model will be un on GPU if requested
    if args.cuda:
        model.cuda()

    # Don't treat categories separately
    cat_names = list(imgs.keys())
    cat_ids = np.concatenate([
        np.repeat(i_cat, len(imgs[c]))
        for i_cat, c in enumerate(cat_names)
    ])
    cat_img_ixs = [ # Within-category image counter
        np.cumsum(cat_ids == i) - 1
        for i in range(len(cat_names))]
    cat_ns = {c: len(imgs[c]) for c in imgs}
    ys = np.concatenate([a for a in ys.values()])
    imgs = torch.cat([a for a in imgs.values()])

    # Setup outputs
    # We have to wait to initialize the hdf5 arrays until we
    # know what size the wrt layers will be but can setup the
    # file pointer now.
    img_count = 0
    if not args.no_back:
        outputs = h5py.File(args.output_path, 'w')
    else:
        outputs = None
    if args.regs is not None:
        fn_output_fname = os.path.join(
            os.path.dirname(args.output_path),
            'fn_' + os.path.basename(args.output_path))
        fn_outputs = h5py.File(fn_output_fname, 'w')
        # Save image metadata along with behavioral outputs
        ids_ds = fn_outputs.create_dataset('cat_ids', (cat_ids.size,), np.int8)
        ids_ds[...] = cat_ids
        ids_ds.attrs['cat_names'] = np.array(cat_names).astype('S')
        ys_ds = fn_outputs.create_dataset('true_ys', (cat_ids.size,), np.int8)
        ys_ds[...] = ys
    else:
        fn_outputs = None

    # Adjust batch size to processor if requested:
    if args.batch_size == int(args.batch_size) or not args.cuda:
        args.batch_size = int(args.batch_size)
    else:
        tot_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        args.batch_size = int(args.batch_size * tot_mem)
        print(f"[bp] tot_mem {tot_mem} batch_size {args.batch_size}")

    # ======   Run backprop   ======

    batches = (np.arange(len(imgs)), )
    if args.batch_size > 0:
        if len(imgs) > args.batch_size:
            n_batch = np.ceil(len(imgs)/args.batch_size)
            batches = np.array_split(batches[0], n_batch)

    # NOTE: code assumes we'll still see images in expected order
    for batch_n, batch_ix in enumerate(batches):
        if args.verbose:
            print(f"Batch {batch_n+1} / {len(batches)}")

        run_batch(
            model, imgs, batch_ix, units, att_mods,
            fn_outputs, outputs, regs,
            cat_names, cat_ids, cat_img_ixs, cat_ns)
        cleanup()

    if not args.no_back:
        outputs.close()
    if args.regs is not None:
        fn_outputs.close()

        

@profile
def run_batch(
    model, imgs, batch_ix, units, att_mods,
    fn_outputs, outputs, regs,
    cat_names, cat_ids, cat_img_ixs, cat_ns):
    """Run gradients calculations for a batch and and them
    to the running totals."""
    
    # Run encodings with voxel gradients tracked
    if args.verbose: print(f"Running")
    decoder_bypass = {l: det.LayerBypass() for l in args.decoders}
    batch_imgs = imgs[batch_ix]

    # Ensure model will be un on GPU if requested
    if args.cuda:
        batch_imgs = batch_imgs.cuda()

    # Run the model
    mgr = nm.NetworkManager.assemble(model, batch_imgs,
        mods = nm.mod_merge(att_mods, decoder_bypass),
        with_grad = True,
        cuda = args.cuda)

    if not args.no_back:
        # Run backprop
        for i_layer, unit_layer in enumerate(units):
            print("Units in layer:", unit_layer)


            # Iterator over masks for the backprop algorithm
            # Each mask will be all zeros except for one element which is
            # set to 1, corresponding to the current unit whose gradients
            # we should measure.
            masks_iter = units[unit_layer].backward_masks_fullbatch(
                mgr, cuda = args.cuda)
            for i, (i_unit, unit_idx, mask) in enumerate(masks_iter):
                if args.verbose:
                    print(f"Unit: {str(i_unit+1)} / "
                        f" {str(units[unit_layer].nvox())}")
                sys.stdout.flush()
                sys.stderr.flush()

                with torch.no_grad():
                    backprop_for_unit(
                        model, mgr, batch_ix,
                        unit_layer, units, i_unit, mask, 
                        outputs, cat_ns)
                    cleanup()
                
            if args.verbose: print()

    # Store behavioral data if running regressions
    if args.regs is not None:
        decisions_to_file(
            fn_outputs, mgr, batch_ix, regs,
            cat_names, cat_ids, cat_img_ixs, cat_ns)

    cleanup()

    mgr.close_hooks()
    del mgr; cleanup()


@profile
def backprop_for_unit(
    model, mgr, batch_ix,
    unit_layer, units, i_unit, mask, 
    outputs, cat_ns):
    """For a single unit in a single batch, calculate all 
    required gradients and add them to running totals."""

    # Zero network gradients
    model.zero_grad()
    for t in mgr.computed:
        if hasattr(t, 'grad_nonleaf'):
            t.grad_nonleaf.zero_()

    # Run the backward pass
    mgr.computed[unit_layer].backward(mask, retain_graph = True)

    # # Create place to store the gradients as we're averaging
    avg_grads = {}
    out_keys = {}

    for wrt_layer in args.wrt_layers:

        unit_lstr = '.'.join(str(i) for i in unit_layer)
        wrt_lstr = '.'.join(str(i) for i in wrt_layer)
        key = f'grads_{unit_lstr}_{wrt_lstr}'
        out_keys[wrt_layer] = key

        # Get ready to update gradient averages
        curr_grad = mgr.grads[wrt_layer].detach()
        curr_grad = curr_grad.sum(dim = 0)

        # Update the gradient average
        curr_grad *= torch.full((), (1/(args.test_n * len(cat_ns))),
            device = curr_grad.device)
        if args.abs:
            curr_grad = torch.abs(curr_grad)
        if key not in avg_grads:
            avg_grads[key] = torch.zeros_like(curr_grad)
            # avg_grads[key] = torch.zeros(19)
        avg_grads[key] += curr_grad
        # del curr_grad; cleanup()

    # Write the calculated gradients to the output file
    grads_to_file(
        outputs, out_keys, avg_grads,
        units, unit_layer, i_unit)
    del avg_grads; cleanup()


    


@profile
def decisions_to_file(
    fn_outputs, mgr, batch_ix, regs,
    cat_names, cat_ids, cat_img_ixs, cat_ns):
    """Sort the encodings out by category, apply the given regressions,
    and write to the output file."""
    enc = mgr.computed[(0,)].cpu().detach().numpy()
    curr_ids = cat_ids[batch_ix]
    for c_id in np.unique(curr_ids):
        img_ixs = np.where(curr_ids == c_id)[0]
        true_ix = cat_img_ixs[c_id][batch_ix[img_ixs]]
        c = cat_names[c_id]
        fn = regs[c].decision_function(enc[img_ixs])
        if c not in fn_outputs.keys():
            fn_outputs.create_dataset(c, (cat_ns[c],))
        fn_outputs[c][list(true_ix)] = fn.ravel()


@profile
def grads_to_file(
    outputs, out_keys, avg_grads,
    units, unit_layer, i_unit):
    """Take the gradients calculated for a given unit and add them
    to the running total in the output file."""

    for key in out_keys.values():
        # Make sure we have a place to write the data to
        if key not in outputs.keys():
            total_units = units[unit_layer].nvox()
            dset_shape = (total_units,) + avg_grads[key].shape
            if args.compress:
                comp = dict(compression = 'lzf')
            else:
                comp = {}
            ds = outputs.create_dataset(key, dset_shape, **comp)
            ds[...] = 0
        detached = avg_grads[key].cpu().clone().detach().numpy()
        outputs[key][i_unit, ...] += detached



main()

th_prof.__exit__(None, None, None)
if do_prof:
    import pickle as pkl
    pkl.dump(th_prof, open(args.cuda_profile, 'wb'))
    th_prof.export_chrome_trace(args.cuda_profile +  '.chrome')

if args.profile:
    profile.print_stats()

print("Success. Exiting.")






