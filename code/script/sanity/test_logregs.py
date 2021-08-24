"""
Test logistic regressions on isolated object detection task,
generating standard ML performance metrics if the provided
dataset is the same as training.
"""

import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import detection_task as det
from proc import cornet
from argparse import ArgumentParser
import numpy as np
import h5py

parser = ArgumentParser(
    description = "Test logistic regressions on isolated"+
                  "object detection task.")
parser.add_argument('output_path',
    help = 'Path to an npz file where output should be stored.')
parser.add_argument("iso_images_path",
    help = 'Path to the HDF5 archive containing the input images.')
parser.add_argument('regs',
    help = 'File giving regressions to apply at the output of ' +
           'the network, the performance of which we are evaluating.')
parser.add_argument("train_n", type = int,
    help = 'Number of images regression was trained on (that '+
           'we should now test on).')
parser.add_argument("test_n", type = int,
    help = 'Number of new test/validation images to run trained'+
           ' regresion layers on.')
parser.add_argument("decoders", nargs = "+",
    help = 'Layer indices that perform decoding and should be '+
           'skipped when generating encodings. Given as '+
           'list of tuples, e.g. "(0, 4, 2)" for cornet-Z')
parser.add_argument("--cats", nargs = '*', default = [],
    help = 'If given, a whitelist for categories to train regressors for.')
parser.add_argument('--verbose', action = "store_true",
    help = 'Run with extra progress output.')
parser.add_argument('--nodata', action = 'store_true',
    help = 'Run without access to true data files. They\'re often top '+
           'large to live on the same machine as scripting is done on.')
parser.add_argument('--cuda', action = 'store_true',
    help = 'Force data and weight tensors to reside on GPU.')
parser.add_argument('--model', type = str, default = None,
    help = 'Python file with a function `get_model` that returns a PyTorch'+
           'model for the script to run backprop on. If not provided, the '+
           'script will use CorNet-Z.')
args = parser.parse_args()
args.decoders = [eval('tuple('+l+')') for l in args.decoders]
if len(args.cats) == 0: args.cats = None


# Set up / load inputs
if args.model is None:
    model, _ = cornet.load_cornet("Z")
else:
    spec = importlib.util.spec_from_file_location(
        "model", args.model)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model = model_module.get_model()
if args.cuda:
    model.cuda()
if args.nodata:
    iso_task = det.FakeDetectionTask(args.cats, 3, 224)
else:
    iso_task = det.DistilledDetectionTask(
       args.iso_images_path, whitelist = args.cats)
regs = det.load_logregs(args.regs)


# Set up / load ouputs
all_cats = [c for c in regs if c in iso_task.cats]
print("Outputting to:", args.output_path) 
outputs = h5py.File(args.output_path, 'w')
for name, dtype in [
        ('ys', 'bool'), ('preds', 'bool'), ('fn', 'float'),
        ('cat', 'uint8')]:
    outputs.create_dataset(f'{name}_train', (args.train_n * len(all_cats),),
        dtype = dtype, compression = 'gzip', compression_opts = 9,
        shuffle = True)
    outputs.create_dataset(f'{name}_test',  (args.test_n * len(all_cats),),
        dtype = dtype, compression = 'gzip', compression_opts = 9,
        shuffle = True)
outputs['cat_train'].attrs['cat_names'] = np.array(all_cats).astype('S')
outputs['cat_test'].attrs['cat_names']  = np.array(all_cats).astype('S')


# Run for each category
for i_cat, cat in enumerate(all_cats):

    # Run on training data
    for cond, N, img_fn in [
            ('test',  args.test_n,  iso_task.val_set),
            ('train', args.train_n, iso_task.train_set)]:

        if args.verbose:
            print(f"Category: {cat}. Image set: {cond}")

        # Load and run the network on the images
        imgs, ys, _ = img_fn(cat, N)
        if args.cuda:
            imgs.cuda()
        enc = det.model_encodings(model, args.decoders, imgs,
            cuda = args.cuda)

        write_to = slice(i_cat * N, (i_cat + 1) * N)
        outputs[f'preds_{cond}'][write_to] = regs[cat].predict(enc)
        outputs[f'fn_{cond}'][write_to] = regs[cat].decision_function(enc)
        outputs[f'ys_{cond}'][write_to] = ys
        outputs[f'cat_{cond}'][write_to] = i_cat


outputs.close()









