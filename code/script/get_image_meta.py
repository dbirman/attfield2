"""
Run a model on a set of inputs and save its activations.
This should usually be done on small sets of images, or the output
files will become prohibitively large.
"""

import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import attention_models as atts

from argparse import ArgumentParser
import pickle as pkl

parser = ArgumentParser(
    description = 
        "Extract metadata from image generator.")
parser.add_argument('output_path',
    help = 'Path to pickle file where metadata should be stored.')
parser.add_argument("image_gen",
    help = 'Path to a python script generating inputs.')
parser.add_argument('--gen_cfg', default = None,
    help = 'Path to a JSON file to pass as kwargs to `generate_set()` '+
           'from the image gen file.')
args = parser.parse_args()


# -------------------------------------- Load inputs ----

# Image gen:
gen_kws = atts.load_cfg(args.gen_cfg)
spec = importlib.util.spec_from_file_location(
    "image_gen", args.image_gen)
image_gen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_gen_module)
generate_set = image_gen_module.generate_set


# ---------------------------------- Run Generator and output ----

meta, imgs = generate_set(**gen_kws)

pkl.dump([
    (key, imgs.shape, meta)
    for ((key, meta), imgs) in zip(meta, imgs())],
    open(args.output_path, 'wb'))











