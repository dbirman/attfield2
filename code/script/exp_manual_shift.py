import importlib.util, os
if os.environ.get('BLVNE', None) is not None:
    libscript="/Users/gru/proj/attfield/code/script/link_libs_blvne.py"
else:
    libscript="/Users/kaifox/GoogleDrive/attfield/code/script/link_libs_kfmbp.py"
spec = importlib.util.spec_from_file_location("link_libs", libscript)
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)


import proc.detection_task as det
import proc.network_manager as nm
import proc.attention_models as atts
import proc.experiment_tools as exp
import proc.attention_models as att
from proc import voxel_selection as vx
from proc.possible_fields import estimates_to_params
from proc import backprop_fields
from proc import lsq_fields
from proc import video_gen
from proc import cornet
import plot.rfs

from pprint import pprint
import itertools as iit
import pickle as pkl
import pandas as pd
import numpy as np
import torch
import tqdm
import time
import sys
import os
import gc


# ================================================= #
#   Parameters                                      #
# ================================================= #

# Experiment Parameters
UNIT_LAYERS = [(0,0,0), (0, 1, 0), (0, 2, 0), (0, 3, 0)]
BETAS = [11.0] #[1.1, 2.0, 4.0, 11.0]
# Attention settings to test
ATTS = {
    'ms-0': lambda b: {
        (0, 0, 0): atts.GaussianManualShiftAttention((56, 56), (56, 56), b)},
}
# What network layers perform decoding of embeddings to categories.
# These will usually be bypassed
DECODERS = [(0, 4, 2)]

# Speed/Quality Parameters
                                # Num units to test at each layer
UNIT_SET = '../100units_4layer.csv'
TRAIN_N = 100                   # Num training examples for regressions
TEST_N = 100                    # Num test images for gradients & performance
FIND_RFS = False                # Optionally receptive fields, just do qualitative
GRAD_APPROX_N = None            # Quality of gradient estimation (None for ideal)
ROTATION_FRAMES = 32            # Number of test images for gradients
                                # Limit on category set (speedup)
CAT_WHITELIST = None
GRAD_HEATMAPS = False           # Plot RF heatmaps? (speedup)
BATCH_SIZE = 10                 # Batch size for model runs

# Path parameters
DATA_OUT = Paths.data("exp/att_models/manual")
PLOT_OUT = Paths.plots("exp/att_models/manual")
ISO_PATH = Paths.data("imagenet/imagenet_iso224.h5")
FOUR_PATH = Paths.data('imagenet/imagenet_four224l0.h5')


# Instantiate the pyTorch model and select test units
model, ckpt = cornet.load_cornet("Z")
'''units = vx.random_voxels_for_model(
    model, UNIT_LAYERS, N_UNIT, 3, 224, 224)'''
# Or reload an old set of units
'''rf_fname = os.path.join(DATA_OUT, "rfs_noatt.csv")
units, _ = lsq_fields.load_rf_csv(rf_fname)'''
units = lsq_fields.load_units_from_csv(os.path.join(DATA_OUT, UNIT_SET))



# ================================================= #
#   Find RFs                                        #
# ================================================= #

if FIND_RFS:
    units = exp.gather_rfs(
        model, units, {}, BETAS,
        224, ROTATION_FRAMES, GRAD_APPROX_N,
        PLOT_OUT, DATA_OUT, GRAD_HEATMAPS)




# ================================================= #
#   Performance Evaluation                          #
# ================================================= #


'''rf_fname = os.path.join(DATA_OUT, "rfs_noatt.csv")
units, _ = lsq_fields.load_rf_csv(rf_fname)'''
exp.run_and_save_performance(
    model, units, ATTS, BETAS,
    ISO_PATH, CAT_WHITELIST, DECODERS, TRAIN_N,
    FOUR_PATH, TEST_N, UNIT_LAYERS, DATA_OUT,
    BATCH_SIZE,
    no_grads = True)




