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
from torch import nn
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
UNIT_LAYERS = [(0,),]
BETAS = [1.1, 2.0, 4.0, 11.0, 50.]
# Attention settings to test
ATTS = {
    'ms': lambda b: {
        (0, 0): atts.GaussianSensitivityGradAttention((56, 56), (56, 56), b)},
}

# Speed/Quality Parameters
                                # Units to test at each layer
N_UNIT = 100
UNIT_SET = '../att_models/100units_4layer.csv'
FIND_RFS = False                # Optionally receptive fields, just do qualitative
GRAD_APPROX_N = None            # Quality of gradient estimation (None for ideal)
ROTATION_FRAMES = 32            # Number of test images for gradients
                                # Limit on category set (speedup)
GRAD_HEATMAPS = False           # Plot RF heatmaps? (speedup)

# Path parameters
DATA_OUT = Paths.data("exp/sens_shift_val/")
PLOT_OUT = Paths.plots("exp/sens_shift_val/")


# Instantiate the pyTorch model and select test units

def EdgeDetector():
    conv = nn.Conv2d(1, 3, kernel_size = 5, padding = 2,
        stride = 1, bias = None)
    conv.weight.data.zero_()
    # Horizontal edges
    conv.weight.data[0, 0, :2, :] = -1
    conv.weight.data[0, 0, 3:, :] = 1
    # Vertical edges
    conv.weight.data[1, 0, :, :2] = -1
    conv.weight.data[1, 0, :, 3:] = 1
    # Curvature
    conv.weight.data[2, 0, :, :] = -1
    conv.weight.data[2, 0, 2, :] = 1
    conv.weight.data[2, 0, :, 2] = 1

    return nn.Sequential(conv)

def edge_images(size, wavelengths = [5, 10, 20], thresh = 0.9):
    R, C = np.meshgrid(np.arange(size), np.arange(size))
    f = 1 / np.array(wavelengths)[:, None, None]
    h_bars = np.sin(R[None, ...] * f) > thresh
    v_bars = np.sin(C[None, ...] * f) > thresh
    return torch.tensor((h_bars | v_bars)[:, None, ...].astype('float')).float()

model = EdgeDetector()
units = vx.random_voxels_for_model(
    model, UNIT_LAYERS, N_UNIT, 1, 224, 224)
# Or reload an old set of units
'''rf_fname = os.path.join(DATA_OUT, "rfs_noatt.csv")
units, _ = lsq_fields.load_rf_csv(rf_fname)'''
#units = lsq_fields.load_units_from_csv(os.path.join(DATA_OUT, UNIT_SET))




# ================================================= #
#   Find RFs                                        #
# ================================================= #


if FIND_RFS:
    units = exp.gather_rfs(
        model, units, ATTS, BETAS,
        224, ROTATION_FRAMES, GRAD_APPROX_N,
        PLOT_OUT, DATA_OUT, GRAD_HEATMAPS,
        N_CHANNEL = 1)



# ================================================= #
#   Extration of attention effects                  #
# ================================================= #

# Place to store results before dump
acts = {}

# Network rerpesentations with NO attention
print("Task with attention: [ None ]")
test_images = edge_images(224)
mgr = nm.NetworkManager.assemble(model, test_images,
    mods = {}, with_grad = False)
acts['ctrl'] = mgr.computed[(0,)]



# Network rerpesentations WITH attention
for (att_name, att_mod_gen), att_b in iit.product(ATTS.items(), BETAS):

    print(f"Task with attention: [ {att_name} : {att_b} ]")
    
    att_mod = att_mod_gen(att_b)
    mgr = nm.NetworkManager.assemble(model, test_images,
        mods = att_mod, with_grad = False)
    acts[(att_name, att_b)] = mgr.computed[(0,)]


pkl.dump(acts, open(os.path.join(DATA_OUT, 'acts.pkl'), 'wb'))







