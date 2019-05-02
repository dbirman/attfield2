import importlib.util
spec = importlib.util.spec_from_file_location("link_libs",
    "/Users/kaifox/projects/art_physio/code/script/link_libs_kfmbp.py")
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import cornet
from proc import detection_task as det

import itertools as iit
import pickle as pkl
import pandas as pd
import numpy as np
import tqdm
import os

import warnings
warnings.filterwarnings("ignore")



# Set parameters
TRAIN_NS = [2, 10, 30, 100]
MAX_VAL_N = 400
OUTPUT = Paths.data('exp/bhv_perf')
ATT_BETAS = [1.1, 1.5, 2., 11., 51., 101.]
ATT_LAYERS = [(0, 0), (0, 1), (0, 2), (0, 3)]
WHITELIST = ['bathtub', 'artichoke']


model, ckpt = cornet.load_cornet("Z")
decoders = [(0, 4, 2)]
isolated_task = det.IsolatedObjectDetectionTask(
    Paths.data.join('imagenet.h5'),
    whitelist = WHITELIST)
att_task = det.FourWayObjectDetectionTask(
    Paths.data.join('imagenet.h5'),
    whitelist = WHITELIST)
VAL_N = min(MAX_VAL_N, min(
    att_task.val_size(max(TRAIN_NS)),
    isolated_task.val_size(max(TRAIN_NS))))
print("Val size:", VAL_N)


val_imgs = {}
val_ys = {}
val_imgs['iso'], val_ys['iso'] = isolated_task.val_set(
    None, VAL_N, shuffle = False)
val_imgs['att'], val_ys['att'], val_locs = att_task.val_set(
    None, VAL_N, shuffle = False)
val_enc = {
    k: det.model_encodings(model, decoders, val_imgs[k])
    for k in val_imgs.keys()}

