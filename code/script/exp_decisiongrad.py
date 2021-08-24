import importlib.util
spec = importlib.util.spec_from_file_location("link_libs",
    "/Users/gru/proj/attfield/code/script/link_libs_blvne.py")
    #"/Users/kaifox/projects/art_physio/code/script/link_libs_kfmbp.py")
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

import proc.detection_task as det
import proc.network_manager as nm
import proc.attention_models as att
import proc.experiment_tools as exp
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





# Experiment Parameters
UNIT_LAYERS = [(0,0,0), (0,2,0)]
ATT_LAYERS = [(0, 0)]
BETAS = [11.0]#[1.1, 2.0, 4.0, 11.0]
ATT_MODES = {#'qg': det.QuadAttention,
             'fs': det.QuadAttentionFullShuf}
DECODERS = [(0, 4, 2)]

# Speed/Quality Parameters
N_UNIT = 500
TRAIN_N = 30
TEST_N = 100
GRAD_APPROX_N = None
ROTATION_FRAMES = 32
CAT_WHITELIST = None #['padlock', 'banana', 'stone wall', 'mortar']
GRAD_HEATMAPS = False

# Path parameters
DATA_OUT = Paths.data("exp/effect_shift_rel")
PLOT_OUT = Paths.plots("exp/effect_shift_rel")
ISO_PATH = Paths.data("imagenet/imagenet_iso224.h5")
FOUR_PATH = Paths.data('imagenet_four224l0.h5')


model, ckpt = cornet.load_cornet("Z")
'''units = vx.random_voxels_for_model(
    model, UNIT_LAYERS, N_UNIT, 3, 224, 224)'''
rf_fname = os.path.join(DATA_OUT, "rfs_noatt.csv")
units, _ = lsq_fields.load_rf_csv(rf_fname)








# ================================================= #
#   Find RFs                                        #
# ================================================= #


'''rf_imgs = video_gen.rf_slider_check(224,
    check_widths = [5, 10],
    speed = CHECK_SPEED)'''
rf_imgs = video_gen.sine_rotation(224, ROTATION_FRAMES,
    freqs = [5, 20], phase_rand = True)



# Unattended ('normal' or 'train-mode') receptive fields
print("RFs without attention")
_, raw_grads, raw_rfs = exp.get_rfs(rf_imgs, model, units, {},
    approx_n = GRAD_APPROX_N)

# Forget units with no gradient found
# (speeds up later 'null' operations dramatically for higher layers)
units = exp.discard_nan_rfs(units, raw_grads, raw_rfs)


# Output data & diagnostics
lsq_fields.save_rf_csv(os.path.join(DATA_OUT, "rfs_noatt.csv"),
        units, raw_rfs)
if GRAD_HEATMAP:
    plot.rfs.grad_heatmap(
        os.path.join(PLOT_OUT, 'rfs_noatt.pdf'),
        list(units.keys()), units, raw_grads)

# Clean up
del raw_grads, raw_rfs, _; gc.collect()



for (att_name, AttClass), att_l, att_b in (
    iit.product(ATT_MODES.items(), ATT_LAYERS, BETAS)):

    print(f"RFs with attention: [ {att_name} - {att_b} x {att_l} ]")
    _, att_grads, att_rfs = exp.get_rfs(rf_imgs, model, units, {
        att_l: AttClass(att_b, 0, profile = 'gauss')},
        approx_n = GRAD_APPROX_N)

    # Output data & diagnostics
    lstr = '-'.join(str(i) for i in att_l)
    fname = f"rfs_a{att_name}_b{att_b}_l{lstr}."
    lsq_fields.save_rf_csv(os.path.join(DATA_OUT, fname + "csv"),
            units, att_rfs)
    if GRAD_HEATMAP:
        plot.rfs.grad_heatmap(
            os.path.join(PLOT_OUT, fname + 'pdf'),
            list(units.keys()), units, att_grads)

# Clean up
del att_grads, att_rfs, _; gc.collect()











# ================================================= #
#   Unit Activations & Decision Gradients           #
# ================================================= #


rf_fname = os.path.join(DATA_OUT, "rfs_noatt.csv")
units, _ = lsq_fields.load_rf_csv(rf_fname)

# Fit logistic regressions
print("Fitting Regressions")
iso_task = det.DistilledDetectionTask(
    ISO_PATH, whitelist = CAT_WHITELIST)
_, _, regs, _, _ = det.fit_logregs(
    model, DECODERS, iso_task, train_size = TRAIN_N,
    shuffle = False)

# Unit gradients with NO attention
print("Unit effects without attention")
four_task = det.DistilledDetectionTask(
    FOUR_PATH, whitelist = CAT_WHITELIST)
val_imgs, val_ys, _ = four_task.val_set(None, TEST_N)
noatt_results = det.voxel_decision_grads(
    units, model, DECODERS, val_imgs, val_ys, regs)

# Generate a table of results for later calls to add to
df = exp.csv_template(units, four_task.cats, UNIT_LAYERS, TEST_N)
exp.append_results(noatt_results, df, "noatt")

# Clean up
del noatt_results, _; gc.collect()



# Unit gradients WITH attention
for (att_name, AttClass), att_l, att_b in (
    iit.product(ATT_MODES.items(), ATT_LAYERS, BETAS)):

    print(f"Unit effects with attention: [ {att_name} - {att_b} x {att_l} ]")
    
    att_results = det.voxel_decision_grads(
        units, model, DECODERS, val_imgs, val_ys, regs, mods = {
        att_l: AttClass(att_b, 0, profile = 'gauss')})

    # Merge data with other conditions
    lstr = '-'.join(str(i) for i in att_l)
    condition = f"a{att_name}_b{att_b}_l{lstr}"
    exp.append_results(att_results, df, condition)


    # Output data
    df.to_csv(os.path.join(DATA_OUT, 'att_bhv_fx_fs11.csv'), index = False, float_format = '%g')

# Clean up
del att_results, regs; gc.collect()






