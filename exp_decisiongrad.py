import importlib.util
spec = importlib.util.spec_from_file_location("link_libs",
    "/Users/gru/proj/attfield/code/script/link_libs_blvne.py")
    #"/Users/kaifox/projects/art_physio/code/script/link_libs_kfmbp.py")
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

import proc.detection_task as det
import proc.network_manager as nm
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





# Experiment Parameters
UNIT_LAYERS = [(0,0,0), (0,2,0)]
ATT_LAYERS = [(0, 0)]
BETAS = [1.1, 2.0, 11.0]
ATT_MODES = {'qg': det.QuadAttention,
             'fs': det.QuadAttentionFullShuf}
DECODERS = [(0, 4, 2)]

# Speed/Quality Parameters
N_UNIT = 1000
TRAIN_N = 30
TEST_N = 10
GRAD_APPROX_N = 50
CHECK_SPEED = 2

# Path parameters
DATA_OUT = Paths.data("exp/effect_shift_rel")
PLOT_OUT = Paths.plots("exp/effect_shift_rel")
ISO_PATH = Paths.data("imagenet/imagenet_iso224.h5")
FOUR_PATH = Paths.data('imagenet_four224l0.h5')
CAT_WHITELIST = ['padlock', 'banana', 'stone wall', 'mortar']


model, ckpt = cornet.load_cornet("Z")

'''
units = vx.random_voxels_for_model(
    model, UNIT_LAYERS, N_UNIT, 3, 224, 224)
'''
rf_fname = os.path.join(DATA_OUT, "rfs_noatt.csv")
units, _ = lsq_fields.load_rf_csv(rf_fname)





# lsq_fields.save_rf_csv does our job for us here

def save_bhv_grads(filename, units, grads):
    cats = list(grads.keys())
    layers = list(grads[cats[0]].keys())
    n_img = len(grads[cats[0]][layers[0]])

    # Arrange other column data into a flattened version of `grads`
    unit = vx.VoxelIndex.serialize(units)
    unit_values = np.concatenate([
        np.tile(np.array(unit[l])[np.newaxis, :], [n_img, 1]).ravel()
        for l in layers for c in cats
    ])
    img_values = np.concatenate([
        np.tile(np.arange(n_img)[:, np.newaxis], [1, len(unit[l])]).ravel()
        for l in layers for c in cats
    ])
    cat_values = np.concatenate([
        np.full([n_img, len(unit[l])], c).ravel()
        for l in layers for c in cats
    ])
    grad_values = np.concatenate([
        grads[c][l].ravel()
        for l in layers for c in cats
    ])
    
    df = pd.DataFrame(dict(
            unit = unit_values, img = img_values,
            cat = cat_values, value = grad_values
    ))
    df.to_csv(filename, index = False)


def get_rfs(inputs, voxels, mods, approx_n = GRAD_APPROX_N):
    manager, grads = backprop_fields.gradients_raw_fullbatch(
        model, inputs, voxels, mods = mods,
        approx_n = approx_n)
    estimated = backprop_fields.parameter_estimates_grads(grads, voxels)
    params = estimates_to_params(estimated)
    return manager, grads, params







# ================================================= #
#   Find RFs                                        #
# ================================================= #


rf_imgs = video_gen.rf_slider_check(224,
    check_widths = [5, 10],
    speed = CHECK_SPEED)




'''
# Unattended ('normal' or 'train-mode') receptive fields
print("RFs without attention")
_, raw_grads, raw_rfs = get_rfs(rf_imgs, units, {})

# Forget units with no gradient found
# (speeds up later operations dramatically for higher layers)
new_units = {}
for layer, layer_vox in units.items():
    nans = np.array(raw_rfs[layer]).T[3]
    new_units[layer] = vx.VoxelIndex(layer,
        [idxs[~np.isnan(nans)] for idxs in layer_vox._idx])
    raw_grads[layer] = raw_grads[layer][~np.isnan(nans)]
    raw_rfs[layer] = np.array(raw_rfs[layer])[~np.isnan(nans)]
    
# Reset the units object since it couldn't change during iteration
units = new_units
if len(units) == 0:
    print("No units with receptive fields found")
    exit()


# Output data & diagnostics
lsq_fields.save_rf_csv(os.path.join(DATA_OUT, "rfs_noatt.csv"),
        units, raw_rfs)
plot.rfs.grad_heatmap(
    os.path.join(PLOT_OUT, 'rfs_noatt.pdf'),
    list(units.keys()), units, raw_grads)

# Clean up
del raw_grads, raw_rfs, _, nans; gc.collect()





for (att_name, AttClass), att_l, att_b in (
    iit.product(ATT_MODES.items(), ATT_LAYERS, BETAS)):

    print(f"RFs with attention: [ {att_name} - {att_b} x {att_l} ]")
    _, att_grads, att_rfs = get_rfs(rf_imgs, units, {
        att_l: AttClass(att_b, 0, profile = 'gauss')})

    # Output data & diagnostics
    lstr = '-'.join(str(i) for i in l)
    fname = f"rfs_a{att_name}_b{att_b}_l{lstr}."
    lsq_fields.save_rf_csv(os.path.join(DATA_OUT, fname + "csv"),
            units, att_rfs)
    plot.rfs.grad_heatmap(
        os.path.join(PLOT_OUT, fname + 'pdf'),
        list(units.keys()), units, att_grads)

# Clean up
del att_grads, att_rfs, _; gc.collect()
'''











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
ugrads, uacts = det.voxel_decision_grads(
    units, model, DECODERS, val_imgs, val_ys, regs)

# Output data
fname = os.path.join(DATA_OUT, '{}_noatt.csv')
save_bhv_grads(fname.format('ugrads'), units, ugrads)
save_bhv_grads(fname.format('uacts'), units, ugrads)

# Clean up
del ugrads, _; gc.collect()



# Unit gradients WITH attention
for (att_name, AttClass), att_l, att_b in (
    iit.product(ATT_MODES.items(), ATT_LAYERS, BETAS)):

    print(f"Unit effects with attention: [ {att_name} - {att_b} x {att_l} ]")
    
    ugrads, uacts = det.voxel_decision_grads(
        units, model, DECODERS, val_imgs, val_ys, regs, mods = {
        att_l: AttClass(att_b, 0, profile = 'gauss')})

    # Output data
    lstr = '-'.join(str(i) for i in att_l)
    fname = f"_a{att_name}_b{att_b}_l{lstr}.csv"
    save_bhv_grads(
        os.path.join(DATA_OUT, 'ugrads' + fname),
        units, ugrads)
    save_bhv_grads(
        os.path.join(DATA_OUT, 'uacts' + fname),
        units, uacts)

# Clean up
del ugrads, uacts, regs; gc.collect()






