# As we develop a standard set of operations to be performed during
# they may be moved to file to become a piece of shared code.


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

import itertools as iit
import pickle as pkl
import pandas as pd
import numpy as np
import torch
import os
import gc


# ============================================================================
# -----------------------------------------------------------------  IO  -----
# ============================================================================


def csv_template(units, cats, layers, n_img):
    '''
    Form the structure of a DataFrame recording information about
    various units under different conditions. (wide form)
    '''

    # Arrange other column data into a flattened version of `values`
    unit = vx.VoxelIndex.serialize(units)
    unit_values = np.concatenate([
        np.tile(np.array(unit[l])[np.newaxis, :], [n_img, 1]).ravel()
        for l in layers for c in cats
    ])
    unit_layer_values = np.concatenate([
        np.full([n_img, len(unit[l])], ".".join(str(i) for i in l)).ravel()
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
    df = pd.DataFrame(dict(
            unit = unit_values, unit_layer = unit_layer_values,
            img = img_values, cat = cat_values,
    ))
    return df


def append_results(results, df, condition):
    """
    Append (columns) of results obtained by testing under a certain condition
    to the template DataFrame from `csv_template`.
    """

    grads, acts, fn, pred, ys, score_cat, score_ovrl = results
    cats = list(acts.keys())
    layers = list(acts[cats[0]].keys())
    n_img = acts[cats[0]][layers[0]].shape[0]
    n_unit = {l: acts[cats[0]][l].shape[1] for l in layers}

    if grads != None:
        df['grad_' + condition] = np.concatenate([
            grads[c][l].ravel() for l in layers for c in cats
        ])
    df['act_' + condition] = np.concatenate([
        acts[c][l].ravel() for l in layers for c in cats
    ])
    df['fn_' + condition] = np.concatenate([
        np.tile(fn[c][:, np.newaxis], [1, n_unit[l]]).ravel()
        for l in layers for c in cats
    ])
    df['pred_' + condition] = np.concatenate([
        np.tile(pred[c][:, np.newaxis], [1, n_unit[l]]).ravel()
        for l in layers for c in cats
    ])
    df['gt_' + condition] = np.concatenate([
        np.tile(ys[c][:, np.newaxis], [1, n_unit[l]]).ravel()
        for l in layers for c in cats
    ])
    df['cscore_' + condition] = np.concatenate([
        np.full([n_img, n_unit[l]], score_cat[c]).ravel()
        for l in layers for c in cats
    ])
    df['oscore_' + condition] = np.concatenate([
        np.full([n_img, n_unit[l]], score_ovrl).ravel()
        for l in layers for c in cats
    ])


def append_results_intermediate(df, condition):
    def callback(c, batch_ix, grads, acts, fn, pred, ys):
        print("Saving data.")

        # Make sure that columns are present
        for col in ('grad_', 'act_', 'fn_', 'pred_', 'gt_'):
            if not (col + condition) in df.columns:
                df[col + condition] = np.nan

        # Figure out where in the dataframe we should be inserting data
        insert_mask = (df['cat'] == c) & (df['img'].isin(batch_ix))

        layers = list(acts.keys())
        n_unit = {l: acts[l].shape[1] for l in layers}
        if grads != None:
            df.loc[insert_mask, 'grad_' + condition] = np.concatenate([
                grads[l].ravel() for l in layers
            ])
        df.loc[insert_mask, 'act_' + condition] = np.concatenate([
            acts[l].ravel() for l in layers
        ])
        df.loc[insert_mask, 'fn_' + condition] = np.concatenate([
            np.tile(fn[:, np.newaxis], [1, n_unit[l]]).ravel()
            for l in layers
        ])
        df.loc[insert_mask, 'pred_' + condition] = np.concatenate([
            np.tile(pred[:, np.newaxis], [1, n_unit[l]]).ravel()
            for l in layers
        ])
        df.loc[insert_mask, 'gt_' + condition] = np.concatenate([
            np.tile(ys[:, np.newaxis], [1, n_unit[l]]).ravel()
            for l in layers
        ])
    return callback




# ============================================================================
# ----------------------------------------------------  Model Execution  -----
# ============================================================================


def get_rfs(inputs, model, voxels, mods, approx_n = None):
    manager, grads = backprop_fields.gradients_raw_fullbatch(
        model, inputs, voxels, mods = mods,
        approx_n = approx_n)
    estimated = backprop_fields.parameter_estimates_grads(grads, voxels)
    params = estimates_to_params(estimated)
    return manager, grads, params


def discard_nan_rfs(units, raw_grads, raw_rfs):
    '''
    Discard units from the experiment for which no gradient was found.
    Note: updates raw_grads and raw_rfs, but returns new VoxelIndex `units`
    '''
    new_units = {}
    for layer, layer_vox in units.items():
        nans = np.array(raw_rfs[layer]).T[3]
        new_units[layer] = vx.VoxelIndex(layer,
            [idxs[~np.isnan(nans)] for idxs in layer_vox._idx])
        raw_grads[layer] = raw_grads[layer][~np.isnan(nans)]
        raw_rfs[layer] = np.array(raw_rfs[layer])[~np.isnan(nans)]
        
    # Reset the units object since it couldn't change during iteration
    if all(len(units[l]._idx[0]) == 0 for l in units):
        print("No units with receptive fields found")
        exit()
    return new_units




def gather_rfs(
    model, units, att_gens, BETAS,
    INPUT_SIZE, ROTATION_FRAMES, GRAD_APPROX_N,
    PLOT_OUT, DATA_OUT, GRAD_HEATMAPS, N_CHANNEL = 3):
    '''
    Measure receptive fields of a model under varied attention conditions.

    ### Returns
    - `units` --- An updated set of units, having dropped any
        with no detected receptive field.
    '''

    '''rf_imgs = video_gen.rf_slider_check(224,
    check_widths = [5, 10],
    speed = CHECK_SPEED)'''
    rf_imgs = video_gen.sine_rotation(INPUT_SIZE, ROTATION_FRAMES,
        freqs = [5, 20], phase_rand = True)
    if N_CHANNEL <= 3: rf_imgs = rf_imgs[:, :N_CHANNEL, :, :]
    else: raise NotImplementedError(
        "Sine-rotation RFs for model with >3 input channels.")


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
    if GRAD_HEATMAPS:
        plot.rfs.grad_heatmap(
            os.path.join(PLOT_OUT, 'rfs_noatt.pdf'),
            list(units.keys()), units, raw_grads)
    grad_outfile = os.path.join(DATA_OUT, "rfs_noatt.pkl")
    with open(grad_outfile, 'wb') as f:
        pkl.dump(raw_grads, f)

    # Clean up
    del raw_grads, raw_rfs, _; gc.collect()


    for (att_name, att_mod_gen), att_b in iit.product(att_gens.items(), BETAS):
        # Instantiate the network mod from generator lambda
        # taking only amplitude
        att_mod = att_mod_gen(att_b)

        print(f"RFs with attention: [ {att_name} : {att_b} ]")
        _, att_grads, att_rfs = exp.get_rfs(rf_imgs, model, units, att_mod,
            approx_n = GRAD_APPROX_N)

        # Output data & diagnostics
        fname = f"rfs_a{att_name}_b{att_b}."
        lsq_fields.save_rf_csv(os.path.join(DATA_OUT, fname + "csv"),
                units, att_rfs)
        if GRAD_HEATMAPS:
            plot.rfs.grad_heatmap(
                os.path.join(PLOT_OUT, fname + 'pdf'),
                list(units.keys()), units, att_grads)
        grad_outfile = os.path.join(DATA_OUT, fname + "pkl")
        with open(grad_outfile, 'wb') as f:
            pkl.dump(att_grads, f)

    return units




def run_and_save_performance(
    model, units, 
    att_gens, BETAS,
    ISO_PATH, CAT_WHITELIST, DECODERS, TRAIN_N, 
    FOUR_PATH, TEST_N, UNIT_LAYERS, DATA_OUT,
    BATCH_SIZE,
    no_grads = False,
    do_noatt = False):

    # Fit logistic regressions
    print("Fitting Regressions")
    iso_task = det.DistilledDetectionTask(
        ISO_PATH, whitelist = CAT_WHITELIST)
    #iso_task = det.FakeDetectionTask(CAT_WHITELIST, 3, 224)
    _, _, regs, _, _ = det.fit_logregs(
        model, DECODERS, iso_task, train_size = TRAIN_N,
        shuffle = False)
    
    four_task = det.DistilledDetectionTask(
        FOUR_PATH, whitelist = CAT_WHITELIST)
    #four_task = det.FakeDetectionTask(CAT_WHITELIST, 3, 224)
    val_imgs, val_ys, _ = four_task.val_set(None, TEST_N)

    # Unit gradients with NO attention
    if do_noatt:
        print("Task with attention: [ None ]")
        noatt_results = det.voxel_decision_grads(
            units, model, DECODERS, val_imgs, val_ys, regs,
            no_grads = no_grads,
            batch_size = BATCH_SIZE)

        # Generate a table of results for later calls to add to
        df = exp.csv_template(units, four_task.cats, UNIT_LAYERS, TEST_N)
        exp.append_results(noatt_results, df, "noatt")
        df.to_csv(
            os.path.join(DATA_OUT, 'bhv_noatt.csv'),
            index = False, float_format = '%g')

        # Clean up
        del noatt_results, _; gc.collect()



    # Unit gradients WITH attention
    for (att_name, att_mod_gen), att_b in (iit.product(att_gens.items(), BETAS)):

        print(f"Task with attention: [ {att_name} : {att_b} ]")
        
        # Parse out the attention model into a dict of network mods
        # and an effective 'layer'. The requisite layer is mainly for
        # backwards compatibility.
        att_mod = att_mod_gen(att_b)

        # Set up data saving callback
        condition = f"a{att_name}_b{att_b}"
        df = exp.csv_template(units, four_task.cats, UNIT_LAYERS, TEST_N)
        data_cb = append_results_intermediate(df, condition)

        att_results = det.voxel_decision_grads(
            units, model, DECODERS, val_imgs, val_ys, regs,
            mods = att_mod,
            no_grads = no_grads,
            batch_size = BATCH_SIZE,
            batch_callback = data_cb)

        # Merge data with other conditions
        df.to_csv(
            os.path.join(DATA_OUT, f'bhv_{condition}.csv'),
            index = False, float_format = '%g')

        # Clean up
        del att_results; gc.collect()








