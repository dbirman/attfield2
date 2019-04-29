from proc import voxel_selection as vx
from proc import network_manager as nm
from proc import backprop_fields as bf
from proc import possible_fields
from proc import lsq_fields
from proc import video_gen
from proc import cornet
import plot.rfs



if __name__ == '__main__':

    # ---- Model & Inputs Setup ----

    model, ckpt = cornet.load_cornet("Z")
    layers = [(0, 0, 0), (0, 1, 0), (0, 2, 0)]
    frame = 64
    voxels = vx.random_voxels_for_model(
        model, layers, 4, 3, frame, frame)
    print("Generating video")
    inputs = {
        'check_slider': video_gen.rf_slider_check(frame,
            check_widths = [50, 30, 5],
            speed = 8),
        'color': video_gen.color_rotation(frame, 128),
        'orientation': video_gen.sine_rotation(frame, 64,
            freqs = [5, 20])
    }



    # ---- Compute Receptive fields, etc. ----

    results = {}
    for k, inp in inputs.items():
        manager = nm.NetworkManager.assemble(model, inp)
        bboxes = possible_fields.rf_bboxes(voxels, manager)

        print("\n[ Gradients: {} ]".format(k))
        _, grads = bf.gradients_raw_fullbatch(
            model, inp, voxels,
            approx_n = 30)
        params = bf.parameter_estimates_grads(grads, voxels)


        true_acts, pred_acts = lsq_fields.voxel_activity(
            voxels, manager, inp.numpy(), params)
        results[k] = {
            'grad': grads, 
            'params': params,
            'true_acts': true_acts,
            'pred_acts': pred_acts,
        }




    # ---- Saving & Plotting RF Diagnostics ----

    print("Saving and Plotting")

    for k, result in results.items():

        lsq_fields.save_rf_csv("data/rft_{}_params.csv".format(k),
            voxels, result['params'])
        plot.rfs.grad_heatmap('plots/rft_{}_grads.pdf'.format(k),
            layers, voxels, result['grad'], bboxes = bboxes)

        lsq_fields.save_activity_csv(
            'data/rft_{}_activity.csv'.format(k),
            voxels, result['true_acts'], result['pred_acts'])
        plot.rfs.activation_trials(
            'plots/rft_{}_activity.pdf'.format(k),
            voxels, result['true_acts'], result['pred_acts'])





