from proc import voxel_selection as vx
from proc import network_manager as nm
from proc import backprop_fields
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
        model, layers, 30, 3, frame, frame)
    print("Generating video")
    inputs = video_gen.rf_slider_check(frame,
        check_widths = [50, 30, 5],
        speed = 8)



    # ---- Compute Receptive fields, etc. ----

    manager = nm.NetworkManager.assemble(model, inputs)
    print("Bounding boxes")
    bboxes = possible_fields.rf_bboxes(voxels, manager)
    estimated_bbox = possible_fields.parameter_estimates_bbox(bboxes, voxels)
   
    print("Gradients")
    _, grads = backprop_fields.gradients_raw_fullbatch(
        model, inputs, voxels,
        approx_n = 30)
    estimated_grad = backprop_fields.parameter_estimates_grads(grads, voxels)
    refined, _ = lsq_fields.fit_rfs(
        inputs.numpy(), manager, voxels, estimated_grad,
        verbose = 1)

    true_acts, pred_acts = lsq_fields.voxel_activity(
        voxels, manager, inputs.numpy(), refined)



    # ---- Saving & Plotting RF Diagnostics ----

    lsq_fields.save_rf_csv("data/cornet_rf_estimates_bbox.csv", voxels,
        possible_fields.estimates_to_params(estimated_bbox))
    lsq_fields.save_rf_csv("data/cornet_rf_estimates_grad.csv", voxels,
        possible_fields.estimates_to_params(estimated_grad))
    lsq_fields.save_rf_csv("data/cornet_rf_refined.csv", voxels, refined)

    plot.rfs.grad_heatmap('plots/cornet_rf_grads.pdf',
        layers, voxels, grads, bboxes = bboxes)
    plot.rfs.gaussian_rfs_refinement('plots/cornet_rf_refinement.pdf',
        layers, voxels, (frame, frame), estimated_grad, refined)

    lsq_fields.save_activity_csv(
        'data/cornet_activity.csv',
        voxels, true_acts, pred_acts)
    plot.rfs.activation_trials(
        'plots/cornet_activity.pdf',
        voxels, true_acts, pred_acts)



    # ---- Saving & Plotting Method Comparisons ----

    plot.rfs.motion_vectors('plots/bbox_grad_estimate_vectors.pdf',
        (frame, frame), None, voxels, 
        possible_fields.estimates_to_params(estimated_bbox), 
        possible_fields.estimates_to_params(estimated_grad))
    plot.rfs.motion_vectors('plots/cornet_rf_refinement_vectors.pdf',
        (frame, frame), None, voxels, 
        possible_fields.estimates_to_params(estimated_grad), 
        refined)





