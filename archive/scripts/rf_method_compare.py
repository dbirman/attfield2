from proc import possible_fields
from proc import lsq_fields
from proc import cornet
import plot.rfs


def rfs_for_method(method, model, voxels, approx_n = 10):
    if method[0] == 'cifar':
        inputs = video_gen.cifar_images(64, approx_n)
    elif mdethod[0] == 'slider_check':
        inputs = [video_gen.rf_slider(64, speed = bar_speed)*255,
                  video_gen.rf_slider_check(64, speed = bar_speed)]
        inputs = np.concatenate(inputs)

    if method[1] == 'grad':
        manager, grads = bf.gradients_raw_fullbatch(
            manager, model, inputs, voxels,
            approx_n = approx_n)
        estimated = backprop_fields.parameter_estimates(grads, voxels)
    elif method[1] ==  'bb':
        manager = nm.NetworkManager.assemble(model, inputs)
        bboxes = possible_fields.rf_bboxes(voxels, manager)

    refined, _ = lsq_fields.fit_rfs(
        inputs.numpy(), manager, voxels, , estimated,
        verbose = 1)

    return manager, estimated, refined




if __name__ == '__main__':

    

    model, ckpt = cornet.load_cornet("Z")
    layers = [(0, 3, 0)]
    voxels = vx.random_voxels_for_model(
        model, layers, 1, 3, 64, 64)
    modes = [('cifar', 'bb'), ('slider_check', 'bb'),
             ('cifar', 'grad'), ('slider_check', 'grad')]
    ns = [10, 30, 60]

    results = {}
    for mode in modes:
        results[mode] = {}
        for n in ns:
            results[mode][n] = rfs_for_method(
                mode, model, voxels,
                approx_n = n)

    from pprint import pprint
    pprint(results)

    # n-wise comparisons
    for mode in modes:
        # Betas_by_n shape (n_Ns, n_vox)
        betas_by_n = [np.array(results[mode][n]).T[3] for n in ns]
        print(betas)
        success_rate = [np.isnan(betas)/len(betas) for betas in betas_by_n]





