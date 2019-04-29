from sklearn.decomposition import PCA
import numpy as np


def bin_cts_grouping(grouping, n):
    bins = np.histogram_bin_edges(grouping, n)
    centers = (bins[:-1] + bins[1:])/2
    return np.digitize(grouping, bins), centers


def tuning_pca(voxels, acts):
    '''
    Run PCA dimensionality reduction to produce a simple
    and interpretable way to visualise tuning.
    
    ### Returns
    - `pca` --- Dictionary giving a sklearn PCA object for
        each layer.
    - `embedded` --- Dictionary mapping embeddings of the
        training set in PCA space, giving a rough measure
        of how voxels are tuned. Each value in the dictionary
        is an array with shape (vox, pca_dim)
    '''
    n_repeat = len(acts)
    pcas = {}
    embedded = {}
    for layer, layer_vox in voxels.items():
        # Average activations for this layer over the grouping
        # `act_avg` will have shape (frame, voxel)
        all_acts = [
            acts[i][layer][:, :]
            for i in range(n_repeat)]
        acts_avg = np.array(all_acts).mean(axis = 0)
        acts_concat = np.concatenate(all_acts, axis = 1)

        print("Fitting for layer " + str(layer))
        pca = PCA().fit(acts_concat.T)
        embedded[layer] = pca.transform(acts_avg.T)
        pcas[layer] = pca
    return pcas, embedded
