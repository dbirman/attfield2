
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import seaborn as sns

from argparse import ArgumentParser
import pandas as pd
import numpy as np
import tqdm, sys
import h5py



def locus_field_ratio(dists, gains, locus = 0.02, field = 0.5):
    locus_mask = dists <= np.quantile(dists, locus)
    field_mask = dists >= np.quantile(dists, field)
    return gains[locus_mask].mean() / gains[field_mask].mean()



def normalization_data(norm_base, norm_focl, layers, exponents, loc, degrees_in_img = 22):

    pre_sds = {k: v for k, v in np.load(norm_base).items()}

    dists = {}
    ws = {}
    hs = {}
    for l in layers:
        ws[l] = pre_sds[l].shape[-1]
        hs[l] = pre_sds[l].shape[-2]
        cs, rs = np.meshgrid(np.linspace(0, 1, ws[l]), np.linspace(0, 1, hs[l]))
        dists[l] = np.sqrt(((cs - loc[0]) * degrees_in_img) ** 2 +
                           ((rs - loc[1]) * degrees_in_img) ** 2)
    
    sd_gains = []
    lfs = []
    zero_div = lambda a,b: np.divide(a, b, out = np.zeros_like(a), where = b!=0)
    for i_f in range(len(norm_focl)):

        sd_gains.append({k: v for k, v in np.load(norm_focl[i_f]).items()})

        lfs.append({})
        for l in layers:
            lfs[i_f][l] = {}
            for i_exp in range(len(exponents)):
                xs = np.tile(dists[l][None, :, :], [sd_gains[i_f][l].shape[-3], 1, 1]).ravel()
                ys = sd_gains[i_f][l][i_exp].ravel()
                lfs[i_f][l][i_exp] = locus_field_ratio(xs, ys)

    return lfs




def normalization_heatmap(lfs, betas, exponents, layer, ax):
    data = np.array([[lfs[i_f][layer][i_exp]
       for i_exp in range(len(exponents))]
       for i_f in range(len(betas))])[::-1]
    
    mappable = ax.imshow(data, cmap = 'bone_r', extent = (0, 1, 0, 1))
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticklabels([np.round(betas[0], 4), "", np.round(betas[-1], 4)])
    ax.set_xticklabels([np.round(exponents[0], 4), "", np.round(exponents[-1], 4)])
    return mappable
























