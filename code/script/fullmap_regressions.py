import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# iso_file = "data/gratings/augment_ccw_iso112.h5"
iso_file = "data/imagenet/imagenet_iso224.h5"
out_file = "data/imagenet/imagenet_iso224_pasted.h5"
n_train = 2000
full_size = 224
iso_size = 112

iso_h5 = h5py.File(iso_file, 'r')
iso_h5.keys()
cats = [c for c in iso_h5.keys() if not c.endswith("_y")]

with h5py.File(out_file, 'w') as out_h5:
    for cat in cats:
        R = C = np.arange(full_size)
        R = R[None, :, None]
        C = C[None, None, :]
        c, r = np.random.randint(full_size - iso_size, size = (2, n_train))[:, :, None, None]
        rmask = (r <= R) & (R < r + iso_size)
        cmask = (c <= C) & (C < c + iso_size)
        full_img = np.ones([n_train, full_size, full_size, 3])
        img_select = np.arange(0, n_train) * (iso_h5[cat].shape[0] // n_train)
        full_img *= iso_h5[cat][0, 0, 0, :][None, None, None, :]
        full_img[rmask & cmask] = iso_h5[cat][:n_train].reshape([-1, 3])
        
        dset_img = out_h5.create_dataset(cat, full_img.shape, full_img.dtype)
        dset_img[...] = full_img
        dset_y = out_h5.create_dataset(cat + '_y', (n_train,), np.bool_)
        dset_y[...] = iso_h5[cat + '_y'][:n_train]



# gratings data
embeddings_file = "data/runs/apool/enc_accw_iso-p.h5"
n_train = 4800
dist_file = "data/runs/acuity/fnenc_task_accw_base.h5"
focl_file = "data/runs/acuity/enc_task_accw_multigauss_b1.4.h5"
plot_file = "plots/runs/apool/accw_iso-p_regs/c-{i_cat}.pdf"
coef_file = "data/runs/apool/accw_isp-p_coefs.npz"

# imagenet data
embeddings_file = "ssddata/apool/enc_ign_iso224.h5"
n_train = 1600
dist_file = "ssddata/runs/fig2/enc_task_imgnet_base.h5" # acuity_behavior.md
focl_file = "data/runs/fig2/enc_task_imgnet_gauss_b4.0.h5" # acuity_behavior.md
plot_file = "plots/runs/apool/ign_iso224_regs/c-{i_cat}.pdf"
coef_file = "data/runs/apool/ign_iso224_coefs.npz"

layer = '0.4.3'




embed_h5 = h5py.File(embeddings_file, 'r')
if not os.path.exists(os.path.dirname(plot_file)): os.mkdir(os.path.dirname(plot_file))



all_coefs = []
all_apool_coefs = []
val_aucs = {'apool': [], 'fullmap': []}
gain_effect = []
gain_effect_best = []
gain_effect_apool = []
for i_cat in range(embed_h5[layer].shape[0]):
    with PdfPages(plot_file.format(i_cat = i_cat)) as pdf:
        print(f"Category {i_cat}")
        feats = embed_h5[layer][i_cat, :n_train]
        map_size = feats.shape[2]
        n_maps = feats.shape[1]
        feats = feats.reshape([n_train, -1])
        # feats = feats.reshape([n_train, n_maps, map_size, map_size]) # inverse
        reg_args = dict(
            solver = 'liblinear',
            max_iter = 1000,
            fit_intercept = False)
        reg = LogisticRegression(**reg_args)
        ys = embed_h5['y'][i_cat, :n_train].astype('bool')
        reg.fit(feats, ys)
        fn_scores = (reg.coef_ * feats).sum(axis = 1)

        val_feats = embed_h5[layer][i_cat, n_train:]
        val_feats = val_feats.reshape([val_feats.shape[0], -1]) 
        val_ys = embed_h5['y'][i_cat, n_train:].astype('bool')
        val_fn_scores = (reg.coef_ * val_feats).sum(axis = 1)

        trn_auc = roc_auc_score(ys, fn_scores)
        val_auc = roc_auc_score(val_ys, val_fn_scores)
        val_aucs['fullmap'].append(val_auc)

        fig, ax = plt.subplots(1, 2)
        for i, (fn, y, auc, ttl) in enumerate([
            (fn_scores, ys, trn_auc, "Train"), 
            (val_fn_scores, val_ys, val_auc, "Val")]):
            ax[i].plot(
                np.random.uniform(-0.1, 0.1, y.sum()), fn[y],
                'C0o', ms = 5, alpha = 0.5)
            ax[i].plot(
                np.random.uniform(0.9, 1.1, (~y).sum()), fn[~y],
                'C3o', ms = 5, alpha = 0.5)
            ax[i].set_title(f"{ttl}, AUC={auc:.3f}")
        pdf.savefig(); plt.close()
        # plt.show()


        fig, ax = plt.subplots(1, 1, figsize = (4, 4))
        for i, (fn, y) in enumerate([(fn_scores, ys), (val_fn_scores, val_ys)]):
            fpr, tpr, thresh = roc_curve(y, fn)
            ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], lw = 1, color = '.8')
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        plt.tight_layout()
        pdf.savefig(); plt.close()
        # plt.show()

        coefs = reg.coef_[0].reshape([n_maps, -1])
        all_coefs.append(coefs)
        all_corrs = np.corrcoef(coefs.T)

        flatmap_ix = np.arange(map_size ** 2).reshape([map_size, map_size])
        fig, ax = plt.subplots(map_size, map_size, figsize = (7, 7))
        for r in range(map_size):
            for c in range(map_size):
                ax[r, c].imshow(all_corrs[flatmap_ix[r, c], flatmap_ix],
                    vmin = 0, vmax = 1)
                ax[r, c].set_axis_off()
        plt.suptitle("Positional correlations. Color=[0, 1]")
        plt.tight_layout()
        pdf.savefig(); plt.close()
        # plt.show()


        trn_scores = np.zeros([map_size, map_size])
        val_scores = np.zeros([map_size, map_size])
        sq_trn_feats = feats.reshape([n_train, n_maps, map_size, map_size])
        sq_val_feats = val_feats.reshape([len(val_ys), n_maps, map_size, map_size])
        for r in range(map_size):
            for c in range(map_size):
                ix_coefs = coefs[None, :, flatmap_ix[r, c], None, None]
                trn_fn_scores = (ix_coefs * sq_trn_feats).sum(axis = (1, 2, 3))
                val_fn_scores = (ix_coefs * sq_val_feats).sum(axis = (1, 2, 3))
                trn_scores[r, c] = roc_auc_score(ys, trn_fn_scores)
                val_scores[r, c] = roc_auc_score(val_ys, val_fn_scores)
        best_coefs = coefs[:, np.argmax(val_scores.ravel())]

        fig, ax = plt.subplots(2, 2, figsize = (3, 5), gridspec_kw = dict(width_ratios = [8, 1]))
        for i, (scores, ttl) in enumerate([(trn_scores, "Train"), (val_scores, "Val")]):
            _ = ax[i, 0].imshow(scores)
            ax[i, 0].set_axis_off()
            ax[i, 0].set_title(ttl)
            plt.colorbar(_, cax = ax[i, 1])
        plt.tight_layout(rect = (0, 0, 1, 0.95))
        plt.suptitle("Best individual regresions.")
        pdf.savefig(); plt.close()
        # plt.show()


        dist_h5 = h5py.File(dist_file, 'r')
        focl_h5 = h5py.File(focl_file, 'r')


        dist_embed = dist_h5[layer][i_cat]
        focl_embed = focl_h5[layer][i_cat]
        n_img = dist_embed.shape[0]
        dist_embed = dist_embed.reshape([n_img, n_maps, -1])
        focl_embed = focl_embed.reshape([n_img, n_maps, -1])
        dist_fn = (dist_embed * coefs[None]).sum(axis = (1, 2))
        focl_fn = (focl_embed * coefs[None]).sum(axis = (1, 2))
        dist_bt = (dist_embed * best_coefs[None, :, None]).sum(axis = (1, 2))
        focl_bt = (focl_embed * best_coefs[None, :, None]).sum(axis = (1, 2))
        cat_ys = dist_h5['y'][i_cat].astype('bool')
        gain_effect.append(
            (roc_auc_score(cat_ys, dist_fn),
             roc_auc_score(cat_ys, focl_fn))
        )
        gain_effect_best.append(
            (roc_auc_score(cat_ys, dist_bt),
             roc_auc_score(cat_ys, focl_bt))
        )

        fig, ax = plt.subplots(1, 2, sharey = True)
        for i, (fn, y, ttl) in enumerate([
            (dist_fn, cat_ys, "Distributed"),
            (focl_fn, cat_ys, "Focal")]):
            ax[i].plot(
                np.random.uniform(-0.1, 0.1, y.sum()), fn[y],
                'C0o', ms = 5, alpha = 0.5)
            ax[i].plot(
                np.random.uniform(0.9, 1.1, (~y).sum()), fn[~y],
                'C3o', ms = 5, alpha = 0.5)
            ax[i].set_title(ttl)
        plt.suptitle("Gain effect: full map")
        pdf.savefig(); plt.close()
        # plt.show()


        apool_trn_feats = (feats
            ).reshape([n_train, n_maps, map_size, map_size]
            ).mean(axis = (2, 3))
        reg = LogisticRegression(**reg_args)
        reg.fit(apool_trn_feats, ys)
        apool_coefs = reg.coef_[0]
        all_apool_coefs.append(apool_coefs)
        apool_trn_fn = (reg.coef_ * apool_trn_feats).sum(axis = 1)

        apool_val_feats = (val_feats
            ).reshape([len(val_feats), n_maps, map_size, map_size]
            ).mean(axis = (2, 3))
        apool_val_fn = (reg.coef_ * apool_val_feats).sum(axis = 1)

        trn_auc = roc_auc_score(ys, apool_trn_fn)
        val_auc = roc_auc_score(val_ys, apool_val_fn)
        val_aucs['apool'].append(val_auc)

        apool_dist_fn = (dist_embed.mean(axis = 2) * apool_coefs[None]).sum(axis = 1)
        apool_focl_fn = (focl_embed.mean(axis = 2) * apool_coefs[None]).sum(axis = 1)
        gain_effect_apool.append(
            (roc_auc_score(cat_ys, apool_dist_fn),
             roc_auc_score(cat_ys, apool_focl_fn))
        )

        fig, ax = plt.subplots(1, 2)
        for i, (fn, y, auc, ttl) in enumerate([
            (apool_trn_fn, ys, trn_auc, "Train"), 
            (apool_val_fn, val_ys, val_auc, "Val")]):
            ax[i].plot(
                np.random.uniform(-0.1, 0.1, y.sum()), fn[y],
                'C0o', ms = 5, alpha = 0.5)
            ax[i].plot(
                np.random.uniform(0.9, 1.1, (~y).sum()), fn[~y],
                'C3o', ms = 5, alpha = 0.5)
            ax[i].set_title(f"{ttl}, AUC={auc:.3f}")
        plt.suptitle("APool scores.")
        pdf.savefig(); plt.close()

        fig, ax = plt.subplots(1, 1, figsize = (4, 4))
        for i, (fn, y) in enumerate([(apool_trn_fn, ys), (apool_val_fn, val_ys)]):
            fpr, tpr, thresh = roc_curve(y, fn)
            ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], lw = 1, color = '.8')
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        plt.suptitle("APool ROC.")
        pdf.savefig(); plt.close()

        apool_corrs = np.zeros([map_size, map_size])
        for r in range(map_size):
            for c in range(map_size):
                apool_corrs[r, c] = np.corrcoef(apool_coefs, coefs[:, flatmap_ix[r, c]])[0, 1]
        fig, ax = plt.subplots(1, 2, figsize = (3, 2.5), gridspec_kw = dict(width_ratios = [8, 1]))
        _ = ax[0].imshow(apool_corrs)
        ax[0].set_axis_off()
        plt.colorbar(_, cax = ax[1])
        plt.suptitle("Correlation to APool weights.")
        plt.tight_layout()
        pdf.savefig(); plt.close()


        fig, ax = plt.subplots(1, 2, sharey = True)
        for i, (fn, y, ttl) in enumerate([
            (apool_dist_fn, cat_ys, "Distributed"),
            (apool_focl_fn, cat_ys, "Focal")]):
            ax[i].plot(
                np.random.uniform(-0.1, 0.1, y.sum()), fn[y],
                'C0o', ms = 5, alpha = 0.5)
            ax[i].plot(
                np.random.uniform(0.9, 1.1, (~y).sum()), fn[~y],
                'C3o', ms = 5, alpha = 0.5)
            ax[i].set_title(ttl)
        plt.suptitle("Gain effect: APool")
        pdf.savefig(); plt.close()

        fig, ax = plt.subplots(1, 2, figsize = (5, 4), gridspec_kw = dict(width_ratios = [3, 1]))
        for data, ttl, color in [
            (gain_effect[-1], "Full map", 'C0'),
            (gain_effect_best[-1], "Best", 'C3'),
            (gain_effect_apool[-1], "APool", 'C4')]:
            ax[0].plot(
                [0, 1], data, 
                '-', color = color, lw = 1,
                label = ttl)
            ax[0].plot(
                [0, 1], data,
                'o', color = color, ms = 6, mew = 2, mec = 'w')
        ax[0].set_xticks([0, 1])
        ax[0].set_xticklabels(["Dist.", "Focal"])
        ax[1].set_axis_off()
        ax[0].legend(frameon = False, loc = 'center left', bbox_to_anchor = (1, 0.5))
        pdf.savefig(); plt.close()
        # plt.show()

fig, ax = plt.subplots(1, 3, figsize = (8, 5), sharey = True)
for i, (data, ttl, color) in enumerate([
    (gain_effect, "Full map", 'C0'),
    (gain_effect_best, "Best", 'C3'),
    (gain_effect_apool, "APool", 'C4')]):
    pltx = [0, 1] * len(data) + np.random.uniform(-0.00, 0.00, 2*len(data))
    plty = np.array(data).ravel()
    ax[i].plot(
        pltx.reshape([-1, 2]).T, plty.reshape([-1, 2]).T,
        '-', color = color, lw = 0.5)
    ax[i].plot(
        pltx, plty,
        'o', color = color, ms = 4, mew = 1, mec = 'w')
    ax[i].set_title(ttl)
    ax[i].set_xticks([0, 1])
    ax[i].set_xticklabels(["Dist.", "Focal"])
plt.tight_layout()
plt.savefig(plot_file.format(i_cat = "all")); plt.close()

np.savez(coef_file,
    fullmap_coefs = all_coefs,
    apool_coefs = all_apool_coefs,
    fullmap_auc = val_aucs['fullmap'],
    apool_auc = val_aucs['apool'],
    n_maps = n_maps,
    map_size = map_size)





# Reconstruct plot
# =========================================================


coef_data = np.load(coef_file)

from plot import readouts

fullmap_readout_data = readouts.readout_data(
    dist_file, focl_file,
    (0, 4, 3))

# without avg pool
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
scores_dict_7x7 = readouts.reconstructed_bhv_auc(fullmap_readout_data,
    coef_data['fullmap_coefs'].reshape([20, 512, 7, 7])
)
readouts.reconstructed_bhv_plot(ax, scores_dict_7x7)
# plot without obtaining scores_dict:
# readouts.reconstructed_bhv(ax, fullmap_readout_data,
#     coef_data['fullmap_coefs'].reshape([20, 512, 7, 7])
# )
plt.savefig("plots/runs/apool/7x7_neccsuff.pdf")
plt.show()

np.savez('data/runs/apool/7x7_neccsuff_scores.npz', **scores_dict_7x7)



# with avg pool
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
readouts.reconstructed_bhv(ax, fullmap_readout_data,
    coef_data['apool_coefs'].reshape([20, 512, 1, 1])
)
plt.savefig("plots/runs/apool/pooled_neccsuff.pdf")
plt.show()





# individual analyses from the 7x7
# =========================================================
"""
Notes:
- rather obviously "fake" and "undo" conditions do not
  change scores achieves by any pixel
- how do we test the relevance of these pixel-wise changes
  in AUC to the result of Fig 7? Still have the intuition
  that really what's happening is the performance of the
  pooled map comes to meet the performance of the pixels
  at the center of the target regime via effectively 
  changing the readout (as seen in the "fake" condition!!)
  and that the effect of these modified pixel-wise AUCs
  do not explain the performance benefits of the Focal
  condition (see "undo" which *still has* these pixel-
  wise improvements) Indeed we DO see shift-based effects
  on readout windows smaller than the gain window (of
  comparable size to gain based effects when the readout
  window is large) but I'm not sure what the physiological
  relevance of small readout windows is..when might other
  areas pull from single IT units?

"""


import sklearn.metrics as skmtr

coef_data = np.load(coef_file)
fullmap_readout_data = readouts.readout_data(
    dist_file, focl_file,
    (0, 4, 3))



def per_pix_auc(pos, neg):
    ytrue = np.concatenate([
        np.ones(pos.shape[0]),
        np.zeros(neg.shape[0])])
    scores = np.zeros([7, 7])
    for r in range(7):
        for c in range(7):
            fn = np.concatenate([
                pos[..., r, c].sum(axis = -1),
                neg[..., r, c].sum(axis = -1)])
            scores[r, c] = skmtr.roc_auc_score(ytrue, fn)
    return scores
    



scores_dict = readouts.reconstructed_bhv_auc(
    fullmap_readout_data,
    coef_data['apool_coefs'].reshape([20, 512, 1, 1]),
    score_func = per_pix_auc)

gain_fx = (scores_dict['score_focl'] - scores_dict['score_dist']).mean(axis = 0)
fake_fx = (scores_dict['score_fake'] - scores_dict['score_dist']).mean(axis = 0)
undo_fx = (scores_dict['score_undo'] - scores_dict['score_dist']).mean(axis = 0)

fig, ax = plt.subplots(1, 1, figsize = (4, 3))
imshow_kw = dict(cmap = "RdBu", vmin = -0.04, vmax = 0.04)
clr = ax.imshow(gain_fx, **imshow_kw)
plt.colorbar(clr).set_label("Average AUC Change")
plt.tight_layout()
plt.savefig("plots/runs/apool/1x1_effect.pdf")
plt.show()

np.savez("data/runs/apool/1x1_scores.npz", **scores_dict)

fig, ax = plt.subplots(1, 3, figsize = (9, 3))
imshow_kw = dict(cmap = "Spectral", vmin = 0.5, vmax = 0.8)
clr = ax[0].imshow(scores_dict['score_dist'].mean(axis = 0), **imshow_kw)
plt.colorbar(clr)
plt.tight_layout()
plt.show()




# masked readout reconstruct plot
# =========================================================

def mask_auc(mask):
    def score_func(pos, neg):
        ytrue = np.concatenate([
            np.ones(pos.shape[0]),
            np.zeros(neg.shape[0])])
        scores = np.zeros([7, 7])

        fn = np.concatenate([
            (mask[None, None] * pos).mean(axis = (-2, -1)).sum(axis = -1),
            (mask[None, None] * neg).mean(axis = (-2, -1)).sum(axis = -1)])
        return skmtr.roc_auc_score(ytrue, fn)
    return score_func


# target 4x4
mask = np.zeros([7, 7])
mask[:4, :4] = 1
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
scores_dict = readouts.reconstructed_bhv_auc(fullmap_readout_data,
    coef_data['apool_coefs'].reshape([20, 512, 1, 1]),
    score_func = mask_auc(mask))
readouts.reconstructed_bhv_plot(ax, scores_dict)
ax.set_ylabel("AUC")
plt.tight_layout()
plt.savefig("plots/runs/apool/4x4_neccsuff.pdf")
plt.show()

np.savez('data/runs/apool/4x4_neccsuff_scores.npz', **scores_dict)


# compare to isolated image performance
scores4x4 = readouts.reconstructed_bhv_auc(fullmap_readout_data,
    coef_data['apool_coefs'].reshape([20, 512, 1, 1]),
    score_func = mask_auc(mask)
)

# note: scores are not *quite* saturated
plt.plot(coef_data['apool_auc'], scores4x4['score_dist'], 'wo', mec = 'C0')
plt.plot(coef_data['apool_auc'], scores4x4['score_focl'], 'C0o')
plt.plot(*(2*[np.sort(coef_data['apool_auc'])]), '.8')
plt.xlabel("Isolated performance")
plt.ylabel("4x4 performance")



# featmap locations : correlation / contribution
# =========================================================
"""
Why don't the AUC shifts at the border improve performance?
Have that cosine image performance with equal norms is
mean performance across spatial locations. Appears that this
is *not* true of AUC. So, what is causing the small changes
in locationwise cosine performance (which add up to small
change in imagewise cosine performance) to show up as large
changes in pixelwise AUC?

Nothing weird about the conversion to AUC - in fact once you
convert to spatially normalized feature maps the conversion 
is pretty straightforward (assuming equal stddev). This
question appears to be based more on an optical illusion than
anything else: the improvement in AUCs along the border does
not outweigh the degredation in AUC elsewhere. In fact the
improvement in imagewise AUC with normalized pixels is *more*
than you would expect if AUC simply summed across component
locationwise AUCs.
"""

# normalized decision axis
axis = coef_data['apool_coefs'].reshape([20, 1, 512, 1, 1])
axis /= np.linalg.norm(axis, axis = 2, keepdims = True)

# select out distributed and focal conditions
pos_dist = fullmap_readout_data.pos_dist
neg_dist = fullmap_readout_data.neg_dist
pos_focl = fullmap_readout_data.pos_focl
neg_focl = fullmap_readout_data.neg_focl
conds = [pos_dist, neg_dist, pos_focl, neg_focl]

# normalize activations
norms = [np.linalg.norm(cond, axis = 2, keepdims = True) for cond in conds]
normd = [cond / norm for cond, norm in zip(conds, norms)]

# pixelwise alignment with decision axis
raw_cos = [(n * axis).sum(axis = 2) for n in normd]
cos1x1_dist = np.concatenate([raw_cos[0], -raw_cos[1]], axis = 1)
cos1x1_focl = np.concatenate([raw_cos[2], -raw_cos[3]], axis = 1)


fig, ax = plt.subplots(1, 1, figsize = (4, 3))
imshow_kw = dict(cmap = "Spectral")
# imshow_kw = dict(cmap = "Spectral", vmin = -0.03, vmax = 0.03)
clr = ax.imshow((cos1x1_focl - cos1x1_dist).mean(axis = (0, 1)), **imshow_kw)
plt.colorbar(clr)
plt.tight_layout()
plt.show()


# weighted sum to calculate full image cosine performance
# this is rather unneccesary -- simple arithmetic result
full_img_norms = [
    np.linalg.norm(
        cond.sum(axis = (-2, -1), keepdims = True),
    axis = 2, keepdims = True)
    for cond in conds
]
cos_weights = [pix / full for full, pix in zip(full_img_norms, norms)]
cweight_dist = np.concatenate([cos_weights[0], cos_weights[1]], axis = 1)
cweight_focl = np.concatenate([cos_weights[2], cos_weights[3]], axis = 1)

wcos1x1_dist = cweight_dist * cos1x1_dist
wcos1x1_focl = cweight_focl * cos1x1_focl



# do changes in pixelwise AUC show up when feature maps
# are spatially normalized?
pxl_dist_normd_auc = np.stack([
    per_pix_auc(normd[0][i] * axis[i], normd[1][i] * axis[i])
    for i in range(20)])
pxl_focl_normd_auc = np.stack([
    per_pix_auc(normd[2][i] * axis[i], normd[3][i] * axis[i])
    for i in range(20)])
img_dist_normd_auc = np.stack([
    readouts.compose_auc(normd[0][i] * axis[i], normd[1][i] * axis[i])
    for i in range(20)])
img_focl_normd_auc = np.stack([
    readouts.compose_auc(normd[2][i] * axis[i], normd[3][i] * axis[i])
    for i in range(20)])


# confirmation: relatively large changes in pixelwise AUC
fig, ax = plt.subplots(1, 1, figsize = (4, 3))
imshow_kw = dict(cmap = "RdBu", vmin = -0.04, vmax = 0.04)
clr = ax.imshow((pxl_focl_normd_auc - pxl_dist_normd_auc).mean(axis = 0), **imshow_kw)
plt.colorbar(clr)
plt.tight_layout()
plt.show()



# but still no large change in total image AUC
fig, ax = plt.subplots(1, 1, figsize = (4, 3))
ax.plot(np.random.uniform(-0.1, 0.1, 20), img_dist_normd_auc, 'C0o', ms = 4)
ax.plot([0], [img_dist_normd_auc.mean()], 'ks', ms = 8, mec = 'w')
ax.plot(np.random.uniform( 0.9, 1.1, 20), img_focl_normd_auc, 'C3o', ms = 4)
ax.plot([1], [img_focl_normd_auc.mean()], 'ks')
plt.tight_layout()
plt.show()

# is this all still roughly additive?
# the spatial average of differences in cosine performance
# is significantly smaller than the 
plt.hist((pxl_focl_normd_auc - pxl_dist_normd_auc).mean(axis = (1,2)))
plt.hist((img_focl_normd_auc - img_dist_normd_auc))

# great plot to see relationship:
fig, ax = plt.subplots(1, 1, figsize = (5, 4))
plt.plot(
    (pxl_focl_normd_auc - pxl_dist_normd_auc).mean(axis = (1,2)),
    (img_focl_normd_auc - img_dist_normd_auc), 'C0o')
plt.plot(
    np.sort(img_focl_normd_auc - img_dist_normd_auc),
    np.sort(img_focl_normd_auc - img_dist_normd_auc), '.8')
plt.xlabel("Mean locationwise AUC")
plt.ylabel("Imagewise AUC")
plt.savefig('plots/runs/apool/auc_aggregate.pdf')
plt.show()


















