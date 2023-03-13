import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)


import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# quick test of sensitivity to flipped images in the encodings


# Form flipped images
# =====================================================

iso_file = "data/imagenet/imagenet_iso224.h5"
out_file = "ssddata/imagenet/imagenet_iso224_flip.h5"

iso_h5 = h5py.File(iso_file, 'r')
cats = [c for c in iso_h5.keys() if not c.endswith('_y')]
# cats = ['bathtub', 'bakery', 'artichoke']

with h5py.File(out_file, 'w') as out_h5:
    for cat in cats:
        print(f"Category: {cat}")
        nimg = iso_h5[cat + '_y'][...].sum()
        to_flip = iso_h5[cat][...][iso_h5[cat + '_y'][...].astype('bool')]
        dset_img = out_h5.create_dataset(cat, to_flip.shape, iso_h5[cat].dtype)
        dset_img[...] = to_flip[:, ::-1, :, :]
        dset_y = out_h5.create_dataset(cat + '_y', (nimg,), np.bool_)
        dset_y[...] = True




# Train logstic regressions on fliped and upright imgages
# =====================================================

flip_encodings = "ssddata/apool/enc_ign_iso112_flip.h5" # flip_detection.md
uprt_encodings = "ssddata/apool/enc_ign_iso112.h5" # flip_detection.md
reg_performance_plot = "plots/runs/flip/reg_performance_112.pdf"
raw_weight_out = 'data/models/opposed_regs_ign112_flip.npz'
reg_out = 'data/models/regs_ign112_flip.npz'
layer = '0.4.3'
n_trn_each = 400
n_val_each = 200

flip_h5 = h5py.File(flip_encodings, 'r')
uprt_h5 = h5py.File(uprt_encodings, 'r')

opposing_weights = {'cats': cats}
combined_regs = {}
weight_corrs = {}
val_aucs = {}
combo_auc = {}
with PdfPages(reg_performance_plot) as pdf:
    for i_cat, cat in enumerate(cats):

        uprt_y = uprt_h5['y'][i_cat].astype('bool')
        trn_uprt_feat = uprt_h5[layer][i_cat][...][uprt_y][:n_trn_each]
        val_uprt_feat = uprt_h5[layer][i_cat][...][uprt_y][n_trn_each:n_trn_each + n_val_each]

        trn_flip_feat = flip_h5[layer][i_cat, :n_trn_each]
        val_flip_feat = flip_h5[layer][i_cat, n_trn_each:n_trn_each + n_val_each]

        all_trn_feat = np.hstack([trn_uprt_feat, trn_flip_feat]
            ).reshape((n_trn_each*2,) + trn_uprt_feat.shape[1:]
            ).mean(axis = (2, 3))
        val_feat = np.hstack([val_uprt_feat, val_flip_feat]
            ).reshape((n_val_each*2,) + val_uprt_feat.shape[1:]
            ).mean(axis = (2, 3))
        trn_isuprt = np.array([1, 0] * n_trn_each).astype('bool')
        val_isuprt = np.array([1, 0] * n_val_each).astype('bool')

        trn_ix = np.arange(n_trn_each * 2) < n_trn_each

        fig, ax = plt.subplots(1, 4, figsize = (10, 3), sharey = True)
        for j, (cond, trn_x, val_x, trn_y, val_y) in enumerate([
                ('uprt', all_trn_feat[ trn_ix], val_feat,  trn_isuprt[ trn_ix],  val_isuprt),
                ('flip', all_trn_feat[~trn_ix], val_feat, ~trn_isuprt[~trn_ix], ~val_isuprt)]):
            reg = LogisticRegression(
                solver = 'liblinear',
                max_iter = 1000,
                fit_intercept = False)
            reg.fit(trn_x, trn_y)
            trn_fn = (reg.coef_ * trn_x).sum(axis = 1)
            val_fn = (reg.coef_ * val_x).sum(axis = 1)
            opposing_weights[f'{cat}:{cond}'] = reg.coef_

            for i, (fn, y, ttl) in enumerate([
                (trn_fn, trn_y, 'Train'),
                (val_fn, val_y, 'Val')]):
                ax[2*j + i].plot(
                    np.random.uniform(-0.1, 0.1, y.sum()), fn[y],
                    'C0o', ms = 5, alpha = 0.5)
                ax[2*j + i].plot(
                    np.random.uniform(0.9, 1.1, (~y).sum()), fn[~y],
                    'C3o', ms = 5, alpha = 0.5)
                ax[2*j + i].set_title(f"{cond} | {ttl} AUC: {roc_auc_score(y, fn):.3f}")
                if ttl == 'Val': val_aucs[f'{cat}:{cond}'] = roc_auc_score(y, fn)
        plt.suptitle(f"Regressions - {cat}")
        plt.tight_layout(rect = (0, 0, 1, 0.95))
        pdf.savefig(); plt.close()
            # plt.show()

        weight_corrs[cat] = np.corrcoef(np.concatenate([
            opposing_weights[f'{cat}:uprt'],
            opposing_weights[f'{cat}:flip']
        ]))[1, 0]

        combo = opposing_weights[f'{cat}:flip'] - opposing_weights[f'{cat}:uprt']
        combined_regs[cat] = combo
        combo_val_fn = (combo * val_feat).sum(axis = 1)
        combo_auc[cat] = roc_auc_score(val_isuprt, combo_val_fn)


    fig, ax = plt.subplots(1, 1, figsize = (12, 5))
    ax.scatter(
        np.arange(len(val_aucs)), val_aucs.values(), s = 20, 
        c = (['C0', 'C0', 'C4', 'C4'] * (len(val_aucs) // 4 + 1))[:len(val_aucs)])
    ax.set_xticks(np.arange(len(val_aucs)))
    ax.set_xticklabels(val_aucs.keys(), rotation = 45, horizontalalignment = 'right')
    ax.set_ylabel("Val AUC")
    plt.tight_layout()
    pdf.savefig(); plt.close()

    fig, ax = plt.subplots(1, 1, figsize = (8, 5))
    ax.plot(
        np.arange(len(weight_corrs)), weight_corrs.values(),
        'C0o')
    ax.set_xticks(np.arange(len(weight_corrs)))
    ax.set_xticklabels(weight_corrs.keys(), rotation = 45, horizontalalignment = 'right')
    ax.set_ylabel("Weight correlation")
    ax.set_ylim(-1.1, 0.1)
    plt.tight_layout()
    pdf.savefig(); plt.close()

    fig, ax = plt.subplots(1, 1, figsize = (8, 5))
    ax.plot(
        np.arange(len(combo_auc)), combo_auc.values(),
        'C0o')
    ax.set_xticks(np.arange(len(combo_auc)))
    ax.set_xticklabels(combo_auc.keys(), rotation = 45, horizontalalignment = 'right')
    ax.set_ylabel("Combined Regression Val AUC")
    plt.tight_layout()
    pdf.savefig(); plt.close()


opposing_weights = {**opposing_weights, **{f'{k}_auc':v for k,v in val_aucs.items()}}
np.savez(raw_weight_out, **opposing_weights)
np.savez(reg_out, **combined_regs)

opposing_weights = np.load(raw_weight_out)


# Form 112x112 imagenet images
# =====================================================

iso_file = "ssddata/imagenet/imagenet_iso224_flip.h5"
out_file = "ssddata/imagenet/imagenet_iso112_flip.h5"

iso_h5 = h5py.File(iso_file, 'r')
cats = [c for c in iso_h5.keys() if not c.endswith('_y')]
# cats = ['bathtub', 'bakery', 'artichoke']

downscale_ax = lambda axis, a: (
    np.moveaxis(np.moveaxis(a, axis, -1
                ).reshape(a.shape[:axis] + a.shape[axis + 1:] + (a.shape[axis] // 2, 2)
                ).mean(axis = -1),
    -1, axis))

with h5py.File(out_file, 'w') as out_h5:
    for cat in cats:
        print(f"Category: {cat}")
        nimg = iso_h5[cat + '_y'][...].sum()
        to_scale = iso_h5[cat][...][iso_h5[cat + '_y'][...].astype('bool')]
        scaled = downscale_ax(1, downscale_ax(2, to_scale))
        dset_img = out_h5.create_dataset(cat, scaled.shape, iso_h5[cat].dtype)
        dset_img[...] = scaled
        dset_y = out_h5.create_dataset(cat + '_y', (nimg,), np.bool_)
        dset_y[...] = True



# Form composites
# =====================================================

uprt_img_file = "data/imagenet/imagenet_iso224.h5"
flip_img_file = "ssddata/imagenet/imagenet_iso224_flip.h5"
out_file = "ssddata/imagenet/imagenet_flip_comp.h5"

flip_h5 = h5py.File(flip_img_file, 'r')
uprt_h5 = h5py.File(uprt_img_file, 'r')
cats = [c for c in flip_h5.keys() if not c.endswith('_y')]
# cats = ['bathtub', 'bakery', 'artichoke']

with h5py.File(out_file, 'w') as out_h5:
    for cat in cats:
        print(f"Category: {cat}")
        n_gen = 450
        flip_start = 450

        flip_imgs = flip_h5[cat][:n_gen]
        uprt_imgs = uprt_h5[cat][...][uprt_h5[cat + '_y'][...].astype('bool')]
        neg_ix = np.stack([
            np.random.choice(len(uprt_imgs), 7, replace = False)
            for _ in range(n_gen)], axis = 1)
        # downscale images
        downscale_ax = lambda axis, a: (
            np.moveaxis(np.moveaxis(a, axis, -1
                        ).reshape(a.shape[:axis] + a.shape[axis + 1:] + (a.shape[axis] // 2, 2)
                        ).mean(axis = -1),
            -1, axis))
        flip_imgs = downscale_ax(1, downscale_ax(2, flip_imgs))
        uprt_imgs = downscale_ax(1, downscale_ax(2, uprt_imgs))

        # interleave positive and negative images
        pos = np.concatenate([
            np.concatenate([           flip_imgs, uprt_imgs[neg_ix[1]]], axis = 2),
            np.concatenate([uprt_imgs[neg_ix[0]], uprt_imgs[neg_ix[2]]], axis = 2)
        ], axis = 1)
        neg = np.concatenate([
            np.concatenate([uprt_imgs[neg_ix[3]], uprt_imgs[neg_ix[5]]], axis = 2),
            np.concatenate([uprt_imgs[neg_ix[4]], uprt_imgs[neg_ix[6]]], axis = 2)
        ], axis = 1)
        all_imgs = np.hstack((pos, neg)).reshape(
            (n_gen * 2, flip_imgs.shape[1] * 2, flip_imgs.shape[2] * 2, flip_imgs.shape[3]))
        gen_ys = np.array([1, 0] * n_gen, dtype = np.bool_)


        dset_img = out_h5.create_dataset(cat, all_imgs.shape, np.float32)
        dset_img[...] = all_imgs
        dset_y = out_h5.create_dataset(cat + '_y', (n_gen * 2,), np.bool_)
        dset_y[...] = gen_ys




# Combined-regressions reconstruction plot
# =====================================================

dist_file = "ssddata/apool/enc_ign_flipcomp.h5" # flip_detection.md
focl_file = "ssddata/apool/enc_ign_flipcomp_b4.0.h5" # flip_detection.md
plot_out = "plots/runs/flip/reconstruct_112.pdf"


from plot import readouts

readout_data = readouts.readout_data(
    dist_file, focl_file,
    (0, 4, 3))


fig, ax = plt.subplots(1, 1, figsize = (7, 5))
readouts.reconstructed_bhv(ax, readout_data,
    combined_regs
)
plt.tight_layout()
plt.savefig(plot_out)
plt.show()




# TI-FC composites
# =====================================================
"""
flip_tifc:
Two intervals of a composite; upright & flipped imgs
One quadrant/image will invert between the two intervals
Which interval has the upright copy of the inverting quadrant?
manyflip_tifc:
Two intervals of a composite; upright & flipped imgs
Some quadrants will invert between the two intervals
Which interval has the upright copy of the image in a given quadrant?
manysrc_tifc
Two intervals of composites; upright & flipped imgs
One quadrant/image will invert between the two intervals,
   and all of the source images will change
Which interval has the upright copy of the inverting quadrant?
"""


uprt_img_file = "ssddata/imagenet/imagenet_iso112.h5"
flip_img_file = "ssddata/imagenet/imagenet_iso112_flip.h5"
out_file = "ssddata/imagenet/imagenet_manysrc_tifc.h5"
mode = 'manysrc'

flip_h5 = h5py.File(flip_img_file, 'r')
uprt_h5 = h5py.File(uprt_img_file, 'r')
cats = [c for c in flip_h5.keys() if not c.endswith('_y')]
# cats = ['bathtub', 'bakery', 'artichoke']

with h5py.File(out_file, 'w') as out_h5:
    for cat in cats:
        print(f"Category: {cat}")
        n_gen = 450
        start = 450

        flip_imgs = flip_h5[cat][start:start+n_gen]
        uprt_imgs = uprt_h5[cat][start:start+n_gen]

        pd_ix = np.stack([ # mode : flip, manyflip, manysrc
            np.random.choice(len(uprt_imgs), 3, replace = False)
            for _ in range(n_gen)], axis = 1)

        if mode in ['flip', 'manyflip']:
            nd_ix = pd_ix.copy() # mode: flip, manyflip
        elif mode in ['manysrc']:
            nd_ix = np.stack([ # mode : manysrc
                np.random.choice(len(uprt_imgs), 3, replace = False)
                for _ in range(n_gen)], axis = 1)

        pd_flp = np.random.choice(2, [3, n_gen], replace = True)
        if mode in ['flip', 'manysrc']:
            nd_flp = pd_flp.copy() # mode: flip, manysrc
        elif mode in ['manyflip']:
            nd_flp = np.random.choice(2, [3, n_gen], replace = True) # mode :manyflip

        pt_ix = np.arange(n_gen) # mode : all
        if mode in ['flip', 'manyflip']:
            nt_ix = np.arange(n_gen) # mode : flip, manyflip
        elif mode in ['manysrc']:
            nt_ix = np.random.permutation(n_gen) # mode : manysrc

        # form uprt and flip composites
        imgs = np.stack([uprt_imgs, flip_imgs])
        pos = np.concatenate([
            np.concatenate([         uprt_imgs[pt_ix], imgs[pd_flp[1], pd_ix[1]]], axis = 2),
            np.concatenate([imgs[pd_flp[0], pd_ix[0]], imgs[pd_flp[2], pd_ix[2]]], axis = 2)
        ], axis = 1)
        neg = np.concatenate([
            np.concatenate([         flip_imgs[nt_ix], imgs[nd_flp[1], nd_ix[1]]], axis = 2),
            np.concatenate([imgs[nd_flp[0], nd_ix[0]], imgs[nd_flp[2], nd_ix[2]]], axis = 2)
        ], axis = 1)
        # interleave positive and negative images
        all_imgs = np.hstack((pos, neg)).reshape(
            (n_gen * 2, flip_imgs.shape[1] * 2, flip_imgs.shape[2] * 2, flip_imgs.shape[3]))
        all_isflipped = np.hstack((pd_flp, nd_flp)).reshape(
            (n_gen * 2, 3))
        gen_ys = np.array([1, 0] * n_gen, dtype = np.bool_)


        dset_img = out_h5.create_dataset(cat, all_imgs.shape, np.float32)
        dset_img[...] = all_imgs
        dset_y = out_h5.create_dataset(cat + '_y', (n_gen * 2,), np.bool_)
        dset_y[...] = gen_ys
        dset_flp = out_h5.create_dataset(cat + '_isflipped.meta', (n_gen * 2, 3), np.bool_)
        dset_flp[...] = all_isflipped




# TI-FC behavior
# =====================================================

tifc_enc_file = 'ssddata/apool/enc_ign_flip_tifc.h5'
plot_out = 'plots/runs/flip/tifc_bycategory.pdf'
layer = '0.4.3'

enc_h5 = h5py.File(tifc_enc_file, 'r')


from sklearn import metrics as skmtr
compose_auc = lambda pos, neg: skmtr.roc_auc_score(
    np.concatenate([
        np.ones(pos.shape[0]),
        np.zeros(neg.shape[0])]),
    np.concatenate([
        pos,
        neg]))

with PdfPages(plot_out) as pdf:
    for i_cat, cat in enumerate(cats):
        print("Category:", cat)
        isuprt = enc_h5['y'][i_cat][...].astype('bool')
        uprt_enc = enc_h5[layer][i_cat][ isuprt]
        flip_enc = enc_h5[layer][i_cat][~isuprt]
        cat_coef = opposing_weights[f'{cat}:uprt']
        uprt_fn = (uprt_enc.mean(axis = (2,3)) * cat_coef).sum(axis = 1)
        flip_fn = (flip_enc.mean(axis = (2,3)) * cat_coef).sum(axis = 1)
        tifc_scores = uprt_fn - flip_fn

        fig, ax = plt.subplots(1, 2, figsize = (6, 3))
        ax[0].axhline(0, lw = 1, color = '.8')
        ax[0].plot(
            np.random.uniform(-0.2, 0.2, len(uprt_fn)),
            uprt_fn, 'C0o', alpha = 0.3)
        ax[0].plot(
            np.random.uniform(0.8, 1.2, len(flip_fn)),
            flip_fn, 'C3o', alpha = 0.3)
        ax[0].set_xticks([0, 1])
        ax[0].set_xticklabels(["TL: Upright", "TL: Flip"])
        ax[0].set_title(cat)
        ax[1].hist(tifc_scores)
        # ax[1].set_title(f"Pct pos: {(tifc_scores>0).mean():.3f}")
        ax[1].set_title(f"AUC: {compose_auc(uprt_fn, flip_fn)}")
        plt.tight_layout()
        pdf.savefig(); plt.close()



# TI-FC reconstruct plot
# =====================================================


dist_file = "ssddata/apool/enc_ign_manysrc_tifc.h5" # flip_detection.md
focl_file = "ssddata/apool/enc_ign_manysrc_tifc_b4.0.h5" # flip_detection.md
opposing_weights = np.load('data/models/opposed_regs_ign112_flip.npz')

from plot import readouts

src_tifc_readout_data = readouts.readout_data(
    dist_file, focl_file,
    (0, 4, 3))



diff_pct_correct = lambda fn_pos, fn_neg: (
    (fn_pos.mean(axis = (2, 3)).sum(axis = 1) -
     fn_neg.mean(axis = (2, 3)).sum(axis = 1)
     ) > 0).mean()
src_scr_dist = []; src_scr_focl = []
src_scr_fake = []; src_scr_undo = []
# for i_cat in range(len(pair_c1)):
for i_cat, cat in enumerate(cats):
    # weights = regs[f'pair{i_cat}'][..., None, None]
    weights = opposing_weights[f'{cat}:uprt'][..., None, None]
    for (cond, cond_list) in [
            ('dist', src_scr_dist), ('focl', src_scr_focl),
            ('fake', src_scr_fake), ('undo', src_scr_undo)]:
        fn_pos = src_tifc_readout_data.__dict__['pos_' + cond][i_cat] * weights
        fn_neg = src_tifc_readout_data.__dict__['neg_' + cond][i_cat] * weights
        cond_list.append(diff_pct_correct(fn_pos, fn_neg))

src_scr_dist = np.stack(src_scr_dist); src_scr_focl = np.stack(src_scr_focl)
src_scr_fake = np.stack(src_scr_fake); src_scr_undo = np.stack(src_scr_undo)


out_plot = "plots/runs/flip/tifc_reconstruct_cls.pdf"
scores = dict(
    score_dist = cls_scr_dist, score_focl = cls_scr_focl,
    score_fake = cls_scr_fake, score_undo = cls_scr_undo)

with PdfPages(out_plot) as pdf:
    fig, ax = plt.subplots(1, 1, figsize = (7, 5))
    readouts.reconstructed_bhv_plot(ax, scores)
    plt.tight_layout()
    pdf.savefig()
    plt.show()


plot_out = "plots/runs/flip/breakout_reconstruct_cls.pdf"
with PdfPages(plot_out) as pdf:
    fig, ax = plt.subplots(1, 3, figsize = (7, 3), sharey = True)
    val_aucs_arr = [val_aucs[f'pair{i}'] for i in range(len(pair_c1))]
    val_auc_rng = (0.925, 1.025)
    dist_kw = dict(color = 'k')
    focl_kw = dict(color = 'C4')
    fake_kw = dict(color = (0,0,0,0), mec = 'C4')
    undo_kw = dict(color = (0,0,0,0), mec = 'k')
    ax[0].plot(val_aucs_arr, cls_scr_dist, 'o', ms = 4, **dist_kw)
    ax[0].plot(val_aucs_arr, cls_scr_fake, 'o', ms = 4, **fake_kw)
    ax[0].set_xlim(*val_auc_rng)
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title("Dist. to Mult")
    ax[1].plot(val_aucs_arr, cls_scr_dist, 'o', ms = 4, **dist_kw)
    ax[1].plot(val_aucs_arr, cls_scr_focl, 'o', ms = 4, **focl_kw)
    ax[1].set_xlim(*val_auc_rng)
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_title("Dist. to Focl")
    ax[2].plot([], [], 'o', ms = 4, **dist_kw, label = "Distributed")
    ax[2].plot(val_aucs_arr, cls_scr_focl, 'o', ms = 4, **focl_kw, label = 'Focal')
    ax[2].plot([], [], 'o', ms = 4, **fake_kw, label = 'Multipled')
    ax[2].plot(val_aucs_arr, cls_scr_undo, 'o', ms = 4, **undo_kw, label = 'Divided')
    ax[2].set_xlim(*val_auc_rng)
    ax[2].set_ylim(-0.05, 1.05)
    ax[2].legend(loc = 'center left', bbox_to_anchor = (1, 0.5), frameon = False)
    ax[2].set_title("Focl. to Divide")
    ax[0].set_ylabel("Accuracy")
    ax[1].set_xlabel("Isolated validation AUC")
    plt.tight_layout()
    pdf.savefig()
    plt.show()

plot_out = "plots/runs/flip/breakout_reconstruct_src.pdf"
with PdfPages(plot_out) as pdf:
    fig, ax = plt.subplots(1, 3, figsize = (7, 3), sharey = True)
    val_aucs_arr = [opposing_weights[f"{cat}:uprt_auc"] for cat in cats]
    val_auc_rng = (0.45, 1.05)
    ax[0].plot(val_aucs_arr, src_scr_dist, 'o', ms = 4, **dist_kw)
    ax[0].plot(val_aucs_arr, src_scr_fake, 'o', ms = 4, **fake_kw)
    ax[0].set_xlim(*val_auc_rng)
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title("Dist. to Mult")
    ax[1].plot(val_aucs_arr, src_scr_dist, 'o', ms = 4, **dist_kw)
    ax[1].plot(val_aucs_arr, src_scr_focl, 'o', ms = 4, **focl_kw)
    ax[1].set_xlim(*val_auc_rng)
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_title("Dist. to Focl")
    ax[2].plot([], [], 'o', ms = 4, **dist_kw, label = "Distributed")
    ax[2].plot(val_aucs_arr, src_scr_focl, 'o', ms = 4, **focl_kw, label = 'Focal')
    ax[2].plot([], [], 'o', ms = 4, **fake_kw, label = 'Multipled')
    ax[2].plot(val_aucs_arr, src_scr_undo, 'o', ms = 4, **undo_kw, label = 'Divided')
    ax[2].set_xlim(*val_auc_rng)
    ax[2].set_ylim(-0.05, 1.05)
    ax[2].legend(loc = 'center left', bbox_to_anchor = (1, 0.5), frameon = False)
    ax[2].set_title("Focl. to Divide")
    ax[0].set_ylabel("Accuracy")
    ax[1].set_xlabel("Isolated validation AUC")
    plt.tight_layout()
    pdf.savefig()
    plt.show()



# TI-FC behavior change
# =====================================================
"""
Sort of makes sense that gain wouldn't make a difference
in the TI-FC setup : already what's happening is that
the averaged representation is shifting towards `uprt`
when the image changes --- there's nothing to "overcome".

Drift diffusion model / Noise?
"""

bhv_change = np.zeros([len(cats), 2, 2])
dist_acc = np.zeros(len(cats))
for i_cat, cat in enumerate(cats):
    isuprt = enc_h5['y'][i_cat][...].astype('bool')
    uprt_enc = enc_h5[layer][i_cat][ isuprt]
    flip_enc = enc_h5[layer][i_cat][~isuprt]
    cat_coef = opposing_weights[f'{cat}:uprt']
    uprt_fn_aftr = (tifc_readout_data.pos_fake[i_cat].mean(axis = (2,3)) * cat_coef).sum(axis = 1)
    flip_fn_aftr = (tifc_readout_data.neg_fake[i_cat].mean(axis = (2,3)) * cat_coef).sum(axis = 1)
    uprt_fn_befr = (tifc_readout_data.pos_dist[i_cat].mean(axis = (2,3)) * cat_coef).sum(axis = 1)
    flip_fn_befr = (tifc_readout_data.neg_dist[i_cat].mean(axis = (2,3)) * cat_coef).sum(axis = 1)
    pred_aftr = (uprt_fn_aftr - flip_fn_aftr) > 0
    pred_befr = (uprt_fn_befr - flip_fn_befr) > 0
    bhv_change[i_cat] = [
        [(pred_aftr)[ pred_befr].mean(), (~pred_aftr)[ pred_befr].mean()],
        [(pred_aftr)[~pred_befr].mean(), (~pred_aftr)[~pred_befr].mean()]]
    # bhv_change[i_cat] = [
    #     [(pred_aftr &  pred_befr).mean(), (~pred_aftr &  pred_befr).mean()],
    #     [(pred_aftr & ~pred_befr).mean(), (~pred_aftr & ~pred_befr).mean()]]
    dist_acc[i_cat] = pred_befr.mean()


fig, ax = plt.subplots(1, 4, figsize = (10, 3), sharey = True)
val_aucs_uprt = [val_aucs[f"{cat}:uprt"] for cat in cats]
for i, ix in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
    ax[i].plot(val_aucs_uprt, bhv_change[:, ix[0], ix[1]], 'ko', ms = 4)
    ax[i].set_xlim(-0.05, 1.05)
    ax[i].set_ylim(-0.05, 1.05)
    ax[i].set_title('TF'[ix[0]] + " to " + 'TF'[ix[1]])
ax[0].set_xlabel("Accuracy (Dist)")
ax[0].set_ylabel("Percent shift")
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1, 4, figsize = (10, 3), sharey = True)
for i, ix in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
    ax[i].hist(bhv_change[:, ix[0], ix[1]],
        bins = np.linspace(0, 1, 30))
    ax[i].set_xlim(0, 1)
    ax[i].set_ylim(0, len(cats))
    ax[i].set_title('TF'[ix[0]] + " to " + 'TF'[ix[1]])
plt.tight_layout()
plt.show()







# Classwise TIFC Composites
# =====================================================
"""
cls_tifc:
Two intervals of composites; Banana and Greenhouse images
All of the quadrants will change between intervals, but only
    one will switch class
Which interval has the Greenhouse copy of the inverting quadrant?
"""


iso_file = "ssddata/imagenet/imagenet_iso112.h5"
out_file = "ssddata/imagenet/imagenet_cls_tifc.h5"

iso_h5 = h5py.File(iso_file, 'r')
cats = [c for c in iso_h5.keys() if not c.endswith('_y')]
n_gen = 450
start = 450

pair_c1 = np.arange(0, (len(cats) // 2) * 2, 2)
pair_c2 = pair_c1 + 1

with h5py.File(out_file, 'w') as out_h5:
    for i_pair, (c1, c2) in enumerate(zip(pair_c1, pair_c2)):
        print(f"Category: pair{i_pair}")
        c1_imgs = iso_h5[cats[c1]][start:start+n_gen]
        c2_imgs = iso_h5[cats[c2]][start:start+n_gen]

        d_ix = np.stack([
            np.random.choice(n_gen, [3, 2], replace = False)
            for _ in range(n_gen)], axis = 1)
        pd_ix = d_ix[:, :, 0]
        nd_ix = d_ix[:, :, 1]
        d_cls = np.random.choice(2, [3, n_gen], replace = True)
        nt_ix = np.random.permutation(n_gen)

        imgs = np.stack([c1_imgs, c2_imgs])
        pos = np.concatenate([
            np.concatenate([                 c1_imgs, imgs[d_cls[1], pd_ix[1]]], axis = 2),
            np.concatenate([imgs[d_cls[0], pd_ix[0]], imgs[d_cls[2], pd_ix[2]]], axis = 2)
        ], axis = 1)
        neg = np.concatenate([
            np.concatenate([          c2_imgs[nt_ix], imgs[d_cls[1], nd_ix[1]]], axis = 2),
            np.concatenate([imgs[d_cls[0], nd_ix[0]], imgs[d_cls[2], nd_ix[2]]], axis = 2)
        ], axis = 1)
        # interleave positive and negative images
        all_imgs = np.hstack((pos, neg)).reshape(
            (n_gen * 2, c1_imgs.shape[1] * 2, c1_imgs.shape[2] * 2, c1_imgs.shape[3]))
        gen_ys = np.array([1, 0] * n_gen, dtype = np.bool_)

        dset_img = out_h5.create_dataset(f"pair{i_pair}", all_imgs.shape, np.float32)
        dset_img[...] = all_imgs
        dset_y = out_h5.create_dataset(f"pair{i_pair}_y", (n_gen * 2,), np.bool_)
        dset_y[...] = gen_ys
        dset_flp = out_h5.create_dataset(f"pair{i_pair}_cls.meta", (n_gen, 3), np.bool_)
        dset_flp[...] = d_cls.T



# Classwise TIFC regressions
# =====================================================



iso_encodings = "ssddata/apool/enc_ign_iso112.h5" # flip_detection.md
reg_performance_plot = "plots/runs/flip/reg_performance_cls.pdf"
reg_out = 'data/models/regs_ign112_pair.npz'
layer = '0.4.3'
n_trn_each = 400
n_val_each = 200

iso_h5 = h5py.File(iso_encodings, 'r')

regs = {}
val_aucs = {}
with PdfPages(reg_performance_plot) as pdf:
    for i_pair, (c1, c2) in enumerate(zip(pair_c1, pair_c2)):
        print(f"Category: pair{i_pair}")

        c1_y = iso_h5['y'][c1].astype('bool')
        trn_c1_feat = iso_h5[layer][c1][...][c1_y][:n_trn_each]
        val_c1_feat = iso_h5[layer][c1][...][c1_y][n_trn_each:n_trn_each + n_val_each]

        c2_y = iso_h5['y'][c2].astype('bool')
        trn_c2_feat = iso_h5[layer][c2][...][c2_y][:n_trn_each]
        val_c2_feat = iso_h5[layer][c2][...][c2_y][n_trn_each:n_trn_each + n_val_each]

        trn_x = np.hstack([trn_c1_feat, trn_c2_feat]
            ).reshape((n_trn_each*2,) + trn_c1_feat.shape[1:]
            ).mean(axis = (2, 3))
        val_x = np.hstack([val_c1_feat, val_c2_feat]
            ).reshape((n_val_each*2,) + val_c1_feat.shape[1:]
            ).mean(axis = (2, 3))
        trn_y = np.array([1, 0] * n_trn_each).astype('bool')
        val_y = np.array([1, 0] * n_val_each).astype('bool')


        reg = LogisticRegression(
            solver = 'liblinear',
            max_iter = 1000,
            fit_intercept = False)
        reg.fit(trn_x, trn_y)
        trn_fn = (reg.coef_ * trn_x).sum(axis = 1)
        val_fn = (reg.coef_ * val_x).sum(axis = 1)
        regs[f'pair{i_pair}'] = reg.coef_

        fig, ax = plt.subplots(1, 2, figsize = (6, 3), sharey = True)
        for i, (fn, y, ttl) in enumerate([
            (trn_fn, trn_y, 'Train'),
            (val_fn, val_y, 'Val')]):
            ax[i].plot(
                np.random.uniform(-0.1, 0.1, y.sum()), fn[y],
                'C0o', ms = 5, alpha = 0.5)
            ax[i].plot(
                np.random.uniform(0.9, 1.1, (~y).sum()), fn[~y],
                'C3o', ms = 5, alpha = 0.5)
            ax[i].set_title(f"{ttl} AUC: {roc_auc_score(y, fn):.3f}")
            if ttl == 'Val': val_aucs[f'pair{i_pair}'] = roc_auc_score(y, fn)
        plt.suptitle(f"pair{i_pair} - {cats[c1]} v. {cats[c2]}")
        plt.tight_layout(rect = (0, 0, 1, 0.95))
        pdf.savefig(); plt.close()


    fig, ax = plt.subplots(1, 1, figsize = (6, 3))
    ax.scatter(
        np.arange(len(val_aucs)), val_aucs.values(), s = 20, color = 'k')
    ax.set_xticks(np.arange(len(val_aucs)))
    ax.set_xticklabels(val_aucs.keys(), rotation = 90,
        horizontalalignment = 'right')
    ax.set_ylabel("Val AUC")
    plt.tight_layout()
    pdf.savefig(); plt.close()


regs = {**regs, **{f'{k}_auc':v for k,v in val_aucs.items()}}
np.savez(reg_out, **regs)





















