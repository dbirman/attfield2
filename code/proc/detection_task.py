from proc import network_manager as nm
from proc import lsq_fields
from proc import video_gen

from sklearn.linear_model import LogisticRegression
from skimage.transform import downscale_local_mean
import scipy.ndimage.interpolation as scimg
import sklearn.metrics as sk_metric
from torch import nn
import pandas as pd
import numpy as np
import skvideo.io
import operator
import torch
import h5py
import tqdm
import os


RAND_LOC = -1
class QuadAttention(nm.LayerMod):

    def __init__(self, beta, locs, profile = 'flat'):
        '''
        ### Arguments
        - `task` --- A FourWayObjectDetectionTask object
        - `locs` --- List of integers with length of expected
            batch size, or (RAW PYTHON) integer (non-numpy).
        - `profile` --- `'flat'` to uniformly amplify gain
            in the selected quad, or `'gauss'` to apply gaussian
            gain centered at that quad and clipped at its edges.
        '''
        self.locs = locs
        self.beta = beta
        self.profile = profile

    def pre_layer(self, inp, *args, **kwargs):
        """
        ### Arguments
        - `inp` --- Main layer input, of shape (batch, channel, row, col)
        """
        scale = self.scale_array(inp.size())
        if inp.is_cuda:
            scaled = inp * scale.to(device = inp.get_device())
        else:
            scaled = inp * scale
        return (scaled,) + args, kwargs, None

    def scale_array(self, shape):
        if shape[-2] % 2 != 0 or shape[-1] % 2 != 0:
            raise ValueError("QuadAttention only valid " + 
                             "for even-sized images") 
        if isinstance(self.locs, int):
            locs = [self.locs] * shape[0]
        else:
            if len(self.locs) != shape[0]:
                raise ValueError("Input to QuadAttention " + 
                             "has {} frames ".format(shape[0]) + 
                             "when given `locs` only has " + 
                             "{} values.".format(len(self.locs))) 
            locs = self.locs
        row_n = shape[-2]//2
        col_n = shape[-1]//2
        imgs = np.concatenate(
            [self.gain_profile(row_n, col_n) + 1, 
             np.ones((3, row_n, col_n))])
        gain = [FourWayObjectDetectionTask._arrange_grid(imgs, locs[i])[0]
                for i in range(shape[0])]
        gain = np.array(gain)
        return torch.tensor(gain[:, np.newaxis, ...]).float()

    def gain_profile(self, row_n, col_n):
        if self.profile == 'flat':
            return np.ones((1, row_n, col_n)) * (self.beta-1)
        elif self.profile == 'gauss':
            gauss = lsq_fields.gauss_with_params_torch(
                col_n, row_n, [col_n/2], [row_n/2],
                [750*(row_n/112)**2], [self.beta-1])
            return gauss


class QuadAttentionFullShuf(QuadAttention):
    '''A control condition on QuadAttention in which pixels of
    the gain filter are shuffled across the entire image.'''
    def scale_array(self, shape):
        arr = QuadAttention.scale_array(self, shape)
        arr_flat = arr.view(-1)[torch.randperm(arr.numel())]
        return arr_flat.view(*arr.shape)

class QuadAttentionCornerShuf(QuadAttention):
    '''A control condition on QuadAttention in which pixels of
    the gain filter are shuffled within the attended corner.'''
    def gain_profile(self, row_n, col_n):
        arr = QuadAttention.gain_profile(self, row_n, col_n)
        arr_flat = torch.reshape(arr, (-1,))
        arr_flat = arr_flat[torch.randperm(arr_flat.size()[0])]
        return torch.reshape(arr_flat, arr.shape) 



class LayerBypass(nm.LayerMod):

    def pre_layer(self, *args, **kwargs):
        return args, kwargs, args[0]
    def post_layer(self, outputs, cache):
        return cache


class ClassifierMod(nm.LayerMod):
    def __init__(self, w, b):
        self.w = w
        self.b = b
    
    def post_layer(self, outputs, cache):
        return self.decision_function(outputs)

    def predict(self, encodings):
        return self.decision_function(encodings) > 0

    def decision_function(self, encodings):
        if isinstance(encodings, np.ndarray):
            return np.dot(encodings, self.w.T[:, 0]) + self.b
        else:
            w = torch.tensor(self.w.T).float()
            b = torch.tensor(self.b).float()
            return torch.mm(encodings, w) + b

    def predict_on_fn(self, decision_fn):
        return decision_fn > 0


def load_logregs(filename):
    archive = np.load(filename)
    if archive['type'] == 'mod':
        return {
            c: ClassifierMod(archive[c+'_w'], archive[c+'_b'])
            for c in archive['categories']
        }
    else:
        regs = {c: LogisticRegression()
                for c in archive['categories']}
        for c in regs:
            regs[c].coef_ = archive[c+'_c']
            regs[c].intercept_ = archive[c+'_i']




def save_logregs(filename, logregs):
    logreg_type = type(logregs[next(iter(logregs.keys()))])
    if logreg_type is ClassifierMod:
        np.savez(filename,
            categories = list(logregs.keys()),
            **{c+"_w": logregs[c].w for c in logregs},
            **{c+"_b": logregs[c].b for c in logregs},
            type = 'mod')
    else:
        np.savez(filename,
            categories = list(logregs.keys()),
            **{c+"_c": logregs[c].coef_ for c in logregs},
            **{c+"_i": logregs[c].intercept_ for c in logregs},
            type = 'sk')


def multi_decision(regs, encodings):
    '''
    Utility function for running the decision function of each
    category regression'''
    return {c: regs[c].decision_function(encodings[c])
            for c in encodings}

def by_cat(guide_dict, func):
    '''
    Utility function for running functions across categories 
    of a task.'''
    return {c: func(c) for c in guide_dict}























class IsolatedObjectDetectionTask():
    def __init__(self, h5file, image_size = 224,
        whitelist = None, _seed = 1):
        '''
        ### Arguments
        - `index` --- Path to an index.csv file
        '''
        self.h5 = h5py.File(h5file, 'r')
        self.seed = _seed

        if whitelist is not None:
            self.cats = [c for c in self.h5.keys() if c in whitelist]
        else:
            self.cats = list(self.h5.keys())

        h5_dim = self.h5[self.cats[0]].shape[1]
        self.image_scale = h5_dim//image_size
        self.image_size = h5_dim//self.image_scale


    def train_set(self, cat, n, **kwargs):
        '''
        Pull images from ILSVRC2014 validation set of a given
        category. These are from the validation set so that
        earlier layers of the model will not have been trained
        on them.
        ### Arguments
        - `cat` --- A category identifier from the self.cats
        - `n` --- Number of positive images to include
        ### Returns
        - `imgs` --- A collection of training images for a
            binary classification of size (2*n, 3, 224, 224).
            These will contain `n` images containing the object
            category and `n` not containing the object category
            in random order.
        - 'ys' --- An array of shape (2*n,) taking values either
            1 or 0 to indicate whether each image in `imgs`
            contains an object of the category
        '''
        if cat is None:
            # Run _load_set for each category and arrange the
            # returned values into dictionaries
            rets = None
            for c in self.cats:
                curr_ret = self._load_set(c, n, 1, **kwargs)
                if rets is None:
                    rets = tuple({} for i in range(len(curr_ret)))
                for i in range(len(curr_ret)):
                    rets[i][c] = curr_ret[i]
            return rets
        else:
            return self._load_set(cat, n, 1, **kwargs)


    def _load_images(self, cat, which, read_order):
        # Tuple indexing does weird things
        if isinstance(which, tuple): which = list(which)

        if hasattr(which, '__iter__'):
            if len(which) == 0:
                h5_dim = self.h5[self.cats[0]].shape[1]
                ret_shp = [0, h5_dim//self.image_scale,
                           h5_dim//self.image_scale, 3]
                return np.empty(ret_shp)
            if min(which) != 0: raise NotImplementedError

            # Reading from h5 is only supported in blocks
            pull_limit = max(which)
            if read_order == -1:
                imgs = self.h5[cat][-pull_limit-1:][::-1]
            else:
                imgs = self.h5[cat][:pull_limit+1]
            # Now do the full indexing
            imgs = imgs[which]
        else:
            ord_offset = -1 if read_order is -1 else 0
            imgs = self.h5[cat][which + ord_offset][np.newaxis, ...]

        # Scale appropriately
        factors = (1, self.image_scale, self.image_scale, 1)
        return downscale_local_mean(imgs, factors)


    
    def _negative_imgs(self, cat, n, read_order):

        # Calculate the number of images required from each category
        cat_n = n // (len(self.cats)-1)
        n_extra = n % (len(self.cats)-1)
        neg_cats = [c for c in self.cats if c != cat]

        # Pull the images from the HDF5 archive
        imgs = np.concatenate(
            [self._load_images(c, range(cat_n), read_order)
             for c in neg_cats] + 
            [self._load_images(c, cat_n, read_order)
             for c in neg_cats[:n_extra]]
            )

        # (Deterministically) shuffle the images
        rng = np.random.RandomState(self.seed)
        imgs = imgs[rng.permutation(len(imgs))]

        return imgs


    def _load_set(self, cat, n, read_order, cache = None,
                  shuffle = False):
        '''
        Pull positive and negative examples of a given category
        These will be the first `n` images of the category, as ordered
        by ncol
        ### Arguments
        - `cat` --- A category identifier from the self.cats
        - `n` --- Number of positive images to include
        - `read_order` --- 1 to read from the front of the archive
            or -1 to read from the back of the archive.
        ### Returns
        - `imgs` --- A collection of images for a binary
            classification. Will be of size (2*n, 3, 224, 224).
            These will contain `n` images containing the object
            category and `n` not containing the object category
            in random order.
        - 'ys' --- An array of shape (2*n,) taking values either
            1 or 0 to indicate whether each image in `imgs`
            contains an object of the category
        '''

        if cache is not None:
            cachefile = os.path.join(cache, cat + str(n) + ncol + 
                                     str(self.image_size) + ".npz")

        if cache is not None and os.path.exists(cachefile):
            archive = np.load(cachefile)
            imgs = archive['imgs']
            ys = archive['ys']

        else:

            true_images = self._load_images(cat, range(n), read_order)
            false_images = self._negative_imgs(cat, n, read_order)
            imgs = np.concatenate([true_images, false_images])
            ys = np.concatenate([np.ones(n, dtype = bool),
                                 np.zeros(n, dtype = bool)])

            if cache is not None:
                np.savez(cachefile, imgs = imgs, ys = ys)

        # Shuffle
        if shuffle:
            shuf = np.random.permutation(2*n)
            ys = ys[shuf]
            imgs = imgs[shuf]

        # Normalize how CORNet expects, and shift channel
        # dimension to torch format
        imgs = torch.tensor(np.moveaxis(imgs, -1, 1)).float()
        imgs = video_gen.batch_transform(imgs, video_gen.normalize)
        return imgs, ys


    def val_size(self, train_size):
        return len(self.h5[self.cats[0]]) - train_size


    def val_set(self, cat, n, **kwargs):
        '''See documentation for train_set'''
        if cat is None:
            # Run _load_set for each category and arrange the
            # returned values into dictionaries
            rets = None
            for c in self.cats:
                curr_ret = self._load_set(c, n, -1, **kwargs)
                if rets is None:
                    rets = tuple({} for i in range(len(curr_ret)))
                for i in range(len(curr_ret)):
                    rets[i][c] = curr_ret[i]
            return rets
        else:
            return self._load_set(cat, n, -1, **kwargs)



class FourWayObjectDetectionTask(IsolatedObjectDetectionTask):

    def __init__(self, *args, **kw):
        '''Same signature as IsolatedObjectDetectionTask.__init__'''
        super(FourWayObjectDetectionTask, self).__init__(*args, **kw)
        # Images returned are twice the 'requested' size
        self.image_size *= 2

    def _load_set(self, cat, n, read_order, cache = None,
        loc = -1, shuffle = False):
        '''
        Pull positive and negative examples of a given category
        These will be the first `n` images of the category, as ordered
        by ncol
        ### Arguments
        - `cat` --- A category identifier from the self.cats
        - `n` --- Number of positive images to include
        - `read_order` --- 1 to read from the front of the archive
            or -1 to read from the back of the archive.
        - `loc` --- Position where the first image from `imgs` should
            go in the grid. The mapping is
            - 0: Top Left
            - 1: Top Right
            - 2: Bottom Left
            - 3: Bottom Right
            - -1: Random
        - `shuffle` --- If true, `imgs` will have all the positive 
            examples grouped together in the beginning of the first
            dimension. If false then `imgs` will be shuffled along 
            the first dimension.
        ### Returns
        - `imgs` --- A collection of images for a binary
            classification. Will be of size (2*n, 3, 224, 224).
            These will contain `n` images containing the object
            category and `n` not containing the object category
            in random order.
        - 'ys' --- An array of shape (2*n,) taking values either
            1 or 0 to indicate whether each image in `imgs`
            contains an object of the category
        '''
        if cache is not None:
            cachefile = os.path.join(cache, "4W" + str(loc) + cat +  
                     str(n) + ncol + str(self.image_size) + ".npz")

        if cache is not None and os.path.exists(cachefile):
            archive = np.load(cachefile)
            imgs = archive['imgs']
            ys = archive['ys']
            locs = archive['locs']

        else:
            # Identify the first `n` images in this category
            true_images = self._load_images(cat, range(n), read_order)
            false_images = self._negative_imgs(cat, 7*n, read_order)

            # Split up the false images into those that will be in
            # grids with a true image and those that will be in
            # negative grids
            false_images_posarr = false_images[:3*n]
            false_images = false_images[3*n:]

            pos_imgs = []
            neg_imgs = []
            pos_locs = []
            neg_locs = []
            for i in range(n):
                array_posimg = false_images_posarr[3*i:3*(i+1)]
                pos_img, pos_loc = self._arrange_grid(
                    np.concatenate([[true_images[i]], array_posimg]),
                    loc = loc)
                neg_img, neg_loc = self._arrange_grid(
                    false_images[4*i:4*(i+1)],
                    loc = loc)
                pos_imgs.append(pos_img)
                neg_imgs.append(neg_img)
                pos_locs.append(pos_loc)
                neg_locs.append(neg_loc)

            imgs = np.concatenate([pos_imgs, neg_imgs])
            ys = np.concatenate([np.ones(n, dtype = bool),
                                 np.zeros(n, dtype = bool)])
            locs = np.concatenate([pos_locs, neg_locs])

            if cache is not None:
                np.savez(cachefile, imgs = imgs, ys = ys, locs = locs)

        # Shuffle
        if shuffle:
            shuf = np.random.permutation(2*n)
            ys = ys[shuf]
            imgs = imgs[shuf]
            locs = locs[shuf]

        # Normalize how CORNet expects, and shift channel
        # dimension to torch format
        imgs = torch.tensor(np.moveaxis(imgs, -1, 1)).float()
        imgs = video_gen.batch_transform(imgs, video_gen.normalize)
        return imgs, ys, locs


    @staticmethod
    def _arrange_grid(imgs, loc = -1):
        '''
        Arrange four images into 2x2 grid, with the first image at a
        specified location and others distributed randomly.
        ### Arguments
        - `imgs` --- A list or array of four images, each with shape
            (h, w, channel) to be arranged into a 2x2 grid.
        - `loc` --- Position where the first image from `imgs` should
            go in the grid. The mapping is
            - 0: Top Left
            - 1: Top Right
            - 2: Bottom Left
            - 3: Bottom Right
            - -1: Random
        ### Returns
        - `grid` --- The image grid, of shape (2*h, 2*w, channel)
        - `loc` --- The passed parameter for `loc`, or the equivalent
            mapping of what was randomly assigned otherwise.
        '''
        # Rearrange `imgs` to put the first image in the right spot
        # and to randomize position of the other images

        if loc == -1:
            loc = np.random.randint(4)

        rand = np.array(imgs)[np.random.permutation(3)+1]
        if loc == 0:
            imgs = [imgs[0], rand[0], rand[1], rand[2]]
        elif loc == 1:
            imgs = [rand[0], imgs[0], rand[1], rand[2]]
        elif loc == 2:
            imgs = [rand[0], rand[1], imgs[0], rand[2]]
        elif loc == 3:
            imgs = [rand[0], rand[1], rand[2], imgs[0]]

        return np.concatenate([
                np.concatenate([imgs[0], imgs[1]], axis = 1),
                np.concatenate([imgs[2], imgs[3]], axis = 1)],
            axis = 0), loc


    def val_size(self, train_size):
        if len(self.cats)-1 > 7:
            # Negative image arrays will be distributed around
            # different categories enough that the number of
            # positive images necessary will dominate
            return len(self.h5[self.cats[0]]) - train_size
        else:
            # Not enough categories to spread negative examples
            # across, the need for 7 times as many negative
            # images dominates
            remaining = len(self.h5[self.cats[0]]) - 7*train_size
            return remaining // 7




























def score_logregs(logregs, encodings, ys):
    # Get predictions and decision function valueås
    preds = {}
    decision = {}
    for c in encodings:        
        preds[c] = logregs[c].predict(encodings[c])
        decision[c] = logregs[c].decision_function(encodings[c])

    # Outsource the computation of scores
    overall, per_cat = score_logregs_precomputed(preds, decision, ys)
    return preds, overall, per_cat


def score_logregs_precomputed(preds, decision, ys):
    per_cat = {}
    for c in encodings:
        per_cat[c] = {
            'acc': sk_metric.accuracy_score(ys[c], preds[c]),
            'precision': sk_metric.precision_score(ys[c], preds[c]),
            'recall': sk_metric.recall_score(ys[c], preds[c]),
            'auc': sk_metric.roc_auc_score(ys[c], decision[c])
        }

    ys_concat = np.concatenate([ys[c] for c in preds])
    preds_concat = np.concatenate([preds[c] for c in preds])
    decision_concat = np.concatenate([decision[c] for c in preds])
    overall = {
        'acc': sk_metric.accuracy_score(
            ys_concat, preds_concat),
        'precision': sk_metric.accuracy_score(
            ys_concat, preds_concat),
        'recall': sk_metric.accuracy_score(
            ys_concat, preds_concat),
        'auc': sk_metric.roc_auc_score(
            ys_concat, decision_concat)
    }
    return overall, per_cat





def model_encodings(model, decoders, imgs, mods = {}):
    '''
    ### Arguments
    - `model` --- A PyTorch model, presumably CORNet
    - `decoders` --- A list of layer indices (tuples of ints)
        giving layers that perform 'decoding' functions in
        the model and so should be skipped.
    - `imgs` --- Either an array of shape (n, h, w, c) giving
        the images to be encoded, or a dictionary mapping
        categories to arrays of that type.
    ### Returns
    - `encodings` --- Model encodings of `imgs`. Either an array
        of shape (n_imgs, n_features) if `imgs` was an array, or
        a dictionary mapping categories to such arrays of `imgs`
        was passed as a dictionary.
    '''
    if isinstance(imgs, dict):
        return {c: model_encodings(model, decoders, imgs[c], mods)
                for c in imgs}
    else:
        decoder_bypass = {l: LayerBypass() for l in decoders}
        mgr = nm.NetworkManager.assemble(model, imgs,
                mods = {**mods, **decoder_bypass})
        return mgr.computed[(0,)].cpu().detach().numpy()


def fit_logregs(model, decoders, task, mods = {},
                train_size = 30, shuffle = False):
    '''
    Fit linear decoders to a model and a task.
    ### Arguments
    - `model` --- The neural network that will be performing the task
    - `decoders` --- A list of layer indexes (tuples of ints) that
        perform decoding or behavioral task operations. These will be 
        skipped when computing the model on task inputs and be replaced
        by a task-specific logistic regression decoder for each class.
        It is expected that the model will give output in a flat format
        with shape (batch, features). If this is not the case once
        decoders are bypassed then a LayerMod should be included on 
        layer (0,) that will flatten the output.
    - `mods` --- Any LayerMods to be applied to `model`, for example
        attention mechanisms could be passed here if decoders need to
        adapt to them.
    '''

    # Compute how the network embeds the images in feature space
    encodings = {}
    all_ys = {}
    all_imgs = {}
    skregs = {}
    regmods = {}

    for c in task.cats:
        imgs, ys = task.train_set(c, train_size, shuffle=shuffle)[:2]
        all_ys[c] = ys
        all_imgs[c] = imgs
        encodings[c] = model_encodings(
            model, decoders, imgs, mods = mods)

        skregs[c] = LogisticRegression(solver = 'liblinear')
        skregs[c].fit(encodings[c], ys)
        regmods[c] = ClassifierMod(
            skregs[c].coef_,
            skregs[c].intercept_)

    return encodings, skregs, regmods, all_imgs, all_ys



def voxel_decision_grads(voxels, model, decoders, imgs, ys, regs,
                         mods = {}):
    grads = {}
    acts = {}

    for c in regs:
        # Run encodings with voxel gradients tracked
        decoder_bypass = {l: LayerBypass() for l in decoders}
        logreg = {(0,): regs[c]}
        mgr = nm.NetworkManager.assemble(model, imgs[c],
                mods = nm.mod_merge(mods, decoder_bypass, logreg),
                with_grad = True)

        # Backprop the gradients to the voxels
        back_mask = torch.ones_like(mgr.computed[(0,)])
        mgr.computed[(0,)].backward(back_mask)

        # Make space for the results of this run
        for res_dict in (grads, acts):
            res_dict[c] = {}

        # Pull gradients from the model and score its performance
        for l in voxels:
            batch = mgr.computed[l].cpu()
            grads[c][l] = voxels[l].index_into_batch(
                batch.grad_nonleaf.detach().numpy(), i_vox = None)
            acts[c][l] = voxels[l].index_into_batch(
                batch.detach().numpy(), i_vox = None)

    return grads, acts



















def generate_dataset(imagenet_index, output, image_size = 224):
    '''Process ImageNet data down to an archived dataset
    that is quickly readable for training.
    ### Arguments
    - `output` --- HDF5 filename to output to. This will contain
        one table for each image category.
    '''

    base_dir = os.path.dirname(imagenet_index)
    index = pd.read_csv(imagenet_index)
    cats = index['category'].unique()

    max_ns = [max(index.iloc[np.where(index['category'] == c)]['n'])
              for c in cats]
    limit_n = min(max_ns)

    with h5py.File(output, "w") as f:
        for cat in tqdm.tqdm(cats, desc = "Category"):
            dset = f.create_dataset(cat,
                (limit_n, image_size, image_size, 3),
                dtype = 'i')

            cat_images = index['category'] == cat
            first_n = index['n'] < limit_n
            idxs = np.where(cat_images & first_n)[0]
            paths = index['path'].iloc[idxs]

            for i, pth in enumerate(tqdm.tqdm(paths, desc = "   Image")):
                img = skvideo.io.vread(os.path.join(base_dir, pth))
                img = img[0]
                factor = image_size/min(img.shape[0], img.shape[1])
                img = scimg.zoom(img, [factor, factor, 1], order = 1)

                # Center crop
                # See https://stackoverflow.com/a/50322574/1888160
                bounding = (image_size, image_size, 3)
                start = tuple(map(lambda a, da: a//2-da//2, img.shape,
                                  bounding))
                end = tuple(map(operator.add, start, bounding))
                slices = tuple(map(slice, start, end))
                dset[i, ...] = img[slices]


def _neg_iter(task, cat, seed = 1):
    max_n = task.h5[task.cats[0]].shape[0]
    rng = np.random.RandomState(seed)
    neg_cats = [c for c in task.cats if c != cat]
    order = rng.permutation(max_n)
    while True:
        for i in range(max_n):
            for c in rng.permutation(neg_cats):
                yield task._load_images(c, order[i], read_order = 1)


def cache_iso_task(imagenet_h5, output, image_size = 224, seed = 1):
    '''Process an imagenet dataset into a format where it can
    be almost instantaneously read as IsolatedObjectDetectionTask data.'''

    task = IsolatedObjectDetectionTask(imagenet_h5, image_size)
    with h5py.File(output, "w") as f:
         for i, cat in enumerate(tqdm.tqdm(task.cats, desc = "Category")):

            # Load and save images
            negs = _neg_iter(task, cat, seed = seed)
            cat_n = task.h5[cat].shape[0]
            dset = f.create_dataset(cat,
                (2*cat_n, image_size, image_size, 3),
                dtype = 'i')
            dset[::2] = task._load_images(cat, range(cat_n),
                                             read_order = 1)
            neg_imgs = [next(negs)[0] for _ in range(cat_n)]
            dset[1::2] = np.array(neg_imgs)

            # Save classification targets
            dset = f.create_dataset(cat + '_y',
                (2*cat_n,), dtype = bool)
            dset[::2] = np.ones(cat_n, dtype = bool)
            dset[1::2] = np.zeros(cat_n, dtype = bool)




def cache_four_task(imagenet_h5, output, image_size = 112, seed = 1, loc = -1):
    task = FourWayObjectDetectionTask(imagenet_h5, image_size)
    with h5py.File(output, "w") as f:
         for i, cat in enumerate(tqdm.tqdm(task.cats, desc = "Category")):

            # Load and save images
            negs = _neg_iter(task, cat, seed = seed)
            cat_n = task.h5[cat].shape[0]

            true_images = task._load_images(cat, range(cat_n), read_order=1)
            false_images = [next(negs)[0]
                for _ in tqdm.trange(7*cat_n, desc = 'Negative')]

            false_images_posarr = false_images[:3*cat_n]
            false_images = false_images[3*cat_n:]

            imgs = f.create_dataset(cat,
                (2*cat_n, 2*image_size, 2*image_size, 3),
                dtype = 'i')
            ys = f.create_dataset(cat + '_y',
                (2*cat_n,), dtype = bool)
            locs = f.create_dataset(cat + '_loc',
                (2*cat_n,), dtype = 'i')

            for i in range(cat_n):
                array_posimg = false_images_posarr[3*i:3*(i+1)]
                imgs[2*i], locs[2*i] = task._arrange_grid(
                    np.concatenate([[true_images[i]], array_posimg]),
                    loc = loc)
                ys[2*i] = True
                imgs[2*i+1], locs[2*i+1] = task._arrange_grid(
                    false_images[4*i:4*(i+1)],
                    loc = loc)
                ys[2*i+1] = False



class DistilledDetectionTask(IsolatedObjectDetectionTask):
    def __init__(self, h5file, whitelist = None):
        '''
        ### Arguments
        - `index` --- Path to an index.csv file
        '''
        self.h5 = h5py.File(h5file, 'r')

        if whitelist is not None:
            self.cats = [c for c in self.h5.keys() if c in whitelist]
        else:
            self.cats = list(self.h5.keys())
            self.cats = [c for c in self.cats
                         if ((not c.endswith('_y')) and
                             (not c.endswith('_loc')))]

        self.image_size = self.h5[self.cats[0]].shape[1]


    def _load_set(self, cat, n, read_order, shuffle = False,
                  load_imgs = True):
        '''
        Pull positive and negative examples of a given category
        These will be the first `n` images of the category, as ordered
        by ncol
        ### Arguments
        - `cat` --- A category identifier from the self.cats
        - `n` --- Number of positive images to include
        - `read_order` --- 1 to read from the front of the archive
            or -1 to read from the back of the archive.
        ### Returns
        - `imgs` --- A collection of images for a binary
            classification. Will be of size (2*n, 3, 224, 224).
            These will contain `n` images containing the object
            category and `n` not containing the object category
            in random order.
        - 'ys' --- An array of shape (2*n,) taking values either
            1 or 0 to indicate whether each image in `imgs`
            contains an object of the category
        '''

        if isinstance(n, tuple):
            start = n[0]
            stop = n[1]
        else:
            start = 0
            stop = n

        if read_order == 1:
            if load_imgs:
                imgs = self.h5[cat][start:stop]
            ys = self.h5[cat+'_y'][start:stop]
            if cat+'_loc' in self.h5.keys():
                locs =  self.h5[cat+'_loc'][start:stop]
        else:
            if load_imgs:
                imgs = self.h5[cat][-stop-1:-start-1]
            ys = self.h5[cat+'_y'][-stop-1:-start-1]
            if cat+'_loc' in self.h5.keys():
                locs =  self.h5[cat+'_loc'][-stop-1:-start-1]

        # Shuffle
        if shuffle:
            shuf = np.random.permutation(len(imgs))
            ys = ys[shuf]
            if load_imgs:
                imgs = imgs[shuf]
            if cat+'_loc' in self.h5.keys():
                locs = locs[shuf]

        # Normalize how CORNet expects, and shift channel
        # dimension to torch format
        if load_imgs:
            imgs = torch.tensor(np.moveaxis(imgs, -1, 1)).float()
            imgs = video_gen.batch_transform(imgs, video_gen.normalize)
            if cat+'_loc' in self.h5.keys():
                return imgs, ys, locs
            else:
                return imgs, ys
        else:
            if cat+'_loc' in self.h5.keys():
                return ys, locs
            else:
                return ys


    def val_size(self, train_size):
        return len(self.h5[self.cats[0]]) - train_size




