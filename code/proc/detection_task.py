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
import collections
import skvideo.io
import operator
import torch
import h5py
import tqdm
import sys
import os
import gc


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
        super(nm.LayerMod, self).__init__()
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
        '''
        ### Arguments
        - 'shape' --- Shape of the tensor that the scaling array will
            be applied to, of form (batch, channel, row, col)
        ### Returns
        - 'arr' --- The scale array, a torch tensor of compatible to
            multiply with a tenor of the shape specified.
        '''
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
        row_odd_fix = shape[-2] % 2
        col_odd_fix = shape[-1] % 2
        imgs = [
            self.gain_profile(
                row_n + row_odd_fix,
                col_n + col_odd_fix
                )[0, ...] + 1,
            np.ones((row_n + row_odd_fix, col_n)),
            np.ones((row_n, col_n + col_odd_fix)),
            np.ones((row_n, col_n))]
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
                [750*(row_n/112)**2], [1])
            return (self.beta-1) * gauss


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
        super(nm.LayerMod, self).__init__()
        self.w = nn.Parameter(torch.tensor(w).float())
        self.register_parameter("w", self.w)
        self.b = nn.Parameter(torch.tensor(b).float())
        self.register_parameter("b", self.b)
    
    def post_layer(self, outputs, cache):
        return self.decision_function(outputs)

    def predict(self, encodings):
        return self.decision_function(encodings) > 0

    def decision_function(self, encodings):
        if isinstance(encodings, np.ndarray):
            w = self.w.data.detach().cpu().numpy().T[:, 0]
            b = self.b.data.detach().cpu().numpy()
            return (encodings * w[None, :]).sum(axis = -1) + b
            return np.dot(encodings, w) + b
        else:
            w = torch.t(self.w.data)
            b = self.b.data
            return torch.mm(encodings, w) + b

    def predict_on_fn(self, decision_fn):
        return decision_fn > 0

def stddiff(a, b):
    mean = (a.mean() + b.mean()) / 2
    std = (a.std() + b.std()) / 2
    return abs(a/std - b/std).mean()

def load_logregs(filename, bias = True):
    archive = np.load(filename)
    bias = float(bias)
    if archive['type'] == 'mod':
        return {
            c: ClassifierMod(archive[c+'_w'], bias * archive[c+'_b'])
            for c in archive['categories']
        }
    else:
        regs = {c: LogisticRegression()
                for c in archive['categories']}
        for c in regs:
            regs[c].coef_ = archive[c+'_c']
            regs[c].intercept_ = bias * archive[c+'_i']
        return regs




def save_logregs(filename, logregs):
    logreg_type = type(logregs[next(iter(logregs.keys()))])
    if logreg_type is ClassifierMod:
        np.savez(filename,
            categories = list(logregs.keys()),
            **{c+"_w": logregs[c].w.detach().numpy() for c in logregs},
            **{c+"_b": logregs[c].b.detach().numpy() for c in logregs},
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
                # loaded_imgs = self.h5[]
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
            Alternatively a list/tuple/array containing `n` integers can be
            provided corresponding to the target locations for each
            of the `n` returned positive images.
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

        if hasattr(loc, '__iter__'):
            loc_iter = True
            loc_tupl = tuple(loc)
        else:
            loc_iter = False

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
                    loc = loc if not loc_iter else loc_tupl[i])
                neg_img, neg_loc = self._arrange_grid(
                    false_images[4*i:4*(i+1)],
                    loc = loc if not loc_iter else loc_tupl[i])
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
        specified location and others distributed around the grid such 
        that if their shapes are uneven they still align correctly.
        ### Arguments
        - `imgs` --- A list or array of four images, each with shape
            (row, col, channel) to be arranged into a 2x2 grid.
        - `loc` --- Position where the first image from `imgs` should
            go in the grid. The mapping is
            - 0: Top Left (NW)
            - 1: Top Right (NE)
            - 2: Bottom Left (SW)
            - 3: Bottom Right (SE)
            - -1: Random
        ### Returns
        - `grid` --- The image grid, of shape (2*h, 2*w, channel)
        - `loc` --- The passed parameter for `loc`, or the equivalent
            mapping of what was randomly assigned otherwise.
        '''
        # Rearrange `imgs` to put the first image in the right spot
        # and to position of the other images in a grid around

        if loc == -1:
            loc = np.random.randint(4)

        #rand = np.array(imgs)[np.random.permutation(3)+1]
        imgs = FourWayObjectDetectionTask.order_by_loc(imgs, loc)

        return np.concatenate([
                np.concatenate([imgs[0], imgs[1]], axis = 1),
                np.concatenate([imgs[2], imgs[3]], axis = 1)],
            axis = 0), loc

    @staticmethod
    def order_by_loc(arr, loc):
        if loc == 0:
            return (arr[0], arr[1], arr[2], arr[3])
        elif loc == 1:
            return (arr[1], arr[0], arr[3], arr[2])
        elif loc == 2:
            return (arr[2], arr[3], arr[0], arr[1])
        elif loc == 3:
            return (arr[3], arr[2], arr[1], arr[0])


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
    for c in preds:
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





def model_encodings(model, decoders, imgs, mods = {}, cuda = False):
    '''
    ### Arguments
    - `model` --- A PyTorch model, presumably CORNet
    - `decoders` --- A list of layer indices (tuples of ints)
        giving layers that perform 'decoding' functions in
        the model and so should be skipped.
    - `imgs` --- Either an array of shape (n, h, w, c) giving
        the images to be encoded, or a dictionary mapping
        categories to arrays of that type.
    - `cuda` --- Force run on GPU, assumes `imgs` are GPU tensors.
    ### Returns
    - `encodings` --- Model encodings of `imgs`. Either an array
        of shape (n_imgs, n_features) if `imgs` was an array, or
        a dictionary mapping categories to such arrays of `imgs`
        was passed as a dictionary.
    '''
    if isinstance(imgs, dict):
        return {c: model_encodings(model, decoders, imgs[c],
                                   mods, cuda = cuda)
                for c in imgs}
    else:
        decoder_bypass = {l: LayerBypass() for l in decoders}
        mgr = nm.NetworkManager.assemble(model, imgs,
                mods = {**mods, **decoder_bypass},
                cuda = cuda)
        return mgr.computed[(0,)].cpu().detach().numpy()


def fit_logregs(model, decoders, task, mods = {},
                train_size = 30, shuffle = False,
                verbose = False):
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
        if verbose:
            print(f"Category: {c}")
        imgs, ys = task.train_set(c, train_size, shuffle=shuffle)[:2]
        all_ys[c] = ys
        all_imgs[c] = imgs
        encodings[c] = model_encodings(
            model, decoders, imgs, mods = mods)

        skregs[c] = LogisticRegression(solver = 'liblinear', max_iter = 1000)
        skregs[c].fit(encodings[c], ys)
        regmods[c] = ClassifierMod(
            skregs[c].coef_,
            skregs[c].intercept_)

    return encodings, skregs, regmods, all_imgs, all_ys



def voxel_decision_grads(voxels, model, decoders, imgs, ys, regs,
                         mods = {}, no_grads = False, batch_size = None,
                         verbose = True, batch_callback = None):
    '''
    ### Arguments
    - `no_grads` --- Don't compute gradients. Useful if you just want
    to get the supporting information without running (slow) backprop.
    ### Returns
    `grads` --- Nested dictionary with outer key giving category
        and inner giving unit layer (or will be `None` if no_grads
        was passed as `True`). Each value is an array giving
        the gradient of a unit on an image, arranged in shape
        (n_img, n_unit).
    `acts` --- Nested dictionary with outer key giving category
        and inner giving unit layer. Each value is an array giving
        the activation of a unit on an image, arranged in shape
        (n_img, n_unit).
    `fn` --- Dictionary mapping categories to arrays of decision
        function values on each image. Values are numpy arrays
        with shape (n_img)
    `pred` --- Dictionary mapping categories to arrays of binary
        classification predictions on each image. Values are
        numpy arrays with shape (n_img)
    `ys` --- Dictionary mapping categories to arrays of binary
        classification ground truth on each image. Values are
        numpy arrays with shape (n_img). This is a simple passthrough
        of the `ys` parameter integrated into the results format.
    `score_cat` --- Dictionary mapping categories to percent
        accuracy scores.
    `score_ovrl` --- A single float giving percent accuracy score.
    `batch_size` --- Size of batches to run. Can be None to run
        without batching the data.
    '''
    grads = {}
    acts = {}
    pred = collections.defaultdict(list)
    fn = collections.defaultdict(list)
    score_cat = {}
    score_ovrl = {}

    for i_c, c in enumerate(regs):
        # Print progress updates
        if verbose:
            print("\rClass: " + str(i_c+1) + " / " + str(len(regs)))

        batches = (np.arange(len(imgs[c])), )
        if batch_size is not None:
            if len(imgs[c]) > batch_size:
                batches = np.array_split(batches[0], len(imgs[c]) // batch_size)

        for res_dict in (grads, acts):
            res_dict[c] = collections.defaultdict(list)

        # NOTE: code assumes we'll still see images in expected order
        for batch_n, batch_ix in enumerate(batches):
            if verbose:
                print(f"Batch {batch_ix} / {len(batches)}")

            # Run encodings with voxel gradients tracked
            if verbose: print(f"Running")
            decoder_bypass = {l: LayerBypass() for l in decoders}
            logreg = {(0,): regs[c]}
            mgr = nm.NetworkManager.assemble(model, imgs[c][batch_ix],
                    mods = nm.mod_merge(mods, decoder_bypass, logreg),
                    with_grad = True)

            # Backprop the gradients to the voxels
            if not no_grads:
                if verbose: print(f"Backpropagating")
                back_mask = torch.ones_like(mgr.computed[(0,)])
                mgr.computed[(0,)].backward(back_mask)

            # Pull gradients from the model and score its performance
            if verbose:
                print(f"Extracting data")
            if not no_grads:
                batch_grads = {}
            batch_acts = {}
            for l in voxels:
                batch = mgr.computed[l].cpu()
                if not no_grads:
                    batch_grads[l] = voxels[l].index_into_batch(
                        batch.grad_nonleaf.detach().numpy(), i_vox = None)
                batch_acts[l] = voxels[l].index_into_batch(
                    batch.detach().numpy(), i_vox = None)
            batch_fn = mgr.computed[(0,)].detach().numpy()
            batch_pred = regs[c].predict_on_fn(batch_fn)

            if batch_callback is None:
                for l in voxels:
                    if not no_grads:
                        grads[c][l].append(batch_grads[l])
                    acts[c][l].append(batch_acts[l])
                fn[c].append(batch_fn)
                pred[c].append(batch_pred)
            else:
                batch_callback(
                    c, batch_ix,
                    batch_grads if not no_grads else None,
                    batch_acts,
                    batch_fn,
                    batch_pred,
                    ys[c][batch_ix])

                # Clean intermediate outputs if possible
                if not no_grads:
                    del batch_grads
                del batch_acts, batch_fn, batch_pred; gc.collect()

            # Definitely clean up the network manager
            del mgr; gc.collect()

        # Concatenate batches
        if batch_callback is None:
            if verbose: print(f"Concatenating")
            for l in voxels:
                if not no_grads:
                    grads[c][l] = np.concatenate(grads[c][l])
                acts[c][l] = np.concatenate(acts[c][l])
            fn[c] = np.concatenate(fn[c])
            pred[c] = np.concatenate(pred[c])

        # End progress update line
        print()

    if batch_callback is None:
        s_ovrl, s_cat = score_logregs_precomputed(pred, fn, ys)
        for c in regs:
            score_cat[c] = s_cat[c]['acc']
            score_ovrl = s_ovrl['acc']

        return (grads if not no_grads else None,
            acts, fn, pred, ys, score_cat, score_ovrl)



















def generate_dataset(imagenet_index, output, image_size = 224, blacklist = (), seed = 1):
    '''Process ImageNet data down to an archived dataset
    that is quickly readable for training.
    ### Arguments
    - `output` --- HDF5 filename to output to. This will contain
        one table for each image category.
    '''

    base_dir = os.path.dirname(imagenet_index)
    index = pd.read_csv(imagenet_index)
    cats = index['category'].unique()
    metacols = [col for col in index.columns if col not in ('category', 'n', 'neg_n')]

    max_ns = [max(index.iloc[np.where(index['category'] == c)]['n'])
              for c in cats]
    limit_n = min(max_ns)
    rng = np.random.RandomState(seed)

    metadata = {'cat': [], 'n':[], **{col: [] for col in metacols}}
    with h5py.File(output, "w") as f:
        for cat in tqdm.tqdm(cats, desc = "Category"):
            if cat in blacklist: continue

            dset = f.create_dataset(cat,
                (limit_n, image_size, image_size, 3),
                dtype = 'i')

            cat_images = index['category'] == cat
            first_n = index['n'] < limit_n
            idxs = np.where(cat_images & first_n)[0]
            paths = index['path'].iloc[idxs]
            index_metadata = index.iloc[idxs]

            ordering = rng.permutation(len(paths))
            for out_i, in_i in enumerate(tqdm.tqdm(ordering, desc = "   Image")):
                filepath = os.path.join(base_dir, paths.iloc[in_i])
                if paths.iloc[in_i].endswith('.tif'):
                    from PIL import Image
                    with Image.open(filepath) as imgfile:
                        img = np.array(imgfile)
                    img = np.tile(img[:, :, None], [1, 1, 3]) * 255
                else:
                    img = skvideo.io.vread(filepath)
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
                dset[out_i, ...] = img[slices]


                metadata['cat'].append(cat)
                metadata['n'].append(out_i)
                for col in metacols:
                    metadata[col].append(index_metadata[col].iloc[in_i])
    if output.endswith('.h5'):
        metadata_output = output[:-3] + '_meta.csv'
    else:
        metadata_output = output + '_meta.csv'
    pd.DataFrame(metadata).to_csv(metadata_output, index = False)


def _neg_iter(task, cat, seed = 1, metadata_df = None):
    max_n = task.h5[task.cats[0]].shape[0]
    rng = np.random.RandomState(seed)
    neg_cats = [c for c in task.cats if c != cat]
    order = rng.permutation(max_n)
    while True:
        for i in range(max_n):
            for c in rng.permutation(neg_cats):
                yield (
                    task._load_images(c, order[i], read_order = 1),
                    None if metadata_df is None else metadata_df[c].iloc[order[i]]
                )


def cache_iso_task(
        imagenet_h5, output, image_size = 224, seed = 1,
        blacklist = (), metadata_csv = None):
    '''Process an imagenet dataset into a format where it can
    be almost instantaneously read as IsolatedObjectDetectionTask data.'''

    task = IsolatedObjectDetectionTask(imagenet_h5, image_size)
    if metadata_csv is not None:
        in_metadata = pd.read_csv(metadata_csv)
        metadata_cols = [col for col in in_metadata.columns if col not in ('cat', 'n')]
        in_metadata = {c: c_data for c, c_data in in_metadata.groupby('cat')}
    else:
        metadata_cols = []
    out_metadata = {'target': [], 'cat': [], 'n': [], **{col: [] for col in metadata_cols}}
    with h5py.File(output, "w") as f:
        for i, cat in enumerate(tqdm.tqdm(task.cats, desc = "Category")):
            if cat in blacklist: continue

            # Load and save images
            negs = _neg_iter(
                task, cat, seed = seed,
                metadata_df = in_metadata if metadata_csv is not None else None)
            cat_n = task.h5[cat].shape[0]
            dset = f.create_dataset(cat,
                (2*cat_n, image_size, image_size, 3),
                dtype = 'i')
            dset[::2] = task._load_images(cat, range(cat_n),
                                             read_order = 1)
            pos_metadata = {
                'cat': [cat] * cat_n,
                'target': [cat] * cat_n,
                'n': np.arange(2*cat_n)[::2]}
            for col in metadata_cols:
                pos_metadata[col] = in_metadata[cat][col].iloc[:cat_n]
            neg_choices = [next(negs) for _ in range(cat_n)]
            neg_imgs, neg_meta = list(zip(*neg_choices))
            dset[1::2] = np.array(neg_imgs)[:, 0]
            neg_metadata = {
                'cat': [row['cat'] for row in neg_meta],
                'target': [cat] * cat_n,
                'n': np.arange(2*cat_n)[1::2]}
            for col in metadata_cols:
                neg_metadata[col] = [row[col] for row in neg_meta]

            # Save classification targets
            dset = f.create_dataset(cat + '_y',
                (2*cat_n,), dtype = bool)
            dset[::2] = np.ones(cat_n, dtype = bool)
            dset[1::2] = np.zeros(cat_n, dtype = bool)

            flatten2d = lambda x, y: [a for b in zip(x, y) for a in b]
            out_metadata['cat'] += flatten2d(pos_metadata['cat'], neg_metadata['cat'])
            out_metadata['target'] += flatten2d(pos_metadata['target'], neg_metadata['target'])
            out_metadata['n'] += flatten2d(pos_metadata['n'], neg_metadata['n'])
            for col in metadata_cols:
                out_metadata[col] += flatten2d(pos_metadata[col], neg_metadata[col])
    if output.endswith('.h5'):
        metadata_output = output[:-3] + '_meta.csv'
    else:
        metadata_output = output + '_meta.csv'
    pd.DataFrame(out_metadata).to_csv(metadata_output, index = False)



def cache_four_task(
        imagenet_h5, output, image_size = 112, seed = 1,
        loc = -1, blacklist = (), metadata_csv = None):
    task = FourWayObjectDetectionTask(imagenet_h5, image_size)
    cats = [c for c in task.cats if c not in blacklist]
    if metadata_csv is not None:
        in_metadata = pd.read_csv(metadata_csv)
        metadata_cols = [col for col in in_metadata.columns if col not in ('cat', 'n')]
        in_metadata = {c: c_data for c, c_data in in_metadata.groupby('cat')}
    else:
        metadata_cols = []
    loc_names = ['tl', 'tr', 'bl', 'br']
    out_metadata = {'target': [], 'target_loc': [],
        **{f'{loc}_cat': [] for loc in loc_names},
        **{f'{loc}_src_n': [] for loc in loc_names},
        **{f'{loc}_{col}': [] for col in metadata_cols for loc in loc_names}}
    with h5py.File(output, "w") as f:
         for i, cat in enumerate(tqdm.tqdm(cats, desc = "Category")):

            # Load and save images
            negs = _neg_iter(task, cat, seed = seed + i,
                metadata_df = in_metadata if metadata_csv is not None else None)
            cat_n = task.h5[cat].shape[0]

            # load target images and associated metadata
            true_images = task._load_images(cat, range(cat_n), read_order=1)
            pos_meta = {
                'cat': pd.Series([cat] * cat_n),
                'src_n': pd.Series(np.arange(cat_n))}
            for col in metadata_cols:
                pos_meta[col] = in_metadata[cat][col].iloc[:cat_n]

            # load distractor images and associated metadata
            false_images, false_meta = list(zip(*[next(negs)
                for _ in tqdm.trange(7*cat_n, desc = 'Negative')]))
            false_images = np.array(false_images)[:, 0]
            neg_meta = {
                'cat': pd.Series([row['cat'] for row in false_meta]),
                'src_n': pd.Series([row['n'] for row in false_meta])}
            for col in metadata_cols:
                neg_meta[col] = pd.Series([row[col] for row in false_meta])

            # separate the distractor images into ones that will go in positive
            # examples (_posarr) and ones that will go in negative examples
            false_images_posarr = false_images[:3*cat_n]
            false_images = false_images[3*cat_n:]
            neg_meta_posarr = {k: v.iloc[:3*cat_n] for k, v in neg_meta.items()}
            neg_meta = {k: v.iloc[3*cat_n:] for k, v in neg_meta.items()}

            # create space in the HDF5 archive
            imgs = f.create_dataset(cat,
                (2*cat_n, 2*image_size, 2*image_size, 3),
                dtype = 'i')
            ys = f.create_dataset(cat + '_y',
                (2*cat_n,), dtype = bool)
            locs = f.create_dataset(cat + '_loc',
                (2*cat_n,), dtype = 'i')

            # arrange the loaded images into composites and put them in the HDF5
            # and put associated metadata into correct column
            for i in range(cat_n):
                array_posimg = false_images_posarr[3*i:3*(i+1)]
                imgs[2*i], locs[2*i] = task._arrange_grid(
                    np.concatenate([[true_images[i]], array_posimg]),
                    loc = loc)
                ys[2*i] = True
                # arrange metadata to agree with location of target and distracotr images
                all_metacols = ['cat', 'src_n', *metadata_cols]
                arranged_meta = task.order_by_loc([
                    {c: pos_meta[c].iloc[i] for c in all_metacols},
                    {c: neg_meta_posarr[c].iloc[3*i] for c in all_metacols},
                    {c: neg_meta_posarr[c].iloc[3*i+1] for c in all_metacols},
                    {c: neg_meta_posarr[c].iloc[3*i+2] for c in all_metacols}
                ], locs[2*i])
                out_metadata['target'].append(cat)
                out_metadata['target_loc'].append(loc_names[locs[2*i]])
                for loc_i, loc_name in enumerate(loc_names):
                    for col in all_metacols:
                        out_metadata[f'{loc_name}_{col}'].append(arranged_meta[loc_i][col])

                imgs[2*i+1], locs[2*i+1] = task._arrange_grid(
                    false_images[4*i:4*(i+1)],
                    loc = loc)
                ys[2*i+1] = False
                # arrange metadata to agree with location of distractor images
                arranged_meta = task.order_by_loc([
                    {c: neg_meta[c].iloc[4*i  ] for c in all_metacols},
                    {c: neg_meta[c].iloc[4*i+1] for c in all_metacols},
                    {c: neg_meta[c].iloc[4*i+2] for c in all_metacols},
                    {c: neg_meta[c].iloc[4*i+3] for c in all_metacols}
                ], locs[2*i + 1])
                out_metadata['target'].append(cat)
                out_metadata['target_loc'].append(loc_names[locs[2*i]])
                for loc_i, loc_name in enumerate(loc_names):
                    for col in all_metacols:
                        out_metadata[f'{loc_name}_{col}'].append(arranged_meta[loc_i][col])
    if output.endswith('.h5'):
        metadata_output = output[:-3] + '_meta.csv'
    else:
        metadata_output = output + '_meta.csv'
    pd.DataFrame(out_metadata).to_csv(metadata_output, index = False)



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
            classification. Will be of size (n, 3, 224, 224).
            These will contain `n` images containing the object
            category and `n` not containing the object category
            in random order.
        - 'ys' --- An array of shape (n,) taking values either
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
            # ixs = self.metadata[cat].iloc[start:stop]
            ixs = np.arange(start, stop)

        else:
            if load_imgs:
                imgs = self.h5[cat][-stop-1:-start-1]
            ys = self.h5[cat+'_y'][-stop-1:-start-1]
            if cat+'_loc' in self.h5.keys():
                locs =  self.h5[cat+'_loc'][-stop-1:-start-1]
            ixs = np.arange(
                len(self.h5[cat])-stop-1,
                len(self.h5[cat])-start-1)

        # Shuffle
        if shuffle:
            shuf = np.random.permutation(len(imgs))
            ys = ys[shuf]
            if load_imgs:
                imgs = imgs[shuf]
            if cat+'_loc' in self.h5.keys():
                locs = locs[shuf]
            ixs = ixs[shuf]

        # Normalize how CORNet expects, and shift channel
        # dimension to torch format
        if load_imgs:
            imgs = torch.tensor(np.moveaxis(imgs, -1, 1)).float()
            imgs = video_gen.batch_transform(imgs, video_gen.normalize)
            if cat+'_loc' in self.h5.keys():
                return imgs, ys, locs, ixs
            else:
                return imgs, ys, ixs
        else:
            if cat+'_loc' in self.h5.keys():
                return ys, locs, ixs
            else:
                return ys, ixs

    def val_size(self, train_size):
        return len(self.h5[self.cats[0]]) - train_size


class FakeDetectionTask(IsolatedObjectDetectionTask):
    '''
    Utility task to emulate an IsolatedObjectDetectionTask that
    simply returns random noise images.

    Mainly intended for local use where one doesn't have the
    capacity to store the imagenet composite dataset but still
    wants to test code.'''

    def __init__(self, cats, image_size, channels,
        sim_imgs = True, sim_locs = True):

        self.cats = cats
        self.sample_shape = (channels, image_size, image_size)
        self.sim_imgs = sim_imgs
        self.sim_locs = sim_locs
        self.channels = channels
        self.image_size = image_size

    def _load_set(self, cat, n, read_order, shuffle = False):
        imgs = torch.empty(n,
            self.channels,
            self.image_size,
            self.image_size
            ).uniform_(-1., 1.)
        imgs[..., self.image_size // 40::self.image_size // 20, :] = 5.
        imgs[..., :, self.image_size // 40::self.image_size // 20] = 5.
        ys = torch.linspace(0, 1, n)[np.random.permutation(n)] > 0.5
        locs = np.int_(np.floor(np.random.rand(n) * 5))
        # Match the return profile of a DistilledDetectionTask
        if self.sim_imgs and self.sim_locs: return imgs, ys, locs
        elif self.sim_imgs: return imgs, ys
        elif self.sim_locs: return ys, locs
        else: return ys

