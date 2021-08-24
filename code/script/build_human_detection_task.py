"""
Usage:
python3 code/script/build_human_detection_task.py
    <input_directory>           # Directory containing input images
    <image_size>                # Size of one corner of the composite
    <n_positive_focl>           # Second stimulus type: target in any corner (to be cued)
    <n_positive_dist>           # Second stimulus type: target in any corner (no cue)
    <n_neg>                    # Third stimulus type: no target present
    <n_exemplars>               # Number of example images to hold out
    <seed>                      # For replicability           
    <input_image_extension>     # Input files named Category_imagenumber.EXT
    <output_directory>          # Where to store output files
    <plot_path>             # Optional path to diagnostic plot
"""

import importlib.util
spec = importlib.util.spec_from_file_location("link_libs",
    "/Users/kaifox/projects/art_physio/code/script/link_libs_kfmbp.py")
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import detection_task as det

import pandas as pd
import numpy as np
import collections
import skimage.io
import scipy.io
import tqdm
import sys
import os

in_dir = sys.argv[1]
image_size = int(sys.argv[2])
n_pos_focal = int(sys.argv[3])
n_pos_dist = int(sys.argv[4])
n_neg = int(sys.argv[5])
n_exemplars = int(sys.argv[6])
seed = int(sys.argv[7])
ext = sys.argv[8]
out_dir = sys.argv[9]
out_diagnost = sys.argv[10] if len(sys.argv) == 11 else None
rng = np.random.RandomState(seed)



# List all images in the input directory and sort into categories
cat_images = collections.defaultdict(list)
cat_img_idxs = collections.defaultdict(list)
for filename in os.listdir(in_dir):
    # Filename will be of the form "category name_200.png"
    if filename.endswith(ext):
        cat = filename.split('_')[0]
        ix = int(filename.split('_')[-1].split('.')[0])
        cat_images[cat].append(filename)
        cat_img_idxs[cat].append(ix)
n_cats = len(cat_images)
print(f"Found {sum(len(cat_images[c]) for c in cat_images)} images.")

# Make sure images are ordered according to index for replicability
for cat in cat_images:
    ix_order = np.argsort(cat_img_idxs[cat])
    cat_images[cat] = np.array(cat_images[cat])[ix_order]
    cat_img_idxs[cat] = np.array(cat_img_idxs[cat])[ix_order]
del cat_img_idxs



# Load the i'th image from a category
def load_cat_img(cat, i):
    filename = cat_images[cat][i % len(cat_images[cat])]
    img = skimage.io.imread(os.path.join(in_dir, filename))
    if img.shape[0] != img.shape[1]:
        raise ValueError(f"Image {filename} not square, has shape {img.shape}.")
    image_scale = img.shape[0] // image_size
    if img.shape[0] / image_size != image_scale:
        raise ValueError(f"Image {filename} has size {img.shape} which is not"
            +f" a multiple of the requested image size {image_size}.")
    factors = (image_scale, image_scale, 1)
    return det.downscale_local_mean(img, factors)

# Helper to convert string category names to integers
cat_int_lut = dict(zip(cat_images.keys(), range(len(cat_images))))


# Pull out a couple of exemplars from each category
exemplars = np.zeros([n_cats * n_exemplars, image_size, image_size, 3])
exemplar_meta = pd.DataFrame()
with tqdm.tqdm(total = n_cats * n_exemplars, desc = "Exemplars") as tq:
    for cat_i, c in enumerate(cat_images.keys()):
        # Precalculate offset in `exemplars`
        cat_ofs = cat_i * n_exemplars

        curr_exemplars = rng.choice(len(cat_images[c]), n_exemplars)
        for i, image_ix in enumerate(curr_exemplars):
            image = load_cat_img(c, image_ix)
            ix = cat_ofs + i
            exemplars[ix] = image
            exemplar_meta = exemplar_meta.append({
                'cat': c,
                'cat_int': cat_int_lut[c],
                'id': cat_images[c][image_ix],
            }, ignore_index = True)
            tq.update()

        # Remove exemplars from the other set of images
        for image_ix in sorted(curr_exemplars, reverse = True):
            cat_images[c] = np.delete(cat_images[c], image_ix)

target_dtypes = {
    'cat': str,
    'cat_int': int,
    'id': str
}
exemplar_meta = exemplar_meta.astype(target_dtypes)
cat_mapping = exemplar_meta.groupby(by = 'cat', as_index = False).first()
cat_mapping = cat_mapping['cat'].values[np.argsort(cat_mapping['cat_int'])]


scipy.io.savemat(
    os.path.join(out_dir, 'exemplars.mat'),
    {'data': exemplars.astype('uint8'),
     'cat_map': cat_mapping,
     'cat': exemplar_meta['cat_int'].values})
exemplar_meta.to_csv(os.path.join(out_dir, 'exemplar_meta.csv'))



# Check we have enough images after exemplars are removed

comp_per_cat = n_pos_focal*4 + n_pos_dist*4 + n_neg
deficient_cats = [
    (c, len(cat_images[c])) for c in cat_images.keys()
    if len(cat_images[c]) < comp_per_cat - 1]
if len(deficient_cats) != 0:
    sys.stderr.write("WARNING: Not enough images to create unique stimuli"
        +" in categories: " + repr(deficient_cats) + '\n')

deficient_cats = [
    c for c in cat_images.keys()
    if len(cat_images[c]) < 0.8 * (comp_per_cat) / (n_cats - 1)]
if len(deficient_cats) != 0:
    sys.stderr.write("WARNING: Not enough images to create unique distractors"
        +" in categories: " + repr(deficient_cats) + '\n')
sys.stderr.flush()

    


# Generate positive and negative example composites for each category
class DistractorIter:

    def __init__(self, all_cats, positive_cat, seed = 1):
        self.max_n = cat_images[all_cats[0]].shape[0]
        self.neg_cats = [c for c in all_cats if c != positive_cat]
        self.counters = {c: 0 for c in self.neg_cats}
        self.order = rng.permutation(self.max_n)


    def get_next(self, k):
        n_neg_cats = len(self.counters)

        # Figure out which images to load
        meta = []

        for i in range(k // n_neg_cats):
            cat_order = rng.permutation(self.neg_cats).tolist()
            meta.extend([
                {'cat': c, 'cat_int': cat_int_lut[c],
                 'image_ix': self.counters[c],
                 'image_id': cat_images[c][self.counters[c]]}
                for c in cat_order])
            for c in cat_order:
                self.counters[c] += 1

        cat_order = rng.permutation(self.neg_cats)[:k % n_neg_cats].tolist()
        meta.extend([
                {'cat': c, 'cat_int': cat_int_lut[c],
                 'image_ix': self.counters[c],
                 'image_id': cat_images[c][self.counters[c]]}
                for c in cat_order])
        for c in cat_order:
            self.counters[c] += 1

        # Load the images
        imgs = []
        for d in meta:
            imgs.append(load_cat_img(d['cat'], d['image_ix']))

        return imgs, meta

# Helpers for generating metadata
def order_for_loc(lut, loc):
    if loc == 0:
        ixs = [0, 1, 2, 3]
    elif loc == 1:
        ixs = [1, 0, 3, 2]
    elif loc == 2:
        ixs = [2, 3, 0, 1]
    elif loc == 3:
        ixs = [3, 2, 1, 0]

    return (
        lut[ixs[0]],
        lut[ixs[1]],
        lut[ixs[2]],
        lut[ixs[3]])


def merge_meta_dicts(prefixes, dcts):
    meta = {}
    for pre, dct in zip(prefixes, dcts):
        meta.update({pre + k: v for k, v in dct.items()})
    return meta




# Generate the i'th positive composite for a category
def positive_composite(cat, i, loc, distractor_iter):

    distractors, distractor_meta = distractor_iter.get_next(3)
    imgs = np.stack([load_cat_img(cat, i)] + distractors)
    comp = det.FourWayObjectDetectionTask._arrange_grid(imgs, loc)[0]

    meta = [{'cat': cat, 'cat_int': cat_int_lut[cat],
             'image_ix': i,
             'image_id': cat_images[cat][i]},] + distractor_meta
    prefixes = order_for_loc(('NW_', 'NE_', 'SW_', 'SE_'), loc)
    meta = merge_meta_dicts(prefixes, meta)
    meta['target_loc'] = loc
    meta['target_cat'] = cat
    meta['target_cat_int'] = cat_int_lut[cat]

    return comp, meta


# Generate a negative composite from a given distractor iterator
def negative_composite(cat, distractor_iter):

    distractors, meta = distractor_iter.get_next(4)
    comp = det.FourWayObjectDetectionTask._arrange_grid(distractors, 0)[0]

    prefixes = order_for_loc(('NW_', 'NE_', 'SW_', 'SE_'), loc)
    meta = merge_meta_dicts(prefixes, meta)
    meta['target_loc'] = -1
    meta['target_cat'] = cat
    meta['target_cat_int'] = cat_int_lut[cat]

    return comp, meta




# Build main matrix of stimuli

composites = np.zeros([n_cats * comp_per_cat, image_size * 2, image_size * 2, 3])
meta_df = pd.DataFrame()
with tqdm.tqdm(total = n_cats * comp_per_cat, desc = "Composites") as tq:
    for cat_i, c in enumerate(cat_images.keys()):
        # Precalculate offset in `composites`
        cat_ofs = cat_i * comp_per_cat

        # Initialize randomized iteration over distractors
        diter = DistractorIter(list(cat_images.keys()), c)

        # Old code for positive composites locked to NE
        """
        for i in range(n_pos_focal):
            comp, meta = positive_composite(c, i, 1, diter)
            ix = cat_ofs + i
            composites[cat_ofs + i] = comp
            meta_df = meta_df.append(
                {**meta, 'stim_ix':ix, 'stim_type':'positive_fixed'},
                ignore_index = True)
            tq.update()
        """
        # Positive composites in each corner for focal examples
        for loc in range(4):
            for i in range(n_pos_focal):
                tgts_used = (loc*n_pos_dist) + i
                comp, meta = positive_composite(c, tgts_used, loc, diter)
                ix = cat_ofs + (loc * n_pos_focal) + i
                composites[ix] = comp
                meta_df = meta_df.append(
                    {**meta, 'stim_ix':ix, 'stim_type':'positive_focal'},
                    ignore_index = True)
                tq.update()

        # Positive composites in each corner for distributed examples
        for loc in range(4):
            for i in range(n_pos_dist):
                tgts_used = (n_pos_focal*4) + (loc*n_pos_dist) + i
                comp, meta = positive_composite(c, tgts_used, loc, diter)
                ix = cat_ofs + (n_pos_focal*4) + (loc * n_pos_dist) + i
                composites[ix] = comp
                meta_df = meta_df.append(
                    {**meta, 'stim_ix':ix, 'stim_type':'positive_dist'},
                    ignore_index = True)
                tq.update()

        # Negative composites
        for i in range(n_neg):
            comp, meta = negative_composite(c, diter)
            ix = cat_ofs + (n_pos_focal*4) + (n_pos_dist*4) + i
            composites[ix] = comp
            meta_df = meta_df.append(
                {**meta, 'stim_ix':ix, 'stim_type': 'negative'},
                ignore_index = True)
            tq.update()

target_dtypes = {
    'NE_cat': str,
    'NE_cat_int': int,
    'NE_image_id': str,
    'NE_image_ix': int,
    'NW_cat': str,
    'NW_cat_int': int,
    'NW_image_id': str,
    'NW_image_ix': int,
    'SE_cat': str,
    'SE_cat_int': int,
    'SE_image_id': str,
    'SE_image_ix': int,
    'SW_cat': str,
    'SW_cat_int': int,
    'SW_image_id': str,
    'SW_image_ix': int,
    'stim_ix': int,
    'stim_type': str,
    'target_cat': str,
    'target_cat_int': int,
    'target_loc': int,
}
meta_df = meta_df.astype(target_dtypes)


# Select only integer metadata for cooperation with matlab
int_meta = meta_df[[
    'stim_ix', 'target_loc', 'target_cat_int',
    'NE_cat_int', 'NW_cat_int', 'SE_cat_int', 'SW_cat_int']]
int_meta.columns = [
    'stim_ix', 'target_loc', 'target_cat',
    'NE_cat', 'NW_cat', 'SE_cat', 'SW_cat']

# --- sanity checks on metadata
bycat = meta_df.groupby('target_cat_int')
target_ids = [
    g[['NE_image_id', 'NW_image_id', 'SW_image_id', 'SE_image_id']].values[
        np.arange(len(g)), g['target_loc'].values]
    for n, g in bycat
]
assert all([len(np.unique(ids)) == len(ids) for ids in target_ids])
image_ids = [
    g[['NE_image_id', 'NW_image_id', 'SW_image_id', 'SE_image_id']].values
    for n, g in bycat]
# no repeated distractors *or* targets within category
assert all([len(np.unique(ids.ravel())) == len(ids.ravel()) for ids in image_ids])
# output images for checking
if out_diagnost is not None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out_diagnost) as pdf:
        stim_i = 0
        for i_c in tqdm.trange(n_cats, desc = "Diagnostic Plot"):
            fig, ax = plt.subplots(nrows = 4, ncols = n_pos_focal, figsize = (n_pos_focal*6, 4*6))
            ax = np.array(ax).reshape([4, n_pos_focal])
            for loc in range(4):
                for i in range(n_pos_focal):
                    ax[loc, i].imshow(composites[stim_i].astype('uint8'))
                    stim_i += 1
            plt.tight_layout()
            pdf.savefig(); plt.close()
            fig, ax = plt.subplots(nrows = 4, ncols = n_pos_dist, figsize = (n_pos_dist*6, 4*6))
            ax = np.array(ax).reshape([4, n_pos_dist])
            for loc in range(4):
                for i in range(n_pos_dist):
                    ax[loc, i].imshow(composites[stim_i].astype('uint8'))
                    stim_i += 1
            plt.tight_layout()
            pdf.savefig(); plt.close()
            ncols = int(np.ceil(n_neg / 4))
            fig, ax = plt.subplots(nrows = 4, ncols = ncols, figsize = (ncols*6, 4*6))
            ax = np.array(ax).reshape([4, ncols])
            col_num = 0; row_num = 0
            for i in range(n_neg):
                ax[row_num, col_num].imshow(composites[stim_i].astype('uint8'))
                stim_i += 1
                col_num += 1
                if col_num % ncols == 0:
                    col_num = 0; row_num += 1
            for i in range(n_neg, 4 * ncols):
                ax[row_num, col_num].set_axis_off()
                col_num += 1
                if col_num % ncols == 0:
                    col_num = 0; row_num += 1
            plt.tight_layout()
            pdf.savefig(); plt.close()

# --- output
scipy.io.savemat(
    os.path.join(out_dir, 'stimuli.mat'),
    {'data': composites.astype('uint8'),
     'cat_map': cat_mapping})
meta_df.to_csv(os.path.join(out_dir, 'stimuli_meta.csv'), index = False)
int_meta.to_csv(os.path.join(out_dir, 'stimuli_meta_ints.csv'), index = False)







