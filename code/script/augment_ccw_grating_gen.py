
import skimage.io as skio
import itertools as iit
import pandas as pd
import numpy as np
import os.path
import tqdm
import os

W = 112
r, c = np.mgrid[:W, :W] - W/2
target_n_imgs = 8000
spacings = np.linspace(3, 10, 10)
angles = np.linspace(np.pi / 64, np.pi / 8, 8)
phase_bins = np.linspace(-np.pi, np.pi, 3)[:-1]
contrasts = np.linspace(0.3, 1, 5)
center_bins = np.linspace(-W//4, W//4, 5)[:-1]
radius_bins = np.linspace(np.sqrt(8), np.sqrt(17), 4)[:-1] ** 2
output_dir = 'data/gratings/augmented_ccw/'


if not os.path.exists(os.path.join(output_dir, 'c')):
    os.mkdir(os.path.join(output_dir, 'c'))
if not os.path.exists(os.path.join(output_dir, 'cw')):
    os.mkdir(os.path.join(output_dir, 'cw'))

index = dict(
    category = [], path = [], n = [], neg_n = [],
    angle = [], angle_n = [], spacing = [], spacing_n = [],
    phase = [], phase_bin = [], phase_n = [],
    cont = [], cont_n = [],
    x_cent = [], x_cent_bin = [], x_cent_n = [],
    y_cent = [], y_cent_bin = [], y_cent_n = [],
    radius = [], radius_bin = [], radius_n = [])

dphase = phase_bins[1] - phase_bins[0] # for phase randomization
dcenter = center_bins[1] - center_bins[0] # for center randomization
dradius = radius_bins[1] - radius_bins[0] # for radius randomization
rng = np.random.RandomState(1)

base_n_imgs = (2 * len(angles) * len(spacings) * len(phase_bins) *
          len(contrasts) * len(center_bins)**2 * len(radius_bins))
included_imgs = np.random.choice(base_n_imgs, target_n_imgs, replace = False)
img_id = 0

with tqdm.tqdm(total = target_n_imgs, position = 0) as tq:
    for i_a, a in enumerate(angles):
        for ccw_tag, mult in (['c', 1], ['cw', -1]):
            theta = mult * a + np.pi / 2
            direction = np.sin(theta) * c + np.cos(theta) * r
            for i_s, s in enumerate(spacings):
                for ((i_phase, phase), (i_cont, cont),
                     (i_x, x_cent), (i_y, y_cent), (i_radius, radius)
                    ) in iit.product(
                     enumerate(phase_bins), enumerate(contrasts),
                     enumerate(center_bins), enumerate(center_bins), enumerate(radius_bins)):

                    img_id += 1
                    if img_id not in included_imgs: continue

                    # random augmentations
                    exact_phase = phase + rng.uniform(high = dphase)
                    exact_x_cent = x_cent + rng.uniform(high = dcenter)
                    exact_y_cent = y_cent + rng.uniform(high = dcenter)
                    exact_radius = radius + rng.uniform(high = dradius)

                    # generate image
                    grating = np.sin(np.pi / s * direction + exact_phase)
                    gaussian_mask = np.exp(-(
                        (r - exact_x_cent)**2 +
                        (c - exact_y_cent)**2) / (
                        exact_radius/1.5)**2)
                    gaussian_mask[np.log(gaussian_mask) < -8] = 0
                    img = (cont * grating * gaussian_mask * 0.49 + 0.5)#.astype('uint8')
                    # save image
                    filename = (f'{ccw_tag}_spac{i_s}_ang{i_a}_' + 
                                f'p{i_phase}_c{i_cont}_x{i_x}_y{i_y}_r{i_radius}.tif')
                    Image.fromarray(img).save(os.path.join(output_dir, ccw_tag, filename))
                    # save image metadata
                    index['category'].append(ccw_tag)
                    index['path'].append(os.path.join(ccw_tag, filename))
                    index['angle'].append(theta)
                    index['angle_n'].append(i_a)
                    index['spacing'].append(s)
                    index['spacing_n'].append(i_s)
                    index['phase'].append(exact_phase)
                    index['phase_bin'].append(phase)
                    index['phase_n'].append(i_phase)
                    index['cont'].append(cont)
                    index['cont_n'].append(i_cont)
                    index['x_cent'].append(exact_x_cent)
                    index['x_cent_bin'].append(x_cent)
                    index['x_cent_n'].append(i_x)
                    index['y_cent'].append(exact_y_cent)
                    index['y_cent_bin'].append(y_cent)
                    index['y_cent_n'].append(i_y)
                    index['radius'].append(exact_radius)
                    index['radius_bin'].append(radius)
                    index['radius_n'].append(i_radius)

                    tq.update()

index_df = pd.DataFrame(index)
cat_ns = {c: len(c_data) for c, c_data in index_df.groupby('category')}
index_df['n'] = 0
for cat in ['c', 'cw']:
    index_df.loc[index_df['category'] == cat, 'n'] = np.arange(cat_ns[cat])
index_df['neg_n'] = index_df.apply(lambda row: cat_ns[row['category']] - row['n'] - 1, axis = 1)
index_df.to_csv(os.path.join(output_dir, 'ccw_index.csv'), index = False)

