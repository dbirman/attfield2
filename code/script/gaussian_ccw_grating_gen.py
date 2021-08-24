
import skimage.io as skio
import pandas as pd
import numpy as np
import os.path
import tqdm
import os

W = 112
RAD = 17
r, c = np.mgrid[:W, :W] - W/2
spacings = np.linspace(5, 10, 10)
angles = np.linspace(np.pi / 64, np.pi / 8, 16)
output_dir = 'data/gratings/masked_ccw/'


if not os.path.exists(os.path.join(output_dir, 'c')):
    os.mkdir(os.path.join(output_dir, 'c'))
if not os.path.exists(os.path.join(output_dir, 'cw')):
    os.mkdir(os.path.join(output_dir, 'cw'))

index = dict(category = [], path = [], n = [], neg_n = [],
    angle = [], angle_n = [], spacing = [], spacing_n = [])
cat_max_i = len(angles) * len(spacings) - 1
gaussian_mask = np.exp(-(r**2 + c**2) / (RAD/1.5)**2)
for i_a, a in tqdm.tqdm(enumerate(angles), total = len(angles), position = 0):
    for ccw_tag, mult in (['c', 1], ['cw', -1]):
        theta = mult * a + np.pi / 2
        direction = np.sin(theta) * c + np.cos(theta) * r
        for i_s, s in enumerate(spacings):
            grating = np.sin(np.pi / s * direction)
            img = (grating * gaussian_mask * 127 + 128).astype('uint8')
            filename = f'{ccw_tag}_spacing{i_s}_angle{i_a}.png'
            skio.imsave(os.path.join(output_dir, ccw_tag, filename), img)
            index['category'].append(ccw_tag)
            index['path'].append(os.path.join(ccw_tag, filename))
            img_i = i_a * len(spacings) + i_s
            index['n'].append(img_i)
            index['neg_n'].append(cat_max_i - img_i)
            index['angle'].append(theta)
            index['angle_n'].append(i_a)
            index['spacing'].append(s)
            index['spacing_n'].append(i_s)

pd.DataFrame(index).to_csv(os.path.join(output_dir, 'ccw_index.csv'), index = False)

