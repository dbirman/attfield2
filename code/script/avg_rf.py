import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from argparse import ArgumentParser
import pandas as pd
import numpy as np
import h5py


class args:
    ellipses = 'data/runs/270420/summ_base_ell.csv'
    margin = 56
    image_size = 224

parser = ArgumentParser(
    description = 
        "Calculate average RF size per layer.")
parser.add_argument('output_csv',
    help = 'Path to csv file where data should be output')
parser.add_argument('ellipses',
    help = 'Path to csv file containing RF ellipse fits.')
parser.add_argument("margin", type = float,
    help = 'Float, number of pixels around border for which RFs are considered' +
           ' too close to the border.')
parser.add_argument('image_size', type = float,
    help = 'Pixel size (width and height) of input space.')
args = parser.parse_args()


ells = pd.read_csv(args.ellipses)
rads = 3 * (np.sqrt(ells.major_sigma) + np.sqrt(ells.minor_sigma))/2
majs = 3 * np.sqrt(ells.major_sigma)
mins = 3 * np.sqrt(ells.minor_sigma)
center_mask = (
    (ells.com_x > args.margin) &
    (ells.com_x < args.image_size - args.margin) &
    (ells.com_x > args.margin) &
    (ells.com_y < args.image_size - args.margin)
)
unit_layers = ells['unit'].apply(lambda u: u.split(';')[0])
layer_names = unit_layers.unique()

avg_radii = []
radius_sd = []
n_fields = []
avg_maj = []
avg_min = []
for l in layer_names:
    lyr_mask = (unit_layers == l)
    avg_radii.append(rads[lyr_mask & center_mask].mean())
    avg_maj.append(majs[lyr_mask & center_mask].mean())
    avg_min.append(mins[lyr_mask & center_mask].mean())
    radius_sd.append(rads[lyr_mask & center_mask].std())
    n_fields.append((lyr_mask & center_mask).sum())

output_df = pd.DataFrame(dict(
    layer = layer_names,
    avg_radius = avg_radii,
    radius_sd = radius_sd,
    avg_maj = avg_maj,
    avg_min = avg_min,
    n_fields = n_fields))
output_df.to_csv(args.output_csv, index = False)

