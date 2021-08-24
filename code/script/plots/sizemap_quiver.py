import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc.spatial_fields import TreeField, LinearField

from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns

from argparse import ArgumentParser
from scipy import spatial
from scipy import ndimage
import pandas as pd
import numpy as np
import h5py


sns.set_context('paper')
sns.set_style('ticks')


parser = ArgumentParser(
    description = 
        "Plot qualitative receptive field location and size change summary.")
parser.add_argument('output_path',
    help = 'Path to PDF file where the plots should go.')
parser.add_argument("uncued",
    help = 'Uncued / before-mod ellipse RF summaries.')
parser.add_argument("cued",
    help = 'Cued / after-mod ellipse RF summaries.')
parser.add_argument("layer",
    help = 'Layer from which to select units out of the RF summary files.')
parser.add_argument('--lim', type = float, default = 224,
    help =  'Axis range / image size')
parser.add_argument('--res', type = int, default = 200,
    help = 'Resolution of the size shift heatmap')
parser.add_argument('--em', type = float, default = 2.1415,
    help = 'Size constant.')
args = parser.parse_args()
args.layer = '.'.join([str(l) for l in eval('tuple('+args.layer+')')])

# Parameters
# LAYER = '0.4.0'
# EM = 3.1415
# LIM = 224
# HEATMAP_RES = 200
LAYER = args.layer
EM = args.em
LIM = args.lim
HEATMAP_RES = args.res


# Load input data
uncued_ells = pd.read_csv(args.uncued)
cued_ells = pd.read_csv(args.cued)

# cued_ells[['com_y', 'com_x']] = cued_ells[['com_x', 'com_y']]


# -----------------------------------------------------------  Preprocess  ----
cued_ell = cued_ells.loc[cued_ells['unit'].map(lambda s: s.startswith(LAYER))]
uncued_ell = uncued_ells.loc[uncued_ells['unit'].map(lambda s: s.startswith(LAYER))]

cued_size = np.sqrt(cued_ell.major_sigma * cued_ell.minor_sigma * np.pi)
uncued_size = np.sqrt(uncued_ell.major_sigma * uncued_ell.minor_sigma * np.pi)

# Build smooth map of size changes
centers = uncued_ell[['com_y', 'com_x']].values
tree = spatial.KDTree(centers)
field_dat = cued_size.values - uncued_size.values
t_field = TreeField(tree, field_dat, field_dat)
cs_grid, rs_grid = np.meshgrid( # Extend beyond [0,1] extent
    np.linspace(-0.1, LIM + 0.1, HEATMAP_RES),
    np.linspace(-0.1, LIM + 0.1, HEATMAP_RES))
field_samp, _ = t_field.query(rs_grid, cs_grid)
smooth_samp = ndimage.gaussian_filter(field_samp, 1, mode = 'nearest')

# -----------------------------------------------------------------  Plot  ----

fig, ax = plt.subplots(figsize = (2*EM, 2*EM))
map_img = smooth_samp #np.log(smooth_samp)
vrng = abs(map_img).max()
plt.imshow(
    map_img, extent = (0, LIM, LIM, 0),
    cmap = 'coolwarm', vmin = -vrng, vmax = vrng)
plt.colorbar()
ax.set_aspect(1.)

quiv_lines = LineCollection(
    np.stack([
        uncued_ell[['com_x', 'com_y']].values,
        cued_ell[['com_x', 'com_y']].values
    ], axis = 1), lw = 1, color = '#880E4F', zorder = 1)
ax.add_collection(quiv_lines)
plt.plot(cued_ell['com_x'], cued_ell['com_y'], 's',
    ms = 2, color = '#311B92', zorder = 2)
plt.xlim(0, LIM)
plt.ylim(LIM, 0)

plt.tight_layout()
plt.savefig(args.output_path)





