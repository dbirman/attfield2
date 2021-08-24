import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc.spatial_fields import TreeField, LinearField

from argparse import ArgumentParser
from scipy import interpolate
from scipy import spatial
from scipy import ndimage
import pickle as pkl
import pandas as pd
import numpy as np

from proc import voxel_selection as vx

parser = ArgumentParser(
    description = 
        " summaries of receptive field statistics.")
parser.add_argument('output_path',
    help = 'Where to store the saved interpolator.')
parser.add_argument("rf_summary",
    help = 'RF summary CSV file containing stat to be interpolated. ' + 
           'Should have `com_x` and `com_y` columns.')
parser.add_argument('center',
    help = 'RF summary on the same units to be subtracted from the main ' +
           'summary elementwise. Should have `com_x` and `com_y` columns.')
parser.add_argument('layer',
    help = 'Layer whose units the field will be estimated from.')
parser.add_argument('norm', type = float,
    help = 'Size of input space to normalize.')
args = parser.parse_args()
args.layer = eval(args.layer)


"""
class args:
    output_path = '/tmp/null.h5'
    rf_summary = 'data/runs/270420/summ_gauss_b11.0_com.csv'
    center = 'data/runs/270420/summ_base_com.csv'
    layer = (0, 4, 0)
    norm = 224
"""


# -------------------------------------- Load inputs ----

# Receptive fields
summ = pd.read_csv(args.rf_summary)
units = vx.VoxelIndex.from_serial(summ['unit'])
summ.set_index('unit', inplace = True)

# Normalization RFs
center = pd.read_csv(args.center)
center.set_index('unit', inplace = True)
if not all(center.index == summ.index):
    raise ValueError(
        f"Unit sets differ in {args.rf_summary} and {args.center}")


# ----------------------------- Compute location & shift ----

# Only look at units from the given layer
lstr = '.'.join(str(i) for i in args.layer)
mask = summ.index.map(lambda u: u.startswith(lstr))

# Calculate stat
shift_r = (summ.loc[mask, 'com_y'] - center.loc[mask, 'com_y']).values
shift_r /= args.norm
shift_c = (summ.loc[mask, 'com_x'] - center.loc[mask, 'com_x']).values
shift_c /= args.norm

# Build spatial map of input receptive fields
center = summ.loc[mask, ['com_y', 'com_x']].values.copy()
center /= args.norm
tree = spatial.KDTree(center)

# ----------------------------- Build interpolator object ----

t_field = TreeField(tree, shift_r, shift_c)
cs_grid, rs_grid = np.meshgrid( # Extend beyond [0,1] extent
    np.linspace(-0.1, 1.1, 50),
    np.linspace(-0.1, 1.1, 50))
grid_rshift, grid_cshift = t_field.query(rs_grid, cs_grid)

# Smooth
smooth_rshift = ndimage.gaussian_filter(grid_rshift, 1.5, mode = 'nearest')
smooth_cshift = ndimage.gaussian_filter(grid_cshift, 1.5, mode = 'nearest')

"""
eval_locs = np.stack([rs_grid, cs_grid], axis = -1).reshape(-1, 2)
rshift_interp = interpolate.LinearNDInterpolator(
    eval_locs, smooth_rshift.ravel())
cshift_interp = interpolate.LinearNDInterpolator(
    eval_locs, smooth_cshift.ravel())
"""

field = LinearField(rs_grid, cs_grid, smooth_rshift, smooth_cshift)
field.save(args.output_path)


# rs = np.linspace(0, 1, 100)
# cs = np.linspace(0, 1, 100)
# rs = np.broadcast_to(rs[:, None], (100, 100))
# cs = np.broadcast_to(cs[None, :], (100, 100))
# shift_r, shift_c = field.query(rs, cs)


"""
rs = np.random.uniform(low = 0, high = 1, size = 20)
cs = np.random.uniform(low = 0, high = 1, size = 20)
sh = np.random.uniform(low = -0.1, high = 0.2, size = 20)

rbfi = interpolate.Rbf(rs, cs, sh)
rbfi = interpolate.LinearNDInterpolator(np.stack([rs, cs]).T, sh)
rs_grid, cs_grid = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
sh_grid = rbfi(rs_grid, cs_grid)
rs_rel, cs_rel = rs_grid, cs_grid
rsh_grid, _ = field.query(rs_grid, cs_grid)

plt.pcolormesh(rs_grid, cs_grid, sh_grid, cmap = 'bone')
plt.gca().set_aspect(1.)
plt.scatter(x = locs[:, 0], y = locs[:, 1],
    c = shift_r, cmap = 'viridis', edgecolor = 'w', linewidth = 0, s = 30)

rsh_grid, csh_grid = field.query(rs_grid, cs_grid)
plt.pcolormesh(rs_grid, cs_grid, rsh_grid, cmap = 'bone')
plt.gca().set_aspect(1.)
plt.scatter(x = field.tree.data[:, 0], y = field.tree.data[:, 1],
    c = field.r_data, cmap = 'viridis',
    edgecolor = 'w', linewidth = 0, s = 30)


from scipy.ndimage import gaussian_filter

gpi = MLPRegressor(hidden_layer_sizes=(2000,), activation = 'tanh').fit(
    locs, shift_r)
eval_locs = np.stack([rs_grid, cs_grid], axis = -1).reshape(-1, 2)
rsh_grid, _ = gpi.predict(eval_locs).reshape([100, 100])

grsh_grid = gaussian_filter(rsh_grid, 5, mode = 'nearest')

ci = interpolate.LinearNDInterpolator(eval_locs, grsh_grid.ravel())
rci_grid = ci(eval_locs).reshape(100, 100)

plt.pcolormesh(rs_grid, cs_grid, rci_grid, cmap = 'bone')
plt.colorbar()
plt.gca().set_aspect(1.)
plt.scatter(x = locs[:, 0], y = locs[:, 1],
    c = shift_r, cmap = 'viridis',
    edgecolor = 'w', linewidth = 0, s = 30)

"""




