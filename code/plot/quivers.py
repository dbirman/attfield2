from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

from proc.spatial_fields import TreeField, LinearField
from plot import kwargs as default_pkws

from scipy import spatial
from scipy import ndimage
import numpy as np


def quiver_data(dist_ells, focl_ells, layer, resolution):
    lstr = '.'.join(str(i) for i in layer)
    focl_ell = focl_ells.loc[focl_ells.index.map(lambda s: s.startswith(lstr))]
    dist_ell = dist_ells.loc[dist_ells.index.map(lambda s: s.startswith(lstr))]

    cued_size = np.sqrt(focl_ell.major_sigma * focl_ell.minor_sigma * np.pi)
    uncued_size = np.sqrt(dist_ell.major_sigma * dist_ell.minor_sigma * np.pi)

    # Build smooth map of size changes
    centers = dist_ell[['com_y', 'com_x']].values
    tree = spatial.KDTree(centers)
    field_dat = focl_ell['size']
    # field_dat = cued_size.values - uncued_size.values
    t_field = TreeField(tree, field_dat, field_dat)
    cs_grid, rs_grid = np.meshgrid( # Extend beyond [0,1] extent
        np.linspace(-0.1, 224 + 0.1, resolution),
        np.linspace(-0.1, 224 + 0.1, resolution))
    field_samp, _ = t_field.query(rs_grid, cs_grid)
    smooth_samp = ndimage.gaussian_filter(field_samp, 1, mode = 'nearest')
    return dist_ell, focl_ell, smooth_samp



def quiverplot(dist_ell, focl_ell, smooth_samp, ax, cmap, vrng = (None, None), pkws = default_pkws):
    map_img = smooth_samp
    norm = colors.TwoSlopeNorm(vmin=vrng[0], vcenter=1., vmax=vrng[1])
    im_obj = ax.imshow(
        map_img, extent = (0, 224, 224, 0),
        cmap = cmap, norm = norm)

    quiv_lines = LineCollection(
        np.stack([
            dist_ell[['com_x', 'com_y']].values,
            focl_ell[['com_x', 'com_y']].values
        ], axis = 1), zorder = 1, **pkws.quiver_line)
    ax.add_collection(quiv_lines)
    ax.plot(focl_ell['com_x'], focl_ell['com_y'], 's',
        zorder = 2, **pkws.quiver_point)

    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    return im_obj

