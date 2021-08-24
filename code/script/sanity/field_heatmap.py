import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from matplotlib.backends.backend_pdf import PdfPages
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib import cm
import numpy as np
import hsluv

from proc import spatial_fields as fld

import matplotlib.pyplot as plt
import seaborn as sns
sns.set('notebook', style = 'ticks')

parser = ArgumentParser(
    description = "Plot tests of  logistic regressions on isolated"+
                  "object detection task.")
parser.add_argument('output_path',
    help = 'Desired path for output PDF.')
parser.add_argument("data_path",
    help = 'Field HDF5 archive.')
parser.add_argument('--res', type = int, default = 100,
    help = 'Resolution of the plot')
parser.add_argument('--norm', type = float, default = 1)
parser.add_argument('--sat_max', type = float, default = None,
    help = 'Maximum value of the saturation color axis.')
args = parser.parse_args()

"""
class args:
    data_path = 'data/runs/270420/field_gauss_b4.0.h5'
    res = 100
    norm = 224
"""


# -------------------------------------- Load inputs ----

field = fld.LinearField.load(args.data_path)

# Compute the field values
rs = np.linspace(0, 1, args.res)
cs = np.linspace(0, 1, args.res)
rs = np.broadcast_to(rs[:, None], (args.res, args.res))
cs = np.broadcast_to(cs[None, :], (args.res, args.res))
shift_r, shift_c = field.query(rs, cs)



# -------------------------------------- Plot heatmap ----


def hsluv2rgb_vec(h, s, l):
    ret = []
    for h_, s_, l_, in zip(h.ravel(), s.ravel(), l.ravel()):
        ret.append(hsluv.hsluv_to_rgb([h_,s_,l_]))
    return np.array(ret).reshape(h.shape + (3,))

# Shift in polar coords
ang = np.angle(shift_r + (1j) * shift_c)
mag = np.sqrt(shift_r ** 2 + shift_c ** 2) * args.norm

sat_max = args.sat_max if args.sat_max is not None else mag.max()
hue = ang * (180. / ang.max())
sat = mag * (100. / sat_max)
lit = mag * (40. / sat_max) + 20.
img = hsluv2rgb_vec(hue, sat, lit)

with PdfPages(args.output_path) as pdf:

    plt.imshow(img / 1.01 + 0.005, extent = (0, args.norm, args.norm, 0))

    sats = hsluv2rgb_vec(
            np.full([360], 30),
            np.linspace(0, 100, 360),
            np.linspace(0, 60, 360)) / 1.01 + 0.005
    sat_cmap = mc.ListedColormap(sats)
    norm = mc.Normalize(0, sat_max)
    sat_map = cm.ScalarMappable(norm = norm, cmap = sat_cmap)
    plt.colorbar(sat_map).ax.set_title("Shift (px)")

    hues = hsluv2rgb_vec(
            np.arange(359),
            np.full([360], 90),
            np.full([360], 50)) / 1.01 + 0.005
    hue_cmap = mc.ListedColormap(hues)
    norm = mc.Normalize(hue.min(), hue.max())
    hue_map = cm.ScalarMappable(norm = norm, cmap = hue_cmap)
    plt.colorbar(hue_map).ax.set_title("Angle (deg)")

    plt.tight_layout()
    pdf.savefig()






