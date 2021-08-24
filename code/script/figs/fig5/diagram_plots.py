import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

import numpy as np
import matplotlib.pyplot as plt

filt = np.array([-1, -1, -1, 1, 1, 1, -1, -1, -1])
x = np.linspace(-3, 3, len(filt))

gauss_near = 1.4 * np.exp(-(x/3-0.5)**2) + 1
gauss_far = 1.4 * np.exp(-(x/3-1.3)**2) + 1
sens_near = gauss_near - gauss_near.mean() + 1
sens_far = gauss_far - gauss_far.mean() + 1

stem_args = dict(
	use_line_collection = True)
fig, axes = plt.subplots(figsize = (9, 12), ncols = 2, nrows = 5)
axes[0, 0].stem(x, filt, **stem_args)
axes[0, 0].set_title('magnitude: %f' % np.sum((filt)**2))
axes[0, 0].set_ylabel("Base Filter")
axes[0, 0].set_ylim(-3, 3)
axes[0, 1].set_axis_off()

axes[1, 0].stem(x, gauss_near, **stem_args)
axes[1, 0].set_ylim(-0.5, 3.)
axes[1, 0].set_ylabel("Attention Field")
axes[1, 1].stem(x, gauss_far, **stem_args)
axes[1, 1].set_ylim(-0.5, 3.)

axes[2, 0].stem(x, filt * gauss_near, **stem_args)
axes[2, 0].set_title('magnitude: %f' % np.sum((filt * gauss_near)**2))
axes[2, 0].set_ylim(-3, 3)
axes[2, 0].set_ylabel("Filter Under Attention")
axes[2, 1].stem(x, filt * gauss_far, **stem_args)
axes[2, 1].set_title('magnitude: %f' % np.sum((filt * gauss_far)**2))
axes[2, 1].set_ylim(-3, 3)

axes[3, 0].stem(x, sens_near, **stem_args)
axes[3, 0].set_ylim(-0.5, 3.)
axes[3, 0].set_ylabel("Normalized Attention")
axes[3, 1].stem(x, sens_far, **stem_args)
axes[3, 1].set_ylim(-0.5, 3.)


axes[4, 0].stem(x, filt * sens_near, **stem_args)
axes[4, 0].set_title('magnitude: %f' % np.sum((filt * sens_near)**2))
axes[4, 0].set_ylim(-3, 3)
axes[4, 0].set_ylabel("Filter Under Normalized Attention")
axes[4, 1].stem(x, filt * sens_far, **stem_args)
axes[4, 1].set_title('magnitude: %f' % np.sum((filt * sens_far)**2))
axes[4, 1].set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('plots/figures/fig5/sens_shift_diagram.pdf')