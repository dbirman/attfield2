import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

f = h5py.File('data/gratings/augment_ccw_four224.h5', 'r')

with PdfPages("plots/runs/acuity/images_accw.pdf") as pdf:
    for cat in [c for c in f.keys()
                 if ((not c.endswith('_y')) and
                     (not c.endswith('_loc')))]:

        nimg = 5
        fig, ax = plt.subplots(2, nimg, figsize = (nimg*2, 6))
        for i in range(nimg):
            imgix = int(i/nimg * (f[cat].shape[0]/2)) * 2
            ax[0, i].imshow(f[cat][imgix])
            ax[0, i].set_title(str(imgix) if i != 0 else f"Positive: {imgix}")
            ax[0, i].set_axis_off()
            ax[1, i].imshow(f[cat][imgix+1])
            ax[1, i].set_title(str(imgix+1) if i != 0 else f"Negative: {imgix+1}")
            ax[1, i].set_axis_off()
        plt.suptitle(f"Class: '{cat}'")
        pdf.savefig()
        plt.close()