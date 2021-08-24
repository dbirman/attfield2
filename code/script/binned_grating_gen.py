from scipy import fft
import pandas as pd
import numpy as np
import os.path
import tqdm
import os


W = 112
r, c = np.mgrid[:W, :W] - W/2
spacings = np.linspace(5, 10, 5)
angles = np.linspace(np.pi / 64, np.pi / 8, 8)
output_dir = 'data/gratings/ccw/'


if not os.path.exists(os.path.join(output_dir, 'c')):
    os.mkdir(os.path.join(output_dir, 'c'))
if not os.path.exists(os.path.join(output_dir, 'cw')):
    os.mkdir(os.path.join(output_dir, 'cw'))

index = dict(category = [], path = [], n = [], neg_n = [])
cat_max_i = len(angles) * len(spacings) - 1
for i_a, a in tqdm.tqdm(enumerate(angles), total = len(angles), position = 0):
    for ccw_tag, mult in (['c', 1], ['cw', -1]):
        theta = mult * a + np.pi / 2
        direction = np.sin(theta) * c + np.cos(theta) * r
        for i_s, s in enumerate(spacings):
            pass
            grating = np.sin(np.pi / s * direction)
            img = (grating * 127 + 128).astype('uint8')
            filename = f'{ccw_tag}_spacing{i_s}_angle{i_a}.png'

            index['category'].append(ccw_tag)
            index['path'].append(os.path.join(ccw_tag, filename))
            img_i = i_a * len(spacings) + i_s
            index['n'].append(img_i)
            index['neg_n'].append(cat_max_i - img_i)

pd.DataFrame(index).to_csv(os.path.join(output_dir, 'ccw_index.csv'), index = False)

plt.imshow(direction); plt.show()

stutter_noise = np.random.normal(size = W)
stutter_spect = fft.rfft(stutter_noise)
freqs = fft.fftfreq(W)[:W // 2 + 1][1:-2]
plt.plot(freqs, abs(stutter_spect)[1:-2])
plt.plot(freqs, abs(stutter_spect[1:-2] * (freqs.min()/freqs)))
plt.yscale('log'); plt.xscale('log')

plt.plot(freqs, (freqs).min() / freqs**2)
plt.yscale('log'); plt.xscale('log')


def noise1d(NW, FALLOFF_LENGTH, FALLOFF_MIN):
    stutter_noise = np.random.normal(size = NW)
    stutter_spect = fft.rfft(stutter_noise)
    freqs = fft.fftfreq(NW)[:NW // 2 + 1][1:-2]
    stutter_spect_filtered = stutter_spect.copy()
    stutter_spect_filtered[1:-2] = stutter_spect[1:-2] * ((freqs).min()/freqs**2)
    falloff = np.tanh(FALLOFF_LENGTH * np.arange(NW//2 - 2) + FALLOFF_MIN)
    stutter_spect_filtered[1:-2] *= falloff
    stutter_filtered = fft.irfft(stutter_spect_filtered)[::NW//W]
    stutter_filtered /= abs(stutter_filtered).max()
    return stutter_filtered

stutter = noise1d(W * 5, 10, 1e-4)
# falter = noise1d(W * 5, 0.5, 1e-1)
noise_bar = 0.2 * stutter[:, None]
freq = np.full([W, W], 1/s) * (1 + noise_bar)

progress = freq.cumsum(axis = 0)

direction = np.sin(theta) * c + np.cos(theta) * r
plt.imshow(direction); plt.show()

