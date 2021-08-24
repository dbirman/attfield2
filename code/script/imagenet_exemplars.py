from matplotlib import pyplot as plt
import skimage.io
import h5py
import sys
import os

imagenet_file = sys.argv[1]
imagenet_h5 = h5py.File(imagenet_file, 'r')
out_dir = sys.argv[2]
c = sys.argv[3]
start_loc = 0 if len(sys.argv) < 5 else int(sys.argv[4])



imgs = imagenet_h5[c][...]
exemplars = set()

# Iterate over images and ask whether they are exemplars
for i in range(start_loc, len(imgs)):
	plt.imshow(imgs[i])
	plt.title(f'{c} : {i}')
	plt.axis('off')
	plt.draw()
	plt.pause(0.01)

	action = ""
	while action not in ['e', 'n', 's']:
		action = input(
			f'[n={len(exemplars)}] ' + 
			'Action? ([e]xemplar, [n]on-Exemplar, [s]top-Category) ')

	plt.close()
	if action == 'e':
		exemplars.add(i)
	elif action == 's':
		break

# Arrange and output exemplars
if not os.path.isdir(out_dir):
	os.mkdir(out_dir)
for i_e, e in enumerate(exemplars):
	skimage.io.imsave(f'{out_dir}/{c}_{e}.png', imgs[e].astype('uint8'))

imagenet_h5.close()

