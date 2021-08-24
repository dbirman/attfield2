import pandas as pd
import numpy as np
import h5py
import os

os.chdir('/Users/kaifox/projects/art_physio')
fname = 'bhv_base_accw'
bhv_h5 = f'data/runs/acuity/112/{fname}.h5'
metadata_csv = 'data/gratings/augment_ccw_four224_meta.csv'
output_fmt = f'data/runs/acuity/112/by_spacing/{fname}' + '_spacing{}.h5'
metacol = 'tl_spacing_n'

metadata = pd.read_csv(metadata_csv)
bhv = h5py.File(bhv_h5, 'r')

for level in metadata[metacol].unique():
	out_path = output_fmt.format(level)
	print(f"Copying {metacol} = {level} to {out_path}")
	out_bhv = h5py.File(out_path, 'w')
	try:
		for c, cat_meta in metadata.groupby('target'):
			at_level_ixs = cat_meta[metacol] == level
			bhv_ixs = np.isin(bhv[f'{c}_ix'], np.where(at_level_ixs.values)[0])
			out_y  = bhv[f'{c}_y' ][bhv_ixs]
			out_fn = bhv[f'{c}_fn'][bhv_ixs]
			out_ix = bhv[f'{c}_ix'][bhv_ixs]
			out_bhv.create_dataset(f'{c}_y',  (len(out_y),), np.int8)
			out_bhv.create_dataset(f'{c}_fn', (len(out_y),), np.float32)
			out_bhv.create_dataset(f'{c}_ix', (len(out_y),), np.float32)
			out_bhv[f'{c}_y' ][...] = out_y
			out_bhv[f'{c}_y' ].attrs['cat_names'] = bhv[f'{c}_y' ].attrs['cat_names']
			out_bhv[f'{c}_fn'][...] = out_fn
			out_bhv[f'{c}_ix'][...] = out_ix
	except:
		print("[errored]")
	finally:
		out_bhv.close()

bhv.close()


