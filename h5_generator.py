import argparse
import os

import h5py
import numpy as np
import pandas as pd
import xarray
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', help='path to the folder \'TC_data\'')
parser.add_argument('--output', help='path to the output hdf5 file', default='./TCSA_data/TCSA_reanalysis.h5')
args = parser.parse_args()

f = h5py.File(args.output, 'w')

# =======matrix data=========
regions = ['1_WP', '2_EP', '3_AL', '4_SH', '5_IO', '6_CP']
data_info = pd.DataFrame()

data_matrix = []
for region in regions:
    print(f'Processing region: {region[2:]}')
    region_path = os.path.join(args.data_folder, region)
    cyclones = sorted(os.listdir(region_path))

    for cyclone in tqdm(cyclones):
        cyclone_path = os.path.join(region_path, cyclone)
        frames = os.listdir(cyclone_path)
        for frame in frames:
            frame_path = os.path.join(cyclone_path, frame)
            frame_data = xarray.open_dataset(frame_path)
            frame_info = pd.json_normalize(frame_data.attrs).drop(columns=['PF_PF'])

            data_info = pd.concat([data_info, frame_info], axis=0)
            # transfer 201*201 data matrix into 64*64 numpy ndarray, select only IR and WV
            data_201x201 = frame_data.to_array().values.transpose([1, 2, 0])
            data_64x64_IR_WV = data_201x201[68:132, 68:132, :2]
            data_matrix.append(data_64x64_IR_WV)

data_info = data_info.reset_index(drop=True).sort_values(['basin', 'TC_ID', 'datetime'])
sorted_idx = np.array(data_info.index)
data_info.reset_index(drop=True, inplace=True)

data_matrix = np.stack(data_matrix)
data_matrix = data_matrix[sorted_idx]
f['matrix'] = data_matrix
del data_matrix

f.close()
data_info.to_hdf(args.output, 'info')
