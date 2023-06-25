import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

pd.options.mode.chained_assignment = None


def remove_outlier_and_nan(numpy_array, upper_bound=1000):
    numpy_array = np.nan_to_num(numpy_array, copy=False)
    numpy_array[numpy_array > upper_bound] = 0
    return numpy_array


def flip_SH_images(image_matrix, info_df):
    SH_idx = info_df.index[info_df.basin == 'SH']
    image_matrix[SH_idx] = np.flip(image_matrix[SH_idx], 1)
    return image_matrix


def data_cleaning_and_organizing(image_matrix, info_df):
    image_matrix = remove_outlier_and_nan(image_matrix)
    image_matrix = flip_SH_images(image_matrix, info_df)

    float_columns = list(info_df.columns)
    str_lsit = ['TC_ID', 'basin', 'LST', 'datetime']
    for elem in str_lsit:
        if elem in float_columns:
            float_columns.remove(elem)

    # convert the specified columns to float32
    info_df[float_columns] = info_df[float_columns].astype('float32')

    info_df.loc[((info_df['SST'].isna()) & (info_df['Dis2Land'] > 500)), 'Dis2Land'] = pd.NA
    info_df['Dis2Land'] = info_df['Dis2Land'].fillna(method='bfill')
    info_df['delta_V_3h'] = info_df['delta_V_3h'].fillna(0)
    info_df['delta_V_6h'] = info_df['delta_V_6h'].fillna(0)
    info_df['R34_PF'] = info_df['R34_PF'].fillna(0)
    info_df['TC_spd'] = info_df['TC_spd'].fillna(method='bfill')
    info_df['TC_dir'] = info_df['TC_dir'].fillna(method='bfill')
    info_df['TC_spd'] = info_df['TC_spd'].fillna(method='ffill')
    info_df['TC_dir'] = info_df['TC_dir'].fillna(method='ffill')
    info_df['SST'] = info_df['SST'].fillna(method='ffill')
    info_df['POT'] = info_df['POT'].fillna(method='ffill')
    
    info_df.loc[(info_df['TC_lat']<0), 'VWSdir'] = (540.0 - info_df.VWSdir) % 360.0
    info_df.loc[(info_df['TC_lat']<0), 'TC_lat'] = -info_df.TC_lat
    
    info_df['hour_transform'] = info_df.apply(lambda x: x.LST.hour / 24 * 2 * math.pi, axis=1)
    info_df['local_time_sin'] = info_df.hour_transform.apply(lambda x: math.sin(x))
    info_df['local_time_cos'] = info_df.hour_transform.apply(lambda x: math.cos(x))
    
    info_df['shr_x'] = info_df.apply(lambda x: x.VWSmag * math.cos(math.radians(x.VWSdir)), axis=1)
    info_df['shr_y'] = info_df.apply(lambda x: x.VWSmag * math.sin(math.radians(x.VWSdir)), axis=1) 
    
    info_df = info_df.replace({'WPAC': 1.0, 'EPAC': 2.0, 'AL': 3.0, 'SH': 4.0, 'CPAC': 5.0, 'IO': 6.0})

    info_df['Vmax'] = info_df['Vmax'] - info_df['TC_spd']*0.54

    return image_matrix, info_df


def normalize(image_matrix, info_df):
    non_str_cols = info_df.select_dtypes(exclude='object').columns
    for col in non_str_cols:
        info_df[col] = (info_df[col] - info_df[col].mean()) / info_df[col].std()
        
    image_matrix = (image_matrix - np.nanmean(image_matrix))/np.nanstd(image_matrix)
    
    return  image_matrix, info_df


def data_split(image_matrix, info_df, phase):
    if phase == 'train':
        target_index = info_df.index[info_df.TC_ID < '2017000']
    elif phase == 'valid':
        target_index = info_df.index[(info_df.TC_ID > '2017000') & (info_df.TC_ID < '2019000')]
    elif phase == 'test':
        target_index = info_df.index[info_df.TC_ID > '2019000']

    new_image_matrix = image_matrix[target_index]
    new_info_df = info_df.loc[target_index].reset_index(drop=True)
    return new_image_matrix, new_info_df


def group_by_id(image_matrix, info_df):
    id2indices_group = info_df.groupby('TC_ID', sort=False).groups
    indices_groups = list(id2indices_group.values())

    image_matrix = [image_matrix[indices] for indices in indices_groups]
    info_df = [info_df.iloc[indices] for indices in indices_groups]

    return image_matrix, info_df


def write_tfrecord(image_matrix, info_df, tfrecord_path):
    def _bytes_feature(value: float) -> tf.train.Feature:
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _encode_tfexample(single_TC_images, single_TC_info):
        history_len = single_TC_info.shape[0]
        frame_ID = single_TC_info.TC_ID + '_' + single_TC_info.datetime

        env_name = ['TC_lat',
                    'POT',
                    'SST',
                    'TPW',
                    'T200',
                    'RH700',
                    'RH850',
                    'VOR850',
                    'VWSmag',
                    'shr_x',
                    'shr_y',
                    'LMFmag',
                    'Dis2Land',
                    'TC_spd',
                    ]
        
        TC_status = ['basin',
                    'local_time_sin',
                    'local_time_cos',
                    'R34_PF',
                    'delta_V_6h',
                    'delta_V_3h',
                    ]
        
        Vmax = single_TC_info.Vmax.to_numpy(dtype='float')
        VWSdir = single_TC_info.VWSdir.to_numpy(dtype='float')
        
        env_data = []
        for data_name in env_name:
            env_data.append(single_TC_info[data_name].to_numpy(dtype='float'))
        env_data = np.array(env_data)
        
        status_data = []
        for data_name in TC_status:
            status_data.append(single_TC_info[data_name].to_numpy(dtype='float'))
        status_data = np.array(status_data)
        
        features = {
            'history_len': _int64_feature(history_len),
            'images': _bytes_feature(np.ndarray.tobytes(single_TC_images)),
            'intensity': _bytes_feature(np.ndarray.tobytes(Vmax)),
            'frame_ID': _bytes_feature(np.ndarray.tobytes(frame_ID.to_numpy('bytes'))),
            'env_feature': _bytes_feature(np.ndarray.tobytes(env_data)),
            'status_feature': _bytes_feature(np.ndarray.tobytes(status_data)),
            'VWSdir': _bytes_feature(np.ndarray.tobytes(VWSdir)),
        }
        return tf.train.Example(features=tf.train.Features(feature=features))

    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        assert len(image_matrix) == len(info_df)
        for single_TC_images, single_TC_info in zip(image_matrix, info_df):
            example = _encode_tfexample(single_TC_images, single_TC_info)
            serialized = example.SerializeToString()
            writer.write(serialized)


def generate_tfrecord(data_folder):
    file_path = Path(data_folder, 'TCRI_reanalysis.h5')
    if not file_path.exists():
        raise ValueError(
            "h5 file and tfrecord not found. Please check h5 file name or link to {data_folder}."
        )

    with h5py.File(file_path, 'r') as hf:
        image_matrix = hf['matrix'][:]
    # collect info from every file in the list
    info_df = pd.read_hdf(file_path, key='info', mode='r')
    info_df['LST'] = pd.to_datetime(info_df['LST'])
    image_matrix, info_df = data_cleaning_and_organizing(image_matrix, info_df)
    image_matrix, info_df = normalize(image_matrix, info_df)

    phase_data = {phase: data_split(image_matrix, info_df, phase) for phase in ['train', 'valid', 'test']}
    del image_matrix, info_df

    for phase, (image_matrix, info_df) in phase_data.items():
        image_matrix, info_df = group_by_id(image_matrix, info_df)
        phase_path = Path(data_folder, f'TCRI.tfrecord.{phase}')
        write_tfrecord(image_matrix, info_df, phase_path)


def get_or_generate_tfrecord(data_folder):
    tfrecord_path = {}
    for phase in ['train', 'valid', 'test']:
        phase_path = Path(data_folder, f'TCRI.tfrecord.{phase}')
        if not phase_path.exists():
            print(f'tfrecord {phase_path} not found! try to generate it!')
            generate_tfrecord(data_folder)
        tfrecord_path[phase] = phase_path

    return tfrecord_path
