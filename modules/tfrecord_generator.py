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

    info_df['Dis2Land'] = info_df['Dis2Land'].fillna(method='bfill')
    info_df['delta_V_3h'] = info_df['delta_V_3h'].fillna(0)
    info_df['TC_spd'] = info_df['TC_spd'].fillna(method='bfill')
    info_df['TC_dir'] = info_df['TC_dir'].fillna(method='bfill')
    info_df['SST'] = info_df['SST'].fillna(method='ffill')
    info_df['POT'] = info_df['POT'].fillna(method='ffill')

    return image_matrix, info_df


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

        region_one_hot = {'WPAC': 1.0, 'EPAC': 2.0, 'AL': 3.0, 'SH': 4.0, 'CPAC': 5.0, 'IO': 6.0}
        region_string = list(single_TC_info.basin)
        region = []
        for i in range(len(region_string)):
            region.append(region_one_hot[region_string[i]])
        region = np.array(region)

        # --- time feature ---
        single_TC_info['local_time'] = pd.to_datetime(single_TC_info.LST)
        single_TC_info['hour_transform'] = single_TC_info.apply(
            lambda x: x.local_time.hour / 24 * 2 * math.pi, axis=1
        )
        single_TC_info['hour_sin'] = single_TC_info.hour_transform.apply(lambda x: math.sin(x))
        single_TC_info['hour_cos'] = single_TC_info.hour_transform.apply(lambda x: math.cos(x))
        local_time_sin = single_TC_info.hour_sin.to_numpy(dtype='float')
        local_time_cos = single_TC_info.hour_cos.to_numpy(dtype='float')

        lat = single_TC_info.TC_lat.to_numpy(dtype='float')
        Vmax = single_TC_info.Vmax.to_numpy(dtype='float')
        delta_V_3h = single_TC_info.delta_V_3h.to_numpy(dtype='float')
        TC_spd = single_TC_info.TC_spd.to_numpy(dtype='float')
        SST = single_TC_info.SST.to_numpy(dtype='float')
        POT = single_TC_info.POT.to_numpy(dtype='float')
        TPW = single_TC_info.TPW.to_numpy(dtype='float')
        T200 = single_TC_info.T200.to_numpy(dtype='float')
        RH700 = single_TC_info.RH700.to_numpy(dtype='float')
        RH850 = single_TC_info.RH850.to_numpy(dtype='float')
        VOR850 = single_TC_info.VOR850.to_numpy(dtype='float')
        VWSdir = single_TC_info.VWSdir.to_numpy(dtype='float')
        VWSmag = single_TC_info.VWSmag.to_numpy(dtype='float')
        LMFmag = single_TC_info.LMFmag.to_numpy(dtype='float')

        if region_string[0] == 'SH':
            VWSdir = (540.0 - VWSdir) % 360.0
            lat = -lat

        shr_x = []
        shr_y = []
        for i in range(len(region_string)):
            shr_x.append(VWSmag[i] * math.cos(math.radians(VWSdir[i])))
            shr_y.append(VWSmag[i] * math.sin(math.radians(VWSdir[i])))

        env_feature = np.array(
            [
                region,
                local_time_sin,
                local_time_cos,
                lat,
                delta_V_3h,
                POT,
                TC_spd,
                SST,
                POT,
                TPW,
                T200,
                RH700,
                RH850,
                VOR850,
                VWSmag,
                shr_x,
                shr_y,
                LMFmag,
            ]
        )

        features = {
            'history_len': _int64_feature(history_len),
            'images': _bytes_feature(np.ndarray.tobytes(single_TC_images)),
            'intensity': _bytes_feature(np.ndarray.tobytes(Vmax)),
            'frame_ID': _bytes_feature(np.ndarray.tobytes(frame_ID.to_numpy('bytes'))),
            'env_feature': _bytes_feature(np.ndarray.tobytes(env_feature)),
            'VWSdir': _bytes_feature(np.ndarray.tobytes(VWSdir)),
            'TC_spd': _bytes_feature(np.ndarray.tobytes(TC_spd)),
        }
        return tf.train.Example(features=tf.train.Features(feature=features))

    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        assert len(image_matrix) == len(info_df)
        for single_TC_images, single_TC_info in zip(image_matrix, info_df):
            example = _encode_tfexample(single_TC_images, single_TC_info)
            serialized = example.SerializeToString()
            writer.write(serialized)


def generate_tfrecord(data_folder):
    file_path = Path(data_folder, 'TCSA_reanalysis.h5')
    if not file_path.exists():
        raise ValueError(
            "h5 file and tfrecord not found. Please check h5 file name or link to {data_folder}."
        )

    with h5py.File(file_path, 'r') as hf:
        image_matrix = hf['matrix'][:]
    # collect info from every file in the list
    info_df = pd.read_hdf(file_path, key='info', mode='r')
    image_matrix, info_df = data_cleaning_and_organizing(image_matrix, info_df)

    phase_data = {phase: data_split(image_matrix, info_df, phase) for phase in ['train', 'valid', 'test']}
    del image_matrix, info_df

    for phase, (image_matrix, info_df) in phase_data.items():
        image_matrix, info_df = group_by_id(image_matrix, info_df)
        phase_path = Path(data_folder, f'TCSA.tfrecord.{phase}')
        write_tfrecord(image_matrix, info_df, phase_path)


def get_or_generate_tfrecord(data_folder):
    tfrecord_path = {}
    for phase in ['train', 'valid', 'test']:
        phase_path = Path(data_folder, f'TCSA.tfrecord.{phase}')
        if not phase_path.exists():
            print(f'tfrecord {phase_path} not found! try to generate it!')
            generate_tfrecord(data_folder)
        tfrecord_path[phase] = phase_path

    return tfrecord_path
