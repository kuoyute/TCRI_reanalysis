from functools import partial

import tensorflow as tf
import tensorflow_addons as tfa

from modules.tfrecord_generator import get_or_generate_tfrecord


def ascii_array_to_string(ascii_array):
    string = ''
    for ascii_code in ascii_array:
        string += chr(ascii_code)
    return string


def deserialize(serialized_TC_history):
    features = {
        'history_len': tf.io.FixedLenFeature([], tf.int64),
        'images': tf.io.FixedLenFeature([], tf.string),
        'intensity': tf.io.FixedLenFeature([], tf.string),
        'frame_ID': tf.io.FixedLenFeature([], tf.string),
        'env_feature': tf.io.FixedLenFeature([], tf.string),
        'status_feature': tf.io.FixedLenFeature([], tf.string),
        'VWSdir': tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(serialized_TC_history, features)
    history_len = tf.cast(example['history_len'], tf.int32)

    images = tf.reshape(tf.io.decode_raw(example['images'], tf.float64), [history_len, 100, 100, 1])
    images = tf.cast(images, tf.float32)

    intensity = tf.reshape(tf.io.decode_raw(example['intensity'], tf.float64), [history_len])
    intensity = tf.cast(intensity, tf.float32)

    VWSdir = tf.reshape(tf.io.decode_raw(example['VWSdir'], tf.float64), [history_len])
    VWSdir = tf.cast(VWSdir, tf.float32)
    
    env_feature = tf.reshape(tf.io.decode_raw(example['env_feature'], tf.float64), [-1, history_len])
    env_feature = tf.cast(env_feature, tf.float32)

    status_feature = tf.reshape(tf.io.decode_raw(example['status_feature'], tf.float64), [-1, history_len])
    status_feature = tf.cast(status_feature, tf.float32)

    frame_ID_ascii = tf.reshape(tf.io.decode_raw(example['frame_ID'], tf.uint8), [history_len, -1])

    return images, intensity, env_feature, history_len, frame_ID_ascii, VWSdir, status_feature


def breakdown_into_sequence(
    images, intensity, env_feature, history_len, frame_ID_ascii, status_feature, encode_length, estimate_distance
):
    sequence_num = history_len - (encode_length + estimate_distance) -1
    starting_index = tf.range(1, sequence_num+1)

    image_sequences = tf.map_fn(
        lambda start: images[start : start + encode_length], starting_index, fn_output_signature=tf.float32
    )
    
#     intensity -= TC_spd * 0.54  # km/hr to kt

    starting_frame_ID_ascii = frame_ID_ascii[encode_length: -estimate_distance-1]    
    starting_intensity = intensity[encode_length : -estimate_distance-1]

    ending_intensity = intensity[encode_length + estimate_distance :-1]

    labels = ending_intensity
    labels_prior = intensity[encode_length + estimate_distance -1 :-2]
    labels_latter = intensity[encode_length + estimate_distance +1  :]
    
    intensity_change = ending_intensity - starting_intensity

    starting_env_feature = env_feature[:, encode_length: -estimate_distance-1]
    ending_env_feature = env_feature[:, encode_length + estimate_distance :-1]
    
    starting_status_feature = status_feature[:-1, encode_length : -estimate_distance-1]
    dv_3h = status_feature[-1, encode_length : -estimate_distance-1]
    
    feature = tf.concat([[starting_intensity], starting_env_feature, ending_env_feature, starting_status_feature], 0)
    feature = tf.transpose(feature)
    
    prior_starting_env_feature = env_feature[:, encode_length-1: -estimate_distance-1-1]
    prior_ending_env_feature = env_feature[:, encode_length + estimate_distance-1 :-1-1]    
    prior_starting_intensity = intensity[encode_length-1 : -estimate_distance-1-1]
    
    prior_feature = tf.concat([[prior_starting_intensity], prior_starting_env_feature, prior_ending_env_feature, starting_status_feature], 0)
    prior_feature = tf.transpose(prior_feature)
    
    
    latter_starting_env_feature = env_feature[:, encode_length+1: -estimate_distance-1+1]
    latter_ending_env_feature = env_feature[:, encode_length + estimate_distance+1 :]
    latter_starting_intensity = intensity[encode_length+1 : -estimate_distance-1+1]
    
    latter_feature = tf.concat([[latter_starting_intensity], latter_starting_env_feature, latter_ending_env_feature, starting_status_feature], 0)
    latter_feature = tf.transpose(latter_feature)
    
    
    return tf.data.Dataset.from_tensor_slices(
        (image_sequences, labels, labels_prior, labels_latter, feature, prior_feature, latter_feature, starting_frame_ID_ascii, dv_3h, intensity_change)
    )


def image_preprocessing(
    images, intensity, env_feature, history_len, frame_ID_ascii, VWSdir, status_feature, rotate_type, input_image_type
):
    images_channels = tf.gather(images, input_image_type, axis=-1)
    if rotate_type == 'single':
        angles = tf.random.uniform([history_len], maxval=360)
        rotated_images = tfa.image.rotate(images_channels, angles=angles)
    elif rotate_type == 'series':
        angles = tf.ones([history_len]) * tf.random.uniform([1], maxval=360)
        rotated_images = tfa.image.rotate(images_channels, angles=angles)
    elif rotate_type == 'shear':
        print('this is the shear rotation run')
        rotated_images = tfa.image.rotate(images_channels, angles=-VWSdir * 0.0174533)
    else:
        rotated_images = images_channels
        
    images_64x64 = tf.image.central_crop(rotated_images, 0.64)
    
    return images_64x64, intensity, env_feature, history_len, frame_ID_ascii, status_feature


def get_tensorflow_datasets(
    data_folder, batch_size, encode_length, estimate_distance, rotate_type, input_image_type
):
    tfrecord_paths = get_or_generate_tfrecord(data_folder)
    datasets = dict()

    min_history_len = encode_length + estimate_distance+2

    for phase, record_path in tfrecord_paths.items():
        serialized_TC_histories = tf.data.TFRecordDataset([record_path], num_parallel_reads=8)
        TC_histories = serialized_TC_histories.map(deserialize, num_parallel_calls=tf.data.AUTOTUNE)

        long_enough_histories = TC_histories.filter(lambda a, b, c, d, e, f, g: d >= min_history_len)

        preprocessed_histories = long_enough_histories.map(
            partial(image_preprocessing, rotate_type=rotate_type, input_image_type=input_image_type),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        TC_sequence = preprocessed_histories.flat_map(
            partial(
                breakdown_into_sequence, encode_length=encode_length, estimate_distance=estimate_distance
            )
        )

        dataset = TC_sequence.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(4)
        datasets[phase] = dataset

    return datasets
