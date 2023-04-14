import tensorflow as tf


def calculate_metric_dict(model, dataset):
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    mse_metric = tf.keras.metrics.MeanSquaredError()

    for image_sequences, labels, feature, frame_ID_ascii, dV in dataset:
        pred = model(image_sequences, feature, training=False)
        sample_weight = tf.math.tanh((dV - 20) / 10) * 1000 + 1000.1

        mae_metric.update_state(tf.reshape(labels, (-1, 1)), pred, tf.reshape(sample_weight, (-1, 1)))
        mse_metric.update_state(tf.reshape(labels, (-1, 1)), pred, tf.reshape(sample_weight, (-1, 1)))

    MAE = mae_metric.result()
    MSE = mse_metric.result()

    return dict(MAE=MAE, MSE=MSE)
