import tensorflow as tf


def calculate_metric_dict(model, dataset):
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    mse_metric = tf.keras.metrics.MeanSquaredError()

    for image_sequences, labels, labels_prior, labels_latter, feature, _, _, _, dv3h, dV in dataset:
        data_stack = tf.stack([labels, labels_prior, labels_latter], axis=1)
        new_label = tf.math.reduce_max(data_stack, axis=1)
            
        pred = model(image_sequences, feature, dv3h, training=False)[:,0]
        sample_weight = tf.math.tanh((dV - 0.5)*1.5) * 20000 + 20001

        mae_metric.update_state(tf.reshape(new_label, (-1, 1)), pred, tf.reshape(sample_weight, (-1, 1)))
        mse_metric.update_state(tf.reshape(new_label, (-1, 1)), pred, tf.reshape(sample_weight, (-1, 1)))

    MAE = mae_metric.result()
    MSE = mse_metric.result()

    return dict(MAE=MAE, MSE=MSE)
