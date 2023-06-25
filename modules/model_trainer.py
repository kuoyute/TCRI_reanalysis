from collections import defaultdict

import numpy as np
import tensorflow as tf

from modules.training_helper import calculate_metric_dict


def train(model, datasets, summary_writer, saving_path, max_epoch, evaluate_freq, learning_rate, class_weight  
):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_function = tf.keras.losses.MeanSquaredError()
    avg_losses = defaultdict(lambda: tf.keras.metrics.Mean(dtype=tf.float32))

    @tf.function
    def train_step(model, image_sequences, labels, labels_prior, labels_latter, feature, prior_feature, latter_feature, dv3h, dV):
        with tf.GradientTape() as tape:
            
#             random_indices = tf.random.uniform(shape=(feature.shape[0], feature.shape[1]), minval=0, maxval=3, dtype=tf.int32)
#             selected_feature = tf.where(random_indices == 0, feature,
#                                          tf.where(random_indices == 1, prior_feature, latter_feature))
            
            model_output = model(image_sequences, feature, dv3h, training=True)

            sample_weight = tf.math.tanh((dV - 0.5)*1.5) * 20000 + 20001
            sample_weight = tf.expand_dims(sample_weight, axis=1)
            
            data_stack = tf.stack([labels, labels_prior, labels_latter], axis=1)
            new_label = tf.math.reduce_max(data_stack, axis=1)
#             new_label = tf.map_fn(lambda x: x[tf.random.uniform([], maxval=3, dtype=tf.int32)], data_stack, dtype=tf.float32)
            
            target_wmse = tf.reduce_sum(((model_output[:, 0] - new_label)**2) * sample_weight) / tf.reduce_sum(sample_weight)
            short_term_guess_wmse = tf.reduce_sum(((model_output[:, 1] - dv3h)**2) * sample_weight) / tf.reduce_sum(sample_weight)
            batch_loss_wmse = class_weight * target_wmse + short_term_guess_wmse

        gradients = tape.gradient(batch_loss_wmse, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        avg_losses['mean square error'].update_state(batch_loss_wmse)
        return

    best_MAE = np.inf
    best_MSE = np.inf
    for epoch_index in range(1, max_epoch + 1):
        print(f'Executing epoch #{epoch_index}')

        for image_sequences, labels, labels_prior, labels_latter, feature, prior_feature, latter_feature, frame_ID_ascii, dv3h,  dV in datasets['train']:
            train_step(model, image_sequences, labels, labels_prior, labels_latter, feature, prior_feature, latter_feature, dv3h, dV)

        with summary_writer['train'].as_default():
            for loss_name, avg_loss in avg_losses.items():
                tf.summary.scalar(loss_name, avg_loss.result(), step=epoch_index)
                avg_loss.reset_states()

        if epoch_index % evaluate_freq == 0:
            print(f'Completed {epoch_index} epochs, do some evaluation')

            for phase in ['test', 'valid']:
                metric_dict = calculate_metric_dict(model, datasets[phase])
                with summary_writer[phase].as_default():
                    for metric_name, metric_value in metric_dict.items():
                        tf.summary.scalar(metric_name, metric_value, step=epoch_index)

            valid_MAE = metric_dict['MAE']
            valid_MSE = metric_dict['MSE']
            if best_MAE > valid_MAE:
                best_MAE = valid_MAE
                model.save_weights(saving_path / 'best-MAE', save_format='tf')
            if best_MSE > valid_MSE:
                best_MSE = valid_MSE
                model.save_weights(saving_path / 'best-MSE', save_format='tf')
                print('update network')
