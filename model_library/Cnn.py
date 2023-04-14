import tensorflow as tf
from tensorflow.keras import layers


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.input_norm = layers.BatchNormalization()
        self.input_norm_aux = layers.BatchNormalization()

        self.encoding_layer = [
            layers.Conv2D(filters=16, kernel_size=(3, 3), activation='LeakyReLU'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='LeakyReLU'),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='LeakyReLU'),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='LeakyReLU'),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='LeakyReLU'),
        ]

        self.output_layers = [
            layers.Dense(units=512, activation='LeakyReLU'),
            layers.Dropout(rate=0.4),
            layers.Dense(units=64, activation='LeakyReLU'),
            layers.Dense(units=1, activation='LeakyReLU'),
        ]

    def apply_list_of_layers(self, input, list_of_layers, training):
        x = input
        for layer in list_of_layers:
            x = layer(x, training=training)
        return x

    def call(self, image_sequences, feature, training):
        batch_size, encode_length, height, width, channels = image_sequences.shape

        # image_encoder block
        images = tf.reshape(image_sequences, [batch_size * encode_length, height, width, channels])

        normalized_images = self.input_norm(images, training=training)
        encoded_images = self.apply_list_of_layers(normalized_images, self.encoding_layer, training)

        flatten_feature = tf.reshape(encoded_images, [batch_size, -1])

        feature = self.input_norm_aux(feature, training=training)

        combine_feature = tf.concat([flatten_feature, feature], 1)

        # output block
        output = self.apply_list_of_layers(combine_feature, self.output_layers, training)
        return output
