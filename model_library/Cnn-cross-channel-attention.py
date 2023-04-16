import tensorflow as tf
from tensorflow.keras import layers

class CrossChannelPooling(layers.Layer):
    def __init__(self, reduction_ratio=16):
        super().__init__()
        self.reduction_ratio = reduction_ratio
    
    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.pooling = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(units=self.filters//self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(units=self.filters, activation='sigmoid')
    
    def call(self, inputs):
        # Compute global average pooling
        avg_pool = self.pooling(inputs)
        
        # Compute feature-wise statistics
        fc1_out = self.fc1(avg_pool)
        fc2_out = self.fc2(fc1_out)
        
        # Expand dimensions to allow for broadcasting
        fc2_out = layers.Reshape((1,1,self.filters))(fc2_out)
        
        # Compute output
        out = inputs * fc2_out
        
        return out


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.input_norm = layers.BatchNormalization()
        self.input_norm_aux = layers.BatchNormalization()

        self.encoding_layer = [
            layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        ]

        self.attention = CrossChannelPooling(reduction_ratio=8)

        self.global_pooling = layers.GlobalMaxPooling2D()
        
        self.output_layers = [
            layers.Dense(units=512, activation='LeakyReLU'),
            layers.Dropout(rate=0.05),
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
        images = tf.reshape(image_sequences, [batch_size, height, width, channels * encode_length])

        normalized_images = self.input_norm(images, training=training)
        encoded_images = self.apply_list_of_layers(normalized_images, self.encoding_layer, training=training)

        # Apply channel attention mechanism
        x = self.attention(encoded_images)

        flatten_feature = self.global_pooling(x)
        
        feature = self.input_norm_aux(feature, training=training)

        combine_feature = tf.concat([flatten_feature, feature], 1)

        # output block
        output = self.apply_list_of_layers(combine_feature, self.output_layers, training)
        return output
