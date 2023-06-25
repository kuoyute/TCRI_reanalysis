import tensorflow as tf
from tensorflow.keras import layers


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.encoding_layer = [
            layers.Conv2D(filters=16, kernel_size=(3, 3), activation='LeakyReLU'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='LeakyReLU'),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='LeakyReLU'),
        ]
        
        self.spatial_attention = layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid')
        
        self.global_pooling = layers.GlobalMaxPooling2D()
        
        self.short_term_output_layers = [
            layers.Dense(units=512, activation='LeakyReLU'),
            layers.Dropout(rate=0.1),
            layers.Dense(units=64, activation='LeakyReLU'),
            layers.Dense(units=1, activation='LeakyReLU'),
        ]
        
        self.final_output_layers= [
            layers.Dense(units=512, activation='LeakyReLU'),
            layers.Dropout(rate=0.1),
            layers.Dense(units=64, activation='LeakyReLU'),
            layers.Dense(units=1, activation='LeakyReLU'),
        ]

    def apply_list_of_layers(self, input, list_of_layers, training):
        x = input
        for layer in list_of_layers:
            x = layer(x, training=training)
        return x


    def call(self, image_sequences, feature, dv_3h, training):
        batch_size, encode_length, height, width, channels = image_sequences.shape

        # image_encoder block
        images = tf.reshape(image_sequences, [batch_size, height, width, channels * encode_length])

        encoded_images = self.apply_list_of_layers(images, self.encoding_layer, training)

        # Spatial Attention
        attention = self.spatial_attention(encoded_images)
        attention_out = encoded_images * attention
        
        # Global Pooling
        global_pooling_features = self.global_pooling(attention_out)

        short_term_change = self.apply_list_of_layers(
            global_pooling_features, self.short_term_output_layers, training
        )
                
        combine_feature = tf.concat([global_pooling_features, feature, tf.reshape(dv_3h, [batch_size, -1])-short_term_change], 1)

        # output block
        output = self.apply_list_of_layers(
            combine_feature, self.final_output_layers, training
        )
        
        output = tf.concat([output, short_term_change], 1)
        
        return output
