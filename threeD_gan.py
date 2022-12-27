import numpy
import tensorflow as tf
import variables as var
import data_processing as dp
import dataIO as d
import custom_layers

# Weight initialization described in Radford et al. [2016]
# Filter values are equivalent to channels in Wu et al. [2016]
def make_generator_model():
    res = var.threeD_gan_resolution
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv3DTranspose(filters=var.threeD_gan_generator_first_filter, 
        kernel_size=(4, 4, 4), strides=1, use_bias=False, input_shape=(1, 1, 1, var.threeD_gan_z_size), 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    assert model.output_shape == (None, 4, 4, 4, 512)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    layer_width = 4
    for filter in var.threeD_gan_generator_intermediate_filters:
        layer_width = layer_width * 2
        model.add(tf.keras.layers.Conv3DTranspose(filters=filter, kernel_size=(4, 4, 4), strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
        assert model.output_shape == (None, layer_width, layer_width, layer_width, filter)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv3DTranspose(filters=var.threeD_gan_generator_final_filter, kernel_size=(4, 4, 4), 
        strides=2, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), 
        activation='sigmoid'))
        # activation='tanh'))
    assert model.output_shape == (None, res, res, res, 1)

    return model


def make_discriminator_model():
    res = var.threeD_gan_resolution
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv3D(filters=var.threeD_gan_discriminator_first_filter, kernel_size=4, 
        strides=2, padding='same', use_bias=False, input_shape=(res, res, res, 1),
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))

    for filter in var.threeD_gan_discriminator_intermediate_filters:
        model.add(tf.keras.layers.Conv3D(filters=filter, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(0.2))
    
    model.add(tf.keras.layers.Conv3D(filters=var.threeD_gan_discriminator_final_filter, 
        kernel_size=4, strides=1, padding='valid', use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
        activation='sigmoid'))
    model.add(tf.keras.layers.Flatten())

    return model


# takes a 3x256x256 image as input and outputs a 200 dimensional vector
def make_encoder_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(filters=64, kernel_size=11, strides=4, padding="same", use_bias=False,
                input_shape=(256, 256, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=400, kernel_size=8, strides=1, padding="valid", use_bias=False),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            # At this point we a have a 400 dimensional gaussian latent space, that we need to sample a 200 dimensional vector from
            custom_layers.Sampling(),
        ]
    )
    print(model.output_shape)

