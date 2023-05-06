import numpy
import tensorflow as tf
import variables as var
import data_processing as dp
import dataIO as d
import custom_layers
import os

# Based on the "GENERATIVE ADVERSARIAL NETWORKS FOR SINGLE PHOTO 3D RECONSTRUCTION" by V.V.Kniaz et al.
# and "Image-to-Voxel Model Translation with Conditional Adversarial Networks" by V.V.Kniaz et al.
# pix2pix (Isola et al., 2017) (U-net) is used as the starting point for this model
# the only changes are 3D deconvolution and the required skip connection changes for that
# https://www.tensorflow.org/tutorials/generative/pix2pix
# https://github.com/vlkniaz/Z_GAN/blob/master/models/networks.py
# the result of the generator should be a view centered voxel model
# "We hypothesize that the performance can be improved if the voxel model will be aligned with the input image."

def downsample(filters, kernel_size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, kernel_size, strides=2, padding="same",
        kernel_initializer=initializer, use_bias=False))
    if (apply_batchnorm):
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU(0.2))

    return result

def upsample(filters, kernel_size, strides=2, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv3DTranspose(filters, kernel_size, strides=strides, padding='same',
            kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())

    return result

def make_generator_model():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
        
    down_stack = [
        # (batch_size, 256, 256, 3)
        downsample(64, 4, apply_batchnorm=False), # (batch_size, 128, 128, 64)
        downsample(128, 4), # (batch_size, 64, 64, 128)
        downsample(256, 4), # (batch_size, 32, 32, 256)
        downsample(512, 4), # (batch_size, 16, 16, 512)
        downsample(512, 4), # (batch_size, 8, 8, 512)
        downsample(512, 4), # (batch_size, 4, 4, 512)
        downsample(512, 4), # (batch_size, 2, 2, 512)
        downsample(512, 4), # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        # (batch_size, 1, 1, 1, 1024)
        upsample(512, 4, apply_dropout=True), # (batch_size, 2, 2, 2, 1024)
        # upsample(512, (2,4,4), (1,2,2), apply_dropout=True), #
        upsample(512, 4, apply_dropout=True), # (batch_size, 4, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (batch_size, 8, 8, 8, 1024)
        # upsample(512, (2,4,4), (1,2,2)),
        upsample(512, 4), # (batch_size, 16, 16, 16, 1024)
        upsample(256, 4), # (batch_size, 32, 32, 32, 512)
        upsample(128, 4), # (batch_size, 64, 64, 64, 256)
        upsample(64, 4), # (batch_size, 128, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv3DTranspose(filters=1, kernel_size=4, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False,
        activation='tanh') 

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        print(x.shape)
        skips.append(tf.expand_dims(x, 3))
        while skips[-1].shape[3] != x.shape[2]:
            skips[-1] = tf.concat([skips[-1], skips[-1]], 3)
        print(skips[-1].shape)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    print('upsample')
    x = tf.expand_dims(x, 3)
    print(x.shape)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        print(x.shape)
        x = tf.keras.layers.Concatenate()([x, skip])
        print(x.shape)

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(disc_generated_output, generated_objects, real_objects):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    LAMBDA = 100 # Decided by the authors of the pix2pix paper

    gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)
    recon_loss = tf.keras.losses.mean_absolute_error(real_objects, generated_objects)
    total_gen_loss = gan_loss + (LAMBDA * recon_loss)

    return total_gen_loss, gan_loss, recon_loss

# The papers only say that it is a PatchGAN discriminator with 2D convolution in stead of 3D
def make_discriminator_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers
        ]
    )