import numpy as np
import tensorflow as tf
import dataIO as d
import custom_layers
import os
import time
import gc

import Z_gan.z_gan_blocks as blocks
# import ThreeD_gan.threeD_gan_metrics as metrics
# import ThreeD_gan.threeD_gan_helpers as helpers

from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
import tensorflow_probability as tfp


class Generator(tf.keras.layers.Layer):
    def __init__(self):
        '''
            downsample_channels = {64, 128, 256, 512, 512, 512, 512, 512}
            upsample_channels = {512, 512, 512, 512, 256, 128, 64}
            input shape = (256, 256, 3)
            kernel size = 4 x 4 x 4
            strides = 2
        '''
        super(Generator, self).__init__()
        self.input_layer = tf.keras.layers.Input(shape=[256, 256, 3])
        self.down_stack = [
            # (batch_size, 256, 256, 3)
            blocks.DownsampleBlock(64, 4, use_batchnorm=False), # (batch_size, 128, 128, 64)
            blocks.DownsampleBlock(128, 4), # (batch_size, 64, 64, 128)
            blocks.DownsampleBlock(256, 4), # (batch_size, 32, 32, 256)
            blocks.DownsampleBlock(512, 4), # (batch_size, 16, 16, 512)
            blocks.DownsampleBlock(512, 4), # (batch_size, 8, 8, 512)
            blocks.DownsampleBlock(512, 4), # (batch_size, 4, 4, 512)
            blocks.DownsampleBlock(512, 4), # (batch_size, 2, 2, 512)
            blocks.DownsampleBlock(512, 4), # (batch_size, 1, 1, 512)
        ]
        self.up_stack = [
            # (batch_size, 1, 1, 1, 1024)
            blocks.UpsampleBlock(512, 4, use_dropout=True), # (batch_size, 2, 2, 2, 1024)
            blocks.UpsampleBlock(512, 4, use_dropout=True), # (batch_size, 4, 4, 4, 1024)
            blocks.UpsampleBlock(512, 4, use_dropout=True), # (batch_size, 8, 8, 8, 1024)
            blocks.UpsampleBlock(512, 4), # (batch_size, 16, 16, 16, 1024)
            blocks.UpsampleBlock(256, 4), # (batch_size, 32, 32, 32, 512)
            blocks.UpsampleBlock(128, 4), # (batch_size, 64, 64, 64, 256)
            blocks.UpsampleBlock(64, 4), # (batch_size, 128, 128, 128, 128)
        ]
        self.final_layer = tf.keras.layers.Conv3DTranspose(filters=1, kernel_size=4, strides=2, padding='same',
            kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False, activation='tanh') 
        

    def call(self, inputs):
        x = self.input_layer(inputs)

        # Downsampling through the model
        skips = []
        for down in self.down_stack:
            x = down(x)
            # add a third dimmension for the skip
            skips.append(tf.expand_dims(x, 3))
            # increse the size of the new dimension to match the other two dimmensions
            while tf.shape(skips[-1])[3] != tf.shape(x)[2]:
                skips[-1] = tf.concat([skips[-1], skips[-1]], 3)

        # Upsampling and establishing the skip connections
        skips = reversed(skips[:-1]) # Omits the last skip connection from the list, because there will be a direct connection there
        x = tf.expand_dims(x, 3)
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        return x
    
# The papers only say that it is a PatchGAN discriminator with 3D convolution in stead of 2D
class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        '''
            channels = {64, 128, 256, 512}
            input shape = (128, 128, 128, 1)
            kernel size = 4 x 4 x 4
            strides = 2
        '''
        super(Discriminator, self).__init__()
        self.block_1 = blocks.DiscriminatorBlock(64, input_shape=(128, 128, 128, 1))
        self.block_2 = blocks.DiscriminatorBlock(128, strides=1)
        self.block_3 = blocks.DiscriminatorBlock(256, strides=1)
        self.block_4 = blocks.DiscriminatorBlock(512, strides=1)
        self.block_5 = blocks.DiscriminatorBlock(1, strides=1, activation='sigmoid')

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return tf.keras.layers.Flatten()(x)


class Z_gan(tf.keras.Model):
    def __init__(self):
        super(Z_gan, self).__init__()
        self.discriminator = Discriminator()
        self.generator = Generator()

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, inputs):
        encoded_noise = self.encoder(inputs)
        generated_models = self.generator(encoded_noise)
        return generated_models
    
    def compile(self, d_optimizer, g_optimizer):
        super(Z_gan, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def train_step(self, data):
        print("TODO")