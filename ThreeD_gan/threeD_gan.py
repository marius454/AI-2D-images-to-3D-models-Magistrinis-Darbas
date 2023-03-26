import numpy as np
import tensorflow as tf
import dataIO as d
import custom_layers
import os
import time
import gc


from .. import variables as var
from .. import data_processing as dp
import threeD_gan_blocks as blocks
import threeD_gan_metrics as metrics
import threeD_gan_helpers as helpers

from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
import tensorflow_probability as tfp

        
class Generator(tf.keras.Layer):
    def __init__(self):
        '''
            channels = {512, 256, 128, 64, 1}
            input shape = (1, 1, 1, z_size)
            kernel size = 4 x 4 x 4
            strides = {1, 2, 2, 2, 2}
        '''
        super(Generator, self).__init__()
        self.block_1 = blocks.Conv3DTransposeBlock(512, input_shape=(1, 1, 1, 200), strides=1, padding = 'valid')
        self.block_2 = blocks.Conv3DTransposeBlock(256)
        self.block_3 = blocks.Conv3DTransposeBlock(128)
        self.block_4 = blocks.Conv3DTransposeBlock(64)
        self.block_5 = blocks.Conv3DTransposeBlock(1, activation='sigmoid')

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return x



class Discriminator(tf.keras.Layer):
    def __init__(self):
        '''
            channels = {64, 128, 256, 512, 1}
            input shape = (res, res, res, 1)
            kernel size = 4 x 4 x 4
            strides = {2, 2, 2, 2, 1}
        '''
        super(Discriminator, self).__init__()
        self.block_1 = blocks.Conv3DBlock(64, input_shape=(64, 64, 64, 1))
        self.block_2 = blocks.Conv3DBlock(128)
        self.block_3 = blocks.Conv3DBlock(256)
        self.block_4 = blocks.Conv3DBlock(512)
        self.block_5 = blocks.Conv3DBlock(1, strides=1, padding = 'valid', activation='sigmoid')

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return tf.keras.layers.Flatten()(x)


class Encoder(tf.keras.Layer):
    def __init__(self):
        '''
            channels = {64, 128, 256, 512, 400}
            input shape = (imgRes, imgRes, 3)
            kernel size = {11x11, 5x5, 5x5, 5x5, 8x8}
            strides = {4, 2, 2, 2, 1}
        '''
        super(Encoder, self).__init__()
        self.block_1 = blocks.Conv3DBlock(64, input_shape=(256, 256, 3), kernel_size = 11, strides = 4)
        self.block_2 = blocks.Conv3DBlock(128)
        self.block_3 = blocks.Conv3DBlock(256)
        self.block_4 = blocks.Conv3DBlock(512)
        self.block_5 = blocks.Conv3DBlock(400, kernel_size = 8, strides = 1, padding = 'valid')

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x, use_batch_norm = False, use_ReLU = False)
        return tf.keras.layers.Flatten()(x)
    

class ThreeD_gan(tf.keras.Model):
    def __init__(self):
        super(ThreeD_gan, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.encoder = Encoder()

        self.G_optimizer = tf.keras.optimizers.Adam(0.0025, 0.5, 0.5)
        self.D_optimizer = tf.keras.optimizers.Adam(1e-5, 0.5, 0.5)
        self.E_optimizer = tf.keras.optimizers.Adam(0.0003)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, inputs):
        encoded_noise = self.encoder(inputs)
        generated_models = self.generator(encoded_noise)
        return generated_models
    

    def train_step(self, data):
        # Data will be a tensorflow dataset containing (image, 3d_model) pairs
        # train_step is called for every batch of data

        images, real_shapes = data
        real_shapes = tf.cast(real_shapes, tf.float32)
        real_shapes = tf.reshape(real_shapes, (real_shapes.shape[0], real_shapes.shape[1], 
                                               real_shapes.shape[2], real_shapes.shape[3], 1))
        noise = get_z(0, 1, len(images))

        # Train the discriminator:
        generated_shapes = self.generator(inputs=noise, training=False)
        with tf.GradientTape() as disc_tape:
            real_output = self.discriminator(inputs=real_shapes, training=True)
            generated_output = self.discriminator(inputs=generated_shapes, training=True)

            disc_loss = metrics.discriminator_loss(self.cross_entropy, real_output, generated_output)
        
        disc_accuracy = metrics.discriminator_accuracy(real_output, generated_output)
        if (disc_accuracy < 0.8):
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
            self.D_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_weights))

        # Train the encoder:
        with tf.GradientTape() as ecn_tape:
            encoder_output = self.encoder(inputs=images, training=True)
            z_means, z_vars = tf.split(encoder_output, num_or_size_splits=2, axis=1)
            z = get_z(z_means, z_vars, len(images))

            enc_loss = metrics.vae_loss(z_means, z_vars, real_shapes, 
                                        generated_shapes = self.generator(inputs=z, training=False))

        gradients_of_encoder = ecn_tape.gradient(enc_loss, self.encoder.trainable_weights)
        self.E_optimizer.apply_gradients(zip(gradients_of_encoder, self.encoder.trainable_weights))

        # Train the generator:
        encoder_output = self.encoder(inputs=images, training=False)
        z_means, z_vars = tf.split(encoder_output, num_or_size_splits=2, axis=1)
        z = get_z(z_means, z_vars, len(images))

        with tf.GradientTape() as gen_tape:
            shapes_from_noise = self.generator(inputs=noise, training=True)
            output_from_noise = self.discriminator(inputs=shapes_from_noise, training=False)
            del shapes_from_noise

            generated_shapes = self.generator(inputs=z, training=True)
            gen_loss = metrics.generator_loss(self.cross_entropy, output_from_noise, real_shapes, generated_shapes)

        # gen_accuracy = metrics.generator_accuracy(output_from_noise)
        # if gen_accuracy < 0.8:
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
        self.G_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_weights))
