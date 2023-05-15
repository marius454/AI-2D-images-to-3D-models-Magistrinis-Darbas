import numpy as np
import tensorflow as tf
import dataIO as d
import custom_layers
import os
import time

import Ha_gan.ha_gan_blocks as blocks
import Ha_gan.ha_gan_options as opt

# G^L
class Sub_Generator(tf.keras.layers.Layer):
    def __init__(self):
        '''
            input shape = (64 x 64 x 64 x 64)
            kernel size = 3 x 3 x 3
            strides = {1, 1, 1, 1, 1}
        '''
        super(Sub_Generator, self).__init__()

        self.conv_block_1 = blocks.Conv3DBlock(32, input_shape=(64, 64, 64, 64))
        self.conv_block_2 = blocks.Conv3DBlock(16)
        self.conv_block_3 = blocks.Conv3DBlock(1, activation='tanh')

    def call(self, inputs):
        x = self.conv_block_1(inputs, use_interpolation=False)
        assert tf.shape(x) == (None, 64, 64, 64, 32)
        x = self.conv_block_2(x, use_interpolation=False)
        assert tf.shape(x) == (None, 64, 64, 64, 16)
        x = self.conv_block_3(x, use_group_norm=False, use_ReLU=False, use_interpolation=False)
        assert tf.shape(x) == (None, 64, 64, 64, 1)

        return x


class Generator(tf.keras.layers.Layer):
    def __init__(self, mode='train', latent_dim=opt.latent_dim):
        '''
            input shape = (1, 1024)
            kernel size = 3 x 3 x 3
            strides = {1, 1, 1, 1, 1}
        '''
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.mode = mode

        # G^A
        self.fully_connected = tf.keras.layers.Dense(4*4*4*512)
        self.conv_block_1 = blocks.GNConv3DBlock(512)
        self.conv_block_2 = blocks.GNConv3DBlock(512)
        self.conv_block_3 = blocks.GNConv3DBlock(256)
        self.conv_block_4 = blocks.GNConv3DBlock(128)
        self.conv_block_5 = blocks.GNConv3DBlock(64, use_interpolation=False)

        # G^H
        self.interpolate = tf.keras.layers.UpSampling3D(2)
        self.conv_block_6 = blocks.GNConv3DBlock(32)
        self.conv_block_7 = blocks.GNConv3DBlock(1, activation='tanh', use_group_norm=False, use_ReLU=False, use_interpolation=False)

        # G^L
        self.sub_G = Sub_Generator()

    def call(self, x, r = None):
        if (r != None or self.mode == 'eval'):
            # G^A
            x = self.fully_connected(x)
            x = tf.keras.layers.Reshape((-1, 4, 4, 4, 512))(x)

            x = self.conv_block_1(x)
            assert tf.shape(x) == (None, 8, 8, 8, 512)
            x = self.conv_block_2(x)
            assert tf.shape(x) == (None, 16, 16, 16, 512)
            x = self.conv_block_3(x)
            assert tf.shape(x) == (None, 32, 32, 32, 256)
            x = self.conv_block_4(x)
            assert tf.shape(x) == (None, 64, 64, 64, 128)
            x_latent = self.conv_block_5(x)
            assert tf.shape(x_latent) == (None, 64, 64, 64, 64)

            if (self.mode == 'train'):
                x_small = self.sub_G(x_latent)
                x = x_latent[:, r//4 : r//4+8, :, :, :] # Crop out (8, 64, 64) curretly from x axis, might be better from y axis.
            else:
                x = x_latent

        # G^H
        x = self.interpolate(x)
        assert tf.shape(x) == (None, 128, 128, 128, 64) or tf.shape(x) == (None, 16, 128, 128, 64)
        x = self.conv_block_6(x)
        assert tf.shape(x) == (None, 256, 256, 256, 32) or tf.shape(x) == (None, 32, 256, 256, 32) 
        x = self.conv_block_7(x)
        assert tf.shape(x) == (None, 256, 256, 256, 1) or tf.shape(x) == (None, 32, 256, 256, 1)
        
        if (r != None) and self.mode == 'train':
            return x, x_small
        return x


# D^L
class Sub_Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        '''
            input shape = (64, 64, 64, 1)
            kernel size = 4 x 4 x 4
            strides = {2, 2, 2, 2, 1}
        '''
        super(Sub_Discriminator, self).__init__()

        self.conv_block_1 = blocks.SNConv3DBlock(32, input_shape=(64, 64, 64, 1))
        self.conv_block_2 = blocks.SNConv3DBlock(64)
        self.conv_block_3 = blocks.SNConv3DBlock(128)
        self.conv_block_4 = blocks.SNConv3DBlock(256)
        self.conv_block_5 = blocks.SNConv3DBlock(1, strides = 1, padding='valid', use_norm=False, use_ReLU=False)
    
    def call (self, inputs):
        x = self.conv_block_1(inputs)
        assert tf.shape(x) == (None, 32, 32, 32, 32)
        x = self.conv_block_2(x)
        assert tf.shape(x) == (None, 16, 16, 16, 64)
        x = self.conv_block_3(x)
        assert tf.shape(x) == (None, 8, 8, 8, 128)
        x = self.conv_block_4(x)
        assert tf.shape(x) == (None, 4, 4, 4, 256)
        x = self.conv_block_5(x)
        assert tf.shape(x) == (None, 1, 1, 1, 1)

        return tf.keras.layers.Reshape((-1, 1))(x)


class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        '''
            input shape = (32, 256, 256, 1)
            kernel size = {4, 4, 4, 2x4x4, 2x4x4, 1x4x4}
            strides = {2, 2, 2, 2, 1, 1, 1}
        '''
        super(Discriminator, self).__init__()
        self.conv_block_1 = blocks.SNConv3DBlock(16, input_shape=(32, 256, 256, 1))
        self.conv_block_2 = blocks.SNConv3DBlock(32)
        self.conv_block_3 = blocks.SNConv3DBlock(64)
        self.conv_block_4 = blocks.SNConv3DBlock(128, kernel_size=(2,4,4))
        ## IN THE CODE
        # self.conv_block_5 = blocks.SNConv3DBlock(256, kernel_size=(2,4,4))
        # self.conv_block_6 = blocks.SNConv3DBlock(512, kernel_size=(1,4,4), strides=(1,2,2))
        # self.conv_block_7 = blocks.SNConv3DBlock(128, kernel_size=(1,4,4), strides=1, padding = 'valid')
        ## IN THE PAPER
        self.conv_block_5 = blocks.SNConv3DBlock(256, kernel_size=(2,4,4), strides=1)
        self.conv_block_6 = blocks.SNConv3DBlock(512, kernel_size=(1,4,4), strides=1)
        self.conv_block_7 = blocks.SNConv3DBlock(128, kernel_size=(1,4,4), strides=1, padding = 'valid')

        self.fully_connceted_1 = blocks.SNDenseBlock(64)
        ## IN THE CODE
        # self.fully_connceted_2 = blocks.SNDenseBlock(1)
        ## IN THE PAPER
        self.fully_connceted_2 = blocks.SNDenseBlock(32)
        self.fully_connceted_3 = tf.keras.layers.Dense(1)

        # D^L
        self.sub_D = Sub_Discriminator()

    def call (self, x, x_small, r = None):
        x = self.conv_block_1(x)
        assert tf.shape(x) == (None, 16, 128, 128, 16)
        x = self.conv_block_2(x)
        assert tf.shape(x) == (None, 8, 64, 64, 32)
        x = self.conv_block_3(x)
        assert tf.shape(x) == (None, 4, 32, 32, 64)
        x = self.conv_block_4(x)
        assert tf.shape(x) == (None, 2, 16, 16, 128)
        x = self.conv_block_5(x)
        assert tf.shape(x) == (None, 1, 8, 8, 256)
        x = self.conv_block_6(x)
        assert tf.shape(x) == (None, 1, 4, 4, 512)
        x = self.conv_block_7(x)
        assert tf.shape(x) == (None, 1, 1, 1, 128)
        x = tf.keras.layers.Flatten()(x)
        ## IN THE CODE
        # x = tf.keras.layers.Concatenate(-1)[x, r / 224. * tf.ones([tf.shape(x)[0], 1])]
        
        x = self.fully_connceted_1(x)
        assert tf.shape(x) == (None, 64)
        ## IN THE CODE
        # x_logit = self.fully_connceted_2(x)
        # assert tf.shape(x) == (None, 1)
        ## IN THE PAPER
        x = self.fully_connceted_2(x)
        assert tf.shape(x) == (None, 32)
        x_logit = self.fully_connceted_3(x)
        assert tf.shape(x) == (None, 1)

        x_small_logit = self.sub_D(x_small)

        return (x_logit + x_small_logit) / 2.

class Encoder(tf.keras.layers.Layer):
    # Input: (-1, 256, 256, 3)
    # GNconv1: (128, 128, 32)
    # GNconv2: (128, 128, 32)
    # GNconv3: (64, 64, 64)
    # GNconv4: (32, 32, 32)
    # GNconv5: (16, 16, 64)
    # GNconv6: (8, 8, 128)
    # GNconv7: (4, 4, 256)
    # conv2d: (1, 1, 1024)

    def __init__(self):
        '''
            input shape = (256, 256, 3)
            kernel size = {4, 3, 4, 4, 4, 4, 4, 4}
            strides = {2, 1, 2, 2, 2, 2, 2, 1}
        '''
        super(Encoder, self).__init__()
        self.conv_block_1 = blocks.Conv2DBlock(32, input_shape=(256, 256, 3))
        self.conv_block_2 = blocks.Conv2DBlock(32, kernel_size=3, strides=1)
        self.conv_block_3 = blocks.Conv2DBlock(64)
        self.conv_block_4 = blocks.Conv2DBlock(32)
        self.conv_block_5 = blocks.Conv2DBlock(64)
        self.conv_block_6 = blocks.Conv2DBlock(128)
        self.conv_block_7 = blocks.Conv2DBlock(256)
        self.conv_block_8 = blocks.Conv2DBlock(1024, strides=1, padding='valid', use_norm=False, use_ReLU=False)

    def call(self, x):
        x = self.conv_block_1(x)
        assert tf.shape(x) == (None, 128, 128, 32)
        x = self.conv_block_2(x)
        assert tf.shape(x) == (None, 128, 128, 32)
        x = self.conv_block_3(x)
        assert tf.shape(x) == (None, 64, 64, 64)
        x = self.conv_block_4(x)
        assert tf.shape(x) == (None, 32, 32, 32)
        x = self.conv_block_5(x)
        assert tf.shape(x) == (None, 16, 16, 64)
        x = self.conv_block_6(x)
        assert tf.shape(x) == (None, 8, 8, 128)
        x = self.conv_block_7(x)
        assert tf.shape(x) == (None, 4, 4, 256)
        x = self.conv_block_8(x)
        assert tf.shape(x) == (None, 1, 1, 1024)

        return tf.keras.layers.Flatten()(x)



class HA_GAN(tf.keras.Model):
    def __init__(self):
        super(HA_GAN, self).__init__()
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.encoder = Generator()
        
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
        self.enc_loss_tracker = tf.keras.metrics.Mean(name="enc_loss")
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="gen_loss")