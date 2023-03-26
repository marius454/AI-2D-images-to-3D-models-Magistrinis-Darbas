import numpy as np
import tensorflow as tf
import dataIO as d
import custom_layers
import os
import time
import gc

from .. import variables as var
from .. import data_processing as dp

class Conv3DTransposeBlock(tf.keras.layers.Layer):
    def __init__(self, filters, input_shape = None, kernel_size = 4, strides = 2, padding = 'same', activation = None,
                 initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)):
        super(Conv3DTransposeBlock, self).__init__()
        self.conv_layer = tf.keras.layers.Conv3DTranspose(filters,
            use_bias=False,
            kernel_size=kernel_size,
            strides=strides,
            input_shape=input_shape,
            padding=padding,
            kernel_initializer=initializer,
            activation=activation)
        
        self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        self.ReLU_layer = tf.keras.layers.ReLU()

    def call(self, inputs, use_batch_norm=True, use_ReLU=True):
        x = self.conv_layer(inputs)
        if use_batch_norm:
            x = self.batch_norm_layer(x)
        if use_ReLU:
            x = self.ReLU_layer(x)
        return x


class Conv3DBlock(tf.keras.layers.Layer):
    def __init__(self, filters, input_shape = None, kernel_size = 4, strides = 2, padding = 'same', activation = None,
                 initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)):
        super(Conv3DBlock, self).__init__()
        self.conv_layer = tf.keras.layers.Conv3D(filters,
            use_bias=False,
            kernel_size=kernel_size,
            strides=strides,
            input_shape=input_shape,
            padding=padding,
            kernel_initializer=initializer,
            activation=activation)
        
        self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        self.ReLU_layer = tf.keras.layers.LeakyReLU(0.2)

    def call(self, inputs, use_batch_norm=True, use_ReLU=True):
        x = self.conv_layer(inputs)
        if use_batch_norm:
            x = self.batch_norm_layer(x)
        if use_ReLU:
            x = self.ReLU_layer(x)
        return x
    

class Conv2DBlock(tf.keras.layers.Layer):
    def __init__(self, filters, input_shape = None, kernel_size = 5, strides = 2, padding = 'same', activation = None,
                 initializer = 'glorot_uniform'):
        super(Conv2DBlock, self).__init__()
        self.conv_layer = tf.keras.layers.Conv2D(filters,
            use_bias=False,
            kernel_size=kernel_size,
            strides=strides,
            input_shape=input_shape,
            padding=padding,
            kernel_initializer=initializer,
            activation=activation)
        
        self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        self.ReLU_layer = tf.keras.layers.ReLU()

    def call(self, inputs, use_batch_norm=True, use_ReLU=True):
        x = self.conv_layer(inputs)
        if use_batch_norm:
            x = self.batch_norm_layer(x)
        if use_ReLU:
            x = self.ReLU_layer(x)
        return x
    


# def Conv3DTransposeBlock(self, input, input_shape, filter, strides, padding='same', use_batch_norm=True, use_ReLU=True):
#     x = tf.keras.layers.Conv3DTranspose(filter, kernel_size=(4, 4, 4), 
#         strides=strides, use_bias=False, input_shape=input_shape, padding=padding,
#         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(input)

#     if use_batch_norm:
#         x = tf.keras.layers.BatchNormalization()(x)
#     if use_ReLU:
#         x = tf.keras.layers.ReLU()(x)

#     return x