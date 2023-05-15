import numpy as np
import tensorflow as tf
import dataIO as d
import custom_layers
import os
import time
import gc


class DownsampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, input_shape = None, kernel_size = 4, strides = 2, padding = 'same', activation = None,
                 initializer = tf.random_normal_initializer(0., 0.02), use_bias = False,
                 use_batch_norm=True, use_ReLU=True):
        super(DownsampleBlock, self).__init__()
        params = {
            "filters": filters,
            "use_bias": use_bias,
            "kernel_size": kernel_size,
            "strides": strides,
            "input_shape": input_shape,
            "padding": padding,
            "kernel_initializer": initializer,
            "activation": activation,
        }
        params = {key:value for key, value in params.items() if value is not None}

        self.conv_layer = tf.keras.layers.Conv2D(**params)
        self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        self.ReLU_layer = tf.keras.layers.LeakyReLU(0.2)

        self.use_batch_norm = use_batch_norm
        self.use_ReLU = use_ReLU

    def call(self, inputs):
        x = self.conv_layer(inputs)
        if self.use_batch_norm:
            x = self.batch_norm_layer(x)
        if self.use_ReLU:
            x = self.ReLU_layer(x)
        return x
    

class UpsampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, input_shape = None, kernel_size = 4, strides = 2, padding = 'same', activation = None,
                 initializer = tf.random_normal_initializer(0., 0.02), use_bias = False, 
                 use_batch_norm=True, use_ReLU=True, use_dropout=False):
        super(UpsampleBlock, self).__init__()
        params = {
            "filters": filters,
            "use_bias": use_bias,
            "kernel_size": kernel_size,
            "strides": strides,
            "input_shape": input_shape,
            "padding": padding,
            "kernel_initializer": initializer,
            "activation": activation,
        }
        params = {key:value for key, value in params.items() if value is not None}

        self.conv_layer = tf.keras.layers.Conv3DTranspose(**params)
        self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        self.dropout_layer = tf.keras.layers.Dropout(0.5)
        self.ReLU_layer = tf.keras.layers.ReLU()

        self.use_batch_norm = use_batch_norm
        self.use_ReLU = use_ReLU
        self.use_dropout = use_dropout

    def call(self, inputs):
        x = self.conv_layer(inputs)
        if self.use_batch_norm:
            x = self.batch_norm_layer(x)
        if self.use_dropout:
            x = self.dropout_layer(x)
        if self.use_ReLU:
            x = self.ReLU_layer(x)
        return x
    
class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, filters, input_shape = None, kernel_size = 4, strides = 2, padding = 'same', activation = None,
                 initializer = tf.random_normal_initializer(0., 0.02), use_bias = False,
                 use_batch_norm=True, use_ReLU=True):
        super(DiscriminatorBlock, self).__init__()
        params = {
            "filters": filters,
            "use_bias": use_bias,
            "kernel_size": kernel_size,
            "strides": strides,
            "input_shape": input_shape,
            "padding": padding,
            "kernel_initializer": initializer,
            "activation": activation,
        }
        params = {key:value for key, value in params.items() if value is not None}

        self.conv_layer = tf.keras.layers.Conv3D(**params)
        self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        self.ReLU_layer = tf.keras.layers.LeakyReLU(0.2)

        self.use_batch_norm = use_batch_norm
        self.use_ReLU = use_ReLU

    def call(self, inputs):
        x = self.conv_layer(inputs)
        if self.use_batch_norm:
            x = self.batch_norm_layer(x)
        if self.use_ReLU:
            x = self.ReLU_layer(x)
        return x