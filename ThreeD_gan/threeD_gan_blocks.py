import numpy as np
import tensorflow as tf
import dataIO as d
import backups.custom_layers as custom_layers
import os
import time
import gc


class Conv3DTransposeBlock(tf.keras.layers.Layer):
    def __init__(self, filters, input_shape = None, kernel_size = 4, strides = 2, padding = 'same', activation = None,
                 initializer = None, use_bias = False):
        super(Conv3DTransposeBlock, self).__init__()
        if initializer == None:
            initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
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
                 initializer = None, use_bias = False):
        super(Conv3DBlock, self).__init__()
        if initializer == None:
            initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
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

    def call(self, inputs, use_batch_norm=True, use_ReLU=True):
        x = self.conv_layer(inputs)
        if use_batch_norm:
            x = self.batch_norm_layer(x)
        if use_ReLU:
            x = self.ReLU_layer(x)
        return x
    

class Conv2DBlock(tf.keras.layers.Layer):
    def __init__(self, filters, input_shape = None, kernel_size = 5, strides = 2, padding = 'same', activation = None,
                 initializer = 'glorot_uniform', activity_regularizer = None, use_bias = False):
        super(Conv2DBlock, self).__init__()
        params = {
            "filters": filters,
            "use_bias": use_bias,
            "kernel_size": kernel_size,
            "strides": strides,
            "input_shape": input_shape,
            "padding": padding,
            "kernel_initializer": initializer,
            "activation": activation,
            "activity_regularizer": activity_regularizer,
        }
        params = {key:value for key, value in params.items() if value is not None}

        self.conv_layer = tf.keras.layers.Conv2D(**params)
        self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        self.ReLU_layer = tf.keras.layers.ReLU()

    def call(self, inputs, use_batch_norm=True, use_ReLU=True):
        x = self.conv_layer(inputs)
        if use_batch_norm:
            x = self.batch_norm_layer(x)
        if use_ReLU:
            x = self.ReLU_layer(x)
        return x
    