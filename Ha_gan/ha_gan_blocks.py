import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import dataIO as d
import backups.custom_layers as custom_layers
import os
import time


# 3D conv block with group normalization used for generator
class GNConv3DBlock(tf.keras.layers.Layer):
    def __init__(self, filters, norm_groups = 8, input_shape = None, kernel_size = 3, strides = 1, padding = 'same', activation = None,
                 initializer = None, use_bias = True, use_norm=True, use_ReLU=True, use_interpolation=True, name=None):
        super(GNConv3DBlock, self).__init__()
        params = {
            "filters": filters, #filters == out_channels in pytorch
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
        if (name):
            self._name = name

        self.layers = []
        if (use_norm):
            self.layers.append(tfa.layers.GroupNormalization(norm_groups))
        if (use_ReLU):
            self.layers.append(tf.keras.layers.ReLU())
        if (use_interpolation):
            self.layers.append(tf.keras.layers.UpSampling3D(2))

    def call(self, inputs):
        x = self.conv_layer(inputs)
        for layer in self.layers:
            x = layer(x)

        return x
    

# 3D conv block with spectral normalization used for discriminator
class SNConv3DBlock(tf.keras.layers.Layer):
    def __init__(self, filters, input_shape = None, kernel_size = 4, strides = 2, padding = 'same', activation = None,
                 initializer = None, use_bias = True, use_norm=True, use_ReLU=True, name=None):
        super(SNConv3DBlock, self).__init__()
        params = {
            "filters": filters, #filters == out_channels in pytorch
            "use_bias": use_bias,
            "kernel_size": kernel_size,
            "strides": strides,
            "input_shape": input_shape,
            "padding": padding,
            "kernel_initializer": initializer,
            "activation": activation,
        }
        params = {key:value for key, value in params.items() if value is not None}
        if (name):
            self._name = name

        if (use_norm):
             # Can't be sure if implementation of SpectralNormalization is correct
            self.conv_layer = tfa.layers.SpectralNormalization(tf.keras.layers.Conv3D(**params))
        else:
            self.conv_layer = tf.keras.layers.Conv3D(**params)

        self.layers = []
        if (use_ReLU):
            self.layers.append(tf.keras.layers.LeakyReLU(0.2))

    def call(self, inputs):
        x = self.conv_layer(inputs)
        for layer in self.layers:
            x = layer(x)

        return x
    
# 3D conv block with spectral normalization used for discriminator
class SNDenseBlock(tf.keras.layers.Layer):
    def __init__(self, units, activation = None, initializer = None, use_bias = True, use_norm=True, use_ReLU=True, 
                 name=None):
        super(SNDenseBlock, self).__init__()
        params = {
            "units": units,
            "activation": activation,
        }
        params = {key:value for key, value in params.items() if value is not None}
        self.dense_layer = tfa.layers.SpectralNormalization(tf.keras.layers.Dense(**params))
        if (name):
            self._name = name

        self.layers = []
        if (use_ReLU):
            self.layers.append(tf.keras.layers.LeakyReLU(0.2))

    def call(self, inputs):
        x = self.dense_layer(inputs)
        for layer in self.layers:
            x = layer(x)

        return x
    

class Conv2DBlock(tf.keras.layers.Layer):
    pass
    def __init__(self, filters, norm_groups = 8, input_shape = None, kernel_size = 4, strides = 2, padding = 'same', activation = None,
                 initializer = None, use_bias = False, use_norm=True, use_ReLU=True, name=None):
        super(Conv2DBlock, self).__init__()
        params = {
            "filters": filters, #filters == out_channels in pytorch
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
        if (name):
            self._name = name

        self.layers = []
        if (use_norm):
            self.layers.append(tfa.layers.GroupNormalization(norm_groups))
        if (use_ReLU):
            self.layers.append(tf.keras.layers.ReLU())

    def call(self, inputs):
        x = self.conv_layer(inputs)
        for layer in self.layers:
            x = layer(x)

        return x
