import numpy as np
import tensorflow as tf
import variables as var
import data_processing as dp
import dataIO as d
import custom_layers
import os
import time
import gc

class Conv3DTransposeBlock(tf.keras.layers.Layer):
    def __init__(self, input_shape, filter, strides, padding='same'):
        super(Conv3DTransposeBlock, self).__init__()
        self.conv_layer = tf.keras.layers.Conv3DTranspose(filter, kernel_size=(4, 4, 4), 
            strides=strides, use_bias=False, input_shape=input_shape, padding=padding,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        self.ReLU_layer = tf.keras.layers.ReLU()

    def call(self, input, use_batch_norm=True, use_ReLU=True):
        x = self.conv_layer(input)
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