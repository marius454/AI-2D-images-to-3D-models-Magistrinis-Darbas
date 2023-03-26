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
    def __init__(self, input_shape, filter, strides, padding='same', use_batch_norm=True, use_ReLU=True):
        super(Conv3DTransposeBlock, self).__init__()
        self.conv_layer = tf.keras.layers.Conv3DTranspose(filter, kernel_size=(4, 4, 4), 
            strides=strides, use_bias=False, input_shape=input_shape, padding=padding,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        if use_batch_norm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        if use_ReLU:
            self.ReLU_layer = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv_layer(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)