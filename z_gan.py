import numpy
import tensorflow as tf
import variables as var
import data_processing as dp
import dataIO as d
import custom_layers
import os

# pix2pix (Isola et al., 2017) (U-net) is used as the starting point for this model
# Many parts of the implementation are done consulting the tensorflow pix2pix guide - https://www.tensorflow.org/tutorials/generative/pix2pix
# the result of the generator should be a view centered voxel model
# "We hypothesize that the performance can be improved if the voxel model will be aligned with the input image."

def downsample(filters, kernel_size, batchNorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filers=filters, kernel_size=kernel_size, strides=2, padding="same",
        kernel_initializer=initializer))
    if (batchNorm):
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())

    return result

def make_generator_model():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(32, 4, batchNorm=False),
        downsample(64, 4),
        downsample(128, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
    ]