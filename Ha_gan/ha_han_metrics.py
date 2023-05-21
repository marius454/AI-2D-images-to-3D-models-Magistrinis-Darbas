import numpy as np
import tensorflow as tf
import dataIO as d
import backups.custom_layers as custom_layers
import os
import time
import gc

import Ha_gan.ha_gan_options as opt

def discriminator_accuracy(predictions_real, predictions_fake):
    real_acu = tf.cast(tf.math.greater_equal(predictions_real[:,0], 0.5), tf.float16)
    fake_acu = tf.cast(tf.math.less(predictions_fake[:,0], 0.5), tf.float16)
    total_acu = tf.math.reduce_mean(tf.concat([real_acu, fake_acu], 0))
    return total_acu

def generator_accuracy(generated_output):
    predictions = tf.cast(tf.math.greater_equal(generated_output, 0.5), tf.float16)
    accuracy = tf.math.reduce_mean(predictions)
    return accuracy

def voxel_accuracy(real_shapes, generated_shapes, threshold = 0):
    real_binary = tf.cast(tf.math.greater_equal(real_shapes, threshold), tf.float16)
    generated_binary = tf.cast(tf.math.greater_equal(generated_shapes, threshold), tf.float16)
    mae = tf.keras.losses.mean_absolute_error(real_binary, generated_binary)

    return 100 * (1-mae)
