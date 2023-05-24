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

def reconstruction_loss(real_shapes, generated_shapes):
    """
    Calculate the Euclidan distance ||G(E(y)) − x||2\n
    Sum over batch (mean) reduction is applied
    """
    # USE SQARED EUCLIDEAN DISTACE
    euclidean_distance_per_batch = tf.reduce_sum(tf.math.squared_difference(
            tf.reshape(generated_shapes, (-1, 64 * 64 * 64)), 
            tf.reshape(real_shapes, (-1, 64 * 64 * 64)),
        ), 1)
    return tf.reduce_mean(euclidean_distance_per_batch)
