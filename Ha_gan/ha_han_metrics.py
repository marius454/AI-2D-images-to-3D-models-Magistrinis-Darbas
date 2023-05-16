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