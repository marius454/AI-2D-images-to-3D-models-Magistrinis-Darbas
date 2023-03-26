import numpy as np
import tensorflow as tf
import dataIO as d
import custom_layers
import os
import time
import gc


def get_z(z_means, z_vars, batch_size, z_size = 200):
    eps = tf.random.normal(shape = (batch_size, z_size))
    z = eps * tf.exp(z_vars * .5) + z_means
    z = tf.reshape(z, (batch_size, 1, 1, 1, z_size))
    return z