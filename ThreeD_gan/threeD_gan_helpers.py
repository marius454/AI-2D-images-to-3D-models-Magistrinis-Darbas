import numpy as np
import tensorflow as tf
import dataIO as d
import backups.custom_layers as custom_layers
import os
import time
import gc

import ThreeD_gan.threeD_gan_options as opt


def get_z(z_means, z_vars, batch_size, z_size = opt.z_size):
    eps = tf.random.normal(shape = (batch_size, z_size))
    z = eps * tf.exp(z_vars * .5) + z_means
    z = tf.reshape(z, (batch_size, 1, 1, 1, z_size))
    return z