import tensorflow as tf
import numpy as np
# import os
# import sys
# import tensorflow_datasets as tfds
# import time
# import string
import matplotlib.pyplot as plt
import binvox_rw as bv

import data_processing as dp
import ThreeD_gan.threeD_gan as threeD_gan
from  ThreeD_gan.threeD_gan_run import run_3D_VAE_GAN, load_and_show_3D_VAE_GAN
import z_gan
# import variables as var


# prior_means = tf.fill([tf.shape(z_means)[0], z_size], 0.0)
# prior_vars = tf.fill([tf.shape(z_vars)[0], z_size], 1.0)

# tf.keras.losses.KLDivergence()([prior_means, prior_vars], [z_means, z_vars])

print(tf.keras.losses.KLDivergence()([0, 0, 0], [0.8, 0.8, 0.8]))
