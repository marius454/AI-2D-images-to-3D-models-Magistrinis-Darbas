import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
import time
import string
import matplotlib.pyplot as plt

import data_processing as dp
import threeD_gan
import variables as var

# image = dp.load_single_image('chair.png')
# dp.show_single_image(image)

# print (threeD_gan.initialiseWeights())

# generator = threeD_gan.make_generator_model()


# z = tf.random.uniform((1, 200), minval=0, maxval=1)
# z = tf.reshape(z, (-1, 1, 1, 1, var.threeD_gan_z_size))
# generated_3d_model = generator(z, training=False)
# generated_3d_model = tf.where(generated_3d_model > 0.5, True, False).numpy()

# # print (generated_3d_model[0, :, :, :, 0])

# discriminator = threeD_gan.make_discriminator_model()
# decision = discriminator(generated_3d_model)
# print (decision)

encoder = threeD_gan.make_encoder_model()

