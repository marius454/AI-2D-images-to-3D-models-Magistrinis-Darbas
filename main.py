import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
import time
import string
import matplotlib.pyplot as plt

import data_processing as dp
import threeD_gan
import z_gan
import variables as var

## TODO list:
# In the paper, the final result is the largest connected component
# Reference the Shapenet database in my report 
# Might need to fix images to be loaded as floats in range [0; 1]

tf.keras.mixed_precision.set_global_policy('mixed_float16')

# image = dp.load_single_image('chair.png')
# dp.show_single_image(image)


# generator = threeD_gan.make_generator_model()
# z = tf.random.uniform((1, 200), minval=0, maxval=1)
# z = tf.reshape(z, (-1, 1, 1, 1, var.threeD_gan_z_size))
# generated_3d_model = generator(z, training=False)
# generated_3d_model = tf.where(generated_3d_model > 0.5, True, False).numpy()
# # print (generated_3d_model[0, :, :, :, 0])

# discriminator = threeD_gan.make_discriminator_model()
# decision = discriminator(generated_3d_model)
# print (decision)

# encoder = threeD_gan.make_encoder_model()

# directory = "./Ikea Data/model/"
# data = dp.load_directory_mat(directory)
# print(data)

# data = dp.load_file_mat("./Ikea Data/model/IKEA_bed_BRIMNES_4b534bead1e06a7f9ef2df9927efa75_obj0_object.mat")
# print(data)
# dp.plot_3d_model(data)

# generator = z_gan.make_generator_model()
# print(generator)


shape_codes = dp.get_shape_code_list("./Data/ShapeNetSem/Table.csv")
shapes = dp.get_shapes(shape_codes, "./Data/ShapeNetSem/models-binvox/")
shape_screenshots = dp.get_shape_screenshots(shape_codes, "./Data/ShapeNetSem/screenshots/")


# shape_keys = list(shapes.keys())
# shape_screenshot_keys = list(shape_screenshots.keys())
# last_shape_screenshots = shape_screenshots[shape_screenshot_keys[-1]]

# print (shape_keys[-1])
# last_shape = shapes[shape_keys[-1]]
# print (last_shape)
# print (last_shape.dims)
# print (last_shape.scale)
# print (last_shape.translate)
# print (last_shape.data)
# dp.plot_3d_model(last_shape.data, last_shape.dims)

# print (shape_screenshot_keys[-1])
# print (len(last_shape_screenshots))
# print (last_shape_screenshots[0])
# dp.show_single_image(last_shape_screenshots[0])
# dp.show_single_image(last_shape_screenshots[1])
# dp.show_single_image(last_shape_screenshots[2])
# dp.show_single_image(last_shape_screenshots[3])
# dp.show_single_image(last_shape_screenshots[4])
# dp.show_single_image(last_shape_screenshots[5])
