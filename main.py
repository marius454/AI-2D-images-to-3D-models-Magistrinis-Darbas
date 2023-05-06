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
import variables as var

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# run_3D_VAE_GAN("shapenet_tables", epochs=100)
run_3D_VAE_GAN("shapenet_tables2", epochs=100)
# run_3D_VAE_GAN("shapenet_limited_tables", epochs=10)
# run_3D_VAE_GAN("shapenet_single_table", epochs=3)

# load_and_show_3D_VAE_GAN("./training_checkpoints/3D_GAN_shapenet_tables_disc_lr-0.0001_gen_lr-0.0025_a1-5_a2-1e-1")




# shape = dp.get_shape('./Data/ShapeNetSem/models-binvox-custom/db80fbb9728e5df343f47bfd2fc426f7.binvox', 2)
# dp.plot_3d_model(shape.data, shape.dims)

# shape = dp.get_shape('./Data/ShapeNetSem/models-binvox-custom/1a10ecfcaac04c882d17d82c03b66.binvox', 2)
# dp.plot_3d_model(shape.data, shape.dims)



## TODO list:
# Search "TODO" in all files, for other entries.
# 
# In the paper, the final result is the largest connected component
# Might need to fix images to be loaded as floats in range [0; 1]
# + Get a way to downscale the ShapeNet images
# Try to speed up training
# + Calculate accuracy of discriminator and only update it if the accuracy is less than 80%




## Load and how single image
# image = dp.load_single_image('chair.png')
# dp.show_single_image(image)

## Example code for working with shapes
# shape_codes = get_shape_code_list("./Data/ShapeNetSem/single_table.csv")
# shapes = get_shapes(shape_codes, "./Data/ShapeNetSem/models-binvox/")
# shape_screenshots = get_shape_screenshots(shape_codes, "./Data/ShapeNetSem/screenshots/")
# print (shapes[code])
# print (shapes[code].dims)
# print (shapes[code].scale)
# print (shapes[code].translate)
# print (shapes[code].data)
# dp.plot_3d_model(shapes[code].data, shapes[code].dims)
# dp.show_single_image(shape_screenshots[code][0])
# dp.show_single_image(shape_screenshots[code][1])
# dp.show_single_image(shape_screenshots[code][2])
# dp.show_single_image(shape_screenshots[code][3])

## Example code of downscaling model
# shape = dp.downscale_binvox(shape)
# print (shape.dims)
# print (shape.data.shape)
# dp.plot_3d_model(shape.data, shape.dims)


# Example code for declaring training checkpoints
# checkpoint_dir = './training_checkpoints/'
# checkpoint = tf.train.Checkpoint(G_optimizer=G_optimizer,
#                                 D_optimizer=D_optimizer,
#                                 E_optimizer=E_optimizer,
#                                 generator=generator,
#                                 discriminator=discriminator,
#                                 encoder=encoder)
# print(tf.train.latest_checkpoint(checkpoint_dir))
# checkpoint.restore(checkpoint_dir + "ckpt-2")


## Load single image to test model
# image = dp.load_single_image('./Data/ShapeNetSem/screenshots/db80fbb9728e5df343f47bfd2fc426f7/db80fbb9728e5df343f47bfd2fc426f7-7.png')
# image = tf.reshape(image, (-1, 256, 256, 3))
# image = dp.normalize_image(image)
# TODO use model here
# dp.plot_3d_model(generated_shape[0, :, :, :, 0], (64, 64, 64))





