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
from  ThreeD_gan.threeD_gan_run import run_3D_VAE_GAN, load_and_show_3D_VAE_GAN, continue_from_checkpoint
import z_gan
import variables as var

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# run_3D_VAE_GAN("shapenet_tables", epochs=100)
run_3D_VAE_GAN("shapenet_tables2", epochs=100, user_lr_schedule=False)
# run_3D_VAE_GAN("shapenet_limited_tables", epochs=10)
# run_3D_VAE_GAN("shapenet_single_table", epochs=3, save_checkpoints = True)

# continue_from_checkpoint("./training_checkpoints/3D_GAN_shapenet_tables2_disc_lr-2e-05_gen_lr-0.0024_enc_lr-0.0005_a1-5_a2-0.0005_batch_size-64_latest_epoch",
#                          dataset_name="shapenet_tables2", epochs=17)

## lOAD MODEL WEIGHTS FROM CHECKPOINT AND SHOW THE RESULTS
# load_and_show_3D_VAE_GAN(
#     # checkpoint_path = "../backups/checkpoints/2023-05-07 epoch 99 d_lr-5e-5 batch-64/3D_GAN_shapenet_tables_disc_lr-5e-05_gen_lr-0.0025_enc_lr-0.001_a1-5_a2-0.0005_voxel_weight-1_batch_size-64"
#     checkpoint_path = "./training_checkpoints/3D_GAN_shapenet_tables2_disc_lr-2e-05_gen_lr-0.0021_enc_lr-0.001_a1-5_a2-0.0005_batch_size-64_best_train",
#     # shape_code = "7d3fc73ccd968863e40d907aaaf9adfd",
#     # shape_code = "8938cbf318d05cf44854921d37f7e048",
#     # shape_code = "582343e970abd505f155d75bbf62b80", # from test set
#     # shape_code = "4b22b93f9f881fe3434cc1450456532d", # from test set
#     # screenshot_number = 6,
# )


## RANDOM PIECES OF CODE USED FOR VARIOUS TESTING
# shape_code = "7807caccf26f7845e5cf802ea0702182"
# image = dp.load_single_image(f'./Data/ShapeNetSem/screenshots/{shape_code}/{shape_code}-6.png', 512)
# # dp.show_single_image(image)
# model = dp.get_shape(f"C:/old pc/Downloads/ShapeNetSem/tables-binvox-64/{shape_code}.binvox")
# dp.show_image_and_shape(image, model.data, model.dims)

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





