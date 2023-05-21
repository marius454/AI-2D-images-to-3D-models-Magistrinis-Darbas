import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import binvox_rw as bv
import torch

import data_processing as dp
import ThreeD_gan.threeD_gan as threeD_gan
import ThreeD_gan.threeD_gan_run as threeD_gan_run

import Ha_gan.ha_gan as ha_gan
import Ha_gan.ha_gan_run as ha_gan_run

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

## Train 3D-VAE-GAN model
# threeD_gan_run.run_3D_VAE_GAN("shapenet_tables", epochs=100)
# threeD_gan_run.run_3D_VAE_GAN("shapenet_tables2", epochs=100, use_lr_schedule=False)
# threeD_gan_run.run_3D_VAE_GAN("shapenet_limited_tables", epochs=10)
# threeD_gan_run.run_3D_VAE_GAN("shapenet_single_table", epochs=3, save_checkpoints = True)

## Train 3D-VAE-GAN model continuing from checkpoint
# threeD_gan_run.continue_from_checkpoint("./training_checkpoints/3D_GAN_shapenet_tables2_disc_lr-2e-05_gen_lr-0.0021_enc_lr-0.001_a1-5_a2-0.0005_batch_size-64_latest_epoch",
#                          dataset_name="shapenet_tables2", epochs=50)

## Load 3D-VAE-GAN weights from checkpoint and show the result
# threeD_gan_run.load_and_show_3D_VAE_GAN(
#     # checkpoint_path = "../backups/checkpoints/2023-05-07 epoch 99 d_lr-5e-5 batch-64/3D_GAN_shapenet_tables_disc_lr-5e-05_gen_lr-0.0025_enc_lr-0.001_a1-5_a2-0.0005_voxel_weight-1_batch_size-64",
#     checkpoint_path = "C:/Users/mariu/Desktop/epoch 100 disc_lr-3.3 gen_lr 0.0025/3D_GAN_shapenet_tables2_disc_lr-3.3e-05_gen_lr-0.0025_enc_lr-0.001_a1-5_a2-0.0005_batch_size-64_latest_epoch",
#     # checkpoint_path = "C:/Users/mariu/Desktop/epoch 100 disc_lr-3.3 gen_lr 0.0025/epoch 150/3D_GAN_shapenet_tables2_disc_lr-3.3e-05_gen_lr-0.0025_enc_lr-0.001_a1-5_a2-0.0005_batch_size-64_latest_epoch",
#     # checkpoint_path = "C:/Users/mariu/Desktop/epoch 100 disc_lr-2e-05_gen_lr-0.0021_enc_lr-0.001_a1-5_a2-0.0005/3D_GAN_shapenet_tables2_disc_lr-2e-05_gen_lr-0.0021_enc_lr-0.001_a1-5_a2-0.0005_batch_size-64_latest_epoch",
#     # checkpoint_path = "./training_checkpoints/3D_GAN_shapenet_tables2_disc_lr-2e-05_gen_lr-0.0021_enc_lr-0.001_a1-5_a2-0.0005_batch_size-64_latest_epoch",
#     # shape_code = "7d3fc73ccd968863e40d907aaaf9adfd",
#     shape_code = "8938cbf318d05cf44854921d37f7e048",
#     # shape_code = "582343e970abd505f155d75bbf62b80", # from test set
#     # shape_code = "4b22b93f9f881fe3434cc1450456532d", # from test set
#     # shape_code = "c0470c413b0a260979368d1198f406e7", # from test set
#     # screenshot_number = 6,
#     threshold=0.1
# )

## Train HA-GAN model
# ha_gan_run.run_HA_GAN("shapenet_tables", epochs=100)
ha_gan_run.run_HA_GAN("shapenet_tables2", epochs=500, use_test_data=False)
# ha_gan_run.run_HA_GAN("shapenet_limited_tables", epochs=5)
# ha_gan_run.run_HA_GAN("shapenet_five_tables", epochs=4, use_test_data=False)

## Continue HA-GAN training from checkpoint
# ha_gan_run.continue_from_checkpoint(checkpoint_path = "./training_checkpoints/HA-GAN_shapenet_five_tables_disc_lr-0.0001_gen_lr-0.0001_enc_lr-0.0004_lambda-10_batch_size-4_epoch-04",
#     dataset_name="shapenet_five_tables", epochs = 6, initial_epoch=4, use_test_data=False)
# ha_gan_run.continue_from_checkpoint(checkpoint_path = "./training_checkpoints/HA-GAN_shapenet_tables2_disc_lr-0.0001_gen_lr-0.0001_enc_lr-0.0004_lambda-10_batch_size-4_latest_epoch",
#     dataset_name="shapenet_tables2", epochs = 200, initial_epoch=20, use_test_data=False)

## Creating Tensorflow datasets:
# ha_gan_run.save_HA_GAN_dataset("shapenet_tables")

## Load 3D-VAE-GAN weights from checkpoint and show the result
# ha_gan_run.load_and_show_HA_GAN(
#     checkpoint_path = "./training_checkpoints/HA-GAN_shapenet_tables2_disc_lr-0.0001_gen_lr-0.0002_enc_lr-0.0004_lambda-10_batch_size-4_e_iter-3_g_iter2_epoch-020",
#     # shape_code = "7d3fc73ccd968863e40d907aaaf9adfd",
#     # shape_code = "8938cbf318d05cf44854921d37f7e048",
#     # shape_code = "582343e970abd505f155d75bbf62b80", # from test set
#     # shape_code = "4b22b93f9f881fe3434cc1450456532d", # from test set
#     # shape_code = "c0470c413b0a260979368d1198f406e7", # from test set
#     # screenshot_number = 6,
#     threshold = -0.5
# )

## TEST VOXEL TO MESH TRANSFORMATION, SMOOTHING AND MESH PLOTING
# shape_code = "7807caccf26f7845e5cf802ea0702182"
# # shape_code = "db80fbb9728e5df343f47bfd2fc426f7"
# # shape_code = "8938cbf318d05cf44854921d37f7e048"
# shape = dp.get_shape(f"C:/old pc/Downloads/ShapeNetSem/tables-binvox-64/{shape_code}.binvox")
# vertices, triangles = dp.voxel_to_mesh(shape.data, use_smoothing = False)
# dp.plot_3d_mesh(vertices, triangles)





## Testing the differences between pytorch L1loss and tf.keras.losses.MeanAbsoluteError()
# y_true = tf.Variable([[-1., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]])
# y_pred = tf.Variable([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])

# mae = tf.keras.losses.MeanAbsoluteError()
# print (mae(y_true, y_pred))
# print (tf.keras.losses.mae(y_true, y_pred))
# print (tf.abs(y_pred - y_true))
# print (tf.reduce_mean(tf.abs(y_pred - y_true), 1))
# print (tf.reduce_mean(tf.reduce_sum(tf.abs(y_pred - y_true), 1)))

# y_true = torch.tensor([[0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]])
# y_pred = torch.tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
# print (torch.nn.L1Loss()(y_pred, y_true))




## PLOT MODEL LAYERS TO PNG
# model = threeD_gan.Generator()
# # model = threeD_gan.Discriminator()
# # model = threeD_gan.Encoder()
# # model.build_graph().summary()
# tf.keras.utils.plot_model(model.build_graph(),
#                            show_shapes=True,
#                            to_file="Generator.png")


## RANDOM PIECES OF CODE USED FOR VARIOUS TESTING
# shape_code = "7807caccf26f7845e5cf802ea0702182"
# image = dp.load_single_image(f'./Data/ShapeNetSem/screenshots/{shape_code}/{shape_code}-6.png', 512)
# # dp.show_single_image(image)
# model = dp.get_shape(f"C:/old pc/Downloads/ShapeNetSem/tables-binvox-64/{shape_code}.binvox")
# dp.show_image_and_shape(image, model.data, model.dims)

# shape = dp.get_shape('./Data/ShapeNetSem/tables-binvox-64/db80fbb9728e5df343f47bfd2fc426f7.binvox')
# dp.plot_3d_model(shape.data, shape.dims)

# shape = dp.get_shape('./Data/ShapeNetSem/tables-binvox-128/1a10ecfcaac04c882d17d82c03b66.binvox', 2)
# dp.plot_3d_model(shape.data, shape.dims)



## TODO list:
# Search "TODO" in all files, for other entries.
# 
# In the paper, the final result is the largest connected component
# + Might need to fix images to be loaded as floats in range [0; 1]
# + Get a way to downscale the ShapeNet images
# :( Try to speed up training
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





