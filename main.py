import tensorflow as tf
import numpy as np
import os
import sys
import tensorflow_datasets as tfds
import time
import string
import matplotlib.pyplot as plt
import binvox_rw as bv

import data_processing as dp
import threeD_gan
import z_gan
import variables as var




# tf.keras.mixed_precision.set_global_policy('mixed_float16')

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

# dataset = dp.load_data()
# print (list(dataset.as_numpy_iterator()))

# shape_codes = dp.get_shape_code_list("./Data/ShapeNetSem/single_table.csv")
# shapes = dp.get_shapes(shape_codes, "./Data/ShapeNetSem/models-binvox/")
# # for code in shape_codes:
# #     shapes[code].data = 
# shape_data = np.array(shapes[shape_codes[0]].data)
# shape_data = np.where(shape_data, 1.0, 0.0).astype(np.float32)
# print (sys.getsizeof(shape_data))
# list1 = [shape_data, shape_data, shape_data]
# print (sys.getsizeof(list1))
# # list1 = [[1,2,3], [1,2,3], [1,2,3]]
# list2 = [1,2,3]
# dataset = tf.data.Dataset.from_tensor_slices((list2, list1))
# print (list(dataset.as_numpy_iterator()))



# shape_codes = dp.get_shape_code_list("./Data/ShapeNetSem/single_table.csv")
# shapes = dp.get_shapes(shape_codes, "./Data/ShapeNetSem/models-binvox-solid/")
# shape_screenshots = dp.get_shape_screenshots(shape_codes, "./Data/ShapeNetSem/screenshots/")

# with open("./Data/ShapeNetSem/models-binvox-custom/db80fbb9728e5df343f47bfd2fc426f7.binvox", 'rb') as f:
#     shape = bv.read_as_3d_array(f)

# shape = dp.downscale_binvox(shape)
# print (shape.dims)
# print (shape.data.shape)
# dp.plot_3d_model(shape.data, shape.dims)




generator = threeD_gan.make_generator_model()
discriminator = threeD_gan.make_discriminator_model()
encoder = threeD_gan.make_encoder_model()

# Declare optimizers
G_optimizer = tf.keras.optimizers.Adam(var.threeD_gan_generator_learning_rate, var.threeD_gan_adam_beta1, var.threeD_gan_adam_beta2)
D_optimizer = tf.keras.optimizers.Adam(var.threeD_gan_discriminator_learning_rate, var.threeD_gan_adam_beta1, var.threeD_gan_adam_beta2)
E_optimizer = tf.keras.optimizers.Adam(var.threeD_gan_encoder_learning_rate, var.threeD_gan_adam_beta1, var.threeD_gan_adam_beta2)

# declare training checkpoints
checkpoint_dir = './training_checkpoints/'
checkpoint = tf.train.Checkpoint(G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                E_optimizer=E_optimizer,
                                generator=generator,
                                discriminator=discriminator,
                                encoder=encoder)

# print(tf.train.latest_checkpoint(checkpoint_dir))
checkpoint.restore(checkpoint_dir + "ckpt-2")

image = dp.load_single_image('./Data/ShapeNetSem/screenshots/db80fbb9728e5df343f47bfd2fc426f7/db80fbb9728e5df343f47bfd2fc426f7-7.png')
image = tf.reshape(image, (-1, 256, 256, 3))
image = dp.normalize_image(image)

encoder_output = encoder(inputs=image, training=False)
z_mean, z_var = tf.split(encoder_output, num_or_size_splits=2, axis=1)
z = threeD_gan.get_z(z_mean, z_var, 1)

generated_shape = generator(z, training=False)
print(generated_shape[0, :, :, :, 0])
generated_shape = tf.where(generated_shape > 0.5, True, False).numpy()


dp.plot_3d_model(generated_shape[0, :, :, :, 0], (64, 64, 64))




# threeD_gan.train_3d_vae_gan(epochs=20)




## TODO list:
# In the paper, the final result is the largest connected component
# Might need to fix images to be loaded as floats in range [0; 1]
#
# Get a way to downscale the ShapeNet images
# Try to speed up training
# Calculate accuracy of discriminator and only update it if the accuracy is less than 80%