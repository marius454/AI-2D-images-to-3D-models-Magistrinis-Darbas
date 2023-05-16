import numpy as np
import tensorflow as tf
import dataIO as d
import math
import os
import time
import gc

import data_processing as dp
from ThreeD_gan.callbacks import DisplayCallback, LR_callback

from ThreeD_gan import threeD_gan
import ThreeD_gan.threeD_gan_options as opt

def run_3D_VAE_GAN(dataset_name, batch_size = opt.batch_size, epochs = opt.epochs, save_checkpoints = True, user_lr_schedule = False):
    ## Load a dataset as tf.Dataset
    print('Loading data')
    dataset = dp.load_data(dataset_name, shapes_dir=opt.shape_data_dir, image_res=opt.image_res)
    train_data = dataset.take(math.trunc(len(dataset) * opt.training_split))
    test_data = dataset.skip(math.trunc(len(dataset) * opt.training_split))
    train_data = (
        train_data
        .shuffle(len(train_data))
        .batch(batch_size)
    )
    test_data = (
        test_data.batch(batch_size)
    )

    print("\nGenerating 3D-VAE-GAN model")
    model = threeD_gan.ThreeD_gan()

    print("\nCompiling 3D-VAE-GAN model")
    model.compile(
        g_optimizer = tf.keras.optimizers.Adam(opt.generator_lr[0], opt.adam_beta),
        d_optimizer = tf.keras.optimizers.Adam(opt.discriminator_lr[0], opt.adam_beta),
        e_optimizer = tf.keras.optimizers.Adam(opt.encoder_lr[0]),
    )
    model.run_eagerly = opt.use_eager_mode
    callbacks = []
    if (user_lr_schedule):
        callbacks.append(LR_callback())
    if (save_checkpoints):
        save_best_val_callback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/3D_GAN_{dataset_name}_disc_lr-{opt.discriminator_lr[0]}_gen_lr-{opt.generator_lr[0]}_enc_lr-{opt.encoder_lr[0]}_a1-{opt.alpha_1}_a2-{opt.alpha_2}_batch_size-{batch_size}_best_val",
                                                      save_weights_only = True,
                                                      save_best_only = True,
                                                      monitor = "val_overall_loss",
                                                      mode = "min",
                                                      # save_freq=5*var.STEPS_PER_EPOCH + 1,
                                                      verbose=1)
        save_best_train_callback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/3D_GAN_{dataset_name}_disc_lr-{opt.discriminator_lr[0]}_gen_lr-{opt.generator_lr[0]}_enc_lr-{opt.encoder_lr[0]}_a1-{opt.alpha_1}_a2-{opt.alpha_2}_batch_size-{batch_size}_best_train",
                                                      save_weights_only = True,
                                                      save_best_only = True,
                                                      monitor = "overall_loss",
                                                      mode = "min",
                                                      # save_freq=5*var.STEPS_PER_EPOCH + 1,
                                                      verbose=1)
        save_last_callback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/3D_GAN_{dataset_name}_disc_lr-{opt.discriminator_lr[0]}_gen_lr-{opt.generator_lr[0]}_enc_lr-{opt.encoder_lr[0]}_a1-{opt.alpha_1}_a2-{opt.alpha_2}_batch_size-{batch_size}_latest_epoch",
                                                      save_weights_only = True,
                                                      save_freq='epoch',
                                                      # save_freq=5*var.STEPS_PER_EPOCH + 1,
                                                      verbose=0)
        callbacks.append(save_best_val_callback)
        callbacks.append(save_best_train_callback)
        callbacks.append(save_last_callback)

    callbacks.append(DisplayCallback())

    print("\nStarting 3D-VAE-GAN fit")
    model.fit(
        train_data,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = test_data,
        callbacks = callbacks,
    )
    
def load_and_show_3D_VAE_GAN(checkpoint_path, real_shape_dir = opt.shape_data_dir, shape_code = "db80fbb9728e5df343f47bfd2fc426f7", screenshot_number = 7):
    model = threeD_gan.ThreeD_gan()
    # NOT SURE IF THIS IS NECESSARY
    # model.compile(
    #     g_optimizer = tf.keras.optimizers.Adam(opt.generator_lr[0], opt.adam_beta),
    #     d_optimizer = tf.keras.optimizers.Adam(opt.discriminator_lr[0], opt.adam_beta),
    #     e_optimizer = tf.keras.optimizers.Adam(opt.encoder_lr[0]),
    # )
    model.load_weights(checkpoint_path).expect_partial()

    print("Loading image")
    image = dp.load_single_image(f'./Data/ShapeNetSem/table_screenshots/{shape_code}/{shape_code}-{screenshot_number}.png', image_res = opt.image_res)
    image = dp.normalize_image(image)
    image = tf.reshape(image, (-1, opt.image_res, opt.image_res, 3))

    print("Loading real 3D model")
    real_shape = dp.get_shape(f"{real_shape_dir}/{shape_code}.binvox")

    print("Generating 3D model")
    generated_shape = model.predict(image)
    generated_shape = tf.math.greater_equal(generated_shape, 0.5)
    # print(generated_shape[0, :, :, :, 0].numpy())
    
    print("Ploting image and models")
    image = tf.reshape(image, (opt.image_res, opt.image_res, 3))
    dp.show_image_and_shapes(image, real_shape.data, generated_shape[0, :, :, :, 0].numpy(), real_shape.dims)


def continue_from_checkpoint(checkpoint_path, dataset_name, batch_size = opt.batch_size, epochs = opt.epochs, save_checkpoints = True):
    print('Loading data')
    dataset = dp.load_data(dataset_name, shapes_dir=opt.shape_data_dir, image_res=opt.image_res)
    train_data = dataset.take(math.trunc(len(dataset) * opt.training_split))
    test_data = dataset.skip(math.trunc(len(dataset) * opt.training_split))
    train_data = (
        train_data
        .shuffle(len(train_data))
        .batch(batch_size)
    )
    test_data = (
        test_data.batch(batch_size)
    )

    print("\nGenerating 3D-VAE-GAN model")
    model = threeD_gan.ThreeD_gan()

    print("\nCompiling 3D-VAE-GAN model")
    model.compile(
        g_optimizer = tf.keras.optimizers.Adam(opt.generator_lr[0], opt.adam_beta),
        d_optimizer = tf.keras.optimizers.Adam(opt.discriminator_lr[0], opt.adam_beta),
        e_optimizer = tf.keras.optimizers.Adam(opt.encoder_lr[0]),
    )
    model.run_eagerly = opt.use_eager_mode

    print('loading checkpoint weights')
    model.load_weights(checkpoint_path).expect_partial()

    callbacks = []
    if (save_checkpoints):
        save_best_val_callback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/3D_GAN_{dataset_name}_disc_lr-{opt.discriminator_lr[0]}_gen_lr-{opt.generator_lr[0]}_enc_lr-{opt.encoder_lr[0]}_a1-{opt.alpha_1}_a2-{opt.alpha_2}_batch_size-{batch_size}_best_val",
                                                      save_weights_only = True,
                                                      save_best_only = True,
                                                      monitor = "val_overall_loss",
                                                      mode = "min",
                                                      verbose=1)
        save_best_train_callback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/3D_GAN_{dataset_name}_disc_lr-{opt.discriminator_lr[0]}_gen_lr-{opt.generator_lr[0]}_enc_lr-{opt.encoder_lr[0]}_a1-{opt.alpha_1}_a2-{opt.alpha_2}_batch_size-{batch_size}_best_train",
                                                      save_weights_only = True,
                                                      save_best_only = True,
                                                      monitor = "overall_loss",
                                                      mode = "min",
                                                      verbose=1)
        save_last_callback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/3D_GAN_{dataset_name}_disc_lr-{opt.discriminator_lr[0]}_gen_lr-{opt.generator_lr[0]}_enc_lr-{opt.encoder_lr[0]}_a1-{opt.alpha_1}_a2-{opt.alpha_2}_batch_size-{batch_size}_latest_epoch",
                                                      save_weights_only = True,
                                                      save_freq='epoch',
                                                      verbose=0)
        callbacks = [save_best_val_callback, save_best_train_callback, save_last_callback, DisplayCallback()]
    else:
        callbacks = [DisplayCallback()]

    print("\nStarting 3D-VAE-GAN fit")
    model.fit(
        train_data,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = test_data,
        callbacks = callbacks,
    )