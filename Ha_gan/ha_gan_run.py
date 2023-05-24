import numpy as np
import tensorflow as tf
import dataIO as d
import math
import os
import time
import gc

import data_processing as dp
from Ha_gan.ha_gan_callbacks import DisplayCallback

import Ha_gan.ha_gan as ha_gan
import Ha_gan.ha_gan_options as opt

def save_HA_GAN_dataset(dataset_name):
    print('Loading data')
    dataset = dp.load_shapenet_data_groups(dataset_name, shapes64_dir=opt.shapes64_dir, shapes256_dir=opt.shapes256_dir, image_res=opt.image_res)
    train_data = dataset.take(math.trunc(len(dataset) * opt.training_split))
    test_data = dataset.skip(math.trunc(len(dataset) * opt.training_split))

    train_data.save(f"./Data/Custom_Datasets/ha_gan_{dataset_name}_train")
    test_data.save(f"./Data/Custom_Datasets/ha_gan_{dataset_name}_test")
    
    
def run_HA_GAN(dataset_name, batch_size = opt.batch_size, epochs = opt.epochs, save_checkpoints = True, use_test_data = True):
    ## Load a dataset as tf.Dataset
    print("\nGenerating HA-GAN model")
    model = ha_gan.HA_GAN()

    print("\nCompiling HA-GAN model")
    model.compile(
        g_optimizer = tf.keras.optimizers.Adam(opt.generator_lr, opt.adam_beta1, opt.adam_beta2),
        d_optimizer = tf.keras.optimizers.Adam(opt.discriminator_lr, opt.adam_beta1, opt.adam_beta2),
        e_optimizer = tf.keras.optimizers.Adam(opt.encoder_lr, opt.adam_beta1, opt.adam_beta2),
    )
    model.run_eagerly = opt.use_eager_mode
    callbacks = []
    
    if (save_checkpoints):
        save_best_train_callback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/HA-GAN_{dataset_name}_disc_lr-{opt.discriminator_lr}_gen_lr-{opt.generator_lr}_enc_lr-{opt.encoder_lr}_lambda-{opt.lambda_1}_batch_size-{batch_size}_e_iter-{opt.e_iter}_g_iter-{opt.g_iter}_best_train",
                                                      save_weights_only = True, save_best_only = True, 
                                                      monitor = "overall_loss", mode = "min", verbose=1)
        callbacks.append(save_best_train_callback)
        save_last_callback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/HA-GAN_{dataset_name}_disc_lr-{opt.discriminator_lr}_gen_lr-{opt.generator_lr}_enc_lr-{opt.encoder_lr}_lambda-{opt.lambda_1}_batch_size-{batch_size}_e_iter-{opt.e_iter}_g_iter-{opt.g_iter}_latest_epoch",
                                                      save_weights_only = True, save_freq='epoch', verbose=0)
        callbacks.append(save_last_callback)
        save_periodic_callback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/HA-GAN_{dataset_name}_disc_lr-{opt.discriminator_lr}_gen_lr-{opt.generator_lr}_enc_lr-{opt.encoder_lr}_lambda-{opt.lambda_1}_batch_size-{batch_size}_e_iter-{opt.e_iter}_g_iter-{opt.g_iter}_epoch-" + "{epoch:03d}",
                                                      save_weights_only = True, save_freq='epoch', verbose=1, 
                                                      period = opt.save_callback_period)
        callbacks.append(save_periodic_callback)
        if (use_test_data):
            save_best_val_callback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/HA-GAN_{dataset_name}_disc_lr-{opt.discriminator_lr}_gen_lr-{opt.generator_lr}_enc_lr-{opt.encoder_lr}_lambda-{opt.lambda_1}_batch_size-{batch_size}_e_iter-{opt.e_iter}_g_iter-{opt.g_iter}_best_val",
                                                        save_weights_only = True, save_best_only = True,
                                                        monitor = "val_overall_loss", mode = "min", verbose=1)
            callbacks.append(save_best_val_callback)

    callbacks.append(DisplayCallback())

    print('Loading data')
    # dataset = dp.load_shapenet_data_groups(dataset_name, shapes64_dir=opt.shapes64_dir, shapes256_dir=opt.shapes256_dir, image_res=opt.image_res)
    # train_data = dataset.take(math.trunc(len(dataset) * opt.training_split))
    # test_data = dataset.skip(math.trunc(len(dataset) * opt.training_split))
    train_data = tf.data.Dataset.load(f"./Data/Custom_Datasets/ha_gan_{dataset_name}_train")
    train_data = (
        train_data
        .shuffle(len(train_data))
        .batch(batch_size)
        .prefetch(2)
    )
    
    if (use_test_data):
        test_data = tf.data.Dataset.load(f"./Data/Custom_Datasets/ha_gan_{dataset_name}_test")
        test_data = (
            test_data
            .batch(opt.val_batch_size)
            .prefetch(3)
        )
        print("\nStarting HA-GAN fit")
        model.fit(
            train_data,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = test_data,
            callbacks = callbacks,
        )
    else:
        model.fit(
            train_data,
            batch_size = batch_size,
            epochs = epochs,
            callbacks = callbacks,
        )
    
def load_and_show_HA_GAN(checkpoint_path, threshold = 0, shape_res = 256, shape_code = "db80fbb9728e5df343f47bfd2fc426f7", screenshot_number = 7):
    if (shape_res == 256):
        model = ha_gan.HA_GAN(mode = 'eval')
        shapes_dir = opt.shapes256_dir
    elif (shape_res == 64):
        model = ha_gan.HA_GAN(mode = 'eval_small')
        shapes_dir = opt.shapes64_dir
    else:
        raise Exception('undefined shape_res given to load_and_show_HA_GAN()')
    model.load_weights(checkpoint_path).expect_partial()

    print("Loading image")
    image = dp.load_single_image(f'./Data/ShapeNetSem/table_screenshots/{shape_code}/{shape_code}-{screenshot_number}.png', image_res = opt.image_res)
    image = dp.normalize_image(image)
    image = tf.reshape(image, (-1, opt.image_res, opt.image_res, 3))

    print("Loading real 3D model")
    real_shape = dp.get_shape(f"{shapes_dir}/{shape_code}.binvox")

    print("Generating 3D model")
    # generated_shape = model(image)
    generated_shape = model(image, use_encoder = False)
    generated_shape = tf.math.greater_equal(generated_shape, threshold)
    
    print("Ploting image and models")
    image = tf.reshape(image, (opt.image_res, opt.image_res, 3))
    # dp.show_image_and_shapes(image, real_shape.data, generated_shape[0, :, :, :, 0].numpy(), real_shape.dims)
    dp.show_image_and_shape(image, generated_shape[0, :, :, :, 0].numpy(), real_shape.dims)


def continue_from_checkpoint(checkpoint_path, dataset_name, initial_epoch, batch_size = opt.batch_size, epochs = opt.epochs, save_checkpoints = True, use_test_data = True):
    print("\nGenerating HA-GAN model")
    model = ha_gan.HA_GAN()

    print("\nCompiling HA-GAN model")
    model.compile(
        g_optimizer = tf.keras.optimizers.Adam(opt.generator_lr, opt.adam_beta1, opt.adam_beta2),
        d_optimizer = tf.keras.optimizers.Adam(opt.discriminator_lr, opt.adam_beta1, opt.adam_beta2),
        e_optimizer = tf.keras.optimizers.Adam(opt.encoder_lr, opt.adam_beta1, opt.adam_beta2),
    )
    model.run_eagerly = opt.use_eager_mode

    print('loading checkpoint weights')
    model.load_weights(checkpoint_path).expect_partial()

    callbacks = []
    if (save_checkpoints):
        save_best_train_callback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/HA-GAN_{dataset_name}_disc_lr-{opt.discriminator_lr}_gen_lr-{opt.generator_lr}_enc_lr-{opt.encoder_lr}_lambda-{opt.lambda_1}_batch_size-{batch_size}_e_iter-{opt.e_iter}_g_iter-{opt.g_iter}_best_train",
                                                      save_weights_only = True, save_best_only = True, 
                                                      monitor = "overall_loss", mode = "min", verbose=1)
        callbacks.append(save_best_train_callback)
        save_last_callback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/HA-GAN_{dataset_name}_disc_lr-{opt.discriminator_lr}_gen_lr-{opt.generator_lr}_enc_lr-{opt.encoder_lr}_lambda-{opt.lambda_1}_batch_size-{batch_size}_e_iter-{opt.e_iter}_g_iter-{opt.g_iter}_latest_epoch",
                                                      save_weights_only = True, save_freq='epoch', verbose=0)
        callbacks.append(save_last_callback)
        save_periodic_callback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/HA-GAN_{dataset_name}_disc_lr-{opt.discriminator_lr}_gen_lr-{opt.generator_lr}_enc_lr-{opt.encoder_lr}_lambda-{opt.lambda_1}_batch_size-{batch_size}_e_iter-{opt.e_iter}_g_iter-{opt.g_iter}_epoch-" + "{epoch:03d}",
                                                      save_weights_only = True, save_freq='epoch', verbose=1, 
                                                      period = opt.save_callback_period)
        callbacks.append(save_periodic_callback)
        if (use_test_data):
            save_best_val_callback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/HA-GAN_{dataset_name}_disc_lr-{opt.discriminator_lr}_gen_lr-{opt.generator_lr}_enc_lr-{opt.encoder_lr}_lambda-{opt.lambda_1}_batch_size-{batch_size}_e_iter-{opt.e_iter}_g_iter-{opt.g_iter}_best_val",
                                                        save_weights_only = True, save_best_only = True,
                                                        monitor = "val_overall_loss", mode = "min", verbose=1)
            callbacks.append(save_best_val_callback)

    callbacks.append(DisplayCallback())

    print('Loading data')
    train_data = tf.data.Dataset.load(f"./Data/Custom_Datasets/ha_gan_{dataset_name}_train")
    train_data = (
        train_data
        .shuffle(len(train_data))
        .batch(batch_size)
        .prefetch(2)
    )
    
    if (use_test_data):
        test_data = tf.data.Dataset.load(f"./Data/Custom_Datasets/ha_gan_{dataset_name}_test")
        test_data = (
            test_data
            .batch(opt.val_batch_size)
            .prefetch(3)
        )
        print("\nStarting HA-GAN fit")
        model.fit(
            train_data,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = test_data,
            callbacks = callbacks,
            initial_epoch=initial_epoch,
        )
    else:
        model.fit(
            train_data,
            batch_size = batch_size,
            epochs = epochs,
            callbacks = callbacks,
            initial_epoch=initial_epoch,
        )


    