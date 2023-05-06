import numpy as np
import tensorflow as tf
import dataIO as d
import custom_layers
import os
import time
import gc

import data_processing as dp
from callbacks import DisplayCallback

from ThreeD_gan import threeD_gan
import ThreeD_gan.threeD_gan_options as opt

def run_3D_VAE_GAN(dataset_name, batch_size = opt.batch_size, epochs = 100):
    ## Load a dataset as tf.Dataset
    print('Loading data')
    dataset = dp.load_data(dataset_name, downscale_factor=2)
    dataset = (
        dataset
        .shuffle(len(dataset))
        .batch(batch_size)
    )

    print("\nGenerating 3D-VAE-GAN model")
    model = threeD_gan.ThreeD_gan()

    print("\nCompiling 3D-VAE-GAN model")
    model.compile(
        g_optimizer = tf.keras.optimizers.Adam(opt.generator_lr, opt.adam_beta),
        d_optimizer = tf.keras.optimizers.Adam(opt.discriminator_lr, opt.adam_beta),
        e_optimizer = tf.keras.optimizers.Adam(opt.encoder_lr),
    )
    model.run_eagerly = opt.use_eager_mode
    saveCallback = tf.keras.callbacks.ModelCheckpoint(filepath = f"./training_checkpoints/3D_GAN_{dataset_name}_disc_lr-{opt.discriminator_lr}_gen_lr-{opt.generator_lr}_enc_lr-{opt.encoder_lr}_a1-{opt.alpha_1}_a2-{opt.alpha_2}_voxel_weight-{opt.voxel_weight}_recon-sqsum",
                                                      save_weights_only = True,
                                                      save_best_only = True,
                                                      monitor = "overall_loss",
                                                      mode = "min",
                                                      # save_freq=5*var.STEPS_PER_EPOCH + 1,
                                                      verbose=1)

    print("\nStarting 3D-VAE-GAN fit")
    model.fit(
        dataset,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[DisplayCallback(), saveCallback],
    )
    
def load_and_show_3D_VAE_GAN(checkpoint_path):
    model = threeD_gan.ThreeD_gan()
    # NOT SURE IF THIS IS NECESSARY
    # model.compile(
    #     g_optimizer = tf.keras.optimizers.Adam(opt.generator_lr, 0.5),
    #     d_optimizer = tf.keras.optimizers.Adam(opt.discriminator_lr, 0.5),
    #     e_optimizer = tf.keras.optimizers.Adam(0.0003)
    # )
    model.load_weights(checkpoint_path)

    print("Loading image")
    image = dp.load_single_image('./Data/ShapeNetSem/screenshots/db80fbb9728e5df343f47bfd2fc426f7/db80fbb9728e5df343f47bfd2fc426f7-7.png')
    image = tf.reshape(image, (-1, 256, 256, 3))
    image = dp.normalize_image(image)

    print("Generating 3D model")
    generated_shape = model.predict(image)
    generated_shape = tf.math.greater_equal(generated_shape, 0.5)
    print(generated_shape[0, :, :, :, 0].numpy())
    
    print("Ploting 3D model")
    dp.plot_3d_model(generated_shape[0, :, :, :, 0].numpy(), (64, 64, 64))
