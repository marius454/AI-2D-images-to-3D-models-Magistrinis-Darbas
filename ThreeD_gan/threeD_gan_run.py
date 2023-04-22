import numpy as np
import tensorflow as tf
import dataIO as d
import custom_layers
import os
import time
import gc

import data_processing as dp
from ThreeD_gan import threeD_gan

def run_3D_VAE_GAN(dataset_name):
    ## Load a dataset as tf.Dataset
    batch_size = 100
    print('Loading data')
    dataset = dp.load_data(dataset_name, downscale_factor=2)
    dataset = (
        dataset
        .shuffle(len(dataset))
        .batch(batch_size)
    )

    print("\nGenerating 3D-VAE-GAN model")
    gan = threeD_gan.ThreeD_gan()

    print("\nCompiling 3D-VAE-GAN model")
    gan.compile(
        g_optimizer = tf.keras.optimizers.Adam(0.0025, 0.5),
        d_optimizer = tf.keras.optimizers.Adam(1e-5, 0.5),
        e_optimizer = tf.keras.optimizers.Adam(0.0003)
        #TODO add combined loss function and metric tracking
    )
    saveCallback = tf.keras.callbacks.ModelCheckpoint(filepath = "./training_checkpoints/3D_GAN_" + dataset_name,
                                                      save_weights_only=True,
                                                      save_best_only = True,
                                                      monitor = "overall_loss",
                                                      mode = "min",
                                                      # save_freq=5*var.STEPS_PER_EPOCH + 1,
                                                      verbose=1)

    print("\nStarting 3D-VAE-GAN fit")
    gan.fit(
        dataset,
        batch_size=batch_size,
        epochs=100,
        callbacks=[saveCallback],
    )