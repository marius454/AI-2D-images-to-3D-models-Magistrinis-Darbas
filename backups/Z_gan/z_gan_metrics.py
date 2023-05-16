import numpy as np
import tensorflow as tf
import dataIO as d
import backups.custom_layers as custom_layers
import os
import time
import gc

def generator_loss(disc_generated_output, generated_objects, real_objects):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    LAMBDA = 100 # Decided by the authors of the pix2pix paper

    gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)
    recon_loss = tf.keras.losses.mean_absolute_error(real_objects, generated_objects)
    total_gen_loss = gan_loss + (LAMBDA * recon_loss)

    return total_gen_loss, gan_loss, recon_loss