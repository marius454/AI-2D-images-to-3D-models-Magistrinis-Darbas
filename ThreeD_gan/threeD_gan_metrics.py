import numpy as np
import tensorflow as tf
import dataIO as d
import custom_layers
import os
import time
import gc

from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
import tensorflow_probability as tfp


def discriminator_loss(cross_entropy, real_output, generated_output):
    '''
    Provide:\n
    `cross_entropy` - keras BinaryCrossentropy loss object with desired settings\n
    `real_output` - discriminator output when given real 3D models\n
    `generated_output` - discriminator output when given 3D models created by the generator network
    '''
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(generated_output), generated_output)
    total_loss = real_loss + fake_loss
    return total_loss

def discriminator_accuracy(real_output, generated_output):
    real_acu = tf.cast(tf.math.greater_equal(real_output[:, 0], 0.5), tf.float32)
    fake_acu = tf.cast(tf.math.less(generated_output[:, 0], 0.5), tf.float32)
    total_acu = tf.math.reduce_mean(tf.concat([real_acu, fake_acu], 0))
    return total_acu

# Compare the discriminators decisions on the generated images to an array of 1s.
def generator_loss(cross_entropy, generated_output, real_shapes, generated_shapes):
    '''
    Provide:\n
    `cross_entropy` - keras BinaryCrossentropy loss object with desired settings\n
    `generated_output` - discriminator output when given 3D models created by the generator network\n
    `real_shapes` - an array of discriminator predictions on the real objects\n
    `generated_shapes` - an array of discriminator predictions on the fake objects
    '''
    gen_loss = cross_entropy(tf.ones_like(generated_output), generated_output)
    recon_loss = reconstruction_loss(real_shapes, generated_shapes)
    
    # Reconstruction loss weight given in the paper, it is unclear how it is supposed to be used
    # a2 = 1e-4     
    # total_loss = gen_loss + a2*recon_loss
    # return total_loss

    total_loss = gen_loss + recon_loss
    return total_loss

def generator_accuracy(generated_output):
    predictions = tf.cast(tf.math.greater_equal(generated_output[:, 0], 0.5), tf.float32)
    accuracy = tf.math.reduce_mean(predictions)
    return accuracy
    

def vae_loss(z_means, z_vars, real_shapes, generated_shapes):
    """
    Provide:\n
    `z_means` - an array of real objects coresponding to some images\n
    `z_vars` - an array objects generated from those images\n
    `real_shapes` - an array of discriminator predictions on the real objects\n
    `generated_shapes` - an array of discriminator predictions on the fake objects
    """
    # print(f"KL Divergence: {KL_loss}")
    # print(f"Reconstruction loss: {recon_loss}")

    a1 = 5        # KL divergence loss weight given in the paper
    a2 = 1e-4     # Reconstruction loss weight given in the paper

    # ONLY RECONSTRUCTION
    # return reconstruction_loss(real_shapes, generated_shapes)
    # return reconstruction_loss(real_shapes, generated_shapes) * a2

    # ONLY KL DIVERGENCE
    # return kullback_leiber_loss(z_means, z_vars)
    # return kullback_leiber_loss(z_means, z_vars) * a1

    # RECONSTRUCTION AND KL DIVERGENCE
    KL_loss = kullback_leiber_loss(z_means, z_vars)
    recon_loss = reconstruction_loss(real_shapes, generated_shapes)
    # print("KL divergence loss: " + str(KL_loss))
    # print("Reconstruction loss: " + str(recon_loss))
    total_loss = KL_loss + recon_loss
    return total_loss

    # KL_loss = kullback_leiber_loss(z_means, z_vars)
    # recon_loss = reconstruction_loss(real_shapes, generated_shapes)
    # total_loss = a1*KL_loss + a2*recon_loss
    # return total_loss



def reconstruction_loss(real_shapes, generated_shapes):
    """
    Calculate the Euclidan distance ||G(E(y)) − x||2\n
    Sum over batch (mean) reduction is applied
    """
    l2_norm = tf.norm(generated_shapes - real_shapes) # square root of squared differences in every coordinate
    l2_norm = l2_norm / tf.cast(tf.shape(real_shapes)[0], tf.float32) # imitating the SUM_OVER_BATCH reduction
    return l2_norm



def kullback_leiber_loss(z_means, z_vars, z_size = 200):
    """
    Calculate the Kullback-Leiber dvergence DKL(q(z|y) || p(z))\n
    Sum over batch (mean) reduction is applied
    """
    prior_means = tf.fill([tf.shape(z_means)[0], z_size], 0.0)
    prior_vars = tf.fill([tf.shape(z_vars)[0], z_size], 1.0)

    # USE TENSORFLOW_PROBABILITY
    # prior = tfp.distributions.MultivariateNormalDiag(prior_means, prior_vars)
    # z_distribution = tfp.distributions.MultivariateNormalDiag(z_means, z_vars)
    # KL_loss = tf.reduce_mean(kl_lib.kl_divergence(z_distribution, prior))
    # return KL_loss

    # USE KERAS LOSSES
    return tf.keras.losses.KLDivergence()([prior_means, prior_vars], [z_means, z_vars])


