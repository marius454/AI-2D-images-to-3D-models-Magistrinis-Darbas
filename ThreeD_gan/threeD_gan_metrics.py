import numpy as np
import tensorflow as tf
import dataIO as d
import custom_layers
import os
import time
import gc

from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
import tensorflow_probability as tfp

import ThreeD_gan.threeD_gan_options as opt


def discriminator_loss(cross_entropy, real_output, generated_output, batch_size):
    '''
    Provide:\n
    `cross_entropy` - keras BinaryCrossentropy loss object with desired settings\n
    `real_output` - discriminator output when given real 3D models\n
    `generated_output` - discriminator output when given 3D models created by the generator network
    '''
    labels_real = tf.ones((batch_size, 1))
    labels_fake = tf.zeros((batch_size, 1))

    if (opt.add_noise_to_discriminator_labels):
        labels_real += 0.05 * tf.random.uniform(tf.shape(labels_real), minval=-1, maxval=1)
        labels_fake += 0.05 * tf.random.uniform(tf.shape(labels_fake), minval=-1, maxval=1)

    real_loss = cross_entropy(labels_real, real_output)
    fake_loss = cross_entropy(labels_fake, generated_output)
    # total_loss = (real_loss + fake_loss) / 2
    total_loss = real_loss + fake_loss
    if (opt.use_eager_mode):
        print(f"real_loss: {real_loss}")
        print(f"fake_loss: {fake_loss}")
        print(f"total_loss: {total_loss}")
    return total_loss

def discriminator_accuracy(real_output, generated_output):
    real_acu = tf.cast(tf.math.greater_equal(real_output[:,0], 0.5), tf.float16)
    fake_acu = tf.cast(tf.math.less(generated_output[:,0], 0.5), tf.float16)
    total_acu = tf.math.reduce_mean(tf.concat([real_acu, fake_acu], 0))
    return total_acu



def generator_loss(cross_entropy, generated_output, real_shapes, generated_shapes):
    '''
    Provide:\n
    `cross_entropy` - keras BinaryCrossentropy loss object with desired settings\n
    `generated_output` - discriminator output when given 3D models created by the generator network\n
    `real_shapes` - an array of discriminator predictions on the real objects\n
    `generated_shapes` - an array of discriminator predictions on the fake objects
    '''
    labels = tf.ones_like(generated_output)
    if (opt.add_noise_to_discriminator_labels):
        labels += 0.05 * tf.random.uniform(tf.shape(labels), minval=-1, maxval=1)

    gen_loss = cross_entropy(labels, generated_output)
    recon_loss = reconstruction_loss(real_shapes, generated_shapes)
    total_loss = gen_loss + opt.alpha_2 * recon_loss

    if (opt.use_eager_mode):
        print(f"gen_loss: {gen_loss}")
        print(f"a2*recon_loss: {opt.alpha_2 * recon_loss}")
        print(f"total_loss: {total_loss}")
    return total_loss

    # total_loss = gen_loss + recon_loss
    # return total_loss

def generator_accuracy(generated_output):
    predictions = tf.cast(tf.math.greater_equal(generated_output, 0.5), tf.float16)
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

    # ONLY RECONSTRUCTION
    # return reconstruction_loss(real_shapes, generated_shapes)
    # return reconstruction_loss(real_shapes, generated_shapes) * a2

    # ONLY KL DIVERGENCE
    # return kullback_leiber_loss(z_means, z_vars)
    # return kullback_leiber_loss(z_means, z_vars) * a1

    # RECONSTRUCTION AND KL DIVERGENCE
    # KL_loss = kullback_leiber_loss(z_means, z_vars)
    # recon_loss = reconstruction_loss(real_shapes, generated_shapes)
    # total_loss = KL_loss + recon_loss
    # return total_loss

    KL_loss = kullback_leiber_loss(z_means, z_vars)
    recon_loss = reconstruction_loss(real_shapes, generated_shapes)
    total_loss = opt.alpha_1 * KL_loss + opt.alpha_2 * recon_loss

    if (opt.use_eager_mode):
        print(f"a1*KL_loss: {opt.alpha_1 * KL_loss}")
        print(f"a2*recon_loss: {opt.alpha_2 * recon_loss}")
        print(f"total_loss: {total_loss}")
    return total_loss



def reconstruction_loss(real_shapes, generated_shapes):
    """
    Calculate the Euclidan distance ||G(E(y)) − x||2\n
    Sum over batch (mean) reduction is applied
    """
    # USE NORMAL EUCLIDEAN DISTANCE
    # l2_norm = tf.norm(generated_shapes - real_shapes) # square root of squared differences in every coordinate
    # l2_norm = l2_norm / tf.cast(tf.shape(real_shapes)[0], tf.float32) # imitating the SUM_OVER_BATCH reduction
    # return l2_norm

    # USE SQARED EUCLIDEAN DISTACE
    euclidean_distance_per_batch = tf.reduce_sum(tf.math.squared_difference(
            tf.reshape(generated_shapes, (-1, 64 * 64 * 64)) * opt.voxel_weight, 
            tf.reshape(real_shapes, (-1, 64 * 64 * 64)) * opt.voxel_weight,
        ), 1)
    return tf.reduce_mean(euclidean_distance_per_batch)
    # return euclidean_distance_per_batch





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
    KL_loss = tf.keras.losses.KLDivergence()([prior_means, prior_vars], [z_means, z_vars])
    return KL_loss / tf.cast(tf.shape(z_means)[0], tf.float32)
    # return KL_loss

    # USE FORMULA FROM EXAMPLES
    # KL_divergence_per_batch = -0.5 * tf.reduce_sum(1.0 + 2.0 * z_vars - z_means ** 2 - tf.exp(2.0 * z_vars), 1)
    # return tf.reduce_mean(KL_divergence_per_batch)
    # # return KL_divergence_per_batch


