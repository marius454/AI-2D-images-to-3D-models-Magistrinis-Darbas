import numpy
import tensorflow as tf
import variables as var
import data_processing as dp
import dataIO as d
import custom_layers
import os

# Weight initialization described in Radford et al. [2016]
# Filter values are equivalent to channels in Wu et al. [2016]
def make_generator_model():
    res = var.threeD_gan_resolution
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv3DTranspose(filters=var.threeD_gan_generator_first_filter, 
        kernel_size=(4, 4, 4), strides=1, use_bias=False, input_shape=(1, 1, 1, var.threeD_gan_z_size), 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    assert model.output_shape == (None, 4, 4, 4, 512)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    layer_width = 4
    for filter in var.threeD_gan_generator_intermediate_filters:
        layer_width = layer_width * 2
        model.add(tf.keras.layers.Conv3DTranspose(filters=filter, kernel_size=(4, 4, 4), strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
        assert model.output_shape == (None, layer_width, layer_width, layer_width, filter)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv3DTranspose(filters=var.threeD_gan_generator_final_filter, kernel_size=(4, 4, 4), 
        strides=2, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), 
        activation='sigmoid'))
        # activation='tanh'))
    assert model.output_shape == (None, res, res, res, 1)

    return model


def make_discriminator_model():
    res = var.threeD_gan_resolution
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv3D(filters=var.threeD_gan_discriminator_first_filter, kernel_size=4, 
        strides=2, padding='same', use_bias=False, input_shape=(res, res, res, 1),
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))

    for filter in var.threeD_gan_discriminator_intermediate_filters:
        model.add(tf.keras.layers.Conv3D(filters=filter, kernel_size=4, strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(0.2))
    
    model.add(tf.keras.layers.Conv3D(filters=var.threeD_gan_discriminator_final_filter, 
        kernel_size=4, strides=1, padding='valid', use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
        activation='sigmoid'))
    model.add(tf.keras.layers.Flatten())

    return model


# takes a 3x256x256 image as input and outputs a 200 dimensional vector
def make_encoder_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(filters=64, kernel_size=11, strides=4, padding="same", use_bias=False,
                input_shape=(256, 256, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=400, kernel_size=8, strides=1, padding="valid", use_bias=False),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            # At this point we a have a 400 dimensional gaussian latent space, that we need to sample a 200 dimensional vector from
            # custom_layers.Sampling(),
        ]
    )
    print(model.output_shape)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Compare the discriminators decisions on the generated images to an array of 1s.
def generator_loss(fake_output):
    G_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return 

def vae_loss(encoder_output, real_objects, generated_objects, real_output, fake_output):
    """
    Provide:
    an array of real objects coresponding to some images
    an array objects generated from those images
    an array of discriminator predictions on the real objects
    an array of discriminator predictions on the fake objects
    """
    a1 = 5            # KL divergence loss weight given in the paper
    a2 = pow(10, -4)  # Reconstruction loss weight given in the paper
    prior_mean = tf.fill([None, 200], 0)
    prior_var = tf.fill([None, 200], 1)
    prior = tf.concat([prior_mean, prior_var], 1)
    KL_loss = tf.keras.losses.kl_divergence(encoder_output, prior)
    recon_loss = tf.keras.losses.mean_squared_error(real_objects, generated_objects)

    total_loss = a1*KL_loss + a2*recon_loss
    return total_loss


def train_3d_vae_gan(data):
    # declare neural network models
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    encoder = make_encoder_model()

    # declare optimizers
    G_optimizer = tf.keras.optimizers.Adam(var.threeD_gan_adam_beta)
    D_optimizer = tf.keras.optimizers.Adam(var.threeD_gan_adam_beta)
    E_optimizer = tf.keras.optimizers.Adam(var.threeD_gan_adam_beta)

    # declare training checkpoints
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(G_optimizer=G_optimizer,
                                    D_optimizer=D_optimizer,
                                    E_optimizer=E_optimizer,
                                    generator=generator,
                                    discriminator=discriminator,
                                    encoder=encoder)

    # seed required for better evaluation of training results with the same parameters
    num_examples_to_generate = 4
    seed = tf.random.normal([num_examples_to_generate, var.threeD_gan_z_size])

    z = tf.random.uniform((1, var.threeD_gan_z_size), minval=0, maxval=1)
    z = tf.reshape(z, (-1, 1, 1, 1, var.threeD_gan_z_size))



# def train_3d_VAE_gan():