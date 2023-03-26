import numpy as np
import tensorflow as tf
import variables as var
import data_processing as dp
import dataIO as d
import custom_layers
import os
import time
import gc

from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
import tensorflow_probability as tfp


# Weight initialization described in Radford et al. [2016]
# Filter values are equivalent to channels in Wu et al. [2016]
def make_generator_model():
    """Creates generator model as discribed in Wu et al. [2016], but according to the settings in variables.py file"""
    res = var.threeD_gan_resolution
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv3DTranspose(filters=var.threeD_gan_generator_first_filter, 
        kernel_size=(4, 4, 4), strides=1, use_bias=False, input_shape=(1, 1, 1, var.threeD_gan_z_size), padding='valid',
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    print(model.output_shape)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    layer_width = 4
    for filter in var.threeD_gan_generator_intermediate_filters:
        layer_width = layer_width * 2
        model.add(tf.keras.layers.Conv3DTranspose(filters=filter, kernel_size=(4, 4, 4), strides=2, padding='same',
            use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
        print(model.output_shape)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv3DTranspose(filters=var.threeD_gan_generator_final_filter, kernel_size=(4, 4, 4), 
        strides=2, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), 
        activation='sigmoid'))
        # activation='tanh'))
    # model.add(custom_layers.Threshold(trainable = False))
    print(model.output_shape)
    assert model.output_shape == (None, res, res, res, 1)

    return model


def make_discriminator_model():
    """Creates discriminator model as discribed in Wu et al. [2016], but according to the settings in variables.py file"""
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
    print(model.output_shape)

    return model


# takes a 256x256x3 image as input and outputs a 200 dimensional vector
def make_encoder_model():
    """Creates encoder model as discribed in Wu et al. [2016]"""
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
            # The JiaJun Wu paper states that the last layer is a convolution layer, 
            # however the Larsen paper states that the last layer should be a fully connected layer
            tf.keras.layers.Conv2D(filters=400, kernel_size=8, strides=1, padding="valid", use_bias=False),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(var.threeD_gan_z_size + var.threeD_gan_z_size),
            # At this point we a have a 400 dimensional gaussian latent space, that we need to sample a 200 dimensional vector from
            # custom_layers.Sampling(),
        ]
    )
    print(model.output_shape)
    assert model.output_shape == (None, var.threeD_gan_z_size + var.threeD_gan_z_size)

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)

# Compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.
def discriminator_loss(real_output, generated_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(generated_output), generated_output)
    total_loss = real_loss + fake_loss
    return total_loss

def discriminator_accuracy(real_output, generated_output):
    real_acu = tf.cast(tf.math.greater_equal(real_output[:, 0], 0.5), tf.float32)
    fake_acu = tf.cast(tf.math.less_equal(generated_output[:, 0], 0.5), tf.float32)
    total_acu = tf.math.reduce_mean(tf.concat([real_acu, fake_acu], 0))
    return total_acu

# Compare the discriminators decisions on the generated images to an array of 1s.
def generator_loss(generated_output, real_shapes, generated_shapes):
    gen_loss = cross_entropy(tf.ones_like(generated_output), generated_output)
    recon_loss = reconstruction_loss(real_shapes, generated_shapes)
    
    # a2 = 1e-4     # Reconstruction loss weight given in the paper
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
    Provide:
    `z_means` - an array of real objects coresponding to some images\n
    `z_vars` - an array objects generated from those images\n
    `real_shapes` - an array of discriminator predictions on the real objects\n
    `generated_shapes` - an array of discriminator predictions on the fake objects
    """
    KL_loss = kullback_leiber_loss(z_means, z_vars)
    recon_loss = reconstruction_loss(real_shapes, generated_shapes)
    print(f"KL Divergence: {KL_loss}")
    print(f"Reconstruction loss: {recon_loss}")

    # a1 = 5        # KL divergence loss weight given in the paper
    # a2 = 1e-4     # Reconstruction loss weight given in the paper
    # total_loss = a1*KL_loss + a2*recon_loss
    # return total_loss

    total_loss = KL_loss + recon_loss
    return total_loss

def reconstruction_loss(real_shapes, generated_shapes):
    """
    Calculate the Euclidan distance ||G(E(y)) − x||2\n
    Sum over batch (mean) reduction is applied
    """
    l2_norm = tf.norm(generated_shapes - real_shapes)
    l2_norm = l2_norm / len(real_shapes)
    return l2_norm

def kullback_leiber_loss(z_means, z_vars):
    """
    Calculate the Kullback-Leiber dvergence DKL(q(z|y) || p(z))\n
    Sum over batch (mean) reduction is applied
    """
    prior_mean = tf.fill((var.threeD_gan_z_size), 0.0)
    prior_var = tf.fill((var.threeD_gan_z_size), 1.0)
    
    prior = tfp.distributions.MultivariateNormalDiag(prior_mean, prior_var)
    z_distribution = tfp.distributions.MultivariateNormalDiag(z_means, z_vars)
    KL_loss = tf.reduce_mean(kl_lib.kl_divergence(z_distribution, prior))
    return KL_loss


def train_3d_vae_gan(epochs = var.threeD_gan_epochs):
    # declare neural network models
    print('Initializing generator')
    generator = make_generator_model()
    print('Initializing discriminator')
    discriminator = make_discriminator_model()
    print('Initializing encoder')
    encoder = make_encoder_model()

    # Declare optimizers
    G_optimizer = tf.keras.optimizers.Adam(var.threeD_gan_generator_learning_rate, var.threeD_gan_adam_beta1, var.threeD_gan_adam_beta2)
    D_optimizer = tf.keras.optimizers.Adam(var.threeD_gan_discriminator_learning_rate, var.threeD_gan_adam_beta1, var.threeD_gan_adam_beta2)
    E_optimizer = tf.keras.optimizers.Adam(var.threeD_gan_encoder_learning_rate, var.threeD_gan_adam_beta1, var.threeD_gan_adam_beta2)
    # E_optimizer = tf.keras.optimizers.Adam(var.threeD_gan_adam_beta1)

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
    # num_examples_to_generate = 4
    # seed = tf.random.normal([num_examples_to_generate, var.threeD_gan_z_size])


    print('Loading data')
    dataset = dp.load_data("shapenet_tables", downscale_factor=2)
    dataset = (
        dataset
        .shuffle(len(dataset))
        .batch(var.threeD_gan_batch_size)
    )

    print('Initiating training')
    for epoch in range(epochs):
        start = time.time()

        batch_nr = 0
        for batch in dataset:
            print('')

            images, real_shapes = batch
            real_shapes = tf.cast(real_shapes, tf.float32)
            real_shapes = tf.reshape(real_shapes, (real_shapes.shape[0], real_shapes.shape[1], real_shapes.shape[2], real_shapes.shape[3], 1))
            # Last batch is smaller than batch_size
            noise = get_z(0, 1, len(images))

            train_step1(real_shapes, noise, generator, discriminator, D_optimizer)
            train_step2(images, real_shapes, generator, encoder, E_optimizer)
            train_step3(images, real_shapes, noise, generator, discriminator, encoder, G_optimizer)

            batch_nr = batch_nr + 1
            print(f"Completed batch {batch_nr} of epoch {epoch + 1}")

        # TODO generate and save progress images

        if (epoch + 1) % var.threeD_gan_checkpoint_frequency == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print (f'Time for epoch {epoch + 1} is {time.time()-start} sec')



def train_step1(real_shapes, noise, generator, discriminator, D_optimizer, seed=None):
    with tf.GradientTape() as disc_tape:
        generated_shapes = generator(inputs=noise, training=False)
        real_output = discriminator(inputs=real_shapes, training=True)
        generated_output = discriminator(inputs=generated_shapes, training=True)

        disc_loss = discriminator_loss(real_output, generated_output)
        disc_accuracy = discriminator_accuracy(real_output, generated_output)

    print(f"Discriminator loss: {disc_loss}; Discriminator accuracy: {disc_accuracy * 100}%")
    if (disc_accuracy < 0.8):
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        D_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train_step2(images, real_shapes, generator, encoder, E_optimizer, seed=None):
    with tf.GradientTape() as ecn_tape:
        encoder_output = encoder(inputs=images, training=True)
        z_means, z_vars = tf.split(encoder_output, num_or_size_splits=2, axis=1)
        z = get_z(z_means, z_vars, len(images))
        generated_shapes = generator(inputs=z, training=False)

        enc_loss = vae_loss(z_means, z_vars, real_shapes, generated_shapes)

    print(f"Encoder loss: {enc_loss}")
    gradients_of_encoder = ecn_tape.gradient(enc_loss, encoder.trainable_variables)
    E_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))


def train_step3(images, real_shapes, noise, generator, discriminator, encoder, G_optimizer, seed=None):
    with tf.GradientTape() as gen_tape:
        shapes_from_noise = generator(inputs=noise, training=True)
        output_from_noise = discriminator(inputs=shapes_from_noise, training=False)

        encoder_output = encoder(inputs=images, training=False)
        z_means, z_vars = tf.split(encoder_output, num_or_size_splits=2, axis=1)
        z = get_z(z_means, z_vars, len(images))
        generated_shapes = generator(inputs=z, training=True)

        gen_loss = generator_loss(output_from_noise, real_shapes, generated_shapes)
        gen_accuracy = generator_accuracy(output_from_noise)

    print(f"Generator loss: {gen_loss}; Generator accuracy against the discriminator: {gen_accuracy * 100}%")
    if gen_accuracy < 0.8:
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        G_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))



def get_z(z_means, z_vars, batch_size):
    # if isinstance(z_means, int) or isinstance(z_means, float):
    #     z_means = tf.fill((batch_size, var.threeD_gan_z_size), z_means)
    #     z_means = tf.cast(z_means, tf.float32)
    # if isinstance(z_vars, int) or isinstance(z_vars, float):
    #     z_vars = tf.fill((batch_size, var.threeD_gan_z_size), z_vars)
    #     z_vars = tf.cast(z_vars, tf.float32)

    # z = []
    # for i in range(batch_size):
    #     # z.append(tf.random.normal((var.threeD_gan_z_size, ), z_means[i], tf.sqrt(z_vars[i])))
    #     z.append()
    # z = tf.convert_to_tensor(z)
    # z = tf.reshape(z, (batch_size, 1, 1, 1, var.threeD_gan_z_size))

    # return z

    eps = tf.random.normal(shape = (batch_size, var.threeD_gan_z_size))
    z = eps * tf.exp(z_vars * .5) + z_means
    z = tf.reshape(z, (batch_size, 1, 1, 1, var.threeD_gan_z_size))
    return z







# with tf.GradientTape() as disc_tape, tf.GradientTape() as ecn_tape, tf.GradientTape() as gen_tape:
                # Train discriminator
                # shapes_from_noise = generator(inputs=noise, training=True)
                
                # disc_real_output = discriminator(inputs=real_shapes, training=True)
                # disc_gen_noise_output = discriminator(inputs=shapes_from_noise, training=True)

                # disc_loss = discriminator_loss(disc_real_output, disc_gen_noise_output)
                # print(f"Discriminator loss: {disc_loss}")
                # gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                # D_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

                # # Train encoder
                # encoder_output = encoder(inputs=images, training=True)
                # z_means, z_vars = tf.split(encoder_output, num_or_size_splits=2, axis=1)
                # z = get_z(z_means, z_vars)
                
                # shapes_from_encoder = generator(inputs=z, training=True)

                # enc_loss = vae_loss(z_means, z_vars, real_shapes, shapes_from_encoder[:, :, :, :, 0])
                # del encoder_output
                # del shapes_from_encoder

                # print(f"Encoder loss: {enc_loss}")
                # gradients_of_encoder = ecn_tape.gradient(enc_loss, encoder.trainable_variables)
                # E_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))

                # # Train generator with updated encoder and discriminator
                # del disc_gen_noise_output

                # disc_gen_noise_output = discriminator(inputs=shapes_from_noise, training=True)
                # encoder_output = encoder(inputs=images, training=True)

                # z_means, z_vars = tf.split(encoder_output, num_or_size_splits=2, axis=1)
                # z = get_z(z_means, z_vars)
                # shapes_from_encoder = generator(inputs=z, training=True)

                # gen_loss = generator_loss(disc_gen_noise_output, real_shapes, shapes_from_encoder[:, :, :, :, 0])
                # print(f"Generator loss: {gen_loss}")
                # gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
                # G_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))



# def kl_mvn(m0, S0, m1, S1):
#     """
#     Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
#     Also computes KL divergence from a single Gaussian pm,pv to a set
#     of Gaussians qm,qv.
    
#     From wikipedia
#     KL( (m0, S0) || (m1, S1))
#          = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
#                   (m1 - m0)^T S1^{-1} (m1 - m0) - N )
#     """
#     # Turn arrays into diagonal matrices
#     md0 = np.diag(m0)
#     md1 = np.diag(m1)
#     Sd0 = np.diag(S0)
#     Sd1 = np.diag(S1)

#     for i in range(len(S0)):
#         if (S0[i] == 0.0):
#             print (i)

#     # store inv diag covariance of S1 and diff between means
#     N = md0.shape[0]
#     iS1 = np.linalg.inv(Sd1)
#     diff = md1 - md0

#     # kl is made of three terms
#     tr_term   = np.trace(iS1 @ Sd0)
#     det_S1 = np.prod(S1)
#     det_S0 = np.prod(S0)
#     det_term  = np.sum(np.log(det_S1)) - np.sum(np.log(det_S0)) # np.log(det_S1/det_S0)
#     quad_term = np.sum(diff.T @ iS1 @ diff) # np.sum( (diff*diff) * iS1, axis=1)
#     #print(tr_term,det_term,quad_term)
#     return .5 * (tr_term + det_term + quad_term - N) 


# # Sum of squared errors (SSE)
# # recon_loss = tf.reduce_sum(tf.pow(generated_shapes - tf.cast(real_shapes, tf.float32), 2))
# # recon_loss = recon_loss / len(real_shapes)
# # return gen_loss + a2*recon_loss

# recon_loss = tf.keras.losses.MeanAbsoluteError()(real_shapes, generated_shapes)


