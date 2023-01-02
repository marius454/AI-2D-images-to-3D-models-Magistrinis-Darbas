import numpy
import tensorflow as tf
import variables as var
import data_processing as dp
import dataIO as d
import custom_layers
import os
import time

# Weight initialization described in Radford et al. [2016]
# Filter values are equivalent to channels in Wu et al. [2016]
def make_generator_model():
    res = var.threeD_gan_resolution
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv3DTranspose(filters=var.threeD_gan_generator_first_filter, 
        kernel_size=(4, 4, 4), strides=1, use_bias=False, input_shape=(1, 1, 1, var.threeD_gan_z_size), 
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
    print(model.output_shape)
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


# takes a 256x256x3 image as input and outputs a 200 dimensional vector
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
            tf.keras.layers.Conv2D(filters=400, kernel_size=8, strides=1, padding="valid", use_bias=False, activation="sigmoid"),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            # At this point we a have a 400 dimensional gaussian latent space, that we need to sample a 200 dimensional vector from
            # custom_layers.Sampling(),
        ]
    )
    print(model.output_shape)

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.
def discriminator_loss(real_output, generated_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(generated_output), generated_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Compare the discriminators decisions on the generated images to an array of 1s.
def generator_loss(generated_output, real_shapes, generated_shapes):
    gen_loss = cross_entropy(tf.ones_like(generated_output), generated_output)
    recon_loss = tf.keras.losses.mean_squared_error(real_shapes, generated_shapes)
    return gen_loss + recon_loss

def vae_loss(encoder_output, real_shapes, generated_shapes):
    """
    Provide:
    an array of real objects coresponding to some images
    an array objects generated from those images
    an array of discriminator predictions on the real objects
    an array of discriminator predictions on the fake objects
    """
    a1 = 5            # KL divergence loss weight given in the paper
    a2 = pow(10, -4)  # Reconstruction loss weight given in the paper
    prior_mean = tf.fill([len(encoder_output), var.threeD_gan_z_size], 0.0)
    prior_var = tf.fill([len(encoder_output), var.threeD_gan_z_size], 1.0)
    prior = tf.concat([prior_mean, prior_var], 1)
    KL_loss = tf.keras.losses.KLDivergence()(encoder_output, prior)
    recon_loss = tf.reduce_sum(tf.pow((generated_shapes - real_shapes), 2)) # Manualy calculating mean squared error
    
    # print(KL_loss)
    # print(recon_loss)
    total_loss = a1*KL_loss + a2*recon_loss
    return total_loss



def train_3d_vae_gan(epochs = var.threeD_gan_epochs):
    # declare neural network models
    print('Initializing generator')
    generator = make_generator_model()
    print('Initializing discriminator')
    discriminator = make_discriminator_model()
    print('Initializing encoder')
    encoder = make_encoder_model()

    # Declare optimizers
    G_optimizer = tf.keras.optimizers.Adam(var.threeD_gan_generator_learning_rate, var.threeD_gan_adam_beta)
    D_optimizer = tf.keras.optimizers.Adam(var.threeD_gan_discriminator_learning_rate, var.threeD_gan_adam_beta)
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

    
    print('Loading data')
    dataset = dp.load_data()
    dataset = (
        dataset
        .shuffle(len(dataset))
        .batch(var.threeD_gan_batch_size)
    )

    print('Initiating training')
    for epoch in range(epochs):
        start = time.time()

        for batch in dataset:
            train_step1(batch, generator, discriminator, D_optimizer)
            train_step2(batch, generator, encoder, E_optimizer)
            train_step3(batch, generator, discriminator, encoder, G_optimizer)

        # TODO generate and save progress images

        if (epoch + 1) % var.threeD_gan_checkpoint_frequency == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print (f'Time for epoch {epoch + 1} is {time.time()-start} sec')


def train_step1(data, generator, discriminator, D_optimizer, seed=None):
    noise = tf.random.uniform((var.threeD_gan_batch_size, var.threeD_gan_z_size), minval=0, maxval=1)
    noise = tf.reshape(noise, (var.threeD_gan_batch_size, 1, 1, 1, var.threeD_gan_z_size))

    with tf.GradientTape() as disc_tape:
        generated_shapes = generator(inputs=noise, training=True)
        images, real_shapes = data

        real_output = discriminator(inputs=real_shapes, training=True)
        generated_output = discriminator(inputs=generated_shapes, training=True)

        disc_loss = discriminator_loss(real_output, generated_output)
        print(f"Discriminator loss: {disc_loss}")
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        D_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train_step2(data, generator, encoder, E_optimizer, seed=None):
    with tf.GradientTape() as ecn_tape:
        images, real_shapes = data
        
        encoder_output = encoder(inputs=images, training=True)
        z_means, z_vars = tf.split(encoder_output, num_or_size_splits=2, axis=1)
        z = []
        for i in range(len(images)):
            z.append(tf.random.normal((1, var.threeD_gan_z_size), z_means[i], tf.pow(z_vars[i], 0.5)))
        z = tf.convert_to_tensor(z)
        z = tf.reshape(z, (len(images), 1, 1, 1, var.threeD_gan_z_size))
        print(z[0])
        generated_shapes = generator(inputs=z, training=True)[:, :, :, :, 0]
        # generated_shapes = tf.where(generated_shapes > 0.5, 1., 0.)

        enc_loss = vae_loss(encoder_output, real_shapes, generated_shapes)
        print(f"Encoder loss: {enc_loss}")
        gradients_of_encoder = ecn_tape.gradient(enc_loss, encoder.trainable_variables)
        E_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))

def train_step3(data, generator, discriminator, encoder, G_optimizer, seed=None):
    noise = tf.random.uniform((var.threeD_gan_batch_size, var.threeD_gan_z_size), minval=0, maxval=1)
    noise = tf.reshape(noise, (var.threeD_gan_batch_size, 1, 1, 1, var.threeD_gan_z_size))

    with tf.GradientTape() as gen_tape:
        images, real_shapes = data

        shapes_from_noise = generator(inputs=noise, training=True)
        output_from_noise = discriminator(inputs=shapes_from_noise, training=True)
        real_output = discriminator(inputs=real_shapes, training=True)

        encoder_output = encoder(inputs=images, training=True)
        z_means, z_vars = tf.split(encoder_output, num_or_size_splits=2, axis=1)
        z = []
        for i in range(len(data)):
            z.append(tf.random.normal((1, var.threeD_gan_z_size), z_means[i], tf.pow(z_vars[i], 0.5)))
        z = tf.convert_to_tensor(z)
        z = tf.reshape(z, (len(data), 1, 1, 1, var.threeD_gan_z_size))
        generated_shapes2 = generator(inputs=z, training=True)

        gen_loss = generator_loss(output_from_noise, real_shapes, generated_shapes2)
        print(f"Generator loss: {gen_loss}")
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        G_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

