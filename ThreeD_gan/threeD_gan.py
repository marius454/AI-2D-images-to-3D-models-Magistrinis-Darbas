import numpy as np
import tensorflow as tf
import dataIO as d
import backups.custom_layers as custom_layers
import os
import time
import gc

import ThreeD_gan.threeD_gan_blocks as blocks
import ThreeD_gan.threeD_gan_metrics as metrics
import ThreeD_gan.threeD_gan_helpers as helpers
import ThreeD_gan.threeD_gan_options as opt
import data_processing as dp

from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
import tensorflow_probability as tfp

        
class Generator(tf.keras.layers.Layer):
    def __init__(self):
        '''
            channels = {512, 256, 128, 64, 1}
            input shape = (1, 1, 1, z_size)
            kernel size = 4 x 4 x 4
            strides = {1, 2, 2, 2, 2}
        '''
        super(Generator, self).__init__()
        self.block_1 = blocks.Conv3DTransposeBlock(512, input_shape=(1, 1, 1, opt.z_size), strides=1, padding = 'valid')
        self.block_2 = blocks.Conv3DTransposeBlock(256)
        self.block_3 = blocks.Conv3DTransposeBlock(128)
        self.block_4 = blocks.Conv3DTransposeBlock(64)
        self.block_5 = blocks.Conv3DTransposeBlock(1, activation=opt.discriminator_activation)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x, use_batch_norm=False, use_ReLU=False)
        return x


class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        '''
            channels = {64, 128, 256, 512, 1}
            input shape = (shape_res, shape_res, shape_res, 1)
            kernel size = 4 x 4 x 4
            strides = {2, 2, 2, 2, 1}
        '''
        super(Discriminator, self).__init__()
        self.block_1 = blocks.Conv3DBlock(64, input_shape=(opt.shape_res, opt.shape_res, opt.shape_res, 1))
        self.block_2 = blocks.Conv3DBlock(128)
        self.block_3 = blocks.Conv3DBlock(256)
        self.block_4 = blocks.Conv3DBlock(512)
        self.block_5 = blocks.Conv3DBlock(1, strides=1, padding = 'valid', activation=opt.generator_activation)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x, use_batch_norm=False, use_ReLU=False)
        return tf.keras.layers.Flatten()(x)

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        '''
            channels = {64, 128, 256, 512, 400}
            input shape = (image_res, image_res, 3)
            kernel size = {11x11, 5x5, 5x5, 5x5, 8x8}
            strides = {4, 2, 2, 2, 1}
        '''
        super(Encoder, self).__init__()
        self.block_1 = blocks.Conv2DBlock(64, input_shape=(opt.image_res, opt.image_res, 3), kernel_size = 11, strides = 4)
        self.block_2 = blocks.Conv2DBlock(128)
        self.block_3 = blocks.Conv2DBlock(256)
        self.block_4 = blocks.Conv2DBlock(512)
        self.block_5 = blocks.Conv2DBlock(400, kernel_size = 8, strides = 1, padding = 'valid')

        self.z_means = tf.keras.layers.Dense(200, name = 'z_means')
        self.z_vars = tf.keras.layers.Dense(200, name = 'z_vars')
        self.z = tf.keras.layers.Lambda(self.sampling, name='z')


    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = tf.keras.layers.Flatten()(x)

        z_means = self.z_means(x)
        z_vars = self.z_vars(x)
        z = self.z([z_means, z_vars])
        
        return z_means, z_vars, z
    
    def sampling(self, args):
        z_means, z_vars = args
        eps = tf.random.normal(shape = (tf.shape(z_means)[0], 200))
        z = eps * tf.exp(z_vars * .5) + z_means
        return tf.reshape(z, (tf.shape(z_means)[0], 1, 1, 1, 200))

    

class ThreeD_gan(tf.keras.Model):
    def __init__(self):
        super(ThreeD_gan, self).__init__()
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.encoder = Encoder()

        self.disc_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.gen_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
        self.enc_loss_tracker = tf.keras.metrics.Mean(name="enc_loss")
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="gen_loss")

    @property
    def metrics(self):
        return [
            self.disc_loss_tracker,
            self.enc_loss_tracker,
            self.gen_loss_tracker,
        ]

    def call(self, inputs):
        z_means, z_vars, z = self.encoder(inputs)
        return self.generator(z)
    
    def compile(self, d_optimizer, g_optimizer, e_optimizer):
        super(ThreeD_gan, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.e_optimizer = e_optimizer

    def train_step(self, data):
        # Data will be a tensorflow dataset containing (image, 3d_model) pairs
        # train_step is called for every batch of data
        if (opt.use_eager_mode):
            print('')
            print('------------------------------------------')

        images, real_shapes = data
        # for i in range(tf.shape(images)[0].numpy()):
        #     dp.show_image_and_shape(images[i].numpy(), real_shapes[i].numpy(), (64, 64, 64))
        real_shapes = tf.cast(real_shapes, tf.float32)
        real_shapes = tf.expand_dims(real_shapes, -1)

        if (opt.add_noise_to_input_shapes):
            real_shapes += 0.05 * tf.random.uniform(tf.shape(real_shapes), minval=-1, maxval=1)
        
        batch_size = tf.shape(images)[0]
        noise = helpers.get_z(0, 1, batch_size)
        # noise = tf.random.normal([batch_size, opt.z_size])
        # noise = tf.reshape(noise, (batch_size, 1, 1, 1, opt.z_size))

        disc_accuracy = self.train_step1(noise, real_shapes, batch_size)
        self.train_step2(images, real_shapes)
        gen_accuracy = self.train_step3(noise, images, real_shapes)

        if (opt.use_eager_mode):
            print (f"Overall train loss: {self.disc_loss_tracker.result() + self.enc_loss_tracker.result() + self.gen_loss_tracker.result()}")
            print('------------------------------------------')
            print('')

        return {
            "disc_loss": self.disc_loss_tracker.result(),
            "disc_accuracy": disc_accuracy,
            "enc_loss": self.enc_loss_tracker.result(),
            "gen_loss": self.gen_loss_tracker.result(),
            "gen_accuracy": gen_accuracy,
            "overall_loss": self.disc_loss_tracker.result() + self.enc_loss_tracker.result() + self.gen_loss_tracker.result(),
            "g_lr": self.g_optimizer.learning_rate,
            "d_lr": self.d_optimizer.learning_rate,
            "e_lr": self.e_optimizer.learning_rate,
        }

    def test_step(self, data):
        images, real_shapes = data
        real_shapes = tf.cast(real_shapes, tf.float32)
        real_shapes = tf.expand_dims(real_shapes, -1)

        batch_size = tf.shape(images)[0]
        noise = helpers.get_z(0, 1, batch_size)
        # noise = tf.random.normal([batch_size, opt.z_size])
        # noise = tf.reshape(noise, (batch_size, 1, 1, 1, opt.z_size))

        generated_shapes = self.generator(inputs=noise, training=False)
        predictions_real = self.discriminator(inputs = real_shapes, training = False)
        predictions_fake = self.discriminator(inputs = generated_shapes, training = False)
        disc_loss = metrics.discriminator_loss(self.disc_loss_fn, predictions_real, predictions_fake, batch_size)

        z_means, z_vars, z = self.encoder(inputs=images, training=False)
        generated_shapes = self.generator(inputs=z, training=False)
        enc_loss = metrics.vae_loss(z_means, z_vars, real_shapes, generated_shapes)
        
        if (opt.use_eager_mode):
            print (f"Overall val loss: {disc_loss + enc_loss}")
            print('------------------------------------------')
            print('')
            
        return {
            "disc_loss": disc_loss,
            "enc_loss": enc_loss,
            "overall_loss": disc_loss + enc_loss,
        }


    @tf.function
    def train_step1(self, noise, real_shapes, batch_size):
        with tf.GradientTape() as disc_tape:
            generated_shapes = self.generator(inputs=noise, training=True)
            predictions_real = self.discriminator(inputs = real_shapes, training = True)
            predictions_fake = self.discriminator(inputs = generated_shapes, training = True)
            disc_loss = metrics.discriminator_loss(self.disc_loss_fn, predictions_real, predictions_fake, batch_size)

        disc_accuracy = metrics.discriminator_accuracy(predictions_real, predictions_fake)
        self.disc_loss_tracker.update_state(disc_loss)

        if (opt.use_eager_mode):
            print(f"mean predictions_real: {tf.math.reduce_mean(predictions_real[:, 0]).numpy()}")
            print(f"mean predictions_fake: {tf.math.reduce_mean(predictions_fake[:, 0]).numpy()}")
            print(f"DISCRIMINATOR ACCURACY: {disc_accuracy}")
            print('')

        if (disc_accuracy < opt.discriminator_training_threshold):
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return disc_accuracy


    @tf.function
    def train_step2(self, images, real_shapes):
        with tf.GradientTape() as enc_tape:
            z_means, z_vars, z = self.encoder(inputs=images, training=True)
            generated_shapes = self.generator(inputs=z, training=True)

            enc_loss = metrics.vae_loss(z_means, z_vars, real_shapes, generated_shapes)

        gradients_of_encoder = enc_tape.gradient(enc_loss, self.encoder.trainable_variables)
        self.e_optimizer.apply_gradients(zip(gradients_of_encoder, self.encoder.trainable_variables))
        
        self.enc_loss_tracker.update_state(enc_loss)

    @tf.function
    def train_step3(self, noise, images, real_shapes):
        if (opt.use_eager_mode):
            print('')

        with tf.GradientTape() as gen_tape:
            shapes_from_noise = self.generator(inputs=noise, training=True)
            output_from_noise = self.discriminator(inputs=shapes_from_noise, training=True)
            del shapes_from_noise

            z_means, z_vars, z = self.encoder(inputs=images, training=True)
            generated_shapes = self.generator(inputs=z, training=True)
            gen_loss = metrics.generator_loss(self.gen_loss_fn, output_from_noise, real_shapes, generated_shapes)

        gen_accuracy = metrics.generator_accuracy(output_from_noise)
        self.gen_loss_tracker.update_state(gen_loss)
        if (opt.use_eager_mode):
            print(f"mean output_from_noise: {tf.math.reduce_mean(output_from_noise[:, 0]).numpy()}")
            print(f"GENERATOR_ACCURACY: {gen_accuracy}")

        # if (gen_accuracy < 0.9):
        #     gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        #     self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
        return gen_accuracy









# import numpy as np
# import tensorflow as tf
# import dataIO as d
# import backups.custom_layers as custom_layers
# import os
# import time
# import gc

# import ThreeD_gan.threeD_gan_blocks as blocks
# import ThreeD_gan.threeD_gan_metrics as metrics
# import ThreeD_gan.threeD_gan_helpers as helpers
# import ThreeD_gan.threeD_gan_options as opt
# import data_processing as dp

# from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
# import tensorflow_probability as tfp

        
# class Generator(tf.keras.layers.Layer):
#     def __init__(self):
#         '''
#             channels = {512, 256, 128, 64, 1}
#             input shape = (1, 1, 1, z_size)
#             kernel size = 4 x 4 x 4
#             strides = {1, 2, 2, 2, 2}
#         '''
#         super(Generator, self).__init__()
#         self.block_1 = blocks.Conv3DTransposeBlock(512, input_shape=(1, 1, 1, opt.z_size), strides=1, padding = 'valid', name = 'conv3d_transpose__batch_norm__ReLU_1')
#         self.block_2 = blocks.Conv3DTransposeBlock(256, name = 'conv3d_transpose__batch_norm__ReLU_2')
#         self.block_3 = blocks.Conv3DTransposeBlock(128, name = 'conv3d_transpose__batch_norm__ReLU_3')
#         self.block_4 = blocks.Conv3DTransposeBlock(64, name = 'conv3d_transpose__batch_norm__ReLU_4')
#         self.block_5 = blocks.Conv3DTransposeBlock(1, activation=opt.discriminator_activation, use_batch_norm=False, use_ReLU=False, name = 'conv3d_transpose__sigmoid')

#     def call(self, inputs):
#         x = self.block_1(inputs)
#         x = self.block_2(x)
#         x = self.block_3(x)
#         x = self.block_4(x)
#         x = self.block_5(x)
#         return x
    
#     def build_graph(self):
#         x = tf.keras.Input(shape=(1, 1, 1, opt.z_size))
#         return tf.keras.Model(inputs=[x], outputs=self.call(x))


# class Discriminator(tf.keras.layers.Layer):
#     def __init__(self):
#         '''
#             channels = {64, 128, 256, 512, 1}
#             input shape = (shape_res, shape_res, shape_res, 1)
#             kernel size = 4 x 4 x 4
#             strides = {2, 2, 2, 2, 1}
#         '''
#         super(Discriminator, self).__init__()
#         self.block_1 = blocks.Conv3DBlock(64, input_shape=(opt.shape_res, opt.shape_res, opt.shape_res, 1), name = 'conv3d__batch_norm__leakyReLU_1')
#         self.block_2 = blocks.Conv3DBlock(128, name = 'conv3d__batch_norm__leakyReLU_2')
#         self.block_3 = blocks.Conv3DBlock(256, name = 'conv3d__batch_norm__leakyReLU_3')
#         self.block_4 = blocks.Conv3DBlock(512, name = 'conv3d__batch_norm__leakyReLU_4')
#         self.block_5 = blocks.Conv3DBlock(1, strides=1, padding = 'valid', activation=opt.generator_activation, use_batch_norm=False, use_ReLU=False, name = 'conv3d__sigmoid')

#     def call(self, inputs):
#         x = self.block_1(inputs)
#         x = self.block_2(x)
#         x = self.block_3(x)
#         x = self.block_4(x)
#         x = self.block_5(x)
#         return tf.keras.layers.Flatten()(x)
    
#     def build_graph(self):
#         x = tf.keras.Input(shape=(opt.shape_res, opt.shape_res, opt.shape_res, 1))
#         return tf.keras.Model(inputs=[x], outputs=self.call(x))

# class Encoder(tf.keras.layers.Layer):
#     def __init__(self):
#         '''
#             channels = {64, 128, 256, 512, 400}
#             input shape = (image_res, image_res, 3)
#             kernel size = {11x11, 5x5, 5x5, 5x5, 8x8}
#             strides = {4, 2, 2, 2, 1}
#         '''
#         super(Encoder, self).__init__()
#         self.block_1 = blocks.Conv2DBlock(64, input_shape=(opt.image_res, opt.image_res, 3), kernel_size = 11, strides = 4, name = 'conv2d__batch_norm__ReLU_1')
#         self.block_2 = blocks.Conv2DBlock(128, name = 'conv2d__batch_norm__ReLU_2')
#         self.block_3 = blocks.Conv2DBlock(256, name = 'conv2d__batch_norm__ReLU_3')
#         self.block_4 = blocks.Conv2DBlock(512, name = 'conv2d__batch_norm__ReLU_4')
#         self.block_5 = blocks.Conv2DBlock(400, kernel_size = 8, strides = 1, padding = 'valid', name = 'conv2d')

#         self.z_means = tf.keras.layers.Dense(200, name = 'dense_z_means')
#         self.z_vars = tf.keras.layers.Dense(200, name = 'dense_z_vars')
#         self.z = tf.keras.layers.Lambda(self.sampling, name='sampling_z')


#     def call(self, inputs):
#         x = self.block_1(inputs)
#         x = self.block_2(x)
#         x = self.block_3(x)
#         x = self.block_4(x)
#         x = self.block_5(x)
#         x = tf.keras.layers.Flatten()(x)

#         z_means = self.z_means(x)
#         z_vars = self.z_vars(x)
#         z = self.z([z_means, z_vars])
        
#         return z_means, z_vars, z
    
#     def sampling(self, args):
#         z_means, z_vars = args
#         eps = tf.random.normal(shape = (tf.shape(z_means)[0], 200))
#         z = eps * tf.exp(z_vars * .5) + z_means
#         return tf.reshape(z, (tf.shape(z_means)[0], 1, 1, 1, 200))
    
#     def build_graph(self):
#         x = tf.keras.Input(shape=(opt.image_res, opt.image_res, 3))
#         return tf.keras.Model(inputs=[x], outputs=self.call(x))

    

# class ThreeD_gan(tf.keras.Model):
#     def __init__(self):
#         super(ThreeD_gan, self).__init__()
#         self.discriminator = Discriminator()
#         self.generator = Generator()
#         self.encoder = Encoder()

#         self.disc_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
#         self.gen_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

#         self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
#         self.enc_loss_tracker = tf.keras.metrics.Mean(name="enc_loss")
#         self.gen_loss_tracker = tf.keras.metrics.Mean(name="gen_loss")

#     @property
#     def metrics(self):
#         return [
#             self.disc_loss_tracker,
#             self.enc_loss_tracker,
#             self.gen_loss_tracker,
#         ]

#     def call(self, inputs):
#         z_means, z_vars, z = self.encoder(inputs)
#         return self.generator(z)
    
#     def compile(self, d_optimizer, g_optimizer, e_optimizer):
#         super(ThreeD_gan, self).compile()
#         self.g_optimizer = g_optimizer
#         self.d_optimizer = d_optimizer
#         self.e_optimizer = e_optimizer

#     def train_step(self, data):
#         # Data will be a tensorflow dataset containing (image, 3d_model) pairs
#         # train_step is called for every batch of data
#         if (opt.use_eager_mode):
#             print('')
#             print('------------------------------------------')

#         images, real_shapes = data
#         # for i in range(tf.shape(images)[0].numpy()):
#         #     dp.show_image_and_shape(images[i].numpy(), real_shapes[i].numpy(), (64, 64, 64))
#         real_shapes = tf.cast(real_shapes, tf.float32)
#         real_shapes = tf.expand_dims(real_shapes, -1)

#         if (opt.add_noise_to_input_shapes):
#             real_shapes += 0.05 * tf.random.uniform(tf.shape(real_shapes), minval=-1, maxval=1)
        
#         batch_size = tf.shape(images)[0]
#         noise = helpers.get_z(0, 1, batch_size)
#         # noise = tf.random.normal([batch_size, opt.z_size])
#         # noise = tf.reshape(noise, (batch_size, 1, 1, 1, opt.z_size))

#         disc_accuracy = self.train_step1(noise, real_shapes, batch_size)
#         self.train_step2(images, real_shapes)
#         gen_accuracy = self.train_step3(noise, images, real_shapes)

#         if (opt.use_eager_mode):
#             print (f"Overall val loss: {self.disc_loss_tracker.result() + self.enc_loss_tracker.result() + self.gen_loss_tracker.result()}")
#             print('------------------------------------------')
#             print('')

#         return {
#             "disc_loss": self.disc_loss_tracker.result(),
#             "disc_accuracy": disc_accuracy,
#             "enc_loss": self.enc_loss_tracker.result(),
#             "gen_loss": self.gen_loss_tracker.result(),
#             "gen_accuracy": gen_accuracy,
#             "overall_loss": self.disc_loss_tracker.result() + self.enc_loss_tracker.result() + self.gen_loss_tracker.result(),
#             "g_lr": self.g_optimizer.learning_rate,
#             "d_lr": self.d_optimizer.learning_rate,
#             "e_lr": self.e_optimizer.learning_rate,
#         }

#     def test_step(self, data):
#         images, real_shapes = data
#         real_shapes = tf.cast(real_shapes, tf.float32)
#         real_shapes = tf.expand_dims(real_shapes, -1)

#         batch_size = tf.shape(images)[0]
#         noise = helpers.get_z(0, 1, batch_size)
#         # noise = tf.random.normal([batch_size, opt.z_size])
#         # noise = tf.reshape(noise, (batch_size, 1, 1, 1, opt.z_size))

#         generated_shapes = self.generator(inputs=noise, training=False)
#         predictions_real = self.discriminator(inputs = real_shapes, training = False)
#         predictions_fake = self.discriminator(inputs = generated_shapes, training = False)
#         disc_loss = metrics.discriminator_loss(self.disc_loss_fn, predictions_real, predictions_fake, batch_size)

#         z_means, z_vars, z = self.encoder(inputs=images, training=False)
#         generated_shapes = self.generator(inputs=z, training=False)
#         enc_loss = metrics.vae_loss(z_means, z_vars, real_shapes, generated_shapes)
        
#         if (opt.use_eager_mode):
#             print (f"Overall val loss: {disc_loss + enc_loss}")
#             print('------------------------------------------')
#             print('')
            
#         return {
#             "disc_loss": disc_loss,
#             "enc_loss": enc_loss,
#             "overall_loss": disc_loss + enc_loss,
#         }


#     @tf.function
#     def train_step1(self, noise, real_shapes, batch_size):
#         with tf.GradientTape() as disc_tape:
#             generated_shapes = self.generator(inputs=noise, training=True)
#             predictions_real = self.discriminator(inputs = real_shapes, training = True)
#             predictions_fake = self.discriminator(inputs = generated_shapes, training = True)
#             disc_loss = metrics.discriminator_loss(self.disc_loss_fn, predictions_real, predictions_fake, batch_size)

#         disc_accuracy = metrics.discriminator_accuracy(predictions_real, predictions_fake)
#         self.disc_loss_tracker.update_state(disc_loss)

#         if (opt.use_eager_mode):
#             print(f"mean predictions_real: {tf.math.reduce_mean(predictions_real[:, 0]).numpy()}")
#             print(f"mean predictions_fake: {tf.math.reduce_mean(predictions_fake[:, 0]).numpy()}")
#             print(f"DISCRIMINATOR ACCURACY: {disc_accuracy}")
#             print('')

#         if (disc_accuracy < opt.discriminator_training_threshold):
#             gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
#             self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
#         return disc_accuracy


#     @tf.function
#     def train_step2(self, images, real_shapes):
#         with tf.GradientTape() as enc_tape:
#             z_means, z_vars, z = self.encoder(inputs=images, training=True)
#             generated_shapes = self.generator(inputs=z, training=True)

#             enc_loss = metrics.vae_loss(z_means, z_vars, real_shapes, generated_shapes)

#         gradients_of_encoder = enc_tape.gradient(enc_loss, self.encoder.trainable_variables)
#         self.e_optimizer.apply_gradients(zip(gradients_of_encoder, self.encoder.trainable_variables))
        
#         self.enc_loss_tracker.update_state(enc_loss)

#     @tf.function
#     def train_step3(self, noise, images, real_shapes):
#         if (opt.use_eager_mode):
#             print('')

#         with tf.GradientTape() as gen_tape:
#             shapes_from_noise = self.generator(inputs=noise, training=True)
#             output_from_noise = self.discriminator(inputs=shapes_from_noise, training=True)
#             del shapes_from_noise

#             z_means, z_vars, z = self.encoder(inputs=images, training=True)
#             generated_shapes = self.generator(inputs=z, training=True)
#             gen_loss = metrics.generator_loss(self.gen_loss_fn, output_from_noise, real_shapes, generated_shapes)

#         gen_accuracy = metrics.generator_accuracy(output_from_noise)
#         self.gen_loss_tracker.update_state(gen_loss)
#         if (opt.use_eager_mode):
#             print(f"mean output_from_noise: {tf.math.reduce_mean(output_from_noise[:, 0]).numpy()}")
#             print(f"GENERATOR_ACCURACY: {gen_accuracy}")

#         # if (gen_accuracy < 0.9):
#         #     gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
#         #     self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

#         gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
#         self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
#         return gen_accuracy






    
