import numpy as np
import tensorflow as tf
import dataIO as d
import backups.custom_layers as custom_layers
import os
import time

import Ha_gan.ha_gan_blocks as blocks
import Ha_gan.ha_gan_options as opt
import Ha_gan.ha_han_metrics as metrics

# G^L
class Sub_Generator(tf.keras.layers.Layer):
    def __init__(self):
        '''
            input shape = (64 x 64 x 64 x 64)
            kernel size = 3 x 3 x 3
            strides = {1, 1, 1, 1, 1}
        '''
        super(Sub_Generator, self).__init__()

        self.conv_block_1 = blocks.Conv3DBlock(32, input_shape=(64, 64, 64, 64))
        self.conv_block_2 = blocks.Conv3DBlock(16)
        self.conv_block_3 = blocks.Conv3DBlock(1, activation='tanh')

    def call(self, inputs):
        x = self.conv_block_1(inputs, use_interpolation=False)
        assert tf.shape(x) == (None, 64, 64, 64, 32)
        x = self.conv_block_2(x, use_interpolation=False)
        assert tf.shape(x) == (None, 64, 64, 64, 16)
        x = self.conv_block_3(x, use_group_norm=False, use_ReLU=False, use_interpolation=False)
        assert tf.shape(x) == (None, 64, 64, 64, 1)

        return x


class Generator(tf.keras.layers.Layer):
    def __init__(self, mode='train', latent_dim=opt.latent_dim):
        '''
            input shape = (1, 1024)
            kernel size = 3 x 3 x 3
            strides = {1, 1, 1, 1, 1}
        '''
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.mode = mode

        # G^A
        self.fully_connected = tf.keras.layers.Dense(4*4*4*512)
        self.conv_block_1 = blocks.GNConv3DBlock(512)
        self.conv_block_2 = blocks.GNConv3DBlock(512)
        self.conv_block_3 = blocks.GNConv3DBlock(256)
        self.conv_block_4 = blocks.GNConv3DBlock(128)
        self.conv_block_5 = blocks.GNConv3DBlock(64, use_interpolation=False)

        # G^H
        self.interpolate = tf.keras.layers.UpSampling3D(2)
        self.conv_block_6 = blocks.GNConv3DBlock(32)
        self.conv_block_7 = blocks.GNConv3DBlock(1, activation='tanh', use_group_norm=False, use_ReLU=False, use_interpolation=False)

        # G^L
        self.sub_G = Sub_Generator()

    def call(self, x, r = None):
        if (r != None or self.mode == 'eval'):
            # G^A
            x = self.fully_connected(x)
            x = tf.keras.layers.Reshape((-1, 4, 4, 4, 512))(x)

            x = self.conv_block_1(x)
            assert tf.shape(x) == (None, 8, 8, 8, 512)
            x = self.conv_block_2(x)
            assert tf.shape(x) == (None, 16, 16, 16, 512)
            x = self.conv_block_3(x)
            assert tf.shape(x) == (None, 32, 32, 32, 256)
            x = self.conv_block_4(x)
            assert tf.shape(x) == (None, 64, 64, 64, 128)
            x_latent = self.conv_block_5(x)
            assert tf.shape(x_latent) == (None, 64, 64, 64, 64)

            if (self.mode == 'train'):
                x_small = self.sub_G(x_latent)
                x = x_latent[:, :, r//4 : r//4+8, :, :] # Crop out (64, 8, 64) curretly from y axis, in the paper it is the x axis (because torch uses channels_first)
            else:
                x = x_latent

        # G^H
        x = self.interpolate(x)
        assert tf.shape(x) == (None, 128, 128, 128, 64) or tf.shape(x) == (None, 128, 16, 128, 64)
        x = self.conv_block_6(x)
        assert tf.shape(x) == (None, 256, 256, 256, 32) or tf.shape(x) == (None, 256, 32, 256, 32) 
        x = self.conv_block_7(x)
        assert tf.shape(x) == (None, 256, 256, 256, 1) or tf.shape(x) == (None, 256, 32, 256, 1)
        
        if (r != None) and self.mode == 'train':
            return x, x_small
        return x


# D^L
class Sub_Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        '''
            input shape = (64, 64, 64, 1)
            kernel size = 4 x 4 x 4
            strides = {2, 2, 2, 2, 1}
        '''
        super(Sub_Discriminator, self).__init__()

        self.conv_block_1 = blocks.SNConv3DBlock(32, input_shape=(64, 64, 64, 1))
        self.conv_block_2 = blocks.SNConv3DBlock(64)
        self.conv_block_3 = blocks.SNConv3DBlock(128)
        self.conv_block_4 = blocks.SNConv3DBlock(256)
        self.conv_block_5 = blocks.SNConv3DBlock(1, strides = 1, padding='valid', use_norm=False, use_ReLU=False)
    
    def call (self, inputs):
        x = self.conv_block_1(inputs)
        assert tf.shape(x) == (None, 32, 32, 32, 32)
        x = self.conv_block_2(x)
        assert tf.shape(x) == (None, 16, 16, 16, 64)
        x = self.conv_block_3(x)
        assert tf.shape(x) == (None, 8, 8, 8, 128)
        x = self.conv_block_4(x)
        assert tf.shape(x) == (None, 4, 4, 4, 256)
        x = self.conv_block_5(x)
        assert tf.shape(x) == (None, 1, 1, 1, 1)

        return tf.keras.layers.Reshape((-1, 1))(x)


class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        '''
            input shape = (32, 256, 256, 1)
            kernel size = {4, 4, 4, 2x4x4, 2x4x4, 1x4x4}
            strides = {2, 2, 2, 2, 1, 1, 1}
        '''
        super(Discriminator, self).__init__()
        self.conv_block_1 = blocks.SNConv3DBlock(16, input_shape=(256, 32, 256, 1))
        self.conv_block_2 = blocks.SNConv3DBlock(32)
        self.conv_block_3 = blocks.SNConv3DBlock(64)
        self.conv_block_4 = blocks.SNConv3DBlock(128, kernel_size=(4,2,4))
        ## IN THE CODE
        self.conv_block_5 = blocks.SNConv3DBlock(256, kernel_size=(4,2,4))
        self.conv_block_6 = blocks.SNConv3DBlock(512, kernel_size=(4,1,4), strides=(2,1,2))
        self.conv_block_7 = blocks.SNConv3DBlock(128, kernel_size=(4,1,4), strides=1, padding = 'valid')
        ## IN THE PAPER
        # self.conv_block_5 = blocks.SNConv3DBlock(256, kernel_size=(4,2,4), strides=1)
        # self.conv_block_6 = blocks.SNConv3DBlock(512, kernel_size=(4,1,4), strides=1)
        # self.conv_block_7 = blocks.SNConv3DBlock(128, kernel_size=(4,1,4), strides=1, padding = 'valid')

        self.fully_connceted_1 = blocks.SNDenseBlock(64)
        ## IN THE CODE
        self.fully_connceted_2 = blocks.SNDenseBlock(1)
        ## IN THE PAPER
        # self.fully_connceted_2 = blocks.SNDenseBlock(32)
        # self.fully_connceted_3 = tf.keras.layers.Dense(1)

        # D^L
        self.sub_D = Sub_Discriminator()

    def call (self, x, x_small, r = None):
        x = self.conv_block_1(x)
        assert tf.shape(x) == (None, 128, 16, 128, 16)
        x = self.conv_block_2(x)
        assert tf.shape(x) == (None, 64, 8, 64, 32)
        x = self.conv_block_3(x)
        assert tf.shape(x) == (None, 32, 4, 32, 64)
        x = self.conv_block_4(x)
        assert tf.shape(x) == (None, 16, 2, 16, 128)
        x = self.conv_block_5(x)
        assert tf.shape(x) == (None, 8, 1, 8, 256)
        x = self.conv_block_6(x)
        assert tf.shape(x) == (None, 4, 1, 4, 512)
        x = self.conv_block_7(x)
        assert tf.shape(x) == (None, 1, 1, 1, 128)
        x = tf.keras.layers.Flatten()(x)
        ## IN THE CODE
        x = tf.keras.layers.Concatenate(-1)[x, r / 224. * tf.ones([tf.shape(x)[0], 1])]
        
        x = self.fully_connceted_1(x)
        assert tf.shape(x) == (None, 64)
        ## IN THE CODE
        x_logit = self.fully_connceted_2(x)
        assert tf.shape(x) == (None, 1)
        ## IN THE PAPER
        # x = self.fully_connceted_2(x)
        # assert tf.shape(x) == (None, 32)
        # x_logit = self.fully_connceted_3(x)
        # assert tf.shape(x) == (None, 1)

        x_small_logit = self.sub_D(x_small)

        return (x_logit + x_small_logit) / 2.

class Encoder(tf.keras.layers.Layer):
    # Input: (-1, 256, 256, 3)
    # GNconv1: (128, 128, 32)
    # GNconv2: (128, 128, 32)
    # GNconv3: (64, 64, 64)
    # GNconv4: (32, 32, 32)
    # GNconv5: (16, 16, 64)
    # GNconv6: (8, 8, 128)
    # GNconv7: (4, 4, 256)
    # conv2d: (1, 1, 1024)

    def __init__(self):
        '''
            input shape = (256, 256, 3)
            kernel size = {4, 3, 4, 4, 4, 4, 4, 4}
            strides = {2, 1, 2, 2, 2, 2, 2, 1}
        '''
        super(Encoder, self).__init__()
        self.conv_block_1 = blocks.Conv2DBlock(32, input_shape=(256, 256, 3))
        self.conv_block_2 = blocks.Conv2DBlock(32, kernel_size=3, strides=1)
        self.conv_block_3 = blocks.Conv2DBlock(64)
        self.conv_block_4 = blocks.Conv2DBlock(32)
        self.conv_block_5 = blocks.Conv2DBlock(64)
        self.conv_block_6 = blocks.Conv2DBlock(128)
        self.conv_block_7 = blocks.Conv2DBlock(256)
        self.conv_block_8 = blocks.Conv2DBlock(1024, strides=1, padding='valid', use_norm=False, use_ReLU=False)

    def call(self, x):
        x = self.conv_block_1(x)
        assert tf.shape(x) == (None, 128, 128, 32)
        x = self.conv_block_2(x)
        assert tf.shape(x) == (None, 128, 128, 32)
        x = self.conv_block_3(x)
        assert tf.shape(x) == (None, 64, 64, 64)
        x = self.conv_block_4(x)
        assert tf.shape(x) == (None, 32, 32, 32)
        x = self.conv_block_5(x)
        assert tf.shape(x) == (None, 16, 16, 64)
        x = self.conv_block_6(x)
        assert tf.shape(x) == (None, 8, 8, 128)
        x = self.conv_block_7(x)
        assert tf.shape(x) == (None, 4, 4, 256)
        x = self.conv_block_8(x)
        assert tf.shape(x) == (None, 1, 1, 1024)

        return tf.keras.layers.Flatten()(x)



class HA_GAN(tf.keras.Model):
    def __init__(self):
        super(HA_GAN, self).__init__()
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.encoder = Generator()

        self.BCE_with_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.MAE_loss = tf.keras.losses.MeanAbsoluteError()
        
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
        z = self.encoder(inputs)
        self.generator(z)
        return self.generator(z)
    
    def compile(self, d_optimizer, g_optimizer, e_optimizer):
        super(HA_GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.e_optimizer = e_optimizer


    def train_step(self, data):
        # Data will be a tensorflow dataset containing ([image_list], 3d_model_256, 3d_model_64) groups
        # train_step is called for every batch of data
        if (opt.use_eager_mode):
            print('')
            print('------------------------------------------')

        images, real_shapes, real_shapes_small = data
        batch_size = tf.shape(real_shapes)[0]
        # for i in range(tf.shape(images)[0].numpy()):
        #     dp.show_image_and_shape(images[i][1].numpy(), real_shapes[i].numpy(), (64, 64, 64))

        real_shapes = tf.cast(real_shapes, tf.float32)
        real_shapes = tf.expand_dims(real_shapes, -1)
        if (opt.add_noise_to_input_shapes):
            real_shapes += 0.05 * tf.random.uniform(tf.shape(real_shapes), minval=-1, maxval=1)
        real_shapes_crop = real_shapes[:, :, r : r + opt.shape_res//8, :, :] # crop (256, 32, 256).

        labels_real = tf.ones((batch_size, 1))
        labels_fake = tf.zeros((batch_size, 1))
        if (opt.add_noise_to_discriminator_labels):
            labels_real += 0.05 * tf.random.uniform(tf.shape(labels_real), minval=-1, maxval=1)
            labels_fake += 0.05 * tf.random.uniform(tf.shape(labels_fake), minval=-1, maxval=1)
        r = tf.random.uniform(shape=[1], minval=0, maxval=opt.shape_res*7/8+1)

        disc_accuracy = self.train_discriminator(real_shapes_crop, real_shapes_small, batch_size, labels_real, labels_fake, r)
        gen_accuracy = self.train_generator(batch_size, labels_fake, r)
        self.train_encoder(images, real_shapes_crop, real_shapes_small, r)

        if (opt.use_eager_mode):
            print (f"Overall val loss: {self.disc_loss_tracker.result() + self.enc_loss_tracker.result() + self.gen_loss_tracker.result()}")
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

    @tf.function
    def train_discriminator(self, real_shapes_crop, real_shapes_small, batch_size, labels_real, labels_fake, r):
        noise = tf.random.normal([batch_size, opt.latent_dim])

        with tf.GradientTape() as disc_tape:
            # predictions_real = self.discriminator(real_shapes_crop, real_shapes_small) ## IN THE PAPER
            predictions_real = self.discriminator(real_shapes_crop, real_shapes_small, r) ## IN THE CODE
            disc_loss_real = self.BCE_with_logits(labels_real, predictions_real)

            generated_shapes_crop, generated_shapes_small = self.generator(noise, r)
            # predictions_fake = self.discriminator(generated_shapes_crop, generated_shapes_small) ## IN THE PAPER
            predictions_fake = self.discriminator(generated_shapes_crop, generated_shapes_small, r) ## IN THE CODE
            disc_loss_fake = self.BCE_with_logits(labels_fake, predictions_fake)
            disc_loss = disc_loss_real + disc_loss_fake

        self.disc_loss_tracker.update_state(disc_loss)
        disc_accuracy = metrics.discriminator_accuracy(predictions_real, predictions_fake)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return disc_accuracy


    @tf.function
    def train_generator(self, batch_size, labels_fake, r):
        for iteration in range(opt.g_iter):
            noise = tf.random.normal([batch_size, opt.latent_dim])

            with tf.GradientTape() as gen_tape:
                generated_shapes_crop, generated_shapes_small = self.generator(noise, r)
                # predictions_fake = self.discriminator(generated_shapes_crop, generated_shapes_small) ## IN THE PAPER
                predictions_fake = self.discriminator(generated_shapes_crop, generated_shapes_small, r) ## IN THE CODE

                gen_loss = self.BCE_with_logits(labels_fake, predictions_fake)

            self.gen_loss_tracker.update_state(gen_loss)
        
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
        gen_accuracy = metrics.generator_accuracy(predictions_fake)
        return gen_accuracy

    @tf.function
    def train_encoder(self, images, real_shapes_crop, real_shapes_small, r):
        with tf.GradientTape() as enc_tape:
            enc_loss = 0
            for i in range(tf.shape(images)[1]):
                z = self.encoder(images[:, i])
                encoded_shapes_crop, encoded_shapes_small = self.generator(z, r)

                enc_loss = enc_loss + (self.MAE_loss(real_shapes_crop, encoded_shapes_crop) 
                                       + self.MAE_loss(real_shapes_small, encoded_shapes_small))
            enc_loss = enc_loss / tf.shape(images)[1]
        
        self.enc_loss_tracker.update_state(enc_loss)

        gradients_of_encoder = enc_tape.gradient(enc_loss, self.encoder.trainable_variables)
        self.e_optimizer.apply_gradients(zip(gradients_of_encoder, self.encoder.trainable_variables))


