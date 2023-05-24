import numpy as np
import tensorflow as tf
import dataIO as d
import backups.custom_layers as custom_layers
import os
import time
import random
import data_processing as dp

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

        self.conv_block_1 = blocks.GNConv3DBlock(32, input_shape=(64, 64, 64, 64), use_interpolation=False)
        self.conv_block_2 = blocks.GNConv3DBlock(16, use_interpolation=False)
        self.conv_block_3 = blocks.GNConv3DBlock(1, activation='tanh', use_norm=False, use_ReLU=False, use_interpolation=False)

    def call(self, inputs):
        x = self.conv_block_1(inputs) # (None, 64, 64, 64, 32)
        x = self.conv_block_2(x) # (None, 64, 64, 64, 16)
        x = self.conv_block_3(x) # (None, 64, 64, 64, 1)

        return x
    
    def build_graph(self):
        x = tf.keras.Input(shape=(64, 64, 64, 64))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


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
        self.conv_block_7 = blocks.GNConv3DBlock(1, activation='tanh', use_norm=False, use_ReLU=False, use_interpolation=False)

        # G^L
        self.sub_G = Sub_Generator()

    def call(self, x, r = None):
        # G^A
        x = self.fully_connected(x)
        x = tf.keras.layers.Reshape((4, 4, 4, 512))(x)

        x = self.conv_block_1(x) # (None, 8, 8, 8, 512)
        x = self.conv_block_2(x) # (None, 16, 16, 16, 512)
        x = self.conv_block_3(x) # (None, 32, 32, 32, 256)
        x = self.conv_block_4(x) # (None, 64, 64, 64, 128)
        x_latent = self.conv_block_5(x) # (None, 64, 64, 64, 64)

        if (self.mode == 'train' or self.mode == 'eval_small'):
            x_small = self.sub_G(x_latent) # (None, 64, 64, 64, 1)
            if (r != None):
                # x = x_latent[:, :, r//4 : r//4+8, :, :] # Crop out (64, 8, 64, 64) by y axis, in the paper it is the x axis
                x = x_latent[:, r//4 : r//4+8, :, :, :] # Crop out (8, 64, 64, 64) by x axis
        else:
            x = x_latent

        if (r != None or self.mode == 'eval'):
            # G^H
            x = self.interpolate(x) # (None, 128, 128, 128, 64) or (None, 128, 16, 128, 64)
            x = self.conv_block_6(x) # (None, 256, 256, 256, 32) or (None, 256, 32, 256, 32) 
            x = self.conv_block_7(x) # (None, 256, 256, 256, 1) or (None, 256, 32, 256, 1)
        
        if (r != None and self.mode == 'train'):
            return x, x_small
        elif (r == None and (self.mode == 'train' or self.mode == 'eval_small')):
            return x_small
        return x
    
    def build_graph(self):
        x = tf.keras.Input(shape=(64, 64, 64, 64))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


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
        x = self.conv_block_1(inputs) # (None, 32, 32, 32, 32)
        x = self.conv_block_2(x) # (None, 16, 16, 16, 64)
        x = self.conv_block_3(x) # (None, 8, 8, 8, 128)
        x = self.conv_block_4(x) # (None, 4, 4, 4, 256)
        x = self.conv_block_5(x) # (None, 1, 1, 1, 1)

        return tf.keras.layers.Flatten()(x)


class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        '''
            input shape = (32, 256, 256, 1), (64, 64, 64, 1)
            kernel size = {4, 4, 4, 2x4x4, 2x4x4, 1x4x4}
            strides = {2, 2, 2, 2, 1, 1, 1}
        '''
        super(Discriminator, self).__init__()
        self.conv_block_1 = blocks.SNConv3DBlock(16, input_shape=(32, 256, 256, 1))
        self.conv_block_2 = blocks.SNConv3DBlock(32)
        self.conv_block_3 = blocks.SNConv3DBlock(64)
        self.conv_block_4 = blocks.SNConv3DBlock(128, kernel_size=(2,4,4))
        ## IN THE CODE
        self.conv_block_5 = blocks.SNConv3DBlock(256, kernel_size=(2,4,4))
        self.conv_block_6 = blocks.SNConv3DBlock(512, kernel_size=(1,4,4), strides=(1,2,2))
        self.conv_block_7 = blocks.SNConv3DBlock(128, kernel_size=(1,4,4), strides=1, padding = 'valid')
        ## IN THE PAPER
        # self.conv_block_5 = blocks.SNConv3DBlock(256, kernel_size=(2,4,4), strides=1)
        # self.conv_block_6 = blocks.SNConv3DBlock(512, kernel_size=(1,4,4), strides=1)
        # self.conv_block_7 = blocks.SNConv3DBlock(128, kernel_size=(1,4,4), strides=1, padding = 'valid')

        self.fully_connceted_1 = blocks.SNDenseBlock(64)
        ## IN THE CODE
        self.fully_connceted_2 = blocks.SNDenseBlock(1)
        ## IN THE PAPER
        # self.fully_connceted_2 = blocks.SNDenseBlock(32)
        # self.fully_connceted_3 = tf.keras.layers.Dense(1)

        # D^L
        self.sub_D = Sub_Discriminator()

    def call (self, x, x_small, r = None):
        x = self.conv_block_1(x) # (batch_size, 128, 16, 128, 16)
        x = self.conv_block_2(x) # (None, 64, 8, 64, 32)
        x = self.conv_block_3(x) # (None, 32, 4, 32, 64)
        x = self.conv_block_4(x) # (None, 16, 2, 16, 128)
        x = self.conv_block_5(x) # (None, 8, 1, 8, 256)
        x = self.conv_block_6(x) # (None, 4, 1, 4, 512)
        x = self.conv_block_7(x) # (None, 1, 1, 1, 128)
        x = tf.keras.layers.Flatten()(x) # (None, 128)
        ## IN THE CODE
        x = tf.keras.layers.concatenate([x, r / 224. * tf.ones([tf.shape(x)[0], 1])], axis = -1) # (None, 129)
        
        x = self.fully_connceted_1(x) # (None, 64)
        ## IN THE CODE
        x_logit = self.fully_connceted_2(x) # (None, 1)
        ## IN THE PAPER
        # x = self.fully_connceted_2(x) # (None, 32)
        # x_logit = self.fully_connceted_3(x) # (None, 1)

        x_small_logit = self.sub_D(x_small)

        return (x_logit + x_small_logit) / 2.
    
    def build_graph(self):
        x = tf.keras.Input(shape=(32, 256, 256, 1))
        x_small = tf.keras.Input(shape=(64, 64, 64, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x, x_small, 100))
    


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
        # self.conv_block_2 = blocks.Conv2DBlock(32, kernel_size=3, strides=1)
        self.conv_block_3 = blocks.Conv2DBlock(64)
        self.conv_block_4 = blocks.Conv2DBlock(32)
        self.conv_block_5 = blocks.Conv2DBlock(64)
        self.conv_block_6 = blocks.Conv2DBlock(128)
        self.conv_block_7 = blocks.Conv2DBlock(256)
        self.conv_block_8 = blocks.Conv2DBlock(1024, strides=1, padding='valid', use_norm=False, use_ReLU=False)

    def call(self, x):
        x = self.conv_block_1(x) # (None, 128, 128, 32)
        # x = self.conv_block_2(x) # (None, 128, 128, 32)
        x = self.conv_block_3(x) # (None, 64, 64, 64)
        x = self.conv_block_4(x) # (None, 32, 32, 32)
        x = self.conv_block_5(x) # (None, 16, 16, 64)
        x = self.conv_block_6(x) # (None, 8, 8, 128)
        x = self.conv_block_7(x) # (None, 4, 4, 256)

        # self.z_means = tf.keras.layers.Dense(opt.latent_dim, name = 'z_means')
        # self.z_vars = tf.keras.layers.Dense(opt.latent_dim, name = 'z_vars')
        # self.z = tf.keras.layers.Lambda(self.sampling, name='z')

        x = self.conv_block_8(x) # (None, 1, 1, 1024)
        return tf.keras.layers.Flatten()(x)
    
    def sampling(self, args):
        z_means, z_vars = args
        eps = tf.random.normal(shape = (tf.shape(z_means)[0], opt.latent_dim))
        z = eps * tf.exp(z_vars * .5) + z_means
        return z

    def build_graph(self):
        x = tf.keras.Input(shape=(opt.image_res, opt.image_res, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class HA_GAN(tf.keras.Model):
    def __init__(self, mode = 'train'):
        super(HA_GAN, self).__init__()
        self.discriminator = Discriminator()
        self.generator = Generator(mode = mode)
        self.encoder = Encoder()

        self.BCE_with_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # self.recon_loss = tf.keras.losses.MeanAbsoluteError()
        self.recon_loss = tf.keras.losses.MeanSquaredError()
        self.KL_loss = tf.keras.losses.KLDivergence()
        
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
        self.enc_loss_tracker = tf.keras.metrics.Mean(name="enc_loss")
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="gen_loss")
        # self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")

    @property
    def metrics(self):
        return [
            self.disc_loss_tracker,
            self.enc_loss_tracker,
            self.gen_loss_tracker,
            # self.kl_loss_tracker,
            self.recon_loss_tracker,
        ]
    
    def call(self, inputs, use_encoder = True):
        if (use_encoder):
            z = self.encoder(inputs, training = False)
            generated_shapes = self.generator(z, training = False)
        else:
            noise = tf.random.normal([1, opt.latent_dim])
            generated_shapes = self.generator(noise, training = False)
        return generated_shapes
    
    def compile(self, d_optimizer, g_optimizer, e_optimizer):
        super(HA_GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.e_optimizer = e_optimizer


    def train_step(self, data):
        # Data will be a tensorflow dataset containing ([image_list], 3d_model_256, 3d_model_64) groups
        # train_step is called for every batch of data
        if (opt.use_eager_mode and opt.print_progress):
            print('')
            print('------------------------------------------')

        images, real_shapes_small, real_shapes = data
        batch_size = tf.shape(real_shapes)[0]
        r = np.random.randint(0,opt.shape_res*7/8+1)

        real_shapes_crop = real_shapes[:, r : r + opt.shape_res//8, :, :] # crop (32, 256, 256).
        real_shapes_crop = tf.cast(real_shapes_crop, tf.float32) 
        real_shapes_crop = (real_shapes_crop * 2) - 1 # transform into range [-1, 1]
        real_shapes_crop = tf.expand_dims(real_shapes_crop, -1)

        real_shapes_small = tf.cast(real_shapes_small, tf.float32) #
        real_shapes_small = (real_shapes_small * 2) - 1 # transform into range [-1, 1]
        real_shapes_small = tf.expand_dims(real_shapes_small, -1)

        if (opt.add_noise_to_input_shapes):
            real_shapes_crop += 0.005 * tf.random.uniform(tf.shape(real_shapes_crop), minval=-1, maxval=1)
            real_shapes_small += 0.005 * tf.random.uniform(tf.shape(real_shapes_small), minval=-1, maxval=1)
        

        labels_real = tf.ones((batch_size, 1))
        labels_fake = tf.zeros((batch_size, 1))
        if (opt.add_noise_to_discriminator_labels):
            labels_real += 0.005 * tf.random.uniform(tf.shape(labels_real), minval=-1, maxval=1)
            labels_fake += 0.005 * tf.random.uniform(tf.shape(labels_fake), minval=-1, maxval=1)

        ## TRAINING WITHOUT RECON LOSS IN GENERATOR
        # disc_accuracy = self.train_discriminator(real_shapes_crop, real_shapes_small, batch_size, labels_real, labels_fake, r)
        # gen_accuracy = self.train_generator(batch_size, labels_real, r)
        # self.train_encoder(images, real_shapes_small)
        # # self.train_encoder(images, real_shapes_small, real_shapes_crop, r)

        ## TRAINING WITH RECON LOSS IN GENERATOR
        disc_accuracy = self.train_discriminator(real_shapes_crop, real_shapes_small, batch_size, labels_real, labels_fake, r)
        self.train_encoder(images, real_shapes_small, batch_size = batch_size)
        gen_accuracy = self.train_generator(images, real_shapes_small, batch_size, labels_real, r)

        if (opt.use_eager_mode and opt.print_progress):
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
            # "kl_loss": self.kl_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "g_lr": self.g_optimizer.learning_rate,
            "d_lr": self.d_optimizer.learning_rate,
            "e_lr": self.e_optimizer.learning_rate,
        }
    
    def test_step(self, data):
        images, real_shapes_small, real_shapes = data

        if (opt.use_eager_mode and opt.print_progress):
            print('')
            print('------------------------------------------')

        real_shapes = tf.cast(real_shapes, tf.float32) 
        real_shapes = (real_shapes * 2) - 1 # transform into range [-1, 1]
        real_shapes = tf.expand_dims(real_shapes, -1)

        generator_mode = self.generator.mode 
        self.generator.mode = "eval"

        enc_loss = 0
        for i in range(opt.e_max_iter):
            image = images[:, i]
            z = self.encoder(image, training = False)
            generated_shapes = self.generator(z, training = False)
            # enc_loss = enc_loss + (self.recon_loss(real_shapes, generated_shapes) * opt.lambda_1)
            enc_loss = enc_loss + metrics.reconstruction_loss(real_shapes, generated_shapes)

        enc_loss = enc_loss / tf.cast(tf.shape(images)[1], tf.float32) * opt.lambda_1
        self.generator.mode = generator_mode

        if (opt.use_eager_mode and opt.print_progress):
            print (f"val encoder loss: {enc_loss}")
            print('------------------------------------------')
            print('')

        return {
            "overall_loss": enc_loss
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


    # @tf.function
    # def train_generator(self, batch_size, labels_real, r):
        # for iteration in range(opt.g_iter):
        #     noise = tf.random.normal([batch_size, opt.latent_dim])

        #     with tf.GradientTape() as gen_tape:
        #         generated_shapes_crop, generated_shapes_small = self.generator(noise, r)
        #         # predictions_fake = self.discriminator(generated_shapes_crop, generated_shapes_small) ## IN THE PAPER
        #         predictions_fake = self.discriminator(generated_shapes_crop, generated_shapes_small, r) ## IN THE CODE

        #         gen_loss = self.BCE_with_logits(labels_real, predictions_fake)

        #     self.gen_loss_tracker.update_state(gen_loss)
        
        #     gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        #     self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
        # gen_accuracy = metrics.generator_accuracy(predictions_fake)
        # return gen_accuracy
    @tf.function
    def train_generator(self, images, real_shapes_small, batch_size, labels_real, r):
        for iteration in range(opt.g_iter):
            encoder_index = np.random.choice(range(opt.e_max_iter), size=1, replace=False)
            noise = tf.random.normal([batch_size, opt.latent_dim])

            with tf.GradientTape() as gen_tape:
                generated_shapes_crop, generated_shapes_small = self.generator(noise, r)
                # predictions_fake = self.discriminator(generated_shapes_crop, generated_shapes_small) ## IN THE PAPER
                predictions_fake = self.discriminator(generated_shapes_crop, generated_shapes_small, r) ## IN THE CODE
                gen_loss_bce = self.BCE_with_logits(labels_real, predictions_fake)

                z = self.encoder(images[:, encoder_index[0]])
                encoded_shapes_small = self.generator(z)
                # gen_loss_recon = self.recon_loss(real_shapes_small, encoded_shapes_small)
                gen_loss_recon = metrics.reconstruction_loss(real_shapes_small, encoded_shapes_small)
                gen_loss = gen_loss_bce + opt.lambda_1*gen_loss_recon

            self.gen_loss_tracker.update_state(gen_loss)
            self.recon_loss_tracker.update_state(gen_loss_recon)
        
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
        gen_accuracy = metrics.generator_accuracy(predictions_fake)
        return gen_accuracy

    @tf.function
    def train_encoder(self, images, real_shapes_small, real_shapes_crop = None, r = None, batch_size = None):
        # index_list = np.random.choice(range(opt.e_max_iter), size=opt.e_iter, replace=False)
        # for i in index_list:
        #     with tf.GradientTape() as enc_tape:
        #         image = images[:, i]
        #         z = self.encoder(image)
        #         if (r == None or real_shapes_crop == None):
        #             generated_shapes_small = self.generator(z)
        #             enc_loss = self.recon_loss(real_shapes_small, generated_shapes_small) * opt.lambda_1
        #         else:
        #             generated_shapes_crop, generated_shapes_small = self.generator(z, r)
        #             enc_loss_crop = self.recon_loss(real_shapes_crop, generated_shapes_crop)
        #             enc_loss_small = self.recon_loss(real_shapes_small, generated_shapes_small)
        #             enc_loss = ((enc_loss_crop + enc_loss_small) / 2) * opt.lambda_1

        #     gradients_of_encoder = enc_tape.gradient(enc_loss, self.encoder.trainable_variables)
        #     self.e_optimizer.apply_gradients(zip(gradients_of_encoder, self.encoder.trainable_variables))

        #     self.enc_loss_tracker.update_state(enc_loss)

        index_list = np.random.choice(range(opt.e_max_iter), size=opt.e_iter, replace=False)
        with tf.GradientTape() as enc_tape:
            enc_loss_total = 0
            for i in index_list:
                image = images[:, i]
                z = self.encoder(image)
                # if (batch_size != None):
                #     noise = tf.random.normal([batch_size, opt.latent_dim])
                #     kl_loss = abs(self.KL_loss(noise, z))
                #     enc_loss_total = enc_loss_total + opt.kl_weight*kl_loss

                if (r == None or real_shapes_crop == None):
                    encoded_shapes_small = self.generator(z)
                    # enc_loss = self.recon_loss(real_shapes_small, encoded_shapes_small)
                    enc_loss = metrics.reconstruction_loss(real_shapes_small, encoded_shapes_small)
                else:
                    encoded_shapes_crop, encoded_shapes_small = self.generator(z, r)
                    # enc_loss_crop = self.recon_loss(real_shapes_crop, encoded_shapes_crop)
                    # enc_loss_small = self.recon_loss(real_shapes_small, encoded_shapes_small)
                    enc_loss_crop = metrics.reconstruction_loss(real_shapes_crop, encoded_shapes_crop)
                    enc_loss_small = metrics.reconstruction_loss(real_shapes_small, encoded_shapes_small)
                    enc_loss = ((enc_loss_crop + enc_loss_small) / 2)

                enc_loss_total = enc_loss_total + opt.lambda_1*enc_loss

            enc_loss_total = enc_loss_total / opt.e_iter


        gradients_of_encoder = enc_tape.gradient(enc_loss_total, self.encoder.trainable_variables)
        self.e_optimizer.apply_gradients(zip(gradients_of_encoder, self.encoder.trainable_variables))

        self.enc_loss_tracker.update_state(enc_loss_total)
        # self.kl_loss_tracker.update_state(kl_loss)
        self.recon_loss_tracker.update_state(enc_loss)


                # for j in range(tf.shape(images)[0].numpy()):
                #     print ("ploting image and shape")
                #     real_shape = tf.math.greater_equal(real_shapes_small[j, :, :, :, 0], 0)
                #     dp.show_image_and_shape(image[j].numpy(), real_shape.numpy(), (32, 256, 256))



    # def train_encoder(self, images, real_shapes_crop, real_shapes_small, r):
    #     with tf.GradientTape() as enc_tape:
    #         enc_loss = 0
    #         for i in range(tf.shape(images)[1]):
    #             z = self.encoder(images[:, i])
    #             encoded_shapes_crop, encoded_shapes_small = self.generator(z, r)

    #             enc_loss = enc_loss + (self.recon_loss(real_shapes_crop, encoded_shapes_crop) 
    #                                    + self.recon_loss(real_shapes_small, encoded_shapes_small))
    #         enc_loss = enc_loss / tf.shape(images)[1]
        
    #     self.enc_loss_tracker.update_state(enc_loss)

    #     gradients_of_encoder = enc_tape.gradient(enc_loss, self.encoder.trainable_variables)
    #     self.e_optimizer.apply_gradients(zip(gradients_of_encoder, self.encoder.trainable_variables))

