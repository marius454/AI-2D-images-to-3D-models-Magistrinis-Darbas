import numpy as np
import tensorflow as tf
import data_processing as dp

import ThreeD_gan.threeD_gan_options as opt

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if ((epoch + 1) % opt.display_callback_frequency == 0):
            print("Loading image")
            image = dp.load_single_image(f'./Data/ShapeNetSem/table_screenshots/db80fbb9728e5df343f47bfd2fc426f7/db80fbb9728e5df343f47bfd2fc426f7-7.png', image_res = opt.image_res)
            image = dp.normalize_image(image)
            image = tf.reshape(image, (-1, opt.image_res, opt.image_res, 3))

            print("Loading real 3D model")
            real_shape = dp.get_shape(f"{opt.shape_data_dir}/db80fbb9728e5df343f47bfd2fc426f7.binvox")

            print("Generating 3D model")
            generated_shape = self.model.predict(image)
            generated_shape = tf.math.greater_equal(generated_shape, 0.5)
            # print(generated_shape[0, :, :, :, 0].numpy())
            
            print("Ploting image and models")
            image = tf.reshape(image, (opt.image_res, opt.image_res, 3))
            dp.show_image_and_shapes(image, real_shape.data, generated_shape[0, :, :, :, 0].numpy(), real_shape.dims)

class LR_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if ((epoch + 1) < 5):
            self.model.g_optimizer.learning_rate = opt.generator_lr[0]
            self.model.d_optimizer.learning_rate = opt.discriminator_lr[0]
            self.model.e_optimizer.learning_rate = opt.encoder_lr[0]
        elif ((epoch + 1) < 20):
            self.model.g_optimizer.learning_rate = opt.generator_lr[1]
            self.model.d_optimizer.learning_rate = opt.discriminator_lr[1]
            self.model.e_optimizer.learning_rate = opt.encoder_lr[1]
        elif ((epoch + 1) < 50):
            self.model.g_optimizer.learning_rate = opt.generator_lr[2]
            self.model.d_optimizer.learning_rate = opt.discriminator_lr[2]
            self.model.e_optimizer.learning_rate = opt.encoder_lr[2]
        elif ((epoch + 1) < 80):
            self.model.g_optimizer.learning_rate = opt.generator_lr[3]
            self.model.d_optimizer.learning_rate = opt.discriminator_lr[3]
            self.model.e_optimizer.learning_rate = opt.encoder_lr[3]