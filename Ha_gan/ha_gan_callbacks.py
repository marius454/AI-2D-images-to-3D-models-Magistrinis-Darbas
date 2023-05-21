import numpy as np
import tensorflow as tf
import data_processing as dp

import Ha_gan.ha_gan_options as opt

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if ((epoch + 1) % opt.display_callback_frequency == 0):
            print("Loading image")
            image = dp.load_single_image(f'./Data/ShapeNetSem/table_screenshots/db80fbb9728e5df343f47bfd2fc426f7/db80fbb9728e5df343f47bfd2fc426f7-7.png', image_res = opt.image_res)
            image = dp.normalize_image(image)
            image = tf.reshape(image, (-1, opt.image_res, opt.image_res, 3))

            print("Loading real 3D model")
            real_shape = dp.get_shape(f"{opt.shapes256_dir}/db80fbb9728e5df343f47bfd2fc426f7.binvox")

            print("Generating 3D model")
            self.model.generator.mode = 'eval'
            generated_shape = self.model(image)
            self.model.generator.mode = 'train'
            generated_shape = tf.math.greater_equal(generated_shape, -0.2)
            # print(generated_shape[0, :, :, :, 0].numpy())
            
            print("Ploting image and models")
            image = tf.reshape(image, (opt.image_res, opt.image_res, 3))
            # dp.show_image_and_shapes(image, real_shape.data, generated_shape[0, :, :, :, 0].numpy(), real_shape.dims)
            dp.show_image_and_shape(image, generated_shape[0, :, :, :, 0].numpy(), real_shape.dims)
