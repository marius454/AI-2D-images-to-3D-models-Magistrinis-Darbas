import numpy as np
import tensorflow as tf
import data_processing as dp

import ThreeD_gan.threeD_gan_options as opt

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if ((epoch + 1) % opt.display_callback_frequency == 0):
            print("Loading image")
            image = dp.load_single_image('./Data/ShapeNetSem/screenshots/db80fbb9728e5df343f47bfd2fc426f7/db80fbb9728e5df343f47bfd2fc426f7-7.png')
            image = tf.reshape(image, (-1, 256, 256, 3))
            image = dp.normalize_image(image)

            print("Generating 3D model")
            generated_shape = self.model.predict(image)
            generated_shape = tf.math.greater_equal(generated_shape, 0.5)
            # print(generated_shape[0, :, :, :, 0].numpy())
            
            print("Ploting 3D model")
            dp.plot_3d_model(generated_shape[0, :, :, :, 0].numpy(), (64, 64, 64))