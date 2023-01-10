import tensorflow as tf
import variables as var

class Sampling(tf.keras.layers.Layer):
    """Samples an n dimmentional latent vector z from an n*2 dimmensional Gaussian latent space."""

    def call(self, inputs):
        # tf.split axis is the dimension to split along, so if we have a tensor of shape (None, 400)
        # where None is the batch number and 400 is the vector size, we split along the vector size
        assert len(inputs.shape) == 2
        z_mean, z_var = tf.split(inputs, num_or_size_splits=2, axis=1)
        z = tf.random.normal((1, var.threeD_gan_z_size), z_mean, tf.pow(z_var, 0.5))
        return z

class Split(tf.keras.layers.Layer):
    """Splits input tensor into two parts"""

    def call(self, inputs):
        # tf.split axis is the dimension to split along, so if we have a tensor of shape (None, 400)
        # where None is the batch number and 400 is the vector size, we split along the vector size
        assert len(inputs.shape) == 2
        z_mean, z_var = tf.split(inputs, num_or_size_splits=2, axis=1)
        
        return z_mean, z_var

class Threshold(tf.keras.layers.Layer):
    """Thresholds tensor values into bool values"""

    def call(self, inputs, threshold = 0.5):
        outputs = tf.where(inputs > threshold, 1.0, 0.0)
        
        return outputs