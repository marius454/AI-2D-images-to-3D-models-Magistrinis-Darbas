import numpy as np
import tensorflow as tf
import dataIO as d
import custom_layers
import os
import time
import gc


from .. import variables as var
from .. import data_processing as dp
import threeD_gan_blocks as blocks

from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
import tensorflow_probability as tfp

class Generator(tf.keras.Layer):
    def __init__(self):
        super(Generator, self).__init__()
        self.block_1 = blocks.Conv3DTransposeBlock()

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()

class ThreeD_gan(tf.keras.Model):
    def __init__(self):
        super(ThreeD_gan, self).__init__()