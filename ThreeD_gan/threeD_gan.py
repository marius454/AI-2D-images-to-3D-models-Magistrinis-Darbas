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

class threeD_gan(tf.keras.Model):
    def __init__(self):
        super(threeD_gan, self).__init__()