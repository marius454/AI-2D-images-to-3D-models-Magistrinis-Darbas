import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
import time
import string

import file_processing as fp

image = fp.load_single_image('chair.png')
fp.show_single_image(image)