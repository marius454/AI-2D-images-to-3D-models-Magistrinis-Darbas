import tensorflow as tf
import os
import variables as var
from time import sleep
import numpy as np
from random import randrange
import matplotlib.pyplot as plt

def load_single_image(image_path: str):
    print('load image')
    image = tf.io.read_file(image_path)
    if image_path.endswith(".jpg"):
        image = tf.io.decode_jpeg(image, channels=3)
    elif image_path.endswith(".png"):
        image = tf.io.decode_png(image, channels=3)

    return image

def show_single_image(image):
    print('show image')
    plt.figure(figsize=(10, 5))

    plt.title('Image')
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def load_multiple_images():
    print('load_multiple_images')

