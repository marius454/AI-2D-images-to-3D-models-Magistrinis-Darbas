import tensorflow as tf
import os
import variables as var
from time import sleep
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
import binvox_rw as bv
import scipy.io

def load_single_image(imagePath: str):
    print('load image')
    image = tf.io.read_file(imagePath)
    if imagePath.endswith(".jpg"):
        image = decode_jpeg(image)
    elif imagePath.endswith(".png"):
        image = decode_png(image)

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

def decode_jpeg(image):
    image = tf.io.decode_jpeg(image, channels=3)
    return tf.image.resize(image, [var.imageHeight, var.imageWidth])

def decode_png(image):
    image = tf.io.decode_png(image, channels=3)
    return tf.image.resize(image, [var.imageHeight, var.imageWidth])

def plot_3d_model(model):
    """Provide a 3 dimensional array to plot in matplotlib"""
    x, y, z = np.indices((65, 65, 65))
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(x, y, z, model)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.set_aspect('equal')
    plt.show()


def load_file_mat(filepath):
    """Load a .mat file into a numpy array"""
    data = scipy.io.loadmat(filepath)
    data = np.array(data['voxel'])
    data = np.where(data == 0, False, True)
    return data

def load_directory_mat(directory):
    """
    Load all .mat files from a directory into an list of numpy arrays
    directory path must end with a '/'
    """
    data_dict = {}
    for file in os.listdir(directory):
        data = load_file_mat(directory + file)
        data_dict[file] = data
    return data_dict


