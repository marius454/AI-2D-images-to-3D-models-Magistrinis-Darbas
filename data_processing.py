import tensorflow as tf
import os
import variables as var
from time import sleep
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
import binvox_rw as bv
import scipy.io
from pathlib import Path
import random

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
    plt.imshow(tf.cast(image, tf.uint8))
    plt.axis('off')
    plt.show()

def load_multiple_images():
    print('load_multiple_images')

def decode_jpeg(image, resize_method='bilinear'):
    image = tf.io.decode_jpeg(image, channels=3)
    return tf.image.resize(image, [var.imageHeight, var.imageWidth], method=resize_method)
    # image = tf.image.resize(image, [var.imageHeight, var.imageWidth], method=resize_method)
    # return tf.cast(image, tf.uint8)

def decode_png(image, resize_method='bilinear'):
    image = tf.io.decode_png(image, channels=3)
    return tf.image.resize(image, [var.imageHeight, var.imageWidth], method=resize_method)
    # image = tf.image.resize(image, [var.imageHeight, var.imageWidth], method=resize_method)
    # return tf.cast(image, tf.uint8)

def plot_3d_model(model, dimensions):
    """
    Provide a 3 dimensional array to plot in matplotlib
    Provide dimension size as a list or tuple of three values eg. (64, 64, 64)
    """
    x, y, z = np.indices((dimensions[0]+1, dimensions[1]+1, dimensions[0]+1))
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
    """
    data_dict = {}
    for file in os.listdir(directory):
        filepath = directory + file
        if directory[-1] != "/":
            filepath = directory + "/" + file
        data = load_file_mat(filepath)
        data_dict[file] = data
    return data_dict

def get_shape_code_list(filepath):
    shape_codes = []
    with open(filepath) as file:
        for line in file:
            shape_codes.append(line.rstrip())
    
    return shape_codes

def get_shapes(shape_codes, directory):
    shapes = {}
    for file in os.listdir(directory):
        if Path(file).stem in shape_codes:
            print ("getting shapes: " + Path(file).stem)
            filepath = directory + file
            if directory[-1] != "/":
                filepath = directory + "/" + file
            with open(filepath, "rb") as f:
                shapes[Path(file).stem] = bv.read_as_3d_array(f)
    return shapes

def get_shape_screenshots(shape_codes, directory):
    shape_screenshots = {}
    for folder in os.listdir(directory):
        if folder in shape_codes:
            print ("getting screenshots: " + folder)
            images = []
            folderpath = directory + folder
            if directory[-1] != "/":
                folderpath = directory + "/" + folder

            for i in [6, 7, 9, 10, 11, 13]:
                images.append(decode_png(tf.io.read_file(f"{folderpath}/{folder}-{i}.png")))
            shape_screenshots[folder] = images
    
    return shape_screenshots








# def get_shape_screenshots(shape_codes, directory):
#     shape_screenshots = {}
#     for folder in os.listdir(directory):
#         if folder in shape_codes:
#             print ("getting screenshots: " + folder)
#             images = []
#             folderpath = directory + folder
#             if directory[-1] != "/":
#                 folderpath = directory + "/" + folder
#             for file in random.sample(os.listdir(folderpath), 6):
#                 if file.endswith(".png") or file.endswith(".jpg"):
#                     image = tf.io.read_file(folderpath + "/" + file)
#                     if file.endswith(".png"):
#                         images.append(decode_png(image))
#                     elif file.endswith(".jpg"):
#                         images.append(decode_jpeg(image))

#             shape_screenshots[folder] = images
    
#     return shape_screenshots
