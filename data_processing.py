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
from skimage.transform import resize

def load_single_image(imagePath: str):
    """Load image file into code as tensor."""
    print('load image')
    image = tf.io.read_file(imagePath)
    if imagePath.endswith(".jpg"):
        image = decode_jpeg(image)
    elif imagePath.endswith(".png"):
        image = decode_png(image)

    return image

def show_single_image(image):
    """Display a single 2D image using matplotlib."""
    print('show image')
    plt.figure(figsize=(10, 5))

    plt.title('Image')
    plt.imshow(tf.cast(image, tf.uint8))
    plt.axis('off')
    plt.show()



def decode_jpeg(image, resize_method='bilinear'):
    """Load jpeg image into a tensor."""
    image = tf.io.decode_jpeg(image, channels=3)
    return tf.image.resize(image, [var.imageHeight, var.imageWidth], method=resize_method)

def decode_png(image, resize_method='bilinear'):
    """Load png image into a tensor."""
    image = tf.io.decode_png(image, channels=3)
    return tf.image.resize(image, [var.imageHeight, var.imageWidth], method=resize_method)

def normalize_image(image):
    """Normalize 8 bit image into range [0, 1]."""
    image = tf.cast(image, tf.float32) / 255.0
    return image

# def add_sample_weights(shape, weight):
#     """Add weights to positions with voxels in the ground truth shapes"""
#     # Set class weights
#     class_weights = tf.constant([var.threeD_gan_background_weight, var.threeD_gan_voxel_weight])
#     # Normalize to range [0, 1]
#     class_weights = tf.cast((class_weights/tf.reduce_sum(class_weights)), dtype=tf.float16)
#     # Create sample weights
#     sample_weights = tf.gather(class_weights, indices=tf.cast(shape, tf.int32))



def plot_3d_model(model, dimensions = (128, 128, 128)):
    """
    Plots a 3D model using matplotlib (binvox formal highly preferable, the code will not receive an error but might not work).

    model - a 3 dimensional array to plot in matplotlib.
    dimensions - list or tuple of x,y,z axis sizes for the object space eg. (64, 64, 64).
    """
    x, y, z = np.indices((dimensions[0]+1, dimensions[1]+1, dimensions[2]+1))
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(x, y, z, model)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.set_aspect('equal')
    plt.show()


# TODO make sure this is functional
def show_image_and_shape(image, real_shape, encoder, generator, dimensions = (128, 128, 128)):
    """Show the image of an object, it's ground trush 3D model, and the generated model."""
    plt.figure(figsize=(10, 5))

    title = ['Input Image', 'True shape', 'Generated shape']
    x, y, z = np.indices((dimensions[0]+1, dimensions[1]+1, dimensions[2]+1))

    plt.subplot(1, len(title), 1)
    plt.title('Input Image')
    plt.imshow(tf.keras.utils.array_to_img(image))
    plt.axis('off')

    plt.subplot(1, len(title), 2)
    plt.title(title[0])
    plt.voxels(x, y, z, real_shape)
    plt.axis('off')

    z = encoder(image)
    generated_shape = generator(z, training = False)

    plt.subplot(1, len(title), 2)
    plt.title(title[0])
    plt.voxels(x, y, z, generated_shape)
    plt.axis('off')
        
    plt.show()



def load_file_mat(filepath):
    """Load a .mat file into a numpy array."""
    data = scipy.io.loadmat(filepath)
    data = np.array(data['voxel'])
    data = np.where(data == 0, False, True)
    return data

def load_directory_mat(directory):
    """Load all .mat files from a directory into an list of numpy arrays."""
    data_dict = {}
    for file in os.listdir(directory):
        filepath = directory + file
        if directory[-1] != "/":
            filepath = directory + "/" + file
        data = load_file_mat(filepath)
        data_dict[file] = data
    return data_dict



def get_shape_code_list(filepath):
    """Load a list of ShapeNet shape codes from a .txt or .csv file."""
    shape_codes = []
    with open(filepath) as file:
        for line in file:
            shape_codes.append(line.rstrip())
    return shape_codes


def get_shapes(shape_codes, directory, downscale_factor = None):
    """Load a dictionary a 3D models according to their shape codes."""
    shapes = {}
    if directory[-1] != "/":
        directory = directory + "/"

    for file in os.listdir(directory):
        if Path(file).stem in shape_codes:
            print ("getting shapes: " + Path(file).stem)
            with open(directory + file, "rb") as f:
                shapes[Path(file).stem] = bv.read_as_3d_array(f)
                if downscale_factor:
                    shapes[Path(file).stem] = downscale_binvox(shapes[Path(file).stem], factor=downscale_factor)

    return shapes

def get_shape_screenshots(shape_codes, directory):
    """
    Load a dictionary of lists containing images that render a shape referenced by a shape code.
    Assumed that ShapeNet image data is kept in PNG format.
    """
    shape_screenshots = {}
    if directory[-1] != "/":
        directory = directory + "/"

    for folder in os.listdir(directory):
        if folder in shape_codes:
            print ("getting screenshots: " + folder)
            images = []

            # Load images from the selections of angles in range [1, 13]
            for i in [6, 7, 8, 9, 10, 11, 12, 13]:
                image = decode_png(tf.io.read_file(f"{directory + folder}/{folder}-{i}.png"))
                image = normalize_image(image)
                images.append(image)
            shape_screenshots[folder] = images
    
    return shape_screenshots


# TODO Will have to make this more portable, currently other users will have to set up datasets themselves
def load_data(dataset = 'shapenet_tables', downscale_factor = None):
    """
    Load shape and image data from a predifened shape code file into a tensor dataset

    `shapenet_tables` - load dataset of 436 table and desk objects collected based on their metadata wnsynet codes\n
    `shapenet_tables2` - load dataset of 677 table and desk objects collected based on their metadata categories\n
    `shapenet_limited_tables` - load a subset of shapenet_tables with a limited amount of entries (currently 300, but might change)\n
    `shapenet_single_table` - load a single table object from the shapenet_tables set
    """
    datasets = ['shapenet_tables', 'shapenet_tables2', 'shapenet_limited_tables', 'shapenet_single_table'] 
    if dataset not in datasets:
        raise Exception('undefined dataset name given to load_data()')

    if dataset == "shapenet_tables":
        result = load_shapenet_data("./Data/ShapeNetSem/Table.csv", downscale_factor)
        return result
    if dataset == "shapenet_tables2":
        result = load_shapenet_data("./Data/ShapeNetSem/Table2.csv", downscale_factor)
        return result
    if dataset == "shapenet_limited_tables":
        result = load_shapenet_data("./Data/ShapeNetSem/limited_table.csv", downscale_factor)
        return result
    if dataset == "shapenet_single_table":
        result = load_shapenet_data("./Data/ShapeNetSem/single_table.csv", downscale_factor)
        return result


def load_shapenet_data(codes_directory, downscale_factor = None):
    """Load shapenet data into (shape, image) pairs tensor."""
    shape_codes = get_shape_code_list(codes_directory)
    shapes = get_shapes(shape_codes, "./Data/ShapeNetSem/models-binvox-custom/", downscale_factor)
    shape_screenshots = get_shape_screenshots(shape_codes, "./Data/ShapeNetSem/screenshots/")

    print("\nFormating data as (image, shape) pairs")
    inputs = tf.nest.flatten(shape_screenshots.values())
    labels = []
    for code in shape_codes:
        shape_data = np.array(shapes[code].data)
        for i in range(len(shape_screenshots[code])):
            labels.append(shape_data)

    return tf.data.Dataset.from_tensor_slices((inputs, labels))

# This is a single use code for downscaling an entire directory of binvox models, uncommend if needed.
# def downscale_binvox_dir(input_dir, output_dir, factor = 2):
#     if input_dir[-1] != "/":
#         input_dir = input_dir + '/'
#     if output_dir[-1] != "/":
#         output_dir = output_dir + '/'

#     for file in os.listdir(input_dir):
#         with open(input_dir + file, 'rb') as f:
#             shape = bv.read(f)
#             shape.data = downscale_binvox(shape.data, factor)

#         # with open(output_dir)

def downscale_binvox(binvox, factor = 2):
    binvox.dims = [dim // 2 for dim in binvox.dims]
    binvox.data = resize(binvox.data, (binvox.dims[0], binvox.dims[1], binvox.dims[2]))
    return binvox

