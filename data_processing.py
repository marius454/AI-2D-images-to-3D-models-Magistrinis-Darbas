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

# TODO: need to fix with image_res when decoding
def load_single_image(imagePath: str, image_res):
    """Load image file into code as tensor."""
    image = tf.io.read_file(imagePath)
    if imagePath.endswith(".jpg"):
        image = decode_jpeg(image, image_res)
    elif imagePath.endswith(".png"):
        image = decode_png(image, image_res)

    return image

def show_single_image(image):
    """Display a single 2D image using matplotlib."""
    plt.figure(figsize=(8, 5))

    plt.title('Image')
    plt.imshow(tf.cast(image, tf.uint8))
    plt.axis('off')
    plt.show()



def decode_jpeg(image, image_res, resize_method='bilinear'):
    """Load jpeg image into a tensor."""
    image = tf.io.decode_jpeg(image, channels=3)
    return tf.image.resize(image, [image_res, image_res], method=resize_method)

def decode_png(image, image_res, resize_method='bilinear'):
    """Load png image into a tensor."""
    image = tf.io.decode_png(image, channels=3)
    return tf.image.resize(image, [image_res, image_res], method=resize_method)

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

def show_image_and_shapes(image, real_shape, generated_shape, dimensions = (128, 128, 128)):
    """Show the image of an object, it's ground trush 3D model, and the generated model."""
    title = ['Input Image', 'True Shape', 'Generated Shape']
    x, y, z = np.indices((dimensions[0]+1, dimensions[1]+1, dimensions[2]+1))

    plot = plt.figure(figsize=(15, 5))

    ix = plot.add_subplot(1, len(title), 1)
    ix.set_title(title[0])
    ix.imshow(tf.keras.utils.array_to_img(image))
    ix.axis('off')

    rx = plot.add_subplot(1, len(title), 2, projection='3d')
    rx.voxels(x, y, z, real_shape)
    rx.set_title(title[1])
    rx.set(xlabel='x', ylabel='y', zlabel='z')
    rx.set_aspect('equal')
    # rx.axis('off')

    gx = plot.add_subplot(1, len(title), 3, projection='3d')
    gx.voxels(x, y, z, generated_shape)
    gx.set_title(title[2])
    gx.set(xlabel='x', ylabel='y', zlabel='z')
    gx.set_aspect('equal')
    # gx.axis('off')
        
    plt.show()

def show_image_and_shape(image, shape, dimensions = (128, 128, 128)):
    """Show the image of an object and it's ground trush 3D model"""
    title = ['Image', 'Shape']
    x, y, z = np.indices((dimensions[0]+1, dimensions[1]+1, dimensions[2]+1))

    plot = plt.figure(figsize=(12, 5))

    ix = plot.add_subplot(1, len(title), 1)
    ix.set_title(title[0])
    ix.imshow(tf.keras.utils.array_to_img(image))
    ix.axis('off')

    rx = plot.add_subplot(1, len(title), 2, projection='3d')
    rx.voxels(x, y, z, shape)
    rx.set_title(title[1])
    rx.set(xlabel='x', ylabel='y', zlabel='z')
    rx.set_aspect('equal')
    # rx.axis('off')
        
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

def get_shape(filepath, downscale_factor = None):
    with open(filepath, "rb") as f:
        shape = bv.read_as_3d_array(f)
        if downscale_factor:
            shape = downscale_binvox(shape, factor=downscale_factor)
    return shape


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
            # print ("getting shape: " + Path(file).stem)
            with open(directory + file, "rb") as f:
                shapes[Path(file).stem] = bv.read_as_3d_array(f)
                if downscale_factor:
                    shapes[Path(file).stem] = downscale_binvox(shapes[Path(file).stem], factor=downscale_factor)

    return shapes

def get_shape_screenshots(shape_codes, directory, image_res):
    """
    Load a dictionary of lists containing images that render a shape referenced by a shape code.
    Assumed that ShapeNet image data is kept in PNG format.
    """
    shape_screenshots = {}
    if directory[-1] != "/":
        directory = directory + "/"

    for folder in os.listdir(directory):
        if folder in shape_codes:
            # print ("getting screenshots: " + folder)
            images = []

            # Load images from the selections of angles in range [1, 13]
            # 8 and 12 might not be good, because with some models, the image might be straight on and not provido much 3D information.
            # for i in [6, 7, 8, 9, 10, 11, 12, 13]:
            for i in [6, 7, 9, 10, 11, 13]:
                image = decode_png(tf.io.read_file(f"{directory + folder}/{folder}-{i}.png"), image_res)
                image = normalize_image(image)
                images.append(image)
            shape_screenshots[folder] = images
    
    return shape_screenshots


# TODO Will have to make this more portable, currently other users will have to set up datasets themselves
def load_data(dataset, shapes_dir, image_res, downscale_factor = None):
    """
    Load shape and image data from a predifened shape code file into a tensor dataset

    `shapenet_tables` - load dataset of 431 table and desk objects collected based on their metadata wnsynet codes\n
    `shapenet_tables2` - load dataset of 670 table and desk objects collected based on their metadata categories\n
    `shapenet_limited_tables` - load a subset of shapenet_tables with a limited amount of entries (currently 100, but might change)\n
    `shapenet_single_table` - load a single table object from the shapenet_tables set
    """
    datasets = {
        'shapenet_tables': "./Data/ShapeNetSem/Table.csv", 
        'shapenet_tables2': "./Data/ShapeNetSem/Table2.csv", 
        'shapenet_limited_tables': "./Data/ShapeNetSem/limited_table.csv", 
        'shapenet_single_table': "./Data/ShapeNetSem/single_table.csv"
    }
    if dataset not in datasets.keys():
        raise Exception('undefined dataset name given to load_data()')

    return load_shapenet_data(datasets[dataset], shapes_dir, image_res, downscale_factor)


def load_shapenet_data(codes_filepath, shapes_dir, image_res, downscale_factor = None):
    """Load shapenet data into (shape, image) pairs tensor."""
    shape_codes = get_shape_code_list(codes_filepath)
    shapes = get_shapes(shape_codes, shapes_dir, downscale_factor)
    shape_screenshots = get_shape_screenshots(shape_codes, "./Data/ShapeNetSem/screenshots/", image_res)

    print("\nFormating data as (image, shape) pairs")
    inputs = []
    labels = []
    for code in shape_codes:
        for i in range(len(shape_screenshots[code])):
            inputs.append(shape_screenshots[code][i])
            labels.append(np.array(shapes[code].data))

    ## TEST IF SHAPES ARE BEING PAIRED CORRECTLY
    # for i in range(len(inputs)):
    #     show_image_and_shape(inputs[i], labels[i], shapes[shape_codes[0]].dims)

    return tf.data.Dataset.from_tensor_slices((inputs, labels))

def downscale_binvox(binvox, factor = 2):
    binvox.dims = [dim // 2 for dim in binvox.dims]
    binvox.data = resize(binvox.data, (binvox.dims[0], binvox.dims[1], binvox.dims[2]))
    return binvox




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
