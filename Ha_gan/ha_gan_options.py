batch_size = 4  # 4 (in the paper or example code)
epochs = 100  # 80000 batches
shape_res = 256  # 256
latent_dim = 1024  # 2

g_iter = 1  # number of generator runs per batch

image_res = 256  # 2D images are not used in the paper, this is original

use_eager_mode = False # Need to also comment out or uncomment the @tf.function descriptors from threeD_gan.py train steps accordingly
add_noise_to_input_shapes = False
add_noise_to_discriminator_labels = False
