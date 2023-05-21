batch_size = 4  # 4 (in the paper or example code)
epochs = 100  # 80000 batches
shape_res = 256  # 256
latent_dim = 1024  # 2
generator_lr = 0.0004  # 0.0001
discriminator_lr = 0.0001  # 0.0001
encoder_lr = 0.0004  # 0.0004
adam_beta1 = 0  # 0
adam_beta2 = 0.999  # 0.999
lambda_1 = 10  # 5

g_iter = 2  # number of generator runs per batch
e_iter = 3  # number of photos use when training encoder (selected randomly)
e_max_iter = 6 # how many photos are selected when loading data

image_res = 256  # 2D images are not used in the paper, this is original
shapes64_dir = "./Data/ShapeNetSem/tables-binvox-64"
shapes256_dir = "./Data/ShapeNetSem/tables-binvox-256"
training_split = 0.8
display_callback_frequency = 70
save_callback_period = 20
val_batch_size = 2

use_eager_mode = False # Need to also comment out or uncomment the @tf.function descriptors from threeD_gan.py train steps accordingly
print_progress = False
add_noise_to_input_shapes = False
add_noise_to_discriminator_labels = False
