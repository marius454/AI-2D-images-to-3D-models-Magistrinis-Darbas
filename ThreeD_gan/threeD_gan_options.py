z_size = 200  # 200 (in the paper)
epochs = 100  # 100
batch_size = 64  # 100 
generator_lr = [0.0021, 0.00255, 0.0023, 0.002]  # 0.0025
discriminator_lr = [0.00002, 0.00005, 0.000035, 0.00002]  # 1e-5 / 0.00001 (resume training with 0.000024 if 0.000025 does not work, stopped after epoch 6)
encoder_lr = [0.001, 0.0008, 0.0006, 0.0004]  # 0.0003 (given in Larsen et al. (2016))
discriminator_training_threshold = 0.8  # 0.8
adam_beta = 0.5  # 0.5
image_res = 256  # 256
shape_res = 64  # 64

# KL divergence and reconstruction loss weights described in the paper
# it is unclear how exactly they are supposed to be used
alpha_1 = 5  # 5
alpha_2 = 0.0005  # 1e-4 / 0.0001

discriminator_activation = 'sigmoid'  # 'sigmoid'
generator_activation = 'sigmoid'  # 'sigmoid'

use_eager_mode = False # Need to also comment out or uncomment the @tf.function descriptors from threeD_gan.py train steps accordingly
add_noise_to_input_shapes = True
add_noise_to_discriminator_labels = True
backgroud_weight = 1
voxel_weight = 1
display_callback_frequency = 100
training_split = 0.8
shape_data_dir = "./Data/ShapeNetSem/tables-binvox-64"



#### Experiments that I have done:
# Set training to false for models that are not being trained
# Apply all training within a single gradient tape environtment
# only train the generator when the accuracy agains the discriminator less than 80%
# different methods of loss calculation, most of which are still saved in the metrics file
# eager execution and graph execution
# one of the issues was, that the discriminator would bouse around between marking all generated output as either 1 or 0
# especially with generated shapes, where discriminator accuracy is often either 100% or 0%
# to resolve this I first tried to add some noise to the real_shapes

# very commonly the training converges on the generator creating empty shapes

# See squared euclidean distance in wikipedia - https://en.wikipedia.org/wiki/Euclidean_distance
# "minimizing squared distance is equivalent to minimizing the Euclidean distance, 
# so the optimization problem is equivalent in terms of either, but easier to solve using squared distance"

# eager execution with tables2 takes ~1100-1200 seconds per epoch, graph executions takes about 800-900 seconds per epoch


# different training optimizer and a1/a2 values:
## disc_lr = 1e-4, gen_lr = 0,0025, a1 = 5, a2 = 1e-1
# discriminator learning significantly quicker than generator but there are some signs of a fight at first, 
# but then the generator gets stuck at 0% accuracy
# a1*KL_loss and a2*recon_loss eventually become about equal

# https://github.com/FairyPig/3D-VAE-GAN/tree/master settings test
## disc_lr = 5e-5, gen_lr = 1e-4, a1 = 5, a2 = 5e-4 now calculating recon loss as a sum of squared differences
## and KL_loss based on example formula. Reduce mean is aplied when calculating final loss values
# generator accuracy gets stuck to 0, encoder loss is very high and might be detrimental to results, and makes it difficult 
# to choose a metric to judge checkpoints by.
# after 5 epochs the result is an empty shape

## disc_lr = 5e-5, gen_lr = 0.003, enc_lr = 0.001, a1 = 5, a2 = 5e-4 calculating recon loss as a sum of squared differences
# generator gets overtrained realy quickly

## disc_lr = 5e-5, gen_lr = 0.0025, enc_lr = 0.001, a1 = 5, a2 = 5e-4
# there are some signs of a fight between the generator and discriminator, but gen_accuracy is bouncing around between 0 and 1
# KL_loss becomes close to 0, so a lower weight for a2 will be better for next time?
# disc and gen accuracies are showing signs of equalizing at about the middle of the of epoch 2, but go back to bouncing arround quite quickly
# the overall tendency is leading towards generator domination, will increace the disc_lr slightly for the next test
# at the start of epoch 4 the discriminator is dooing much better, and real fight happens by the middle
# will try more epochs for this
# both accuracies being low seems to be a good thing, because that means, that accuracies are swaping places on every batch
# overall tendencies are good, but reconstruction loss hits a wall at about 8
# after 10 epochs an empty shape is returned. 10th epoch results are returned, can check them with different images

## disc_lr = 7e-5, gen_lr = 0.0025, enc_lr = 0.001, a1 = 5, a2 = 1e-4 al tables, no test set
# a much larger number of epochs was use, generator accuracy overall was pretty low, but was never
# permanently stuck at 0
# the generator started to generated a large cube in the middle of the voxel space, and then 
# slowly making it smaller, which will hopefully later result in something at least resembling 
# a 3d model of a table
# At epoch 56 the reconstruction is still being reduced at a pretty slow but steady pace.
# at epoch 60 the cube gets bigger again when testing with the standar table with bench image. Need to test with other images.
# by the end disc accuracy always stays above 80%


## tables1, disc_lr = 5e-5, gen_lr = 0.0025, enc_lr = 0.001, a1 = 5, a2 = 5e-4, tables 1, without images 8 and 12, batch size 64
# discriminator cannot keep up with the generator
# actually got usable results, however the image and shape pairs do not match, perhaps there was a issue when making image-shape pairs
# there was indeed an issue with the training data images were not being paired up with the correct shapes.

## tables2 with train/test 80/20%, disc_lr = 5e-5, gen_lr = 0.0025, enc_lr = 0.001, a1 = 5, a2 = 5e-4, without images 8 and 12, batch size 64
# generator falls behind, recon loss on the test set does not decrese
# try settings from the paper 

## tables2 with train/test 80/20%, disc_lr = 1e-5, gen_lr = 0.0025, enc_lr = 0.001, a1 = 5, a2 = 1e-4, without images 8 and 12, batch size 64 after fixiing image rotation
# generator get overtrained and stuck 100% accuracy with the discriminator guessing that all shapes are real

## tables2 with train/test 80/20%, disc_lr = 4e-5, gen_lr = 0.0025, enc_lr = 0.001, a1 = 5, a2 = 4e-4, without images 8 and 12, batch size 64
# generator get overtrained and stuck 100% accuracy with the discriminator guessing that all shapes are real


## tables2 with train/test 80/20%, disc_lr = 5e-5, gen_lr = 0.0025, enc_lr = 0.001, a1 = 5, a2 = 5e-4, without images 8 and 12, batch size 64
# generator accuracy very low, slow progress
# Stoped after 49 epochs

## tables2 with train/test 80/20%, disc_lr = 4e-5, gen_lr = 0.0025, enc_lr = 0.001, a1 = 5, a2 = 5e-4, without images 8 and 12, batch size 64
# generator accuracy very low, basically 0, slow progress
# Stoped after 40 epochs

## tables2 with train/test 80/20%, disc_lr = 2.5e-5, gen_lr = 0.0025, enc_lr = 0.001, a1 = 5, a2 = 5e-4, without images 8 and 12, batch size 64
# stoped after 63 epochs, discriminator too ahead
# continued with gen_lr = 0.0024, disc_lr = 0.00002, enc_lr = 0.0005
