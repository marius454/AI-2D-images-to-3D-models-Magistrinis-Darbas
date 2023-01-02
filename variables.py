## General variables
# Resolution of the input images
imageWidth = 256
imageHeight = 256


## 3D-GAN variables
threeD_gan_z_size = 200
threeD_gan_epochs = 50 # the paper does not the number of epochs
threeD_gan_checkpoint_frequency = 5
threeD_gan_batch_size = 20
threeD_gan_generator_learning_rate = 0.0025
threeD_gan_discriminator_learning_rate = pow(10, -5)
threeD_gan_discriminator_training_threshold = 80
threeD_gan_adam_beta = 0.5

# variables for 64x64x64 resolution
# threeD_gan_resolution = 64
# threeD_gan_generator_first_filter = 512
# threeD_gan_generator_intermediate_filters = [256, 128, 64]
# threeD_gan_generator_final_filter = 1
# threeD_gan_discriminator_first_filter = 64
# threeD_gan_discriminator_intermediate_filters = [128, 256, 512]
# threeD_gan_discriminator_final_filter = 1

# variables for 128x128x128 resolution
threeD_gan_resolution = 128
threeD_gan_generator_first_filter = 1024
threeD_gan_generator_intermediate_filters = [512, 256, 128, 64]
threeD_gan_generator_final_filter = 1
threeD_gan_discriminator_first_filter = 64
threeD_gan_discriminator_intermediate_filters = [128, 256, 512, 1024]
threeD_gan_discriminator_final_filter = 1

## Z-GAN variables

