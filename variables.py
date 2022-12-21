## General variables
# The shape of the model and the resolution input images must be resized to
imageWidth = 256
imageHeight = 256


## 3D-GAN variables
threeD_gan_z_size = 200
threeD_gan_batch_size = 100

# change these variables accordingly, to change resolution of generated 3d models
threeD_gan_resolution = 64
threeD_gan_generator_first_filter = 512
threeD_gan_generator_intermediate_filters = [256, 128, 64]
threeD_gan_generator_final_filter = 1
threeD_gan_discriminator_first_filter = 64
threeD_gan_discriminator_intermediate_filters = [128, 256, 512]
threeD_gan_discriminator_final_filter = 1

## Z-GAN variables

