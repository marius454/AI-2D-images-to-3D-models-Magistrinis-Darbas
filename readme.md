# Magistro baigiamasis darbas: Generatyviniai besivaržantys neuroniniai tinklai 3D modeliams generuoti iš 2D nuotraukų
# Master's thesis: Generative Adversarial Networks for 3D Model Reconstruction from 2D Images

## Placeholder


## Code
```data_processing.py```:
contains methods for inputing, outputing and transforming 3D or 2D image data.

```threeD_gan.py```:
Code for create 3D digitization neural network as discribed in Wu et al. [2016]


### Code usage

#### General notes
- To set mixed precision add ```tf.keras.mixed_precision.set_global_policy('mixed_float16')``` to main.py

#### Load and display a single image:
```
image = dp.load_single_image('chair.png')
dp.show_single_image(image)
```
