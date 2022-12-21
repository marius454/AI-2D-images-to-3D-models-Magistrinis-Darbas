The IKEA dataset for single-image 3D reconstruction 
======================================================

This dataset is based on the original IKEA dataset (http://ikea.csail.mit.edu/) by Joseph J. Lim, Hamed Pirsiavash, and Antonio Torralba. Here, we provide cropped images of each object and corresponding voxelized shapes.

- ./chair/: testing images for chairs
- ./list/: shape IDs for images of each shape category
- ./model/: voxelized 3D shapes

For visualization scripts, please refer to the MarrNet repo (https://github.com/jiajunwu/marrnet).

======================================================
=== References
======================================================

If you use this dataset in your work, please cite the following two papers:

@inproceedings{3dgan,
  title={{Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling}},
  author={Wu, Jiajun and Zhang, Chengkai and Xue, Tianfan and Freeman, William T and Tenenbaum, Joshua B},
  booktitle={Advances in Neural Information Processing Systems (NIPS)},
  pages={82--90},
  year={2016}
}

@inproceedings{lpt2013ikea,
   title={{Parsing IKEA Objects: Fine Pose Estimation}},
   author={Joseph J. Lim and Hamed Pirsiavash and Antonio Torralba},
   booktitle={IEEE International Conference on Computer Vision (ICCV)},
   year={2013}
}

======================================================
=== Contact info
======================================================
For questions, please contact: Jiajun Wu (jiajunwu.cs@gmail.com), Chengkai Zhang (ckzhang@mit.edu), Tianfan Xue (tianfan.xue@gmail.com)

