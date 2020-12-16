# Multi-task Learning for Macromolecule Classification, Segmentation-Implementation
Pytorch Implementation for the Segmentation Part of [**Multi-task Learning for Macromolecule Classification, Segmentation and Coarse Structural Recovery in Cryo-Tomography**](https://arxiv.org/pdf/1805.06332)

# Execution Guideline
## Requirements
pytorch==0.4.1  
tensorboardX==1.1  
torchvision==0.2.1  
numpy==1.14.2  
mrcfile

## My Environment
Ubuntu16.04  
CUDA 9.0  
cudnn 7.0  
python 3.6.5  
GeForce GTX 1080 1080ti  
8GB RAM

## Execution Guide
- For training: usage: train.py
- For inference:  test.py

# Detailed Guideline
### CustomDataset Class
Your custom dataset should contain `images`, `masks` folder.
  - In each folder, the filenames should be matched. 
  - The `images`, `masks`should be 2D images.
  - eg. ```images/a.jpg masks/a.jpg```

##  Referencess
[1]  2018_Multi-task Learning for Macromolecule Classification, Segmentation and Coarse Structural Recovery in Cryo-Tomography_BMVC_c

[3] https://github.com/xulabs/projects/tree/master/segmentation