# Macromolecule Structure Segmentation-Implementation
Pytorch Implementation for the Segmentation Part of [**Domain Randomization for Macromolecule Structure Classification and Segmentation in Electron Cyro-tomograms**](https://repository.kaust.edu.sa/bitstream/handle/10754/663480/2019_domainrandomization_bibm.pdf?sequence=1&isAllowed=y)

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

### Unet Model
The file of 'unet2D' contains 2D Unet model
The file of 'unet3D' contains 3D Unet model

##  References
[1]  2019_Domain Randomization for Macromolecule Structure Classification and Segmentation in Electron Cyro-tomograms_BIBM_b

[3]https://github.com/milesial/Pytorch-UNet