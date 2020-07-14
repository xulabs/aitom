# PUB-SalNet-Implementation
Pytorch Implementation of [**PUB-SalNet: A Pre-Trained Unsupervised Self-Aware Backpropagation Network for Biomedical Salient Segmentation**](https://www.mdpi.com/1999-4893/13/5/126/htm)

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

- For inference:  image_test.py

- To report score:  measure_test.py

All of the parameters of above mentioned files are in the **config**dict.

# Detailed Guideline
## Dataset
### CustomDataset Class
Your custom dataset should contain `images`, `masks` folder.
  - In each folder, the filenames should be matched. 
  - The `images`, `masks`should be 2D images.
  - eg. ```images/a.jpg masks/a.jpg```

### Directory & Name Format of .ckpt files
<code>
        "models/state_dict/<datetime(Month,Date,Hour,Minute)>/<#epo_#step>.ckpt"
</code>

##  References
[1] 2018_PiCANet_ Learning pixel-wise contextual attention for saliency detection_CVPR

[2] 2020_PUB-SalNet_ A Pre-Trained Unsupervised Self-Aware Backpropagation Network for Biomedical Salient Segmentation

[3] https://github.com/Ugness/PiCANet-Implementation