# One-Shot Learning With Attention-Guided Segmentation in Cryo-Electron Tomography

Bo Zhou, Haisu Yu, Xiangrui Zeng, Xiaoyan Yang, Jing Zhang, and Min Xu

Frontiers in Molecular Biosciences (AI in Biological and Biomedical Imaging), 2020

[[Paper](https://www.frontiersin.org/articles/10.3389/fmolb.2020.613347/full)]

This repository contains the PyTorch implementation of COS-Net.

### Citation
If you use this code for your research or project, please cite:

    @article{zhou2020one,
      title={One-Shot Learning With Attention-Guided Segmentation in Cryo-Electron Tomography},
      author={Zhou, Bo and Yu, Haisu and Zeng, Xiangrui and Yang, Xiaoyan and Zhang, Jing and Xu, Min},
      journal={Frontiers in Molecular Biosciences},
      volume={7},
      year={2020},
      publisher={Frontiers Media SA}
    }


### Environment and Dependencies
Requirements:
* Python 3.7
* Pytorch 0.4.1
* scipy
* scikit-image
* opencv-python
* denseCRF3D

Our code has been tested with Python 3.7, Pytorch 0.4.1, CUDA 10.0 on Ubuntu 18.04.


### Dataset Setup
    ../
    Data/
    ├── subtomograms_snrINF      # dataset with SNR = INF       
    │   ├── cnt.npy              # 22x1 array -- contains the number of subtomograms for each classes (22 classes in total)
    │   ├── data.npy             # 22x1000x32x32x32 array -- contains 22 classes' subtomograms with size of 32x32x32. 1000 subtomogram storage place give, and is indexed by cnt.npy
    │   ├── data_seg.npy         # 22x1000x32x32x32 array -- contains 22 classes' subtomogram segmentation with size of 32x32x32, corresponding with data.npy
    │   ├── testtom_vol_00.npy   # 1xcountsx32x32x32 array -- contains 1st class's subtomograms with size of 32x32x32, counts means number of subtomogram in this class.
    │   ├── testtom_seg_00.npy   # 1xcountsx32x32x32 array -- contains 1st class's subtomogram segmentation with size of 32x32x32, counts means number of subtomogram in this class.
    │   ├── testtom_vol_01.npy   # please refer to above comments
    │   ├── testtom_seg_01.npy 
    │   ├── testtom_vol_02.npy  
    │   ├── testtom_seg_02.npy  
    │   ├── testtom_vol_03.npy  
    │   ├── testtom_seg_03.npy  
    │   ├── testtom_vol_04.npy  
    │   ├── testtom_seg_04.npy 
    │   ├── testtom_vol_05.npy  
    │   ├── testtom_seg_05.npy  
    │   ├── testtom_vol_06.npy  
    │   ├── testtom_seg_06.npy 
    │   ├── testtom_vol_07.npy  
    │   ├── testtom_seg_07.npy  
    │   ├── testtom_vol_08.npy  
    │   ├── testtom_seg_08.npy  
    │   ├── testtom_vol_09.npy  
    │   ├── testtom_seg_09.npy 
    │   ├── testtom_vol_10.npy  
    │   ├── testtom_seg_10.npy 
    │   ├── testtom_vol_11.npy  
    │   ├── testtom_seg_11.npy  
    │   ├── testtom_vol_12.npy  
    │   ├── testtom_seg_12.npy 
    │   ├── testtom_vol_13.npy  
    │   ├── testtom_seg_13.npy  
    │   ├── testtom_vol_14.npy  
    │   ├── testtom_seg_14.npy  
    │   ├── testtom_vol_15.npy  
    │   ├── testtom_seg_15.npy 
    │   ├── testtom_vol_16.npy  
    │   ├── testtom_seg_16.npy  
    │   ├── testtom_vol_17.npy  
    │   ├── testtom_seg_17.npy 
    │   ├── testtom_vol_18.npy  
    │   ├── testtom_seg_18.npy  
    │   ├── testtom_vol_19.npy  
    │   ├── testtom_seg_19.npy  
    │   ├── testtom_vol_20.npy  
    │   ├── testtom_seg_20.npy 
    │   ├── testtom_vol_21.npy  
    │   └── testtom_seg_21.npy 
    │
    ├── subtomograms_snr10000    # dataset with SNR = 10000  
    │   ├── cnt.npy
    │   └── ... 
    └── 

Please refer to the above data directory setup to prepare you data, and please carefully ead the comments for the file contents. \
The one-shot training and testing process use cnt.py / data.npy / data_seg.npy files. Our dataloader will split the first 1-14 classes as training class and 15-22 classes as testing class. \
The segmentation testing process use testtom_vol_nn.npy files.

### To Run Our Code
- Train and Test the one-shot model
```bash
python main_oneshot.py --dataset 'snrINF' --data_root '../Data/subtomograms_snrINF/' --net 'dusescnn' --N 2 --K 1
```
where \
`--dataset` is the dataset's name. \
`--data_root`  provides the data folder directory (with structure illustrated above). \
`--net` is the network block structrue. \
`--N` is the number of way/class. \
`--K` is the number of shot/sample per class. \
Other hyperparameters can be adjusted in the code as well.

- Test the model's segmentation output
```bash
python main_segment.py --dataset 'snrINF' --data_root '../Data/subtomograms_snrINF/' --resume './outputs/2-way-1-shot_dusescnn_snrINF/model_25.pkl' --net 'dusescnn' --N 2 --K 1 
```
where \
`--resume` defines which checkpoint for segmentation test evaluation. \
The segmentation visualizations will be saved under './outputs/2-way-1-shot_dusescnn_snrINF/segment/'. \
The quantitative results per class will be saved in './outputs/2-way-1-shot_dusescnn_snrINF/segment/dice.txt'. \

Sample training/test scripts are provided under './scripts/' and can be directly executed.


### Contact 
If you have any question, please file an issue or contact the author:
```
Bo Zhou: bo.zhou@yale.edu
```