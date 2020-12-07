# Gum-Net
Geometric unsupervised matching Net-work (Gum-Net) finds the geometric correspondence between two images with application to 3D subtomogram alignment and averaging. We introduce an end-to-end trainable architecture with three novel modules specifically designed for preserving feature spatial information and propagating feature matching information. 

<p align="center">
<img src="https://user-images.githubusercontent.com/31047726/84725693-2ec78800-af59-11ea-94a3-fdd6b5242645.png" width="800">
</p>

The training is performed in a fully unsupervised fashion to optimize a matching metric. No ground truth transformation information nor category-level or instance-level matching supervision information is needed. As the first 3D unsupervised geometric matching method for images of strong transformation variation and high noise level, Gum-Net significantly improved the accuracy and efficiency of subtomogram alignment. 

<p align="center">
<img src="https://user-images.githubusercontent.com/31047726/84724490-536e3080-af56-11ea-93b8-b31bd4f18cd6.gif" width="400">
</p>

Please refer to our paper for more details:

Zeng, Xiangrui, and Min Xu. "Gum-Net: Unsupervised Geometric Matching for Fast and Accurate 3D Subtomogram Image Alignment and Averaging." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4073-4084. 2020. [[CVPR 2020 open access](http://openaccess.thecvf.com/content_CVPR_2020/html/Zeng_Gum-Net_Unsupervised_Geometric_Matching_for_Fast_and_Accurate_3D_Subtomogram_CVPR_2020_paper.html)]


## Package versions
* keras==2.2.4
* tensorflow-gpu==1.12.0
* h5py==2.10.0



## Installation 
Please follow the installation guide of AITom. 

Alternatively, you could download all the scripts in aitom.align.gum and modify the lines for importing modules to run Gum-Net independently. 

## Demo

### Dataset

The [demo dataset](https://cmu.box.com/s/la07ke48s6vkv8y4ntv7yn1hlgwo9ybn) consists of 100 subtomogram pairs (20 of each structure) simulated at SNR 0.1. Transformation ground truth is provided for evaluation. 

Masks of observed region and missing region in Fourier space are provided for imputation in the spatial transformation step. Tilt angle range masks can be generated using functions in aitom.image.vol.wedge.util.

### Trained model

The [model](https://cmu.box.com/s/ymjit1ta5svqb8hyegwf5rqk2m46ouz7) is trained on the simulated dataset at SNR 100 from the paper.

### Training code

The training code finetunes the trained model (from SNR 100 dataset) on the demo dataset (SNR 0.1) for 20 iterations. 

```
python Gum-Net.py
```

Output:

```
Before finetuning:
Rotation error:  1.3030150126200715 +/- 0.8484602493466796 Translation error:  5.723414606003282 +/- 3.9436690083966606 ----------

Training Iteration 0
......
......
......
Training Iteration 19
......

After finetuning:
Rotation error:  1.0768166138653037 +/- 0.7477417154213482 Translation error:  3.5317874399013327 +/- 2.4426374023491872 ----------
```


### BibTeX

If you use or modify the code from this project in your project, please cite:
```bibtex
@inproceedings{zeng2020gum,
  title={Gum-Net: Unsupervised Geometric Matching for Fast and Accurate 3D Subtomogram Image Alignment and Averaging},
  author={Zeng, Xiangrui and Xu, Min},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4073--4084},
  year={2020}
}
```
Thank you!
