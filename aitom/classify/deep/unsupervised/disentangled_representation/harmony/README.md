# Harmony

Harmony, A Generic Unsupervised Approach for Disentangling Semantic Content From Parameterized Transformations, is a generic unsupervised framework that simultaneously and explicitly disentangles semantic content from multiple parameterized transformations. In Harmony, we used the power of cross-contrastive learning to explicitly disentangle transformations and semantic content. As an application of Harmony, we disentangle semantic content from multiple geometric and lighting condition transformations in various imaging datasets. With Harmony, we resolved transformation-invariant conformations of proteins from 2D single-particle cryo-EM images. We disentangled transformation parameters from 3D images and applied them to model structural heterogeneity of extremely noisy real and simulated 3D cryo-ET subtomograms. 


<p align="center">
<img src="https://user-images.githubusercontent.com/14123565/180104959-46ecb2b6-03c6-49de-b296-b6546d62c98a.png" width="600">
</p>

Please refer to our paper for more details:

Uddin, Mostofa Rafid, Gregory Howe, Xiangrui Zeng, and Min Xu. "Harmony: A Generic Unsupervised Approach for Disentangling Semantic Content From Parameterized Transformations." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20646-20655, 2022. [[CVPR 2022 open access](https://openaccess.thecvf.com/content/CVPR2022/papers/Uddin_Harmony_A_Generic_Unsupervised_Approach_for_Disentangling_Semantic_Content_From_CVPR_2022_paper.html)]

## Usage

### Cryo-EM dataset

To disentangle rotation and translation from semantic content on 2D cryo-EM images, you can run the following:

```
python main2D.py --dataset <dataset_name> --num-epochs <number of epochs> --z-dim <dimension of semantic latent factor> --pixel <number of pixels per dimension> --batch-size <Batch size for training> --learning-rate <learning-rate for training> --gamma <gamma parameter in Harmony loss function>
```

If gamma is not set from the command line, the model will estimate the optimal gamma from the training dataset and batch-size. Such estimation worked well for our experiments. 

To perform disentanglement on the codhacs dataset used in the paper, follow the procedures:

<ol>

<li>Create a 'data' folder in the working directory</li> 
<li>Change directory to the 'data' folder</li>
<li>Download the dataset from [CODH/ACS EM images](http://bergerlab-downloads.csail.mit.edu/spatial-vae/codhacs.tar.gz) and extract them inside the 'data' folder</li>
<li>Run the following command:

```
python main2D.py --dataset codhacs --num-epochs 100 --z-dim 1 --pixel 40 --batch-size 100
```
</li>
</ol>

### Disentangling full affine matrix 

To disentangle full affine matrix (rotation, translation, and scaling) add a --scale command in the above mentioned command:

```
python main2D.py --dataset <dataset_name> --scale --num-epochs <number of epochs> --z-dim <dimension of semantic latent factor> --pixel <number of pixels per dimension> --batch-size <Batch size for training> --learning-rate <learning-rate for training> --gamma <gamma parameter in Harmony loss function>
```

Disentangling transformations other than rotation, translation, and scaling has not been implemented for 2D grayscale images yet. But it can be implemented easily by modifying the model.py in model2D folder. 

### Disentangling 3D rotation and translation

To disentangle rotation and translation from semantic content on 2D cryo-EM images, you can run the following:

```
python main3D.py --dataset <dataset_name> --num-epochs <number of epochs> --z-dim <dimension of semantic latent factor> --batch-size <Batch size for training> --learning-rate <learning-rate for training> --gamma <gamma parameter in Harmony loss function>
```

For a demo, create a 'data' folder, download the [[demo dataset](https://cmu.box.com/s/vx45o7xa3qbz6tyd0ri6sxgpxus613wy)] from here, put the pickle files inside the 'data' folder and run the following command:

```
python main3D.py --dataset harmony_3d_demo --num-epochs 200 --z-dim 1 --batch-size 100
```

### Disentangling contrast from colored (RGB) images

To disentangle semantic content (e.g., facial identity) from lighting condition transformation (e.g.,contrast), you can run the following:

```
python main-color.py --dataset <dataset_name> --num-epochs <number of epochs> --z-dim <dimension of semantic latent factor> --batch-size <Batch size for training> --learning-rate <learning-rate for training> --gamma <gamma parameter in Harmony loss function>
```

If gamma is not set from the command line, the model will estimate the optimal gamma from the training dataset and batch-size. Such estimation worked well for our experiments. 

## Installation 
Please follow the installation guide of AITom. 

Alternatively, you could download all the scripts in aitom.classify.unsupervised.disentangled.harmony and modify the lines for importing modules to run Harmony independently. 

### BibTeX

If you use or modify the code from this project in your project, please cite:
```bibtex
@inproceedings{uddin2022harmony,
  title={Harmony: A Generic Unsupervised Approach for Disentangling Semantic Content From Parameterized Transformations},
  author={Uddin, Mostofa Rafid and Howe, Gregory and Zeng, Xiangrui and Xu, Min},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20646--20655},
  year={2022}
}
```
Thank you!
