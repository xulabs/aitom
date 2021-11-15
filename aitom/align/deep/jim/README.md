# Jim-Net
Computing dense pixel-to-pixel image correspondences is a fundamental task of computer vision. Often, the objective is to align image pairs from the same semantic category for manipulation or segmentation purposes. Despite achieving superior performance, existing deep learning alignment methods cannot cluster images; consequently, clustering and pairing images needed to be a separate laborious and expensive step.

<p align="center">
<img src="https://user-images.githubusercontent.com/31047726/136886457-0f279bef-c9b8-44d7-ac30-67db3efeff28.png" width="1400">
</p>

Given a dataset with diverse semantic categories, we propose a multi-task model, Jim-Net, that can directly learn to cluster and align images without any pixel-level or image-level annotations. We design a pair-matching alignment unsupervised training algorithm that selectively matches and aligns image pairs from the clustering branch. Our unsupervised Jim-Net achieves comparable accuracy with state-of-the-art supervised methods on benchmark 2D image alignment dataset PF PASCAL. Specifically, we apply Jim-Net to cryo-electron tomography, a revolutionary 3D microscopy imaging technique of native subcellular structures. After extensive evaluation on seven datasets, we demonstrate that Jim-Net enables systematic discovery and recovery of representative macromolecular structures in situ, which is essential for revealing molecular mechanisms underlying cellular functions. To our knowledge, Jim-Net is the first end-to-end model that can simultaneously align and cluster images, which significantly improves the performance as compared to performing each task alone.


Please refer to our paper for more details:

Zeng, Xiangrui, Howe, Gregory, and Xu, Min. "End-to-End Robust Joint Unsupervised Image Alignment and Clustering." In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 3854-3866. [[ICCV 2021 open access](https://openaccess.thecvf.com/content/ICCV2021/html/Zeng_End-to-End_Robust_Joint_Unsupervised_Image_Alignment_and_Clustering_ICCV_2021_paper.html)]


## Package versions
* keras==2.2.4
* tensorflow-gpu==1.12.0
* h5py==2.10.0



## Installation 
We have uploaded the code for 2D data. Please see the corresponding folder. We are organizing code for 3D data and will upload it soon.


### BibTeX

If you use or modify the code from this project in your project, please cite:
```bibtex
@inproceedings{zeng2021end,
  title={End-to-End Robust Joint Unsupervised Image Alignment and Clustering},
  author={Zeng, Xiangrui and Howe, Gregory and Xu, Min},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3854--3866},
  year={2021}
}
```
Thank you!

