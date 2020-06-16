# Gum-Net
Geometric unsupervised matching Net-work (Gum-Net) finds the geometric correspondence between two images with application to 3D subtomogram alignment and averaging. We introduce an end-to-end trainable architecture with three novel modules specifically designed for preserving feature spatial information and propagating feature matching information. 

<img src="https://user-images.githubusercontent.com/31047726/84725693-2ec78800-af59-11ea-94a3-fdd6b5242645.png" width="800">


The training is performed in a fully unsupervised fashion to optimize a matching metric. No ground truth transformation information nor category-level or instance-level matching supervision information is needed. As the first 3D unsupervised geometric matching method for images of strong transformation variation and high noise level, Gum-Net significantly improved the accuracy and efficiency of subtomogram alignment. 

<img src="https://user-images.githubusercontent.com/31047726/84724490-536e3080-af56-11ea-93b8-b31bd4f18cd6.gif" width="400">


Please refer to our paper for more details:

Zeng, Xiangrui, and Min Xu. "Gum-Net: Unsupervised Geometric Matching for Fast and Accurate 3D Subtomogram Image Alignment and Averaging." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4073-4084. 2020.

http://openaccess.thecvf.com/content_CVPR_2020/html/Zeng_Gum-Net_Unsupervised_Geometric_Matching_for_Fast_and_Accurate_3D_Subtomogram_CVPR_2020_paper.html 




## Key prerequisites
* keras==2.2.4
* tensorflow-gpu==1.12.0



## Installation 
Please follow the installation guide of AITom. 

Alternatively, you could download all the scripts in aitom.align.gum and modify the lines for importing modules to run Gum-Net independently. 


