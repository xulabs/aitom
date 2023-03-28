# DISCA
DISCA is a high-throughput template-and-label-free deep learning approach that automatically discovers subsets of homogeneous structures by learning and modeling 3D structural features and their distributions. The training is fully unsupervised.

<p align="center">
<img src="https://user-images.githubusercontent.com/31047726/228112212-f1ed62f5-5c7d-4c34-8614-37ee4f58f045.png" width="600">
</p>

Please refer to our paper for more details:

Zeng, X., Kahng, A., Xue, L., Mahamid, J., Chang, Y.W., and Xu, M. "High-throughput cryo-et structural pattern mining by deep iterative unsupervised clustering." 


## Package versions
* keras==2.9.0
* tensorflow-gpu==2.9.0
* scikit-learn==1.1.3
* scipy==1.4.1



## Installation 
Please follow the installation guide of AITom. 

Alternatively, you could download all the scripts in this module and modify the lines for importing modules to run DISCA independently. 

## Training

### Dataset

The subtomogram dataset can be prepared using the Difference of Gaussians particle picking [module](https://github.com/xiangruz/aitom/blob/master/doc/tutorials/008_particle_picking.py) in AITom.


### Training code

The training code is available in DISCA.py.


### BibTeX

If you use or modify the code from this project in your project, please cite:
```bibtex
@article{zeng2021disca,
  title={DISCA: high-throughput cryo-ET structural pattern mining by deep unsupervised clustering},
  author={Zeng, Xiangrui and Kahng, Anson and Xue, Liang and Mahamid, Julia and Chang, Yi-Wei and Xu, Min},
  journal={bioRxiv},
  pages={2021--05},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
Thank you!
