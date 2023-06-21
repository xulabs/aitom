# DISCA
DISCA is a high-throughput template-and-label-free deep learning approach that automatically discovers subsets of homogeneous structures by learning and modeling 3D structural features and their distributions. The training is fully unsupervised.

<p align="center">
<img src="https://user-images.githubusercontent.com/31047726/228112212-f1ed62f5-5c7d-4c34-8614-37ee4f58f045.png" width="600">
</p>

Please refer to our paper for more details:

Zeng, X., Kahng, A., Xue, L., Mahamid, J., Chang, Y.W., and Xu, M. "High-throughput cryo-et structural pattern mining by deep iterative unsupervised clustering." [[PNAS (direct submission)](https://www.pnas.org/doi/abs/10.1073/pnas.2213149120)]


## Package versions
For keras version (DISCA.py)
* keras==2.9.0
* tensorflow-gpu==2.9.0
* scikit-learn==1.1.3
* scipy==1.4.1

For pytorch version (torch_disca.py)
* scikit-learn==1.1.3
* numpy==1.23.5
* torch==2.0.1
* tqdm==4.64.1
* scipy==1.8.1
* torchvision==0.15.2


## Installation 
Please follow the installation guide of AITom. 

Alternatively, you could download all the scripts in this module and modify the lines for importing modules to run DISCA independently. 

## Training

### Dataset

The subtomogram dataset can be prepared using the Difference of Gaussians particle picking [module](https://github.com/xiangruz/aitom/blob/master/doc/tutorials/008_particle_picking.py) in AITom.


### Training code

The training code is available in DISCA.py.

## Subtomogram averages

The subtomogram average of macromolecular complexes (Fig. 4 and 5 of the manuscript) from the *Rattus* neuron dataset and the *Mycoplasma pneumoniae* dataset have been deposited in the EM Data Bank with accession numbers EMD-40043, -40087, -40089, and -40090. The subtomogram average of macromolecular complexes from the *Synechocystis* cell dataset, the *Cercopithecus aethiops* kidney dataset, and the *Murinae* embryonic fibroblast are available [here](https://github.com/xulabs/aitom/files/11392586/DISCA_subtomogram_averages.zip) as .mrc files.

## BibTeX

If you use or modify the code from this project in your project, please cite:
```bibtex
@article{zeng2023high,
  title={High-throughput cryo-ET structural pattern mining by unsupervised deep iterative subtomogram clustering},
  author={Zeng, Xiangrui and Kahng, Anson and Xue, Liang and Mahamid, Julia and Chang, Yi-Wei and Xu, Min},
  journal={Proceedings of the National Academy of Sciences},
  volume={120},
  number={15},
  pages={e2213149120},
  year={2023},
  publisher={National Acad Sciences}
}
```
Thank you!
