# List of publications and locations of corresponding code in AITom

**Particle Picking**

1. Pei L, Xu M, Frazier Z, Alber F. Simulating Cryo-Electron Tomograms of Crowded Mixtures of Macromolecular Complexes and Assessment of Particle Picking. BMC Bioinformatics. 2016; 17: 405 [code](https://github.com/xulabs/aitom/tree/master/aitom/simulation/tomogram/single_bounding_sphere) (pei2016simulating)

**Subtomogram Classification**

1. Xu M, Chai X, Muthakana H, Liang X, Yang G, Zeev-Ben-Mordehai T, Xing E. Deep learning based subdivision approach for large scale macromolecules structure recovery from electron cryo tomograms.  ISMB 2017, Bioinformatics [doi:10.1093/bioinformatics/btx230](http://dx.doi.org/10.1093/bioinformatics/btx230). [code](https://github.com/xulabs/aitom/blob/master/aitom/classify/deep/supervised/cnn/subdivide.py) (xu2017deep)
2. Liu C, Zeng X, Wang K, Guo Q, Xu M. Multi-task learning for macromolecule classification, segmentation and coarse structural recovery in cryo-tomography. 2018, arXiv preprint arXiv:1805.06332 [code](https://github.com/xulabs/aitom/tree/master/aitom/classify/deep/supervised/cnn/mult_task) (liu2018multi)
3. Lin R, Zeng X, Kitani K, Xu M. Adversarial domain adaptation for cross data source macromolecule *in situ* structural classification in cellular electron cryo-tomograms [code](https://github.com/xulabs/aitom/tree/master/aitom/classify/deep/supervised/cnn/domain_adaptation_adversarial) (lin2019adversarial)
4. Che C, Xian Z, Zeng X, Gao X, Xu M. Domain Randomization for Macromolecule Structure Classification and Segmentation in Electron Cyro-tomograms, 2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2019: 6-11. [code](https://github.com/xulabs/aitom/tree/master/aitom/classify/deep/supervised/cnn/domain_randomization) (che2019domain)
5. Yu L, Li R, Zeng X, Wang H, Jin J, Yang G, Jiang R, Xu M. Few Shot Domain Adaptation Macromolecule Classification]{Few Shot Domain Adaptation for in situ Macromolecule Structural Classification in Cryo-electron Tomograms. Bioinformatics (2020). [doi:10.1093/bioinformatics/btaa671](https://doi.org/10.1093/bioinformatics/btaa671). [arXiv:2007.15422](https://arxiv.org/abs/2007.15422) [code](https://github.com/xulabs/aitom/tree/master/aitom/classify/deep/supervised/cnn/few_shot/domain_adaptation) (yu2020few)
6. Li R, Yu L, Zhou B, Zeng X, Wang Z, Yang X, Zhang J, Gao X, Jang R, Xu M. Few-shot learning for classification of novel macromolecular structures in cryo-electron tomograms. PLOS Computational Biology. [doi:10.1371/journal.pcbi.1008227](https://doi.org/10.1371/journal.pcbi.1008227) [code](https://github.com/xulabs/aitom/tree/master/aitom/classify/deep/supervised/cnn/few_shot/protonet) (li2020few)
7. Zhou B, Yu H, Zeng X, Yang X, Zhang J, Xu M. One-shot Learning with Attention-guided Segmentation in Cryo-Electron Tomography. Frontiers in Molecular Biosciences. [doi:10.3389/fmolb.2020.613347](https://doi.org/10.3389/fmolb.2020.613347) [code](https://github.com/xulabs/aitom/tree/master/aitom/classify/deep/supervised/cnn/one_shot) (zhou2020one)
8. Du X, Wang H, Zhu Z, Zeng X, Chang Y, Zhang J, Xing E, Xu M. Active learning to classify macromolecular structures in situ for less supervision in cryo-electron tomography. Bioinformatics. [doi:10.1093/bioinformatics/btab123](https://doi.org/10.1093/bioinformatics/btab123) [arXiv:2102.12040](https://arxiv.org/abs/2102.12040) [code](https://github.com/xulabs/aitom/tree/master/aitom/classify/deep/supervised/cnn/active/hal) 

**Subtomogram Segmentation**

1. Xu M, and Frank A. Automated target segmentation and real space fast alignment methods for high-throughput classification and averaging of crowded cryo-electron subtomograms. Bioinformatics 29, no. 13 (2013): i274-i282 [code](https://github.com/xulabs/aitom/tree/master/aitom/segmentation) (xu2013automated)
2. Zeng X, Leung M, Zeev-Ben-Mordehai T, Xu M. A convolutional autoencoder approach for mining features in cellular electron cryo-tomograms and weakly supervised coarse segmentation . Journal of Structural Biology. 2018 May;202(2):150-160. doi:10.1016/j.jsb.2017.12.015 [code](https://github.com/xulabs/projects/tree/master/autoencoder) (zeng2018convolutional)
3. Zhao G, Zhou B, Wang K, Jiang R, Xu M. Respond-CAM: Analyzing Deep Models for 3D Imaging Data by Visualizations. Medical Image Computing & Computer Assisted Intervention (MICCAI) 2018. [code]() (zhao2018respond)
4. Yang Y, Ma Y, Zhang J, Gao X, Xu M. AttPNet: Attention-Based Deep Neural Network for 3D Point Set Analysis. Sensors, 2020, 20(19): 5455. [code](https://github.com/xulabs/aitom/tree/master/aitom/segmentation/point_cloud/deep/attpnet) (yang2020attpnet)
5. Chen F, Jiang Y, Zeng X, Zhang J, Gao X, Min X. PUB-SalNet: A Pre-Trained Unsupervised Self-Aware Backpropagation Network for Biomedical Salient Segmentation. Algorithms 13.5 (2020): 126 [code](https://github.com/xulabs/aitom/tree/master/aitom/segmentation/saliency/deep/unsupervised/pub_salnet) (chen2020pub)

**Subtomogram Alignment and Averaging**

1. Xu M, Martin B, Frank A. High-throughput subtomogram alignment and classification by Fourier space constrained fast volumetric matching. Journal of structural biology 178, no. 2 (2012): 152-164. [code](https://github.com/xulabs/aitom/tree/master/aitom/align/fast) (xu2012high-throughput)
2. Zeng X, Xu M. Gum-Net: Unsupervised geometric matching for fast and accurate 3D subtomogram image alignment and averaging. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR 2020). [code](https://github.com/xulabs/aitom/tree/master/aitom/align/deep/gum) (zeng2020gum)

**Tomogram Segmentation and Object Detection**
1. Zhou B, Guo Q, Zeng X, Gao X, Xu M. Feature Decomposition Based Saliency Detection in Electron Cryo-Tomograms. arXiv:1801.10562. IEEE International Conference on Bioinformatics & Biomedicine, Workshop on Machine Learning in High Resolution Microscopy (BIBM-MLHRM 2018) [code](https://github.com/xulabs/aitom/tree/master/aitom/segmentation/saliency/feature_decomposition) (zhou2018feature)
2. Li R, Zeng X, Siegmund S, Lin R, Zhou B, Liu C, Wang K, Jiang R, Freyberg Z, Lv H, Xu M. Automatic Localization and Identification of Mitochondria in Cellular Electron Cryo-Tomography using Faster-RCNN. BMC Bioinformatics. 201920 (Suppl 3) :132 doi:10.1186/s12859-019-2650-7. [code](https://github.com/xulabs/aitom/tree/master/aitom/segmentation/detection/organelle/frcnn) (li2019automatic)

**Tomominer**

1. Xu M, Singla J, Tocheva E, Chang Y, Stevens R, Jensen G, Alber F.  De novo structural pattern mining in cellular electron cryo-tomograms.  Structure. 2019 Apr 2;27(4):679-691.e14.[code](https://github.com/xulabs/aitom/tree/master/aitom/tomominer) (xu2019novo)
2. Frazier Z, Xu M, Alber F. Tomominer and tomominer cloud: A software platform for large-scale subtomogram structural analysis. Structure, Volume 25, Issue 6, p951–961.e2, 6 June 2017  [code](https://github.com/xulabs/aitom/tree/master/aitom/tomominer) (frazier2017tomominer)

