The demo single particle tomogram can be downloaded from: https://cmu.box.com/s/l6ij6ntbqj52tj6m1o8krtlnv0cughbl.

![image](https://user-images.githubusercontent.com/17937329/69974384-221a9f80-14f3-11ea-9681-f5f4bbd82029.png)

'original.png': One slice from the de-noised tomogram.

![image](https://user-images.githubusercontent.com/17937329/69974383-21820900-14f3-11ea-9d5a-4bab82fab94a.png)

'SLIC.png': SLIC supervoxel over-segmentation result.

![image](https://user-images.githubusercontent.com/17937329/69974382-21820900-14f3-11ea-8557-b6d67d1925dc.png)

'saliency_map': Visualization of the slice of saliency map (saliency level below (max+min)/2 set to 0).

Above results are under such conditions on a Ubuntu 16.04 server:

gaussian_sigma=2.5, gabor_sigma=9.0, gabor_lambda=9.0, cluster_center_number=10000

Reference:
Zhou B, Guo Q, Zeng X, Gao X, Xu M. Feature Decomposition Based Saliency Detection in Electron Cryo-Tomograms. [arXiv:1801.10562](https://arxiv.org/abs/1801.10562). IEEE International Conference on Bioinformatics & Biomedicine, Workshop on Machine Learning in High Resolution Microscopy (BIBM-MLHRM 2018). [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6571026/)

