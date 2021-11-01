# End-to-End Robust Joint Unsupervised Image Alignment and Clustering

### Dependencies

This code is implemented in Python 3.8.10. All required packages can be found in requirements.txt.

### Training

The model is trained with three separate scripts in the `scripts` directory that should be executed in the following order. 1) `train_strong_random_affine_pascal.sh`, which pretrains the affine alignment branch of Jim-Net. 2) `train_strong_random_tps_pascal.sh`, which pretrains the thin plate spline alignment branch of Jim-Net. 3) `train_jim_net.sh`, which trains all branches of Jim-Net.

Steps 1-2 are trained using the self-supervised method proposed in [Convolutional neural network architecture for geometric matching](http://www.di.ens.fr/willow/research/cnngeometric/).

### Evaluation

Use the `test_jim_net.sh` script to evaluate Jim-Net on PF-Pascal test set. 

## BibTeX 

This code was built ontop of the code provided in
````
@inproceedings{rocco2018end,
  title={End-to-end weakly-supervised semantic alignment},
  author={Rocco, Ignacio and Arandjelovi{\'c}, Relja and Sivic, Josef},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6917--6925},
  year={2018}
}
````


