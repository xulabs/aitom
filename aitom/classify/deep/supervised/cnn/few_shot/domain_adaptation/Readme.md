# The usage of our code in Few Shot Domain Adaptation for in situ Macromolecule Structural Classification in Cryo-electron Tomograms 


## 1. the dependency of our code

* Python 3.7.3
* Pytorch 1.1.0
* CUDA 10.0.130
## 2. The description of the code

### main.py: the main script to run our code
### CORAL.py: the script of  Unsupervised Domain Adaptation
### CECT_dataloader.py: the script to process CECT dataset
### models: the defination of our model

## 3. run our code

```
python main.py
```

Specifically, we explain some of the parameters of main.py:

* n_target_samples: the number of samples in target domain in stage 3

* classes_num: the total of category

* CORAL_batch_size：the minibatch of Deep CORAL

## Reference

Yu L, Li R, Zeng X, Wang H, Jin J, Yang G, Jiang R, Xu M. Few Shot Domain Adaptation Macromolecule Classification]{Few Shot Domain Adaptation for in situ Macromolecule Structural Classification in Cryo-electron Tomograms. Bioinformatics (2020). [doi:10.1093/bioinformatics/btaa671](https://doi.org/10.1093/bioinformatics/btaa671). [arXiv:2007.15422](https://arxiv.org/abs/2007.15422). [code](https://github.com/xulabs/aitom/tree/master/aitom/classify/deep/supervised/cnn/few_shot/domain_adaptation)
