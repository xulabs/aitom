# AL-GradNorm
Pytorch implementation of our paper: *Boosting Active Learning via Improving Test Performance*, AAAI 2022. 


## Introduction
* With the aid of the influence function, we derive that unlabeled data of higher gradient norm should be selected
for annotation in active learning.
* This work explains why some data can benefit test performance whereas some data cannot. 


## Requirements
numpy

pytorch 1.7+

torchvision 0.10+

## Data Prep
Download the dataset (e.g. Cifar-10) and unzip it to your preferred folder. 

Specify the path of your folder in line 43-45 of the `main.py` file. 

## Run the Code 
`python main.py`

## Notes
The config.py file includes all the hyper-parameters. Set SCHEME=0 for the expected-gradnorm scheme and SCHEME=1 for the entropy-gradnorm scheme. Set NUM_CLASS accordingly, for example, 10 for Cifar10.

Currently, the TRIALS is set to 3 in config.py because the reported results are averaged over 3 runs. One can change it to 1 if using 3 GPUs to run the program concurrently.

## Citation 
`@inproceedings{wang2022boosting,
  title={Boosting Active Learning via Improving Test Performance}, 
  author={Wang, Tianyang and Li, Xingjian and Yang, Pengkun and Hu, Guosheng and Zeng, Xiangrui and Huang, Siyu and Xu, Cheng-Zhong and Xu, Min},  
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},  
  year={2022}
 }`
 
## Contact
[Tianyang Wang](https://tianyangwang.org/)

toseattle@siu.edu
