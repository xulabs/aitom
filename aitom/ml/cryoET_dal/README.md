\*\*\* This sample code is for the simulated cryo-ET dataset of 50 classes with a SNR of 0.05 \*\*\*


## Environments:

1. Python 3.8   2. Pytorch 1.7   3. mrcfile 1.3



## How to use the code:

1. Specify the data path in line 28 of the main.py 

2. Run this command to train the task model and test it

   python main.py --num_classes 50 --target_snr_type "SNR005"

3. Go to config.py to setup hyper-parameters if needed. For example, the experiment 
   currently runs for 3 trials on a single GPU. The 'TRIAL' can be changed to 1 if
   running the experiment with 3 GPUs concurrently. 
   
4. python main.py 



## Data availability:

We refer readers to the following paper for data preparations. 

```
@inproceedings{liu2020efficient,
  title={Efficient cryo-electron tomogram simulation of macromolecular crowding with application to SARS-CoV-2},
  author={Liu, Sinuo and Ma, Yan and Ban, Xiaojuan and Zeng, Xiangrui and Nallapareddy, Vamsi and Chaudhari, Ajinkya and Xu, Min},
  booktitle={2020 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={80--87},
  year={2020},
  organization={IEEE}
}
```

## Contact
[Tianyang Wang](https://tianyangwang.org/)

toseattle@siu.edu
