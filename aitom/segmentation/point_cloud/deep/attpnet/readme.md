# AttPNet

Attention-based Point Network (AttPNet) uses attention mechanism for both global feature masking and channel weighting to focus on characteristic regions and channels. There are two branches in our model. The first branch calculates an attention mask for every point. The second branch uses convolution layers to abstract global features from point sets, where channel attention block is adapted to focus on important channels.


![avatar](https://www.mdpi.com/sensors/sensors-20-05455/article_deploy/html/images/sensors-20-05455-g002.png)

Evaluations on the ModelNet40 benchmark dataset show that our model outperforms the existing best model in classification tasks by 0.7% without voting. In addition, experiments on augmented data demonstrate that our model is robust to rotational perturbations and missing points.

Please refer to our paper for more details:

> Yang, Y.; Ma, Y.; Zhang, J.; Gao, X.; Xu, M. AttPNet: Attention-Based Deep Neural Network for 3D Point Set Analysis. Sensors 2020, 20, 5455. [MDPI Access](https://www.mdpi.com/1424-8220/20/19/5455)

# Package Versions

- torch==1.0.0
- h5py==2.8.0
- tqdm==4.32.1

# Installation

Please follow the installation guide of AITom.

Alternatively, you could download all the scripts in aitom/segmentation/point_cloud/deep/attpnet/ and modify the lines for importing modules to run AttPNet independently.

# Demo

## Dataset

For the classification task, we evaluate our model on **ModelNet40** and **Electron Cryo-Tomography (ECT)**. For the part segmentation task, we evaluate our model on **ShapeNet**.


- The ModelNet40 dataset is made up of 40 common object categories with 100 CAD models per category, among which all the point sets are augmented by scaling, translation, and shuffling.

- The single-particle ECT dataset consists of 3D images of seven classes of macrocellular structures. We apply constant sampling to generate 400 point cloud data for each class. Compared with other general point cloud dataset, the structures between different classes in ECT dataset are more similar to each other

-  The ShapeNet dataset contains 16 categories of objects and consists of 50 different parts in total. Each category has been annotated with two to six parts unequally. The training and testing 3D point sets are 14,006 and 2874, respectively. 

## Training Code

To run a demo on the ModelNet40 dataset:

`python train.py --batchSize XX`

where argument `batchSize` is the size of input data batch.

Output:

```
final accuracy 0.9359805510534847
Class Acc:
0.9901 1.0000 0.9802 0.8824 0.9314 ... 0.8182
mA: 0.9017
```

where `final accuracy` is the overall accuracy of all 40 categories. `Class Acc` are the results of 40 categories respectively. `mA` is the mean value of all Class Acc. 


# BibTeX

If you use or modify the code from this project in your project, please cite:

> @article{Yang_2020, title={AttPNet: Attention-Based Deep Neural Network for 3D Point Set Analysis}, volume={20}, ISSN={1424-8220}, url={http://dx.doi.org/10.3390/s20195455}, DOI={10.3390/s20195455}, number={19}, journal={Sensors}, publisher={MDPI AG}, author={Yang, Yufeng and Ma, Yixiao and Zhang, Jing and Gao, Xin and Xu, Min}, year={2020}, month={Sep}, pages={5455}}

Thank you!

