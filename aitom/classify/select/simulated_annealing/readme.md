# A simulated annealing approach for resolution guided homogeneous cryo-electron microscopy image selection

### Background

It is significant for improving quality of 3D reconstruction of large macromolecular complexes to select homogeneous images from all obtained projection images obtained by Cryo-electron microscopy (Cryo-EM) and tomography (Cryo-ET) and then gain higher resolution by image averaging.

A simulated annealing-based algorithm (SA) performs better than baseline methods, image matching algorithm (MA) and genetic algorithm (GA), on both accuracy and speed after testing on simulated and experimental datasets. 

#### Reference
Shi J, Zeng X, Jiang R, Jiang T, Xu M. A simulated annealing approach for resolution guided homogeneous cryo electron microscopy image selection. Quantitative Biology. doi:10.1007/s40484-019-0191-8

### Environment

Python 2.7

Install packages: pickle, pyExcelerator, xlrd, xlutils

### Dataset

This project uses labelled data. The dataset is divided into homogeneous part and heterogeneous part for both 2D Cryo-EM image and 3D Cryo-ET image. 

Save image sets as Python dictionary and compress them by "pickle". Each item in the dictionary describes an image. For 3D dataset, each image is also saved as a dictionary like "{'v': , 'm' }".
* 'v' - value of the 3D image
* 'm' - mask of the 3D image

Finally, name homogeneous image set "true__out__images.pickle" and heterogeneous image set "rotation_var__out__images.pickle"

In this project, we set same numbers of images in homogeneous set and heterogeneous set.

### Parameters

| Plugin | README |
| ------ | ------ |
| ITERATION | number of iterations |
| BETA | beta used in F-meature |
| RATIO | ratio = number of image selected from homogeneous image set / number of image selected from heterogeneous image set|
| NUM_FIRST_G | number of candidate in the first generation (just used in genetic algorithm) |

### Usage

Files in the folder "2d" are used on 2D Cryo-EM image dataset, and files in the folder "3d" are aimed at 3D Cryo-ET image dataset.

For 2D dataset,
 - put two data files in the folder "2d"
 ```sh
 $ cd 2d
 ``` 
 - If choosing SA, adjust parameters (ITERATION, BETA, RATIO) in "sa.py", then run the program.
 ```sh
 $ python sa.py
 ```
 - If choosing GA, adjust parameters (ITERATION, BETA, RATIO, NUM_FIRST_G) in "ga.py", then run the program.
 ```sh
 $ python ga.py
 ```
- If choosing MA, adjust parameters (BETA, RATIO) in "ma.py", then run the program.
 ```sh
 $ python ma.py
 ```
 
For 3D data, 
 - put two data files in the folder "3d"
 ```sh
 $ cd 3d
 ``` 
 - If choosing SA, adjust parameters (ITERATION, BETA, RATIO) in "sa.py", then run the program.
 ```sh
 $ python sa.py
 ```
 - If choosing GA, adjust parameters (ITERATION, BETA, RATIO, NUM_FIRST_G) in "ga.py", then run the program.
 ```sh
 $ python ga.py
 ```
 - If choosing MA, adjust parameters (BETA, RATIO) in "ma.py", then run the program.
 ```sh
 $ python ma.py
 ```
 - If choosing SA with restarting, adjust parameters (ITERATION, BETA, RATIO) in "sa_restart.py", then run the program.
 ```sh
 $ python sa_restart.py
 ```

### Output

The program will write results in a file named "record.xls" as the following format.

| resolution | precision | recall | F_beta | time |
| ------ | ------ | ------ | ------ | ------ |
| ... | ... | ... | ... | ... |

Meanwhile, it will also print resolution and numbers of final selected images.

