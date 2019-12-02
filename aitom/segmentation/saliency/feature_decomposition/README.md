The demo single particle tomogram can be downloaded from: https://cmu.box.com/s/9hn3qqtqmivauus3kgtasg5uzlj53wxp.

![image](https://github.com/zhuzhenxi/aitom_doc/blob/master/tutorials/015_saliency_detection/original.png)

'original.png': One slice from the de-noised tomogram.

![image](https://github.com/zhuzhenxi/aitom_doc/blob/master/tutorials/015_saliency_detection/SLIC.png)

'SLIC.png': SLIC supervoxel over-segmentation result.

![image](https://github.com/zhuzhenxi/aitom_doc/blob/master/tutorials/015_saliency_detection/saliency_map.png)

'saliency_map': Visualization of the slice of saliency map (saliency level below (max+min)/2 set to 0).

Above results are under such conditions on a Ubuntu 16.04 server:

gaussian_sigma=2.5, gabor_sigma=9.0, gabor_lambda=9.0, cluster_center_number=10000
