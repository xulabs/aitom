'''
Tutorial on autoencoder
'''


'''
Step1:Prepare input dataset

You can download the example dataset from {this will be added in the future}, extract it into your present working directory.

Here's four parameters and dataset format.
1.A python pickle data file of CECT small subvolumes, this data file should be prepared as follows:
    d is the small subvolume data file.
    d is a dictionary consists 'v_siz' and 'vs'.
    d['v_siz'] is an numpy.ndarray specifying the shape of the small subvolume. For example, d['v_siz'] = array([32,32,32]).
    d['vs'] is a dictionary with keys of uuids specifying each small subvolume.
    d['vs'][an example uuid] is a dictionary consists 'center', 'id', and 'v'.
    d['vs'][an example uuid]['center'] is the center of the small subvolume in the tomogram. For example, d['vs'][an example uuid]['center'] = [110,407,200].
    d['vs'][an example uuid]['id'] is the specific uuid.
    d['vs'][an example uuid]['v'] are voxel values of the small subvolume, which is an numpy.ndarray of shape d['v_siz'].
2.A tomogram file in .rec format, which is only required when performing pose normalization.

3.Whether the optional pose normalization step should be applied. Input should be True or False.

4.The number of clusters. This should be an positive integer such as 100.

'''


'''
Step2 Train the auto encoder. Given the example dataset, you can use parameters1 or parameters2.
'''
import aitom.classify.deep.unsupervised.autoencoder.autoencoder as AE


parameters1 = ["example/subvolumes_example_2.pickle","None","False","4"]
parameters2 = ["example/subvolumes_example_1.pickle","example/tomogram.rec","True","100"]


d = AE.auto.pickle_load(parameters1[0])#pickle data file of CECT small subvolumes
img_org_file = parameters1[1]#A tomogram file in .rec format, which can be None when pose normalization is not required
pose = eval(parameters1[2])#Whether the optional pose normalization step should be applied  True or False
clus_num = int(parameters1[3])# The number of clusters


AE.encoder_simple_conv_test(d=d, pose=pose, img_org_file=img_org_file, out_dir=AE.os.getcwd(), clus_num=clus_num)
AE.kmeans_centers_plot(AE.op_join(AE.os.getcwd(), 'clus-center'))

'''
Step 3. Manual selection of small subvolume clusters

Autoencoder3D training step will have two output folders.
'model' directory saved the trained models
'clus-center' directory for the resulting clusters.
 
There should be two pickle files in 'clus-center'. 
'kmeans.pickle' stores the uuids for each cluster.
'ccents.pickle' stores the decoded cluster centers.

The 'fig' folder under 'clus-center' directory contains the 2D slices of decoded cluster center. 
User can use the figures as guide for manual selection.
  
Manual selection clues are provided in the folder 'fig' under 'clus-center'.
Each picture is a 2D slices presentation of a decoded small subvolume cluster center.
The picture name such as '035--47.png' refers to cluster 35 which consists 47 small subvolumes.
'''

