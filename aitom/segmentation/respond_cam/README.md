# Summary

It is the code for the following paper:

Zhao G, Zhou B, Wang K, Jiang R, Xu M. Respond-CAM: Analyzing Deep Models for 3D Imaging Data by Visualizations. Medical Image Computing & Computer Assisted Intervention (MICCAI) 2018.

https://arxiv.org/abs/1806.00102


* `cnn_models.py`: structural definations and operations of CNNs used.

* `respond_cam.py`: the implementation of Respond-CAM and Grad-CAM for 2D and 3D data.

* `figure_util.py`: figure generation given the CAM and input data.

* `data/`: a folder for our CNN models and datasets. *Now it is an empty folder; the download link for our data is coming soon*.

* `demo/`: a folder to demonstrate how to use our code. We take our figures and table as examples. For instance, for Figure 3 just `cd demo/` and run `python demo_for_figure_3.py`. *They are unavailable until the CNN models and datasets are downloaded and put into the `data/` folder.*

# Walk-through
<img src="https://user-images.githubusercontent.com/31047726/51214222-437a9900-18eb-11e9-877d-2360bc068cdb.jpg" width="900">


# Key prerequisites

* __MayaVi__

* keras

* tensorflow-gpu

* numpy

* scipy

* cv2


# Example models and datasets

*download link coming soon...*

For running our demo, please download the file above.
