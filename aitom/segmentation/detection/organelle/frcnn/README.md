# Automatic localization and identification of mitochondria in cellular electron cryo-tomography using faster-RCNN
Code for the paper [Automatic localization and identification of mitochondria in cellular electron cryo-tomography using faster-RCNN], which proposed a simple yet effective automatic image analysis approach based on Faster-RCNN to automatically localize and recognize in situ cellular structures of interest captured by cryo-ET.

## Requirements
Basically, this code supports both python2.7 and python3.5, the following package should installed:
* tensorflow
* keras
* scipy
* cv2
* eman2 for tomo2jpeg.py

## Data Preparation
tomo2jpeg.py
The projection images in the .mrc or .st files will be saved into JPEG images for later Faster RCNN training and test.

image_preprocessing.py
The images will be preprocessed, first Bilateral filtering, then Histogram Equalization. The parameters and metrics can be changed if needed.

xml2simplefile.py
The annotation files(.xml file for me) should be converted into this format:
`/path/to/img.jpg,x1,y1,x2,y2,class_name`
where x1,y1,x2,y2 is the coordinates of the bounding box. All the annotations will be saved into a TXT file.(like [mito_simple_label.txt](https://github.com/xulabs/aitom/files/5862323/mito_simple_label.txt))

data_split.py
The dataset will be splited into training set and test set in a ratio of 5:1, and saved into two TXT files.([mito_train_label.txt](https://github.com/xulabs/aitom/files/5862325/mito_train_label.txt) and [mito_test_label.txt](https://github.com/xulabs/aitom/files/5862324/mito_test_label.txt) for example)

(You may need to make some adjustments to the paths in the .py files above and regenerate the simple files since the path to the images are changed.)

## Train
After data preparation, simply run:
```
python train_frcnn_mito.py
```
The model will be saved in .hdf5 file.

## Predict
If you want see how good your trained model is, simply run:
```
python test_frcnn_mito.py
```
You can also use `-p` to choose a single image to predict, or send a path contains many images, and `-m` to choose the model you want to use.
A model trained to detect mitochondria is saved in ./model.

## Evaluate
The measure_map.py file can help you to get the mAP, F1 score, mIoU and precision-recall curve for the model and dataset you provide. You can use `-p` to choose the path to test data, which can be either a simple file or in the format of pascal voc( I chose a simple file `mito_test_label.txt` as default), and `-m` is for the path to the model you want to evaluate.

## Else
visualization.py can help to plot the model we used if you need.

## Reference
The implementation of Faster-RCNN model refers to the the work of Yann Henon in 2017 (https://github.com/yhenon/keras-frcnn).

**That's all, and please feel free to contact me (liranran1226@gmail.com) if you have any question!**