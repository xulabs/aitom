# Remote 3D Annotation

A jupyter notebook platform that allows users to interactively annotate CryoET files in .mrc or .rec form stored on remote server. Multiple annotation could be made on a single file. Each file is visualized as multiple xy-plane slices which can be navigated through the z bar. Z-direction boundaries can be manually set by manually selecting the values for z_lower and z_upper bars where the xy-direction boundaries can be interactively set by clicking and dragging mouse on the current slice shown and can be adjusted later. 


## Requirements

matplotlib (Don't use matplotlib==3.3.2)
numpy
ipywidgets
skimage


## Annotation Steps
1. Open jupyter notebook from a remote server. Specific instruction can be found [here](https://docs.anaconda.com/anaconda/user-guide/tasks/remote-jupyter-notebook/) 
2. Import the CryoET data from appropriate path
3. Start the annotation. Skim through all slices through z bar and confirm z_lower, z_upper
4. Click the "Newbox" button to enable interactive box drawing on xy-plane, then unclick it
5. Draw bounding box use mouse and change z to adjust the size of the box
6. Enter label for the object
7. Click "FinishCurrentBox" to save the current annotation result, then unclick it
8. Repeat step 3-7 for each potential object contained in the file
9. When finished all annotation save the results


## Annotation results
Annotation results will be stored in the "AnnotationResults" folder. For each file annotated, there will be two corresponding file generated: filename_resized.npy, where stores the resized 3D CryoET data that is 0.2 times of the original x, y, z; and filename_annotation.csv, which each row stores an individual object in the CryoET file and there are a total of 7 column that stores the object’s name, xmin, ymin, z_lower, x-span,y-span, and z-span respectively. 


