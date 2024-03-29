# Subtomogram Classification

## 1. Configure environment

### 1.1. Create environment
```
conda create -n subtomogram_Classification python=3.9
```

### 1.2. Activate environment
```
conda activate subtomogram_Classification
```

### 1.3. Install dependencies
```
torch==1.13.1
tqdm==4.66.2
mrcfile==1.5.0
SimpleITK==2.3.1
natsort==8.4.0
pillow==10.2.0
```

## 2. Data

### 2.1. Prepare data
You can first put all files into ./data/train, just like the following example, where the MRC file is the input Subtomogram file, and the JSON file is the corresponding classification label.
> Notes: 
You need to maintain the order for each class. For example, if each class has 500 items, then 0-499 belongs to the first class, and 500-999 belongs to the second class.
```
./
└── data
      └── train
            ├── subtomogram_mrc
                  ├── tomotarget1001.mrc
                  ├── tomotarget1002.mrc
                  ├── ...
            └── json_label
                  ├── target1001.json
                  ├── target1002.json
                  ├── ...
```

Sample data (simulated data) can be found [here](https://drive.google.com/drive/folders/18E6VlejGMbyihr2tQ06-amE8XLE8QbX5).

### 2.2. Generate validation and test set
The default parameter setting is train:val:test=7:2:1. You can adjust the generation of the validation set and test set using the following parameters.
```
-- num_class
You can adjust the quantity for each class. For example, if each class has 500 items, then this parameter would be 500.
-- val_rate
The proportion of the validation set relative to the dataset.
-- test_rate
The proportion of the test set relative to the dataset.
```
Below is an example of the generated result.
```
./
└── data
      └── train
            ├── subtomogram_mrc
                  ├── tomotarget1001.mrc
                  ├── tomotarget1002.mrc
                  ├── ...
            └── json_label
                  ├── target1001.json
                  ├── target1002.json
                  ├── ...
      └── val
            ├── subtomogram_mrc
                  ├── tomotarget1023.mrc
                  ├── tomotarget1032.mrc
                  ├── ...
            └── json_label
                  ├── target1023.json
                  ├── target1032.json
                  ├── ...
      └── test
            ├── subtomogram_mrc
                  ├── tomotarget1048.mrc
                  ├── tomotarget1052.mrc
                  ├── ...
            └── json_label
                  ├── target1048.json
                  ├── target1052.json
                  ├── ...
```

## 3. Train

### 3.1. Model training
The following are explanations of the training parameters.
```
--arch
Choose the model you want to use. Currently, there are three models available: RB3D, DSRF3D_v2, and YOPO. You can customize new models in the model folder.
--lr 
Learning rate.If you're customizing your model, you can use the default learning rate, but you can also adjust the learning rate to achieve a better model performance.
--schedule 
Learning rate schedule.
--epochs 
Number of total epochs to run.
--gpu 
Select the GPU you want to use.
```

### 3.2. Customize model
You can customize your model in `2.1 Model Definition`. Then add it to the following list.
```
model_dictionary = {'RB3D': RB3D, 
                    'DSRF3D_v2': DSRF3D_v2,
                    'YOPO': YOPO}
```

### 3.3. Customize dataset
You can use a custom dataset, employing the methods we mentioned, but you might need to adjust the following parameters in `1.3 Data load`.
```
-- data augmentation
We provide default data augmentation, but if you want to increase the diversity of the model, you can adjust the data augmentation based on your dataset.
-- normalize
We provide default parameters for general normalization, but you may also need to adjust them based on your dataset.
```


## 4. test


You can test your model in `2.3 test`.
The following are adjustable parameters.
```
-- arch
The model you want to test.
-- gpu
choose the gpu to test
-- checkpoint_path
The path where the model is saved
```



