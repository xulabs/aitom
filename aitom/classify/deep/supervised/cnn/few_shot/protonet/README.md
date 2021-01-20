# Few-shot learning for subtomogram classification

Code for the paper [Few-shot learning for classification of novel macromolecular structures in cryo-electron tomograms]

The code presents two models proposed in the paper: a basic ProtoNet3D for subtomogram classification, and a combined model ProtoNet-CE.

## Training a network

### Install dependencies

* This code has been tested on macOS Mojave 10.14.6  with Python 3.6 and PyTorch 1.5.
* Install [PyTorch and torchvision](http://pytorch.org/).
* Install [torchnet](https://github.com/pytorch/tnt) by running `pip install git+https://github.com/pytorch/tnt.git@master`.
* Other dependencies: numpy (tested with 1.14.3) and tqdm.

### Data preparing
* The dataset should be organized in the following form:
  *Dataset
    *data
      *class_name_1
        *subtomo_1.npy
        *subtomo_2.npy
        *...
        *subtomo_n1.npy
      *class_name_2
      *...
      *class_name_n
    *splits
      *split_name_1
        *train.txt (List the class names in the training set, one row for one class.)
        *val.txt (List the class names in the validation set, one row for one class.)
        *test.txt (List the class names in the test set, one row for one class.)
        *trainval.txt (Optional, combination of the training set and the validation set.)
      *split_name_2 (Optional, if you need different ways of splits.)
        *train.txt
        *val.txt
        *test.txt
        *trainval.txt
      *...

### Train the model

* Run `python run_train.py`. This will run training and place the results into `results`.
  * You can specify a different output directory by passing in the option `--log.exp_dir EXP_DIR`, where `EXP_DIR` is your desired output directory.
  * You can specify the dataset directory by passing in the option `--data.path DATAPATH`, where `DATAPATH` is your desired dataset directory.
  * If you are running on a GPU you can pass in the option `--data.cuda`.
  * If you are training a basic ProtoNet3D model, set the option `--model.stage` to `protonet` as default. If you are training a ProtoNet-CE model, you need to run `python run_train.py` with the default option at first, and then run `python run_train.py --model.stage feat`.
* Re-run in trainval mode `python run_trainval.py`. This will save your model into `results/trainval` by default. (Optional, if you hope to re-train the model with the combination of training set and validation set.)

### Evaluate

* Run evaluation as: `python run_eval.py --model.model_path results/best_model.pt`.
  *If you want to evaluate a basic ProtoNet3D model, you can set the option `--stage` to `protonet` by running `python run_eval.py --model.model_path results/best_model.pt --stage protonet`. By default, the `--stage` is set to `feat` to evaluate the ProtoNet-CE model.
