#!/bin/bash
# 14:45 start time

if [ -z "$1" ]
then
        echo "Error: No GPU selected"
        echo "Please choose a GPU in the arguments with the -g flag."
        echo "Example Usage: sh run_models.sh 1"
        echo "Exiting."
        exit 0
else
        echo "GPU Chosen: $1";
fi

if [ -z "$2" ]
then
        name="cb3d";
else
        name=$2;
fi

if [ $name == "cb3d" ];
then
    name_cap="cb3d";
fi


if [ $name == "fsda" ];
then
    name_cap="FSDA";
fi

if [ $name == "dense" ];
then
    name_cap="dense3D";
fi

if [ $name == "dsrf" ];
then
    name_cap="dsrf3D";
fi

if [ $name == "resnet" ];
then
    name_cap="resnet3D";
fi


echo "Training for ${name} with name_cap as ${name_cap}"


echo "Training Discriminator"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 python3 step_0.py 
echo "Training Base Network"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 python3 step_1.py $name
echo "Training Final Model"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 python3 step_2.py $name 4
echo "Test Base Performance"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 python3 eval_cls.py running_weights/${name_cap}_basic_cls.pt
echo "Test Final Performance"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 python3 eval_cls.py running_weights/${name_cap}_final_cls.pt
echo "Done"