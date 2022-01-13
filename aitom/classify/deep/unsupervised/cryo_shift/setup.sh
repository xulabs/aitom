#!/bin/bash

if [ -d "./data/" ]
then
    echo "Setup has been done before"
else
    echo "Starting Setup"
    sh download.sh
    echo "Setup Complete. Run run_models.sh to train"
fi