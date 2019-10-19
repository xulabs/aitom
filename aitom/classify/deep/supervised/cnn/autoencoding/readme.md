Source Codes for Autoencoding classifier
The files are organized as follows:

- auto_classifier.py: main program of autoencoding classifier
- auto_classifier_model.py: model definition of autoencoding classifier
- utils.py: helper functions to load and visualize subtomograms

##Train
you can run the program by:
python auto_classifier.py dataset option

- dataset: path of your dataset
- option: train or test

For different formats of dataset, you need to modify the function preprocess in auto_classifier.py to support the formats of your dataset.