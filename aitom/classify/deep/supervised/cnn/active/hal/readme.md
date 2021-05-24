# Active Learning to Classify Macromolecular Structures in situ for Less Supervision in Cryo-Electron Tomography

This repository contains the code used to run the active learning experiments detailed in our paper.

## Dependencies

In order to run our code, you'll need these main packages:

- [Python](https://www.python.org/)>=3.5
- [Numpy](http://www.numpy.org/)>=1.14.3
- [Scipy](https://www.scipy.org/)>=1.0.0
- [TensorFlow](https://www.tensorflow.org/)==1.13.1
- [Keras](https://keras.io/)==2.2.4

## Running the Code

The code is run using the main.py file in the following way:

    python main.py --lambda_e xxx --classes xxx --batch_size xxx --initial_size xxx --iterations xxx --gpu xxx

- lambda_e: a hyperparameter that balances entropy and discriminative scores.
- classes: the number of classes for classification.
- batch_size: the size of the batch of examples to be labeled in every iteration.
- initial_size: the amount of labeled examples to start the experiment with (chosen randomly).
- iterations: the amount of active learning iterations to run in the experiment.

