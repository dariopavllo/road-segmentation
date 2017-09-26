# Satellite Image Segmentation with Convolutional Neural Networks
##### Dario Pavllo, Mattia Martinelli, and Chanhee Hwang
This project presents a technique for segmenting satellite images (labeling each pixel) by using convolutional neural networks. It was developed as final work for the course "Pattern Classification and Machine Learning" taught at EPFL in 2016. In this scenario, the neural net implements a binary classifier that labels each pixel as road or background, but can be adapted to detect any kind of feature (provided that a suitable training set is available).

Our team achieved an accuracy of 93.6% on the test set, and reached the first place on the Kaggle competition organized by EPFL. The details of our technique are explained in the report `report.pdf` and in the poster `conference_poster.pdf`. The results were presented at the Applied Machine Learning Days 2017 in Lausanne.

This repository includes a pre-trained model that can be used to generate predictions, as well as the code to train the model from scratch. The original training and test sets can be downloaded from the Release section.

### Libraries
The following libraries must be installed to run the project:

- Keras 1.1.2
- Theano 0.8.2

Keras is a deep learning library that can use either Theano or TensorFlow as backend. We used Theano, and therefore, this project must be run with Theano backend to ensure reproducibility (since the model weights are saved according to Theano's structure). If it not already so, Keras must be configured to use Theano by editing the `keras.json` file.

### Setup
The model has been trained using GPU acceleration, with the following setup:

- Windows 8.1 x64
- Intel Core i5-4460 @3.2 GHz
- NVIDIA GeForce GTX 960 (with 2 GB of RAM)
- 16 GB of system memory
- GPU Drivers: ForceWare 369.30
- Keras 1.1.2 with Theano 0.8.2 backend + CuDNN 5.1
- Theano flags: fastmath = True, optimizer = fast_run, floatX = float32


### How to run
To avoid re-training the model, we have provided its weights in the file `weights.h5`. Therefore, to generate the predictions, it is only necessary to run the script `run.py`. The test set images must be put in the directory `test_set_images`.

Theano can be configured to use either the CPU or the GPU (both the methods have been tested and will produce the same results). In the first case, a BLAS library should be used.

Sample configuration for `.theanorc` (using CPU):
```
[global]
device = cpu
floatX = float32

[blas]
ldflags=-LC:\\openblas\\bin -LC:\\openblas\\lib -lopenblas
```

Sample configuration for GPU:
```
[global]
device = gpu
floatX = float32
optimizer_including=cudnn
optimizer_excluding=low_memory
allow_gc=False
optimizer = fast_run

[lib]
cnmem = 0.8

[nvcc]
fastmath = True
compiler_bindir=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin
```

### How to train
We have provided a notebook `train.ipynb` that can be used to train the model from scratch. Depending on the available computational power, the process can take several hours.

### Description of the files
All models are grouped into classes, in order to improve code readability and reusability. The following three models have been supplied:

- `naive_model.py`: this model classifies all patches as background, and has been used a baseline and for debug purposes.
- `logistic_model.py`: classifier based on logistic regression.
- `cnn_model.py`: classifier based on convolutional neural networks. This file contains the neural network structure, as well as the training parameters.

Furthermore, the following files contain utility methods:
- `helpers.py`: contains the image processing methods.
- `cross_validation.py`: utilities for cross-validation (including k-fold cross validation).

Finally, the following scripts are the ones that can be run:
- `run.py`: reads the weights from a file and performs classification on the test set, using convolutional neural networks.
- `train.ipynb`: this notebook can be used to train the network from scratch.
- `validation.ipynb`: this toolkit can be used to perform cross-validation (either static or full k-fold) on a certain model.
- `visualization.ipynb`: this notebook can be used to evaluate how the neural network actually performs. It contains methods for viewing the road segmentation results, as well as the convolution filters.
