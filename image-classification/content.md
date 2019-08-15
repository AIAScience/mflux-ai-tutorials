# Image classification

This tutorial shows how you can train a image classification model. We will develop a Convolutional Neural Network (CNN) for the classification. We use the MNIST dataset which is an image dataset of handwritten digits. It has has 60,000 training images and 10,000 test images, each of which are grayscale 28 x 28 sized images.

## Log in to MFlux.ai

## Install Anaconda on your computer

Download and install Anaconda. Select the Python 3.* version):
https://www.anaconda.com/download/

When Anaconda is installed, open "Anaconda Prompt" or any other terminal where you have ```conda``` available now.


## Make an isolated Python environment
Run ```conda create --name image-classification python=3.6``` in your terminal.
Then, to activate your new environment, run ```conda activate image-classification```.


##  Install the required packages

Run ```pip install mlflow[extras] mflux-ai Keras==2.2```  in your terminal.


## Loading imports

```python
import sys
import warnings

import keras
import numpy as np
from keras import backend as K

import mlflow.sklearn

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.datasets import mnist
from keras import optimizers

import mflux_ai
```


##  Loading and pre-processing data

The Keras library provides a database of these digits in its keras.datasets module. First we load the MNIST data into train and test sets.

```python
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
```
Next, we do some pre-processing of the images. We reshape the images to a tensor of shape (num_samples image_height, image_width, num_channels), i.e (num_samples, 28, 28, 1) where num_samples = 60,000 for train dataset and num_samples = 10,00 for test dataset.

```python

img_rows, img_cols = 28, 28
num_classes = 10

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
```


We will re-scale the image data to a value between 0.0 and 1.0.
```python
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

```

Also, we will one-hot-encode the labels.

```python
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

## Define and train a CNN model
Define a CNN model
 ```python
        # Define model architecture
model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=input_shape))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
```


Define the optimizer.
```python
adam = optimizers.Adam()
```

Compile the model.

```python
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
```


Fit the model using the training dataset.
```python
batch_size = 20
nb_epoch=10
model.fit(x_train, y_train,
          batch_size=20, nb_epoch=10, verbose=1)
 ```
Evaluate the model using test data
 ```python
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
 ```

## Log metrics and store machine learning model in MFLux.ai


Let's log the parameters, validation loss and accuracy metric and store the model in MFlux.ai

 ```python

mflux_ai.set_env_vars("Your_key")

mlflow.log_param("batch_size", batch_size)
mlflow.log_param("nb_epochs", nb_epochs)
mlflow.log_metric("loss", score[0])
mlflow.log_metric("accuracy", score[1])

mlflow.sklearn.log_model(model, "model")
```

## Check your tracking UI

You should now be able to see the metric and model that you logged in your MLflow tracking UI


