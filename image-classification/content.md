## Tutorial

This tutorial shows how you can use MLFlow to train a image classification model. We will develop a Convolutional Neural Network (CNN) for the classification. We use the MNIST dataset which is an image dataset of handwritten digits. It has has 60,000 training images and 10,000 test images, each of which are grayscale 28 x 28 sized images.

## Install MLflow and the dependencies
```
pip install mlflow[extras]
```



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
```


##  Loading &amp; Pre-processing Data

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
x_train = x_train.astype(&#39;float32&#39;)
x_test = x_test.astype(&#39;float32&#39;)
x_train /= 255
x_test /= 255

```

Also, we will one-hot-encode the labels.

```python

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

```
## Extract the input parameters
We extract the input parameters: learning rate, batch size and the number of epochs.
```python


lr = float(sys.argv[1]) if len(sys.argv) &gt; 1 else 0.001
batch_size = int(sys.argv[2]) if len(sys.argv) &gt; 1 else 32
nb_epochs = int(sys.argv[3]) if len(sys.argv) &gt; 1 else 10
```

## Set MLflow tracking server

Configure your connection to your tracking server by running mlflow.set_tracking_uri. We will use localhost. We will also set a name for the model experiment.

```python

mlflow.set_tracking_uri(uri=&#39;http://localhost:5000&#39;)

experiment_name = &quot;mlflow-cnn-experiment-0&quot;
expr_id = mlflow.set_experiment(experiment_name)
```

## Define and train a cnn model

Use with mlflow.start_run to create a new MLflow run.

```python
with mlflow.start_run(experiment_id=expr_id, run_name=&quot;running-01&quot;):
```
### Model definition and training
Inside the MLflow run, define a CNN model.
 ```python

# Define model architecture
model = Sequential()

model.add(Convolution2D(32, 3, 3, activation=&#39;relu&#39;, input_shape=input_shape))
model.add(Convolution2D(32, 3, 3, activation=&#39;relu&#39;))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation=&#39;relu&#39;))
model.add(Dropout(0.5))
model.add(Dense(10, activation=&#39;softmax&#39;))

#Define optimizer
adam = optimizers.Adam(lr=lr)
# Compile model

model.compile(loss=&#39;categorical_crossentropy&#39;,
              optimizer=adam,
              metrics=[&#39;accuracy&#39;])
 ```
Fit the model using the training dataset.
```python

model.fit(x_train, y_train,
          batch_size=batch_size, nb_epoch=nb_epochs, verbose=1)
 ```
Evaluate the model using test data
 ```python

score = model.evaluate(x_test, y_test, verbose=0)
print(&quot;CNN model&quot;)
print(&quot;Learning rate: {}, batch_size: {}, nb_epochs: {}&quot;.format(lr, batch_size, nb_epochs))
print(&#39;Test loss:&#39;, score[0])
print(&#39;Test accuracy:&#39;, score[1])
 ```
Use MLflow to record the model&#39;s parameters, metrics and model.
 ```python

mlflow.log_param(&quot;lr&quot;, lr)
mlflow.log_param(&quot;batch_size&quot;, batch_size)
mlflow.log_param(&quot;nb_epochs&quot;, nb_epochs)
mlflow.log_metric(&quot;loss&quot;, score[0])
mlflow.log_metric(&quot;accuracy&quot;, score[1])
mlflow.sklearn.log_model(model, &quot;model&quot;)
```

##  Full code
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

if __name__ == &quot;__main__&quot;:

    # Load pre-shuffled MNIST data into train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_rows, img_cols = 28, 28
    num_classes = 10


    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype(&#39;float32&#39;)
    x_test = x_test.astype(&#39;float32&#39;)
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    lr = float(sys.argv[1]) if len(sys.argv) &gt; 1 else 0.001
    batch_size = int(sys.argv[2]) if len(sys.argv) &gt; 1 else 32
    nb_epochs = int(sys.argv[3]) if len(sys.argv) &gt; 1 else 10

    mlflow.set_tracking_uri(uri=&#39;http://localhost:5000&#39;)
    experiment_name = &quot;mlflow-cnn-experiment-0&quot;
    expr_id = mlflow.set_experiment(experiment_name)

    with mlflow.start_run(experiment_id=expr_id, run_name=&quot;running-01&quot;):
        # Define model architecture
        model = Sequential()

        model.add(Convolution2D(32, 3, 3, activation=&#39;relu&#39;, input_shape=input_shape))
        model.add(Convolution2D(32, 3, 3, activation=&#39;relu&#39;))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation=&#39;relu&#39;))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation=&#39;softmax&#39;))

        #Define optimizer
        adam = optimizers.Adam(lr=lr)
        # Compile model

        model.compile(loss=&#39;categorical_crossentropy&#39;,
                      optimizer=adam,
                      metrics=[&#39;accuracy&#39;])

        model.fit(x_train, y_train,
                  batch_size=batch_size, nb_epoch=nb_epochs, verbose=1)

        # Evaluate model on test data
        score = model.evaluate(x_test, y_test, verbose=0)

        print(&quot;CNN model&quot;)
        print(&quot;Learning rate: {}, batch_size: {}, nb_epochs: {}&quot;.format(lr, batch_size, nb_epochs))
        print(&#39;Test loss:&#39;, score[0])
        print(&#39;Test accuracy:&#39;, score[1])
        mlflow.log_param(&quot;lr&quot;, lr)
        mlflow.log_param(&quot;batch_size&quot;, batch_size)
        mlflow.log_param(&quot;nb_epochs&quot;, nb_epochs)
        mlflow.log_metric(&quot;loss&quot;, score[0])
        mlflow.log_metric(&quot;accuracy&quot;, score[1])

        mlflow.sklearn.log_model(model, &quot;model&quot;)
```

You can run this example using different hyperparameters as follows:

```python
python example_name.py &lt;lr&gt; &lt;batch_size&gt; &lt;nb_epochs&gt;
```

Each time you run the example, MLflow logs information about your experiment runs in the directory mlruns.

## Comparing the models
Next, use the MLflow UI to compare the models that you have produced.  In the same current working directory as the one that contains the mlruns run
```python
mlflow ui
```
You can view it at
 http://localhost:5000