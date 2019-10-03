# MLflow API-usage Example
This is the MFlux/MLflow cheat sheet in example form. It is an example of using MFlux and MLflow to share data and 
machine learning models with a team.

This example handles a very simple machine learning task. The focus is on using MFlux and MLflow to log and share models 
and datasets.

## Dependencies
You will need MFlux, MLflow, keras, scikit-learn, python, numpy and matplotlib. You may want to create a separate conda 
environment, and run:

`conda install "python=3.6.6" "numpy=1.16" "Keras=2.3" "scikit-learn=0.21" "matplotlib=3.1"`

`pip install mlflow[extra]==1.2.0 "mflux-ai>=0.5.1"`

## Creating a shared project
The MFlux dashboard lets us easily create a new shared project. First, create the team for the project, by going to the 
teams tab on the dashboard, pressing "+ New Team" and specifying team name and members. You can add and remove members at 
any time. 

When you go to the projects tab and press "+ New Project" you can choose the team you have created for it. Now click this 
project to see the tracking UI for it. There is nothing there yet, but click the settings tab. There you will find the 
unique token for this project.

## Uploads
first we will make up some data, run some algorithms on it and upload everything to MFlux.

### Imports and Data
Make a new file called `uploads.py`. These are the necessary imports:
```python
import mflux_ai
import mlflow.keras
import mlflow.sklearn
import numpy as np
import sklearn.linear_model as lm
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot
```

And let's initialize the project with the token found on the project's settings page
```python
# initialize MFlux project
mflux_ai.init("your_project_token")
```

We will need some toy data to build the example around.
```python
# make up toy data
x = np.array([0, 1, 2, 3, 4])
x = x.reshape(-1, 1)
y1 = np.array([0, 1, 4, 9, 16])
y1 = y1.reshape(-1, 1)
reg1 = np.concatenate((x, y1), axis=1)
x_pred = np.array([0, 1, 2, 3, 4, 5, 6])
x_pred = x_pred.reshape(-1, 1)
y1_target = np.array([0, 1, 4, 9, 16, 25, 36])
y1_target = y1_target.reshape(-1, 1)
target1 = np.concatenate((x_pred, y1_target), axis=1)
```

Now we upload this data to MFlux, so anyone working on the project can get it preprocessed from there:
```python
# upload toy data to MFlux
mflux_ai.put_dataset(reg1, "reg1.pkl")
mflux_ai.put_dataset(target1, "target1.pkl")
```

### Linear Regression
The following function does linear regression on the toy data. It takes as input the training set (x and y), the x-values
we want to predict the y-values of, and the correct y-values for these for comparison. It returns the predicted y-values, 
the model (in this case the linear formula) and the score of the prediction.
```python
def lin_reg(x, y, x_pred, target):
    lr_model = lm.LinearRegression().fit(x, y)
    y_pred = lr_model.predict(x_pred)
    score = lr_model.score(x_pred, target)
    pyplot.plot(target)
    pyplot.plot(y_pred, color='red')
    pyplot.show()
    return y_pred, lr_model, score
```

Now we run this function:
```python
y_pred, lr_model, score = lin_reg(x, y1, x_pred, y1_target)
```

We upload the result to MFlux. Notice that the model is logged with `log_model()` and the score with `log_metric`
```python
# upload linear regression model to MFlux
mlflow.set_experiment("Linear Regression")
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(lr_model, "model")
    mlflow.log_param("copy_X", lr_model.get_params()["copy_X"])
    mlflow.log_metric("score", score, step=None)
```

### Neural Net
Now we train a tiny neural net on the data. First, let's create variables controlling the amount of training epochs and
how many times we will log the models score progress.
```python
epochs = 10000
log_steps = 25
```

The following function creates and trains the network, and plots the result. The variable `history` contains data about the
training process.
```python
def net(x, y, x_pred, target, epochs):
    nn_model = Sequential()
    nn_model.add(Dense(1, input_dim=1, activation='exponential'))
    nn_model.compile(optimizer='adam', loss='mse')
    nn_model.summary()
    history = nn_model.fit(x, y, epochs=epochs)
    y_pred = nn_model.predict(x_pred)
    pyplot.plot(target)
    pyplot.plot(y_pred, color='red')
    pyplot.show()
    return y_pred, nn_model, history
```

We run this function
```python
y_pred, nn_model, history = net(x, y1, x_pred, y1_target, epochs)
```

And upload the result to MFlux
```python
y_pred, nn_model, history = net(x, y1, x_pred, y1_target, epochs)
# upload neural net model to MFlux
mlflow.set_experiment("Neural Net")
with mlflow.start_run() as run:
    mlflow.keras.log_model(nn_model, "model")
    w = nn_model.get_weights()[0][0][0]
    b = nn_model.get_weights()[1][0]
    tags = {"w": w, "b": b}
    mlflow.set_tags(tags)
    for i in range(0, epochs, epochs//log_steps):
        mlflow.log_metric(key="mse", value=history.history["loss"][i], step=i)
```

## Downloads
Now we want to fetch the data and models on MFlux and run them on our computer. 
You can call this file `downloads.py`. The imports are
```python
import mflux_ai
import mlflow.keras
import mlflow.sklearn
from matplotlib import pyplot
```

And we want to initiate MFlux
```python
mflux_ai.init("your_project_token")
```

We get the data, and reshape it (we could have put it on MFlux in this shape, but chose to only upload two files instead 
of four):
```python
# get data from MFlux
train = mflux_ai.get_dataset("reg1.pkl")
extended = mflux_ai.get_dataset("target1.pkl")
x = train[:, 0].reshape(-1, 1)
y = train[:, 1].reshape(-1, 1)
x_pred = extended[:, 0].reshape(-1, 1)
target = extended[:, 1].reshape(-1, 1)
```

We will use two functions to run the linear and neural models. These are just the prediction and plot part of the functions
that trained these models. Notice that we have to pass the actual model as input this time.
```python
def lin_reg(x_pred, target, lr_model):
    y_pred = lr_model.predict(x_pred)
    pyplot.plot(target)
    pyplot.plot(y_pred, color='red')
    pyplot.show()
    return y_pred


def net(x_pred, target, nn_model):
    y_pred = nn_model.predict(x_pred)
    pyplot.plot(target)
    pyplot.plot(y_pred, color='red')
    pyplot.show()
    return y_pred
```

We want to run a model from the "Linear Regression" experiment. We need to set the experiment we are searching through.
Then we get the id of the first run/model, load this model and run it through the linear regression function.
```python
mlflow.set_experiment("Linear Regression")
runs = mlflow.search_runs()
run_id = runs.iloc[0, :]["run_id"]
model_uri = "runs:/" + run_id + "/model"
model = mlflow.sklearn.load_model(model_uri=model_uri)
lin_reg(x_pred, target, model)
```

Now we want to examine a model from the "Neural Net" experiment. This time it is worthwhile to sort the runs by score, so
we can examine the best model.
```python
mlflow.set_experiment("Neural Net")
runs = mlflow.search_runs()
runs = runs.sort_values(by=['metrics.mse'])
run_id = runs.iloc[0, :]["run_id"]
model_uri = "runs:/" + run_id + "/model"
model = mlflow.keras.load_model(model_uri=model_uri)
lin_reg(x_pred, target, model)
```

## A final puzzle
The neural net in this example consists of one input and one output node, with no hidden layers. This means that the entire
manipulation done by the network can be written as:

`output = f(w * input + b)`

where `w` is the single weight parameter, `b` is the single bias parameter and `f` is the activation function. By tweaking
these it is possible to get a perfect prediction of the test data. Can you do that?

Hint: Pay special attention to the activation function, it is possible to create your own..


