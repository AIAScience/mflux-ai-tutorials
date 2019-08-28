# Model serving


## Install Anaconda on your computer

Download and install Anaconda. Select the Python 3.* version):
https://www.anaconda.com/download/

When Anaconda is installed, open "Anaconda Prompt" or any other terminal where you have ```conda``` available now.

## Make an isolated Python environment
Run ```conda create --name ml-pipeline python=3.5``` in your terminal.
Then, to activate your new environment, run ```conda activate ml-pipeline```.


##  Install the required packages

Run ```pip install mlflow[extras]==1.2.0 "mflux-ai>=0.3.0" kedro==0.15.0 keras==2.2.4 tensorflow==1.14```  in your terminal.

## Tutorial

In this tutorial we will deploy a machine learning model as a REST API using Flask RESTful

### Train and save a model

We train and save a decision tree classifier on the [Iris](https://en.wikipedia.org/wiki/Iris_flower_data_set) data set.
Create a file build_model.py and paste the following code:

```python
import mlflow.sklearn
import os
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()

input_data = iris["data"]
target_data = iris["target"]

model = DecisionTreeClassifier()
model.fit(input_data, target_data)
with open('model.pkl', 'wb') as pickle_file:
    pickle.dump(model, pickle_file)
```

Now, run this file to train and save the model: ```python build_model.py```.

## Defining the REST API

Create a file app.py and paste the following code:

```python
import numpy as np
import pickle
from flask import Flask, request, send_from_directory

from flask import jsonify

app = Flask(__name__)


def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


@app.route("/predict/", methods=['Get', 'POST'])
def predict():
    response = {"success": False}
    if request.method == "POST":
        if request.data:
            data = request.get_json()['data']
            predictions = model.predict(np.asarray(data))
            response["success"] = True
            response["predictions"] = predictions.tolist()

    return jsonify(response)


if __name__ == "__main__":
    model = load_model()
    app.run()
```

Let's go through the code. The first code snippet imports the packages and initalizes the Flask application


```python
import numpy as np
import pickle
from flask import Flask, request, send_from_directory

from flask import jsonify

app = Flask(__name__)
```


Next, we have a method for loading our trained model.
```python
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model
```


We define the ```predict```function which processes any requests to
the ```/predict/``` endpoint.
```python
@app.route("/predict/", methods=['POST'])
def predict():
    response = {"success": False}
    if request.method == "POST":
        if request.data:
            data = request.get_json()['data']
            predictions = model.predict(np.asarray(data))
            response["success"] = True
            response["predictions"] = predictions.tolist()

    return jsonify(response)

```
The function takes the incoming data and feeds its into the model. It
then returns the predictions to the client in JSON format.


```python
if __name__ == "__main__":
    model = load_model()
    app.run()
```

The main method loads the model and launches the app.

## Start the API

Run ```python app.py```






### Using cURL to test the REST API
You can test the API by using cURL.

Run ```curl localhost:5000/predict/ -d '{"data": [[5.1, 3.5, 1.4, 0.2], [3.1 3.5, 1.4, 0.2]]}' -H 'Content-Type: application/json' ``` in the terminal.
You will then receive a json with the predictions:
```
{
  "predictions": [
    0,
    0
  ],
  "success": true
}
```
