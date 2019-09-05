# Model serving


## Install Docker on you computer

Install Docker from https://docs.docker.com/install/

## Install Anaconda on your computer

Download and install Anaconda. Select the Python 3.* version):
https://www.anaconda.com/download/

When Anaconda is installed, open "Anaconda Prompt" or any other terminal where you have ```conda``` available now.

## Make an isolated Python environment
Run ```conda create --name model-serving python=3.5``` in your terminal.
Then, to activate your new environment, run ```conda activate model-serving```.


##  Install the required packages

Run ```pip install mlflow[extras]==1.2.0 "mflux-ai>=0.3.0"```  in your terminal.


## Tutorial

In this tutorial we will deploy a machine learning model as a REST API using Flask in a [Docker](https://www.docker.com/) container.

### Train and save a model

We train and save a decision tree classifier on the [Iris](https://en.wikipedia.org/wiki/Iris_flower_data_set) data set.
The model wil be stored in MFlux.ai
Create a file train_model.py and paste the following code:

```python
import mflux_ai
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

# Note: in the following line, insert the project token shown on your dashboard page.
mflux_ai.init("your_project_token_goes_here")
mlflow.sklearn.log_model(model, "model")

```

Now, run this file: ```python train_model.py```.

### Defining the REST API

Create a file app.py inside a folder named ```app``` and paste the following code:

```python
import mflux_ai
import mlflow.sklearn
import numpy as np
import pickle
from flask import Flask, request, send_from_directory
from flask import jsonify

app = Flask(__name__, static_folder="app")

# Note: in the following line, insert the project token shown on your dashboard page.
mflux_ai.init("your_project_token_goes_here")

model = mlflow.sklearn.load_model(
    "s3://mlflow/0/RUN_ID_GOES_HERE/artifacts/model"
)


@app.route("/predict/", methods=['POST'])
def predict():
    response = {"success": False}
    if request.method == "POST":
        if request.data:
            data = request.json.get('data')
            predictions = model.predict(np.asarray(data))
            response["success"] = True
            response["predictions"] = predictions.tolist()

    return jsonify(response)

```

In ```app.py```, replace ```RUN_ID_GOES_HERE``` with the actual run id that you found in the model tracking UI.

Let's go through the code. The first code snippet imports the packages and initalizes the Flask application


```python
import mflux_ai
import mlflow.sklearn
import numpy as np
import pickle
from flask import Flask, request, send_from_directory
from flask import jsonify

app = Flask(__name__, static_folder="app")
```


Next, we load our trained model from MFlux.ai.
```python
# Note: in the following line, insert the project token shown on your dashboard page.
mflux_ai.init("your_project_token_goes_here")

model = mlflow.sklearn.load_model(
    "s3://mlflow/0/RUN_ID_GOES_HERE/artifacts/model"
)
```


We define the ```predict```function which processes any requests to
the ```/predict/``` endpoint.
```python
@app.route("/predict/", methods=['POST'])
def predict():
    response = {"success": False}
    if request.method == "POST":
        if request.data:
            data = request.json.get('data')
            predictions = model.predict(np.asarray(data))
            response["success"] = True
            response["predictions"] = predictions.tolist()

    return jsonify(response)
```
The function takes the incoming data and feeds its into the model. It
then returns the predictions to the client in JSON format.


### Dockerfile

Make a ```Dockerfile``` and paste the following code:
```Dockerfile
FROM continuumio/miniconda3:4.6.14

RUN pip install pyuwsgi==2.0.18

RUN mkdir /app/
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app

ENTRYPOINT ["pyuwsgi", "--http", ":5000", "--wsgi-file", "app/app.py", "--callable", "app", "--enable-threads"]
```

### Requirements

Make a  ```requirements.txt``` file to install the required packages. Paste the following code in the file:

```
mlflow==1.2.0
mflux-ai>=0.4.0
boto3==1.9.215
minio==4.0.20
scikit-learn==0.21.3
```

### Directory structure

The directory structure should be like this:

```
-/model-serving/
      -Dockerfile
      -requirements.txt
      -/app/
          -app.py
```

### Build the docker image

Run ```docker build -t model-serving .```

### Launch a docker container

Run ```docker run -p 5000:5000 model-server```


### Make requests to the API
You can test the API by using cURL.

Run
```curl http://0.0.0.0:5000/predict/ -d '{"data": [[5.1, 3.5, 1.4, 0.2], [3.1,  3.5, 1.4, 0.2]]}' -H 'Content-Type: application/json'```
in the terminal.
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

You can also make requests using python:
```python
import requests
import numpy as np
url= 'http://0.0.0.0:5000/predict/'
data ={'data': [[5.1, 3.5, 1.4, 0.2], [3.1, 3.5, 1.4, 0.2]]}
response = requests.post(url, json = data)
response.json()
```


## Serve model using MFlux.ai


