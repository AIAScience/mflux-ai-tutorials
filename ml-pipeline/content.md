# ML Pipeline


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

In this tutorial we will create a complete machine learning pipeline using [Kedro](https://github.com/quantumblacklabs/kedro).
We will create a pipline for the video classification problem.

## Creating the tutorial project

Run ```kedro new``` to create a new empty template project. Call the project ml-pipeline. Choose ```n``` to
create a project template without a dummy dataset example. Within your project's root directory, install project
dependencies by running ```kedro install```.

## Add datasets

[Download](https://github.com/AIAScience/deep-learning-intro/tree/master/data) and save the two json files to the ```data/01_raw/```
folder.

## Reference all datasets

Register the datasets in the ```conf/base/catalog.yml``` file by adding
this to the file:

```yaml
categories:
  type: JSONLocalDataSet
  filepath: data/01_raw/categories.json

videos:
  type: JSONLocalDataSet
  filepath: data/01_raw/videos.json
```


## Creating the data pre-processing pipeline

We will now create a pipeline from a set of nodes, which are Python functions. This
pipeline will preprocess the data by extracting feature vectors and target vectors.
We create a file for processing the data called ```data_engineering.py```
inside the ```nodes``` folder. Add the following code in the file:

```python
import json

import numpy as np


def vectorize_video_input(video: dict, num_tags: int, tag_to_index: dict) -> np.array:
    """
    Vectorize the video.
    :param video: video data.
    :param num_tags: number of tags.
    :param tag_to_index: dict which maps a tag to an index.
    :return: feature vector.
    """
    input_vector = [0] * num_tags
    for tag in video["tags"]:
        tag_index = tag_to_index.get(tag, None)
        if tag_index is not None:
            input_vector[tag_index] = 1
    return input_vector


def vectorize_video_target(video: dict, num_categories: int, category_id_to_index: dict) -> np.array:
    """
    Vectorize the video target.
    :param video: video data.
    :param num_categories: number of categories.
    :param category_id_to_index: dict which maps a category to an index.
    :return: target vector.
    """
    target_vector = [0] * num_categories
    category_index = category_id_to_index.get(video["target_category_id"], None)
    if category_index is not None:
        target_vector[category_index] = 1
    return target_vector


def create_video_features(videos: json) -> np.array:
    """
    Create feature vectors for the videos.
    :param videos: video data.
    :return: feature vectors.
    """
    tags = set()
    for video in videos:
        for tag in video["tags"]:
            tags.add(tag)
    num_tags = len(tags)
    tag_to_index = {tag: index for index, tag in enumerate(tags)}
    input_vectors = [vectorize_video_input(video, num_tags, tag_to_index) for video in videos]
    input_vectors = np.array(input_vectors)

    return input_vectors


def create_video_targets(videos: json, categories: json) -> np.array:
    """
    Create target vectors for the videos.
    :param videos: video data.
    :param categories:  video categories.
    :return: target vectors.
    """
    num_categories = len(categories)
    category_id_to_index = {
        category["id"]: index for index, category in enumerate(categories)
    }
    target_vectors = [vectorize_video_target(video, num_categories, category_id_to_index) for video in videos]
    target_vectors = np.array(target_vectors)

    return target_vectors


def extract_num_categories(categories: json) -> int:
    """
    Extract how many categories the data set contains.
    :param categories: categories data.
    :return: number of categories.
    """
    return len(categories)
```

We can use these functions as nodes into the pipeline in ```pipeline.py```. The
```create_pipeline``` will look like this:

```python
from kedro.pipeline import node, Pipeline
from ml_pipeline.nodes.data_engineering import (
    create_video_features,
    create_video_targets,
extract_num_categories,
)

from ml_pipeline.nodes.video_classification import split_data, train_model, evaluate_model


def create_pipeline(**kwargs):
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        Pipeline: The resulting pipeline.

    """
    de_pipeline = Pipeline(
        [
            node(create_video_features, "videos", "features", name="preprocess1"),
            node(create_video_targets, ["videos","categories"], "targets", name="preprocess2"),
            node(extract_num_categories,"categories", "num_categories")
        ]
    )

    return de_pipeline


```

This pipeline will extract feature and target vectors from the data and also extract the number of categories.
If you want any of this data to persist after the pipeline is finished running you can add them in the
```conf/base/catalog.yml```.

### Creating the data science pipeline

Next, we will make a pipeline for training and validation a video classification model. Create a file
```src/ml_pipeline/nodes/video_classification.py``` and add the following code to it:


```python
import logging
from typing import Dict, List

import keras
import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split


def split_data(x: np.array, y: np.array, parameters: Dict) -> List:
    """
    Splits data into training and test sets.
    :param x: feature vector.
    :param y: target vector.
    :param parameters: Parameters defined in parameters.yml.
    :return: A list containing split data.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return [x_train, x_test, y_train, y_test]


def train_model(x_train: np.ndarray, y_train: np.ndarray, num_categories: int) -> keras.models.Model:
    """
    Train the model.
    :param x_train: training data of features.
    :param y_train: training data for labels.
    :param num_categories: number of different categories to predict.
    :return: Trained model.
    """
    num_hidden_nodes = 10
    model = Sequential()
    model.add(Dense(num_hidden_nodes, input_dim=x_train.shape[1], activation="relu"))
    model.add(Dense(num_categories, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer=SGD(momentum=0.0), metrics=["accuracy"]
    )
    model.fit(x_train, y_train, epochs=50)

    return model


def evaluate_model(model: keras.models.Model, x_test: np.ndarray, y_test: np.ndarray):
    """
    Calculate the validation accuracy and the validation loss.
    :param model: Trained model.
    :param x_test: Testing data of features.
    :param y_test: Testing data for target.
    """
    evaluation_scores = model.evaluate(x_test, y_test)
    logger = logging.getLogger(__name__)
    for i, metric_name in enumerate(model.metrics_names):
        logger.info("Validation {}: {:.3f}".format(metric_name, evaluation_scores[i]))
```

Add the following to ```conf/base/parameters.yml```:

```python
test_size: 0.2
random_state: 3
```

We will also save the trained model by adding the following to ```conf/base/catalog.yml```:

```python
model:
  type: PickleLocalDataSet
  filepath: data/06_models/regressor.pickle
  versioned: true
```


We can now create a pipeline for training and validating a classification model by updating the ```create_pipeline()```:

```python
def create_pipeline(**kwargs):
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        Pipeline: The resulting pipeline.

    """
    de_pipeline = Pipeline(
        [
            node(create_video_features, "videos", "features", name="preprocess1"),
            node(create_video_targets, ["videos","categories"], "targets", name="preprocess2"),
            node(extract_num_categories,"categories", "num_categories")
        ]
    )

    ds_pipeline = Pipeline(
        [
            node(
                split_data,
                ["features", "targets", "parameters"],
                ["X_train", "X_test", "y_train", "y_test"],
            ),
            node(train_model, ["X_train", "y_train","num_categories" ], "model"),
            node(evaluate_model, ["model", "X_test", "y_test"], None),
        ]
    )

    return de_pipeline + ds_pipeline
```

The two pipelines are merged together in ```de_pipeline + ds_pipeline```. Both pipelines will be executed when you invoke the following:

```kedro run```

The de_pipeline will preprocess the data, and ds_pipeline will then create features, train and evaluate the model.



## Log metrics and store machine learning model in MFlux.ai
Let's log the metrics and store the model in MFlux.ai.

Remove the model from the data set definition in ```conf/base/catalog.yml```.
MFlux.ai will instead take care of the model storing and versioning.

In the file ```src/ml_pipeline/nodes/video_classification.py``` add the following imports

```python
import mlflow
import mlflow.keras
import mflux_ai
```

In the same file replace the method ```evaluate_model()``` with this new definition:
```python
def evaluate_model(model: keras.models.Model, x_test: np.ndarray, y_test: np.ndarray):
    """
    Calculate the validation accuracy and the validation loss. Log it to MFlux.ai. Store
    the model in MFlux.ai
    :param model: Trained model.
    :param x_test: Testing data of features.
    :param y_test: Testing data for target.
    """
    mflux_ai.init("Your Key")
    evaluation_scores = model.evaluate(x_test, y_test)
    for i, metric_name in enumerate(model.metrics_names):
        mlflow.log_metric("validation_"+ str(metric_name), evaluation_scores[i])
    mlflow.log_param("model_type", model.__class__.__name__)
    mlflow.keras.log_model(model, "model")
```

Remember to replace the "Your key" field with your MFlux.ai key.

## Check your tracking UI
You should now be able to see the metric and model that you logged in your MLflow tracking UI.
