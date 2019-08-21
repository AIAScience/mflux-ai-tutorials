# ML Pipeline


## Install Anaconda on your computer

Download and install Anaconda. Select the Python 3.* version):
https://www.anaconda.com/download/

When Anaconda is installed, open "Anaconda Prompt" or any other terminal where you have ```conda``` available now.

## Make an isolated Python environment
Run ```conda create --name ml-pipeline python=3.5``` in your terminal.
Then, to activate your new environment, run ```conda activate ml-pipeline```.


##  Install the required packages

Run ```pip install mlflow[extras]==1.1.0 mflux-ai kedro keras tensorflow```  in your terminal.

## Tutorial

In this tutorial we will create a complete machine learning pipeline using [Kedro](https://github.com/quantumblacklabs/kedro).
We will use create a pipline for the video classification problem.

## Creating the tutorial project

Run ```kedro new``` to create a new empty template project. Call the project ml-pipeline. Choose ```n``` to
create a project template without a dummy dataset example. Within your project's root directory, install project
dependencies by running ```kedro install```.

## Add datasets

[Download](https://github.com/AIAScience/deep-learning-intro/tree/master/data) and and save the two json files to the ```data/01_raw/```
folder.

## Reference all datasets

Register the datasets in the ```conf/base/catalog.yml``` file by adding
this to the file:

```yaml
companies:
  type: JSONLocalDataSet
  filepath: data/01_raw/companies.csv

reviews:
  type: JSONLocalDataSet
  filepath: data/01_raw/reviews.csv
```


## Creating the data pre-processing pipeline

We will now create a pipeline from a set of nodes, which are Python functions. This
pipeline will preprocess the data by extracting feature vectors and target vectors.
We create a file for processing the data called ```data_engineering.py```
inside the ```nodes``` folder. Add the following code in the file:

```python
import json

import numpy as np


def vectorize_video_input(video, num_tags, tag_to_index):
    input_vector = [0] * num_tags
    for tag in video["tags"]:
        tag_index = tag_to_index.get(tag, None)
        if tag_index is not None:
            input_vector[tag_index] = 1
    return input_vector


def vectorize_video_target(video, num_categories, category_id_to_index):
    target_vector = [0] * num_categories
    category_index = category_id_to_index.get(video["target_category_id"], None)
    if category_index is not None:
        target_vector[category_index] = 1
    return target_vector


def create_video_features(videos: json) -> np.array:
    tags = set()
    for video in videos:
        for tag in video["tags"]:
            tags.add(tag)
    num_tags = len(tags)
    tag_to_index = {tag: index for index, tag in enumerate(tags)}
    input_vectors = [vectorize_video_input(video, num_tags, tag_to_index) for video in videos]
    input_vectors = np.array(input_vectors)

    return input_vectors


def create_video_targets(videos:json, categories: json) -> np.array:
    num_categories = len(categories)
    category_id_to_index = {
        category["id"]: index for index, category in enumerate(categories)
    }
    target_vectors = [vectorize_video_target(video,  num_categories, category_id_to_index) for video in videos]
    target_vectors = np.array(target_vectors)

    return target_vectors


def extract_num_categories(categories: json) -> int:
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

Next, we will make a pipeline for a video classification model. Create a file
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


def split_data(X: np.array, y:np.array, parameters: Dict) -> List:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return [X_train, X_test, y_train, y_test]


def train_model(X_train: np.ndarray, y_train: np.ndarray, num_categories: int) -> keras.models.Model:
    num_hidden_nodes = 10
    model = Sequential()
    model.add(Dense(num_hidden_nodes, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(num_categories, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer=SGD(momentum=0.0), metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=50)

    return model


def evaluate_model(model: keras.models.Model, X_test: np.ndarray, y_test: np.ndarray):
    evaluation_scores = model.evaluate(X_test, y_test)
    logger = logging.getLogger(__name__)
    for i, metric_name in enumerate(model.metrics_names):

```

Add the following to ```conf/base/parameters.yml```:

```python
test_size: 0.2
random_state: 3
```

We will also save the trained model by adding the following to ```conf/base/catalog.yml```

```python
model:
  type: PickleLocalDataSet
  filepath: data/06_models/regressor.pickle
  versioned: true
```


We can now create a pipeline for a video classification model by updating the ```create_pipeline()```

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