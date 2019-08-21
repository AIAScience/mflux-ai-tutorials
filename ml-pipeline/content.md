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

Run ```kedro new``` to create a new empty template project. Call the project ml-pipeline. Choose ```n`` to
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


##Creating a pipeline

We will now create a pipeline from a set of nodes, which are Python functions.
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
```create_pipeline`` will look like this:

```python
fro kedro.pipeline import node, Pipeline
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

This pipeline will extract features and targets from the data and also extract the number of categories.
If you want  any of this data to persist after the pipeline is finished running you can add them in the
```conf/base/catalog.yml``.

### Creating nodes

