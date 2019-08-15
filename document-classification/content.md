# Set up Python with machine learning libraries

Downloading and installing data science-related python packages manually can be hard and tedious. Anaconda to the rescue! It's basically a bundle that includes python and a number of common python dependencies. Download and install it if you don't have it (you should get the Python 3.* version):
https://www.anaconda.com/download/

When Anaconda is installed, open "Anaconda Prompt" or any other terminal where you have `conda` available now.

Run the following command to install Scikit-Learn, TensorFlow and Keras (common machine learning libraries):

* `conda install "python=3.6.6" "scipy=1.1" "h5py<3" "tensorflow=1.11" "Keras=2.2" "scikit-learn=0.20"`

This may take a couple of minutes.

Check your installation by running `python -c "import keras"`. If no error appears, the installation should be fine. If it outputs "Using TensorFlow backend.", that's okay.

When you're ready, move on to the tutorial:

# Install MLflow and M

Run the following command in the terminal to install MLflow and MFlux.ai.

```pip install mlflow[extra] mflux-ai```



# Tutorial

In this tutorial, we will create a simple video classifier model. It will input video metadata and output a category prediction. Here is some info on the dataset:

https://github.com/AIAScience/deep-learning-intro/tree/master/data

Ready to start coding? Great! Clone (or download) the [deep-learning-intro](https://github.com/AIAScience/deep-learning-intro) repository so you have a local copy. Inside the `deep-learning-intro` folder, create a file called `tutorial.py` and paste in the following code for importing the dataset:

```python
import json
import os
import pprint

with open(os.path.join('data', 'videos.json')) as json_file:
    videos = json.load(json_file)

print('We have {} videos'.format(len(videos)))
print("Data for the first video looks like this:")
pprint.pprint(videos[0])
```

Now try to run the program (`python tutorial.py`). If you have everything set up correctly, it should print

```
We have 279 videos
{'relevant_topics': [],
 'tags': ['Funny',
          'smosh',
          'video',
          'Massively Multiplayer Online Role-playing Game (Video Game Genre)',
          'World Of Warcraft (Video Game)',
          'WoW',
          'elf',
          'warlords of draenor',
          'smosh games',
          'RPG',
          'honest trailer',
          'game trailer',
          'Video Game (Industry)',
          'Video Game Culture',
          'Role-playing Game (Game Genre)',
          'anthony',
          'fun',
          'ian',
          'spoof',
          'honest game trailers',
          'orc',
          'comedy',
          'smoshgames',
          'video games',
          'honest',
          'Warcraft (Fictional Universe)',
          'games',
          'MMORPG',
          'parody'],
 'target_category_id': 3,
 'title': 'WORLD OF WARCRAFT (Honest Game Trailers)',
 'topics': []}
```

As you can see, each video has a number of `tags`. For simplicity, let's use `tags` as our only input to the machine learning model we are going to make. First, we need to calculate the set of unique tags we are dealing with. Add the following snippet of python code to your tutorial.py:

```
tags = set()

for video in videos:
    for tag in video["tags"]:
        tags.add(tag)

num_tags = len(tags)

print("We have {} unique tags".format(num_tags))
```

This should print something like

```
We have 3184 unique tags
```

Since machine learning models work on vectors/matrices/tensors (collections of numbers), we need to make a vector representation of each video. We can do that with a binary vector. The size of the vector will be equal to the number of unique tags. Each position in the vector will refer to a unique tag, and the number at that position (either 0 or 1) represents the presence of that tag in the video metadata.

```
tag_to_index = {tag: index for index, tag in enumerate(tags)}


def vectorize_video_input(video):
    input_vector = [0] * num_tags
    for tag in video["tags"]:
        tag_index = tag_to_index.get(tag, None)
        if tag_index is not None:
            input_vector[tag_index] = 1
    return input_vector


print('The first video input in vector form looks like this:')
print(vectorize_video_input(videos[0]))
```

This will print something like this:

```
The first video in vector form looks like this:
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, ... <snipped for brevity> , 0, 0, 0, 0, 0, 0, 0]
```

We also need to vectorize the target data. In other words, the ground truth category must be vectorized. We can do that with a one-hot-vector. A one-hot-vector is a list of numbers: One of the numbers is `1` (corresponds to the ground truth category), while the rest are `0`. Code:

```
with open(os.path.join("data", "categories.json")) as json_file:
    categories = json.load(json_file)

num_categories = len(categories)

print("We have {} categories:".format(num_categories))
pprint.pprint(categories)

category_id_to_index = {
    category["id"]: index for index, category in enumerate(categories)
}
print('Category id to index in target vector:')
print(category_id_to_index)


def vectorize_video_target(video):
    target_vector = [0] * num_categories
    category_index = category_id_to_index.get(video["target_category_id"], None)
    if category_index is not None:
        target_vector[category_index] = 1
    return target_vector


print('The first video target category in one-hot-vector form looks like this:')
print(vectorize_video_target(videos[0]))
```

This will print something like this:

```
We have 3 categories:
[{'id': 1, 'name': 'Cats'},
 {'id': 2, 'name': 'Magic'},
 {'id': 3, 'name': 'Funny Gaming Clips'}]
Category id to index in target vector:
{1: 0, 2: 1, 3: 2}
The first video target category in vector form looks like this:
[0, 0, 1]
```

Next, let's vectorize all videos and put the vectors in numpy arrays. Numpy arrays are typically what ML models in Python expect.

```
input_vectors = [vectorize_video_input(video) for video in videos]
target_vectors = [vectorize_video_target(video) for video in videos]

import numpy as np

input_vectors = np.array(input_vectors)
target_vectors = np.array(target_vectors)
```

Now, if you want to inspect the numpy arrays, you can do that:
```
print("Input vectors:")
print(input_vectors.dtype)
print(input_vectors.shape)
print(input_vectors)

print("Target vectors:")
print(target_vectors.dtype)
print(target_vectors.shape)
print(target_vectors)
```

Next, split up the dataset into two groups: Training data and validation data. You might want to read up on cross-validation here:

[https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Holdout_method](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Holdout_method)

```
training_fraction = 0.8  # use e.g. 80 % of data for training, 20 % for validation
split_index = int(len(input_vectors) * training_fraction)
training_input_vectors = input_vectors[0:split_index]
training_target_vectors = target_vectors[0:split_index]
validation_input_vectors = input_vectors[split_index:]
validation_target_vectors = target_vectors[split_index:]
```

Set up your neural network (machine learning model):

```
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD

num_hidden_nodes = 10
model = Sequential()
model.add(Dense(num_hidden_nodes, input_dim=num_tags, activation="relu"))
model.add(Dense(num_categories, activation="softmax"))

model.compile(
    loss="categorical_crossentropy", optimizer=SGD(momentum=0.0), metrics=["accuracy"]
)
```

Train the model:
```
model.fit(training_input_vectors, training_target_vectors, epochs=50)
```

Evaluate the accuracy of your model on the validation set:

```
evaluation_scores = model.evaluate(validation_input_vectors, validation_target_vectors)

for i, metric_name in enumerate(model.metrics_names):
    print("Validation {}: {:.3f}".format(metric_name, evaluation_scores[i]))
```

It'll print something like this (the numbers may vary from run to run):

```
Validation loss: 0.754
Validation acc: 0.625
```

A perfect validation loss would be 0, and a perfect validation accuracy would be 1 (or 100 %).

Was the validation accuracy bad? Try to tweak the following parameters and see how much you can improve the validation accuracy (you should be able to exceed 90 % accuracy):

* `momentum` (keep it somewhere between 0 and 0.99)
* `epochs` (keep it somewhere between 20 and 500)
* `num_hidden_nodes` (keep it somewhere between 3 and 20)

Here's an example of how we can get a prediction from the model, and compare it with the ground truth:

```
print('The last video (in the validation set):')
pprint.pprint(videos[-1])

# model.predict expects a list of examples (a 2D numpy array)
# We will put only one example in our list of examples
output_vectors = model.predict(np.array([input_vectors[-1]]))
output_vector = output_vectors[0]

print('Output vector: {}'.format(str(output_vector)))
print('Target vector: {}'.format(str(target_vectors[-1])))
```

It may print something like this:

```
The last video (in the validation set):
{'relevant_topics': ['/m/0bzvm2'],
 'tags': [],
 'target_category_id': 3,
 'title': "PTL Presents: Dyrus' Dome",
 'topics': ['/m/0x1_g_7']}
Output vector: [0.4169381  0.20512761 0.3779343 ]
Target vector: [0 0 1]
```

If the model was wrong, why do you think it failed? What can you do to make your model more robust in this case? Discuss with other people.

# Log metrics and store machine learning model in MFLux.ai

Let's log the validation loss and accuracy metric and store the model in MFlux.ai

```python
import mlflow
import mlflow.sklearn
import mflux_ai

mflux_ai.set_env_vars("Insert your key here")

for i, metric_name in enumerate(model.metrics_names):
    mlflow.log_metric("validation_"+ metric_name, evaluation_scores[i])
mlflow.sklearn.log_model(model, "model")
```

# Check your tracking UI

You should now be able to see the metric and model that you logged in your MLflow tracking UI.



# Things to do after you finish the tutorial

* Tweak the parameters of the neural network model. For example, change the number of hidden nodes.
* Try to add [dropout](https://keras.io/layers/core/#dropout) to the input layer, and tweak the `rate` parameter
* Try to disregard all tags that occur only once in the dataset. This will result in a smaller input vector.
* Play around with simple neural network architectures here: https://playground.tensorflow.org/
* Include information from `topics` and `relevant_topics` in the input vector. You can use the same technique as you used for tags.
* Include information from `title` in the input vector. Here is an example of how you can deal with this:
    * Split the title into words
    * Remove punctuation, whitespace and perhaps numbers
    * Perhaps remove words that appear only once in the dataset?
    * Apply TF-IDF (look this up on the internet)
