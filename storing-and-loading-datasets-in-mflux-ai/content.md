By storing files/datasets/objects in a central location, it becomes easier for your data science
team to collaborate. MFlux.ai makes it easy to store files/datasets/objects and make them
available to your data science team and servers (e.g. for model training and serving).
MFlux.ai's object storage system is powered by [MinIO](https://min.io/), an open source,
enterprise-grade Amazon S3-compatible Object Storage system.

# Install requirements

`pip install mflux_ai numpy`

# Store a dataset

The following code snippet shows how to store a serializable (picklable) python object, in this
case a numpy array.

```python
import mflux_ai
import numpy as np

# Note: in the following line, insert the project token shown on your dashboard page.
mflux_ai.init("your_project_token_goes_here")

# Generate some dummy data (a numpy array)
my_dataset = np.zeros(shape=(10000, 100), dtype=np.float32)
object_name = "my-dataset.pkl"

mflux_ai.put_dataset(my_dataset, object_name)

print("Dataset uploaded")
```

When you run the script above successfully, it should output `Dataset uploaded!`

# Loading a dataset

Assuming you want to load the numpy array that you uploaded in the previous script, here's code
for doing that:

```python
import mflux_ai

# Note: in the following line, insert the project token shown on your dashboard page.
mflux_ai.init("your_project_token_goes_here")

object_name = "my-dataset.pkl"
my_loaded_dataset = mflux_ai.get_dataset(object_name)
```

# Listing stored objects

You can use the MinIO python client API to list stored objects. Here's how:

```python
import mflux_ai

mflux_ai.init("your_project_token_goes_here")
minio_client = mflux_ai.get_minio_client()

my_objects = minio_client.list_objects_v2("datasets")
for my_object in my_objects:
    pass  # do something with my_object
```

## MinIO API reference

To see what else you can do with the MinIO python client, check out the official guides and API references:

* [https://docs.min.io/docs/python-client-quickstart-guide.html](https://docs.min.io/docs/python-client-quickstart-guide.html)
* [https://docs.min.io/docs/python-client-api-reference.html](https://docs.min.io/docs/python-client-api-reference.html)

# Object storage web UI

An object storage web UI will be an integrated part of the MFlux.ai dashboard. Until then,
here is a workaround that lets you see the MinIO web UI. Run the following code snippet to
reveal the instructions:

```python
import os

import mflux_ai

mflux_ai.init("your_project_token_goes_here")

print("Visit the following URL in your browser:", os.environ["MLFLOW_S3_ENDPOINT_URL"])
print("Use this access key:", os.environ["AWS_ACCESS_KEY_ID"])
print("Use this secret key:", os.environ["AWS_SECRET_ACCESS_KEY"])
```
