It's possible to store datasets in MFlux.ai and make them available to your data science team.
MFlux.ai's object storage system is powered by [MinIO](https://min.io/), an open source,
enterprise-grade Amazon S3-compatible Object Storage system.

# Install requirements

`pip install joblib==0.13.2 minio==4.0.19 numpy`

# Store a dataset

```
import os

import joblib
import numpy as np
from minio import Minio
from minio.error import ResponseError, BucketAlreadyOwnedByYou, BucketAlreadyExists

# Insert the correct MFlux.ai environment variables here, as shown on the dashboard page
os.environ["MLFLOW_TRACKING_URI"] = "http://159.69.151.173:5000/"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://159.69.151.173:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "updsxtjymzwe"
os.environ["AWS_SECRET_ACCESS_KEY"] = "qydujhbxclvz"


# For the sake of this example, we'll generate a 2D numpy array that consists of 10000 columns
# and 100 rows. Alternatively, one can think of it as a list of 10000 vectors, where each vector
# is 100-dimensional.
my_dataset = np.zeros(shape=(10000, 100), dtype=np.float32)
dataset_filename = "my-dataset.pkl"

# Store the dataset in pickle format on the local disk first
os.makedirs("data", exist_ok=True)
dataset_file_path = os.path.join("data", dataset_filename)
joblib.dump(my_dataset, dataset_file_path, compress=True)

# Initialize minioClient with an endpoint and access/secret keys.
minioClient = Minio(
    os.environ["MLFLOW_S3_ENDPOINT_URL"]
    .replace("http://", "")
    .replace(":9000/", ":9000"),
    access_key=os.environ["AWS_ACCESS_KEY_ID"],
    secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    secure=False,
)

# Make a bucket with the make_bucket API call
# For info on bucket name restrictions, see https://docs.aws.amazon.com/AmazonS3/latest/dev/BucketRestrictions.html
bucket_name = "my-dataset-bucket"
try:
    minioClient.make_bucket(bucket_name)
except BucketAlreadyOwnedByYou as err:
    pass
except BucketAlreadyExists as err:
    pass
except ResponseError as err:
    raise

try:
    minioClient.fput_object(
        bucket_name, object_name=dataset_filename, file_path=dataset_file_path
    )
    print("Dataset uploaded to {}/{}".format(bucket_name, dataset_filename))
except ResponseError as err:
    print(err)
```

When you run the script above successfully, it should output the following:

```
Dataset uploaded to my-dataset-bucket/my-dataset-396b76f9-322f-42e4-89e1-118be0f4514e.pkl
```
 
# Loading a dataset

Assuming you want to load the dataset that you uploaded in the previous script, here's code for
doing that:

```
import os

import joblib
from minio import Minio
from minio.error import ResponseError

# Insert the correct MFlux.ai environment variables here, as shown on the dashboard page
os.environ["MLFLOW_TRACKING_URI"] = "http://159.69.151.173:5000/"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://159.69.151.173:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "updsxtjymzwe"
os.environ["AWS_SECRET_ACCESS_KEY"] = "qydujhbxclvz"

dataset_filename = "my-dataset.pkl".format()

# Store the dataset in pickle format on the local disk first
os.makedirs("downloaded_data", exist_ok=True)

# Initialize minioClient with an endpoint and access/secret keys.
minio_client = Minio(
    os.environ["MLFLOW_S3_ENDPOINT_URL"]
    .replace("http://", "")
    .replace(":9000/", ":9000"),
    access_key=os.environ["AWS_ACCESS_KEY_ID"],
    secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    secure=False,
)

bucket_name = "my-dataset-bucket"
downloaded_file_path = os.path.join("downloaded_data", dataset_filename)
try:
    data = minio_client.get_object(bucket_name, dataset_filename)
    with open(downloaded_file_path, "wb") as file_data:
        for d in data.stream(32 * 1024):
            file_data.write(d)

    print("The file has been downloaded and saved to {}".format(downloaded_file_path))
except ResponseError as err:
    print(err)

my_dataset = joblib.load(downloaded_file_path)
```

# MinIO API reference

[https://docs.min.io/docs/python-client-api-reference.html](https://docs.min.io/docs/python-client-api-reference.html)
