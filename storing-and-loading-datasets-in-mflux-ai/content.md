It's easy to store datasets in MFlux.ai and make them available to your data science team.
MFlux.ai's object storage system is powered by [MinIO](https://min.io/), an open source,
enterprise-grade Amazon S3-compatible Object Storage system.

# Install requirements

`pip install mflux_ai numpy`

# Store a dataset

```
import mflux_ai

# Note: in the following line, insert the project token shown on your dashboard page.
mflux.init("your_project_token_goes_here")

# Generate some dummy data (a numpy array)
my_dataset = np.zeros(shape=(10000, 100), dtype=np.float32)
object_name = "my-dataset.pkl"

mflux_ai.put_dataset(my_dataset, object_name)

print("Dataset uploaded")
```

When you run the script above successfully, it should output `Dataset uploaded!`

# Loading a dataset

Assuming you want to load the dataset that you uploaded in the previous script, here's code for
doing that:

```
import mflux_ai

# Note: in the following line, insert the project token shown on your dashboard page.
mflux.init("your_project_token_goes_here")

object_name = "my-dataset.pkl"
my_loaded_dataset = mflux_ai.get_dataset(object_name)
```

# MinIO API reference

[https://docs.min.io/docs/python-client-api-reference.html](https://docs.min.io/docs/python-client-api-reference.html)
