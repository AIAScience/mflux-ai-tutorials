If you have some code that logs many metrics (or parameters or tags) to an MLflow tracking
server, you should know that each request takes time due to network roundtrips, and it quickly
adds up. In the following example, we assume that you have a deep learning script that trains
for 50 epochs, and you want to log 4 metrics for each epoch. For the sake of this example, let's
log metrics from a keras history object. The script that would generate this history object
could look like this:

```python
model = Sequential()
model.add(Dense(num_categories, activation="softmax"))
model.compile(
    loss="categorical_crossentropy", optimizer=SGD(), metrics=["accuracy"]
)
history = model.fit(
    training_input_vectors,
    training_target_vectors,
    epochs=50,
    validation_data=(validation_input_vectors, validation_target_vectors),
)
``` 

## Slow example

~200 requests, takes ~55 seconds, **do not use** this approach:

```python
import mlflow

for metric_name in history.history:
    for i in range(len(history.history[metric_name])):
        mlflow.log_metric(
            key=metric_name, value=history.history[metric_name][i], step=i
        )
```


To speed it up by a factor of 4, you can use `log_metrics` instead of `log_metric` to reduce the
amount of requests from 200 to 50:

## Faster example

~50 requests, takes ~14 seconds:

```python
import mlflow

for i in tqdm(history.epoch, desc="Logging metrics"):
    metrics = {}
    for metric_name in history.history:
        metrics[metric_name] = history.history[metric_name][i]
    mlflow.log_metrics(metrics, step=i)
```

The speed of this approach is similar to `mlflow.keras.autolog()`, which performs one
`log_metrics` request after each epoch during training.

If you want to further improve the speed, you can use MLflow's `log_batch` method to log all
metrics in a single request instead of 50 requests:

## Even faster example

Takes ~3 seconds:

```python
import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient


with mlflow.start_run() as active_run:
    mlflow_client = MlflowClient()
    all_metrics = []
    for metric_name in history.history:
        for i in history.epoch:
            metric = Metric(
                key=metric_name,
                value=history.history[metric_name][i],
                timestamp=0,
                step=i,
            )
            all_metrics.append(metric)

    mlflow_client.log_batch(run_id=active_run.info.run_id, metrics=all_metrics)
```

Note that there is a limit on the number of metrics that you can log in a single `log_batch`
call. This limit is typically 1000. If you exceed the limit, you'll get an error like this:

> A batch logging request can contain at most 1000 metrics. Got 2000 metrics. Please split up
metrics across multiple requests and try again.
