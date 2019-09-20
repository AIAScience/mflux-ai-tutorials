```python
import mlflow
import mflux_ai
import mlflow.keras
```
_keras_ functions are the same for _sklearn_

|Command|Help|
|---|---|
|**`with mlflow.start_run() as run:`**|exits after indent|
|**&nbsp;&nbsp;&nbsp;&nbsp;`mlflow.keras.log_model(model)`**||
|**`mlflow.search_runs`**(experiment_ids=None, filter_string='')|_Get a DataFrame of runs_|
|**`mlflow.set_experiment`**(experiment_name)|_Set as active experiment, create if not existing_|
|**`mlflow.experiments.list_experiments()`**|_Get a list of experiments_|
|**`mlflow.log_param`**(param_name, value)||
|**`mlflow.log_params`**(params)|params = {"k1": "v1", "k2": 2} **Similarly for log_metrics, set_tags and log_artifacts. See also mlflow.log_metrics**|
|**`mlflow.log_metric`**(metric_name, value, step=None)|"mse", 123, step=2|
|**`mlflow.set_tag`**(tag_name, value)|key is string, value will be stringified|
|**`mlflow.delete_tag`**(tag_name)|_irreversible_|
|**`mlflow.log_artifact`**(local_path, artifact_path=None)||
|**`mlflow.get_artifact_uri`**(artifact_path=None)||
|**`mlflow.keras.save_model`**(model, path)||
|**`mlflow.keras.log_model`**(model, artifact_path)||
|**`mlflow.keras.load_model`**(model_uri, **kwargs)||
|**`mflux_ai.put_dataset(dataset, "name.pkl")`**|_Upload dataset_|
|**`mflux_ai.get_dataset("name.pkl")`**|_Download dataset_|