import time

from mlflow.tracking import MlflowClient
from mlflow.entities import Metric, Param, RunTag

client = MlflowClient()
experiment_id = client.create_experiment("new expenza")
# client.set_experiment(experiment_id)


def print_run_info(r):
	print(f"run_id: {r.info.run_id}")
	print(f"params: {r.data.params}")
	print(f"metrics: {r.data.metrics}")
	print(f"tags: {r.data.tags}")
	print(f"status: {r.info.status}")


# Create Mlflow entities and a run under the default experiment
# (whose id is "0")
timestamp = int(time.time() * 1000)
metrics = [Metric("m", 1.5, timestamp, 1)]
params = [Param("p", "p")]
tags = [RunTag("t", "t")]
experiment_id = "1"

run = client.create_run(experiment_id=experiment_id)

# log entities, terminate the run, and fetch run status
client.log_batch(run.info.run_id, metrics=metrics, params=params, tags=tags)
client.set_terminated(run.info.run_id)
run = client.get_run(run.info.run_id)
print_run_info(run)


















































































































