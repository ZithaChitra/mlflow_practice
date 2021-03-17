
import mlflow

mlflow.set_tracking_uri() # connects to a tracking uri
mlflow.tracking.get_tracking_uri() # returns current tracking uri

experiment_id = mlflow.create_experiment("Social NLP Experiments") # creates a new exp and returns it's id
experiment = mlflow.get_experiment(experiment_id)

# Set an experiment name, which must be unique and case sensitive
# if you do not specify an experiment in mlflow.start_run(), new
# runs are launched under this experiment.
mlflow.set_experiment("Social NLP Experiments")

# Get Experiment Details
experiment = mlflow.get_experiment_by_name("Social NLP Experiments")
print(experiment.experiment_id)
print(experiment.artifact_location)
print(experiment.tags)
print(experiment.lifecycle_stage)



# Start a new MLflow run, setting it as the active run under which
# metrics and parameters will be logged.
with mlflow.start_run(run_name="RARENT_RUN") as parent_run:
	mlflow.log_param("parent", "yes")
	with mlflow.start_run(run_name="CHILD_RUN", nested=True) as child_run:
		mlflow.log_param("child", "yes")


mlflow.autolog() # call before training code
with mlflow.start_run():
	for epoch in range(0, 3):
		mlflow.log_param(key="quality", value=2*epoch, step=epoch)


# or use library specific auto-log calls
mlflow.tensorflow.autolog() # call before training code
mlflow.keras.autolog() # call before training code







# MLflow provides a more detailed Tracking Service API for 
# managing experiments and runs directly, which is available through
# client SDK in the mlflow.tracking module

from mlflow.tracking import MlflowClient


client = MlflowClient()
experiments = client.list_experiments()
run = client.create_run(experiments[0].experiment_id)
client.log_param(run.info.run_id, "hello", "world")
client.set_terminated(run.info.run_id)




# logging to a remote server 
remote_server_uri = "..." 
mlflow.set_tracking_uri(remote_server_uri)
# On Databricks, the experiment name passed to mlflow.set_experiment
# must be a valid path in the workspace
mlflow.set_experiment("/my-experiment")
with mlflow.start_run():
	mlflow.log_param("a", 1)
	mlflow.log_param("b", 2)

















