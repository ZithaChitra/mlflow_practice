# Adding an mlflow model to the model registry
from random import random, randint
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("sqlite:///mlruns.db")
with mlflow.start_run(run_name="YOUR_RUN_NAME") as run:
	params = {"n_estimators": 5, "random_state": 42}
	sk_learn_rfr = RandomForestRegressor(**params)


	# Log parameters and metrics using the MLflow APIs
	mlflow.log_params(params)
	mlflow.log_param("param_1", randint(0, 100))
	mlflow.log_metrics({"metric_1": random(), "metric_2": random()})

	# log the sklearn model and register as version 1
	mlflow.sklearn.log_model(
		sk_model=sk_learn_rfr,
		artifact_path="sklearn-model"
	)


client = MlflowClient()
client.create_registered_model("sk-learn-random_forest-reg-model")

client.create_model_version(
	name="sk-learn-random-forest-reg-model",
	source = "",
	run_id = ""
)





# Fetching an mlflow model from the model registry

import mlflow.pyfunc

model_name = "sk-learn-random-forest-reg-model"
model_version = 1


data = ""

model = mlflow.pyfunc.load_model(
	model_uri = f"models:/{model_name}/{model_version}"
)

model.predict(data)





# Fetch the latest model version in a specific stage


































































































































































