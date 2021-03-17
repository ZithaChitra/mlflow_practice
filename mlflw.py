# import mlflow.sklearn
# from mlflow.tracking import MlflowClient
# from sklearn.ensemble import RandomForestRegressor

# def print_models_info(mv):
# 	for m in mv:
# 		print(m.name)
# 		print(m.version)
# 		print(m.run_id)
# 		print(m.current_stage)


# mlflow.set_tracking_uri("sqlite:///mlruns.db")

# # Create two runs and log Mlflow entities
# with mlflow.start_run() as run1:
# 	params = {"n_estimators": 3, "random_state": 42}
# 	rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])
# 	mlflow.log_params(params)
# 	mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")


# with mlflow.start_run() as run2:
# 	params = {"n_estimators": 6, "random_state": 42}
# 	rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])
# 	mlflow.log_params(params)
# 	mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")


# # Register model name in the model registries
# name = "RandomForestRegression"
# client = MlflowClient()
# client.create_registered_model(name)


# # Create two versions of the rfr model under the registered model name
# for run_id in [run1.info.run_id, run2.info.run_id]:
# 	model_uri = "runs:/{}/sklearn-model".format(run_id)
# 	mv = client.create_model_version(name, model_uri, run_id)
# 	print("model versiom {} created".format(mv.version))



# print("--")


# # fetch latest version; this will be version 2
# print("Deleting latest versiom {}".format(mv.version))
# client.delete_model_version(name, mv.version)
# models = client.get_latest_versions(name, stages=["None"])
# print_models_info(models)


import mlflow
from mlflow.tracking import MlflowClient

def print_registered_models_info(r_models):
	print("--")
	for rm in r_models:
		print(f"name: {rm.name}")
		print(f"tags: {rm.tags}")
		print(f"description: {rm.description}")


mlflow.set_tracking_uri("sqlite:///mlruns.db")
client = MlflowClient()

# Register a couple of models with respctive names, tags and descs
for name, tags, desc in [("name1", {"t1":"t1"}, "description1"),
						("name2", {"t2":"t2"}, "description2")]:
						client.create_registered_model(name, tags, desc)

# Fetch all registered models
print_registered_models_info(client.list_registered_models())

# Delete one registered model and fetch again
client.delete_registered_model("name1")
print_registered_models_info(client.list_registered_models())














































































































































