import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomRorestRegressor

def print_models_info(mv):
	for m in mv:
		print(m.name)
		print(m.version)
		print(m.run_id)
		print(m.current_stage)


mlflow.set_tracking_uri("sqlite:///mlruns.db")

# Create two runs and 