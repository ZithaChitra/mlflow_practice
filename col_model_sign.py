# To include a signature with your model, pass To include a 
# signature object as an argument to the appropriate log_model call
# , e.g. sklearn.log_model().
# The model signature can be created by hand or infered from datasets
# with valid model inputs (e.g. training data)


# This implementation inferes from valid data to create a
# signature object
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient


client = MlflowClient()
experiment_id = client.create_experiment("col_model")
print(f"experiment_id: {experiment_id}")
iris = datasets.load_iris()
iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
with mlflow.start_run(experiment_id=experiment_id) as run:
	clf = RandomForestClassifier(max_depth=7, random_state=0)
	clf.fit(iris_train, iris.target)
	signature = infer_signature(iris_train, clf.predict(iris_train))
	mlflow.sklearn.log_model(clf, "iris_rf", signature=signature)



# This implementation creates the signature explicitly/manually
# from mlflow.models.signature import ModelSignature
# from mlflow.types.schema import Schema, ColSpec

# input_schema = Schema([
# 	ColSpec("double", "sepal length (cm)"),
# 	ColSpec("double", "sepal width (cm)"),
# 	ColSpec("double", "petal length (cm)"),
# 	ColSpec("double", "petal width (cm)")
# ])

# output_schema = Schema([ ColSpec("long") ])
# signature = ModelSignature(inputs=input_schema, outputs=output_schema)























































































































































































