# The following example demonstrates how to store a model 
# signature for a simple classifier trained on the mnist dataset.
# A tensor based signature

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten 
from keras.optimizers import SGD
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
trainX = train_X.reshape((train_X.shape[0], 28, 28, 1))
testX = test_X.reshape((test_X.shape[0], 28, 28, 1))
trainY = to_categorical(train_Y)
testY = to_categorical(test_Y)




mlflow.set_tracking_uri("sqlite:///mlruns.db")
client = MlflowClient()
# experiment_id = client.create_experiment("mnist classifier")


with mlflow.start_run() as run:
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY))
	mlflow.keras.autolog()
	signature = infer_signature(testX, model.predict(testX))
	mlflow.keras.log_model(model, "mnist_classifier", signature=signature)



















