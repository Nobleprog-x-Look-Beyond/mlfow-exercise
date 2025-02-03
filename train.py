# TASK: Import necessary libraries (mlflow, sklearn, joblib, etc.)
# TASK: Load the iris(not Wine) dataset using sklearn.datasets
# TASK: Split the data into train and validation sets
# TASK: Define a function to train and log the model to MLflow
# TASK: Within that function, log hyperparameters and metrics, create confusion matrix plots
# TASK: Save the trained model locally, log the model artifact
# TASK: Perform a hyperparameter sweep in the main() function
# TASK: Log the best model and register it in the MLflow Model Registry
# TASK: Transition the newly registered model to the desired stage (Staging/Production)
# TASK: Print final results and accuracy

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    mlflow.set_experiment('iris_experiment')

    with mlflow.start_run():

        iris = load_iris(as_frame=True)
        print(iris.data) # type: ignore
        print(iris.feature_names) # type: ignore

        X, y = iris.data, iris.target # type: ignore

        # Split the data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Print the shapes of the train and validation sets
        print(f"Training features shape: {X_train.shape}")
        print(f"Validation features shape: {X_val.shape}")
        print(f"Training target shape: {y_train.shape}")
        print(f"Validation target shape: {y_val.shape}")

        # Define a function to train it
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        # Log the accuracy
        print(f"Accuracy: {accuracy}")
        mlflow.log_metric('accuracy', float(accuracy))

        # Save the model
        joblib.dump(model, 'models/model.pkl')
        mlflow.sklearn.log_model(model, 'model')

main()
