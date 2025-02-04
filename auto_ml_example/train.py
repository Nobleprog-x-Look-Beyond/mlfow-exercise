# TASK: Import necessary libraries (mlflow, sklearn, joblib, etc.)
# TASK: Load the Wine dataset using sklearn.datasets
# TASK: Split the data into train and validation sets
# TASK: Define a function to train and log the model to MLflow
# TASK: Within that function, log hyperparameters and metrics, create confusion matrix plots
# TASK: Save the trained model locally, log the model artifact
# TASK: Perform a hyperparameter sweep in the main() function
# TASK: Log the best model and register it in the MLflow Model Registry
# TASK: Transition the newly registered model to the desired stage (Staging/Production)
# TASK: Print final results and accuracy
from sklearn.datasets import load_iris, load_wine
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

import mlflow
import mlflow.sklearn

from tpot import TPOTClassifier


tpot_config = {
    "generations": 5,
    "population_size": 20,
    "verbosity": 2,
    "random_state": 42,
}

def main():
    mlflow.set_experiment('tpot_experiment')

    with mlflow.start_run():

        # Load the Iris dataset
        iris = load_wine()
        data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        data['target'] = iris.target

        # define X and y (features and target)
        X = data.drop('target', axis=1)
        y = data['target']

        # Split the data into train and validation sets, split 80/20
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define a function to train it
        model = TPOTClassifier()

        print("Training model...")
        model.fit(X_train, y_train)

        print("Evaluating model...")
        # Evaluate the best model
        accuracy = model.score(X_val, y_val)
        

        # Log the accuracy
        print(f"Accuracy: {accuracy}")
        mlflow.log_metric('accuracy', accuracy)

        # Save the model
        mlflow.sklearn.log_model(model, 'model')

        # Export the model training script
        model.export("tpot_training.py")
        mlflow.log_artifact("tpot_training.py")


if __name__ == "__main__":
    main()