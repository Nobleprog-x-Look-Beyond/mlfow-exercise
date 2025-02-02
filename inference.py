"""
inference.py

Loads the "best" saved model (or any model file) and performs inference on a sample data point.
You can adapt this to load a model from MLflow's model registry if you want to skip local artifacts entirely.
"""

import mlflow.pyfunc
import numpy as np
from mlflow.tracking import MlflowClient


def evaluate_model_stages(client: MlflowClient, model_name):
    # TODO: Get a list of latest model versions
    model_versions = []
    # This is a sample input from the wine dataset, it can be replaced with any other input though
    sample_input = np.array(
        [[12.7, 3.43, 2.36, 21.0, 111.0, 1.19, 1.61, 0.48, 0.99, 3.13, 1.27, 2.4, 463]]
    )

    best_model = None
    best_prediction = None
    best_accuracy = -1

    for version in model_versions:
        # TODO: Load the model from MLflow and evaluate the best one
        pass

    if best_model:
        # TODO: move the best model to production
        print(f"Deployed model version {best_model.version} to Production")

    print(f"Sample input: {sample_input}")
    print(f"Best Predicted Wine Class: {best_prediction[0]}")


def main():
    # Load the models from MLflow
    model_name = "WineRandomForest"
    client = MlflowClient()
    evaluate_model_stages(client, model_name)


def evaluate_model(prediction):
    # Implement your evaluation logic here
    # Return a random accuracy for demonstration purposes
    return np.random.rand()


if __name__ == "__main__":
    main()
