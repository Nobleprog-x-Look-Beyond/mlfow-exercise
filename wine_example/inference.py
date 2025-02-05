"""
inference.py

Loads the "best" saved model (or any model file) and performs inference on a sample data point.
You can adapt this to load a model from MLflow's model registry if you want to skip local artifacts entirely.
"""

import mlflow.pyfunc
import numpy as np
from mlflow.tracking import MlflowClient


def evaluate_model(prediction):
    # Implement your evaluation logic here
    # Return a random accuracy for demonstration purposes
    return np.random.rand()


def evaluate_model_stages(client: MlflowClient, model_name):
    # Get a list of latest model versions

    model_versions = client.get_latest_versions(
        model_name, stages=["Staging", "Production"]
    )
    # This is a sample input from the wine dataset, it can be replaced with any other input though
    sample_input = np.array(
        [[12.7, 3.43, 2.36, 21.0, 111.0, 1.19, 1.61, 0.48, 0.99, 3.13, 1.27, 2.4, 463]]
    )

    best_model = None
    best_accuracy = -1
    for version in model_versions:
        # TODO: Load the model from MLflow and evaluate the best one
        model_uri = f"models:/{model_name}/{version.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        prediction = model.predict(sample_input)

        accuracy = evaluate_model(prediction)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = version

    if best_model:
        # move the best model to production
        client.transition_model_version_stage(
            name=model_name, version=best_model.version, stage="Production"
        )
        print(f"Deployed model version {best_model.version} to Production")

        print(f"Sample input: {sample_input}")


def main():
    # Load the models from MLflow
    model_name = "best_wine_model"
    client = MlflowClient()
    evaluate_model_stages(client, model_name)


if __name__ == "__main__":
    main()
