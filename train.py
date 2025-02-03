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
from sklearn.datasets import load_iris
import pandas as pd

def main():
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target

    x = data.drop('target', axis=1)
    y = data['target']

main()