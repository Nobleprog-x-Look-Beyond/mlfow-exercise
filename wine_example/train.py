# TASK: Define a function to train and log the model to MLflow
# TASK: Within that function, log hyperparameters and metrics, create confusion matrix plots
# TASK: Save the trained model locally, log the model artifact
# TASK: Perform a hyperparameter sweep in the main() function
# TASK: Log the best model and register it in the MLflow Model Registry
# TASK: Transition the newly registered model to the desired stage (Staging/Production)
# TASK: Print final results and accuracy
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import joblib

import mlflow
import mlflow.sklearn


def plot_confusion_matrix(cm: ConfusionMatrixDisplay, max_depth, n_estimators):
    cm.plot()
    plt.title(f"Confusion Matrix (max_depth={max_depth}, n_estimators={n_estimators})")
    plt.tight_layout()

    plot_path = f"plots/confusion_matrix_max_depth_{max_depth}_n_estimators_{n_estimators}.png"
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path)


def train_model(X_train, y_train, X_val, y_val, max_depth, n_estimators):
    with mlflow.start_run(nested=True) as child_run:
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('n_estimators', n_estimators)
        # Define the model
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
        # Fit the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        mlflow.log_metric('accuracy', accuracy)

        # Log the confusion matrix
        cm = ConfusionMatrixDisplay.from_estimator(model, X_val, y_val, cmap='Blues')
        plot_confusion_matrix(cm, max_depth, n_estimators)

        mlflow.sklearn.log_model(model, 'model')
        print("accuracy: ", accuracy)

        return model, accuracy, child_run.info.run_id



def main():
    mlflow.set_experiment('wine_experiment')

    # Load the Iris dataset
    wine = load_wine()
    data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    data['target'] = wine.target
    data.to_csv('data/wine.csv', index=False)

    # define X and y (features and target)
    X = data.drop('target', axis=1)
    y = data['target']

    # Split the data into train and validation sets, split 80/20
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameters to try
    max_depth_values = [2, 10]
    n_estimators_values = [10, 100]


    # Initialize variables to keep track of the best model
    best_accuracy = 0
    best_run_id = None
    best_parameters = None
    best_model = None

    with mlflow.start_run():
        for max_depth in max_depth_values:
            for n_estimator in n_estimators_values:
                print(f"Training model with max_depth={max_depth} and n_estimators={n_estimator}")

                # Train the model
                model, accuracy, run_id= train_model(X_train, y_train, X_val, y_val, max_depth, n_estimator)

                # Check if this model is better than the previous best one
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_run_id = run_id
                    best_parameters = {'max_depth': max_depth, 'n_estimators': n_estimator}

        # Log the best model
        registered_model = mlflow.register_model(f"runs:/{best_run_id}/model", "best_jan_model")
        mlflow.log_metric('best_accuracy', best_accuracy)
        mlflow.log_params(best_parameters)

        model_version = registered_model.version
        print("Best accuracy: ", best_accuracy)
        print("Best parameters: ", best_parameters)
        print(f"Best model version: {model_version}")


        # Staging the model
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="best_jan_model",
            version=model_version,
            stage="Staging"
        )
        print(f"Model with version {model_version} transitioned to Staging")



if __name__ == "__main__":
    main()