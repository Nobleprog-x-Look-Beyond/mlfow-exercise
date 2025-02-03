import joblib
import numpy as np
import mlflow

@mlflow.trace(name="example_inference", attributes={"key": "value"})
def predict(input, model):
    prediction = model.predict(input)
    return prediction

def main():
    # Load the model
    model = joblib.load('models/model.pkl')
    mlflow.set_experiment('iris_experiment')


    # Define a sample input
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])

    # Perform inference
    with mlflow.start_run():
        prediction = predict(sample_input, model)
        print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()