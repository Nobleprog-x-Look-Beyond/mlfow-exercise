import joblib
import numpy as np


def main():
    # Load the model
    model = joblib.load('models/model.pkl')

    # Define a sample input
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])

    # Perform inference
    prediction = model.predict(sample_input)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()