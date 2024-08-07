import pandas as pd
from model import FraudDetectionModel

def make_predictions(input_path, model_path):
    """
    Make predictions using the trained model.

    Parameters:
    - input_path: str, path to the processed data CSV file.
    - model_path: str, path to the trained model file.
    """
    # Load processed data
    data = pd.read_csv(input_path)
    print("Processed data loaded.")

    # Separate features from labels if necessary
    X = data.drop('fraudulent', axis=1, errors='ignore')
    
    # Load the trained model
    model = FraudDetectionModel()
    model.load(model_path)

    # Make predictions
    predictions = model.predict(X)
    data['predictions'] = predictions
    print("Predictions made.")

    # Save predictions
    data.to_csv('data/predictions.csv', index=False)
    print("Predictions saved to 'data/predictions.csv'.")

if __name__ == '__main__':
    make_predictions('data/processed_data_with_features.csv', 'src/fraud_detection_model.pkl')
