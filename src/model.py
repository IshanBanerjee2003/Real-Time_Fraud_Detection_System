from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd

class FraudDetectionModel:
    def __init__(self):
        """
        Initialize the FraudDetectionModel with a RandomForestClassifier.
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X, y):
        """
        Train the Random Forest model on the given data.

        Parameters:
        - X: DataFrame, feature set for training.
        - y: Series, target labels for training.
        """
        self.model.fit(X, y)
        print("Model training completed.")

    def evaluate(self, X, y):
        """
        Evaluate the model on the given test data.

        Parameters:
        - X: DataFrame, feature set for testing.
        - y: Series, target labels for testing.

        Returns:
        - accuracy: float, the accuracy of the model on the test data.
        """
        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)
        print("Model evaluation completed.")
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(classification_report(y, predictions))
        return accuracy

    def predict(self, X):
        """
        Make predictions with the trained model.

        Parameters:
        - X: DataFrame, feature set for predictions.

        Returns:
        - predictions: ndarray, predicted labels for the input data.
        """
        return self.model.predict(X)

    def save(self, model_path):
        """
        Save the trained model to a file.

        Parameters:
        - model_path: str, path where the model will be saved.
        """
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load(self, model_path):
        """
        Load a trained model from a file.

        Parameters:
        - model_path: str, path from where the model will be loaded.
        """
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

if __name__ == '__main__':
    # Load the data with features
    data = pd.read_csv('data/processed_data_with_features.csv')
    X = data.drop('fraudulent', axis=1)
    y = data['fraudulent']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize, train, and evaluate the model
    model = FraudDetectionModel()
    model.train(X_train, y_train)
    model.evaluate(X_test, y_test)

    # Save the model
    model.save('src/fraud_detection_model.pkl')
