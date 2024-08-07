import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_path, output_path):
    """
    Preprocess the transaction data.
    
    Parameters:
    - input_path: str, path to the raw data CSV file.
    - output_path: str, path where the processed data CSV will be saved.
    """
    # Load raw data
    data = pd.read_csv(input_path)
    print("Data loaded successfully.")

    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    print("Missing values handled.")

    # Encode categorical variables
    le = LabelEncoder()
    data['category'] = le.fit_transform(data['category'])
    print("Categorical variables encoded.")

    # Normalize numerical features
    scaler = StandardScaler()
    data[['amount', 'balance']] = scaler.fit_transform(data[['amount', 'balance']])
    print("Numerical features normalized.")

    # Save processed data
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to '{output_path}'.")

if __name__ == '__main__':
    preprocess_data('data/raw_data.csv', 'data/processed_data.csv')
