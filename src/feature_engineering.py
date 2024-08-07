import pandas as pd

def extract_features(data):
    """
    Extracts and engineers features from the transaction data.

    Parameters:
    - data: DataFrame, the processed transaction data.

    Returns:
    - DataFrame, with new features added.
    """
    # Extract hour from transaction_time
    data['transaction_hour'] = pd.to_datetime(data['transaction_time']).dt.hour
    print("Transaction hour extracted.")

    # Create a feature for amount squared
    data['amount_squared'] = data['amount'] ** 2
    print("Amount squared feature created.")

    # Create a feature for balance ratio (balance/amount)
    data['balance_ratio'] = data['balance'] / (data['amount'] + 1)  # Add 1 to avoid division by zero
    print("Balance ratio feature created.")

    return data

if __name__ == '__main__':
    # Load processed data
    data = pd.read_csv('data/processed_data.csv')
    print("Processed data loaded.")

    # Extract features
    data = extract_features(data)

    # Save the data with new features
    data.to_csv('data/processed_data_with_features.csv', index=False)
    print("Data with new features saved to 'data/processed_data_with_features.csv'.")
