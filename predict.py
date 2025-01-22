import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime
import json


class TransactionPredictor:
    def __init__(self, model_path, data_path, encoders_path, account_no_map_path):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load the preprocessed data (X and y)
        self.X = np.load(f"{data_path}/X.npy")
        self.y = np.load(f"{data_path}/y.npy")
        
        # Load label encoders for decoding predictions
        with open(encoders_path, 'rb') as f:
            self.encoders = pickle.load(f)
        
        # Load the account_no to index mapping
        with open(account_no_map_path, 'r') as f:
            self.account_no_to_index = json.load(f)

    def get_account_history(self, account_no):
        """
        Retrieve the transaction history for a specific account.
        :param account_no: The account number.
        :return: Flattened account transaction history as a 2D NumPy array.
        """
        if str(account_no) not in self.account_no_to_index:
            raise ValueError(f"No transaction history found for account_no: {account_no}")
        
        account_index = self.account_no_to_index[str(account_no)]
        account_history = self.X[account_index]

        # Flatten the history (remove sequence dimension)
        return account_history.reshape(-1, self.X.shape[2])  # Shape: (num_samples, feature_dim)

    def predict(self, account_no):
        """
        Predict the top 3 most likely transaction details for an account.
        :param account_no: The account number.
        :return: Decoded top 3 predictions with probabilities as a dictionary.
        """
        # Retrieve account transaction history
        account_history = self.get_account_history(account_no)

        # Ensure account history matches model input shape
        input_features = account_history[:, :8]  # Use only the first 8 features expected by the model

        # Debugging: Check input shape
        print(f"Input shape for prediction: {input_features.shape}")

        # Predict the next transaction
        predictions = self.model.predict(input_features)

        # Helper function to get top 3 predictions
        def get_top_predictions(probabilities, encoder):
            top_indices = np.argsort(probabilities[0])[-3:][::-1]
            return [
                {
                    "label": encoder.inverse_transform([idx])[0] if idx < len(encoder.classes_) else "Unknown",
                    "probability": probabilities[0][idx]
                }
                for idx in top_indices
            ]

        # Decode predictions with top 3 for each category
        top_transaction_types = get_top_predictions(predictions[0], self.encoders['transaction_type'])
        top_transaction_currencies = get_top_predictions(predictions[1], self.encoders['transaction_currency_code'])
        top_transaction_branches = get_top_predictions(predictions[2], self.encoders['transaction_branch'])

        return {
            "Transaction Types": top_transaction_types,
            "Transaction Currencies": top_transaction_currencies,
            "Transaction Branches": top_transaction_branches
        }



if __name__ == "__main__":
    model_path = '/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/out/best_model.h5'
    data_path = '/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/out'
    encoders_path = '/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/out/encoders.pkl'
    account_no_map_path = '/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/userstore/user_id_to_index.json'
    
    predictor = TransactionPredictor(model_path, data_path, encoders_path, account_no_map_path)

    # Input account_no
    account_no = 29235  # Replace with the desired account number
    
    # Get top 3 predictions for each category
    predictions = predictor.predict(account_no)
    
    # Display predictions in a user-friendly format
    print(f"\nTop Predictions for Account No {account_no}:\n")
    print("Transaction Types:")
    for i, pred in enumerate(predictions["Transaction Types"], 1):
        print(f"  {i}. {pred['label']} ({pred['probability']:.2%})")
    
    print("\nTransaction Currencies:")
    for i, pred in enumerate(predictions["Transaction Currencies"], 1):
        print(f"  {i}. {pred['label']} ({pred['probability']:.2%})")
    
    print("\nTransaction Branches:")
    for i, pred in enumerate(predictions["Transaction Branches"], 1):
        print(f"  {i}. {pred['label']} ({pred['probability']:.2%})")
