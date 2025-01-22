from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import TransactionPredictor
import numpy as np
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Paths to necessary files
model_path = '/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/out/best_model.h5'
data_path = '/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/out'
encoders_path = '/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/out/encoders.pkl'
account_no_map_path = '/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/userstore/user_id_to_index.json'

# Initialize predictor
predictor = TransactionPredictor(model_path, data_path, encoders_path, account_no_map_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    account_no = data.get('account_no')
    
    if not account_no:
        return jsonify({'error': 'Account number (account_no) is required.'}), 400
    
    try:
        predictions = predictor.predict(account_no)

        # Convert all numpy objects to Python native types
        predictions = {
            key: [
                {subkey: (float(value) if isinstance(value, np.float32) else value)
                 for subkey, value in pred.items()}
                for pred in predictions[key]
            ]
            for key in predictions
        }
        return jsonify(predictions)
    except ValueError as e:
        app.logger.error(f"ValueError: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500




if __name__ == '__main__':
    app.run(debug=True)
