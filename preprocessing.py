import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_and_save(input_file, output_dir):
    # Load data
    data = pd.read_excel(input_file)
    
    # Drop unnecessary columns
    data.drop(columns=['transaction_id', 'account_type', 'branch_id', 'account_status',
                       'customer_no', 'posted_date', 'transaction_source', 
                       'transaction_code', 'tran_set_id', 'original_currency_code'], inplace=True, errors='ignore')
    
    # Fill missing values
    data.fillna(method='ffill', inplace=True)
    
    # Create new features
    data['transaction_date'] = pd.to_datetime(data['transaction_date'])
    data['day_of_week'] = data['transaction_date'].dt.dayofweek  # Monday = 0, Sunday = 6
    data['hour_of_day'] = data['transaction_date'].dt.hour  # Hour of the day (0-23)
    
    # Cyclical encoding for day_of_week and hour_of_day
    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    data['hour_of_day_sin'] = np.sin(2 * np.pi * data['hour_of_day'] / 24)
    data['hour_of_day_cos'] = np.cos(2 * np.pi * data['hour_of_day'] / 24)

        # Convert transaction_date to day_of_year features
    data['day_of_year'] = data['transaction_date'].dt.dayofyear
    data['day_of_year_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
    data['day_of_year_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365)
    
    # Compute branch popularity
    branch_popularity = data['transaction_branch'].value_counts(normalize=True)
    data['branch_popularity'] = data['transaction_branch'].map(branch_popularity)
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    data['amount'] = scaler.fit_transform(data[['amount']])
    data['branch_popularity'] = scaler.fit_transform(data[['branch_popularity']])
    
    # Encode categorical data
    encoders = {}
    for column in ['transaction_type', 'transaction_currency_code', 'transaction_branch']:
        encoders[column] = LabelEncoder()
        data[column] = encoders[column].fit_transform(data[column])
    
    # Group data by user_id
    grouped_data = []
    target_data = []
    for _, group in data.groupby('account_no'):
        if len(group) == 1:  # Handle single transaction users
            group = pd.concat([group, group])  # Duplicate the single transaction
        
        grouped_data.append(group[['amount', 'day_of_week_sin', 'day_of_week_cos',
                                'hour_of_day_sin', 'hour_of_day_cos',
                                'branch_popularity', 'day_of_year_sin', 'day_of_year_cos']].values)

        target_data.append(group[['transaction_type', 'transaction_currency_code', 'transaction_branch']].values)
    
    # Pad sequences to the same length for LSTM
    X = pad_sequences(grouped_data, dtype='float32', padding='post', value=0.0)
    y = pad_sequences(target_data, dtype='int32', padding='post', value=-1)  # Use -1 for masked padding
    
    # Save encoders, scaler, and processed sequences
    with open(f'{output_dir}/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    with open(f'{output_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    np.save(f'{output_dir}/X.npy', X)
    np.save(f'{output_dir}/y.npy', y)
    
    print("Preprocessing complete. Files saved!")

# Example usage
preprocess_and_save('/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/data/TranactionExtractForWendel.xlsx', '/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/out')
