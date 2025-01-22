import pandas as pd
import json

# Load the Excel file
excel_file = '/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/data/TranactionExtractForWendel.xlsx'
data = pd.read_excel(excel_file)


# Create a unique list of user_ids
unique_user_ids = data['account_no'].unique()

# Create a mapping of user_id to index
user_id_to_index = {str(user_id): idx for idx, user_id in enumerate(unique_user_ids)}

# Save the mapping as a JSON file
output_path = '/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/userstore/user_id_to_index.json'
with open(output_path, 'w') as f:
    json.dump(user_id_to_index, f)

print(f"account_no to Index mapping saved at {output_path}")
