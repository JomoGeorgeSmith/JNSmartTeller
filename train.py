import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

def train_model(X_file, y_file, output_dir):
    # Load preprocessed data
    X = np.load(X_file)  # Input features
    y = np.load(y_file)  # Target variables

    # Ensure X and y have matching sequence counts
    assert X.shape[0] == y.shape[0], f"Inconsistent data shapes: X={X.shape}, y={y.shape}"

    # Reshape targets
    y_type = y[:, :, 0]
    y_currency = y[:, :, 1]
    y_branch = y[:, :, 2]

    # Flatten inputs and outputs for Dense layers
    num_sequences, sequence_length, feature_dim = X.shape
    X = X.reshape(num_sequences * sequence_length, feature_dim)
    y_type = y_type.reshape(-1)
    y_currency = y_currency.reshape(-1)
    y_branch = y_branch.reshape(-1)

    # Filter out padding (-1 in targets) for all target variables
    valid_indices = y_type != -1
    X = X[valid_indices]
    y_type = y_type[valid_indices]
    y_currency = y_currency[valid_indices]
    y_branch = y_branch[valid_indices]

    # Ensure consistent sample counts
    assert len(X) == len(y_type) == len(y_currency) == len(y_branch), "Mismatch in sample sizes after filtering"

    # Split data into training and testing sets
    X_train, X_test, y_train_type, y_test_type = train_test_split(X, y_type, test_size=0.2, random_state=42)
    _, _, y_train_currency, y_test_currency = train_test_split(X, y_currency, test_size=0.2, random_state=42)
    _, _, y_train_branch, y_test_branch = train_test_split(X, y_branch, test_size=0.2, random_state=42)

    # Model definition
    input_layer = Input(shape=(feature_dim,))


    dense_layer = Dense(128, activation='relu')(input_layer)
    dense_layer = Dropout(0.2)(dense_layer)

    output_type = Dense(len(np.unique(y_type)), activation='softmax', name='transaction_type')(dense_layer)
    output_currency = Dense(len(np.unique(y_currency)), activation='softmax', name='transaction_currency')(dense_layer)

    branch_dense = Dense(64, activation='relu')(dense_layer)
    branch_dense = Dropout(0.2)(branch_dense)
    output_branch = Dense(len(np.unique(y_branch)), activation='softmax', name='transaction_branch')(branch_dense)

    model = Model(inputs=input_layer, outputs=[output_type, output_currency, output_branch])

    # Model compilation with weighted loss
    model.compile(
        optimizer='adam',
        loss={
            'transaction_type': 'sparse_categorical_crossentropy',
            'transaction_currency': 'sparse_categorical_crossentropy',
            'transaction_branch': 'sparse_categorical_crossentropy',
        },
        metrics={
            'transaction_type': ['accuracy'],
            'transaction_currency': ['accuracy'],
            'transaction_branch': ['accuracy']
        },
        loss_weights={
            'transaction_type': 1.0,        # Normal weight
            'transaction_currency': 1.0,   # Normal weight
            'transaction_branch': 2.0      # Higher weight to focus on branch prediction
        }
    )

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f'{output_dir}/best_model.h5', save_best_only=True)
    print(f"Shape of X for training: {X.shape}")  # Ensure this matches the model's expected input

    # Train model
    history = model.fit(
        X_train,
        [y_train_type, y_train_currency, y_train_branch],
        validation_data=(X_test, [y_test_type, y_test_currency, y_test_branch]),
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=2
    )

    # Save final model
    model.save(f'{output_dir}/final_model.h5')
    print("Training complete. Final model saved!")

    # Load the best model
    best_model = tf.keras.models.load_model(f'{output_dir}/best_model.h5')
    evaluation = best_model.evaluate(
        X_test,
        [y_test_type, y_test_currency, y_test_branch],
        verbose=2
    )

    # Output evaluation metrics
    print("\nBest Model Evaluation Metrics:")
    print(f"Total Loss: {evaluation[0]:.4f}")
    print(f"Transaction Type Loss: {evaluation[1]:.4f}, Accuracy: {evaluation[4]:.4f}")
    print(f"Transaction Currency Loss: {evaluation[2]:.4f}, Accuracy: {evaluation[5]:.4f}")
    print(f"Transaction Branch Loss: {evaluation[3]:.4f}, Accuracy: {evaluation[6]:.4f}")

# Example usage
train_model(
    '/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/out/X.npy', 
    '/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/out/y.npy', 
    '/Users/jomosmith/Desktop/repos/Jn Smart Teller 2/out'
)
