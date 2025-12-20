import numpy as np
import os
import pickle
import sys
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add project root folder to path for importing utils
sys.path.append(f"{CURRENT_DIR}/../")

try:
    from utils import DataPoint, ScorerStepByStep
except ModuleNotFoundError as e:
    print("ModuleNotFoundError occurred:", e)
    print("sys.path was:", sys.path)

## --- My LSTM Model ---
from train import MyLSTM

## Hyperparameters
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.0
NUM_FEATURES = 32

class PredictionModel:
    
    def __init__(self):
        """
        Initialize and load model and artifacts.
        """
        # For hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_features = NUM_FEATURES
        self.num_layers = NUM_LAYERS
        self.hidden_size = HIDDEN_SIZE
        self.dropout = DROPOUT

        # Initialise model
        self.model = MyLSTM(
            input_size=self.num_features, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            output_size=self.num_features,
            dropout=DROPOUT
        )
        
        # Load the saved weights inside model.pth file at root
        self.model.load_state_dict(torch.load("model.pth"))
        # Set model to evaluation mode
        self.model.eval()

        # Load the saved scaler from scaler.pkl at root
        with open("scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        # Initialize variables to track the model's sequence
        self.current_seq_id = -1
        self.current_hidden_state = None

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict a market state for every single step in the dataset if needed.
        """
        # Check if this is a new sequence
        if data_point.seq_ix != self.current_seq_id:
            self.current_seq_id = data_point.seq_ix
            # Reset the model's internal state
            self.current_hidden_state = self.model.init_hidden()

        # If current data does not need prediction
        if not data_point.need_prediction:
            return None

        # --- Prepare input data ---
        # Reshape (N,) to (1, N) for the scaler
        current_state = data_point.state.reshape(1, -1)
        # Scale the data using the loaded scaler
        scaled_state = self.scaler.transform(current_state)
        # Convert to a tensor -> (batch_size, seq_len, features)
        state_tensor = torch.tensor(scaled_state, dtype=torch.float32).unsqueeze(0)

        # Run the model for inference
        with torch.no_grad(): 
            prediction_tensor, next_hidden_state = self.model(
                state_tensor, 
                self.current_hidden_state
            )
        
        # Update the model's state for the next call 
        self.current_hidden_state = next_hidden_state
        
        # --- Unscaling the predicted state vector --- 
        # Squeeze and convert to numpy
        scaled_prediction = prediction_tensor.squeeze(0).numpy()
        
        # Use the scaler's inverse_transform
        unscaled_prediction = self.scaler.inverse_transform(scaled_prediction)
        
        # Return the final result as a flat (N,) numpy array
        return unscaled_prediction.flatten()
    
if __name__ == "__main__":
    # Check existence of test file
    test_file = f"{CURRENT_DIR}/../datasets/train.parquet"

    # Load data into scorer
    scorer = ScorerStepByStep(test_file)

    # Create and test our model
    model = PredictionModel()

    print("Testing LSTM model...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")

    # Evaluate our solution
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")

    print(f"\nTotal features: {len(scorer.features)}")

    print("\n" + "=" * 60)
    print("Try submitting an archive with solution.py file")
    print("to test the solution submission mechanism!")
    print("=" * 60)
