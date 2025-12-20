import numpy as np
import os
import pickle
import sys
import torch
from collections import deque 
from train import MyTransformer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add project root folder to path for importing utils
sys.path.append(f"{CURRENT_DIR}/../")

try:
    from utils import DataPoint, ScorerStepByStep
except ModuleNotFoundError as e:
    print("ModuleNotFoundError occurred:", e)
    print("sys.path was:", sys.path)

## --- Hyperparameters ---
N_FEATURES = 32 # Given
D_MODEL = 64
N_HEAD = 4
NUM_LAYERS = 2
CONTEXT_LENGTH = 200  

class PredictionModel:
    def __init__(self):
        ## Set Hardware
        self.device = torch.device("cpu")
        
        # Load Model
        self.model = MyTransformer(
            input_size=N_FEATURES,
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_layers=NUM_LAYERS,
            output_size=N_FEATURES
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(torch.load("model_transformer.pth", map_location=self.device))
        self.model.eval()
        
        # Load Scaler
        with open("scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
            
        # Initialize State
        self.current_seq_id = -1
        self.buffer = deque(maxlen=CONTEXT_LENGTH)

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # --- Reset on New Sequence ---
        if data_point.seq_ix != self.current_seq_id:
            self.current_seq_id = data_point.seq_ix
            self.buffer.clear()

        # --- Check if need prediction --- 
        if not data_point.need_prediction:
            return None
            
        # --- Pre-process Input ---
        # Scale the raw input
        # Reshape to (1, N) for scaler
        raw_state = data_point.state.reshape(1, -1)
        scaled_state = self.scaler.transform(raw_state) # Returns (1, N)
        
        # Add to History Buffer
        # Flatten it to (N,) before appending so the deque is a list of 1D arrays
        self.buffer.append(scaled_state.flatten())
            
        # --- Prepare Transformer Input ---
        # Stack buffer into a single array: (Seq_Len, N_features)
        seq_array = np.array(self.buffer)
        
        # Convert to Tensor and add Batch Dimension: (1, Seq_Len, N_features)
        input_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # --- Inference ---
        with torch.no_grad():
            # The model returns predictions for the whole sequence
            # output shape: (1, Seq_Len, N_features)
            output_sequence = self.model(input_tensor)
            
            # last_step_pred = output_sequence[:, -1, :] # Shape: (1, N_features)
            # Add change to the current state to get final prediction
            current_val = data_point.state # (Unscaled)
            last_step_pred = current_val + self.scaler.inverse_transform(output_sequence[:, -1, :]) # is a numpy
        
        # Un-scale
        pred_unscaled = self.scaler.inverse_transform(last_step_pred)
        
        return pred_unscaled.flatten()

if __name__ == "__main__":
    # Check existence of test file
    test_file = f"{CURRENT_DIR}/../datasets/train.parquet"

    # Load data into scorer
    scorer = ScorerStepByStep(test_file)

    # Create and test our model
    model = PredictionModel()
    # test stat: ~0.31

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
