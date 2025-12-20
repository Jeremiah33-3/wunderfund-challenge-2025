import numpy as np
import os
import pickle
import sys
import torch
from collections import deque
from model import MyMambaModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add project root folder to path for importing utils
sys.path.append(f"{CURRENT_DIR}/../")

try:
    from utils import DataPoint, ScorerStepByStep
except ModuleNotFoundError as e:
    print("ModuleNotFoundError occurred:", e)
    print("sys.path was:", sys.path)


class PredictionModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Load Checkpoint ---
        try:
            checkpoint = torch.load("model_mamba.pth", map_location=self.device)
        except FileNotFoundError:
            # Fallback for local testing if file isn't found
            print("Warning: model_mamba.pth not found.")
            return

        config = checkpoint['config']
        
        # --- Initialize Model (Standard Float32 is fine for Pure PyTorch) ---
        self.n_features = config['n_features']
        self.model = MyMambaModel(
            input_size=self.n_features,
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            output_size=self.n_features
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        # --- Load Scaler ---
        with open("scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
            
        # --- Sliding Window Buffer ---
        self.context_len = 200 
        self.buffer = deque(maxlen=self.context_len)
        self.current_seq_id = -1

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        
        # --- Reset on new sequence ---
        if data_point.seq_ix != self.current_seq_id:
            self.current_seq_id = data_point.seq_ix
            self.buffer.clear()

        if not data_point.need_prediction:
            return None
            
        # --- Pre-process ---
        # Reshape for scaler: (1, N)
        raw_state = data_point.state.reshape(1, -1)
        scaled_state = self.scaler.transform(raw_state)
        
        # Add flat array to buffer
        self.buffer.append(scaled_state.flatten())
            
        # --- Prepare Input ---
        # Convert buffer to array -> Tensor
        # Shape: (1, Seq_Len, N)
        input_seq = np.array(self.buffer)
        input_tensor = torch.tensor(input_seq, dtype=torch.float32) \
                            .unsqueeze(0) \
                            .to(self.device)
        
        # --- Inference ---
        with torch.no_grad():
            output_seq = self.model(input_tensor)
            last_pred = output_seq[:, -1, :] # Get last step
            
        # --- Post-process ---
        pred_unscaled = self.scaler.inverse_transform(last_pred.cpu().numpy())
        
        return pred_unscaled.flatten()

# --- Main logic ---
if __name__ == "__main__":
    # Ensure we don't test on the exact same file we trained on (ideally)
    test_file = "datasets/train.parquet" 
        
    if os.path.exists(test_file):
        scorer = ScorerStepByStep(test_file)
        model = PredictionModel()
        print("Running local scoring...")
        results = scorer.score(model)
        print(f"Mean R2: {results['mean_r2']:.5f}")