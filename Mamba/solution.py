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
        
        # 1. Load Checkpoint
        checkpoint = torch.load("model_mamba.pth", map_location=self.device)
        config = checkpoint['config']
        
        # 2. Initialize Model
        self.n_features = config['n_features']
        self.model = MyMambaModel(
            input_size=self.n_features,
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            output_size=self.n_features
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.half()
        self.model.eval()
        
        # 3. Load Scaler
        with open("scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
            
        # 4. Sliding Window Buffer
        # Note: can experiment a larger context if needed
        self.context_len = 200 
        self.buffer = deque(maxlen=self.context_len)
        self.current_seq_id = -1

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # Reset on new sequence
        if data_point.seq_ix != self.current_seq_id:
            self.current_seq_id = data_point.seq_ix
            self.buffer.clear()
        
        if not data_point.need_prediction:
            return None
            
        # Scale and buffer
        # (1, N)
        raw_state = data_point.state.reshape(1, -1)
        scaled_state = self.scaler.transform(raw_state)
        
        # Store flat array in deque
        self.buffer.append(scaled_state.flatten())

        raw_seq = np.array(self.buffer)
        current_len = len(raw_seq)
        
        # Calculate how much padding we need
        # e.g. if len=10, nearest multiple of 8 is 16. Padding=6.
        remainder = current_len % 8
        if remainder != 0:
            padding_needed = 8 - remainder
            pads = np.zeros((padding_needed, self.n_features))
            # Stack pads BEFORE the data (Left Padding)
            input_seq = np.vstack([pads, raw_seq])
        else:
            input_seq = raw_seq
            
        # Prepare Input: (1, Seq_Len, N)
        input_tensor = torch.tensor(input_seq) \
            .unsqueeze(0) \
            .contiguous() \
            .to(device=self.device, dtype=torch.float16)
        
        # Inference
        with torch.no_grad():
            # Mamba processes the sequence
            output_seq = self.model(input_tensor)
            
            # Take the last step prediction
            last_pred = output_seq[:, -1, :]
            
        # Unscale
        pred_unscaled = self.scaler.inverse_transform(last_pred.float().cpu().numpy())
        
        return pred_unscaled.flatten()
    
if __name__ == "__main__":
    # Check existence of test file
    test_file = f"{CURRENT_DIR}/../datasets/train.parquet"

    # Load data into scorer
    scorer = ScorerStepByStep(test_file)

    # Create and test our model
    model = PredictionModel()

    print("Testing Mamba model...")
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
