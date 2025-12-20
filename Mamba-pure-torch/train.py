import pyarrow.parquet as pq
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from model import MyMambaModel

# ==========================================
# 🔧 CONFIGURATION & HYPERPARAMETERS
# ==========================================
FINAL_SUBMISSION_MODE = False  # <--- Set TRUE before generating your final zip
                               # <--- Set FALSE to see validation scores

# --- Hyperparameters ---
N_FEATURES = 32    # Matches the real dataset
D_MODEL = 64       # Safe size (multiple of 8)
N_LAYERS = 2
LR = 1e-3
BATCH_SIZE = 24
EPOCHS = 30
WARMUP_STEPS = 100

class MambaDataset(Dataset):
    def __init__(self, data, seq_ids, feature_cols, scaler):
        self.data = data.set_index('seq_ix')
        self.seq_ids = seq_ids
        self.feature_cols = feature_cols
        self.scaler = scaler
        
    def __len__(self): 
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        seq_df = self.data.loc[seq_id].sort_values('step_in_seq')
        
        # Transform using values to avoid warning
        features = self.scaler.transform(seq_df[self.feature_cols].values)
        
        # Input: 1-999, Target: 2-1000, 0-indexed
        X = features[:-1]
        Y = features[1:]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

if __name__ == "__main__": 
    # Hardware configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on: {device}")

    # --- Load Data ---
    print("Loading data...")
    try:
        data = pq.read_table("datasets/train.parquet").to_pandas()
    except FileNotFoundError:
        print("Error: datasets/train.parquet not found.")
        exit()

    # 2. Prepare Features
    # Assuming columns are named '0', '1'... or 'feature_0'...
    # Auto-detect for safety
    all_cols = data.columns.tolist()
    # Filter for numeric feature columns (exclude seq_ix, step, etc)
    exclude = ['seq_ix', 'step_in_seq', 'need_prediction']
    FEATURE_COLUMNS = [c for c in all_cols if c not in exclude]
    
    # Update N_FEATURES based on actual data
    N_FEATURES = len(FEATURE_COLUMNS)
    print(f"Detected {N_FEATURES} features.")

    # 3. Split & Scale
    all_ids = data['seq_ix'].unique()
    if FINAL_SUBMISSION_MODE:
        print("⚠️ SUBMISSION MODE: Training on 100% of data. Validation disabled.")
        train_ids = all_ids
        val_ids = []
    else:
        print("🧪 EXPERIMENT MODE: Splitting 80/20 for validation.")
        train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

    # --- 4. Fit Scaler ---
    print("Fitting Scaler...")
    scaler = StandardScaler()
    # Only fit on training data to avoid leakage
    train_data = data[data['seq_ix'].isin(train_ids)]
    scaler.fit(train_data[FEATURE_COLUMNS].values)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # 4. Setup Loader
    train_ds = MambaDataset(train_data, train_ids, FEATURE_COLUMNS, scaler)
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        persistent_workers=True
    )
    
    # If in validation mode
    val_loader = None
    if len(val_ids) > 0:
        val_data = data[data['seq_ix'].isin(val_ids)]
        val_ds = MambaDataset(val_data, val_ids, FEATURE_COLUMNS, scaler)
        val_loader = DataLoader(
            val_ds, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=2,
            persistent_workers=True
        )
    # 5. Setup Model
    model = MyMambaModel(
        input_size=N_FEATURES,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        output_size=N_FEATURES
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # 6. Training Loop
    print("Starting Training...")
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        total_train_loss = 0
        
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            
            preds = model(X)
            
            # Calculate loss on steps after warmup
            # preds shape: (Batch, 999, Features)
            loss = loss_fn(preds[:, WARMUP_STEPS-1:, :], Y[:, WARMUP_STEPS-1:, :])
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATE (If exists) ---
        val_msg = ""
        if val_loader:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for X, Y in val_loader:
                    X, Y = X.to(device), Y.to(device)
                    preds = model(X)
                    loss = loss_fn(preds[:, WARMUP_STEPS-1:, :], Y[:, WARMUP_STEPS-1:, :])
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            val_msg = f"| Val Loss: {avg_val_loss:.6f}"
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} {val_msg}")

    # 7. Save Artifacts
    checkpoint = {
        'state_dict': model.state_dict(),
        'config': {
            'n_features': N_FEATURES, 
            'd_model': D_MODEL, 
            'n_layers': N_LAYERS
        }
    }
    torch.save(checkpoint, "model_mamba.pth")
    print("Training complete. Saved model_mamba.pth and scaler.pkl")
