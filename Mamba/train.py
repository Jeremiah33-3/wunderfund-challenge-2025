import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import MyMambaModel
from tqdm.auto import tqdm

class MambaDataset(Dataset):
    def __init__(self, data, seq_ids, scaler):
        self.data = data.set_index('seq_ix')
        self.seq_ids = seq_ids
        self.scaler = scaler
        
    def __len__(self): 
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        seq_df = self.data.loc[seq_id].sort_values('step_in_seq')
        features = self.scaler.transform(seq_df[FEATURE_COLUMNS].values)
        
        # X is 1-999 steps
        # Y is 2-1000 steps 
        X = features[:-1]
        Y = features[1:]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

if __name__ == "__main__":
    #  --- Configuration ---
    N_FEATURES = 32    # Change to match your actual dataset (10 for dummy, 32 for real)
    D_MODEL = 64
    N_LAYERS = 2
    LR = 1e-3
    BATCH_SIZE = 24
    EPOCHS = 30
    WARMUP_STEPS = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- Data Loading ---
    data = pd.read_parquet("datasets/train.parquet")
    FEATURE_COLUMNS = [f'{i}' for i in range(N_FEATURES)] # Ensure names match data
    all_ids = data['seq_ix'].unique()
    train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    train_data = data[data['seq_ix'].isin(train_ids)]
    scaler.fit(train_data[FEATURE_COLUMNS].values)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    train_loader = DataLoader(
        MambaDataset(train_data, train_ids, scaler), 
        batch_size=BATCH_SIZE,
        shuffle=True
        )
    
    val_data = data[data['seq_ix'].isin(val_ids)]
    
    # --- Model Setup ---
    model = MyMambaModel(
        input_size=N_FEATURES,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        output_size=N_FEATURES
    ).to(device)

    # Use standard parameters
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # --- Training ---
    print("Starting Mamba-2 training...")
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        total_loss = 0
        
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(X)
            
            loss = loss_fn(preds[:, WARMUP_STEPS-1:, :], Y[:, WARMUP_STEPS-1:, :])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.6f}")

    # Save config with weights to avoid shape errors later
    checkpoint = {
        'state_dict': model.state_dict(),
        'config': {'n_features': N_FEATURES, 'd_model': D_MODEL, 'n_layers': N_LAYERS}
    }
    torch.save(checkpoint, "model_mamba.pth")
    print("Training finished.")
