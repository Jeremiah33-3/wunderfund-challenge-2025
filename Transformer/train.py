import math
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pyarrow.parquet as pq 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create a long matrix of positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # We slice the pe matrix to the current sequence length
        return x + self.pe[:, :x.size(1), :]

class MyTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, output_size=None, dropout=0.1):
        super().__init__()
        if output_size is None:
            output_size = input_size
            
        self.d_model = d_model
        
        # Embed input features to model dimension
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder (acting as Decoder with causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_size)

    def _generate_causal_mask(self, size):
        """
        Generates a mask like this:
        [0, -inf, -inf]
        [0, 0,    -inf]
        [0, 0,    0   ]
        This ensures step T can only attend to 0...T
        """
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src):
        # src shape: (batch, seq_len, input_size)
        
        # Create causal mask dynamically based on current sequence length
        seq_len = src.size(1)
        mask = self._generate_causal_mask(seq_len).to(src.device)
        
        # Project and add position info
        x = self.input_proj(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        # We use is_causal=True for optimization if available (PyTorch 2.0+), 
        # but explicit mask is safer for compatibility.
        x = self.transformer(x, mask=mask)
        
        # Project to output
        output = self.output_proj(x)
        return output
    
class TransformerDataset(Dataset):
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
        
        # Transformer Input: Steps 1 to 999
        # Transformer Target: Steps 2 to 1000
        X = features[:-1]
        Y = features[1:] - features[:-1] # try modifying target as change
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

## -- Main Logic --

if __name__ == "__main__":
    print("Starting Training")
    
    # --- Hyperparameters ---
    N_FEATURES = 32     # Given
    D_MODEL = 64        # Internal dimension
    N_HEAD = 4          # Attention heads
    NUM_LAYERS = 2      # Depth
    LR = 0.001
    BATCH_SIZE = 16
    EPOCHS = 30
    WARM_UP = 100  # Given

    ## --- Hardware Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Data ---
    data = pq.read_table("datasets/train.parquet").to_pandas()
    FEATURE_COLUMNS = [f'{i}' for i in range(N_FEATURES)]
    all_ids = data['seq_ix'].unique()
    train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

    # Fit scaler on training Data 
    scaler = StandardScaler()
    train_data = data[data['seq_ix'].isin(train_ids)]
    scaler.fit(train_data[FEATURE_COLUMNS].values)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Training Data
    train_ds = TransformerDataset(train_data, train_ids, scaler)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    # Validation Data
    val_data = data[data['seq_ix'].isin(val_ids)]
    val_ds = TransformerDataset(val_data, val_ids, scaler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

    # --- Model Setup ---
    model = MyTransformer(
        input_size=N_FEATURES,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        output_size=N_FEATURES
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # --- Training Loop ---
    print("Starting Transformer training...")
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        total_train_loss = 0
        
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass (Masking is handled inside the model)
            preds = model(X_batch)
            
            # Loss calculation
            # preds shape: (batch, 999, features)
            loss = loss_fn(preds[:, WARM_UP-1:, :], Y_batch[:, WARM_UP-1:, :])
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation ---
        print("Validating...")
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

                preds = model(X_batch)

                val_loss = loss_fn(preds[:, WARM_UP-1:, :], Y_batch[:, WARM_UP-1:, :])
                total_val_loss += val_loss.item()
            
        avg_val_loss = total_val_loss / len(val_loader)
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    torch.save(model.state_dict(), "model_transformer.pth")
