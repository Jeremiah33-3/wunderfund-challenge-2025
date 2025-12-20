import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pyarrow.parquet as pq 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# -- 0. My LSTM Model -- 
class MyLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Defines the layers of the model.
        
        Args:
            input_size (int): The number of features (N)
            hidden_size (int): The number of units in the LSTM's hidden state
            num_layers (int): The number of stacked LSTM layers
            output_size (int): The number of features to predict (also N)
        """
        super(MyLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Hardware efficiency: check for GPU/CPU environments
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # The LSTM layer
        # batch_first=True means the input tensor shape is (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Takes the LSTM's output (hidden_size) and maps it to the prediction (output_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        """
        Runs model(input) forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            hidden_state (tuple): A tuple (h_n, c_n) containing the 
                                  hidden and cell state from the previous step.
        
        Returns:
            A tuple (output, (h_n, c_n)):
            - output: The prediction tensor of shape (batch_size, seq_len, output_size)
            - (h_n, c_n): The new hidden and cell state (num_layers, batch_size, hidden_size)
        """
        
        # x's shape is (batch_size, seq_len, input_size)
        # hidden_state's shape is ( (num_layers, batch_size, hidden_size), (num_layers, batch_size, hidden_size)
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        
        # Pass LSTM's output (the "hidden states" at each time step)
        # through the final linear layer.
        output = self.fc(lstm_out)
        
        return output, new_hidden_state

    def init_hidden(self, batch_size=1):
        """
        Generates a zero-initialized hidden state.
        To be called at the start of every new sequence.
        """
        # The shape is (num_layers, batch_size, hidden_size)
        # One for the hidden state (h_n) and one for the cell state (c_n)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        return (h0, c0)

## PyTorch Dataset
class SequenceDataset(Dataset):
    def __init__(self, data, seq_ids, feature_cols, scaler):
        self.data = data.set_index('seq_ix')
        self.seq_ids = seq_ids
        self.feature_cols = feature_cols
        self.scaler = scaler

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        seq_df = self.data.loc[seq_id].sort_values(by='step_in_seq')
        
        # Scale features
        features = self.scaler.transform(seq_df[self.feature_cols].values)
        
        # Create inputs (X) and targets (Y)
        # X: step t (1-999) with y: step t+1 (2-1000)
        X = features[:-1]
        Y_delta = features[1:]
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y_delta, dtype=torch.float32)

## -- Main Logic -- 
if __name__ == "__main__":
    ## --- Load Data ---
    data = pq.read_table("datasets/train.parquet").to_pandas()

    # N coloumns
    N = data.shape[1] - 3
    FEATURE_COLUMNS = [f"{i}" for i in range(N)] 

    ## --- Create Train/Validation Split by Sequence ID ---
    all_seq_ids = data['seq_ix'].unique()
    train_ids, val_ids = train_test_split(all_seq_ids, test_size=0.2, random_state=42)

    train_data = data[data['seq_ix'].isin(train_ids)]
    val_data = data[data['seq_ix'].isin(val_ids)]

    ## --- Fit and Save the Scaler ---
    # Fit only the training data at this stage
    scaler = StandardScaler()
    scaler.fit(train_data[FEATURE_COLUMNS].values)

    # Save the fitted scaler for inference
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    ## --- Define and Train the Model ---

    # Best practise: determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters (possible for hyperparameter tuning)
    BATCH_SIZE = 16
    HIDDEN_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 47
    NUM_LAYERS = 2
    DROPOUT = 0.0
    WARM_UP = 100 # Given

    # Model, optimiser and loss function
    model = MyLSTM(
        input_size=N, 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS,
        output_size=N,
        dropout=DROPOUT
    ).to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    # Load dataset
    train_dataset = SequenceDataset(train_data, train_ids, FEATURE_COLUMNS, scaler)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # Validation set
    val_dataset = SequenceDataset(val_data, val_ids, FEATURE_COLUMNS, scaler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ## --- Training Loop ---
    print("Training...")
    for epoch in tqdm(range(NUM_EPOCHS)):
        model.train()
        total_train_loss = 0
        # Train by batch
        for X_batch, Y_batch in train_loader:
            # X_batch shape: (BATCH_SIZE, 999, N_FEATURES)
            # Y_batch shape: (BATCH_SIZE, 999, N_FEATURES)
            
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            # Initialize hidden state for the batch
            hidden = model.init_hidden(X_batch.size(0))
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            Y_pred, hidden = model(X_batch, hidden)
            
            # Calculate loss
            # Only score predictions from step 100 onwards
            loss = loss_fn(Y_pred[:, WARM_UP-1:, :], Y_batch[:, WARM_UP-1:, :])
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation step
        print("Validating...")
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

                hidden = model.init_hidden(X_batch.size(0))

                Y_pred, hidden = model(X_batch, hidden)

                val_loss = loss_fn(Y_pred[:, WARM_UP-1:, :], Y_batch[:, WARM_UP-1:, :])
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    ## --- Save the Final Trained Model ---
    # Save only the model weights (state dictionary)
    torch.save(model.state_dict(), "model.pth")

    print("Training complete. model.pth and scaler.pkl are saved.")
