import subprocess
import sys
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba2
    print(f"Torch CUDA: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("Mamba import successful!")
except ImportError:
    print("ERROR: mamba_ssm not installed. Run: pip install mamba-ssm")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mamba-ssm"])
    # Mamba2 = None # Placeholder to prevent immediate crash during file scan

class MyMambaModel(nn.Module):
    def __init__(self, input_size, d_model=64, n_layers=2, output_size=None, dropout=0.1):
        super().__init__()
        if output_size is None:
            output_size = input_size
            
        self.d_model = d_model
        
        # 1. Project Input to Model Dimension
        self.input_proj = nn.Linear(input_size, d_model)
        
        # 2. Mamba Layers
        # We stack Mamba2 blocks. 
        self.layers = nn.ModuleList([
            Mamba2(
                d_model=d_model, 
                d_state=64,   # SSM state expansion
                d_conv=4,     # Local convolution width
                expand=2      # Block expansion factor
            ) for _ in range(n_layers)
        ])
        
        # 3. Normalization (RMSNorm is standard for Mamba)
        self.norm_f = nn.RMSNorm(d_model)
        
        # 4. Output Projection
        self.output_proj = nn.Linear(d_model, output_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # Safety check for contiguous memory (Critical for Mamba Kernels)
        if not x.is_contiguous():
            x = x.contiguous()

        x = self.input_proj(x)
        
        # Pass through Mamba layers
        # Mamba blocks usually have residual connections built-in or applied manually
        for layer in self.layers:
            x_out = layer(x) 
            x = x + x_out # Residual connection
            x = self.dropout(x)
            
        x = self.norm_f(x)
        output = self.output_proj(x)
        
        return output