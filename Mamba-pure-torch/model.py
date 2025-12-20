import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@torch.jit.script
def mamba_scan_loop(seq_len: int, batch: int, d_inner: int, 
                    dt: torch.Tensor, A: torch.Tensor, 
                    B: torch.Tensor, C: torch.Tensor, 
                    x_conv: torch.Tensor, device: torch.device):
    """
    This function is compiled by TorchScript to run at C++ speeds
    Avoids the slow Python interpreter overhead in the loop.
    """
    y_ssm = torch.zeros(batch, seq_len, d_inner, device=device)
    h = torch.zeros(batch, d_inner, 16, device=device) # Hidden state
    
    for t in range(seq_len):
        # We must be explicit with tensor shapes for JIT
        dt_step = dt[:, t, :].unsqueeze(-1)
        
        # Calculate A and B terms
        dA = torch.exp(dt_step * A)
        dB = dt_step * B[:, t, :].unsqueeze(1)
        
        # Recurrence: h_t = dA * h_{t-1} + dB * x_t
        x_step = x_conv[:, t, :].unsqueeze(-1)
        h = dA * h + dB * x_step
        
        # Output: y_t = C * h_t
        C_step = C[:, t, :].unsqueeze(1)
        y_out = torch.sum(h * C_step, dim=-1)
        
        # Store result (JIT doesn't like list.append, so we assign to tensor)
        y_ssm[:, t, :] = y_out
        
    return y_ssm

class MyMambaModel(nn.Module):
    def __init__(self, input_size, d_model=64, n_layers=2, output_size=None, dropout=0.1):
        """
        A Mamba-style model built with pure PyTorch operations.
        Compatible with Windows/WSL without custom CUDA kernels.
        """
        super().__init__()
        if output_size is None:
            output_size = input_size
            
        self.d_model = d_model
        
        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Stack Mamba Blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=16, d_conv=4, expand=2) 
            for _ in range(n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Size)
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
            
        x = self.norm_f(x)
        output = self.output_proj(x)
        return output

class MambaBlock(nn.Module):
    """
    Pure PyTorch implementation of the Mamba block.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Standard Conv1d
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
        )
        
        self.activation = nn.SiLU()
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D Initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        # 1. Input Projection
        x_and_z = self.in_proj(x)
        x_in, z = x_and_z.chunk(2, dim=-1)
        
        # 2. 1D Convolution
        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len] # Crop padding
        x_conv = self.activation(x_conv).transpose(1, 2)
        
        # 3. State Space Model (JIT function)
        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, 16, 16], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        
        A = -torch.exp(self.A_log.float()) # (D_inner, d_state)
        
        y = mamba_scan_loop(
            seq_len, batch, self.d_inner,
            dt, A, B, C, x_conv, x.device
        )
            
        # 4. Gating
        output = y * self.activation(z)
        return self.out_proj(output)
