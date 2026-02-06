"""
Memory Fusion Network (MFN)
引入记忆机制捕捉跨模态的长期依赖
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryCell(nn.Module):
    """Memory cell for storing cross-modal information"""
    
    def __init__(self, hidden_dim, memory_size):
        super(MemoryCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        # Memory matrix
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_dim))
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, memory_size),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, query):
        """
        Args:
            query: (batch_size, hidden_dim)
        Returns:
            output: (batch_size, hidden_dim)
        """
        # Compute attention weights
        attn_weights = self.attention(query)  # (batch_size, memory_size)
        
        # Weighted memory read
        output = torch.matmul(attn_weights, self.memory)  # (batch_size, hidden_dim)
        
        return output


class MFN(nn.Module):
    """
    Memory Fusion Network
    
    Paper: "Memory Fusion Network for Multi-view Sequential Learning"
    """
    
    def __init__(self, input_dims, hidden_dim, output_dim, memory_size=8, num_layers=2, dropout=0.1):
        """
        Args:
            input_dims: list of input dimensions for each modality
            hidden_dim: hidden dimension
            output_dim: output dimension
            memory_size: size of memory bank
            num_layers: number of LSTM layers
            dropout: dropout rate
        """
        super(MFN, self).__init__()
        
        self.num_modalities = len(input_dims)
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        # Encoders for each modality
        self.encoders = nn.ModuleList([
            nn.LSTM(dim, hidden_dim, num_layers=num_layers, 
                   batch_first=True, dropout=dropout if num_layers > 1 else 0)
            for dim in input_dims
        ])
        
        # Memory cells for cross-modal fusion
        self.memory_cells = nn.ModuleList([
            MemoryCell(hidden_dim, memory_size) for _ in range(self.num_modalities)
        ])
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, *modalities):
        """
        Args:
            modalities: tuple of tensors, each (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, output_dim)
        """
        batch_size = modalities[0].size(0)
        
        # Encode each modality
        encoded = []
        for i, mod in enumerate(modalities):
            # LSTM encoding
            lstm_out, (h_n, c_n) = self.encoders[i](mod)
            # Use last hidden state
            encoded_i = h_n[-1]  # (batch_size, hidden_dim)
            encoded.append(encoded_i)
        
        # Memory-based fusion
        memory_outputs = []
        for i, enc in enumerate(encoded):
            # Read from memory
            mem_out = self.memory_cells[i](enc)
            # Combine with encoded representation
            combined = enc + mem_out
            memory_outputs.append(combined)
        
        # Cross-modal attention
        # Stack all modalities
        stacked = torch.stack(memory_outputs, dim=1)  # (batch_size, num_modalities, hidden_dim)
        
        # Self-attention across modalities
        attn_out, _ = self.cross_modal_attention(stacked, stacked, stacked)
        # Average pooling
        attn_out = attn_out.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Alternative: concatenate and fuse
        concat = torch.cat(memory_outputs, dim=1)  # (batch_size, num_modalities * hidden_dim)
        fused = self.fusion_layer(concat)
        
        # Combine attention and fusion
        final = (attn_out + fused) / 2
        
        # Output
        output = self.output_layer(final)
        
        return output







