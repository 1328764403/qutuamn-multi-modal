"""
Multimodal Transformer (MulT)
基于多头跨模态注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return x


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
        Returns:
            output: (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        residual = query
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        
        return output


class MulT(nn.Module):
    """
    Multimodal Transformer
    
    Paper: "Multimodal Transformer for Unaligned Multimodal Language Sequences"
    """
    
    def __init__(self, input_dims, d_model, output_dim, num_heads=8, num_layers=4, dropout=0.1):
        """
        Args:
            input_dims: list of input dimensions for each modality
            d_model: model dimension (should be divisible by num_heads)
            output_dim: output dimension
            num_heads: number of attention heads
            num_layers: number of transformer layers
            dropout: dropout rate
        """
        super(MulT, self).__init__()
        
        self.num_modalities = len(input_dims)
        self.input_dims = input_dims
        self.d_model = d_model
        
        # Input projections
        self.input_projs = nn.ModuleList([
            nn.Linear(dim, d_model) for dim in input_dims
        ])
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict()
            for i in range(self.num_modalities):
                layer[f'mod_{i}'] = CrossModalAttention(d_model, num_heads, dropout)
            self.cross_modal_layers.append(layer)
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model)
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * self.num_modalities, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )
        
    def forward(self, *modalities):
        """
        Args:
            modalities: tuple of tensors, each (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, output_dim)
        """
        batch_size = modalities[0].size(0)
        
        # Project inputs
        projected = []
        for i, mod in enumerate(modalities):
            proj = self.input_projs[i](mod)  # (batch_size, seq_len, d_model)
            # Add positional encoding
            proj = proj.transpose(0, 1)  # (seq_len, batch_size, d_model)
            proj = self.pos_encoding(proj)
            proj = proj.transpose(0, 1)  # (batch_size, seq_len, d_model)
            projected.append(proj)
        
        # Cross-modal attention
        for layer_idx, layer in enumerate(self.cross_modal_layers):
            new_projected = []
            for i, mod_i in enumerate(projected):
                # Attend to all other modalities
                attended = mod_i
                for j, mod_j in enumerate(projected):
                    if i != j:
                        attended = layer[f'mod_{i}'](attended, mod_j, mod_j)
                
                # Feed-forward
                residual = attended
                attended = self.ffns[layer_idx](attended)
                attended = attended + residual
                
                new_projected.append(attended)
            projected = new_projected
        
        # Pooling: take mean over sequence length
        pooled = [mod.mean(dim=1) for mod in projected]  # List of (batch_size, d_model)
        
        # Concatenate and output
        concat = torch.cat(pooled, dim=1)  # (batch_size, num_modalities * d_model)
        output = self.output_proj(concat)
        
        return output







