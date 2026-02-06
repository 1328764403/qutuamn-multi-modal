"""
Tensor Fusion Network (TFN)
通过张量融合层建模模态间交互
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TFN(nn.Module):
    """
    Tensor Fusion Network for multimodal fusion
    
    Paper: "Tensor Fusion Network for Multimodal Sentiment Analysis"
    """
    
    def __init__(self, input_dims, hidden_dim, output_dim, dropout=0.1):
        """
        Args:
            input_dims: list of input dimensions for each modality [dim1, dim2, dim3, ...]
            hidden_dim: hidden dimension for fusion
            output_dim: output dimension
            dropout: dropout rate
        """
        super(TFN, self).__init__()
        
        self.num_modalities = len(input_dims)
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        
        # Project each modality to hidden dimension
        self.proj_layers = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        
        # Tensor fusion layer
        # For 3 modalities: [1, m1, m2, m3, m1*m2, m1*m3, m2*m3, m1*m2*m3]
        fusion_dim = 2 ** self.num_modalities
        self.fusion_layer = nn.Linear(fusion_dim * hidden_dim, hidden_dim)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, *modalities):
        """
        Args:
            modalities: tuple of tensors, each of shape (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, output_dim)
        """
        batch_size = modalities[0].size(0)
        
        # Project each modality
        projected = []
        for i, mod in enumerate(modalities):
            # Average pooling over sequence length if exists
            if len(mod.shape) == 3:
                mod = mod.mean(dim=1)  # (batch_size, input_dim)
            proj = self.proj_layers[i](mod)  # (batch_size, hidden_dim)
            projected.append(proj)
        
        # Build tensor fusion representation
        # Start with ones for the bias term
        fusion_parts = [torch.ones(batch_size, self.hidden_dim, device=modalities[0].device)]
        
        # Add individual modalities
        fusion_parts.extend(projected)
        
        # Add pairwise interactions
        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                interaction = projected[i] * projected[j]
                fusion_parts.append(interaction)
        
        # Add three-way interactions (if 3+ modalities)
        if self.num_modalities >= 3:
            for i in range(self.num_modalities):
                for j in range(i + 1, self.num_modalities):
                    for k in range(j + 1, self.num_modalities):
                        interaction = projected[i] * projected[j] * projected[k]
                        fusion_parts.append(interaction)
        
        # Concatenate all fusion parts
        fusion = torch.cat(fusion_parts, dim=1)  # (batch_size, fusion_dim * hidden_dim)
        
        # Fusion layer
        fused = self.fusion_layer(fusion)
        fused = F.relu(fused)
        
        # Output
        output = self.output_layer(fused)
        
        return output







