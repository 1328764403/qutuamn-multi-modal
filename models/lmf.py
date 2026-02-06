"""
Low-rank Multimodal Fusion (LMF)
在 TFN 基础上引入低秩分解，降低计算复杂度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LMF(nn.Module):
    """
    Low-rank Multimodal Fusion
    
    Paper: "Efficient Low-rank Multimodal Fusion with Modality-Specific Factors"
    """
    
    def __init__(self, input_dims, hidden_dim, output_dim, rank=4, dropout=0.1):
        """
        Args:
            input_dims: list of input dimensions for each modality
            hidden_dim: hidden dimension for fusion
            output_dim: output dimension
            rank: rank for low-rank decomposition
            dropout: dropout rate
        """
        super(LMF, self).__init__()
        
        self.num_modalities = len(input_dims)
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.rank = rank
        
        # Project each modality to hidden dimension
        self.proj_layers = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        
        # Low-rank factors for each modality
        self.factors = nn.ModuleList([
            nn.Linear(hidden_dim, rank, bias=False) for _ in range(self.num_modalities)
        ])
        
        # Fusion weights
        fusion_dim = 2 ** self.num_modalities
        self.fusion_weights = nn.Parameter(torch.randn(rank, fusion_dim, hidden_dim))
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, *modalities):
        """
        Args:
            modalities: tuple of tensors
        Returns:
            output: (batch_size, output_dim)
        """
        batch_size = modalities[0].size(0)
        device = modalities[0].device
        
        # Project each modality
        projected = []
        for i, mod in enumerate(modalities):
            if len(mod.shape) == 3:
                mod = mod.mean(dim=1)
            proj = self.proj_layers[i](mod)
            projected.append(proj)
        
        # Compute low-rank factors
        factors = [self.factors[i](projected[i]) for i in range(self.num_modalities)]
        
        # Build fusion representation using low-rank approximation
        # Start with bias and individual modalities
        fusion_parts = [torch.ones(batch_size, self.hidden_dim, device=device)]
        fusion_parts.extend(projected)
        
        # Pairwise interactions using low-rank approximation
        interaction_idx = 1 + self.num_modalities
        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                # Low-rank approximation: (factors[i] * factors[j]) @ weight
                # Element-wise product of factors
                factor_product = factors[i] * factors[j]  # (batch_size, rank)
                
                # Use fusion weights to project to hidden_dim
                if interaction_idx < self.fusion_weights.size(1):
                    weight = self.fusion_weights[:, interaction_idx, :]  # (rank, hidden_dim)
                    interaction = torch.matmul(factor_product, weight)  # (batch_size, hidden_dim)
                    fusion_parts.append(interaction)
                interaction_idx += 1
        
        # Concatenate all fusion parts
        fusion = torch.cat(fusion_parts, dim=1)  # (batch_size, total_dim)
        
        # Map to hidden_dim using a learned projection
        if fusion.size(1) > self.hidden_dim:
            # Use first hidden_dim dimensions + sum of rest
            fusion = fusion[:, :self.hidden_dim] + fusion[:, self.hidden_dim:].sum(dim=1, keepdim=True).expand(-1, self.hidden_dim)
        elif fusion.size(1) < self.hidden_dim:
            # Pad with zeros
            padding = torch.zeros(batch_size, self.hidden_dim - fusion.size(1), device=device)
            fusion = torch.cat([fusion, padding], dim=1)
        
        fused = F.relu(fusion)
        
        # Output
        output = self.output_layer(fused)
        
        return output

