"""
Graph-based Baselines
GCN and Hypergraph NN for modeling inter-modal topological relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HypergraphConv
from torch_geometric.data import Data, Batch


class GCNFusion(nn.Module):
    """
    Graph Convolutional Network for Multimodal Fusion
    Each modality is treated as a node in the graph
    """
    
    def __init__(self, input_dims, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        """
        Args:
            input_dims: list of input dimensions for each modality
            hidden_dim: hidden dimension for GCN
            output_dim: output dimension
            num_layers: number of GCN layers
            dropout: dropout rate
        """
        super(GCNFusion, self).__init__()
        
        self.num_modalities = len(input_dims)
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        
        # Input projections
        self.input_projs = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def build_graph(self, batch_size, device):
        """
        Build a fully connected graph where each node represents a modality
        """
        # Create edges: fully connected graph
        edge_index = []
        for i in range(self.num_modalities):
            for j in range(self.num_modalities):
                if i != j:
                    edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        
        # Repeat for batch
        edge_indices = []
        node_offsets = []
        for b in range(batch_size):
            offset = b * self.num_modalities
            edge_indices.append(edge_index + offset)
            node_offsets.append(offset)
        
        batch_edge_index = torch.cat(edge_indices, dim=1)
        
        return batch_edge_index
    
    def forward(self, *modalities):
        """
        Args:
            modalities: tuple of tensors, each (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, output_dim)
        """
        batch_size = modalities[0].size(0)
        device = modalities[0].device
        
        # Project and pool each modality
        node_features = []
        for i, mod in enumerate(modalities):
            if len(mod.shape) == 3:
                mod = mod.mean(dim=1)  # (batch_size, input_dim)
            proj = self.input_projs[i](mod)  # (batch_size, hidden_dim)
            node_features.append(proj)
        
        # Stack node features: (batch_size * num_modalities, hidden_dim)
        x = torch.cat(node_features, dim=0)
        
        # Build graph
        edge_index = self.build_graph(batch_size, device)
        
        # GCN layers
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Reshape: (batch_size, num_modalities, hidden_dim)
        x = x.view(batch_size, self.num_modalities, self.hidden_dim)
        
        # Flatten and output
        x = x.view(batch_size, -1)  # (batch_size, num_modalities * hidden_dim)
        output = self.output_layer(x)
        
        return output


class HypergraphFusion(nn.Module):
    """
    Hypergraph Neural Network for Multimodal Fusion
    Models higher-order relationships between modalities
    """
    
    def __init__(self, input_dims, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        """
        Args:
            input_dims: list of input dimensions for each modality
            hidden_dim: hidden dimension
            output_dim: output dimension
            num_layers: number of hypergraph layers
            dropout: dropout rate
        """
        super(HypergraphFusion, self).__init__()
        
        self.num_modalities = len(input_dims)
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        
        # Input projections
        self.input_projs = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        
        # Hypergraph layers
        self.hypergraph_layers = nn.ModuleList()
        self.hypergraph_layers.append(HypergraphConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.hypergraph_layers.append(HypergraphConv(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def build_hyperedge_index(self, batch_size, device):
        """
        Build hyperedges: each hyperedge connects all modalities
        For simplicity, we create one hyperedge per batch item
        """
        # Each hyperedge contains all modalities
        hyperedge_index = []
        for b in range(batch_size):
            for m in range(self.num_modalities):
                node_idx = b * self.num_modalities + m
                hyperedge_idx = b
                hyperedge_index.append([node_idx, hyperedge_idx])
        
        hyperedge_index = torch.tensor(hyperedge_index, dtype=torch.long, device=device).t().contiguous()
        
        return hyperedge_index
    
    def forward(self, *modalities):
        """
        Args:
            modalities: tuple of tensors
        Returns:
            output: (batch_size, output_dim)
        """
        batch_size = modalities[0].size(0)
        device = modalities[0].device
        
        # Project and pool each modality
        node_features = []
        for i, mod in enumerate(modalities):
            if len(mod.shape) == 3:
                mod = mod.mean(dim=1)
            proj = self.input_projs[i](mod)
            node_features.append(proj)
        
        # Stack node features
        x = torch.cat(node_features, dim=0)  # (batch_size * num_modalities, hidden_dim)
        
        # Build hyperedge index
        hyperedge_index = self.build_hyperedge_index(batch_size, device)
        
        # Hypergraph layers
        for i, hg_layer in enumerate(self.hypergraph_layers):
            x = hg_layer(x, hyperedge_index)
            if i < len(self.hypergraph_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Reshape
        x = x.view(batch_size, self.num_modalities, self.hidden_dim)
        
        # Flatten and output
        x = x.view(batch_size, -1)
        output = self.output_layer(x)
        
        return output







