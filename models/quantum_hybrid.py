"""
Quantum Hybrid Model for Multimodal Fusion
Uses quantum circuits for fusion operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("Warning: PennyLane not available. Quantum layers will use classical approximation.")


class QuantumFusionLayer(nn.Module):
    """
    Quantum fusion layer using Variational Quantum Circuit (VQC)
    """
    
    def __init__(self, input_dim, output_dim, n_qubits=4, n_layers=2):
        """
        Args:
            input_dim: input dimension
            output_dim: output dimension
            n_qubits: number of qubits in quantum circuit
            n_layers: number of layers in VQC
        """
        super(QuantumFusionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        if PENNYLANE_AVAILABLE:
            # Create quantum device
            self.dev = qml.device('default.qubit', wires=n_qubits)
            
            # Variational parameters
            self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
            
            # Classical pre/post processing
            self.pre_proj = nn.Linear(input_dim, n_qubits)
            self.post_proj = nn.Linear(n_qubits, output_dim)
            
            # Create quantum circuit
            @qml.qnode(device=self.dev, interface='torch')
            def quantum_circuit(inputs, params):
                # Encode classical data into quantum state
                for i in range(n_qubits):
                    qml.RY(inputs[i], wires=i)
                
                # Variational layers
                for layer in range(n_layers):
                    for i in range(n_qubits):
                        qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)
                    
                    # Entangling gates
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                
                # Measure expectation values
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
            self.quantum_circuit = quantum_circuit
        else:
            # Classical approximation
            self.pre_proj = nn.Linear(input_dim, n_qubits * 2)
            self.post_proj = nn.Linear(n_qubits, output_dim)
            self.approx_layer = nn.Sequential(
                nn.Linear(n_qubits * 2, n_qubits * 4),
                nn.Tanh(),
                nn.Linear(n_qubits * 4, n_qubits)
            )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            output: (batch_size, output_dim)
        """
        if PENNYLANE_AVAILABLE:
            # Pre-process
            x_proj = self.pre_proj(x)  # (batch_size, n_qubits)
            
            # Quantum circuit
            batch_size = x.size(0)
            quantum_outputs = []
            for i in range(batch_size):
                q_out = self.quantum_circuit(x_proj[i], self.q_params)
                quantum_outputs.append(torch.stack(q_out))
            
            q_out = torch.stack(quantum_outputs)  # (batch_size, n_qubits)
            # Convert to float32 to match PyTorch's default dtype
            q_out = q_out.to(torch.float32)
        else:
            # Classical approximation
            x_proj = self.pre_proj(x)
            q_out = self.approx_layer(x_proj)
        
        # Post-process
        output = self.post_proj(q_out)
        
        return output


class QuantumHybridModel(nn.Module):
    """
    Quantum Hybrid Model for Multimodal Fusion
    Combines classical encoders with quantum fusion layers
    """
    
    def __init__(self, input_dims, hidden_dim, output_dim, n_qubits=4, n_quantum_layers=2, dropout=0.1):
        """
        Args:
            input_dims: list of input dimensions for each modality
            hidden_dim: hidden dimension for encoders
            output_dim: output dimension
            n_qubits: number of qubits in quantum circuit
            n_quantum_layers: number of layers in VQC
            dropout: dropout rate
        """
        super(QuantumHybridModel, self).__init__()
        
        self.num_modalities = len(input_dims)
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        
        # Classical encoders for each modality
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ) for dim in input_dims
        ])
        
        # Quantum fusion layers
        self.quantum_fusion = nn.ModuleList([
            QuantumFusionLayer(hidden_dim, hidden_dim, n_qubits, n_quantum_layers)
            for _ in range(self.num_modalities)
        ])
        
        # Cross-modal quantum entanglement
        self.cross_quantum = QuantumFusionLayer(
            hidden_dim * self.num_modalities, 
            hidden_dim, 
            n_qubits * 2, 
            n_quantum_layers
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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
            # Pool over sequence if exists
            if len(mod.shape) == 3:
                mod = mod.mean(dim=1)  # (batch_size, input_dim)
            
            # Classical encoding
            enc = self.encoders[i](mod)  # (batch_size, hidden_dim)
            
            # Quantum fusion
            q_enc = self.quantum_fusion[i](enc)  # (batch_size, hidden_dim)
            
            encoded.append(q_enc)
        
        # Cross-modal quantum entanglement
        concat = torch.cat(encoded, dim=1)  # (batch_size, num_modalities * hidden_dim)
        fused = self.cross_quantum(concat)  # (batch_size, hidden_dim)
        
        # Output
        output = self.output_layer(fused)
        
        return output

