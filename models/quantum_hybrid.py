"""
Quantum Hybrid Model for Multimodal Fusion
Uses quantum circuits for fusion operations

Enhanced version with multiple quantum circuit architectures:
1. Basic VQC: Basic variational quantum circuit
2. EfficientSU2: EfficientSU2 ansatz
3. RealAmplitudes: Real amplitudes ansatz
4. MultiControlled: Multi-controlled gates
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
    Enhanced version with multiple ansatz options
    """

    def __init__(self, input_dim, output_dim, n_qubits=4, n_layers=2, ansatz='basic'):
        """
        Args:
            input_dim: input dimension
            output_dim: output dimension
            n_qubits: number of qubits in quantum circuit
            n_layers: number of layers in VQC
            ansatz: type of quantum circuit ansatz
                - 'basic': Basic VQC with RY and CNOT
                - 'efficient_su2': EfficientSU2 ansatz
                - 'real_amplitudes': Real Amplitudes ansatz
                - 'strong_entanglement': Strong entanglement pattern
        """
        super(QuantumFusionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz

        if PENNYLANE_AVAILABLE:
            # Create quantum device
            self.dev = qml.device('default.qubit', wires=n_qubits)

            # Variational parameters
            self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

            # Classical pre/post processing
            self.pre_proj = nn.Linear(input_dim, n_qubits)
            self.post_proj = nn.Linear(n_qubits, output_dim)

            # Create quantum circuit based on ansatz
            self.quantum_circuit = self._create_circuit()

        else:
            # Classical approximation
            self.pre_proj = nn.Linear(input_dim, n_qubits * 2)
            self.post_proj = nn.Linear(n_qubits, output_dim)
            self.approx_layer = nn.Sequential(
                nn.Linear(n_qubits * 2, n_qubits * 4),
                nn.Tanh(),
                nn.Linear(n_qubits * 4, n_qubits)
            )

    def _create_circuit(self):
        """Create quantum circuit based on ansatz type"""

        n_qubits = self.n_qubits
        n_layers = self.n_layers

        if self.ansatz == 'basic':
            @qml.qnode(device=self.dev, interface='torch')
            def circuit(inputs, params):
                # Encode classical data into quantum state
                for i in range(n_qubits):
                    qml.RY(inputs[i], wires=i)

                # Variational layers
                for layer in range(n_layers):
                    for i in range(n_qubits):
                        qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)

                    # Entangling gates (linear chain)
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])

                # Measure expectation values
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        elif self.ansatz == 'efficient_su2':
            @qml.qnode(device=self.dev, interface='torch')
            def circuit(inputs, params):
                # Initial encoding
                for i in range(n_qubits):
                    qml.RY(inputs[i], wires=i)

                # Variational layers with EfficientSU2-style gates
                for layer in range(n_layers):
                    # Rotation layer
                    for i in range(n_qubits):
                        qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)

                    # Entangling layer (alternating CNOTs)
                    if layer % 2 == 0:
                        for i in range(0, n_qubits - 1, 2):
                            qml.CNOT(wires=[i, i + 1])
                    else:
                        for i in range(1, n_qubits - 1, 2):
                            qml.CNOT(wires=[i, i + 1])

                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        elif self.ansatz == 'real_amplitudes':
            @qml.qnode(device=self.dev, interface='torch')
            def circuit(inputs, params):
                # Initial RY gates
                for i in range(n_qubits):
                    qml.RY(inputs[i], wires=i)

                # Variational layers
                for layer in range(n_layers):
                    # Rotation gates
                    for i in range(n_qubits):
                        qml.RY(params[layer, i, 0], wires=i)

                    # Entangling (ring topology)
                    for i in range(n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % n_qubits])

                    # Additional rotations
                    for i in range(n_qubits):
                        qml.RZ(params[layer, i, 1], wires=i)

                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        elif self.ansatz == 'strong_entanglement':
            @qml.qnode(device=self.dev, interface='torch')
            def circuit(inputs, params):
                # Initial encoding with RX gates
                for i in range(n_qubits):
                    qml.RX(inputs[i], wires=i)

                # Variational layers with strong entanglement
                for layer in range(n_layers):
                    # Rotation layer
                    for i in range(n_qubits):
                        qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)

                    # Full entanglement
                    for i in range(n_qubits):
                        for j in range(i + 1, n_qubits):
                            qml.CNOT(wires=[i, j])

                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        else:
            # Default to basic
            @qml.qnode(device=self.dev, interface='torch')
            def circuit(inputs, params):
                for i in range(n_qubits):
                    qml.RY(inputs[i], wires=i)

                for layer in range(n_layers):
                    for i in range(n_qubits):
                        qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])

                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        return circuit

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

            # Process in batches for memory efficiency
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


class MultiHeadQuantumFusion(nn.Module):
    """
    Multi-head quantum fusion for capturing different aspects of fusion
    """

    def __init__(self, input_dim, output_dim, n_qubits=4, n_layers=2, n_heads=2, dropout=0.1):
        super(MultiHeadQuantumFusion, self).__init__()

        self.n_heads = n_heads
        self.head_dim = output_dim // n_heads

        # Create multiple quantum fusion heads
        self.heads = nn.ModuleList([
            QuantumFusionLayer(input_dim, self.head_dim, n_qubits, n_layers)
            for _ in range(n_heads)
        ])

        # Attention weights
        self.attention = nn.Linear(input_dim, n_heads)

        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            output: (batch_size, output_dim)
        """
        # Compute attention weights
        attn_weights = torch.softmax(self.attention(x), dim=-1)  # (batch_size, n_heads)

        # Compute head outputs
        head_outputs = []
        for head in self.heads:
            head_out = head(x)  # (batch_size, head_dim)
            head_outputs.append(head_out)

        # Concatenate head outputs
        concat_out = torch.cat(head_outputs, dim=-1)  # (batch_size, output_dim)

        # Apply attention-weighted combination
        weighted_out = concat_out * attn_weights

        # Output projection
        output = self.output_proj(weighted_out)
        output = self.dropout(output)

        return output


class QuantumHybridModel(nn.Module):
    """
    Quantum Hybrid Model for Multimodal Fusion
    Combines classical encoders with quantum fusion layers

    Enhanced version with support for:
    - Multiple quantum circuit ansatzes
    - Multi-head quantum fusion
    - Residual connections
    """

    def __init__(self, input_dims, hidden_dim, output_dim, n_qubits=4, n_quantum_layers=2,
                 dropout=0.1, ansatz='basic', use_multi_head=False, n_heads=2):
        """
        Args:
            input_dims: list of input dimensions for each modality
            hidden_dim: hidden dimension for encoders
            output_dim: output dimension
            n_qubits: number of qubits in quantum circuit
            n_quantum_layers: number of layers in VQC
            dropout: dropout rate
            ansatz: type of quantum circuit ansatz
            use_multi_head: whether to use multi-head quantum fusion
            n_heads: number of attention heads
        """
        super(QuantumHybridModel, self).__init__()

        self.num_modalities = len(input_dims)
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.use_multi_head = use_multi_head

        # Classical encoders for each modality
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ) for dim in input_dims
        ])

        # Quantum fusion layers
        if use_multi_head:
            self.quantum_fusion = nn.ModuleList([
                MultiHeadQuantumFusion(
                    hidden_dim, hidden_dim, n_qubits, n_quantum_layers, n_heads, dropout
                )
                for _ in range(self.num_modalities)
            ])
        else:
            self.quantum_fusion = nn.ModuleList([
                QuantumFusionLayer(hidden_dim, hidden_dim, n_qubits, n_quantum_layers, ansatz)
                for _ in range(self.num_modalities)
            ])

        # Cross-modal quantum entanglement
        cross_n_qubits = min(n_qubits * 2, 8)  # Limit qubits for cross-modal
        self.cross_quantum = QuantumFusionLayer(
            hidden_dim * self.num_modalities,
            hidden_dim,
            cross_n_qubits,
            n_quantum_layers,
            ansatz
        )

        # Output layer with residual connection
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # Residual connection
        self.residual_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else None

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

            # Classical encoding with residual
            enc = self.encoders[i](mod)  # (batch_size, hidden_dim)

            # Quantum fusion
            q_enc = self.quantum_fusion[i](enc)  # (batch_size, hidden_dim)

            encoded.append(q_enc)

        # Cross-modal quantum entanglement
        concat = torch.cat(encoded, dim=1)  # (batch_size, num_modalities * hidden_dim)
        fused = self.cross_quantum(concat)  # (batch_size, hidden_dim)

        # Output with residual connection
        output = self.output_layer(fused)

        if self.residual_proj is not None:
            output = output + self.residual_proj(fused)

        return output


class QuantumHybridModelV2(nn.Module):
    """
    Quantum Hybrid Model V2
    Improved architecture with:
    - Attention-based modality weighting
    - Hierarchical fusion
    - Better regularization
    """

    def __init__(self, input_dims, hidden_dim, output_dim, n_qubits=4, n_quantum_layers=2,
                 dropout=0.1, ansatz='efficient_su2'):
        super(QuantumHybridModelV2, self).__init__()

        self.num_modalities = len(input_dims)
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        # Modality-specific encoders
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ) for dim in input_dims
        ])

        # Modality attention weights
        self.modality_attention = nn.Parameter(torch.ones(num_modalities) / num_modalities)

        # Quantum fusion for each modality
        self.quantum_fusion = nn.ModuleList([
            QuantumFusionLayer(hidden_dim, hidden_dim, n_qubits, n_quantum_layers, ansatz)
            for _ in range(self.num_modalities)
        ])

        # Hierarchical fusion
        # Level 1: Pairwise fusion
        self.pairwise_fusion = nn.ModuleList([
            QuantumFusionLayer(hidden_dim * 2, hidden_dim, n_qubits, n_quantum_layers, ansatz)
            for _ in range(num_modalities * (num_modalities - 1) // 2)
        ])

        # Level 2: Global fusion
        self.global_fusion = QuantumFusionLayer(
            hidden_dim * num_modalities,
            hidden_dim,
            min(n_qubits * 2, 8),
            n_quantum_layers,
            ansatz
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
            modalities: tuple of tensors
        Returns:
            output: (batch_size, output_dim)
        """
        batch_size = modalities[0].size(0)

        # Encode each modality
        encoded = []
        for i, mod in enumerate(modalities):
            if len(mod.shape) == 3:
                mod = mod.mean(dim=1)
            enc = self.encoders[i](mod)
            q_enc = self.quantum_fusion[i](enc)
            encoded.append(q_enc)

        # Modality attention
        attn_weights = torch.softmax(self.modality_attention, dim=0)
        attended = [enc * w for enc, w in zip(encoded, attn_weights)]

        # Pairwise fusion
        pairwise_outputs = []
        idx = 0
        for i in range(len(encoded)):
            for j in range(i + 1, len(encoded)):
                pair = torch.cat([encoded[i], encoded[j]], dim=1)
                fused = self.pairwise_fusion[idx](pair)
                pairwise_outputs.append(fused)
                idx += 1

        # Global fusion
        concat = torch.cat(encoded + pairwise_outputs, dim=1)
        fused = self.global_fusion(concat)

        # Output
        output = self.output_layer(fused)

        return output
