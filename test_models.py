"""
Quick test script to verify all models can be instantiated and run
"""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import TFN, LMF, MFN, MulT, GCNFusion, HypergraphFusion, QuantumHybridModel


def test_model(model_class, model_name, input_dims, batch_size=4, seq_len=10):
    """Test a model"""
    print(f"\nTesting {model_name}...")
    try:
        # Create model
        if model_name == 'TFN':
            model = model_class(input_dims=input_dims, hidden_dim=64, output_dim=1)
        elif model_name == 'LMF':
            model = model_class(input_dims=input_dims, hidden_dim=64, output_dim=1, rank=4)
        elif model_name == 'MFN':
            model = model_class(input_dims=input_dims, hidden_dim=64, output_dim=1, memory_size=4, num_layers=1)
        elif model_name == 'MulT':
            model = model_class(input_dims=input_dims, d_model=64, output_dim=1, num_heads=4, num_layers=2)
        elif model_name == 'GCN':
            model = model_class(input_dims=input_dims, hidden_dim=64, output_dim=1, num_layers=2)
        elif model_name == 'Hypergraph':
            model = model_class(input_dims=input_dims, hidden_dim=64, output_dim=1, num_layers=2)
        elif model_name == 'QuantumHybrid':
            model = model_class(input_dims=input_dims, hidden_dim=64, output_dim=1, n_qubits=2, n_quantum_layers=1)
        else:
            print(f"  Unknown model: {model_name}")
            return False
        
        # Create dummy input
        modalities = [torch.randn(batch_size, seq_len, dim) for dim in input_dims]
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(*modalities)
        
        print(f"  ✓ {model_name} works!")
        print(f"    Input shapes: {[m.shape for m in modalities]}")
        print(f"    Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"  ✗ {model_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("Testing All Models")
    print("="*60)
    
    input_dims = [32, 64, 48]
    batch_size = 4
    seq_len = 10
    
    models_to_test = [
        (TFN, 'TFN'),
        (LMF, 'LMF'),
        (MFN, 'MFN'),
        (MulT, 'MulT'),
        (GCNFusion, 'GCN'),
        (HypergraphFusion, 'Hypergraph'),
        (QuantumHybridModel, 'QuantumHybrid'),
    ]
    
    results = {}
    for model_class, model_name in models_to_test:
        results[model_name] = test_model(model_class, model_name, input_dims, batch_size, seq_len)
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for model_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{model_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\nAll models passed!")
    else:
        print("\nSome models failed. Please check the errors above.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)







