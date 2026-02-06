"""
Quick start script: train all models and compare results
"""

import os
import sys
import subprocess


def main():
    print("="*60)
    print("Quantum Multimodal Fusion Comparison")
    print("="*60)
    
    # Step 1: Test models
    print("\n[Step 1/3] Testing all models...")
    result = subprocess.run([sys.executable, "test_models.py"], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR: Model tests failed!")
        print(result.stderr)
        return False
    
    # Step 2: Train models
    print("\n[Step 2/3] Training all models...")
    print("This may take a while...")
    result = subprocess.run([sys.executable, "train.py", 
                           "--config", "configs/config.yaml"],
                          capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR: Training failed!")
        print(result.stderr)
        return False
    
    # Step 3: Compare results
    print("\n[Step 3/3] Comparing model performance...")
    result = subprocess.run([sys.executable, "compare.py",
                           "--results_dir", "results"],
                          capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR: Comparison failed!")
        print(result.stderr)
        return False
    
    print("\n" + "="*60)
    print("All done! Check the 'results/' directory for outputs.")
    print("="*60)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)







