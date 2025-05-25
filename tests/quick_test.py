#!/usr/bin/env python3
"""
Quick test to verify imports and basic functionality work correctly.
"""

import sys
import os
# Add the parent directory to the Python path so we can import from algorithms/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Testing imports...")
    from algorithms.rwm_gpu import RandomWalkMH_GPU
    from algorithms.rwm import RandomWalkMH
    from target_distributions import MultivariateNormal, MultivariateNormalTorch
    import torch
    import numpy as np
    print("✅ All imports successful!")
    
    print("\nTesting basic instantiation...")
    target_dist_gpu = MultivariateNormalTorch(2)
    target_dist_cpu = MultivariateNormal(2)
    print(f"✅ Created target distribution (GPU): {target_dist_gpu.get_name()}")
    print(f"✅ Created target distribution (CPU): {target_dist_cpu.get_name()}")
    
    rwm_gpu = RandomWalkMH_GPU(
        dim=2,
        var=1.0,
        target_dist=target_dist_gpu,
        standard_rwm=True,
        symmetric=True, 
        pre_allocate_steps=1
    )
    print(f"✅ Created GPU RWM: {rwm_gpu.get_name()}")
    
    rwm_cpu = RandomWalkMH(2, 1.0, target_dist_cpu)
    print(f"✅ Created CPU RWM: {rwm_cpu.get_name()}")
    
    print("\nTesting basic functionality...")
    
    # Test single step
    rwm_gpu._standard_step()
    print("✅ GPU standard step works")
    
    rwm_cpu.step()
    print("✅ CPU step works")
    
    print("\n🎉 All basic tests passed! Ready to run full test suite.")
    print("Now run: python test_rwm_correctness.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 