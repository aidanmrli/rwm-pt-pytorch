#!/usr/bin/env python3
"""
Quick test to verify imports and basic functionality work correctly.
"""

try:
    print("Testing imports...")
    from algorithms.rwm_gpu import RandomWalkMH_GPU
    from algorithms.rwm import RandomWalkMH
    from target_distributions import MultivariateNormal
    import torch
    import numpy as np
    print("✅ All imports successful!")
    
    print("\nTesting basic instantiation...")
    target_dist = MultivariateNormal(2)
    print(f"✅ Created target distribution: {target_dist.get_name()}")
    
    rwm_gpu = RandomWalkMH_GPU(
        dim=2,
        var=1.0,
        target_dist=target_dist,
        standard_rwm=True
    )
    print(f"✅ Created GPU RWM: {rwm_gpu.get_name()}")
    
    rwm_cpu = RandomWalkMH(2, 1.0, target_dist)
    print(f"✅ Created CPU RWM: {rwm_cpu.get_name()}")
    
    print("\nTesting basic functionality...")
    
    # Test single step
    rwm_gpu._standard_step()
    print("✅ GPU standard step works")
    
    rwm_cpu.step()
    print("✅ CPU step works")
    
    print("\n🎉 All basic tests passed! Ready to run full test suite.")
    print("Now run: python test_standard_rwm.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 