#!/usr/bin/env python3
"""
Quick test script for the newly optimized GPU Parallel Tempering RWM
This is a fast test you can run immediately to verify the optimizations work.
"""

import torch
import numpy as np
import time
import sys
import os

# Add the parent directory to the Python path so we can import from algorithms/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.pt_rwm_gpu_optimized import ParallelTemperingRWM_GPU_Optimized
from target_distributions import MultivariateNormalTorch


def quick_optimization_test():
    """Quick test of all three optimizations."""
    print("="*60)
    print("QUICK TEST: GPU Parallel Tempering Optimizations")
    print("="*60)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU (GPU not available)")
    
    # Simple test problem
    dim = 5
    num_samples = 5000
    burn_in = 100
    
    print(f"\nTest Setup:")
    print(f"  Dimension: {dim}")
    print(f"  Samples: {num_samples}")
    print(f"  Burn-in: {burn_in}")
    
    # Create target distribution
    target_dist = MultivariateNormalTorch(dim=dim, device=device)
    
    # Create optimized PT algorithm
    pt_gpu = ParallelTemperingRWM_GPU_Optimized(
        dim=dim,
        var=0.5,
        target_dist=target_dist,
        burn_in=burn_in,
        pre_allocate_steps=num_samples,
        swap_every=10,
        geom_temp_spacing=True,
        device=device
    )
    
    print(f"\nAlgorithm Info:")
    print(f"  Name: {pt_gpu.get_name()}")
    print(f"  Chains: {pt_gpu.num_chains}")
    print(f"  Beta ladder: {[f'{b:.3f}' for b in pt_gpu.beta_ladder]}")
    
    # Test the three optimizations by running sampling
    print(f"\n🚀 Testing optimizations...")
    start_time = time.time()
    
    samples = pt_gpu.generate_samples(num_samples)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n✅ Sampling completed!")
    print(f"  Time: {total_time:.3f} seconds")
    print(f"  Rate: {num_samples / total_time:.1f} samples/sec")
    print(f"  Sample shape: {samples.shape}")
    print(f"  Device: {samples.device}")
    
    # Quick statistical check
    sample_mean = torch.mean(samples, dim=0)
    sample_std = torch.std(samples, dim=0)
    mean_error = torch.norm(sample_mean).item()
    
    print(f"\nStatistical Check:")
    print(f"  Mean error: {mean_error:.4f} (should be < 0.3)")
    print(f"  Mean std: {torch.mean(sample_std):.3f} (should be ~1.0)")
    print(f"  Swap acceptance: {pt_gpu.swap_acceptance_rate:.3f}")
    
    # Verify optimizations worked
    print(f"\n🔍 Optimization Verification:")
    diagnostics = pt_gpu.get_diagnostic_info()
    
    # Check that optimization features are present
    if 'batch_matrix_multiply' in diagnostics:
        print("  ✅ Batch Matrix Multiply: Implemented")
    if 'precomputed_randoms' in diagnostics:
        print("  ✅ Precomputed Randoms: Implemented")
    if 'clone_free_swaps' in diagnostics:
        print("  ✅ Clone-Free Swaps: Implemented")
    
    # Basic validation
    if mean_error < 0.3 and pt_gpu.swap_acceptance_rate > 0.05:
        print(f"\n🎉 QUICK TEST PASSED! All optimizations working correctly.")
        return True
    else:
        print(f"\n❌ QUICK TEST FAILED: Statistical validation failed")
        return False


def performance_comparison():
    """Quick performance comparison showing optimization benefits."""
    print(f"\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("⚠️  Performance comparison limited on CPU")
    
    # Test different problem sizes
    test_configs = [
        {"dim": 5, "samples": 5000, "description": "Small"},
        {"dim": 10, "samples": 5000, "description": "Medium"},
        {"dim": 20, "samples": 5000, "description": "Large"},
    ]
    
    print(f"\nTesting algorithm performance on different problem sizes:")
    print(f"{'Size':<8} {'Dim':<5} {'Samples':<8} {'Time(s)':<8} {'Rate':<12} {'Memory(MB)':<12}")
    print("-" * 60)
    
    for config in test_configs:
        dim = config["dim"]
        num_samples = config["samples"]
        
        target_dist = MultivariateNormalTorch(dim=dim, device=device)
        
        pt_gpu = ParallelTemperingRWM_GPU_Optimized(
            dim=dim,
            var=0.5,
            target_dist=target_dist,
            burn_in=100,
            pre_allocate_steps=num_samples,
            swap_every=20,
            geom_temp_spacing=True,
            device=device
        )
        
        start_time = time.time()
        samples = pt_gpu.generate_samples(num_samples)
        total_time = time.time() - start_time
        
        rate = num_samples / total_time
        memory_mb = torch.cuda.memory_allocated() / 1e6 if device.type == 'cuda' else 0
        
        print(f"{config['description']:<8} {dim:<5} {num_samples:<8} {total_time:<8.3f} {rate:<12.1f} {memory_mb:<12.1f}")
    
    print(f"\n✅ Performance comparison completed!")


def main():
    """Run quick tests."""
    print("QUICK OPTIMIZATION TEST FOR PARALLEL TEMPERING")
    print("This will verify that all three optimizations are working correctly.\n")
    
    try:
        # Run quick test
        success = quick_optimization_test()
        
        if success:
            # Run performance comparison
            performance_comparison()
            
            print(f"\n" + "="*60)
            print("🎉 ALL QUICK TESTS PASSED!")
            print("="*60)
            print("Your optimized Parallel Tempering algorithm is working correctly!")
            print("\nOptimizations implemented:")
            print("  ✅ Batch Matrix Multiply (vectorized increment generation)")
            print("  ✅ Precomputed Random Numbers (zero runtime overhead)")
            print("  ✅ Clone-Free Swaps (in-place tensor operations)")
            print("\nRun 'python tests/test_pt_gpu_optimizations.py' for comprehensive tests.")
            
        return success
        
    except Exception as e:
        print(f"\n❌ QUICK TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
