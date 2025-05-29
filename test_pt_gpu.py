#!/usr/bin/env python3
"""
Test script for GPU-optimized Parallel Tempering Random Walk Metropolis
"""

import torch
import numpy as np
import time
from algorithms.pt_rwm_gpu_optimized import ParallelTemperingRWM_GPU_Optimized
from target_distributions import MultivariateNormalTorch

def test_pt_gpu_basic():
    """Basic functionality test for GPU Parallel Tempering."""
    print("="*60)
    print("TESTING GPU PARALLEL TEMPERING - BASIC FUNCTIONALITY")
    print("="*60)
    
    # Setup parameters
    dim = 5
    var = 0.5
    num_samples = 1000
    burn_in = 100
    
    # Create a simple target distribution
    target_mean = torch.zeros(dim)
    target_cov = torch.eye(dim)
    target_dist = MultivariateNormalTorch(dim, mean=target_mean, cov=target_cov)
    
    # Initialize PT-RWM GPU
    pt_gpu = ParallelTemperingRWM_GPU_Optimized(
        dim=dim,
        var=var,
        target_dist=target_dist,
        burn_in=burn_in,
        pre_allocate_steps=num_samples,
        swap_every=10,
        geom_temp_spacing=True
    )
    
    print(f"Target distribution: {target_dist.get_name()}")
    print(f"Dimension: {dim}")
    print(f"Number of chains: {pt_gpu.num_chains}")
    print(f"Beta ladder: {pt_gpu.beta_ladder}")
    
    # Generate samples
    print(f"\nGenerating {num_samples} samples with {burn_in} burn-in...")
    start_time = time.time()
    
    samples = pt_gpu.generate_samples(num_samples)
    
    end_time = time.time()
    
    print(f"Generation completed in {end_time - start_time:.3f} seconds")
    print(f"Sample shape: {samples.shape}")
    print(f"Samples per second: {num_samples / (end_time - start_time):.1f}")
    
    # Check basic properties
    sample_mean = torch.mean(samples, dim=0)
    sample_cov = torch.cov(samples.T)
    
    print(f"\nSample statistics:")
    print(f"Sample mean: {sample_mean}")
    print(f"Target mean: {target_mean}")
    print(f"Mean error: {torch.norm(sample_mean - target_mean):.4f}")
    
    print(f"\nSample covariance diagonal: {torch.diag(sample_cov)}")
    print(f"Target covariance diagonal: {torch.diag(target_cov)}")
    
    # Display diagnostic information
    print(f"\nAlgorithm diagnostics:")
    diagnostic_info = pt_gpu.get_diagnostic_info()
    for key, value in diagnostic_info.items():
        print(f"  {key}: {value}")
    
    # Performance summary
    pt_gpu.performance_summary()
    
    return True

def test_pt_gpu_vs_cpu_comparison():
    """Compare GPU PT with CPU PT performance."""
    print("\n" + "="*60)
    print("TESTING GPU vs CPU PARALLEL TEMPERING COMPARISON")
    print("="*60)
    
    # This is a placeholder for comparison with CPU version
    # Would need to import and test CPU PT as well
    print("Note: Full CPU vs GPU comparison would require implementing")
    print("CPU PT test here. For now, we focus on GPU functionality.")

def main():
    """Run all tests."""
    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU (performance will be limited).")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    try:
        # Test basic functionality
        test_pt_gpu_basic()
        
        # Test comparison (placeholder)
        test_pt_gpu_vs_cpu_comparison()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 