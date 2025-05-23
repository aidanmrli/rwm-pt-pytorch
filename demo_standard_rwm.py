#!/usr/bin/env python3
"""
Demonstration of Standard vs Batch RWM GPU implementations.

This script demonstrates the difference between:
1. True standard Random Walk Metropolis (sequential proposals)
2. Batch processing approach (multiple proposals from same state)

Both versions use GPU acceleration for computational benefits.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from algorithms.rwm_gpu import RandomWalkMH_GPU
from target_distributions import MultivariateNormal

def run_comparison(dim=2, num_samples=5000, var=1.0):
    """Compare standard vs batch RWM implementations."""
    
    print("=== GPU-Accelerated RWM Comparison ===")
    print(f"Problem: {dim}D Standard Gaussian")
    print(f"Samples: {num_samples}")
    print(f"Proposal variance: {var}")
    print()
    
    # Create target distribution
    target_dist = MultivariateNormal(dim)
    
    # Test 1: Standard RWM (True sequential MCMC)
    print("1. Standard RWM (True sequential MCMC)")
    print("   - Each proposal depends on the previous state")
    print("   - Correct MCMC semantics")
    print("   - GPU acceleration for density evaluation")
    
    rwm_standard = RandomWalkMH_GPU(
        dim=dim, 
        var=var, 
        target_dist=target_dist,
        standard_rwm=True,  # True standard RWM
        pre_allocate_steps=num_samples
    )
    
    samples_standard = rwm_standard.generate_samples(num_samples)
    
    print(f"   - Final acceptance rate: {rwm_standard.acceptance_rate:.3f}")
    print(f"   - ESJD: {rwm_standard.expected_squared_jump_distance_gpu():.4f}")
    print()
    
    # Test 2: Batch RWM (Multiple proposals from same state)
    print("2. Batch RWM (Multiple proposals, non-standard)")
    print("   - Multiple proposals from the same state")
    print("   - Faster due to batch processing")
    print("   - Different mixing properties than standard RWM")
    
    rwm_batch = RandomWalkMH_GPU(
        dim=dim, 
        var=var, 
        target_dist=target_dist,
        standard_rwm=False,  # Batch processing mode
        batch_size=32,
        pre_allocate_steps=num_samples
    )
    
    samples_batch = rwm_batch.generate_samples_batch(num_samples, batch_size=32)
    
    print(f"   - Final acceptance rate: {rwm_batch.acceptance_rate:.3f}")
    print(f"   - ESJD: {rwm_batch.expected_squared_jump_distance_gpu():.4f}")
    print()
    
    # Compare results
    print("=== Comparison Results ===")
    
    # Compute empirical means
    mean_standard = np.mean(samples_standard, axis=0)
    mean_batch = np.mean(samples_batch, axis=0)
    true_mean = np.zeros(dim)
    
    print(f"Empirical means:")
    print(f"  Standard RWM: {mean_standard}")
    print(f"  Batch RWM:    {mean_batch}")
    print(f"  True mean:    {true_mean}")
    print()
    
    # Compute empirical covariances
    cov_standard = np.cov(samples_standard.T)
    cov_batch = np.cov(samples_batch.T)
    true_cov = np.eye(dim)
    
    print(f"Empirical covariance diagonal:")
    print(f"  Standard RWM: {np.diag(cov_standard)}")
    print(f"  Batch RWM:    {np.diag(cov_batch)}")
    print(f"  True cov:     {np.diag(true_cov)}")
    print()
    
    # Visualization for 2D case
    if dim == 2:
        create_visualization(samples_standard, samples_batch, 
                           rwm_standard.acceptance_rate, rwm_batch.acceptance_rate)
    
    return samples_standard, samples_batch

def create_visualization(samples_standard, samples_batch, acc_rate_std, acc_rate_batch):
    """Create visualization comparing the two approaches."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trace plots
    axes[0, 0].plot(samples_standard[:1000, 0], alpha=0.7, label='Standard RWM')
    axes[0, 0].set_title(f'Standard RWM - Trace (Accept: {acc_rate_std:.3f})')
    axes[0, 0].set_ylabel('X₁')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(samples_batch[:1000, 0], alpha=0.7, label='Batch RWM', color='orange')
    axes[0, 1].set_title(f'Batch RWM - Trace (Accept: {acc_rate_batch:.3f})')
    axes[0, 1].set_ylabel('X₁')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter plots
    axes[1, 0].scatter(samples_standard[::10, 0], samples_standard[::10, 1], 
                      alpha=0.5, s=1, label='Standard RWM')
    axes[1, 0].set_title('Standard RWM - Sample Distribution')
    axes[1, 0].set_xlabel('X₁')
    axes[1, 0].set_ylabel('X₂')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')
    
    axes[1, 1].scatter(samples_batch[::10, 0], samples_batch[::10, 1], 
                      alpha=0.5, s=1, color='orange', label='Batch RWM')
    axes[1, 1].set_title('Batch RWM - Sample Distribution')
    axes[1, 1].set_xlabel('X₁')
    axes[1, 1].set_ylabel('X₂')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('rwm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'rwm_comparison.png'")

def performance_comparison():
    """Compare computational performance."""
    print("=== Performance Comparison ===")
    
    # You can add timing comparisons here
    # Note: The standard approach will be slower per sample but more correct
    # The batch approach will be faster but uses non-standard MCMC
    
    print("Standard RWM:")
    print("  + Theoretically correct MCMC")
    print("  + Each step follows proper Markov property")
    print("  - Slower due to sequential nature")
    print("  - Limited batch processing opportunities")
    print()
    
    print("Batch RWM:")
    print("  + Much faster due to vectorization")
    print("  + Better GPU utilization")
    print("  - Non-standard MCMC (multiple proposals from same state)")
    print("  - Different theoretical properties")
    print()
    
    print("GPU Benefits for Standard RWM:")
    print("  + Accelerated density evaluations")
    print("  + GPU random number generation")
    print("  + Pre-allocated memory")
    print("  + Tensor operations for proposals")

if __name__ == "__main__":
    # Run the comparison
    samples_std, samples_batch = run_comparison(dim=2, num_samples=5000, var=1.0)
    
    performance_comparison()
    
    print("\n=== Recommendation ===")
    print("For research purposes requiring theoretically correct MCMC:")
    print("  Use standard_rwm=True")
    print()
    print("For maximum computational speed (with caveats):")
    print("  Use standard_rwm=False (batch mode)")
    print()
    print("Both versions provide significant GPU acceleration over CPU-only implementations.") 