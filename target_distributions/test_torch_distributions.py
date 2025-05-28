#!/usr/bin/env python3
"""
Test script for PyTorch-native target distributions.
Verifies that all implementations work correctly and provide consistent results.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path to import target distributions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from target_distributions import (
    MultivariateNormalTorch, HypercubeTorch, 
    ThreeMixtureDistributionTorch, RoughCarpetDistributionTorch,
    IIDGammaTorch, IIDBetaTorch,
    FullRosenbrockTorch, EvenRosenbrockTorch, HybridRosenbrockTorch
)

def test_distribution(dist_class, dist_name, *args, **kwargs):
    """Test a single distribution class."""
    print(f"\n=== Testing {dist_name} ===")
    
    try:
        # Create distribution
        dist = dist_class(*args, **kwargs)
        print(f"✓ Created {dist.get_name()}")
        
        # Test single point evaluation
        if dist_name == "HybridRosenbrockTorch":
            # For HybridRosenbrockTorch, dim = 1 + n2 * (n1 - 1)
            n1, n2 = args[0], args[1]
            dim = 1 + n2 * (n1 - 1)
        else:
            dim = args[0] if args else 2
        if 'Beta' in dist_name:
            x_single = torch.rand(dim, device=dist.device, dtype=torch.float32) * 0.8 + 0.1  # (0.1, 0.9)
        elif 'Gamma' in dist_name:
            x_single = torch.rand(dim, device=dist.device, dtype=torch.float32) * 5 + 0.1  # (0.1, 5.1)
        elif 'Hypercube' in dist_name:
            x_single = torch.rand(dim, device=dist.device, dtype=torch.float32) * 0.8 + 0.1  # (0.1, 0.9)
        else:
            x_single = torch.randn(dim, device=dist.device, dtype=torch.float32)
        
        density_single = dist.density(x_single)
        log_density_single = dist.log_density(x_single)
        print(f"✓ Single point evaluation: density={density_single.item():.6f}, log_density={log_density_single.item():.6f}")
        
        # Test batch evaluation
        batch_size = 10
        if 'Beta' in dist_name:
            x_batch = torch.rand(batch_size, dim, device=dist.device, dtype=torch.float32) * 0.8 + 0.1
        elif 'Gamma' in dist_name:
            x_batch = torch.rand(batch_size, dim, device=dist.device, dtype=torch.float32) * 5 + 0.1
        elif 'Hypercube' in dist_name:
            x_batch = torch.rand(batch_size, dim, device=dist.device, dtype=torch.float32) * 0.8 + 0.1
        else:
            x_batch = torch.randn(batch_size, dim, device=dist.device, dtype=torch.float32)
        
        density_batch = dist.density(x_batch)
        log_density_batch = dist.log_density(x_batch)
        print(f"✓ Batch evaluation: {batch_size} points, mean_density={density_batch.mean().item():.6f}")
        
        # Test consistency between density and log_density
        density_from_log = torch.exp(log_density_batch)
        max_diff = torch.max(torch.abs(density_batch - density_from_log)).item()
        print(f"✓ Consistency check: max_diff={max_diff:.2e}")
        
        # Test sampling (CPU-based)
        sample = dist.draw_sample()
        print(f"✓ CPU sampling: shape={sample.shape}, mean={np.mean(sample):.3f}")
        
        # Test GPU sampling if available
        if hasattr(dist, 'draw_samples_torch'):
            samples_torch = dist.draw_samples_torch(5)
            print(f"✓ GPU sampling: shape={samples_torch.shape}, mean={samples_torch.mean().item():.3f}")
        
        # Test device movement
        if torch.cuda.is_available():
            original_device = dist.device
            dist.to('cpu')
            print(f"✓ Moved to CPU: {dist.device}")
            dist.to(original_device)
            print(f"✓ Moved back to {original_device}")
        
        print(f"✓ {dist_name} passed all tests!")
        return True
        
    except Exception as e:
        print(f"✗ {dist_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing PyTorch-native Target Distributions")
    print("=" * 50)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test parameters
    dim = 5
    
    # List of distributions to test
    tests = [
        (MultivariateNormalTorch, "MultivariateNormalTorch", dim),
        (HypercubeTorch, "HypercubeTorch", dim),
        (ThreeMixtureDistributionTorch, "ThreeMixtureDistributionTorch", dim),
        (ThreeMixtureDistributionTorch, "ThreeMixtureDistributionTorch (scaled)", dim, True),
        (RoughCarpetDistributionTorch, "RoughCarpetDistributionTorch", dim),
        (RoughCarpetDistributionTorch, "RoughCarpetDistributionTorch (scaled)", dim, True),
        (IIDGammaTorch, "IIDGammaTorch", dim),
        (IIDBetaTorch, "IIDBetaTorch", dim),
        (FullRosenbrockTorch, "FullRosenbrockTorch", dim),
        (EvenRosenbrockTorch, "EvenRosenbrockTorch", 6),  # Must be even dimension
        (HybridRosenbrockTorch, "HybridRosenbrockTorch", 3, 2),  # n1=3, n2=2 -> dim=1+2*(3-1)=5
    ]
    
    # Run tests
    passed = 0
    total = len(tests)
    
    for test_args in tests:
        if len(test_args) == 3:
            dist_class, name, dim_arg = test_args
            success = test_distribution(dist_class, name, dim_arg, device=device)
        elif len(test_args) == 4:
            dist_class, name, arg1, arg2 = test_args
            if name == "HybridRosenbrockTorch":
                # HybridRosenbrockTorch takes n1, n2 parameters
                success = test_distribution(dist_class, name, arg1, arg2, device=device)
            else:
                # Other distributions with scaling parameter
                success = test_distribution(dist_class, name, arg1, arg2, device=device)
        else:
            # Handle other cases if needed
            success = test_distribution(*test_args, device=device)
        
        if success:
            passed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} distributions passed")
    
    if passed == total:
        print("🎉 All tests passed! PyTorch distributions are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main()) 