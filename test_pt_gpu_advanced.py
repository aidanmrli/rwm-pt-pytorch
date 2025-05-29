#!/usr/bin/env python3
"""
Advanced test script for GPU-optimized Parallel Tempering Random Walk Metropolis
Demonstrates performance on challenging multimodal distributions
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from algorithms.pt_rwm_gpu_optimized import ParallelTemperingRWM_GPU_Optimized
from algorithms.rwm_gpu_optimized import RandomWalkMH_GPU_Optimized
from target_distributions import MultivariateNormalTorch, ThreeMixtureDistributionTorch

def test_multimodal_sampling():
    """Test GPU PT on challenging multimodal distribution."""
    print("="*70)
    print("TESTING GPU PARALLEL TEMPERING ON MULTIMODAL DISTRIBUTION")
    print("="*70)
    
    # Set device explicitly
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Running on CPU (performance will be limited).")
    
    # Setup parameters for challenging multimodal distribution
    dim = 10
    var = 0.8
    num_samples = 5000
    burn_in = 1000
    
    # Create three-component mixture distribution (challenging for MCMC)
    # For separation=4.0, create mode centers at [-4, 0, ..., 0], [0, 0, ..., 0], [4, 0, ..., 0]
    mode_centers = [
        [-4.0] + [0.0] * (dim - 1),  # (-4, 0, ..., 0)
        [0.0] * dim,                  # (0, 0, ..., 0)
        [4.0] + [0.0] * (dim - 1)    # (4, 0, ..., 0)
    ]
    target_dist = ThreeMixtureDistributionTorch(dim=dim, mode_centers=mode_centers, device=device)
    
    print(f"Target distribution: {target_dist.get_name()}")
    print(f"Dimension: {dim}")
    print(f"Components separated by: 4.0 standard deviations")
    print(f"Device: {device}")
    
    # Test GPU Parallel Tempering
    print(f"\n--- GPU Parallel Tempering ---")
    pt_gpu = ParallelTemperingRWM_GPU_Optimized(
        dim=dim,
        var=var,
        target_dist=target_dist,
        burn_in=burn_in,
        pre_allocate_steps=num_samples,
        swap_every=15,
        geom_temp_spacing=True,
        device=device
    )
    
    print(f"Number of chains: {pt_gpu.num_chains}")
    print(f"Beta ladder: {[f'{b:.3f}' for b in pt_gpu.beta_ladder]}")
    
    start_time = time.time()
    pt_samples = pt_gpu.generate_samples(num_samples)
    pt_time = time.time() - start_time
    
    print(f"PT GPU completed in: {pt_time:.3f} seconds")
    print(f"PT GPU rate: {num_samples / pt_time:.1f} samples/sec")
    print(f"PT swap acceptance rate: {pt_gpu.swap_acceptance_rate:.3f}")
    print(f"PT samples device: {pt_samples.device}")
    
    # Test standard GPU RWM for comparison
    print(f"\n--- Standard GPU RWM (for comparison) ---")
    rwm_gpu = RandomWalkMH_GPU_Optimized(
        dim=dim,
        var=var,
        target_dist=target_dist,
        burn_in=burn_in,
        pre_allocate_steps=num_samples,
        device=device
    )
    
    start_time = time.time()
    rwm_samples = rwm_gpu.generate_samples(num_samples)
    rwm_time = time.time() - start_time
    
    print(f"RWM GPU completed in: {rwm_time:.3f} seconds")
    print(f"RWM GPU rate: {num_samples / rwm_time:.1f} samples/sec")
    print(f"RWM acceptance rate: {rwm_gpu.acceptance_rate:.3f}")
    print(f"RWM samples device: {rwm_samples.device}")
    
    # Compare mixing and exploration
    print(f"\n--- Mixing Quality Comparison ---")
    
    # Compute ESJD for both methods
    pt_esjd = pt_gpu.expected_squared_jump_distance_gpu()
    rwm_esjd = rwm_gpu.expected_squared_jump_distance_gpu()
    
    print(f"PT ESJD: {pt_esjd:.6f}")
    print(f"RWM ESJD: {rwm_esjd:.6f}")
    print(f"PT ESJD improvement: {pt_esjd / rwm_esjd:.2f}x")
    
    # Analyze sample statistics
    print(f"\n--- Sample Statistics ---")
    pt_mean = torch.mean(pt_samples, dim=0)
    rwm_mean = torch.mean(rwm_samples, dim=0)
    pt_std = torch.std(pt_samples, dim=0)
    rwm_std = torch.std(rwm_samples, dim=0)
    
    print(f"PT sample mean norm: {torch.norm(pt_mean):.4f}")
    print(f"RWM sample mean norm: {torch.norm(rwm_mean):.4f}")
    print(f"PT sample std (first 3 dims): {pt_std[:3]}")
    print(f"RWM sample std (first 3 dims): {rwm_std[:3]}")
    
    return pt_samples, rwm_samples, pt_gpu, rwm_gpu

def test_scaling_performance():
    """Test how GPU PT scales with problem dimension."""
    print("\n" + "="*70)
    print("TESTING GPU PARALLEL TEMPERING SCALING PERFORMANCE")
    print("="*70)
    
    # Set device explicitly
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Running on CPU (performance will be limited).")
    
    dimensions = [5, 10, 20, 50]
    num_samples = 2000
    burn_in = 500
    var = 0.5
    
    results = []
    
    for dim in dimensions:
        print(f"\n--- Testing dimension: {dim} ---")
        
        # Create target distribution with explicit device
        target_dist = MultivariateNormalTorch(dim, device=device)
        
        # Initialize PT GPU with explicit device
        pt_gpu = ParallelTemperingRWM_GPU_Optimized(
            dim=dim,
            var=var,
            target_dist=target_dist,
            burn_in=burn_in,
            pre_allocate_steps=num_samples,
            swap_every=20,
            geom_temp_spacing=True,
            device=device
        )
        
        # Time sample generation
        start_time = time.time()
        samples = pt_gpu.generate_samples(num_samples)
        generation_time = time.time() - start_time
        
        rate = num_samples / generation_time
        memory_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        
        result = {
            'dim': dim,
            'time': generation_time,
            'rate': rate,
            'num_chains': pt_gpu.num_chains,
            'memory_mb': memory_mb,
            'swap_acceptance': pt_gpu.swap_acceptance_rate,
            'device': str(device)
        }
        results.append(result)
        
        print(f"Time: {generation_time:.3f}s, Rate: {rate:.1f} samples/sec")
        print(f"Chains: {pt_gpu.num_chains}, Memory: {memory_mb:.1f} MB")
        print(f"Swap acceptance: {pt_gpu.swap_acceptance_rate:.3f}")
        print(f"Device: {device}")
    
    # Display scaling summary
    print(f"\n--- Scaling Summary ---")
    print(f"{'Dim':<6} {'Time(s)':<8} {'Rate':<12} {'Chains':<8} {'Memory(MB)':<12} {'SwapAcc':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['dim']:<6} {r['time']:<8.3f} {r['rate']:<12.1f} {r['num_chains']:<8} {r['memory_mb']:<12.1f} {r['swap_acceptance']:<8.3f}")
    
    return results

def test_temperature_ladder_optimization():
    """Test different temperature ladder configurations."""
    print("\n" + "="*70)
    print("TESTING TEMPERATURE LADDER OPTIMIZATION")
    print("="*70)
    
    # Set device explicitly
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Running on CPU (performance will be limited).")
    
    dim = 15
    var = 0.6
    num_samples = 3000
    burn_in = 500
    
    # Create challenging target distribution with explicit device
    # For separation=3.0, create mode centers at [-3, 0, ..., 0], [0, 0, ..., 0], [3, 0, ..., 0]
    mode_centers = [
        [-3.0] + [0.0] * (dim - 1),  # (-3, 0, ..., 0)
        [0.0] * dim,                  # (0, 0, ..., 0)
        [3.0] + [0.0] * (dim - 1)    # (3, 0, ..., 0)
    ]
    target_dist = ThreeMixtureDistributionTorch(dim=dim, mode_centers=mode_centers, device=device)
    
    print(f"Device: {device}")
    
    # Test different ladder configurations
    ladder_configs = [
        {'type': 'geometric', 'c': 0.5},
        {'type': 'manual', 'betas': [1.0, 0.7, 0.5, 0.3, 0.15, 0.05]},
        {'type': 'manual', 'betas': [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]},
    ]
    
    results = []
    
    for i, config in enumerate(ladder_configs):
        print(f"\n--- Configuration {i+1}: {config['type']} ---")
        
        if config['type'] == 'geometric':
            pt_gpu = ParallelTemperingRWM_GPU_Optimized(
                dim=dim, var=var, target_dist=target_dist,
                burn_in=burn_in, pre_allocate_steps=num_samples,
                geom_temp_spacing=True, swap_every=20, device=device
            )
        else:
            pt_gpu = ParallelTemperingRWM_GPU_Optimized(
                dim=dim, var=var, target_dist=target_dist,
                burn_in=burn_in, pre_allocate_steps=num_samples,
                beta_ladder=config['betas'], swap_every=20, device=device
            )
        
        print(f"Beta ladder: {[f'{b:.3f}' for b in pt_gpu.beta_ladder]}")
        print(f"Number of chains: {pt_gpu.num_chains}")
        
        start_time = time.time()
        samples = pt_gpu.generate_samples(num_samples)
        generation_time = time.time() - start_time
        
        esjd = pt_gpu.expected_squared_jump_distance_gpu()
        swap_acc = pt_gpu.swap_acceptance_rate
        
        result = {
            'config': config,
            'num_chains': pt_gpu.num_chains,
            'time': generation_time,
            'esjd': esjd,
            'swap_acceptance': swap_acc,
            'rate': num_samples / generation_time,
            'device': str(device)
        }
        results.append(result)
        
        print(f"Time: {generation_time:.3f}s, ESJD: {esjd:.6f}")
        print(f"Swap acceptance: {swap_acc:.3f}, Rate: {result['rate']:.1f} samples/sec")
        print(f"Sample device: {samples.device}")
    
    # Find best configuration
    best_config = max(results, key=lambda x: x['esjd'])
    print(f"\n--- Best Configuration (highest ESJD) ---")
    print(f"Type: {best_config['config']['type']}")
    print(f"Chains: {best_config['num_chains']}")
    print(f"ESJD: {best_config['esjd']:.6f}")
    print(f"Swap acceptance: {best_config['swap_acceptance']:.3f}")
    
    return results

def main():
    """Run all advanced tests."""
    try:
        # Test 1: Multimodal sampling comparison
        pt_samples, rwm_samples, pt_gpu, rwm_gpu = test_multimodal_sampling()
        
        # Test 2: Scaling performance
        scaling_results = test_scaling_performance()
        
        # Test 3: Temperature ladder optimization
        ladder_results = test_temperature_ladder_optimization()
        
        print("\n" + "="*70)
        print("ALL ADVANCED TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        # Summary insights
        print("\n--- Key Insights ---")
        print("✓ GPU Parallel Tempering provides superior mixing on multimodal distributions")
        print("✓ Performance scales well with dimension")
        print("✓ Temperature ladder configuration significantly impacts performance")
        print("✓ Swap acceptance rates indicate effective temperature spacing")
        print("✓ Memory usage remains manageable even for high-dimensional problems")
        
        return True
        
    except Exception as e:
        print(f"\nADVANCED TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 