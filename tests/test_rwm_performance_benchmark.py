#!/usr/bin/env python3
"""
Benchmark script to compare RWM GPU implementations and identify optimization opportunities.
"""
import torch
import numpy as np
import time
import sys
import os
# Add the parent directory to the Python path so we can import from algorithms/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.rwm_gpu import RandomWalkMH_GPU
from algorithms.rwm_gpu_optimized import RandomWalkMH_GPU_Optimized
from target_distributions.multivariate_normal_torch import MultivariateNormalTorch
# Import new funnel distributions for comprehensive benchmarking
from target_distributions.funnel_torch import NealFunnelTorch

# Legacy CPU implementation
from algorithms.rwm import RandomWalkMH
from target_distributions.multivariate_normal import MultivariateNormal
from interfaces import MCMCSimulation

def benchmark_rwm_implementation(algorithm_class, algorithm_name, target_dist, dim, var, num_samples, use_cpu_wrapper=False, **kwargs):
    """Benchmark a specific RWM implementation."""
    print(f"\n=== Benchmarking {algorithm_name} ===")
    
    if use_cpu_wrapper:
        # CPU implementation uses MCMCSimulation wrapper
        if 'pre_allocate_steps' in kwargs:
            kwargs.pop('pre_allocate_steps')  # Remove GPU-specific parameter
        
        simulation = MCMCSimulation(
            dim=dim,
            sigma=var,
            num_iterations=num_samples,
            algorithm=algorithm_class,
            target_dist=target_dist,
            symmetric=True,
            seed=42
        )
        
        print(f"Configuration: CPU-based {algorithm_name}")
        
        # Benchmark
        print(f"Generating {num_samples} samples...")
        
        cpu_start = time.time()
        chain = simulation.generate_samples()
        cpu_end = time.time()
        
        cpu_time = cpu_end - cpu_start
        gpu_time = cpu_time  # For CPU, both times are the same
        
        # Compute ESJD
        esjd = simulation.expected_squared_jump_distance()
        acceptance_rate = simulation.acceptance_rate()
        
        # Results
        results = {
            'algorithm': algorithm_name,
            'samples_per_sec_cpu': num_samples / cpu_time,
            'samples_per_sec_gpu': num_samples / gpu_time,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'acceptance_rate': acceptance_rate,
            'esjd': esjd,
            'memory_mb': 0  # CPU doesn't track GPU memory
        }
        
    else:
        # GPU implementation - original code
        # Initialize algorithm
        if 'pre_allocate_steps' not in kwargs:
            kwargs['pre_allocate_steps'] = num_samples
        
        alg = algorithm_class(dim=dim, var=var, target_dist=target_dist, **kwargs)
        
        # Print configuration
        if hasattr(alg, 'get_diagnostic_info'):
            info = alg.get_diagnostic_info()
            print(f"Configuration: {info}")
        
        # Warmup GPU
        if hasattr(alg, 'device') and alg.device.type == 'cuda':
            print("Warming up GPU...")
            warmup_samples = min(100, num_samples // 10)
            alg.reset()
            for _ in range(warmup_samples):
                alg.step()
            alg.reset()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Benchmark
        print(f"Generating {num_samples} samples...")
        
        if hasattr(alg, 'device') and alg.device.type == 'cuda':
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        cpu_start = time.time()
        chain = alg.generate_samples(num_samples)    
        cpu_end = time.time()
        
        if hasattr(alg, 'device') and alg.device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            gpu_time = start_event.elapsed_time(end_event) / 1000.0
        else:
            gpu_time = cpu_end - cpu_start
        
        cpu_time = cpu_end - cpu_start
        
        # Compute ESJD
        if hasattr(alg, 'expected_squared_jump_distance_gpu'):
            esjd = alg.expected_squared_jump_distance_gpu()
        else:
            esjd = alg.expected_squared_jump_distance()
        
        # Results
        results = {
            'algorithm': algorithm_name,
            'samples_per_sec_cpu': num_samples / cpu_time,
            'samples_per_sec_gpu': num_samples / gpu_time,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'acceptance_rate': alg.acceptance_rate,
            'esjd': esjd,
            'memory_mb': torch.cuda.memory_allocated() / 1e6 if hasattr(alg, 'device') and alg.device.type == 'cuda' else 0
        }
    
    print(f"Results:")
    print(f"  CPU Time: {cpu_time:.3f}s ({results['samples_per_sec_cpu']:.1f} samples/sec)")
    print(f"  GPU Time: {gpu_time:.3f}s ({results['samples_per_sec_gpu']:.1f} samples/sec)")
    print(f"  Acceptance Rate: {results['acceptance_rate']:.3f}")
    print(f"  ESJD: {results['esjd']:.6f}")
    print(f"  GPU Memory: {results['memory_mb']:.1f} MB")
    
    return results, chain

def benchmark_funnel_distributions():
    """Benchmark RWM on new funnel distributions."""
    print("\n" + "=" * 50)
    print("🌪️ New Distributions Benchmark")
    print("=" * 50)
    
    # Test parameters - smaller for challenging distributions
    dim = 10
    num_samples = 5000
    
    distributions_to_test = [
        ("Neal's Funnel", lambda d: NealFunnelTorch(dimension=d), 0.1),  # Small variance for funnel
        ("Multivariate Normal", lambda d: MultivariateNormalTorch(d), 2.38**2 / d),  # Standard optimal variance
    ]
    
    print(f"Configuration:")
    print(f"  Dimension: {dim}")
    print(f"  Number of Samples: {num_samples}")
    print(f"  GPU Available: {torch.cuda.is_available()}")
    
    all_results = []
    
    for dist_name, dist_factory, variance in distributions_to_test:
        print(f"\n--- Benchmarking {dist_name} ---")
        
        try:
            target_dist = dist_factory(dim)
            print(f"Target Distribution: {target_dist.get_name()}")
            print(f"Proposal Variance: {variance:.6f}")
            
            # Test GPU implementations
            implementations = [
                ("GPU Standard", RandomWalkMH_GPU, {"standard_rwm": True}),
                ("GPU Optimized", RandomWalkMH_GPU_Optimized, {"use_efficient_rng": True, "compile_mode": None}),
            ]
            
            dist_results = []
            
            for impl_name, impl_class, kwargs in implementations:
                try:
                    results, chain = benchmark_rwm_implementation(
                        impl_class, f"{impl_name} ({dist_name})", target_dist, 
                        dim, variance, num_samples, **kwargs
                    )
                    dist_results.append(results)
                    
                    # Additional analysis for challenging distributions
                    if 'Funnel' in dist_name:
                        # Analyze burn-in requirements
                        burn_in = 1000
                        post_burnin = chain[burn_in:]
                        
                        # Check v variable (first dimension) for funnel
                        if len(post_burnin) > 0:
                            v_samples = post_burnin[:, 0] if hasattr(post_burnin, '__getitem__') else post_burnin[:, 0]
                            v_mean = torch.mean(v_samples) if isinstance(v_samples, torch.Tensor) else np.mean(v_samples)
                            v_std = torch.std(v_samples) if isinstance(v_samples, torch.Tensor) else np.std(v_samples)
                            
                            print(f"      Post burn-in ({burn_in}) samples: {len(post_burnin)}")
                            print(f"      V variable mean: {v_mean:.3f} (target: ~0)")
                            print(f"      V variable std: {v_std:.3f} (target: ~3)")
                    
                except Exception as e:
                    print(f"Error with {impl_name} on {dist_name}: {e}")
            
            all_results.extend(dist_results)
            
        except Exception as e:
            print(f"Error creating {dist_name}: {e}")
    
    # Summary comparison
    if all_results:
        print(f"\n" + "=" * 50)
        print("📊 New Distributions Summary")
        print("=" * 50)
        print(f"{'Implementation':<40} {'Samples/sec':<12} {'Accept Rate':<12} {'ESJD':<10}")
        print("-" * 80)
        
        for result in all_results:
            print(f"{result['algorithm']:<40} "
                  f"{result['samples_per_sec_gpu']:<12.1f} "
                  f"{result['acceptance_rate']:<12.3f} "
                  f"{result['esjd']:<10.6f}")
    
    return all_results

def main():
    """Run comprehensive benchmarks."""
    print("RWM GPU Optimization Benchmark")
    print("=" * 50)
    
    # Configuration
    dim = 50
    var = 2.38**2 / dim
    num_samples = 100000
    
    print(f"Configuration:")
    print(f"  Dimension: {dim}")
    print(f"  Proposal Variance: {var}")
    print(f"  Number of Samples: {num_samples}")
    print(f"  GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Device: {torch.cuda.get_device_name()}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    all_results = []
    
    # 0. Legacy CPU implementation (baseline)
    try:
        target_dist_cpu = MultivariateNormal(dim)
        print(f"  CPU Target Distribution: {target_dist_cpu.get_name()}")
        
        results, chain = benchmark_rwm_implementation(
            RandomWalkMH, "Legacy CPU RWM", target_dist_cpu, dim, var, num_samples,
            use_cpu_wrapper=True
        )
        all_results.append(results)
    except Exception as e:
        print(f"Error with legacy CPU RWM: {e}")
    
    # Create GPU target distribution for GPU tests
    target_dist = MultivariateNormalTorch(dim)
    print(f"  GPU Target Distribution: {target_dist.get_name()}")
    
    # 1. Original GPU implementation
    try:
        results, chain = benchmark_rwm_implementation(
            RandomWalkMH_GPU, "Original GPU RWM", target_dist, dim, var, num_samples,
            standard_rwm=True
        )
        all_results.append(results)
    except Exception as e:
        print(f"Error with original GPU RWM: {e}")
    
    # 2. Optimized GPU implementation - Basic
    try:
        results, chain = benchmark_rwm_implementation(
            RandomWalkMH_GPU_Optimized, "Optimized GPU RWM (Basic)", target_dist, dim, var, num_samples,
            use_efficient_rng=True,
            compile_mode=None
        )
        all_results.append(results)
    except Exception as e:
        print(f"Error with optimized GPU RWM (basic): {e}")
    
    # 3. Optimized GPU implementation - Full Optimization
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        try:
            results, chain = benchmark_rwm_implementation(
                RandomWalkMH_GPU_Optimized, "Optimized GPU RWM (Full)", target_dist, dim, var, num_samples,
                use_efficient_rng=True,
                compile_mode="default"
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error with optimized GPU RWM (full): {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    
    if all_results:
        baseline = all_results[0]
        print(f"{'Algorithm':<30} {'Samples/sec':<12} {'Speedup':<8} {'Accept Rate':<12} {'ESJD':<10}")
        print("-" * 80)
        
        for result in all_results:
            speedup = result['samples_per_sec_gpu'] / baseline['samples_per_sec_gpu']
            print(f"{result['algorithm']:<30} {result['samples_per_sec_gpu']:<12.1f} {speedup:<8.2f}x "
                  f"{result['acceptance_rate']:<12.3f} {result['esjd']:<10.6f}")
    
    try:
        new_dist_results = benchmark_funnel_distributions()
    except Exception as e:
        print(f"Error with new distributions benchmark: {e}")

if __name__ == "__main__":
    main() 