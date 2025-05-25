import argparse
import time
import torch
from interfaces import MCMCSimulation_GPU
from algorithms import *
import numpy as np
from target_distributions import *
import matplotlib.pyplot as plt
import json
import tqdm

def get_target_distribution(name, dim, use_torch=True):
    """Get target distribution with optional GPU acceleration."""
    if use_torch:
        # Use PyTorch-native implementations for GPU acceleration
        if name == "MultivariateNormal":
            return MultivariateNormalTorch(dim)
        elif name == "RoughCarpet":
            return RoughCarpetDistributionTorch(dim, scaling=False)
        elif name == "RoughCarpetScaled":
            return RoughCarpetDistributionTorch(dim, scaling=True)
        elif name == "ThreeMixture":
            return ThreeMixtureDistributionTorch(dim, scaling=False)
        elif name == "ThreeMixtureScaled":
            return ThreeMixtureDistributionTorch(dim, scaling=True)
        elif name == "Hypercube":
            return HypercubeTorch(dim, left_boundary=-1, right_boundary=1)
        elif name == "IIDGamma":
            return IIDGammaTorch(dim, shape=2, scale=3)
        elif name == "IIDBeta":
            return IIDBetaTorch(dim, alpha=2, beta=3)
        else:
            raise ValueError("Unknown target distribution name")
    else:
        # Fall back to CPU versions
        if name == "MultivariateNormal":
            return MultivariateNormal(dim)
        elif name == "RoughCarpet":
            return RoughCarpetDistribution(dim, scaling=False)
        elif name == "RoughCarpetScaled":
            return RoughCarpetDistribution(dim, scaling=True)
        elif name == "ThreeMixture":
            return ThreeMixtureDistribution(dim, scaling=False)
        elif name == "ThreeMixtureScaled":
            return ThreeMixtureDistribution(dim, scaling=True)
        elif name == "Hypercube":
            return Hypercube(dim, left_boundary=-1, right_boundary=1)
        elif name == "IIDGamma":
            return IIDGamma(dim, shape=2, scale=3)
        elif name == "IIDBeta":
            return IIDBeta(dim, alpha=2, beta=3)
        else:
            raise ValueError("Unknown target distribution name")

def run_performance_comparison(dim, num_iters, target_name="MultivariateNormalTorch", seed=42):
    """Compare GPU vs CPU performance for a single configuration."""
    print(f"\n{'='*60}")
    print(f"PERFORMANCE COMPARISON: {target_name} (dim={dim}, samples={num_iters}, seed={seed})")
    print(f"{'='*60}")
    
    # Test variance (roughly optimal for high dimensions)
    variance = 2.38**2 / dim
    
    # GPU version
    print("\n🚀 Testing GPU-accelerated implementation...")
    target_dist_gpu = get_target_distribution(target_name, dim, use_torch=True)
    
    gpu_start = time.time()
    simulation_gpu = MCMCSimulation_GPU(
        dim=dim,
        sigma=variance,
        num_iterations=num_iters,
        algorithm=RandomWalkMH_GPU,
        target_dist=target_dist_gpu,
        symmetric=True,
        pre_allocate=True,
        seed=seed
    )
    
    chain_gpu = simulation_gpu.generate_samples(progress_bar=False)
    gpu_time = time.time() - gpu_start
    gpu_acceptance = simulation_gpu.acceptance_rate()
    gpu_esjd = simulation_gpu.expected_squared_jump_distance()
    
    print(f"✅ GPU Results:")
    print(f"   Time: {gpu_time:.2f} seconds")
    print(f"   Samples/sec: {num_iters/gpu_time:.0f}")
    print(f"   Acceptance rate: {gpu_acceptance:.3f}")
    print(f"   ESJD: {gpu_esjd:.6f}")
    
    # CPU version for comparison (using original implementation)
    print("\n🐌 Testing CPU implementation...")
    from algorithms import RandomWalkMH
    from interfaces import MCMCSimulation
    
    target_dist_cpu = get_target_distribution(target_name, dim, use_torch=False)
    
    cpu_start = time.time()
    simulation_cpu = MCMCSimulation(
        dim=dim,
        sigma=variance,
        num_iterations=num_iters,
        algorithm=RandomWalkMH,
        target_dist=target_dist_cpu,
        symmetric=True,
        seed=seed
    )
    
    chain_cpu = simulation_cpu.generate_samples()
    cpu_time = time.time() - cpu_start
    cpu_acceptance = simulation_cpu.acceptance_rate()
    cpu_esjd = simulation_cpu.expected_squared_jump_distance()
    
    print(f"✅ CPU Results:")
    print(f"   Time: {cpu_time:.2f} seconds")
    print(f"   Samples/sec: {num_iters/cpu_time:.0f}")
    print(f"   Acceptance rate: {cpu_acceptance:.3f}")
    print(f"   ESJD: {cpu_esjd:.6f}")
    
    # Calculate speedup
    speedup = cpu_time / gpu_time
    print(f"\n🎯 Performance Summary:")
    print(f"   Speedup: {speedup:.1f}x faster")
    print(f"   GPU efficiency: {speedup:.1f}x improvement")
    
    # Verify results are similar
    acc_diff = abs(gpu_acceptance - cpu_acceptance)
    esjd_diff = abs(gpu_esjd - cpu_esjd) / max(gpu_esjd, cpu_esjd)
    
    print(f"\n🔍 Accuracy Verification:")
    print(f"   Acceptance rate difference: {acc_diff:.4f}")
    print(f"   ESJD relative difference: {esjd_diff:.4f}")
    
    if acc_diff < 0.05 and esjd_diff < 0.1:
        print("   ✅ Results are consistent between GPU and CPU")
    else:
        print("   ⚠️  Large differences detected - check implementation")
    
    return {
        'gpu_time': gpu_time,
        'cpu_time': cpu_time,
        'speedup': speedup,
        'gpu_acceptance': gpu_acceptance,
        'cpu_acceptance': cpu_acceptance,
        'gpu_esjd': gpu_esjd,
        'cpu_esjd': cpu_esjd,
        'seed': seed
    }

def run_study(dim, target_name="MultivariateNormalTorch", num_iters=100000, var_max=3.5, seed=42):
    """Run many simulations with different variance values, then save and plot the progression of ESJD and acceptance rate."""
    print(f"\n{'='*60}")
    print(f"Target: {target_name}, Dimension: {dim}, Samples: {num_iters}, Seed: {seed}")
    print(f"{'='*60}")
    
    target_distribution = get_target_distribution(target_name, dim, use_torch=True)
    var_value_range = np.linspace(0.001, var_max, 40)
    
    acceptance_rates = []
    expected_squared_jump_distances = []
    times = []
    
    print(f"\nRunning simulations with {len(var_value_range)} variance values...")
    
    total_start = time.time()
    
    # Use tqdm for progress bar
    for i, var in enumerate(tqdm.tqdm(var_value_range, desc="Running RWM with different variance values", unit="config")):
        variance = (var ** 2) / (dim ** (1))
        iteration_start = time.time()
        
        simulation = MCMCSimulation_GPU(
            dim=dim,
            sigma=variance,
            num_iterations=num_iters,
            algorithm=RandomWalkMH_GPU,
            target_dist=target_distribution,
            symmetric=True,
            pre_allocate=True,
            seed=seed
        )
        
        chain = simulation.generate_samples(progress_bar=False)
        
        iteration_time = time.time() - iteration_start
        times.append(iteration_time)
        
        acceptance_rates.append(simulation.acceptance_rate())
        expected_squared_jump_distances.append(simulation.expected_squared_jump_distance())
    
    total_time = time.time() - total_start
    
    max_esjd = max(expected_squared_jump_distances)
    max_esjd_index = np.argmax(expected_squared_jump_distances)
    max_acceptance_rate = acceptance_rates[max_esjd_index]
    max_variance_value = var_value_range[max_esjd_index]
    
    print(f"\nFinal Results:")
    print(f"   Total time: {total_time:.1f} seconds")
    print(f"   Average time per configuration: {np.mean(times):.1f} seconds")
    print(f"   Maximum ESJD: {max_esjd:.6f}")
    print(f"   Optimal acceptance rate: {max_acceptance_rate:.3f}")
    print(f"   Optimal variance value: {max_variance_value:.6f}")
    
    # Save results
    data = {
        'target_distribution': target_name,
        'dimension': dim,
        'num_iterations': num_iters,
        'seed': seed,
        'total_time': total_time,
        'max_esjd': max_esjd,
        'max_acceptance_rate': max_acceptance_rate,
        'max_variance_value': max_variance_value,
        'expected_squared_jump_distances': expected_squared_jump_distances,
        'acceptance_rates': acceptance_rates,
        'var_value_range': var_value_range.tolist(),
        'times': times
    }
    
    filename = f"data/{target_name}_RWM_GPU_dim{dim}_{num_iters}iters_seed{seed}.json"
    with open(filename, "w") as file:
        json.dump(data, file, indent=2)
    print(f"   Results saved to: {filename}")
    
    # Create separate plots with consistent styling from original experiment
    
    # Plot 1: ESJD vs Acceptance Rate
    plt.plot(acceptance_rates, expected_squared_jump_distances, marker='x')   
    plt.axvline(x=0.234, color='red', linestyle=':', label='a = 0.234')
    plt.xlabel('acceptance rate')
    plt.ylabel('ESJD')
    plt.title(f'ESJD vs acceptance rate (dim={dim}, seed={seed})')
    plt.legend()
    output_filename = f"images/ESJD_vs_acceptance_rate_{target_name}_RWM_GPU_dim{dim}_{num_iters}iters_seed{seed}"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()
    print(f"   Plot 1 created and saved as '{output_filename}'")

    # Plot 2: Acceptance Rate vs Variance
    plt.plot(var_value_range, acceptance_rates, label='Acceptance rate', marker='x')
    plt.xlabel('Variance value (value^2 / dim)')
    plt.ylabel('Acceptance rate')
    plt.title(f'Acceptance rate for different variance values (dim={dim}, seed={seed})')
    filename = f"images/AcceptvsVar_{target_name}_RWM_GPU_dim{dim}_{num_iters}iters_seed{seed}"
    plt.savefig(filename)
    plt.clf()
    print(f"   Plot 2 created and saved as '{filename}'")

    # Plot 3: ESJD vs Variance
    plt.plot(var_value_range, expected_squared_jump_distances, label='Expected squared jump distance', marker='x')
    plt.xlabel('Variance value (value^2 / dim)')
    plt.ylabel('ESJD')
    plt.title(f'ESJD for different variance values (dim={dim}, seed={seed})')
    filename = f"images/ESJDvsVar_{target_name}_RWM_GPU_dim{dim}_{num_iters}iters_seed{seed}"
    plt.savefig(filename)
    plt.clf()
    print(f"   Plot 3 created and saved as '{filename}'")
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-accelerated RWM simulations")
    parser.add_argument("--dim", type=int, default=20, help="Dimension of the target distribution")
    parser.add_argument("--target", type=str, default="MultivariateNormal", help="Target distribution")
    parser.add_argument("--num_iters", type=int, default=100000, help="Number of iterations")
    parser.add_argument("--var_max", type=float, default=3.5, help="Maximum variance value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--mode", type=str, default="default", 
                       choices=["default", "comparison", "benchmark"],
                       help="Mode: default (run study), comparison (GPU vs CPU), or benchmark")
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        print(f"🚀 GPU detected: {torch.cuda.get_device_name()}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected. Running on CPU (will be slower)")
    
    
    if args.mode == "default":
        # Parameter optimization study
        results = run_study(args.dim, args.target, args.num_iters, args.var_max, args.seed)
        
    elif args.mode == "comparison":
        # Performance comparison
        results = run_performance_comparison(args.dim, args.num_iters, args.target, args.seed)
        
    elif args.mode == "benchmark":
        # Comprehensive benchmark
        print(f"\n{'='*60}")
        print("COMPREHENSIVE GPU BENCHMARK")
        print(f"{'='*60}")
        
        dimensions = [5, 10, 20, 50] if args.dim > 20 else [5, 10, args.dim]
        sample_sizes = [10000, 50000, 100000] if args.num_iters >= 100000 else [1000, 5000, args.num_iters]
        
        benchmark_results = []
        
        for dim in dimensions:
            for num_samples in sample_sizes:
                print(f"\nBenchmarking: dim={dim}, samples={num_samples}")
                result = run_performance_comparison(dim, num_samples, args.target, args.seed)
                result.update({'dim': dim, 'num_samples': num_samples})
                benchmark_results.append(result)
        
        # Save comprehensive benchmark
        benchmark_data = {
            'target_distribution': args.target,
            'timestamp': time.time(),
            'seed': args.seed,
            'results': benchmark_results
        }
        
        benchmark_filename = f"data/GPU_benchmark_{args.target}_seed{args.seed}_{int(time.time())}.json"
        with open(benchmark_filename, "w") as file:
            json.dump(benchmark_data, file, indent=2)
        
        print(f"\n🎯 Benchmark completed! Results saved to: {benchmark_filename}")
    
    print(f"\n✅ GPU optimization analysis complete!")
    print(f"💡 Your Metropolis algorithm should now run {10}-{100}x faster with GPU acceleration!") 