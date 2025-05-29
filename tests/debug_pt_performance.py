#!/usr/bin/env python3
"""
Performance degradation debugging script for GPU Parallel Tempering MCMC.

This script systematically tests different potential causes of progressive
performance slowdown in the GPU PT implementation.
"""

import torch
import time
import gc
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np
from algorithms.pt_rwm_gpu_optimized import ParallelTemperingRWM_GPU_Optimized
from target_distributions import MultivariateNormalTorch

def log_memory_stats(step, message=""):
    """Log detailed memory statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e6  # MB
        reserved = torch.cuda.memory_reserved() / 1e6    # MB
        max_allocated = torch.cuda.max_memory_allocated() / 1e6  # MB
        print(f"Step {step:5d} | {message:20s} | GPU Alloc: {allocated:6.1f}MB | Reserved: {reserved:6.1f}MB | Max: {max_allocated:6.1f}MB")
    
    # CPU memory
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1e6  # MB
    print(f"          | CPU Memory: {cpu_mem:6.1f}MB")

def test_basic_performance_degradation():
    """Test 1: Basic performance degradation measurement."""
    print("\n" + "="*80)
    print("TEST 1: BASIC PERFORMANCE DEGRADATION MEASUREMENT")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dim = 30
    target_dist = MultivariateNormalTorch(dim, device=device)
    
    # Create PT algorithm
    pt = ParallelTemperingRWM_GPU_Optimized(
        dim=dim,
        var=(2.38**2)/dim,
        target_dist=target_dist,
        device=device,
        pre_allocate_steps=20000,
        swap_every=100,
        geom_temp_spacing=True
    )
    
    # Measure performance in chunks
    chunk_size = 1000
    num_chunks = 20
    rates = []
    times = []
    
    print(f"\nMeasuring performance over {num_chunks} chunks of {chunk_size} samples each")
    print(f"Expected to see degradation if issue exists...")
    
    for i in range(num_chunks):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        # Run chunk
        for _ in range(chunk_size):
            pt.step()
        
        chunk_time = time.time() - start_time
        rate = chunk_size / chunk_time
        
        rates.append(rate)
        times.append(i * chunk_size)
        
        # Log memory every 5 chunks
        if i % 5 == 0:
            log_memory_stats(i * chunk_size, f"Chunk {i}")
        
        print(f"Chunk {i:2d}: {rate:6.1f} samples/sec (Time: {chunk_time:.3f}s)")
    
    # Analysis
    initial_rate = rates[0]
    final_rate = rates[-1]
    degradation = (initial_rate - final_rate) / initial_rate * 100
    
    print(f"\nResults:")
    print(f"  Initial rate: {initial_rate:.1f} samples/sec")
    print(f"  Final rate:   {final_rate:.1f} samples/sec")
    print(f"  Degradation:  {degradation:.1f}%")
    
    return rates, times

def test_memory_leak_detection():
    """Test 2: Memory leak detection - Focus on specific operations."""
    print("\n" + "="*80)
    print("TEST 2: MEMORY LEAK DETECTION")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 30
    target_dist = MultivariateNormalTorch(dim, device=device)
    
    pt = ParallelTemperingRWM_GPU_Optimized(
        dim=dim,
        var=(2.38**2)/dim,
        target_dist=target_dist,
        device=device,
        pre_allocate_steps=None,  # Test without pre-allocation
        swap_every=100,
        geom_temp_spacing=True
    )
    
    print("Testing individual operations for memory leaks...")
    
    # Test 1: Proposal generation
    print("\nTesting proposal generation...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    initial_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    for i in range(1000):
        increments = pt._generate_all_increments()
        if i % 200 == 0:
            current_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            print(f"  Iteration {i}: Memory: {current_mem/1e6:.1f}MB (Δ{(current_mem-initial_mem)/1e6:.1f}MB)")
    
    # Test 2: Density computation
    print("\nTesting density computation...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    initial_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    for i in range(1000):
        proposals = torch.randn(pt.num_chains, dim, device=device)
        log_densities = pt._compute_log_densities_for_proposals(proposals)
        if i % 200 == 0:
            current_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            print(f"  Iteration {i}: Memory: {current_mem/1e6:.1f}MB (Δ{(current_mem-initial_mem)/1e6:.1f}MB)")
    
    # Test 3: Swap operations
    print("\nTesting swap operations...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    initial_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    for i in range(1000):
        pt._attempt_all_swaps()
        if i % 200 == 0:
            current_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            print(f"  Iteration {i}: Memory: {current_mem/1e6:.1f}MB (Δ{(current_mem-initial_mem)/1e6:.1f}MB)")

def test_tensor_copy_overhead():
    """Test 3: Investigate tensor copy operations."""
    print("\n" + "="*80)
    print("TEST 3: TENSOR COPY OVERHEAD ANALYSIS")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 30
    num_chains = 6
    
    # Create test tensors
    current_states = torch.randn(num_chains, dim, device=device)
    log_densities = torch.randn(num_chains, device=device)
    
    print("Testing different swap implementations...")
    
    # Method 1: Using .clone() (current implementation)
    def swap_with_clone(states, densities, j, k):
        temp_state = states[k].clone()
        states[k] = states[j]
        states[j] = temp_state
        
        temp_density = densities[k].clone()
        densities[k] = densities[j]
        densities[j] = temp_density
        return states, densities
    
    # Method 2: In-place swapping without clone
    def swap_inplace(states, densities, j, k):
        # Direct tensor element swapping
        states[j], states[k] = states[k].clone(), states[j].clone()
        densities[j], densities[k] = densities[k].clone(), densities[j].clone()
        return states, densities
    
    # Method 3: Using torch operations
    def swap_torch_ops(states, densities, j, k):
        # Use indexing operations
        temp_indices = torch.tensor([k, j], device=device)
        states[[j, k]] = states[temp_indices]
        densities[[j, k]] = densities[temp_indices]
        return states, densities
    
    methods = [
        ("Clone Method (Current)", swap_with_clone),
        ("In-place Method", swap_inplace),
        ("Torch Ops Method", swap_torch_ops)
    ]
    
    num_iterations = 10000
    
    for method_name, method_func in methods:
        print(f"\nTesting {method_name}:")
        
        # Reset tensors
        test_states = current_states.clone()
        test_densities = log_densities.clone()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start_time = time.time()
        initial_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        for i in range(num_iterations):
            j, k = i % (num_chains - 1), (i + 1) % num_chains
            try:
                test_states, test_densities = method_func(test_states, test_densities, j, k)
            except Exception as e:
                print(f"  Error: {e}")
                break
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        final_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        elapsed = end_time - start_time
        swaps_per_sec = num_iterations / elapsed if elapsed > 0 else 0
        
        print(f"  Time: {elapsed:.4f}s")
        print(f"  Rate: {swaps_per_sec:.0f} swaps/sec")
        print(f"  Memory change: {(final_mem - initial_mem)/1e6:.1f}MB")

def test_random_number_generation():
    """Test 4: Random number generation patterns."""
    print("\n" + "="*80)
    print("TEST 4: RANDOM NUMBER GENERATION ANALYSIS")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 30
    num_chains = 6
    
    print("Testing different RNG patterns...")
    
    # Method 1: On-demand generation (current for proposals)
    def test_ondemand_rng(iterations):
        times = []
        for i in range(iterations):
            start = time.time()
            raw_increments = torch.randn(num_chains, dim, device=device)
            random_vals = torch.rand(num_chains, device=device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start)
        return times
    
    # Method 2: Pre-allocated random numbers
    def test_prealloc_rng(iterations):
        # Pre-allocate
        all_increments = torch.randn(iterations, num_chains, dim, device=device)
        all_random_vals = torch.rand(iterations, num_chains, device=device)
        
        times = []
        for i in range(iterations):
            start = time.time()
            increments = all_increments[i]
            random_vals = all_random_vals[i]
            times.append(time.time() - start)
        return times
    
    iterations = 1000
    
    print("\nOn-demand RNG:")
    ondemand_times = test_ondemand_rng(iterations)
    print(f"  Mean time per call: {np.mean(ondemand_times)*1000:.3f}ms")
    print(f"  Std time per call: {np.std(ondemand_times)*1000:.3f}ms")
    
    print("\nPre-allocated RNG:")
    prealloc_times = test_prealloc_rng(iterations)
    print(f"  Mean time per call: {np.mean(prealloc_times)*1000:.3f}ms")
    print(f"  Std time per call: {np.std(prealloc_times)*1000:.3f}ms")
    
    # Check if times increase over iterations (indicating degradation)
    print("\nChecking for time degradation...")
    first_half_ondemand = np.mean(ondemand_times[:iterations//2])
    second_half_ondemand = np.mean(ondemand_times[iterations//2:])
    degradation_ondemand = (second_half_ondemand - first_half_ondemand) / first_half_ondemand * 100
    
    print(f"  On-demand RNG degradation: {degradation_ondemand:.2f}%")

def test_cpu_gpu_sync_overhead():
    """Test 5: CPU-GPU synchronization overhead."""
    print("\n" + "="*80)
    print("TEST 5: CPU-GPU SYNCHRONIZATION OVERHEAD")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU sync test")
        return
    
    device = torch.device('cuda')
    dim = 30
    target_dist = MultivariateNormalTorch(dim, device=device)
    
    pt = ParallelTemperingRWM_GPU_Optimized(
        dim=dim,
        var=(2.38**2)/dim,
        target_dist=target_dist,
        device=device,
        pre_allocate_steps=5000,
        swap_every=100,
        geom_temp_spacing=True
    )
    
    print("Testing performance with and without explicit GPU synchronization...")
    
    # Test 1: With frequent synchronization
    print("\nWith frequent sync:")
    start_time = time.time()
    for i in range(1000):
        pt.step()
        if i % 100 == 0:  # Sync every 100 steps
            torch.cuda.synchronize()
    sync_time = time.time() - start_time
    
    # Reset algorithm
    pt.reset()
    
    # Test 2: Without synchronization
    print("Without sync:")
    start_time = time.time()
    for i in range(1000):
        pt.step()
    no_sync_time = time.time() - start_time
    
    torch.cuda.synchronize()  # Final sync
    
    print(f"  With sync: {sync_time:.3f}s ({1000/sync_time:.1f} samples/sec)")
    print(f"  Without sync: {no_sync_time:.3f}s ({1000/no_sync_time:.1f} samples/sec)")
    print(f"  Sync overhead: {(sync_time - no_sync_time)/no_sync_time*100:.1f}%")

def test_thermal_monitoring():
    """Test 6: Monitor GPU temperature and clock speeds."""
    print("\n" + "="*80)
    print("TEST 6: THERMAL AND PERFORMANCE MONITORING")
    print("="*80)
    
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        print("GPU thermal monitoring available")
        
        # Monitor during sustained load
        device = torch.device('cuda')
        dim = 30
        target_dist = MultivariateNormalTorch(dim, device=device)
        
        pt = ParallelTemperingRWM_GPU_Optimized(
            dim=dim,
            var=(2.38**2)/dim,
            target_dist=target_dist,
            device=device,
            pre_allocate_steps=10000,
            swap_every=100,
            geom_temp_spacing=True
        )
        
        temps = []
        clocks = []
        rates = []
        
        chunk_size = 500
        num_chunks = 20
        
        print("Monitoring temperature and performance...")
        
        for i in range(num_chunks):
            start_time = time.time()
            
            for _ in range(chunk_size):
                pt.step()
            
            chunk_time = time.time() - start_time
            rate = chunk_size / chunk_time
            
            # Get thermal data
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            
            temps.append(temp)
            clocks.append(clock)
            rates.append(rate)
            
            if i % 5 == 0:
                print(f"Chunk {i:2d}: {rate:6.1f} samples/sec | Temp: {temp}°C | Clock: {clock}MHz")
        
        # Analysis
        print(f"\nThermal Analysis:")
        print(f"  Temperature range: {min(temps)}°C - {max(temps)}°C")
        print(f"  Clock range: {min(clocks)}MHz - {max(clocks)}MHz")
        print(f"  Performance range: {min(rates):.1f} - {max(rates):.1f} samples/sec")
        
        # Check correlation between temperature and performance
        temp_increase = max(temps) - min(temps)
        perf_decrease = max(rates) - min(rates)
        print(f"  Temperature increase: {temp_increase}°C")
        print(f"  Performance decrease: {perf_decrease:.1f} samples/sec")
        
    except ImportError:
        print("pynvml not available - install with: pip install nvidia-ml-py3")
    except Exception as e:
        print(f"GPU monitoring error: {e}")

def create_performance_visualization(rates, times):
    """Create visualization of performance degradation."""
    plt.figure(figsize=(12, 8))
    
    # Main performance plot
    plt.subplot(2, 2, 1)
    plt.plot(times, rates, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Sample Number')
    plt.ylabel('Sampling Rate (samples/sec)')
    plt.title('Performance Degradation Over Time')
    plt.grid(True, alpha=0.3)
    
    # Calculate trend line
    z = np.polyfit(times, rates, 1)
    p = np.poly1d(z)
    plt.plot(times, p(times), 'r--', alpha=0.8, label=f'Trend: {z[0]:.3f} samples/sec per sample')
    plt.legend()
    
    # Moving average
    plt.subplot(2, 2, 2)
    window_size = 5
    if len(rates) >= window_size:
        moving_avg = np.convolve(rates, np.ones(window_size)/window_size, mode='valid')
        moving_times = times[window_size-1:]
        plt.plot(moving_times, moving_avg, 'g-', linewidth=2)
        plt.xlabel('Sample Number')
        plt.ylabel('Moving Average Rate')
        plt.title(f'Moving Average (window={window_size})')
        plt.grid(True, alpha=0.3)
    
    # Relative performance
    plt.subplot(2, 2, 3)
    relative_perf = np.array(rates) / rates[0] * 100
    plt.plot(times, relative_perf, 'purple', linewidth=2)
    plt.xlabel('Sample Number')
    plt.ylabel('Relative Performance (%)')
    plt.title('Performance Relative to Initial')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.5)
    
    # Distribution of rates
    plt.subplot(2, 2, 4)
    plt.hist(rates, bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Sampling Rate (samples/sec)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sampling Rates')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_degradation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Performance visualization saved as 'performance_degradation_analysis.png'")

def main():
    """Run all diagnostic tests."""
    print("GPU PARALLEL TEMPERING PERFORMANCE DEGRADATION DIAGNOSTIC")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Tests will run on CPU with limited functionality.")
    else:
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    
    # Run all tests
    rates, times = test_basic_performance_degradation()
    test_memory_leak_detection()
    test_tensor_copy_overhead()
    test_random_number_generation()
    test_cpu_gpu_sync_overhead()
    test_thermal_monitoring()
    
    # Create visualization
    create_performance_visualization(rates, times)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nNext steps based on findings:")
    print("1. Check for consistent memory growth (indicates memory leaks)")
    print("2. Compare tensor copy methods for efficiency")
    print("3. Look for thermal throttling if temperature increases significantly")
    print("4. Examine specific operations that show degradation")

if __name__ == "__main__":
    main() 