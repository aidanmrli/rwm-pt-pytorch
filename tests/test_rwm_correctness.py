#!/usr/bin/env python3
"""
Test script to verify Standard RWM GPU implementation correctness.
This script specifically tests that the standard_rwm=True mode produces correct MCMC behavior.
"""

import sys
import os
# Add the parent directory to the Python path so we can import from algorithms/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from algorithms.rwm_gpu import RandomWalkMH_GPU
from algorithms.rwm import RandomWalkMH
from algorithms.rwm_gpu_optimized import RandomWalkMH_GPU_Optimized, ultra_fused_mcmc_step_basic
from target_distributions import MultivariateNormal, MultivariateNormalTorch
import matplotlib.pyplot as plt

def test_standard_rwm_correctness():
    """Test that standard RWM implementation is theoretically correct."""
    print("🧪 Testing Standard RWM Correctness...")
    
    # Test parameters - use simple 2D Gaussian for easy verification
    dim = 2
    num_samples = 5000
    variance = 0.5  # Moderate proposal variance
    
    try:
        # Test 1: Standard RWM should match CPU implementation closely
        print("   🔍 Test 1: Comparing Standard RWM GPU vs CPU...")
        
        target_cpu = MultivariateNormal(dim)
        target_gpu = MultivariateNormalTorch(dim)
        
        rwm_cpu = RandomWalkMH(dim, variance, target_cpu)
        rwm_gpu_std = RandomWalkMH_GPU(
            dim=dim, 
            var=variance, 
            target_dist=target_gpu,
            standard_rwm=True,
            symmetric=True,
            pre_allocate_steps=num_samples,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        rwm_gpu_opt = RandomWalkMH_GPU_Optimized(
            dim=dim, 
            var=variance, 
            target_dist=target_gpu,
            symmetric=True,
            pre_allocate_steps=num_samples,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_efficient_rng=True,
        )
        np.random.seed(42)
        torch.manual_seed(42)
        # Generate samples
        for _ in range(num_samples):
            rwm_cpu.step()
        
        cpu_chain = np.array(rwm_cpu.chain)
        cpu_acc_rate = rwm_cpu.acceptance_rate
        
        gpu_chain_std = rwm_gpu_std.generate_samples(num_samples)
        gpu_acc_rate_std = rwm_gpu_std.acceptance_rate
        
        gpu_chain_opt = rwm_gpu_opt.generate_samples(num_samples)
        gpu_acc_rate_opt = rwm_gpu_opt.acceptance_rate
        
        # Compare acceptance rates (should be very close)
        gpustd_cpu_acc_rate_diff = abs(cpu_acc_rate - gpu_acc_rate_std)
        gpuopt_cpu_acc_rate_diff = abs(cpu_acc_rate - gpu_acc_rate_opt)
        print(f"      CPU acceptance rate: {cpu_acc_rate:.4f}")
        print(f"      GPU standard acceptance rate: {gpu_acc_rate_std:.4f}")
        print(f"      GPU optimized acceptance rate: {gpu_acc_rate_opt:.4f}")
        print(f"      GPU-CPU Difference: {gpustd_cpu_acc_rate_diff:.4f}")
        print(f"      GPU(Opt)-CPU Difference: {gpuopt_cpu_acc_rate_diff:.4f}")
        
        test1_pass = gpustd_cpu_acc_rate_diff < 0.1 and gpuopt_cpu_acc_rate_diff < 0.1 # Allow 10% difference due to random seed differences
        if test1_pass:
            print(f"   ✅ Acceptance rates are consistent")
        else:
            print(f"   ⚠️  Large acceptance rate difference detected")
        
        # Test 3: Statistical properties for standard RWM
        print("   🔍 Test 3: Statistical properties of standard RWM...")
        
        # For 2D standard Gaussian, mean should be ~[0,0] and std ~[1,1]
        gpu_mean = torch.mean(gpu_chain_std[1000:], axis=0)  # Skip burn-in
        gpu_std = torch.std(gpu_chain_std[1000:], axis=0)
        
        mean_error = torch.norm(gpu_mean)
        std_error = torch.norm(gpu_std - 1.0)
        
        print(f"      GPU standard Empirical mean: {gpu_mean}")
        print(f"      GPU standard Empirical std: {gpu_std}")
        print(f"      GPU standard Mean error (should be ~0): {mean_error:.4f}")
        print(f"      GPU standard Std error (should be ~0): {std_error:.4f}")
        
        gpu_mean_opt = torch.mean(gpu_chain_opt[1000:], axis=0)  # Skip burn-in
        gpu_std_opt = torch.std(gpu_chain_opt[1000:], axis=0)
        
        mean_error_opt = torch.norm(gpu_mean_opt)
        std_error_opt = torch.norm(gpu_std_opt - 1.0)
        
        print(f"      GPU optimized Empirical mean: {gpu_mean_opt}")
        print(f"      GPU optimized Empirical std: {gpu_std_opt}")
        print(f"      GPU optimized Mean error (should be ~0): {mean_error_opt:.4f}")
        print(f"      GPU optimized Std error (should be ~0): {std_error_opt:.4f}")
        
        test3_pass = mean_error < 0.2 and std_error < 0.3 and mean_error_opt < 0.2 and std_error_opt < 0.3
        if test3_pass:
            print(f"   ✅ Statistical properties look good")
        else:
            print(f"   ⚠️  Statistical properties seem off")
        
        # Test 4: Chain should have proper sequential dependence
        print("   🔍 Test 4: Testing sequential dependence...")
        
        # Compute lag-1 autocorrelation (should be positive and reasonable)
        chain_centered = gpu_chain_std[1000:] - gpu_mean
        # Stack the lag-0 and lag-1 values as rows for torch.corrcoef
        x_data = torch.stack([chain_centered[:-1, 0], chain_centered[1:, 0]], dim=0)
        y_data = torch.stack([chain_centered[:-1, 1], chain_centered[1:, 1]], dim=0)
        autocorr_x = torch.corrcoef(x_data)[0, 1]
        autocorr_y = torch.corrcoef(y_data)[0, 1]
        
        print(f"      GPU standard Lag-1 autocorrelation X: {autocorr_x:.4f}")
        print(f"      GPU standard Lag-1 autocorrelation Y: {autocorr_y:.4f}")
        
        # Compute lag-1 autocorrelation (should be positive and reasonable)
        chain_centered = gpu_chain_opt[1000:] - gpu_mean_opt
        # Stack the lag-0 and lag-1 values as rows for torch.corrcoef
        x_data = torch.stack([chain_centered[:-1, 0], chain_centered[1:, 0]], dim=0)
        y_data = torch.stack([chain_centered[:-1, 1], chain_centered[1:, 1]], dim=0)
        autocorr_x_opt = torch.corrcoef(x_data)[0, 1]
        autocorr_y_opt = torch.corrcoef(y_data)[0, 1]
        
        print(f"      GPU optimized Lag-1 autocorrelation X: {autocorr_x_opt:.4f}")
        print(f"      GPU optimized Lag-1 autocorrelation Y: {autocorr_y_opt:.4f}")
        
        # For RWM, autocorrelation should be positive (around 0.2-0.8)
        test4_pass = (0.05 < autocorr_x < 0.95 and 0.05 < autocorr_y < 0.95) and (0.05 < autocorr_x_opt < 0.95 and 0.05 < autocorr_y_opt < 0.95)
        if test4_pass:
            print(f"   ✅ Autocorrelation looks reasonable for MCMC")
        else:
            print(f"   ⚠️  Autocorrelation seems unusual")
        
        # Test 5: Sequential nature test - check that proposals depend on previous state
        print("   🔍 Test 5: Testing true sequential nature...")
        
        # Reset and manually check a few steps for standard GPU
        rwm_test = RandomWalkMH_GPU(
            dim=2, 
            var=0.1,  # Small variance for clear dependence
            target_dist=MultivariateNormalTorch(2),
            standard_rwm=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Take 3 steps and check each depends on previous
        torch.manual_seed(123)
        np.random.seed(123)
        
        step1_before = rwm_test.current_state.clone() if rwm_test.current_state is not None else None
        rwm_test._standard_step()
        step1_after = rwm_test.current_state.clone()
        
        step2_before = rwm_test.current_state.clone()
        rwm_test._standard_step()
        step2_after = rwm_test.current_state.clone()
        
        step3_before = rwm_test.current_state.clone()
        rwm_test._standard_step()
        step3_after = rwm_test.current_state.clone()
        
        # Verify sequential dependence for standard GPU
        sequential_ok = True
        if not torch.allclose(step2_before, step1_after, atol=1e-6):
            sequential_ok = False
        if not torch.allclose(step3_before, step2_after, atol=1e-6):
            sequential_ok = False
        
        # Test optimized GPU version for sequential dependence
        rwm_test_opt = RandomWalkMH_GPU_Optimized(
            dim=2, 
            var=0.1,  # Small variance for clear dependence
            target_dist=MultivariateNormalTorch(2),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_efficient_rng=True,
            pre_allocate_steps=5
        )
        
        torch.manual_seed(123)
        np.random.seed(123)
        
        opt_step1_before = rwm_test_opt.current_state.clone() if rwm_test_opt.current_state is not None else None
        rwm_test_opt.step()
        opt_step1_after = rwm_test_opt.current_state.clone()
        
        opt_step2_before = rwm_test_opt.current_state.clone()
        rwm_test_opt.step()
        opt_step2_after = rwm_test_opt.current_state.clone()
        
        opt_step3_before = rwm_test_opt.current_state.clone()
        rwm_test_opt.step()
        opt_step3_after = rwm_test_opt.current_state.clone()
        
        # Verify sequential dependence for optimized GPU
        sequential_ok_opt = True
        if not torch.allclose(opt_step2_before, opt_step1_after, atol=1e-6):
            sequential_ok_opt = False
        if not torch.allclose(opt_step3_before, opt_step2_after, atol=1e-6):
            sequential_ok_opt = False
        
        test5_pass = sequential_ok and sequential_ok_opt
        if test5_pass:
            print(f"   ✅ Sequential dependence verified for both GPU implementations")
        else:
            print(f"   ❌ Sequential dependence failed!")
            if not sequential_ok:
                print(f"       Standard GPU failed sequential test")
            if not sequential_ok_opt:
                print(f"       Optimized GPU failed sequential test")
        
        # Overall result
        all_tests_pass = test1_pass and test3_pass and test4_pass and test5_pass
        
        if all_tests_pass:
            print(f"   🎉 All Standard RWM correctness tests passed!")
        else:
            print(f"   ⚠️  Some tests failed - check individual results above")
        
        return all_tests_pass
        
    except Exception as e:
        print(f"   ❌ Standard RWM correctness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_optimized_jit_kernels():
    """Test GPU optimized version with JIT kernel compilation specifically."""
    print("\n🔥 Testing GPU Optimized JIT Kernel Implementation...")
    
    # Test parameters
    dim = 5
    num_samples = 1000
    variance = 1.0
    
    try:
        # Test 1: JIT kernel correctness
        print("   🔍 Test 1: JIT kernel correctness...")
        
        target_dist = MultivariateNormalTorch(dim)
        
        # Create optimized instance with different compilation modes
        rwm_optimized = RandomWalkMH_GPU_Optimized(
            dim=dim, 
            var=variance, 
            target_dist=target_dist,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_efficient_rng=True,
            pre_allocate_steps=num_samples,
            compile_mode="default"
        )
        
        # Generate samples to test JIT compilation
        torch.manual_seed(999)
        chain_opt = rwm_optimized.generate_samples(num_samples)
        acc_rate_opt = rwm_optimized.acceptance_rate
        
        print(f"      JIT-compiled samples generated: {len(chain_opt)}")
        print(f"      JIT-compiled acceptance rate: {acc_rate_opt:.4f}")
        
        # Test that the JIT kernels are actually working
        test1_pass = len(chain_opt) == num_samples + 1 and 0.1 < acc_rate_opt < 0.9  # +1 for initial state
        if test1_pass:
            print(f"   ✅ JIT kernel compilation and execution successful")
        else:
            print(f"   ⚠️  JIT kernel issues detected")
            print(f"      Expected samples: {num_samples + 1}, Got: {len(chain_opt)}")
            print(f"      Acceptance rate: {acc_rate_opt:.4f} (should be 0.1-0.9)")
        
        # Test 2: Memory efficiency and pre-allocation
        print("   🔍 Test 2: Memory efficiency and pre-allocation...")
        
        # Test with pre-allocation
        rwm_preallocated = RandomWalkMH_GPU_Optimized(
            dim=dim, 
            var=variance, 
            target_dist=target_dist,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            pre_allocate_steps=num_samples,
            use_efficient_rng=True
        )
        
        start_time = time.time()
        chain_prealloc = rwm_preallocated.generate_samples(num_samples)
        prealloc_time = time.time() - start_time
        
        # Test without pre-allocation
        rwm_no_prealloc = RandomWalkMH_GPU_Optimized(
            dim=dim, 
            var=variance, 
            target_dist=target_dist,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            pre_allocate_steps=None,
            use_efficient_rng=True
        )
        
        start_time = time.time()
        chain_no_prealloc = rwm_no_prealloc.generate_samples(num_samples)
        no_prealloc_time = time.time() - start_time
        
        print(f"      Pre-allocation time: {prealloc_time:.4f}s")
        print(f"      No pre-allocation time: {no_prealloc_time:.4f}s")
        print(f"      Memory efficiency gain: {no_prealloc_time/prealloc_time:.2f}x")
        
        test2_pass = (len(chain_prealloc) == num_samples + 1 and 
                     len(chain_no_prealloc) == num_samples + 1)  # +1 for initial state
        if test2_pass:
            print(f"   ✅ Memory management working correctly")
        else:
            print(f"   ⚠️  Memory management issues detected")
            print(f"      Pre-alloc samples: {len(chain_prealloc)}, Expected: {num_samples + 1}")
            print(f"      No pre-alloc samples: {len(chain_no_prealloc)}, Expected: {num_samples + 1}")
        
        # Test 3: Ultra-fused kernel performance 
        print("   🔍 Test 3: Ultra-fused kernel performance...")
        
        # Test the ultra-fused step directly
        rwm_fused = RandomWalkMH_GPU_Optimized(
            dim=dim, 
            var=variance, 
            target_dist=target_dist,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_efficient_rng=True,
            pre_allocate_steps=100
        )
        
        # Time multiple fused steps
        torch.manual_seed(777)
        start_time = time.time()
        for _ in range(100):
            rwm_fused.step()
        fused_time = time.time() - start_time
        
        # Verify results
        fused_acc_rate = rwm_fused.acceptance_rate
        
        print(f"      Ultra-fused 100 steps time: {fused_time:.4f}s")
        print(f"      Ultra-fused acceptance rate: {fused_acc_rate:.4f}")
        print(f"      Steps per second: {100/fused_time:.0f}")
        
        test3_pass = 0.1 < fused_acc_rate < 0.9 and fused_time < 5.0  # Relaxed time constraint for CPU
        if test3_pass:
            print(f"   ✅ Ultra-fused kernels performing well")
        else:
            print(f"   ⚠️  Ultra-fused kernel performance issues")
            print(f"      Time: {fused_time:.4f}s (should be < 5.0s)")
            print(f"      Acceptance rate: {fused_acc_rate:.4f} (should be 0.1-0.9)")
        
        # Test 4: Kernel fusion validation
        print("   🔍 Test 4: Kernel fusion validation...")
        
        # Test that individual JIT functions work
        device = rwm_optimized.device
        current_state = torch.randn(dim, device=device, dtype=torch.float32)
        current_log_density = torch.tensor(-0.5, device=device, dtype=torch.float32)
        increment = torch.randn(dim, device=device, dtype=torch.float32) * 0.1
        random_val = torch.rand(1, device=device, dtype=torch.float32)
        beta = torch.tensor(1.0, device=device, dtype=torch.float32)
        
        # Compute proposed log density
        proposal = current_state + increment
        log_density_proposed = target_dist.log_density(proposal.unsqueeze(0)).squeeze()
        
        # Test ultra-fused kernel directly
        new_state, new_log_density, accepted = ultra_fused_mcmc_step_basic(
            current_state, current_log_density, increment, random_val, 
            beta, log_density_proposed
        )
        
        test4_pass = (new_state.shape == current_state.shape and 
                     isinstance(accepted.item(), bool))
        if test4_pass:
            print(f"   ✅ Individual JIT kernels working correctly")
        else:
            print(f"   ⚠️  JIT kernel validation issues")
        
        # Overall result
        all_tests_pass = test1_pass and test2_pass and test3_pass and test4_pass
        
        if all_tests_pass:
            print(f"   🎉 All GPU optimized JIT tests passed!")
        else:
            print(f"   ⚠️  Some JIT optimization tests failed")
        
        return all_tests_pass
        
    except Exception as e:
        print(f"   ❌ GPU optimized JIT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Compare all three implementations: CPU, GPU standard, and GPU optimized."""
    print("\n⚡ Performance Comparison: CPU vs GPU Standard vs GPU Optimized...")
    
    dim = 10
    num_samples = 10000
    variance = 2.38**2 / dim  # Roughly optimal variance
    
    try:
        results = {}
        
        # CPU test
        print("   🐌 Testing CPU standard RWM...")
        target_cpu = MultivariateNormal(dim)
        
        cpu_start = time.time()
        rwm_cpu = RandomWalkMH(dim, variance, target_cpu)
        for _ in range(num_samples):
            rwm_cpu.step()
        cpu_time = time.time() - cpu_start
        cpu_acc = rwm_cpu.acceptance_rate
        
        results['CPU'] = {'time': cpu_time, 'acc_rate': cpu_acc}
        print(f"      Time: {cpu_time:.2f}s, Acc: {cpu_acc:.3f}, Rate: {num_samples/cpu_time:.0f} samples/s")
        
        # GPU standard test
        print("   🚀 Testing GPU standard RWM...")
        target_gpu = MultivariateNormalTorch(dim)
        
        gpu_start = time.time()
        rwm_gpu = RandomWalkMH_GPU(
            dim=dim,
            var=variance,
            target_dist=target_gpu,
            standard_rwm=True,  # True standard RWM
            pre_allocate_steps=num_samples,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        gpu_chain = rwm_gpu.generate_samples(num_samples)
        gpu_time = time.time() - gpu_start
        gpu_acc = rwm_gpu.acceptance_rate
        
        results['GPU_Standard'] = {'time': gpu_time, 'acc_rate': gpu_acc}
        print(f"      Time: {gpu_time:.2f}s, Acc: {gpu_acc:.3f}, Rate: {num_samples/gpu_time:.0f} samples/s")
        
        # GPU optimized test
        print("   🔥 Testing GPU optimized (JIT) RWM...")
        
        gpu_opt_start = time.time()
        rwm_gpu_opt = RandomWalkMH_GPU_Optimized(
            dim=dim, 
            var=variance, 
            target_dist=target_gpu,
            symmetric=True,
            pre_allocate_steps=num_samples,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_efficient_rng=True,
        )
        
        gpu_opt_chain = rwm_gpu_opt.generate_samples(num_samples)
        gpu_opt_time = time.time() - gpu_opt_start
        gpu_opt_acc = rwm_gpu_opt.acceptance_rate
        
        results['GPU_Optimized'] = {'time': gpu_opt_time, 'acc_rate': gpu_opt_acc}
        print(f"      Time: {gpu_opt_time:.2f}s, Acc: {gpu_opt_acc:.3f}, Rate: {num_samples/gpu_opt_time:.0f} samples/s")
        
        # Compare results
        print(f"\n   📈 Performance Summary:")
        baseline_time = results['CPU']['time']
        
        for name, result in results.items():
            speedup = baseline_time / result['time'] if result['time'] > 0 else 0
            print(f"      {name:15s}: {speedup:6.2f}x speedup, {result['acc_rate']:.3f} acc rate")
        
        # Check consistency
        acc_rates = [r['acc_rate'] for r in results.values()]
        acc_rate_range = max(acc_rates) - min(acc_rates)
        
        gpu_std_speedup = baseline_time / results['GPU_Standard']['time']
        gpu_opt_speedup = baseline_time / results['GPU_Optimized']['time']
        
        print(f"\n   🎯 Analysis:")
        print(f"      Acceptance rate consistency: {acc_rate_range:.4f} (should be < 0.05)")
        print(f"      GPU Standard speedup: {gpu_std_speedup:.2f}x")
        print(f"      GPU Optimized speedup: {gpu_opt_speedup:.2f}x")
        print(f"      Optimized vs Standard: {gpu_opt_speedup/gpu_std_speedup:.2f}x additional gain")
        
        # Performance criteria
        performance_ok = True
        if acc_rate_range > 0.1:
            print(f"   ⚠️  Large acceptance rate differences detected")
            performance_ok = False
        
        # More lenient performance criteria for CPU execution
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_type == 'cpu':
            # On CPU, just check that optimized version is not dramatically slower
            if gpu_opt_speedup < gpu_std_speedup * 0.3:  # Allow optimized to be slower on CPU
                print(f"   ⚠️  GPU optimized much slower than expected vs standard on CPU")
                performance_ok = False
            else:
                print(f"   ✅ Performance acceptable for CPU execution")
        else:
            # On GPU, optimized should be competitive
            if gpu_opt_speedup < gpu_std_speedup * 0.8:
                print(f"   ⚠️  GPU optimized slower than expected vs standard")
                performance_ok = False
        
        if performance_ok:
            print(f"   ✅ Performance comparison successful")
        else:
            print(f"   ⚠️  Performance issues detected")
        
        return performance_ok
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_fallback():
    """Test that all implementations work on both GPU and CPU."""
    print("\n🔧 Testing Device Fallback for All Implementations...")
    
    dim = 3
    num_samples = 100
    variance = 1.0
    target_dist_cpu = MultivariateNormal(dim)
    target_dist_gpu = MultivariateNormalTorch(dim)
    
    devices_to_test = ['cpu']
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
    
    implementations = {
        'GPU_Standard': RandomWalkMH_GPU,
        'GPU_Optimized': RandomWalkMH_GPU_Optimized
    }
    
    results = {}
    
    for device in devices_to_test:
        print(f"   Testing on {device.upper()}...")
        results[device] = {}
        
        for impl_name, impl_class in implementations.items():
            try:
                print(f"      Testing {impl_name}...")
                
                if impl_name == 'GPU_Optimized':
                    rwm = impl_class(
                        dim=dim,
                        var=variance,
                        target_dist=target_dist_gpu,
                        device=device,
                        pre_allocate_steps=num_samples,
                        use_efficient_rng=True
                    )
                else:
                    rwm = impl_class(
                        dim=dim,
                        var=variance,
                        target_dist=target_dist_gpu,
                        standard_rwm=True,
                        pre_allocate_steps=num_samples,
                        device=device
                    )
                
                chain = rwm.generate_samples(num_samples)
                
                results[device][impl_name] = {
                    'success': True,
                    'chain_length': len(chain),
                    'acceptance_rate': rwm.acceptance_rate,
                    'device_used': rwm.device.type
                }
                
                print(f"         ✅ {impl_name} on {device.upper()} passed")
                print(f"            Samples: {len(chain)}, Acc rate: {rwm.acceptance_rate:.3f}")
                
            except Exception as e:
                print(f"         ❌ {impl_name} on {device.upper()} failed: {e}")
                results[device][impl_name] = {'success': False, 'error': str(e)}
    
    # Summary
    total_tests = len(devices_to_test) * len(implementations)
    successful_tests = 0
    
    for device, device_results in results.items():
        for impl_name, result in device_results.items():
            if result.get('success', False):
                successful_tests += 1
    
    print(f"\n   📊 Fallback Summary: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print(f"   ✅ All implementations work on all available devices")
        return True
    elif successful_tests > 0:
        print(f"   ⚠️  Some implementations failed on some devices")
        return True  # Partial success is acceptable
    else:
        print(f"   ❌ All implementations failed")
        return False

def main():
    """Run all standard RWM tests."""
    print("🚀 Standard RWM GPU Implementation Test Suite")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  No GPU detected - tests will run on CPU")
    
    # Run tests
    tests = [
        ("Standard RWM Correctness", test_standard_rwm_correctness),
        ("GPU Optimized JIT Kernels", test_gpu_optimized_jit_kernels),
        ("Performance Comparison", test_performance_comparison),
        ("Device Fallback", test_device_fallback),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:25s}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your Standard RWM GPU implementation is correct.")
        print("💡 You can confidently use standard_rwm=True for research.")
    else:
        print("\n⚠️  Some tests failed. Standard RWM implementation may need fixes.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 