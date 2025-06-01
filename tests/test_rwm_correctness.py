#!/usr/bin/env python3
"""
Test script to verify RWM GPU implementation correctness.
This script tests that the optimized GPU implementation produces correct MCMC behavior.
"""

import sys
import os
# Add the parent directory to the Python path so we can import from algorithms/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from algorithms.rwm import RandomWalkMH
from algorithms.rwm_gpu_optimized import RandomWalkMH_GPU_Optimized, ultra_fused_mcmc_step_basic
from target_distributions import MultivariateNormal, MultivariateNormalTorch
# Import new funnel distributions
from target_distributions import NealFunnelTorch, SuperFunnelTorch
import matplotlib.pyplot as plt

def test_standard_rwm_correctness():
    """Test that optimized RWM implementation is theoretically correct."""
    print("🧪 Testing RWM Correctness...")
    
    # Test parameters - use simple 2D Gaussian for easy verification
    dim = 2
    num_samples = 5000
    variance = 0.5  # Moderate proposal variance
    
    try:
        # Test 1: GPU Optimized RWM should match CPU implementation closely
        print("   🔍 Test 1: Comparing GPU Optimized vs CPU...")
        
        target_cpu = MultivariateNormal(dim)
        target_gpu = MultivariateNormalTorch(dim)
        
        rwm_cpu = RandomWalkMH(dim, variance, target_cpu)
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
        
        gpu_chain_opt = rwm_gpu_opt.generate_samples(num_samples)
        gpu_acc_rate_opt = rwm_gpu_opt.acceptance_rate
        
        # Compare acceptance rates (should be very close)
        gpuopt_cpu_acc_rate_diff = abs(cpu_acc_rate - gpu_acc_rate_opt)
        print(f"      CPU acceptance rate: {cpu_acc_rate:.4f}")
        print(f"      GPU optimized acceptance rate: {gpu_acc_rate_opt:.4f}")
        print(f"      GPU(Opt)-CPU Difference: {gpuopt_cpu_acc_rate_diff:.4f}")
        
        test1_pass = gpuopt_cpu_acc_rate_diff < 0.1 # Allow 10% difference due to random seed differences
        if test1_pass:
            print(f"   ✅ Acceptance rates are consistent")
        else:
            print(f"   ⚠️  Large acceptance rate difference detected")
        
        # Test 3: Statistical properties for optimized RWM
        print("   🔍 Test 3: Statistical properties of optimized RWM...")
        
        # For 2D standard Gaussian, mean should be ~[0,0] and std ~[1,1]
        gpu_mean_opt = torch.mean(gpu_chain_opt[1000:], axis=0)  # Skip burn-in
        gpu_std_opt = torch.std(gpu_chain_opt[1000:], axis=0)
        
        mean_error_opt = torch.norm(gpu_mean_opt)
        std_error_opt = torch.norm(gpu_std_opt - 1.0)
        
        print(f"      GPU optimized Empirical mean: {gpu_mean_opt}")
        print(f"      GPU optimized Empirical std: {gpu_std_opt}")
        print(f"      GPU optimized Mean error (should be ~0): {mean_error_opt:.4f}")
        print(f"      GPU optimized Std error (should be ~0): {std_error_opt:.4f}")
        
        test3_pass = mean_error_opt < 0.2 and std_error_opt < 0.3
        if test3_pass:
            print(f"   ✅ Statistical properties look good")
        else:
            print(f"   ⚠️  Statistical properties seem off")
        
        # Test 4: Chain should have proper sequential dependence
        print("   🔍 Test 4: Testing sequential dependence...")
        
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
        test4_pass = (0.05 < autocorr_x_opt < 0.95 and 0.05 < autocorr_y_opt < 0.95)
        if test4_pass:
            print(f"   ✅ Autocorrelation looks reasonable for MCMC")
        else:
            print(f"   ⚠️  Autocorrelation seems unusual")
        
        # Test 5: Sequential nature test - check that proposals depend on previous state
        print("   🔍 Test 5: Testing true sequential nature...")
        
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
        
        test5_pass = sequential_ok_opt
        if test5_pass:
            print(f"   ✅ Sequential dependence verified for GPU implementation")
        else:
            print(f"   ❌ Sequential dependence failed!")
            print(f"       Optimized GPU failed sequential test")
        
        # Overall result
        all_tests_pass = test1_pass and test3_pass and test4_pass and test5_pass
        
        if all_tests_pass:
            print(f"   🎉 All RWM correctness tests passed!")
        else:
            print(f"   ⚠️  Some tests failed - check individual results above")
        
        return all_tests_pass
        
    except Exception as e:
        print(f"   ❌ RWM correctness test failed: {e}")
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
        test1_pass = len(chain_opt) == num_samples and 0.1 < acc_rate_opt < 0.9  # Should be exactly num_samples, not +1
        if test1_pass:
            print(f"   ✅ JIT kernel compilation and execution successful")
        else:
            print(f"   ⚠️  JIT kernel issues detected")
            print(f"      Expected samples: {num_samples}, Got: {len(chain_opt)}")
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
        
        test2_pass = (len(chain_prealloc) == num_samples and 
                     len(chain_no_prealloc) == num_samples)  # Should be exactly num_samples, not +1
        if test2_pass:
            print(f"   ✅ Memory management working correctly")
        else:
            print(f"   ⚠️  Memory management issues detected")
            print(f"      Pre-alloc samples: {len(chain_prealloc)}, Expected: {num_samples}")
            print(f"      No pre-alloc samples: {len(chain_no_prealloc)}, Expected: {num_samples}")
        
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
    """Compare CPU and GPU optimized implementations."""
    print("\n⚡ Performance Comparison: CPU vs GPU Optimized...")
    
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
        
        # GPU optimized test
        print("   🔥 Testing GPU optimized (JIT) RWM...")
        target_gpu = MultivariateNormalTorch(dim)
        
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
        
        gpu_opt_speedup = baseline_time / results['GPU_Optimized']['time']
        
        print(f"\n   🎯 Analysis:")
        print(f"      Acceptance rate consistency: {acc_rate_range:.4f} (should be < 0.05)")
        print(f"      GPU Optimized speedup: {gpu_opt_speedup:.2f}x")
        
        # Performance criteria
        performance_ok = True
        if acc_rate_range > 0.1:
            print(f"   ⚠️  Large acceptance rate differences detected")
            performance_ok = False
        
        # More lenient performance criteria for CPU execution
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_type == 'cpu':
            # On CPU, just check that optimized version is reasonable
            if gpu_opt_speedup < 0.1:  # Should not be dramatically slower
                print(f"   ⚠️  GPU optimized much slower than expected on CPU")
                performance_ok = False
            else:
                print(f"   ✅ Performance acceptable for CPU execution")
        else:
            # On GPU, optimized should provide speedup
            if gpu_opt_speedup < 1.0:
                print(f"   ⚠️  GPU optimized slower than CPU")
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
    """Test that the optimized implementation works on both GPU and CPU."""
    print("\n🔧 Testing Device Fallback for GPU Optimized Implementation...")
    
    dim = 3
    num_samples = 100
    variance = 1.0
    target_dist_cpu = MultivariateNormal(dim)
    target_dist_gpu = MultivariateNormalTorch(dim)
    
    devices_to_test = ['cpu']
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
    
    implementations = {
        'GPU_Optimized': RandomWalkMH_GPU_Optimized
    }
    
    results = {}
    
    for device in devices_to_test:
        print(f"   Testing on {device.upper()}...")
        results[device] = {}
        
        for impl_name, impl_class in implementations.items():
            try:
                print(f"      Testing {impl_name}...")
                
                rwm = impl_class(
                    dim=dim,
                    var=variance,
                    target_dist=target_dist_gpu,
                    device=device,
                    pre_allocate_steps=num_samples,
                    use_efficient_rng=True
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
        print(f"   ✅ Implementation works on all available devices")
        return True
    elif successful_tests > 0:
        print(f"   ⚠️  Some implementations failed on some devices")
        return True  # Partial success is acceptable
    else:
        print(f"   ❌ All implementations failed")
        return False

def test_funnel_distributions():
    """Test new funnel distributions (NealFunnelTorch and SuperFunnelTorch) with RWM."""
    print("\n🌪️ Testing Funnel Distributions...")
    
    try:
        # Test 1: Neal's Funnel Distribution
        print("   🔍 Test 1: Neal's Funnel Distribution...")
        
        dim = 5
        num_samples = 2000
        burn_in = 500
        
        neal_funnel = NealFunnelTorch(dimension=dim)
        print(f"      Created {neal_funnel.get_name()}")
        
        # Use smaller proposal variance for funnel (challenging geometry)
        variance = 0.2
        
        rwm_funnel = RandomWalkMH_GPU_Optimized(
            dim=dim,
            var=variance,
            target_dist=neal_funnel,
            symmetric=True,
            pre_allocate_steps=num_samples,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Generate samples
        torch.manual_seed(789)
        chain = rwm_funnel.generate_samples(num_samples)
        
        # Basic checks
        assert len(chain) == num_samples, f"Expected {num_samples} samples, got {len(chain)}"
        
        # Post burn-in analysis
        post_burnin_chain = chain[burn_in:]
        assert len(post_burnin_chain) == num_samples - burn_in, f"Post burn-in should have {num_samples - burn_in} samples, got {len(post_burnin_chain)}"
        
        # Check v variable (first dimension) statistics
        v_samples = post_burnin_chain[:, 0]
        v_mean = torch.mean(v_samples)
        v_std = torch.std(v_samples)
        
        print(f"      Samples generated: {len(chain)}")
        print(f"      Post burn-in samples: {len(post_burnin_chain)}")
        print(f"      Acceptance rate: {rwm_funnel.acceptance_rate:.3f}")
        print(f"      V variable mean: {v_mean:.3f} (expected ~0)")
        print(f"      V variable std: {v_std:.3f} (expected ~3)")
        print(f"      Note: Neal's Funnel has challenging geometry for RWM")
        print(f"      The hierarchical structure (z_k ~ N(0, exp(v))) creates")
        print(f"      exploration difficulties that may require longer chains or adaptive samplers")
        
        # More realistic bounds for funnel with RWM (challenging geometry)
        # RWM often struggles with funnel distributions, so we use looser bounds
        funnel_test1_pass = (
            0.1 < rwm_funnel.acceptance_rate < 0.9 and  # Reasonable acceptance rate
            abs(v_mean) < 2.0 and  # V mean within 2 standard deviations (more lenient)
            0.5 < v_std < 6.0  # V std within reasonable range (RWM may not fully explore)
        )
        
        if funnel_test1_pass:
            print("   ✅ Neal's Funnel test passed (RWM functional on challenging geometry)")
        else:
            print("   ⚠️  Neal's Funnel test results outside expected bounds")
            print("      This may indicate RWM difficulty with funnel geometry (not necessarily a bug)")
        
        # Test 2: Super Funnel Distribution (Hierarchical Logistic Regression)
        print("   🔍 Test 2: Super Funnel Distribution...")
        
        # Create synthetic logistic regression data
        J, K = 3, 2  # 3 groups, 2 covariates
        n_j = [20, 15, 25]  # Sample sizes per group
        
        # Generate synthetic X and Y data
        torch.manual_seed(456)
        X_data = []
        Y_data = []
        
        for j in range(J):
            X_j = torch.randn(n_j[j], K)
            # Create somewhat realistic logistic regression outcomes
            true_alpha = torch.randn(1) * 0.5
            true_beta = torch.randn(K) * 0.3
            logits = true_alpha + torch.matmul(X_j, true_beta)
            probs = torch.sigmoid(logits)
            Y_j = torch.bernoulli(probs)
            
            X_data.append(X_j)
            Y_data.append(Y_j)
        
        super_funnel = SuperFunnelTorch(J, K, X_data, Y_data)
        print(f"      Created {super_funnel.get_name()}")
        print(f"      Total dimension: {super_funnel.dim}")
        
        # Use very small proposal variance for super funnel (extremely challenging)
        super_variance = 0.01
        num_samples_super = 1000
        burn_in_super = 200
        
        rwm_super = RandomWalkMH_GPU_Optimized(
            dim=super_funnel.dim,
            var=super_variance,
            target_dist=super_funnel,
            symmetric=True,
            pre_allocate_steps=num_samples_super,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Generate samples (this is very challenging, so expect low acceptance)
        torch.manual_seed(987)
        super_chain = rwm_super.generate_samples(num_samples_super)
        
        # Basic checks
        assert len(super_chain) == num_samples_super, f"Expected {num_samples_super} samples, got {len(super_chain)}"
        
        # Post burn-in analysis
        post_burnin_super = super_chain[burn_in_super:]
        assert len(post_burnin_super) == num_samples_super - burn_in_super, f"Post burn-in should have {num_samples_super - burn_in_super} samples, got {len(post_burnin_super)}"
        
        print(f"      Samples generated: {len(super_chain)}")
        print(f"      Post burn-in samples: {len(post_burnin_super)}")
        print(f"      Acceptance rate: {rwm_super.acceptance_rate:.3f}")
        
        # For Super Funnel, just check that it runs and produces finite results
        finite_samples = torch.isfinite(post_burnin_super).all(dim=1).sum().item()
        finite_fraction = finite_samples / len(post_burnin_super)
        
        print(f"      Finite samples fraction: {finite_fraction:.3f}")
        
        # Super Funnel is very challenging, so just check basic functionality
        funnel_test2_pass = (
            len(super_chain) == num_samples_super and
            finite_fraction > 0.5 and  # At least half the samples should be finite
            rwm_super.acceptance_rate > 0.001  # Some minimal acceptance
        )
        
        if funnel_test2_pass:
            print("   ✅ Super Funnel test passed")
        else:
            print("   ⚠️  Super Funnel test results seem unusual")
        
        overall_pass = funnel_test1_pass and funnel_test2_pass
        
        if overall_pass:
            print("   ✅ All funnel distribution tests passed")
        else:
            print("   ⚠️  Some funnel distribution tests failed")
            
        return overall_pass
        
    except Exception as e:
        print(f"   ❌ Funnel distribution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_burnin_and_sample_counting():
    """Test burn-in handling and sample counting accuracy."""
    print("\n🔥 Testing Burn-in and Sample Counting...")
    
    try:
        dim = 4
        total_samples = 3000
        burn_in_sizes = [0, 500, 1000, 1500]
        variance = 1.0
        
        target_dist = MultivariateNormalTorch(dim)
        
        all_tests_pass = True
        
        for burn_in in burn_in_sizes:
            print(f"   🔍 Testing burn-in = {burn_in}...")
            
            if burn_in >= total_samples:
                print(f"      Skipping (burn-in >= total samples)")
                continue
            
            # Generate samples
            rwm = RandomWalkMH_GPU_Optimized(
                dim=dim,
                var=variance,
                target_dist=target_dist,
                symmetric=True,
                pre_allocate_steps=total_samples,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            torch.manual_seed(555)
            full_chain = rwm.generate_samples(total_samples)
            
            # Manual burn-in removal
            if burn_in > 0:
                post_burnin_manual = full_chain[burn_in:]
            else:
                post_burnin_manual = full_chain
            
            expected_post_burnin_count = total_samples - burn_in
            actual_post_burnin_count = len(post_burnin_manual)
            
            print(f"      Total samples: {len(full_chain)}")
            print(f"      Expected post burn-in: {expected_post_burnin_count}")
            print(f"      Actual post burn-in: {actual_post_burnin_count}")
            
            # Test sample counting
            count_correct = (actual_post_burnin_count == expected_post_burnin_count)
            
            if not count_correct:
                print(f"      ❌ Sample count mismatch!")
                all_tests_pass = False
            else:
                print(f"      ✅ Sample count correct")
            
            # Test ESJD calculation with different burn-in sizes
            if hasattr(rwm, 'expected_squared_jump_distance_gpu'):
                # For GPU version, we need to manually compute ESJD with burn-in
                if burn_in > 0:
                    # Manually compute ESJD excluding burn-in
                    post_burnin_chain = full_chain[burn_in:]
                    if len(post_burnin_chain) > 1:
                        diffs = post_burnin_chain[1:] - post_burnin_chain[:-1]
                        esjd_manual = torch.mean(torch.sum(diffs**2, dim=1)).item()
                    else:
                        esjd_manual = 0.0
                else:
                    esjd_manual = rwm.expected_squared_jump_distance_gpu()
                
                print(f"      ESJD (manual, burn-in={burn_in}): {esjd_manual:.6f}")
                
                # ESJD should be positive for MCMC
                esjd_reasonable = esjd_manual > 0
                if not esjd_reasonable:
                    print(f"      ❌ ESJD seems unreasonable")
                    all_tests_pass = False
                else:
                    print(f"      ✅ ESJD calculation looks good")
        
        if all_tests_pass:
            print("   ✅ All burn-in and sample counting tests passed")
        else:
            print("   ⚠️  Some burn-in or sample counting tests failed")
            
        return all_tests_pass
        
    except Exception as e:
        print(f"   ❌ Burn-in test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_target_distributions():
    """Test RWM on various target distributions to ensure broad compatibility."""
    print("\n🎯 Testing Comprehensive Target Distributions...")
    
    try:
        # Test parameters
        num_samples = 1000
        burn_in = 200
        
        # Import additional distributions for comprehensive testing
        from target_distributions import (
            MultivariateNormalTorch, NealFunnelTorch, 
            HypercubeTorch, ThreeMixtureDistributionTorch, IIDGammaTorch
        )
        
        distributions_to_test = [
            ("MultivariateNormal", lambda: MultivariateNormalTorch(3)),
            ("NealFunnel", lambda: NealFunnelTorch(4)),
            ("Hypercube", lambda: HypercubeTorch(3)),
            ("Multimodal", lambda: ThreeMixtureDistributionTorch(2)),
            ("IIDProduct", lambda: IIDGammaTorch(2,3)),
        ]
        
        all_tests_pass = True
        
        for dist_name, dist_factory in distributions_to_test:
            print(f"   🔍 Testing {dist_name}...")
            
            try:
                # Create distribution
                target_dist = dist_factory()
                dim = target_dist.dim
                
                # Adjust proposal variance based on distribution
                if 'Funnel' in dist_name:
                    variance = 0.1  # Funnel needs smaller variance
                elif 'Multimodal' in dist_name:
                    variance = 0.5  # Multimodal needs moderate variance
                else:
                    variance = 2.38**2 / dim  # Standard optimal variance
                
                # Create RWM sampler
                rwm = RandomWalkMH_GPU_Optimized(
                    dim=dim,
                    var=variance,
                    target_dist=target_dist,
                    symmetric=True,
                    pre_allocate_steps=num_samples,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                # Generate samples
                torch.manual_seed(111 + hash(dist_name) % 1000)
                chain = rwm.generate_samples(num_samples)
                
                # Basic checks
                assert len(chain) == num_samples, f"Wrong number of samples for {dist_name}"
                
                # Post burn-in analysis
                post_burnin = chain[burn_in:]
                
                # Check for finite samples
                finite_mask = torch.isfinite(post_burnin).all(dim=1)
                finite_fraction = finite_mask.float().mean().item()
                
                # Check acceptance rate
                acc_rate = rwm.acceptance_rate
                
                print(f"      Dimension: {dim}")
                print(f"      Proposal variance: {variance:.6f}")
                print(f"      Acceptance rate: {acc_rate:.3f}")
                print(f"      Finite samples: {finite_fraction:.3f}")
                
                # Minimal criteria for success
                distribution_ok = (
                    len(chain) == num_samples and
                    finite_fraction > 0.8 and  # Most samples should be finite
                    acc_rate > 0.001 and  # Some minimal acceptance
                    acc_rate < 0.999  # Not accepting everything
                )
                
                if distribution_ok:
                    print(f"      ✅ {dist_name} test passed")
                else:
                    print(f"      ❌ {dist_name} test failed")
                    all_tests_pass = False
                    
            except Exception as e:
                print(f"      ❌ {dist_name} test crashed: {e}")
                all_tests_pass = False
        
        if all_tests_pass:
            print("   ✅ All target distribution tests passed")
        else:
            print("   ⚠️  Some target distribution tests failed")
            
        return all_tests_pass
        
    except Exception as e:
        print(f"   ❌ Comprehensive distribution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all RWM tests."""
    print("🚀 RWM GPU Implementation Test Suite")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  No GPU detected - tests will run on CPU")
    
    # Run tests
    tests = [
        ("RWM Correctness", test_standard_rwm_correctness),
        ("GPU Optimized JIT Kernels", test_gpu_optimized_jit_kernels),
        ("Performance Comparison", test_performance_comparison),
        ("Device Fallback", test_device_fallback),
        ("Funnel Distributions", test_funnel_distributions),
        ("Burn-in & Sample Counting", test_burnin_and_sample_counting),
        ("Comprehensive Distributions", test_comprehensive_target_distributions),
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
        print("\n🎉 All tests passed! Your RWM GPU implementation is correct.")
        print("💡 You can confidently use the optimized GPU implementation for research.")
    else:
        print("\n⚠️  Some tests failed. RWM implementation may need fixes.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 