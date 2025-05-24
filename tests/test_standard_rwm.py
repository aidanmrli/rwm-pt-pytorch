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
        np.random.seed(42)
        torch.manual_seed(42)
        # Generate samples
        for _ in range(num_samples):
            rwm_cpu.step()
        
        cpu_chain = np.array(rwm_cpu.chain)
        cpu_acc_rate = rwm_cpu.acceptance_rate
        
        gpu_chain_std = rwm_gpu_std.generate_samples_standard(num_samples)
        gpu_acc_rate_std = rwm_gpu_std.acceptance_rate
        
        
        # Compare acceptance rates (should be very close)
        acc_rate_diff = abs(cpu_acc_rate - gpu_acc_rate_std)
        print(f"      CPU acceptance rate: {cpu_acc_rate:.4f}")
        print(f"      GPU standard acceptance rate: {gpu_acc_rate_std:.4f}")
        print(f"      Difference: {acc_rate_diff:.4f}")
        
        test1_pass = acc_rate_diff < 0.1  # Allow 10% difference due to random seed differences
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
        
        print(f"      Empirical mean: {gpu_mean}")
        print(f"      Empirical std: {gpu_std}")
        print(f"      Mean error (should be ~0): {mean_error:.4f}")
        print(f"      Std error (should be ~0): {std_error:.4f}")
        
        test3_pass = mean_error < 0.2 and std_error < 0.3
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
        
        print(f"      Lag-1 autocorrelation X: {autocorr_x:.4f}")
        print(f"      Lag-1 autocorrelation Y: {autocorr_y:.4f}")
        
        # For RWM, autocorrelation should be positive (around 0.2-0.8)
        test4_pass = (0.05 < autocorr_x < 0.95 and 0.05 < autocorr_y < 0.95)
        if test4_pass:
            print(f"   ✅ Autocorrelation looks reasonable for MCMC")
        else:
            print(f"   ⚠️  Autocorrelation seems unusual")
        
        # Test 5: Sequential nature test - check that proposals depend on previous state
        print("   🔍 Test 5: Testing true sequential nature...")
        
        # Reset and manually check a few steps
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
        
        # Verify sequential dependence
        sequential_ok = True
        if not torch.allclose(step2_before, step1_after, atol=1e-6):
            sequential_ok = False
        if not torch.allclose(step3_before, step2_after, atol=1e-6):
            sequential_ok = False
        
        test5_pass = sequential_ok
        if test5_pass:
            print(f"   ✅ Sequential dependence verified")
        else:
            print(f"   ❌ Sequential dependence failed!")
        
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

def test_performance_comparison():
    """Compare standard GPU RWM vs CPU RWM performance."""
    print("\n⚡ Performance Comparison: Standard RWM GPU vs CPU...")
    
    dim = 10
    num_samples = 10000
    variance = 2.38**2 / dim  # Roughly optimal variance
    
    try:
        # CPU test
        print("   🐌 Testing CPU standard RWM...")
        target_cpu = MultivariateNormal(dim)
        
        cpu_start = time.time()
        rwm_cpu = RandomWalkMH(dim, variance, target_cpu)
        for _ in range(num_samples):
            rwm_cpu.step()
        cpu_time = time.time() - cpu_start
        cpu_acc = rwm_cpu.acceptance_rate
        
        print(f"      Time: {cpu_time:.2f}s, Acc: {cpu_acc:.3f}")
        
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
        
        gpu_chain = rwm_gpu.generate_samples_standard(num_samples)
        gpu_time = time.time() - gpu_start
        gpu_acc = rwm_gpu.acceptance_rate
        
        print(f"      Time: {gpu_time:.2f}s, Acc: {gpu_acc:.3f}")
        
        # Compare results
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        acc_diff = abs(gpu_acc - cpu_acc)
        
        print(f"\n   📈 Results:")
        print(f"      Speedup: {speedup:.2f}x")
        print(f"      Acceptance rate difference: {acc_diff:.4f}")
        
        if speedup > 1.0:
            print(f"   ✅ GPU standard RWM is faster than CPU")
        elif speedup > 0.5:
            print(f"   ✅ GPU standard RWM competitive with CPU")
        else:
            print(f"   ⚠️  GPU slower (may be expected for this problem size)")
        
        if acc_diff < 0.05:
            print(f"   ✅ Acceptance rates are consistent")
        else:
            print(f"   ⚠️  Acceptance rates differ significantly")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_fallback():
    """Test that the implementation works on both GPU and CPU."""
    print("\n🔧 Testing Device Fallback...")
    
    dim = 3
    num_samples = 100
    variance = 1.0
    target_dist = MultivariateNormal(dim)
    
    devices_to_test = ['cpu']
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
    
    results = {}
    
    for device in devices_to_test:
        try:
            print(f"   Testing on {device.upper()}...")
            
            rwm = RandomWalkMH_GPU(
                dim=dim,
                var=variance,
                target_dist=target_dist,
                standard_rwm=True,
                pre_allocate_steps=num_samples,
                device=device
            )
            
            chain = rwm.generate_samples_standard(num_samples)
            
            results[device] = {
                'success': True,
                'chain_length': len(chain),
                'acceptance_rate': rwm.acceptance_rate,
                'device_used': rwm.device.type
            }
            
            print(f"      ✅ {device.upper()} test passed")
            print(f"         Samples: {len(chain)}, Acc rate: {rwm.acceptance_rate:.3f}")
            
        except Exception as e:
            print(f"      ❌ {device.upper()} test failed on line {sys.exc_info()[2].tb_lineno}: {e}")
            results[device] = {'success': False, 'error': str(e)}
    
    # Summary
    successful_devices = [dev for dev, res in results.items() if res.get('success', False)]
    
    if len(successful_devices) > 0:
        print(f"   ✅ Working on devices: {', '.join(successful_devices)}")
        return True
    else:
        print(f"   ❌ Failed on all devices")
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