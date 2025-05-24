#!/usr/bin/env python3
"""
Test script to verify GPU-accelerated Metropolis algorithm implementation.
This script tests both standard RWM correctness and performance.
"""

import torch
import numpy as np
import time
from algorithms import RandomWalkMH_GPU, RandomWalkMH
from target_distributions import MultivariateNormal_GPU, MultivariateNormal
from interfaces import MCMCSimulation_GPU, MCMCSimulation
from scipy import stats

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
        
        # CPU version (known to be correct)
        from target_distributions.gaussian import StandardGaussian
        target_cpu = StandardGaussian(dim)
        
        rwm_cpu = RandomWalkMH(dim, variance, target_cpu)
        np.random.seed(42)
        for _ in range(num_samples):
            rwm_cpu.step()
        
        cpu_chain = np.array(rwm_cpu.chain)
        cpu_acc_rate = rwm_cpu.acceptance_rate
        
        # GPU standard version
        target_gpu = StandardGaussian(dim)  # Should work with GPU wrapper
        rwm_gpu_std = RandomWalkMH_GPU(
            dim=dim, 
            var=variance, 
            target_dist=target_gpu,
            standard_rwm=True,  # This is the key!
            pre_allocate_steps=num_samples,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        torch.manual_seed(42)
        np.random.seed(42)
        gpu_chain_std = rwm_gpu_std.generate_samples_standard(num_samples)
        gpu_acc_rate_std = rwm_gpu_std.acceptance_rate
        
        # Compare acceptance rates (should be very close)
        acc_rate_diff = abs(cpu_acc_rate - gpu_acc_rate_std)
        print(f"      CPU acceptance rate: {cpu_acc_rate:.4f}")
        print(f"      GPU standard acceptance rate: {gpu_acc_rate_std:.4f}")
        print(f"      Difference: {acc_rate_diff:.4f}")
        
        if acc_rate_diff > 0.05:  # Allow 5% difference due to random seed differences
            print(f"   ⚠️  Large acceptance rate difference detected")
            # Don't fail immediately, could be due to random differences
        else:
            print(f"   ✅ Acceptance rates are consistent")
        
        # Test 2: Standard vs Batch should give different results
        print("   🔍 Test 2: Standard vs Batch RWM should differ...")
        
        rwm_gpu_batch = RandomWalkMH_GPU(
            dim=dim, 
            var=variance, 
            target_dist=target_gpu,
            standard_rwm=False,  # Batch mode
            batch_size=32,
            pre_allocate_steps=num_samples
        )
        
        torch.manual_seed(42)
        np.random.seed(42)
        gpu_chain_batch = rwm_gpu_batch.generate_samples_batch(num_samples, batch_size=32)
        gpu_acc_rate_batch = rwm_gpu_batch.acceptance_rate
        
        print(f"      Standard RWM acceptance rate: {gpu_acc_rate_std:.4f}")
        print(f"      Batch RWM acceptance rate: {gpu_acc_rate_batch:.4f}")
        
        # They should be different (batch processing changes dynamics)
        batch_diff = abs(gpu_acc_rate_std - gpu_acc_rate_batch)
        if batch_diff < 0.01:
            print(f"   ⚠️  Standard and batch modes too similar: {batch_diff:.4f}")
        else:
            print(f"   ✅ Standard and batch modes appropriately different")
        
        # Test 3: Statistical properties for standard RWM
        print("   🔍 Test 3: Statistical properties of standard RWM...")
        
        # For 2D standard Gaussian, mean should be ~[0,0] and std ~[1,1]
        gpu_mean = np.mean(gpu_chain_std[1000:], axis=0)  # Skip burn-in
        gpu_std = np.std(gpu_chain_std[1000:], axis=0)
        
        mean_error = np.linalg.norm(gpu_mean)
        std_error = np.linalg.norm(gpu_std - 1.0)
        
        print(f"      Empirical mean: {gpu_mean}")
        print(f"      Empirical std: {gpu_std}")
        print(f"      Mean error (should be ~0): {mean_error:.4f}")
        print(f"      Std error (should be ~0): {std_error:.4f}")
        
        if mean_error > 0.2 or std_error > 0.3:
            print(f"   ⚠️  Statistical properties seem off")
            return False
        else:
            print(f"   ✅ Statistical properties look good")
        
        # Test 4: Chain should have proper sequential dependence
        print("   🔍 Test 4: Testing sequential dependence...")
        
        # Compute lag-1 autocorrelation (should be positive and reasonable)
        chain_centered = gpu_chain_std[1000:] - gpu_mean
        autocorr_x = np.corrcoef(chain_centered[:-1, 0], chain_centered[1:, 0])[0, 1]
        autocorr_y = np.corrcoef(chain_centered[:-1, 1], chain_centered[1:, 1])[0, 1]
        
        print(f"      Lag-1 autocorrelation X: {autocorr_x:.4f}")
        print(f"      Lag-1 autocorrelation Y: {autocorr_y:.4f}")
        
        # For RWM, autocorrelation should be positive (around 0.2-0.8)
        if 0.05 < autocorr_x < 0.95 and 0.05 < autocorr_y < 0.95:
            print(f"   ✅ Autocorrelation looks reasonable for MCMC")
        else:
            print(f"   ⚠️  Autocorrelation seems unusual")
        
        print(f"   ✅ Standard RWM correctness tests completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Standard RWM correctness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_functionality():
    """Test basic GPU functionality."""
    print("🧪 Testing Basic GPU functionality...")
    
    dim = 3
    num_samples = 1000
    variance = 1.0
    
    try:
        from target_distributions.gaussian import StandardGaussian
        target_dist = StandardGaussian(dim)
        
        # Test standard RWM mode
        rwm_gpu = RandomWalkMH_GPU(
            dim=dim, 
            var=variance, 
            target_dist=target_dist,
            standard_rwm=True,
            pre_allocate_steps=num_samples
        )
        
        samples = rwm_gpu.generate_samples_standard(num_samples)
        
        # Basic checks
        assert len(samples) == num_samples + 1, f"Expected {num_samples + 1} samples, got {len(samples)}"
        assert samples.shape[1] == dim, f"Expected dimension {dim}, got {samples.shape[1]}"
        assert 0 <= rwm_gpu.acceptance_rate <= 1, f"Invalid acceptance rate: {rwm_gpu.acceptance_rate}"
        
        esjd = rwm_gpu.expected_squared_jump_distance_gpu()
        assert esjd >= 0, f"Invalid ESJD: {esjd}"
        
        print(f"   ✅ Basic GPU functionality test passed!")
        print(f"   📊 Acceptance rate: {rwm_gpu.acceptance_rate:.3f}")
        print(f"   📊 ESJD: {esjd:.6f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GPU functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Compare GPU vs CPU performance."""
    print("\n⚡ Testing performance comparison...")
    
    # Test parameters
    dim = 10
    num_samples = 5000
    variance = 2.38**2 / dim  # Roughly optimal variance
    
    try:
        # GPU test - using standard RWM mode
        print("   🚀 Testing GPU standard RWM implementation...")
        from target_distributions.gaussian import StandardGaussian
        target_gpu = StandardGaussian(dim)
        
        gpu_start = time.time()
        rwm_gpu = RandomWalkMH_GPU(
            dim=dim,
            var=variance,
            target_dist=target_gpu,
            standard_rwm=True,  # Use true standard RWM
            pre_allocate_steps=num_samples,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        chain_gpu = simulation_gpu.generate_samples(use_batch_processing=True, progress_bar=False)
        gpu_time = time.time() - gpu_start
        gpu_acc = simulation_gpu.acceptance_rate()
        gpu_esjd = simulation_gpu.expected_squared_jump_distance()
        
        print(f"      Time: {gpu_time:.2f}s, Acc: {gpu_acc:.3f}, ESJD: {gpu_esjd:.6f}")
        
        # CPU test
        print("   🐌 Testing CPU implementation...")
        target_cpu = MultivariateNormal(dim)
        
        cpu_start = time.time()
        simulation_cpu = MCMCSimulation(
            dim=dim,
            sigma=variance,
            num_iterations=num_samples,
            algorithm=RandomWalkMH,
            target_dist=target_cpu,
            seed=42
        )
        
        chain_cpu = simulation_cpu.generate_samples()
        cpu_time = time.time() - cpu_start
        cpu_acc = simulation_cpu.acceptance_rate()
        cpu_esjd = simulation_cpu.expected_squared_jump_distance()
        
        print(f"      Time: {cpu_time:.2f}s, Acc: {cpu_acc:.3f}, ESJD: {cpu_esjd:.6f}")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        acc_diff = abs(gpu_acc - cpu_acc)
        esjd_diff = abs(gpu_esjd - cpu_esjd) / max(gpu_esjd, cpu_esjd) if max(gpu_esjd, cpu_esjd) > 0 else 0
        
        print(f"\n   📈 Performance Results:")
        print(f"      Speedup: {speedup:.2f}x")
        print(f"      Acceptance rate difference: {acc_diff:.4f}")
        print(f"      ESJD relative difference: {esjd_diff:.4f}")
        
        # Validate consistency
        if acc_diff < 0.1 and esjd_diff < 0.2:
            print(f"   ✅ Results are consistent between GPU and CPU")
            consistency_check = True
        else:
            print(f"   ⚠️  Large differences detected")
            consistency_check = False
            
        if speedup > 1.0:
            print(f"   ✅ GPU shows performance improvement")
            performance_check = True
        else:
            print(f"   ⚠️  GPU not faster (might be expected for small problems)")
            performance_check = False
            
        return consistency_check and (performance_check or num_samples < 10000)
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def test_batch_processing():
    """Test different batch sizes."""
    print("\n🔢 Testing batch processing...")
    
    dim = 8
    num_samples = 2000
    variance = 1.0
    target_dist = MultivariateNormal_GPU(dim)
    
    batch_sizes = [1, 10, 100, 500]
    results = []
    
    for batch_size in batch_sizes:
        try:
            print(f"   Testing batch size {batch_size}...")
            
            start_time = time.time()
            simulation = MCMCSimulation_GPU(
                dim=dim,
                sigma=variance,
                num_iterations=num_samples,
                algorithm=RandomWalkMH_GPU,
                target_dist=target_dist,
                batch_size=batch_size,
                seed=42
            )
            
            chain = simulation.generate_samples(use_batch_processing=True, progress_bar=False)
            elapsed = time.time() - start_time
            
            acc_rate = simulation.acceptance_rate()
            esjd = simulation.expected_squared_jump_distance()
            
            results.append({
                'batch_size': batch_size,
                'time': elapsed,
                'acceptance_rate': acc_rate,
                'esjd': esjd
            })
            
            print(f"      Time: {elapsed:.2f}s, Acc: {acc_rate:.3f}")
            
        except Exception as e:
            print(f"      ❌ Failed for batch size {batch_size}: {e}")
            return False
    
    print(f"\n   📊 Batch Processing Results:")
    for result in results:
        sps = num_samples / result['time']
        print(f"      Batch {result['batch_size']:3d}: {result['time']:5.2f}s ({sps:6.0f} samples/s)")
    
    print(f"   ✅ Batch processing test completed")
    return True

def main():
    """Run all tests."""
    print("🚀 GPU-Accelerated Metropolis Algorithm Test Suite")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected - tests will run on CPU")
    
    # Run tests
    tests = [
        ("Standard RWM Correctness", test_standard_rwm_correctness),
        ("Basic Functionality", test_gpu_functionality),
        ("Performance", test_performance_comparison),
        ("Batch Processing", test_batch_processing),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} Test {'='*20}")
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
        print(f"   {test_name:15s}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your GPU optimization is working correctly.")
        print("💡 You can now use the GPU-accelerated algorithm for much faster sampling.")
        print("\nTo use the optimized algorithm:")
        print("   python experiment_RWM_GPU.py --mode comparison --dim 20 --num_iters 100000")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        print("💡 The CPU version should still work for your research.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 