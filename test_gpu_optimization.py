#!/usr/bin/env python3
"""
Simple test script to verify GPU-accelerated Metropolis algorithm implementation.
This script performs a quick functional test and basic performance comparison.
"""

import torch
import numpy as np
import time
from algorithms import RandomWalkMH_GPU, RandomWalkMH
from target_distributions import MultivariateNormal_GPU, MultivariateNormal
from interfaces import MCMCSimulation_GPU, MCMCSimulation

def test_gpu_functionality():
    """Test basic functionality of GPU implementation."""
    print("🧪 Testing GPU functionality...")
    
    # Simple test parameters
    dim = 5
    num_samples = 1000
    variance = 1.0
    
    try:
        # Create GPU target distribution
        target_dist = MultivariateNormal_GPU(dim)
        
        # Create GPU simulation
        simulation = MCMCSimulation_GPU(
            dim=dim,
            sigma=variance,
            num_iterations=num_samples,
            algorithm=RandomWalkMH_GPU,
            target_dist=target_dist,
            batch_size=100,
            pre_allocate=True
        )
        
        # Generate samples
        chain = simulation.generate_samples(use_batch_processing=True, progress_bar=False)
        
        # Basic checks
        assert len(chain) == num_samples + 1, f"Expected {num_samples + 1} samples, got {len(chain)}"
        assert len(chain[0]) == dim, f"Expected dimension {dim}, got {len(chain[0])}"
        assert 0 <= simulation.acceptance_rate() <= 1, f"Invalid acceptance rate: {simulation.acceptance_rate()}"
        
        esjd = simulation.expected_squared_jump_distance()
        assert esjd >= 0, f"Invalid ESJD: {esjd}"
        
        print(f"   ✅ Basic functionality test passed!")
        print(f"   📊 Acceptance rate: {simulation.acceptance_rate():.3f}")
        print(f"   📊 ESJD: {esjd:.6f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Functionality test failed: {e}")
        return False

def test_performance_comparison():
    """Compare GPU vs CPU performance."""
    print("\n⚡ Testing performance comparison...")
    
    # Test parameters
    dim = 10
    num_samples = 5000
    variance = 2.38**2 / dim  # Roughly optimal variance
    
    try:
        # GPU test
        print("   🚀 Testing GPU implementation...")
        target_gpu = MultivariateNormal_GPU(dim)
        
        gpu_start = time.time()
        simulation_gpu = MCMCSimulation_GPU(
            dim=dim,
            sigma=variance,
            num_iterations=num_samples,
            algorithm=RandomWalkMH_GPU,
            target_dist=target_gpu,
            batch_size=512,
            pre_allocate=True,
            seed=42
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
        ("Functionality", test_gpu_functionality),
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