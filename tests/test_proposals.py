#!/usr/bin/env python3
"""
Comprehensive test script for the new proposal distribution system.
Tests Normal, Laplace, and UniformRadius proposals with:
- Statistical property verification
- Integration with GPU-optimized RWM algorithm
- Multiple target distributions
- Performance comparisons
- Edge cases and error handling
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import warnings

from interfaces import MCMCSimulation_GPU
from algorithms import RandomWalkMH_GPU_Optimized
from proposal_distributions import ProposalDistribution, NormalProposal, LaplaceProposal, UniformRadiusProposal
from target_distributions import MultivariateNormalTorch, ThreeMixtureDistributionTorch, IIDGammaTorch, RoughCarpetDistributionTorch

class ProposalTester:
    """Comprehensive test suite for proposal distributions."""
    
    def __init__(self, device=None, verbose=True):
        """Initialize the tester.
        
        Args:
            device: PyTorch device to use for testing
            verbose: Whether to print detailed progress information
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.verbose = verbose
        self.test_results = {}
        
        if self.verbose:
            print(f"🧪 Comprehensive Proposal Distribution Test Suite")
            print(f"   Device: {self.device}")
            print("=" * 70)
    
    def log(self, message: str, level: int = 1):
        """Log a message with indentation based on level."""
        if self.verbose:
            indent = "   " * level
            print(f"{indent}{message}")
    
    def test_proposal_creation(self) -> Dict[str, bool]:
        """Test basic proposal distribution creation and parameter validation."""
        self.log("🔧 Testing Proposal Creation and Parameter Validation")
        results = {}
        
        dim = 5
        beta = 1.0
        dtype = torch.float32
        
        # Test Normal proposal
        try:
            normal_proposal = NormalProposal(
                dim=dim, 
                base_variance_scalar=0.5, 
                beta=beta,
                device=self.device, 
                dtype=dtype
            )
            self.log(f"✅ NormalProposal created successfully", 2)
            self.log(f"   Name: {normal_proposal.get_name()}", 3)
            self.log(f"   Std dev: {normal_proposal.std_dev.item():.4f}", 3)
            results['Normal'] = True
        except Exception as e:
            self.log(f"❌ NormalProposal failed: {str(e)}", 2)
            results['Normal'] = False
        
        # Test Laplace proposal with vector variance
        try:
            base_variance_vector = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=dtype, device=self.device)
            laplace_proposal = LaplaceProposal(
                dim=dim,
                base_variance_vector=base_variance_vector,
                beta=beta,
                device=self.device,
                dtype=dtype
            )
            self.log(f"✅ LaplaceProposal created successfully", 2)
            self.log(f"   Name: {laplace_proposal.get_name()}", 3)
            self.log(f"   Scale vector: {laplace_proposal.scale_vector.cpu().numpy()}", 3)
            results['Laplace'] = True
        except Exception as e:
            self.log(f"❌ LaplaceProposal failed: {str(e)}", 2)
            results['Laplace'] = False
        
        # Test UniformRadius proposal
        try:
            uniform_proposal = UniformRadiusProposal(
                dim=dim,
                base_radius=1.0,
                beta=beta,
                device=self.device,
                dtype=dtype
            )
            self.log(f"✅ UniformRadiusProposal created successfully", 2)
            self.log(f"   Name: {uniform_proposal.get_name()}", 3)
            self.log(f"   Effective radius: {uniform_proposal.effective_radius.item():.4f}", 3)
            results['UniformRadius'] = True
        except Exception as e:
            self.log(f"❌ UniformRadiusProposal failed: {str(e)}", 2)
            results['UniformRadius'] = False
        
        # Test error cases
        error_tests = {}
        
        # Negative variance for Normal
        try:
            NormalProposal(dim=dim, base_variance_scalar=-0.1, beta=beta, device=self.device, dtype=dtype)
            error_tests['Normal_negative_variance'] = False
        except ValueError:
            error_tests['Normal_negative_variance'] = True
            self.log("✅ Normal proposal correctly rejects negative variance", 2)
        
        # Wrong dimension for Laplace
        try:
            wrong_variance_vector = torch.tensor([0.1, 0.2], dtype=dtype, device=self.device)  # dim=2, but expecting dim=5
            LaplaceProposal(dim=dim, base_variance_vector=wrong_variance_vector, beta=beta, device=self.device, dtype=dtype)
            error_tests['Laplace_wrong_dim'] = False
        except ValueError:
            error_tests['Laplace_wrong_dim'] = True
            self.log("✅ Laplace proposal correctly rejects wrong dimension", 2)
        
        # Negative radius for UniformRadius
        try:
            UniformRadiusProposal(dim=dim, base_radius=-1.0, beta=beta, device=self.device, dtype=dtype)
            error_tests['UniformRadius_negative_radius'] = False
        except ValueError:
            error_tests['UniformRadius_negative_radius'] = True
            self.log("✅ UniformRadius proposal correctly rejects negative radius", 2)
        
        results['error_handling'] = all(error_tests.values())
        return results
    
    def test_statistical_properties(self, n_samples: int = 10000) -> Dict[str, Dict[str, float]]:
        """Test statistical properties of proposal distributions."""
        self.log("📊 Testing Statistical Properties")
        results = {}
        
        dim = 3
        beta = 1.0
        dtype = torch.float32
        
        # Test Normal proposal statistics
        self.log("Testing Normal proposal statistics", 2)
        normal_proposal = NormalProposal(
            dim=dim, base_variance_scalar=1.0, beta=beta, device=self.device, dtype=dtype
        )
        
        normal_samples = normal_proposal.sample(n_samples)
        normal_mean = torch.mean(normal_samples, dim=0)
        normal_var = torch.var(normal_samples, dim=0)
        
        results['Normal'] = {
            'mean_error': torch.max(torch.abs(normal_mean)).item(),
            'variance_error': torch.max(torch.abs(normal_var - 1.0)).item(),
            'sample_shape_correct': normal_samples.shape == (n_samples, dim)
        }
        
        self.log(f"Mean error: {results['Normal']['mean_error']:.6f} (should be ~0)", 3)
        self.log(f"Variance error: {results['Normal']['variance_error']:.6f} (should be ~0)", 3)
        
        # Test Laplace proposal statistics
        self.log("Testing Laplace proposal statistics", 2)
        base_variance_vector = torch.tensor([0.5, 1.0, 2.0], dtype=dtype, device=self.device)
        laplace_proposal = LaplaceProposal(
            dim=dim, base_variance_vector=base_variance_vector, beta=beta, device=self.device, dtype=dtype
        )
        
        laplace_samples = laplace_proposal.sample(n_samples)
        laplace_mean = torch.mean(laplace_samples, dim=0)
        laplace_var = torch.var(laplace_samples, dim=0)
        expected_var = (base_variance_vector / beta).to(self.device)  # Effective variance
        
        results['Laplace'] = {
            'mean_error': torch.max(torch.abs(laplace_mean)).item(),
            'variance_error': torch.max(torch.abs(laplace_var - expected_var)).item(),
            'sample_shape_correct': laplace_samples.shape == (n_samples, dim)
        }
        
        self.log(f"Mean error: {results['Laplace']['mean_error']:.6f} (should be ~0)", 3)
        self.log(f"Variance error: {results['Laplace']['variance_error']:.6f} (should be ~0)", 3)
        
        # Test UniformRadius proposal statistics
        self.log("Testing UniformRadius proposal statistics", 2)
        uniform_proposal = UniformRadiusProposal(
            dim=dim, base_radius=2.0, beta=beta, device=self.device, dtype=dtype
        )
        
        uniform_samples = uniform_proposal.sample(n_samples)
        uniform_mean = torch.mean(uniform_samples, dim=0)
        uniform_norms = torch.linalg.norm(uniform_samples, dim=1)
        max_norm = torch.max(uniform_norms)
        
        results['UniformRadius'] = {
            'mean_error': torch.max(torch.abs(uniform_mean)).item(),
            'max_norm': max_norm.item(),
            'radius_constraint_satisfied': max_norm.item() <= uniform_proposal.effective_radius.item() + 1e-6,
            'sample_shape_correct': uniform_samples.shape == (n_samples, dim)
        }
        
        self.log(f"Mean error: {results['UniformRadius']['mean_error']:.6f} (should be ~0)", 3)
        self.log(f"Max norm: {results['UniformRadius']['max_norm']:.6f}", 3)
        self.log(f"Effective radius: {uniform_proposal.effective_radius.item():.6f}", 3)
        
        return results
    
    def test_mcmc_integration(self, num_iterations: int = 5000) -> Dict[str, Dict[str, Any]]:
        """Test integration with MCMC simulation."""
        self.log("🔗 Testing MCMC Integration")
        results = {}
        
        dim = 5
        target_dist = MultivariateNormalTorch(dim, device=self.device)
        
        # Test configurations for each proposal type
        proposal_configs = {
            'Normal': {'name': 'Normal', 'params': {'base_variance_scalar': 0.5}},
            'Laplace': {'name': 'Laplace', 'params': {'base_variance_vector': [0.3] * dim}},
            'UniformRadius': {'name': 'UniformRadius', 'params': {'base_radius': 1.0}}
        }
        
        for proposal_name, config in proposal_configs.items():
            self.log(f"Testing {proposal_name} with MCMC", 2)
            
            try:
                simulation = MCMCSimulation_GPU(
                    dim=dim,
                    proposal_config=config,
                    num_iterations=num_iterations,
                    algorithm=RandomWalkMH_GPU_Optimized,
                    target_dist=target_dist,
                    device=str(self.device),
                    pre_allocate=True,
                    seed=42,
                    burn_in=500
                )
                
                # Run simulation
                start_time = time.time()
                chain = simulation.generate_samples(progress_bar=False)
                run_time = time.time() - start_time
                
                # Collect statistics
                acceptance_rate = simulation.acceptance_rate()
                esjd = simulation.expected_squared_jump_distance()
                
                results[proposal_name] = {
                    'success': True,
                    'acceptance_rate': acceptance_rate,
                    'esjd': esjd,
                    'run_time': run_time,
                    'samples_per_second': num_iterations / run_time,
                    'chain_length': len(chain) if hasattr(chain, '__len__') else 'unknown'
                }
                
                self.log(f"✅ Success! Acceptance rate: {acceptance_rate:.3f}, ESJD: {esjd:.6f}", 3)
                self.log(f"   Runtime: {run_time:.2f}s ({num_iterations/run_time:.0f} samples/s)", 3)
                
            except Exception as e:
                results[proposal_name] = {
                    'success': False,
                    'error': str(e),
                    'acceptance_rate': None,
                    'esjd': None,
                    'run_time': None
                }
                self.log(f"❌ Failed: {str(e)}", 3)
                import traceback
                if self.verbose:
                    traceback.print_exc()
        
        return results
    
    def test_multiple_target_distributions(self, num_iterations: int = 3000) -> Dict[str, Dict[str, Any]]:
        """Test proposals with multiple target distributions."""
        self.log("🎯 Testing Multiple Target Distributions")
        results = {}
        
        dim = 4
        
        # Different target distributions to test
        target_configs = {
            'MultivariateNormal': MultivariateNormalTorch(dim, device=self.device),
            'ThreeMixture': ThreeMixtureDistributionTorch(dim, device=self.device),
            'IIDGamma': IIDGammaTorch(dim, device=self.device),
            'RoughCarpet': RoughCarpetDistributionTorch(dim, device=self.device)
        }
        
        # Use a single proposal configuration for this test
        proposal_config = {'name': 'Normal', 'params': {'base_variance_scalar': 0.3}}
        
        for target_name, target_dist in target_configs.items():
            self.log(f"Testing with {target_name}", 2)
            
            try:
                simulation = MCMCSimulation_GPU(
                    dim=dim,
                    proposal_config=proposal_config,
                    num_iterations=num_iterations,
                    algorithm=RandomWalkMH_GPU_Optimized,
                    target_dist=target_dist,
                    device=str(self.device),
                    pre_allocate=True,
                    seed=42,
                    burn_in=300
                )
                
                start_time = time.time()
                chain = simulation.generate_samples(progress_bar=False)
                run_time = time.time() - start_time
                
                acceptance_rate = simulation.acceptance_rate()
                esjd = simulation.expected_squared_jump_distance()
                
                results[target_name] = {
                    'success': True,
                    'acceptance_rate': acceptance_rate,
                    'esjd': esjd,
                    'run_time': run_time
                }
                
                self.log(f"✅ Success! AR: {acceptance_rate:.3f}, ESJD: {esjd:.6f}, Time: {run_time:.1f}s", 3)
                
            except Exception as e:
                results[target_name] = {
                    'success': False,
                    'error': str(e),
                    'acceptance_rate': None,
                    'esjd': None,
                    'run_time': None
                }
                self.log(f"❌ Failed: {str(e)}", 3)
        
        return results
    
    def test_performance_comparison(self, num_iterations: int = 10000) -> Dict[str, Dict[str, float]]:
        """Compare performance across different proposal types."""
        self.log("⚡ Testing Performance Comparison")
        results = {}
        
        dim = 6
        target_dist = MultivariateNormalTorch(dim, device=self.device)
        
        # Comparable configurations (roughly similar scale)
        proposal_configs = {
            'Normal': {'name': 'Normal', 'params': {'base_variance_scalar': 0.4}},
            'Laplace': {'name': 'Laplace', 'params': {'base_variance_vector': [0.4] * dim}},
            'UniformRadius': {'name': 'UniformRadius', 'params': {'base_radius': 0.8}}
        }
        
        for proposal_name, config in proposal_configs.items():
            self.log(f"Benchmarking {proposal_name}", 2)
            
            try:
                simulation = MCMCSimulation_GPU(
                    dim=dim,
                    proposal_config=config,
                    num_iterations=num_iterations,
                    algorithm=RandomWalkMH_GPU_Optimized,
                    target_dist=target_dist,
                    device=str(self.device),
                    pre_allocate=True,
                    seed=42,
                    burn_in=1000
                )
                
                # Warm-up run (not timed)
                simulation.generate_samples(progress_bar=False)
                simulation.reset()
                
                # Timed run
                start_time = time.time()
                chain = simulation.generate_samples(progress_bar=False)
                run_time = time.time() - start_time
                
                acceptance_rate = simulation.acceptance_rate()
                esjd = simulation.expected_squared_jump_distance()
                
                results[proposal_name] = {
                    'run_time': run_time,
                    'samples_per_second': num_iterations / run_time,
                    'acceptance_rate': acceptance_rate,
                    'esjd': esjd,
                    'efficiency': esjd * acceptance_rate  # Simple efficiency metric
                }
                
                self.log(f"Time: {run_time:.2f}s, Rate: {num_iterations/run_time:.0f} samples/s", 3)
                self.log(f"AR: {acceptance_rate:.3f}, ESJD: {esjd:.6f}, Efficiency: {results[proposal_name]['efficiency']:.6f}", 3)
                
            except Exception as e:
                results[proposal_name] = {
                    'error': str(e),
                    'run_time': None,
                    'samples_per_second': None,
                    'acceptance_rate': None,
                    'esjd': None,
                    'efficiency': None
                }
                self.log(f"❌ Failed: {str(e)}", 3)
        
        return results
    
    def test_beta_scaling_effects(self) -> Dict[str, Dict[str, List[float]]]:
        """Test how beta (inverse temperature) affects proposal distributions."""
        self.log("🌡️  Testing Beta Scaling Effects")
        results = {}
        
        dim = 3
        beta_values = [0.5, 1.0, 2.0, 4.0]
        n_samples = 5000
        
        for proposal_type in ['Normal', 'Laplace', 'UniformRadius']:
            self.log(f"Testing {proposal_type} beta scaling", 2)
            
            variances = []
            means = []
            
            for beta in beta_values:
                if proposal_type == 'Normal':
                    proposal = NormalProposal(
                        dim=dim, base_variance_scalar=1.0, beta=beta, device=self.device, dtype=torch.float32
                    )
                elif proposal_type == 'Laplace':
                    proposal = LaplaceProposal(
                        dim=dim, base_variance_vector=torch.ones(dim), beta=beta, device=self.device, dtype=torch.float32
                    )
                else:  # UniformRadius
                    proposal = UniformRadiusProposal(
                        dim=dim, base_radius=1.0, beta=beta, device=self.device, dtype=torch.float32
                    )
                
                samples = proposal.sample(n_samples)
                sample_mean = torch.mean(torch.linalg.norm(samples, dim=1)).item()
                sample_var = torch.mean(torch.var(samples, dim=0)).item()
                
                means.append(sample_mean)
                variances.append(sample_var)
                
                self.log(f"β={beta}: mean_norm={sample_mean:.4f}, variance={sample_var:.4f}", 3)
            
            results[proposal_type] = {
                'beta_values': beta_values,
                'mean_norms': means,
                'variances': variances
            }
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        self.log("🚀 Running Comprehensive Proposal Distribution Tests")
        
        all_results = {}
        
        # Run all test categories
        all_results['creation'] = self.test_proposal_creation()
        all_results['statistical_properties'] = self.test_statistical_properties()
        all_results['mcmc_integration'] = self.test_mcmc_integration()
        all_results['multiple_targets'] = self.test_multiple_target_distributions()
        all_results['performance'] = self.test_performance_comparison()
        all_results['beta_scaling'] = self.test_beta_scaling_effects()
        
        # Generate summary
        self.log("\n" + "="*70)
        self.log("📋 COMPREHENSIVE TEST SUMMARY")
        self.log("="*70)
        
        # Creation tests
        creation_success = all(all_results['creation'][key] for key in ['Normal', 'Laplace', 'UniformRadius', 'error_handling'])
        self.log(f"Proposal Creation: {'✅ PASS' if creation_success else '❌ FAIL'}")
        
        # Statistical properties
        stat_success = True
        for prop_type, stats in all_results['statistical_properties'].items():
            if isinstance(stats, dict):
                type_success = stats.get('sample_shape_correct', False) and stats.get('mean_error', 1.0) < 0.1
                stat_success &= type_success
                
        self.log(f"Statistical Properties: {'✅ PASS' if stat_success else '❌ FAIL'}")
        
        # MCMC integration
        mcmc_success = all(result.get('success', False) for result in all_results['mcmc_integration'].values())
        self.log(f"MCMC Integration: {'✅ PASS' if mcmc_success else '❌ FAIL'}")
        
        # Multiple targets
        target_success = all(result.get('success', False) for result in all_results['multiple_targets'].values())
        self.log(f"Multiple Targets: {'✅ PASS' if target_success else '❌ FAIL'}")
        
        # Performance comparison
        perf_success = all('error' not in result for result in all_results['performance'].values())
        self.log(f"Performance Tests: {'✅ PASS' if perf_success else '❌ FAIL'}")
        
        overall_success = all([creation_success, stat_success, mcmc_success, target_success, perf_success])
        self.log(f"\n🎯 OVERALL RESULT: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        if overall_success:
            self.log("\n🎉 Proposal distribution system is working correctly!")
            
            # Print some performance highlights
            if all_results['performance']:
                self.log("\n⚡ Performance Highlights:")
                for prop_name, perf in all_results['performance'].items():
                    if 'samples_per_second' in perf and perf['samples_per_second']:
                        self.log(f"   {prop_name}: {perf['samples_per_second']:.0f} samples/s, efficiency: {perf.get('efficiency', 0):.6f}", 2)
        
        return all_results


def main():
    """Main function to run the comprehensive tests."""
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running tests on device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # Run comprehensive tests
    tester = ProposalTester(device=device, verbose=True)
    results = tester.run_all_tests()
    
    return results


if __name__ == "__main__":
    results = main() 