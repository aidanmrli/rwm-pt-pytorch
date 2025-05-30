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
import os

def calculate_hybrid_rosenbrock_dim(n1, n2):
    """Calculate the dimension for HybridRosenbrock: 1 + n2 * (n1 - 1)"""
    return 1 + n2 * (n1 - 1)

def calculate_super_funnel_dim(J, K):
    """Calculate the dimension for SuperFunnel: J + J*K + 1 + K + 1 + 1"""
    return J + J * K + 1 + K + 1 + 1

def get_target_distribution(name, dim, use_torch=True, device=None, **kwargs):
    """Get target distribution with optional GPU acceleration."""
    
    # Set device if not provided
    if device is None and use_torch:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_torch:
        # Use PyTorch-native implementations for GPU acceleration
        if name == "MultivariateNormal":
            return MultivariateNormalTorch(dim, device=device)
        elif name == "MultivariateNormalScaled":
            return ScaledMultivariateNormalTorch(dim, device=device)
        elif name == "RoughCarpet":
            # Custom mode centers and weights for better separation
            mode_centers = kwargs.get('mode_centers', [-15.0, 0.0, 15.0]) 
            mode_weights = kwargs.get('mode_weights', [0.5, 0.3, 0.2])   
            return RoughCarpetDistributionTorch(dim, scaling=False, device=device, 
                                              mode_centers=mode_centers, mode_weights=mode_weights)
        elif name == "RoughCarpetScaled":
            # Custom mode centers and weights for better separation
            mode_centers = kwargs.get('mode_centers', [-15.0, 0.0, 15.0])
            mode_weights = kwargs.get('mode_weights', [0.5, 0.3, 0.2])   
            return RoughCarpetDistributionTorch(dim, scaling=True, device=device,
                                              mode_centers=mode_centers, mode_weights=mode_weights)
        elif name == "ThreeMixture":
            # Custom mode centers for better separation (4.0 units between adjacent modes)
            mode_centers = kwargs.get('mode_centers', [
                [-15.0] + [0.0] * (dim - 1),  # (-4, 0, ..., 0)
                [0.0] * dim,                  # (0, 0, ..., 0)
                [15.0] + [0.0] * (dim - 1)    # (4, 0, ..., 0)
            ])
            mode_weights = kwargs.get('mode_weights', [1/3, 1/3, 1/3])
            return ThreeMixtureDistributionTorch(dim, scaling=False, device=device,
                                               mode_centers=mode_centers, mode_weights=mode_weights)
        elif name == "ThreeMixtureScaled":
            # Custom mode centers for better separation (4.0 units between adjacent modes)
            mode_centers = kwargs.get('mode_centers', [
                [-15.0] + [0.0] * (dim - 1),  # (-4, 0, ..., 0)
                [0.0] * dim,                  # (0, 0, ..., 0)
                [15.0] + [0.0] * (dim - 1)    # (4, 0, ..., 0)
            ])
            mode_weights = kwargs.get('mode_weights', [1/3, 1/3, 1/3])  
            return ThreeMixtureDistributionTorch(dim, scaling=True, device=device,
                                               mode_centers=mode_centers, mode_weights=mode_weights)
        elif name == "Hypercube":
            return HypercubeTorch(dim, left_boundary=-1, right_boundary=1, device=device)
        elif name == "IIDGamma":
            return IIDGammaTorch(dim, shape=2, scale=3, device=device)
        elif name == "IIDBeta":
            return IIDBetaTorch(dim, alpha=2, beta=3, device=device)
        elif name == "FullRosenbrock":
            a_coeff = kwargs.get('a_coeff', 1.0/20.0)
            b_coeff = kwargs.get('b_coeff', 100.0/20.0)
            mu = kwargs.get('mu', 1.0)
            return FullRosenbrockTorch(dim, a_coeff=a_coeff, b_coeff=b_coeff, mu=mu, device=device)
        elif name == "EvenRosenbrock":
            a_coeff = kwargs.get('a_coeff', 1.0/20.0)
            b_coeff = kwargs.get('b_coeff', 100.0/20.0)
            mu = kwargs.get('mu', 1.0)
            return EvenRosenbrockTorch(dim, a_coeff=a_coeff, b_coeff=b_coeff, mu=mu, device=device)
        elif name == "HybridRosenbrock":
            n1 = kwargs.get('n1', 3)
            n2 = kwargs.get('n2', 5)
            a_coeff = kwargs.get('a_coeff', 1.0/20.0)
            b_coeff = kwargs.get('b_coeff', 100.0/20.0)
            mu = kwargs.get('mu', 1.0)
            # For HybridRosenbrock, dim is calculated from n1 and n2
            return HybridRosenbrockTorch(n1=n1, n2=n2, a_coeff=a_coeff, b_coeff=b_coeff, mu=mu, device=device)
        elif name == "NealFunnel":
            mu_v = kwargs.get('mu_v', 0.0)
            sigma_v_sq = kwargs.get('sigma_v_sq', 9.0)
            mu_z = kwargs.get('mu_z', 0.0)
            return NealFunnelTorch(dim, mu_v=mu_v, sigma_v_sq=sigma_v_sq, mu_z=mu_z, device=device)
        elif name == "SuperFunnel":
            # SuperFunnel requires synthetic data generation
            J = kwargs.get('J', 5)  # Number of groups
            K = kwargs.get('K', 3)  # Number of features
            n_per_group = kwargs.get('n_per_group', 20)  # Observations per group
            prior_hypermean_std = kwargs.get('prior_hypermean_std', 10.0)
            prior_tau_scale = kwargs.get('prior_tau_scale', 2.5)
            
            # Generate synthetic data for SuperFunnel on the correct device
            torch.manual_seed(42)  # For reproducible synthetic data
            X_data = []
            Y_data = []
            for j in range(J):
                # Generate random design matrix for group j
                X_j = torch.randn(n_per_group, K, device=device)
                # Generate synthetic binary outcomes
                # Use simple logistic model: logit(p) = 0.5 * sum(X_j, dim=1)
                logits = 0.5 * torch.sum(X_j, dim=1)
                probs = torch.sigmoid(logits)
                Y_j = torch.bernoulli(probs)
                X_data.append(X_j)
                Y_data.append(Y_j)
            
            return SuperFunnelTorch(J, K, X_data, Y_data, 
                                  prior_hypermean_std=prior_hypermean_std, 
                                  prior_tau_scale=prior_tau_scale, 
                                  device=device)
        else:
            raise ValueError("Unknown target distribution name")
    else:
        # Fall back to CPU versions
        if name == "MultivariateNormal":
            return MultivariateNormal(dim)
        elif name == "RoughCarpet":
            mode_centers = kwargs.get('mode_centers', [-4.0, 0.0, 4.0])  
            mode_weights = kwargs.get('mode_weights', [0.5, 0.3, 0.2])   
            return RoughCarpetDistribution(dim, scaling=False, mode_centers=mode_centers, mode_weights=mode_weights)
        elif name == "RoughCarpetScaled":
            mode_centers = kwargs.get('mode_centers', [-4.0, 0.0, 4.0])  
            mode_weights = kwargs.get('mode_weights', [0.5, 0.3, 0.2])   
            return RoughCarpetDistribution(dim, scaling=True, mode_centers=mode_centers, mode_weights=mode_weights)
        elif name == "ThreeMixture":
            # Custom mode centers for better separation (4.0 units between adjacent modes)
            mode_centers = kwargs.get('mode_centers', [
                [-5.0] + [0.0] * (dim - 1),  # (-4, 0, ..., 0)
                [0.0] * dim,                  # (0, 0, ..., 0)
                [5.0] + [0.0] * (dim - 1)    # (4, 0, ..., 0)
            ])
            mode_weights = kwargs.get('mode_weights', [1/3, 1/3, 1/3])  
            return ThreeMixtureDistribution(dim, scaling=False, mode_centers=mode_centers, mode_weights=mode_weights)
        elif name == "ThreeMixtureScaled":
            # Custom mode centers for better separation (4.0 units between adjacent modes)
            mode_centers = kwargs.get('mode_centers', [
                [-5.0] + [0.0] * (dim - 1),  # (-4, 0, ..., 0)
                [0.0] * dim,                  # (0, 0, ..., 0)
                [5.0] + [0.0] * (dim - 1)    # (4, 0, ..., 0)
            ])
            mode_weights = kwargs.get('mode_weights', [1/3, 1/3, 1/3])  
            return ThreeMixtureDistribution(dim, scaling=True, mode_centers=mode_centers, mode_weights=mode_weights)
        elif name == "Hypercube":
            return Hypercube(dim, left_boundary=-1, right_boundary=1)
        elif name == "IIDGamma":
            return IIDGamma(dim, shape=2, scale=3)
        elif name == "IIDBeta":
            return IIDBeta(dim, alpha=2, beta=3)
        elif name in ["FullRosenbrock", "EvenRosenbrock", "HybridRosenbrock", "NealFunnel", "SuperFunnel"]:
            raise ValueError(f"{name} distribution only available with PyTorch (use_torch=True)")
        else:
            raise ValueError("Unknown target distribution name")

def run_study(dim, target_name="ThreeMixture", num_iters=100000, swap_accept_max=0.5, seed=42, burn_in=1000, **kwargs):
    """Run many parallel tempering simulations with different swap acceptance rates, then save and plot the progression of ESJD and acceptance rate."""
    
    # Set device explicitly
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Running on CPU (performance will be limited).")
    
    # Handle special dimension calculations
    if target_name == "HybridRosenbrock":
        n1 = kwargs.get('n1', 3)
        n2 = kwargs.get('n2', 5)
        actual_dim = calculate_hybrid_rosenbrock_dim(n1, n2)
        print(f"\n{'='*60}")
        print(f"Target: {target_name}, n1={n1}, n2={n2}, Actual Dimension: {actual_dim}, Samples: {num_iters}, Burn-in: {burn_in}, Seed: {seed}")
        print(f"{'='*60}")
    elif target_name == "SuperFunnel":
        J = kwargs.get('J', 5)
        K = kwargs.get('K', 3)
        actual_dim = calculate_super_funnel_dim(J, K)
        print(f"\n{'='*60}")
        print(f"Target: {target_name}, J={J}, K={K}, Actual Dimension: {actual_dim}, Samples: {num_iters}, Burn-in: {burn_in}, Seed: {seed}")
        print(f"{'='*60}")
    else:
        actual_dim = dim
        print(f"\n{'='*60}")
        print(f"Target: {target_name}, Dimension: {dim}, Samples: {num_iters}, Burn-in: {burn_in}, Seed: {seed}")
        print(f"{'='*60}")
    
    target_distribution = get_target_distribution(target_name, dim, use_torch=True, device=device, **kwargs)
    swap_acceptance_rates_range = np.linspace(0.01, swap_accept_max, 30)
    
    acceptance_rates = []
    expected_squared_jump_distances = []
    times = []
    
    print(f"\nRunning parallel tempering simulations with {len(swap_acceptance_rates_range)} swap acceptance rate values...")
    
    total_start = time.time()
    constructed_beta_ladder = None
    
    # Use tqdm for progress bar
    for i, target_swap_rate in enumerate(tqdm.tqdm(swap_acceptance_rates_range, desc="Running PT with target swap rate =", unit="config")):
        # Use standard PT scaling: sigma = (2.38**2) / dim
        proposal_variance = ((2.38 ** 2) / (actual_dim ** (1)))
        iteration_start = time.time()
        
        simulation = MCMCSimulation_GPU(
            dim=actual_dim,
            sigma=proposal_variance,
            num_iterations=num_iters,
            algorithm=ParallelTemperingRWM_GPU_Optimized,
            target_dist=target_distribution,
            symmetric=True,
            pre_allocate=True,
            seed=seed,
            burn_in=burn_in,
            device=device,
            beta_ladder=None, # construct it each new target swap rate
            swap_acceptance_rate=target_swap_rate,
            iterative_temp_spacing=True  # Use iterative ladder construction as default
        )
        
        chain = simulation.generate_samples(progress_bar=False)
        
        iteration_time = time.time() - iteration_start
        times.append(iteration_time)
        
        acceptance_rates.append(simulation.algorithm.swap_acceptance_rate)
        expected_squared_jump_distances.append(simulation.pt_expected_squared_jump_distance())
    
    total_time = time.time() - total_start
    
    max_esjd = max(expected_squared_jump_distances)
    max_esjd_index = np.argmax(expected_squared_jump_distances)
    max_actual_acceptance_rate = acceptance_rates[max_esjd_index]
    max_constr_acceptance_rate = swap_acceptance_rates_range[max_esjd_index]
    
    print(f"\nFinal Results:")
    print(f"   Total time: {total_time:.1f} seconds")
    print(f"   Average time per configuration: {np.mean(times):.1f} seconds")
    print(f"   Maximum ESJD: {max_esjd:.6f}")
    print(f"   (Actual) Swap acceptance rate corresponding to maximum ESJD: {max_actual_acceptance_rate:.3f}")
    print(f"   (Construction) Swap acceptance rate value corresponding to maximum ESJD: {max_constr_acceptance_rate:.3f}")
    
    # Save results
    data = {
        'target_distribution': target_name,
        'dimension': actual_dim,
        'num_iterations': num_iters,
        'seed': seed,
        'total_time': total_time,
        'max_esjd': max_esjd,
        'max_actual_acceptance_rate': max_actual_acceptance_rate,
        'max_constr_acceptance_rate': max_constr_acceptance_rate,
        'expected_squared_jump_distances': expected_squared_jump_distances,
        'acceptance_rates': acceptance_rates,
        'swap_acceptance_rates_range': swap_acceptance_rates_range.tolist(),
        'times': times,
    }
    
    filename = f"data/{target_name}_PT_GPU_dim{actual_dim}_{num_iters}iters_seed{seed}.json"
    with open(filename, "w") as file:
        json.dump(data, file, indent=2)
    print(f"   Results saved to: {filename}")
    
    # Create traceplot using optimal swap acceptance rate
    print(f"\nGenerating traceplot with optimal swap acceptance rate ({max_constr_acceptance_rate:.6f})...")
    
    # Run one more simulation with optimal swap acceptance rate to get chain for traceplot
    traceplot_simulation = MCMCSimulation_GPU(
        dim=actual_dim,
        sigma=proposal_variance,
        num_iterations=num_iters,
        algorithm=ParallelTemperingRWM_GPU_Optimized,
        target_dist=target_distribution,
        symmetric=True,
        pre_allocate=True,
        seed=seed,
        burn_in=burn_in,
        device=device,
        beta_ladder=None,
        swap_acceptance_rate=max_constr_acceptance_rate,
        iterative_temp_spacing=True  # Use iterative ladder construction as default
    )
    
    traceplot_chain = traceplot_simulation.generate_samples(progress_bar=False)
    
    # Create traceplot figure
    plt.figure(figsize=(12, 8))
    
    # Get chain data - use GPU tensor if available for efficiency (cold chain only)
    if hasattr(traceplot_simulation.algorithm, 'get_cold_chain_gpu'):
        chain_data = traceplot_simulation.algorithm.get_cold_chain_gpu().cpu().numpy()
    else:
        chain_data = np.array(traceplot_simulation.algorithm.chain)
    
    # Apply burn-in (skip first burn_in samples for visualization)
    burn_in_samples = burn_in
    if len(chain_data) > burn_in_samples:
        chain_data = chain_data[burn_in_samples:]
    
    # Determine number of dimensions to plot (max 3)
    num_dims_to_plot = min(3, actual_dim)
    
    if num_dims_to_plot == 1:
        # Single dimension plot
        plt.plot(chain_data[:, 0], alpha=0.7, linewidth=0.5, color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title(f'Traceplot - {target_name} (Dimension 1)\nOptimal swap rate: {max_constr_acceptance_rate:.6f}, Actual swap rate: {max_actual_acceptance_rate:.3f}')
        plt.grid(True, alpha=0.3)
    else:
        # Multiple dimensions subplot
        for i in range(num_dims_to_plot):
            plt.subplot(num_dims_to_plot, 1, i + 1)
            plt.plot(chain_data[:, i], alpha=0.7, linewidth=0.5, color=f'C{i}')
            plt.ylabel(f'Dimension {i + 1}')
            plt.grid(True, alpha=0.3)
            
            if i == 0:
                plt.title(f'Traceplot - {target_name} (First {num_dims_to_plot} dimensions)\nOptimal swap rate: {max_constr_acceptance_rate:.6f}, Actual swap rate: {max_actual_acceptance_rate:.3f}')
            if i == num_dims_to_plot - 1:
                plt.xlabel('Iteration')
    
    plt.tight_layout()
    
    # Ensure images directory exists
    os.makedirs("images", exist_ok=True)
    
    # Save traceplot
    output_filename = f"images/traceplot_{target_name}_PT_GPU_dim{actual_dim}_{num_iters}iters_seed{seed}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()
    print(f"   Traceplot created and saved as '{output_filename}'")
    
    # Create 2D density visualization with MCMC trajectory overlay
    if actual_dim >= 2:
        print(f"Creating 2D density visualization with MCMC trajectory...")
        
        plt.figure(figsize=(10, 8))
        
        # Extract first two dimensions of the chain
        x_chain = chain_data[:, 0]
        y_chain = chain_data[:, 1]
        
        # Determine plot bounds based on chain data with very minimal padding
        x_min, x_max = np.min(x_chain), np.max(x_chain)
        y_min, y_max = np.min(y_chain), np.max(y_chain)
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.02  # 2% padding
        
        x_plot_min = x_min - padding * x_range
        x_plot_max = x_max + padding * x_range
        y_plot_min = y_min - padding * y_range
        y_plot_max = y_max + padding * y_range
        
        # Create grid for density evaluation
        x_grid = np.linspace(x_plot_min, x_plot_max, 100)
        y_grid = np.linspace(y_plot_min, y_plot_max, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Evaluate target density on the grid
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Create a point with the first two dimensions from the grid
                # and remaining dimensions set to zero (or mean values)
                point = np.zeros(actual_dim)
                point[0] = X[i, j]
                point[1] = Y[i, j]
                
                # For higher dimensions, use the mean of the chain for other dimensions
                if actual_dim > 2:
                    point[2:] = np.mean(chain_data[:, 2:], axis=0)
                
                # Evaluate density - handle both PyTorch and numpy distributions
                try:
                    if hasattr(target_distribution, 'density'):
                        # Check if it's a PyTorch distribution that needs tensor input
                        if hasattr(target_distribution, 'device') or isinstance(target_distribution, torch.nn.Module):
                            point_tensor = torch.tensor(point, dtype=torch.float32, 
                                                      device=getattr(target_distribution, 'device', 'cpu'))
                            density_val = target_distribution.density(point_tensor)
                            Z[i, j] = density_val.item() if torch.is_tensor(density_val) else density_val
                        else:
                            # CPU/numpy distribution
                            Z[i, j] = target_distribution.density(point)
                    elif hasattr(target_distribution, 'log_density'):
                        # Convert log density to density
                        if hasattr(target_distribution, 'device') or isinstance(target_distribution, torch.nn.Module):
                            point_tensor = torch.tensor(point, dtype=torch.float32, 
                                                      device=getattr(target_distribution, 'device', 'cpu'))
                            log_dens = target_distribution.log_density(point_tensor)
                            Z[i, j] = torch.exp(log_dens).item()
                        else:
                            Z[i, j] = np.exp(target_distribution.log_density(point))
                    else:
                        # Fallback: assume uniform density
                        Z[i, j] = 1.0
                except Exception as e:
                    # If density evaluation fails, set to small positive value
                    Z[i, j] = 1e-10
        
        # Create contour plot of target density
        contour = plt.contourf(X, Y, Z, levels=20, cmap='Greys', alpha=0.7)
        plt.colorbar(contour, label='Target Density')
        
        # Add contour lines for better visualization
        plt.contour(X, Y, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        
        # Plot MCMC trajectory
        # Use 5% of total samples for trajectory visualization
        num_traj_points = int(0.05 * len(x_chain))
        if len(x_chain) > num_traj_points:
            indices = np.linspace(0, len(x_chain)-1, num_traj_points, dtype=int)
            x_traj = x_chain[indices]
            y_traj = y_chain[indices]
        else:
            x_traj = x_chain
            y_traj = y_chain
        
        # Plot trajectory as scattered points
        plt.scatter(x_traj[::max(1, len(x_traj)//200)], y_traj[::max(1, len(y_traj)//200)], 
                   c='red', s=3, alpha=0.6, zorder=5, label='MCMC Samples')
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'2D Target Density with MCMC Samples - {target_name}\n'
                 f'Optimal swap rate: {max_constr_acceptance_rate:.6f}, Actual swap rate: {max_actual_acceptance_rate:.3f}')
        plt.grid(True, alpha=0.3)
        
        # Save 2D visualization
        density_filename = f"images/density2D_{target_name}_PT_GPU_dim{actual_dim}_{num_iters}iters_seed{seed}.png"
        plt.savefig(density_filename, dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
        print(f"   2D density visualization created and saved as '{density_filename}'")

    # Create ESJD plots
    plt.figure(figsize=(12, 5))
    
    # Plot 1: ESJD vs Construction Swap Acceptance Rate
    plt.subplot(1, 2, 1)
    plt.plot(swap_acceptance_rates_range, expected_squared_jump_distances, marker='x')
    plt.axvline(x=0.234, color='red', linestyle=':', label='a = 0.234')
    plt.xlabel('Swap Acceptance Rate (Construction)')
    plt.ylabel('ESJD')
    plt.title(f'ESJD vs Swap Acceptance Rate (dim={actual_dim})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: ESJD vs Actual Swap Acceptance Rate
    plt.subplot(1, 2, 2)
    plt.plot(acceptance_rates, expected_squared_jump_distances, marker='x')   
    plt.axvline(x=0.234, color='red', linestyle=':', label='a = 0.234')
    plt.xlabel('Swap Acceptance Rate (Actual)')
    plt.ylabel('ESJD')
    plt.title(f'ESJD vs Swap Acceptance Rate (dim={actual_dim})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save ESJD plots
    esjd_filename = f"images/ESJD_vs_SwapRate_{target_name}_PT_GPU_dim{actual_dim}_{num_iters}iters_seed{seed}.png"
    plt.savefig(esjd_filename, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()
    print(f"   ESJD plots created and saved as '{esjd_filename}'")

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-accelerated Parallel Tempering simulations")
    parser.add_argument("--dim", type=int, default=20, help="Dimension of the target distribution")
    parser.add_argument("--target", type=str, default="ThreeMixture", help="Target distribution")
    parser.add_argument("--num_iters", type=int, default=200000, help="Number of iterations")
    parser.add_argument("--swap_accept_max", type=float, default=0.5, help="Maximum swap acceptance rate value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--burn_in", type=int, default=1000, help="Number of burn-in samples")
    parser.add_argument("--hybrid_rosenbrock_n1", type=int, default=3, help="Block length parameter for HybridRosenbrock")
    parser.add_argument("--hybrid_rosenbrock_n2", type=int, default=5, help="Number of blocks/rows for HybridRosenbrock")
    
    # NealFunnel parameters
    parser.add_argument("--neal_funnel_mu_v", type=float, default=0.0, help="Mean of v variable for NealFunnel")
    parser.add_argument("--neal_funnel_sigma_v_sq", type=float, default=9.0, help="Variance of v variable for NealFunnel")
    parser.add_argument("--neal_funnel_mu_z", type=float, default=0.0, help="Mean of z variables for NealFunnel")
    
    # SuperFunnel parameters
    parser.add_argument("--super_funnel_J", type=int, default=5, help="Number of groups for SuperFunnel")
    parser.add_argument("--super_funnel_K", type=int, default=3, help="Number of features for SuperFunnel")
    parser.add_argument("--super_funnel_n_per_group", type=int, default=20, help="Observations per group for SuperFunnel")
    parser.add_argument("--super_funnel_prior_hypermean_std", type=float, default=10.0, help="Prior hypermean std for SuperFunnel")
    parser.add_argument("--super_funnel_prior_tau_scale", type=float, default=2.5, help="Prior tau scale for SuperFunnel")
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        print(f"🚀 GPU detected: {torch.cuda.get_device_name()}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected. Running on CPU (will be slower)")
    
    kwargs = {}
    if args.target == "HybridRosenbrock":
        kwargs['n1'] = args.hybrid_rosenbrock_n1
        kwargs['n2'] = args.hybrid_rosenbrock_n2
    elif args.target == "NealFunnel":
        kwargs['mu_v'] = args.neal_funnel_mu_v
        kwargs['sigma_v_sq'] = args.neal_funnel_sigma_v_sq
        kwargs['mu_z'] = args.neal_funnel_mu_z
    elif args.target == "SuperFunnel":
        kwargs['J'] = args.super_funnel_J
        kwargs['K'] = args.super_funnel_K
        kwargs['n_per_group'] = args.super_funnel_n_per_group
        kwargs['prior_hypermean_std'] = args.super_funnel_prior_hypermean_std
        kwargs['prior_tau_scale'] = args.super_funnel_prior_tau_scale
    
    results = run_study(args.dim, args.target, args.num_iters, args.swap_accept_max, args.seed, args.burn_in, **kwargs)
    print(f"🎉 Finished running GPU-accelerated parallel tempering experiment.") 