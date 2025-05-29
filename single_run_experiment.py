import argparse
import time
import torch
from interfaces import MCMCSimulation_GPU
from algorithms import *
import numpy as np
from target_distributions import *
import matplotlib.pyplot as plt
import json
import os

def calculate_hybrid_rosenbrock_dim(n1, n2):
    """Calculate the dimension for HybridRosenbrock: 1 + n2 * (n1 - 1)"""
    return 1 + n2 * (n1 - 1)

def calculate_super_funnel_dim(J, K):
    """Calculate the dimension for SuperFunnel: J + J*K + 1 + K + 1 + 1"""
    return J + J * K + 1 + K + 1 + 1

def get_target_distribution(name, dim, use_torch=True, **kwargs):
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
        elif name == "FullRosenbrock":
            a_coeff = kwargs.get('a_coeff', 1.0/20.0)
            b_coeff = kwargs.get('b_coeff', 100.0/20.0)
            mu = kwargs.get('mu', 1.0)
            device = kwargs.get('device', None)
            return FullRosenbrockTorch(dim, a_coeff=a_coeff, b_coeff=b_coeff, mu=mu, device=device)
        elif name == "EvenRosenbrock":
            a_coeff = kwargs.get('a_coeff', 1.0/20.0)
            b_coeff = kwargs.get('b_coeff', 100.0/20.0)
            mu = kwargs.get('mu', 1.0)
            device = kwargs.get('device', None)
            return EvenRosenbrockTorch(dim, a_coeff=a_coeff, b_coeff=b_coeff, mu=mu, device=device)
        elif name == "HybridRosenbrock":
            n1 = kwargs.get('n1', 3)
            n2 = kwargs.get('n2', 5)
            a_coeff = kwargs.get('a_coeff', 1.0/20.0)
            b_coeff = kwargs.get('b_coeff', 100.0/20.0)
            mu = kwargs.get('mu', 1.0)
            device = kwargs.get('device', None)
            # For HybridRosenbrock, dim is calculated from n1 and n2
            return HybridRosenbrockTorch(n1=n1, n2=n2, a_coeff=a_coeff, b_coeff=b_coeff, mu=mu, device=device)
        elif name == "NealFunnel":
            mu_v = kwargs.get('mu_v', 0.0)
            sigma_v_sq = kwargs.get('sigma_v_sq', 9.0)
            mu_z = kwargs.get('mu_z', 0.0)
            device = kwargs.get('device', None)
            return NealFunnelTorch(dim, mu_v=mu_v, sigma_v_sq=sigma_v_sq, mu_z=mu_z, device=device)
        elif name == "SuperFunnel":
            # SuperFunnel requires synthetic data generation
            J = kwargs.get('J', 5)  # Number of groups
            K = kwargs.get('K', 3)  # Number of features
            n_per_group = kwargs.get('n_per_group', 20)  # Observations per group
            prior_hypermean_std = kwargs.get('prior_hypermean_std', 10.0)
            prior_tau_scale = kwargs.get('prior_tau_scale', 2.5)
            device = kwargs.get('device', None)
            
            # Generate synthetic data for SuperFunnel
            torch.manual_seed(42)  # For reproducible synthetic data
            X_data = []
            Y_data = []
            for j in range(J):
                # Generate random design matrix for group j
                X_j = torch.randn(n_per_group, K)
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
        elif name in ["FullRosenbrock", "EvenRosenbrock", "HybridRosenbrock", "NealFunnel", "SuperFunnel"]:
            raise ValueError(f"{name} distribution only available with PyTorch (use_torch=True)")
        else:
            raise ValueError("Unknown target distribution name")

def run_single_simulation(dim, target_name="MultivariateNormal", num_iters=50000, 
                         proposal_variance=None, seed=42, burn_in=1000, **kwargs):
    """Run a single MCMC simulation with specified proposal variance and plot results immediately."""
    
    # Handle special dimension calculations
    if target_name == "HybridRosenbrock":
        n1 = kwargs.get('n1', 3)
        n2 = kwargs.get('n2', 5)
        actual_dim = calculate_hybrid_rosenbrock_dim(n1, n2)
        print(f"\n{'='*60}")
        print(f"Target: {target_name}, n1={n1}, n2={n2}, Actual Dimension: {actual_dim}")
        print(f"Samples: {num_iters}, Burn-in: {burn_in}, Seed: {seed}")
        print(f"{'='*60}")
    elif target_name == "SuperFunnel":
        J = kwargs.get('J', 5)
        K = kwargs.get('K', 3)
        actual_dim = calculate_super_funnel_dim(J, K)
        print(f"\n{'='*60}")
        print(f"Target: {target_name}, J={J}, K={K}, Actual Dimension: {actual_dim}")
        print(f"Samples: {num_iters}, Burn-in: {burn_in}, Seed: {seed}")
        print(f"{'='*60}")
    else:
        actual_dim = dim
        print(f"\n{'='*60}")
        print(f"Target: {target_name}, Dimension: {dim}")
        print(f"Samples: {num_iters}, Burn-in: {burn_in}, Seed: {seed}")
        print(f"{'='*60}")
    
    # Set default proposal variance if not provided
    if proposal_variance is None:
        # Use a reasonable default: optimal scaling suggests variance ~ 2.38^2 / dim
        proposal_variance = (2.38 ** 2) / actual_dim
        print(f"Using default proposal variance: {proposal_variance:.6f}")
    else:
        print(f"Using specified proposal variance: {proposal_variance:.6f}")
    
    # Create target distribution
    target_distribution = get_target_distribution(target_name, dim, use_torch=True, **kwargs)
    
    print(f"\nRunning MCMC simulation...")
    start_time = time.time()
    
    # Run simulation
    simulation = MCMCSimulation_GPU(
        dim=actual_dim,
        sigma=proposal_variance,
        num_iterations=num_iters,
        algorithm=RandomWalkMH_GPU_Optimized,
        target_dist=target_distribution,
        symmetric=True,
        pre_allocate=True,
        seed=seed,
        burn_in=burn_in
    )
    
    chain = simulation.generate_samples(progress_bar=True)
    
    simulation_time = time.time() - start_time
    acceptance_rate = simulation.acceptance_rate()
    esjd = simulation.expected_squared_jump_distance()
    
    print(f"\nSimulation Results:")
    print(f"   Time: {simulation_time:.1f} seconds")
    print(f"   Acceptance rate: {acceptance_rate:.3f}")
    print(f"   ESJD: {esjd:.6f}")
    print(f"   Samples per second: {num_iters/simulation_time:.0f}")
    
    # Save results
    data = {
        'target_distribution': target_name,
        'dimension': actual_dim,
        'num_iterations': num_iters,
        'proposal_variance': proposal_variance,
        'seed': seed,
        'simulation_time': simulation_time,
        'acceptance_rate': acceptance_rate,
        'esjd': esjd,
        'burn_in': burn_in
    }
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    filename = f"data/{target_name}_single_run_dim{actual_dim}_{num_iters}iters_var{proposal_variance:.6f}_seed{seed}.json"
    with open(filename, "w") as file:
        json.dump(data, file, indent=2)
    print(f"   Results saved to: {filename}")
    
    # Get chain data for plotting
    if hasattr(simulation.algorithm, 'get_chain_gpu'):
        chain_data = simulation.algorithm.get_chain_gpu().cpu().numpy()
    else:
        chain_data = np.array(simulation.algorithm.chain)
    
    # Apply burn-in for visualization
    if len(chain_data) > burn_in:
        chain_data = chain_data[burn_in:]
    
    print(f"\nGenerating plots...")
    
    # Ensure images directory exists
    os.makedirs("images", exist_ok=True)
    
    # Create traceplot
    plt.figure(figsize=(12, 8))
    
    # Determine number of dimensions to plot (max 4)
    num_dims_to_plot = min(4, actual_dim)
    
    if num_dims_to_plot == 1:
        # Single dimension plot
        plt.plot(chain_data[:, 0], alpha=0.7, linewidth=0.5, color='blue')
        plt.xlabel('Iteration (after burn-in)')
        plt.ylabel('Value')
        plt.title(f'Traceplot - {target_name} (Dimension 1)\n'
                 f'Proposal variance: {proposal_variance:.6f}, Acceptance rate: {acceptance_rate:.3f}, ESJD: {esjd:.6f}')
        plt.grid(True, alpha=0.3)
    else:
        # Multiple dimensions subplot
        for i in range(num_dims_to_plot):
            plt.subplot(num_dims_to_plot, 1, i + 1)
            plt.plot(chain_data[:, i], alpha=0.7, linewidth=0.5, color=f'C{i}')
            plt.ylabel(f'Dimension {i + 1}')
            plt.grid(True, alpha=0.3)
            
            if i == 0:
                plt.title(f'Traceplot - {target_name} (First {num_dims_to_plot} dimensions)\n'
                         f'Proposal variance: {proposal_variance:.6f}, Acceptance rate: {acceptance_rate:.3f}, ESJD: {esjd:.6f}')
            if i == num_dims_to_plot - 1:
                plt.xlabel('Iteration (after burn-in)')
    
    plt.tight_layout()
    
    # Save traceplot
    traceplot_filename = f"images/traceplot_{target_name}_single_run_dim{actual_dim}_{num_iters}iters_var{proposal_variance:.6f}_seed{seed}.png"
    plt.savefig(traceplot_filename, dpi=300, bbox_inches='tight')
    plt.show()  # Display immediately
    plt.close()
    print(f"   Traceplot saved as '{traceplot_filename}'")
    
    # Create 2D density visualization if dimension >= 2
    if actual_dim >= 2:
        print(f"   Creating 2D density visualization...")
        
        plt.figure(figsize=(10, 8))
        
        # Extract first two dimensions of the chain
        x_chain = chain_data[:, 0]
        y_chain = chain_data[:, 1]
        
        # Determine plot bounds based on chain data with minimal padding
        x_min, x_max = np.min(x_chain), np.max(x_chain)
        y_min, y_max = np.min(y_chain), np.max(y_chain)
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.05  # 5% padding
        
        x_plot_min = x_min - padding * x_range
        x_plot_max = x_max + padding * x_range
        y_plot_min = y_min - padding * y_range
        y_plot_max = y_max + padding * y_range
        
        # Create grid for density evaluation
        x_grid = np.linspace(x_plot_min, x_plot_max, 80)
        y_grid = np.linspace(y_plot_min, y_plot_max, 80)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Evaluate target density on the grid
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Create a point with the first two dimensions from the grid
                # and remaining dimensions set to mean values from chain
                point = np.zeros(actual_dim)
                point[0] = X[i, j]
                point[1] = Y[i, j]
                
                # For higher dimensions, use the mean of the chain for other dimensions
                if actual_dim > 2:
                    point[2:] = np.mean(chain_data[:, 2:], axis=0)
                
                # Evaluate density
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
        
        # Plot MCMC samples
        # Use subset of samples for visualization (every 10th sample)
        sample_indices = np.arange(0, len(x_chain), max(1, len(x_chain)//1000))
        plt.scatter(x_chain[sample_indices], y_chain[sample_indices], 
                   c='red', s=2, alpha=0.6, zorder=5, label='MCMC Samples')
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'2D Target Density with MCMC Samples - {target_name}\n'
                 f'Proposal variance: {proposal_variance:.6f}, Acceptance rate: {acceptance_rate:.3f}, ESJD: {esjd:.6f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save 2D visualization
        density_filename = f"images/density2D_{target_name}_single_run_dim{actual_dim}_{num_iters}iters_var{proposal_variance:.6f}_seed{seed}.png"
        plt.savefig(density_filename, dpi=300, bbox_inches='tight')
        plt.show()  # Display immediately
        plt.close()
        print(f"   2D density visualization saved as '{density_filename}'")
    
    # Create histogram of marginal distributions
    if actual_dim <= 6:  # Only for low dimensions to keep plots readable
        print(f"   Creating marginal distribution histograms...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(min(actual_dim, 6)):
            axes[i].hist(chain_data[:, i], bins=50, alpha=0.7, density=True, color=f'C{i}')
            axes[i].set_xlabel(f'Dimension {i + 1}')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'Marginal Distribution - Dim {i + 1}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(actual_dim, 6):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Marginal Distributions - {target_name}\n'
                    f'Proposal variance: {proposal_variance:.6f}, Acceptance rate: {acceptance_rate:.3f}')
        plt.tight_layout()
        
        # Save histogram
        hist_filename = f"images/histograms_{target_name}_single_run_dim{actual_dim}_{num_iters}iters_var{proposal_variance:.6f}_seed{seed}.png"
        plt.savefig(hist_filename, dpi=300, bbox_inches='tight')
        plt.show()  # Display immediately
        plt.close()
        print(f"   Marginal histograms saved as '{hist_filename}'")
    
    print(f"\nAll plots generated and displayed!")
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single GPU-accelerated RWM simulation with immediate plotting")
    parser.add_argument("--dim", type=int, default=10, help="Dimension of the target distribution")
    parser.add_argument("--target", type=str, default="MultivariateNormal", help="Target distribution")
    parser.add_argument("--num_iters", type=int, default=50000, help="Number of iterations")
    parser.add_argument("--proposal_variance", type=float, default=None, help="Proposal variance (default: 2.38^2/dim)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--burn_in", type=int, default=1000, help="Burn-in period")
    
    # Distribution-specific parameters
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
    
    results = run_single_simulation(
        dim=args.dim, 
        target_name=args.target, 
        num_iters=args.num_iters, 
        proposal_variance=args.proposal_variance,
        seed=args.seed, 
        burn_in=args.burn_in, 
        **kwargs
    )

    print(f"\nExperiment completed successfully!") 