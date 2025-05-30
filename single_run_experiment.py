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

def get_target_distribution(name, dim, use_torch=True, device=None, **kwargs):
    """Get target distribution with optional GPU acceleration."""
    # Set device if not provided, and use_torch is True
    if device is None and use_torch:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_torch:
        # Pass device to Torch target distributions
        if name == "MultivariateNormal":
            return MultivariateNormalTorch(dim, device=device)
        elif name == "RoughCarpet":
            return RoughCarpetDistributionTorch(dim, scaling=False, device=device)
        elif name == "RoughCarpetScaled":
            return RoughCarpetDistributionTorch(dim, scaling=True, device=device)
        elif name == "ThreeMixture":
            return ThreeMixtureDistributionTorch(dim, scaling=False, device=device)
        elif name == "ThreeMixtureScaled":
            return ThreeMixtureDistributionTorch(dim, scaling=True, device=device)
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
            return HybridRosenbrockTorch(n1=n1, n2=n2, a_coeff=a_coeff, b_coeff=b_coeff, mu=mu, device=device)
        elif name == "NealFunnel":
            mu_v = kwargs.get('mu_v', 0.0)
            sigma_v_sq = kwargs.get('sigma_v_sq', 9.0)
            mu_z = kwargs.get('mu_z', 0.0)
            return NealFunnelTorch(dim, mu_v=mu_v, sigma_v_sq=sigma_v_sq, mu_z=mu_z, device=device)
        elif name == "SuperFunnel":
            J = kwargs.get('J', 5)
            K = kwargs.get('K', 3)
            n_per_group = kwargs.get('n_per_group', 20)
            prior_hypermean_std = kwargs.get('prior_hypermean_std', 10.0)
            prior_tau_scale = kwargs.get('prior_tau_scale', 2.5)
            
            torch.manual_seed(kwargs.get('seed', 42)) # Use provided seed for data generation
            X_data = []
            Y_data = []
            for j_idx in range(J):
                X_j = torch.randn(n_per_group, K, device=device)
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
            raise ValueError(f"Unknown target distribution name: {name}")
    else:
        # Fall back to CPU versions (no device argument needed for these)
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
            raise ValueError(f"Unknown target distribution name: {name}")

def run_single_simulation(dim, target_name="MultivariateNormal", num_iters=50000, 
                         seed=42, burn_in=1000, 
                         proposal_name="Normal",
                         normal_base_variance=None,
                         laplace_base_variance_iso=None,
                         laplace_base_variance_aniso=None,
                         uniform_base_radius=None,
                         use_torch_target=True,
                         device_str=None,
                         **kwargs):
    """Run a single MCMC simulation with specified proposal and plot results immediately."""
    
    # Determine device
    if device_str is None:
        selected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        selected_device = torch.device(device_str)
    
    print(f"Using device: {selected_device}")
    
    # Handle special dimension calculations for target_name
    if target_name == "HybridRosenbrock":
        n1 = kwargs.get('n1', 3)
        n2 = kwargs.get('n2', 5)
        actual_dim = calculate_hybrid_rosenbrock_dim(n1, n2)
    elif target_name == "SuperFunnel":
        J = kwargs.get('J', 5)
        K = kwargs.get('K', 3)
        actual_dim = calculate_super_funnel_dim(J, K)
    else:
        actual_dim = dim

    print(f"\n{'='*60}")
    if target_name == "HybridRosenbrock":
        print(f"Target: {target_name}, n1={n1}, n2={n2}, Actual Dimension: {actual_dim}")
    elif target_name == "SuperFunnel":
        print(f"Target: {target_name}, J={J}, K={K}, Actual Dimension: {actual_dim}")
    else:
        print(f"Target: {target_name}, Dimension: {actual_dim}")
    print(f"Proposal: {proposal_name}")
    print(f"Samples: {num_iters}, Burn-in: {burn_in}, Seed: {seed}")
    print(f"{'='*60}")

    # Construct proposal_config
    proposal_config = {'name': proposal_name, 'params': {}}
    proposal_params_str_list = []

    if proposal_name == "Normal":
        if normal_base_variance is None:
            base_var = (2.38 ** 2) / actual_dim
            proposal_params_str_list.append(f"default base_variance_scalar={base_var:.4f}")
        else:
            base_var = (normal_base_variance ** 2) / actual_dim
            proposal_params_str_list.append(f"base_variance_scalar={base_var:.4f}")
        proposal_config['params']['base_variance_scalar'] = base_var
    elif proposal_name == "Laplace":
        if laplace_base_variance_aniso is not None:
            try:
                # Ensure it's a list of floats if it's a string
                if isinstance(laplace_base_variance_aniso, str):
                    vec = json.loads(laplace_base_variance_aniso)
                else: # Assume it's already a list/tensor
                    vec = laplace_base_variance_aniso

                if not isinstance(vec, list) or not all(isinstance(x, (float, int)) for x in vec):
                    raise ValueError("laplace_base_variance_aniso must be a list of numbers.")
                if len(vec) != actual_dim:
                    raise ValueError(f"Anisotropic Laplace variance vector length must match dimension ({actual_dim}). Got {len(vec)}.")
                proposal_config['params']['base_variance_vector'] = vec
                proposal_params_str_list.append(f"aniso base_variance_vector (first 3): {vec[:3]}...")
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string for laplace_base_variance_aniso.")
        elif laplace_base_variance_iso is not None:
            proposal_config['params']['base_variance_vector'] = torch.full((actual_dim,), float(laplace_base_variance_iso))
            proposal_params_str_list.append(f"iso base_variance_scalar={laplace_base_variance_iso:.4f}")
        else:
            # Default for Laplace if nothing specific is provided
            default_iso_var = (2.38 ** 2) / actual_dim 
            proposal_config['params']['base_variance_vector'] = torch.full((actual_dim,), default_iso_var)
            proposal_params_str_list.append(f"default iso base_variance_scalar={default_iso_var:.4f}")
    elif proposal_name == "UniformRadius":
        if uniform_base_radius is None:
            base_rad = 1.0 # A simple default radius
            proposal_params_str_list.append(f"default base_radius={base_rad:.4f}")
        else:
            base_rad = uniform_base_radius
            proposal_params_str_list.append(f"base_radius={base_rad:.4f}")
        proposal_config['params']['base_radius'] = base_rad
    else:
        raise ValueError(f"Unknown proposal name: {proposal_name}")

    print(f"Proposal Config: Name='{proposal_config['name']}', Params=({', '.join(proposal_params_str_list)})")
    
    # Create target distribution
    # Pass the selected_device to get_target_distribution
    target_distribution_kwargs = {**kwargs, 'device': selected_device, 'seed': seed}
    target_distribution = get_target_distribution(target_name, actual_dim, use_torch=use_torch_target, **target_distribution_kwargs)
    
    print(f"\nRunning MCMC simulation...")
    start_time = time.time()
    
    # Run simulation
    simulation = MCMCSimulation_GPU(
        dim=actual_dim,
        proposal_config=proposal_config,
        num_iterations=num_iters,
        algorithm=RandomWalkMH_GPU_Optimized,
        target_dist=target_distribution,
        symmetric=True,
        pre_allocate=True,
        seed=seed,
        burn_in=burn_in,
        device=str(selected_device)
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
    # Use a more descriptive params string for the filename
    proposal_param_val_str = "_".join(proposal_params_str_list).replace("=","").replace(" ","").replace("(","").replace(")","").replace(",","_").replace(".","p")

    data = {
        'target_distribution': target_name,
        'dimension': actual_dim,
        'num_iterations': num_iters,
        'proposal_name': proposal_name,
        'proposal_config_params': proposal_config['params'],
        'seed': seed,
        'simulation_time': simulation_time,
        'acceptance_rate': acceptance_rate,
        'esjd': esjd,
        'burn_in': burn_in,
        'device_used': str(selected_device)
    }
    
    # Modify filename to include proposal name and its specific parameters
    param_summary_for_filename = f"{proposal_name}_{proposal_param_val_str}"
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    filename = f"data/{target_name}_single_run_dim{actual_dim}_{param_summary_for_filename}_{num_iters}iters_seed{seed}.json"
    with open(filename, "w") as file:
        # Need to handle tensor to list conversion for JSON
        data_to_save = data.copy()
        if 'base_variance_vector' in data_to_save['proposal_config_params'] and isinstance(data_to_save['proposal_config_params']['base_variance_vector'], torch.Tensor):
            data_to_save['proposal_config_params']['base_variance_vector'] = data_to_save['proposal_config_params']['base_variance_vector'].tolist()
        json.dump(data_to_save, file, indent=2)
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
    
    num_dims_to_plot = min(4, actual_dim)
    plot_title_suffix = f'Proposal: {proposal_name} ({", ".join(proposal_params_str_list)})\nAcceptance: {acceptance_rate:.3f}, ESJD: {esjd:.6f}'

    if num_dims_to_plot == 1:
        plt.plot(chain_data[:, 0], alpha=0.7, linewidth=0.5, color='blue')
        plt.xlabel('Iteration (after burn-in)')
        plt.ylabel('Value')
        plt.title(f'Traceplot - {target_name} (Dim 1)\n{plot_title_suffix}')
        plt.grid(True, alpha=0.3)
    else:
        for i in range(num_dims_to_plot):
            plt.subplot(num_dims_to_plot, 1, i + 1)
            plt.plot(chain_data[:, i], alpha=0.7, linewidth=0.5, color=f'C{i}')
            plt.ylabel(f'Dimension {i + 1}')
            plt.grid(True, alpha=0.3)
            if i == 0:
                plt.title(f'Traceplot - {target_name} (First {num_dims_to_plot} Dims)\n{plot_title_suffix}')
            if i == num_dims_to_plot - 1:
                plt.xlabel('Iteration (after burn-in)')
    
    plt.tight_layout()
    traceplot_filename = f"images/traceplot_{target_name}_single_run_dim{actual_dim}_{param_summary_for_filename}_{num_iters}iters_seed{seed}.png"
    plt.savefig(traceplot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Traceplot saved as '{traceplot_filename}'")
    
    if actual_dim >= 2:
        print(f"   Creating 2D density visualization...")
        plt.figure(figsize=(10, 8))
        x_chain = chain_data[:, 0]
        y_chain = chain_data[:, 1]
        x_min, x_max = np.min(x_chain), np.max(x_chain)
        y_min, y_max = np.min(y_chain), np.max(y_chain)
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.05
        x_plot_min, x_plot_max = x_min - padding * x_range, x_max + padding * x_range
        y_plot_min, y_plot_max = y_min - padding * y_range, y_max + padding * y_range
        
        x_grid = np.linspace(x_plot_min, x_plot_max, 80)
        y_grid = np.linspace(y_plot_min, y_plot_max, 80)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)

        for i_idx in range(X.shape[0]):
            for j_idx in range(X.shape[1]):
                point = torch.zeros(actual_dim, device=selected_device, dtype=torch.float32)
                point[0] = torch.tensor(X[i_idx, j_idx], device=selected_device, dtype=torch.float32)
                point[1] = torch.tensor(Y[i_idx, j_idx], device=selected_device, dtype=torch.float32)
                if actual_dim > 2:
                    # Use mean of chain for other dimensions if available, else zeros
                    if chain_data.shape[0] > 0 and chain_data.shape[1] > 2:
                         mean_other_dims = torch.tensor(np.mean(chain_data[:, 2:], axis=0), device=selected_device, dtype=torch.float32)
                         point[2:] = mean_other_dims
                    # else point[2:] remains zeros, which is fine if chain_data is empty or too small
                
                try: # Pass point tensor directly
                    if use_torch_target:
                        if hasattr(target_distribution, 'log_density'):
                             Z[i_idx, j_idx] = torch.exp(target_distribution.log_density(point)).item()
                        elif hasattr(target_distribution, 'density'):
                             Z[i_idx, j_idx] = target_distribution.density(point).item()
                        else: Z[i_idx, j_idx] = 1.0 # Fallback
                    else: # CPU numpy target
                        point_np = point.cpu().numpy()
                        if hasattr(target_distribution, 'log_density'):
                            Z[i_idx, j_idx] = np.exp(target_distribution.log_density(point_np))
                        elif hasattr(target_distribution, 'density'):
                            Z[i_idx, j_idx] = target_distribution.density(point_np)
                        else: Z[i_idx, j_idx] = 1.0 # Fallback
                except Exception as e:
                    Z[i_idx, j_idx] = 1e-10
        
        contour = plt.contourf(X, Y, Z, levels=20, cmap='Greys', alpha=0.7)
        plt.colorbar(contour, label='Target Density')
        plt.contour(X, Y, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        sample_indices = np.arange(0, len(x_chain), max(1, len(x_chain)//1000))
        plt.scatter(x_chain[sample_indices], y_chain[sample_indices], c='red', s=2, alpha=0.6, zorder=5, label='MCMC Samples')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'2D Target Density & Samples - {target_name}\n{plot_title_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        density_filename = f"images/density2D_{target_name}_single_run_dim{actual_dim}_{param_summary_for_filename}_{num_iters}iters_seed{seed}.png"
        plt.savefig(density_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   2D density visualization saved as '{density_filename}'")

    if actual_dim <= 6:
        print(f"   Creating marginal distribution histograms...")
        fig, axes = plt.subplots(2, (actual_dim + 1) // 2 if actual_dim > 1 else 1, figsize=(15, 10 if actual_dim > 3 else 5))
        axes = np.array(axes).flatten()
        for i in range(actual_dim):
            axes[i].hist(chain_data[:, i], bins=50, alpha=0.7, density=True, color=f'C{i}')
            axes[i].set_xlabel(f'Dimension {i + 1}')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'Marginal - Dim {i + 1}')
            axes[i].grid(True, alpha=0.3)
        for i in range(actual_dim, len(axes)):
            axes[i].set_visible(False)
        plt.suptitle(f'Marginal Distributions - {target_name}\n{plot_title_suffix}')
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        hist_filename = f"images/histograms_{target_name}_single_run_dim{actual_dim}_{param_summary_for_filename}_{num_iters}iters_seed{seed}.png"
        plt.savefig(hist_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Marginal histograms saved as '{hist_filename}'")
    
    print(f"\nAll plots generated!")
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single GPU-accelerated RWM simulation with flexible proposals and immediate plotting")
    parser.add_argument("--dim", type=int, default=10, help="Dimension of the target distribution")
    parser.add_argument("--target", type=str, default="MultivariateNormal", help="Target distribution name")
    parser.add_argument("--num_iters", type=int, default=50000, help="Number of MCMC iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--burn_in", type=int, default=1000, help="Burn-in period")
    
    # Proposal related arguments
    parser.add_argument("--proposal_name", type=str, default="Normal", 
                        choices=["Normal", "Laplace", "UniformRadius"], help="Type of proposal distribution")
    parser.add_argument("--normal_base_variance", type=float, default=None, 
                        help="Base variance scalar for Normal proposal (default: 2.38^2/dim)")
    parser.add_argument("--laplace_base_variance_iso", type=float, default=None, 
                        help="Isotropic base variance for Laplace proposal (scalar, applied to all dims).")
    parser.add_argument("--laplace_base_variance_aniso", type=str, default=None, 
                        help="JSON string for anisotropic Laplace base variance vector, e.g., '[0.1, 0.2]'")
    parser.add_argument("--uniform_base_radius", type=float, default=None, 
                        help="Base radius for UniformRadius proposal (default: 1.0).")

    # Device and target type arguments
    parser.add_argument("--use_cpu_target", action="store_true", help="Use CPU (numpy-based) target distribution instead of PyTorch-based one.")
    parser.add_argument("--device", type=str, default=None, help="Specify device ('cuda' or 'cpu'). Auto-detects if None.")

    # Distribution-specific parameters (as before)
    parser.add_argument("--hybrid_rosenbrock_n1", type=int, default=3)
    parser.add_argument("--hybrid_rosenbrock_n2", type=int, default=5)
    parser.add_argument("--neal_funnel_mu_v", type=float, default=0.0)
    parser.add_argument("--neal_funnel_sigma_v_sq", type=float, default=9.0)
    parser.add_argument("--neal_funnel_mu_z", type=float, default=0.0)
    parser.add_argument("--super_funnel_J", type=int, default=5)
    parser.add_argument("--super_funnel_K", type=int, default=3)
    parser.add_argument("--super_funnel_n_per_group", type=int, default=20)
    parser.add_argument("--super_funnel_prior_hypermean_std", type=float, default=10.0)
    parser.add_argument("--super_funnel_prior_tau_scale", type=float, default=2.5)
    
    args = parser.parse_args()

    # Determine target distribution type
    use_torch_target = not args.use_cpu_target
    
    if args.device is None:
        used_device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        used_device_str = args.device

    if used_device_str == 'cuda' and torch.cuda.is_available():
        print(f"🚀 GPU ({torch.cuda.get_device_name()}) will be used for simulation and target (if PyTorch-based).")
    elif used_device_str == 'cpu':
        print(f"⚙️ CPU will be used for simulation and target (if PyTorch-based).")
    else: # GPU requested but not available
        print(f"⚠️ GPU ('{args.device}') was requested but is not available. Falling back to CPU for simulation.")
        print(f"   Target distribution will also use CPU if PyTorch-based.")
        used_device_str = 'cpu'

    kwargs_for_target = {}
    if args.target == "HybridRosenbrock":
        kwargs_for_target['n1'] = args.hybrid_rosenbrock_n1
        kwargs_for_target['n2'] = args.hybrid_rosenbrock_n2
    elif args.target == "NealFunnel":
        kwargs_for_target['mu_v'] = args.neal_funnel_mu_v
        kwargs_for_target['sigma_v_sq'] = args.neal_funnel_sigma_v_sq
        kwargs_for_target['mu_z'] = args.neal_funnel_mu_z
    elif args.target == "SuperFunnel":
        kwargs_for_target['J'] = args.super_funnel_J
        kwargs_for_target['K'] = args.super_funnel_K
        kwargs_for_target['n_per_group'] = args.super_funnel_n_per_group
        kwargs_for_target['prior_hypermean_std'] = args.super_funnel_prior_hypermean_std
        kwargs_for_target['prior_tau_scale'] = args.super_funnel_prior_tau_scale
    
    results = run_single_simulation(
        dim=args.dim, 
        target_name=args.target, 
        num_iters=args.num_iters, 
        seed=args.seed, 
        burn_in=args.burn_in,
        proposal_name=args.proposal_name,
        normal_base_variance=args.normal_base_variance,
        laplace_base_variance_iso=args.laplace_base_variance_iso,
        laplace_base_variance_aniso=args.laplace_base_variance_aniso,
        uniform_base_radius=args.uniform_base_radius,
        use_torch_target=use_torch_target,
        device_str=used_device_str,
        **kwargs_for_target
    )

    print(f"\nExperiment completed successfully!") 