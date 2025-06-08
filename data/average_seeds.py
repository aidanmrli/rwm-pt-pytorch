#!/usr/bin/env python3
"""
Average MCMC Results Across Seeds

This script finds all JSON data files matching a specific experimental configuration
(target distribution, algorithm, dimension, iterations) and averages the Expected 
Squared Jump Distance (ESJD) and acceptance rate values across different random seeds.

This creates smoother curves for plotting by reducing statistical noise from 
individual random seeds.

Usage:
    python average_seeds.py --target MultivariateNormal --algorithm RWM_GPU --dim 20 --iters 1000
    python average_seeds.py --pattern "MultivariateNormal_RWM_GPU_dim20_1000iters"
"""

import argparse
import json
import os
import re
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to JSON file with proper formatting."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)


def find_matching_files(data_dir: str, pattern: str) -> List[str]:
    """
    Find all JSON files matching the experimental pattern.
    
    Args:
        data_dir: Directory containing data files
        pattern: Base pattern without seed specification
        
    Returns:
        List of matching file paths
    """
    matching_files = []
    
    # Create regex pattern to match files with seeds
    # Pattern: {base_pattern}_seed{number}.json or {base_pattern}.json (no seed)
    seed_pattern = f"{pattern}_seed\\d+\\.json"
    no_seed_pattern = f"{pattern}\\.json"
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            if re.match(seed_pattern, filename) or re.match(no_seed_pattern, filename):
                matching_files.append(os.path.join(data_dir, filename))
    
    return sorted(matching_files)


def construct_pattern(target: str, algorithm: str, dim: int, iters: int) -> str:
    """
    Construct the base pattern from experimental parameters.
    
    Args:
        target: Target distribution name
        algorithm: Algorithm name  
        dim: Dimension
        iters: Number of iterations
        
    Returns:
        Base pattern string
    """
    return f"{target}_{algorithm}_dim{dim}_{iters}iters"


def average_numeric_values(values: List[float]) -> float:
    """Average a list of numeric values."""
    return float(np.mean(values))


def average_arrays(arrays: List[List[float]]) -> List[float]:
    """Average corresponding elements across multiple arrays."""
    if not arrays:
        return []
    
    # Convert to numpy arrays for easier manipulation
    np_arrays = [np.array(arr) for arr in arrays]
    
    # Check that all arrays have the same length
    lengths = [len(arr) for arr in np_arrays]
    if len(set(lengths)) > 1:
        raise ValueError(f"Arrays have different lengths: {lengths}")
    
    # Average element-wise
    averaged = np.mean(np_arrays, axis=0)
    return averaged.tolist()


def average_experiment_data(file_paths: List[str]) -> Dict[str, Any]:
    """
    Average experimental data across multiple seed files.
    
    Args:
        file_paths: List of JSON file paths to average
        
    Returns:
        Dictionary containing averaged data
    """
    if not file_paths:
        raise ValueError("No files provided for averaging")
    
    # Load all data files
    all_data = [load_json(path) for path in file_paths]
    
    # Extract seeds for documentation
    seeds = []
    for path in file_paths:
        filename = os.path.basename(path)
        seed_match = re.search(r'seed(\d+)', filename)
        if seed_match:
            seeds.append(int(seed_match.group(1)))
        else:
            seeds.append(None)  # File without explicit seed
    
    # Check for inconsistent array lengths across all relevant fields
    array_fields_to_check = [
        'var_value_range', 
        'expected_squared_jump_distances', 
        'acceptance_rates', 
        'times', 
        'swap_acceptance_rates_range'
    ]

    for field in array_fields_to_check:
        lengths = {}
        # Collect lengths from all files that have this field
        for i, data in enumerate(all_data):
            if field in data and isinstance(data.get(field), list):
                lengths[file_paths[i]] = len(data[field])
        
        # If the field was present and lengths are inconsistent, raise a detailed error
        if lengths and len(set(lengths.values())) > 1:
            error_msg = f"Inconsistent array lengths for field '{field}':\n"
            for path, length in sorted(lengths.items()):
                error_msg += f"  - {os.path.basename(path)}: length {length}\n"
            raise ValueError(error_msg)
    
    # Get reference data structure from first file
    reference_data = all_data[0]
    
    # Average scalar values
    scalar_fields = ['max_esjd', 'max_acceptance_rate', 'max_variance_value']
    averaged_data = {}
    
    for field in scalar_fields:
        if field in reference_data:
            values = [data[field] for data in all_data if field in data]
            averaged_data[field] = average_numeric_values(values)
    
    # Average array values
    array_fields = ['expected_squared_jump_distances', 'acceptance_rates', 'swap_acceptance_rates_range']
    for field in array_fields:
        if field in reference_data:
            arrays = [data[field] for data in all_data if field in data]
            averaged_data[field] = average_arrays(arrays)
    
    # Keep reference values (should be identical across files)
    reference_fields = ['var_value_range', 'target_distribution', 'dimension', 'num_iterations']
    for field in reference_fields:
        if field in reference_data:
            averaged_data[field] = reference_data[field]
    
    # Add metadata about averaging
    averaged_data['averaged_from_seeds'] = [s for s in seeds if s is not None]
    averaged_data['num_files_averaged'] = len(file_paths)
    averaged_data['source_files'] = [os.path.basename(path) for path in file_paths]
    
    return averaged_data


def generate_output_filename(pattern: str, seeds: List[int]) -> str:
    """
    Generate output filename for averaged data.
    
    Args:
        pattern: Base experimental pattern
        seeds: List of seeds that were averaged
        
    Returns:
        Output filename
    """
    if seeds:
        seed_str = f"seeds{'-'.join(map(str, sorted(seeds)))}"
    else:
        seed_str = "averaged"
    
    return f"{pattern}_{seed_str}_averaged.json"


def main():
    parser = argparse.ArgumentParser(
        description="Average MCMC experimental results across random seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using individual parameters
  python average_seeds.py --target MultivariateNormal --algorithm RWM_GPU --dim 20 --iters 1000
  
  # Using pattern directly
  python average_seeds.py --pattern "MultivariateNormal_RWM_GPU_dim20_1000iters"
  
  # Specify custom data directory
  python average_seeds.py --pattern "MultivariateNormal_RWM_GPU_dim20_1000iters" --data_dir ../data
        """
    )
    
    # Pattern specification options
    pattern_group = parser.add_mutually_exclusive_group(required=True)
    pattern_group.add_argument('--pattern', type=str,
                              help='Base pattern to match (e.g., "MultivariateNormal_RWM_GPU_dim20_1000iters")')
    
    # Individual parameter specification
    pattern_group.add_argument('--target', type=str,
                              help='Target distribution name (use with --algorithm, --dim, --iters)')
    
    parser.add_argument('--algorithm', type=str,
                       help='Algorithm name (e.g., RWM, RWM_GPU, PTrwm)')
    parser.add_argument('--dim', type=int,
                       help='Problem dimension')
    parser.add_argument('--iters', type=int,
                       help='Number of iterations')
    
    # Optional arguments
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Directory containing data files (default: current directory)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as data_dir)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.target and not all([args.algorithm, args.dim, args.iters]):
        parser.error("When using --target, must also specify --algorithm, --dim, and --iters")
    
    # Construct pattern
    if args.pattern:
        pattern = args.pattern
    else:
        pattern = construct_pattern(args.target, args.algorithm, args.dim, args.iters)
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.data_dir
    
    if args.verbose:
        print(f"Searching for pattern: {pattern}")
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {output_dir}")
    
    # Find matching files
    matching_files = find_matching_files(args.data_dir, pattern)
    
    if not matching_files:
        print(f"No files found matching pattern: {pattern}")
        print(f"Searched in directory: {args.data_dir}")
        return
    
    if args.verbose:
        print(f"Found {len(matching_files)} matching files:")
        for file_path in matching_files:
            print(f"  {os.path.basename(file_path)}")
    
    # Average the data
    try:
        averaged_data = average_experiment_data(matching_files)
        
        # Generate output filename
        seeds = averaged_data.get('averaged_from_seeds', [])
        output_filename = generate_output_filename(pattern, seeds)
        output_path = os.path.join(output_dir, output_filename)
        
        # Save averaged data
        save_json(averaged_data, output_path)
        
        print(f"Successfully averaged {len(matching_files)} files")
        print(f"Output saved to: {output_filename}")
        
        if args.verbose:
            print(f"Averaged seeds: {seeds}")
            print(f"Max ESJD: {averaged_data.get('max_esjd', 'N/A'):.6f}")
            print(f"Max acceptance rate: {averaged_data.get('max_acceptance_rate', 'N/A'):.6f}")
            print(f"Max variance value: {averaged_data.get('max_variance_value', 'N/A'):.6f}")
        
    except Exception as e:
        print(f"Error averaging data: {e}")
        return


if __name__ == "__main__":
    main() 