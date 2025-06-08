#!/usr/bin/env python3
"""
Batch Average MCMC Results Across Seeds

This script automatically finds all experimental configurations in the data directory
that have multiple seed files and creates averaged versions for smoother plotting.

It groups files by their base pattern (target distribution, algorithm, dimension, iterations)
and creates averaged files for each group that has multiple seeds.

Usage:
    python batch_average_seeds.py
    python batch_average_seeds.py --data_dir ../data --min_seeds 2
"""

import argparse
import json
import os
import re
import numpy as np
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
from collections import defaultdict

# Import functions from average_seeds.py
from average_seeds import (
    load_json, save_json, average_experiment_data, 
    generate_output_filename, construct_pattern
)


def extract_pattern_and_seed(filename: str) -> Tuple[str, int]:
    """
    Extract the base pattern and seed from a filename.
    
    Args:
        filename: JSON filename
        
    Returns:
        Tuple of (base_pattern, seed) or (base_pattern, None) if no seed
    """
    # Remove .json extension
    base = filename.replace('.json', '')
    
    # Check for seed pattern
    seed_match = re.search(r'_seed(\d+)$', base)
    if seed_match:
        seed = int(seed_match.group(1))
        pattern = base[:seed_match.start()]
        return pattern, seed
    else:
        # No seed in filename
        return base, None


def group_files_by_pattern(data_dir: str) -> Dict[str, List[str]]:
    """
    Group JSON files by their base experimental pattern.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Dictionary mapping base patterns to lists of file paths
    """
    pattern_groups = defaultdict(list)
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and not filename.endswith('_averaged.json'):
            # Skip already averaged files and non-experimental files
            if any(skip in filename for skip in ['summary', 'analysis', 'combined']):
                continue
                
            pattern, seed = extract_pattern_and_seed(filename)
            file_path = os.path.join(data_dir, filename)
            pattern_groups[pattern].append(file_path)
    
    return dict(pattern_groups)


def filter_multi_seed_groups(pattern_groups: Dict[str, List[str]], min_seeds: int = 2) -> Dict[str, List[str]]:
    """
    Filter groups to only include those with multiple seed files.
    
    Args:
        pattern_groups: Dictionary mapping patterns to file lists
        min_seeds: Minimum number of seeds required for averaging
        
    Returns:
        Filtered dictionary with only multi-seed groups
    """
    multi_seed_groups = {}
    
    for pattern, files in pattern_groups.items():
        # Count files with seeds
        seed_files = []
        for file_path in files:
            filename = os.path.basename(file_path)
            _, seed = extract_pattern_and_seed(filename)
            if seed is not None:
                seed_files.append(file_path)
        
        if len(seed_files) >= min_seeds:
            multi_seed_groups[pattern] = seed_files
    
    return multi_seed_groups


def process_pattern_group(pattern: str, file_paths: List[str], output_dir: str, verbose: bool = False) -> bool:
    """
    Process a single pattern group by averaging the files.
    
    Args:
        pattern: Base experimental pattern
        file_paths: List of file paths to average
        output_dir: Output directory for averaged file
        verbose: Whether to print verbose output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if verbose:
            print(f"\nProcessing pattern: {pattern}")
            print(f"  Files to average: {len(file_paths)}")
            for file_path in file_paths:
                print(f"    {os.path.basename(file_path)}")
        
        # Average the data
        averaged_data = average_experiment_data(file_paths)
        
        # Generate output filename
        seeds = averaged_data.get('averaged_from_seeds', [])
        output_filename = generate_output_filename(pattern, seeds)
        output_path = os.path.join(output_dir, output_filename)
        
        # Save averaged data
        save_json(averaged_data, output_path)
        
        if verbose:
            print(f"  Successfully created: {output_filename}")
            print(f"  Averaged seeds: {seeds}")
            print(f"  Max ESJD: {averaged_data.get('max_esjd', 'N/A'):.6f}")
            print(f"  Max swap acceptance rate: {averaged_data.get('max_swap_acceptance_rate', 'N/A'):.6f}")
        else:
            print(f"Created: {output_filename}")
        
        return True
        
    except Exception as e:
        print(f"Error processing pattern {pattern}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch average MCMC experimental results across random seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script automatically finds all experimental configurations that have multiple
seed files and creates averaged versions for smoother plotting curves.

Examples:
  # Process all multi-seed groups in current directory
  python batch_average_seeds.py
  
  # Process with custom data directory and minimum seed requirement
  python batch_average_seeds.py --data_dir ../data --min_seeds 3
  
  # Dry run to see what would be processed
  python batch_average_seeds.py --dry_run --verbose
        """
    )
    
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as data_dir)')
    parser.add_argument('--min_seeds', type=int, default=2,
                       help='Minimum number of seeds required for averaging (default: 2)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be processed without creating files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.data_dir
    
    if args.verbose or args.dry_run:
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Minimum seeds required: {args.min_seeds}")
        if args.dry_run:
            print("DRY RUN - No files will be created")
    
    # Group files by pattern
    try:
        pattern_groups = group_files_by_pattern(args.data_dir)
        if args.verbose:
            print(f"\nFound {len(pattern_groups)} unique experimental patterns")
    except Exception as e:
        print(f"Error reading data directory: {e}")
        return
    
    # Filter to multi-seed groups
    multi_seed_groups = filter_multi_seed_groups(pattern_groups, args.min_seeds)
    
    if not multi_seed_groups:
        print(f"No experimental patterns found with {args.min_seeds}+ seed files")
        if args.verbose:
            print("\nAll patterns found:")
            for pattern, files in pattern_groups.items():
                seed_count = sum(1 for f in files if extract_pattern_and_seed(os.path.basename(f))[1] is not None)
                print(f"  {pattern}: {seed_count} seed files")
        return
    
    print(f"\nFound {len(multi_seed_groups)} patterns with {args.min_seeds}+ seeds:")
    
    # Process each group
    success_count = 0
    total_count = len(multi_seed_groups)
    
    for pattern, file_paths in multi_seed_groups.items():
        if args.dry_run:
            seeds = []
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                _, seed = extract_pattern_and_seed(filename)
                if seed is not None:
                    seeds.append(seed)
            
            output_filename = generate_output_filename(pattern, sorted(seeds))
            print(f"Would create: {output_filename}")
            if args.verbose:
                print(f"  From {len(file_paths)} files with seeds: {sorted(seeds)}")
        else:
            if process_pattern_group(pattern, file_paths, output_dir, args.verbose):
                success_count += 1
    
    if not args.dry_run:
        print(f"\nCompleted: {success_count}/{total_count} patterns processed successfully")
        if success_count < total_count:
            print(f"Failed: {total_count - success_count} patterns had errors")


if __name__ == "__main__":
    main() 