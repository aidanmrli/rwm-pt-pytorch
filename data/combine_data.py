import json
import numpy as np
import os

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to average values at each index from two lists
def average_lists(list1, list2):
    array1 = np.array(list1)
    array2 = np.array(list2)
    return ((array1 + array2) / 2).tolist()

# Function to combine two JSON data files
def combine_json(file1, file2, output_file):
    data1 = load_json(file1)
    data2 = load_json(file2)
    
    combined_data = {
        "max_esjd": 0.5 * (data1["max_esjd"] + data2["max_esjd"]),
        "max_acceptance_rate": 0.5 * (data1["max_acceptance_rate"] + data2["max_acceptance_rate"]),
        "max_variance_value": 0.5 * (data1["max_variance_value"] + data2["max_variance_value"]),
        "expected_squared_jump_distances": average_lists(data1["expected_squared_jump_distances"], data2["expected_squared_jump_distances"]),
        "acceptance_rates": average_lists(data1["acceptance_rates"], data2["acceptance_rates"]),
        "var_value_range": data1["var_value_range"]  # Assuming var_value_range is the same in both files
    }
    
    with open(output_file, 'w') as file:
        json.dump(combined_data, file)

# Example usage
file1 = 'Hypercube_RWM_dim5_seed0_100000iters.json'
file2 = 'Hypercube_RWM_dim5_seed1_100000iters.json'
output_file = 'Hypercube_RWM_dim5_seed2_100000iters.json'

combine_json(os.path.join(os.curdir, file1), os.path.join(os.curdir, file2), os.path.join(os.curdir, output_file))
