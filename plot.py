import argparse
import os
from algorithms import *
import numpy as np
from target_distributions import *
import matplotlib.pyplot as plt
import json

# Function to read JSON data
def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

def extract_dimension(filename):
    # Split the filename by underscores
    parts = filename.split('_')
    
    # Find the part that starts with 'dim'
    for part in parts:
        if part.startswith('dim'):
            # Extract the number after 'dim'
            dim = int(part[3:])
            return dim
        
# Function to create the plot
def create_plot(data, filename):
    dim = extract_dimension(filename)
    # plt.figure(figsize=(10, 6))
    
    # Plot ESJD vs acceptance rate
    plt.plot(data['acceptance_rates'], data['expected_squared_jump_distances'], 
            marker='x')
    
    plt.axvline(x=0.234, color='red', linestyle=':', label='a = 0.234')
    plt.xlabel('acceptance rate')
    plt.ylabel('ESJD')
    plt.title(f'ESJD vs acceptance rate (dim={dim})')
    plt.legend()
    output_filename = f"images/publishing/ESJD_vs_acceptance_rate_{os.path.splitext(filename)[0]}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()
    print(f"Plot created and saved as '{output_filename}'")

# Function to process all JSON files in a directory
def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            try:
                data = read_json(file_path)
                create_plot(data, filename)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Main execution
if __name__ == "__main__":
    # Replace 'data_directory' with the path to your directory containing JSON files
    data_directory = 'data/'
    process_directory(data_directory)