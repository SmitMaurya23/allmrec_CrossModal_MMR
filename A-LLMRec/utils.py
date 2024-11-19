import os
import numpy as np
from datetime import datetime
from pytz import timezone
from scipy.linalg import det  # Added for DPP

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ex. target_word: .csv / in target_path find 123.csv file
def find_filepath(target_path, target_word):
    file_paths = []
    for file in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, file)):
            if target_word in file:
                file_paths.append(target_path + file)
            
    return file_paths

# Added DPP utility functions
def compute_dpp_likelihood(kernel_matrix, item_set):
    sub_matrix = kernel_matrix[np.ix_(item_set, item_set)]
    return det(sub_matrix)

def load_kernel_matrix(file_path):
    return np.load(file_path)

def save_kernel_matrix(kernel_matrix, file_path):
    np.save(file_path, kernel_matrix)
