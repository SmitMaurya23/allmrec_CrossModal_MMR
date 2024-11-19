import os
from datetime import datetime
from pytz import timezone
import torch
from scipy.linalg import det

def create_dir(directory):
    """
    Create a directory if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# Example: target_word: .csv / in target_path find 123.csv file
def find_filepath(target_path, target_word):
    """
    Find all file paths in a target directory containing a specific word.
    
    Args:
        target_path (str): Directory to search.
        target_word (str): Keyword to look for in file names.

    Returns:
        list: Paths of files containing the target word.
    """
    file_paths = []
    for file in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, file)):
            if target_word in file:
                file_paths.append(target_path + file)
            
    return file_paths

# DPP Utility Functions
def compute_kernel_matrix(embeddings):
    """
    Compute the DPP kernel matrix from item embeddings.

    Args:
        embeddings (torch.Tensor): Item embeddings of shape [N, D].

    Returns:
        torch.Tensor: Kernel matrix of shape [N, N].
    """
    # Normalize embeddings to ensure cosine similarity-like behavior
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    kernel_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)
    return kernel_matrix

def dpp_selection(kernel_matrix, k):
    """
    Perform DPP-based subset selection for diversity.

    Args:
        kernel_matrix (torch.Tensor): Kernel matrix of shape [N, N].
        k (int): Number of items to select.

    Returns:
        list[int]: Indices of the selected items.
    """
    N = kernel_matrix.shape[0]
    selected_indices = []
    remaining_indices = list(range(N))

    for _ in range(k):
        if not remaining_indices:
            break
        # Compute marginal gains for remaining indices
        scores = []
        for idx in remaining_indices:
            current_matrix = kernel_matrix[np.ix_([idx] + selected_indices, [idx] + selected_indices)]
            scores.append(det(current_matrix.cpu().numpy()))
        # Select the index with the highest marginal gain
        best_idx = remaining_indices[np.argmax(scores)]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return selected_indices
