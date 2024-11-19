import os
from datetime import datetime
from pytz import timezone
import numpy as np

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

def calculate_mmr(items, relevance_scores, lambda_val=0.5):
    """
    Calculate Maximal Marginal Relevance (MMR) for item selection.
    
    Args:
    - items (list): List of items.
    - relevance_scores (dict): Dictionary of item to relevance score.
    - lambda_val (float): Lambda parameter for MMR.
    
    Returns:
    - selected_items (list): List of selected items based on MMR.
    """
    selected_items = []
    while len(selected_items) < len(items):
        marginal_relevance = []
        for item in items:
            if item not in selected_items:
                redundancy = np.max([np.dot(item, selected_item) for selected_item in selected_items]) if selected_items else 0
                mr_score = lambda_val * relevance_scores[item] - (1 - lambda_val) * redundancy
                marginal_relevance.append((item, mr_score))
        if marginal_relevance:
            selected_items.append(max(marginal_relevance, key=lambda x: x[1])[0])
    return selected_items
