import os
from datetime import datetime
from pytz import timezone
import numpy as np
from numpy.linalg import norm

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

# Maximum Marginal Relevance (MMR)
def mmr(candidate_items, query, relevance_scores, diversity_weight=0.5):
    """
    Perform Maximum Marginal Relevance (MMR) to diversify recommendations.

    Args:
        candidate_items (list): List of candidate items for recommendation.
        query (np.array): Query embedding.
        relevance_scores (list): Pre-computed relevance scores for candidate items.
        diversity_weight (float): Weight for diversity (0 = no diversification, 1 = full diversity).

    Returns:
        list: Re-ranked list of recommendations.
    """
    selected = []
    while len(selected) < len(candidate_items):
        mmr_scores = [
            (diversity_weight * relevance_scores[i] -
             (1 - diversity_weight) * max(
                [similarity(candidate_items[i], selected_item) for selected_item in selected], default=0))
            for i in range(len(candidate_items))
        ]
        selected_item_index = np.argmax(mmr_scores)
        selected.append(candidate_items[selected_item_index])
        candidate_items.pop(selected_item_index)
        relevance_scores.pop(selected_item_index)
    return selected

def similarity(item1, item2):
    """
    Compute cosine similarity between two vectors.

    Args:
        item1 (np.array): First item's embedding.
        item2 (np.array): Second item's embedding.

    Returns:
        float: Cosine similarity score between item1 and item2.
    """
    item1 = np.array(item1)
    item2 = np.array(item2)
    if norm(item1) == 0 or norm(item2) == 0:
        return 0  # Avoid division by zero
    return np.dot(item1, item2) / (norm(item1) * norm(item2))
