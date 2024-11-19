import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


def get_answers_predictions(file_path):
    answers = []
    llm_predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            if 'Answer:' == line[:len('Answer:')]:
                answer = line.replace('Answer:', '').strip()[1:-1].lower()
                answers.append(answer)
            if 'LLM:' == line[:len('LLM:')]:
                llm_prediction = line.replace('LLM', '').strip().lower()
                try:
                    llm_prediction = llm_prediction.replace("\"item title\" : ", '')
                    start = llm_prediction.find('"')
                    end = llm_prediction.rfind('"')

                    if (start + end < start) or (start + end < end):
                        print(1/0)
                        
                    llm_prediction = llm_prediction[start + 1:end]
                except Exception as e:
                    print()
                    
                llm_predictions.append(llm_prediction)
                
    return answers, llm_predictions


def evaluate(answers, llm_predictions, k=1, item_embeddings=None, diversity_weight=0.5):
    """
    Evaluates predictions with NDCG and Hit Rate, adding diversity consideration through MMR.
    
    :param answers: List of ground truth answers.
    :param llm_predictions: List of LLM predicted titles.
    :param k: Top-k rank to evaluate.
    :param item_embeddings: List of item embeddings for diversity calculation (optional).
    :param diversity_weight: Weight for diversity consideration (0 for no diversity, 1 for full diversity focus).
    :return: Normalized Discounted Cumulative Gain (NDCG) and Hit rate (HT).
    """
    NDCG = 0.0
    HT = 0.0
    total_diversity = 0.0  # To accumulate diversity scores
    predict_num = len(answers)
    
    for answer, prediction in zip(answers, llm_predictions):
        if k > 1:
            rank = prediction.index(answer)
            if rank < k:
                NDCG += 1 / np.log2(rank + 1)
                HT += 1
        elif k == 1:
            if answer in prediction:
                NDCG += 1
                HT += 1
        
        # If item_embeddings are provided, calculate the diversity score (MMR)
        if item_embeddings is not None:
            diversity_score = compute_diversity(item_embeddings, prediction, diversity_weight)
            total_diversity += diversity_score

    # If diversity was calculated, normalize by number of predictions
    if item_embeddings is not None:
        average_diversity = total_diversity / predict_num
        print(f"Average Diversity (MMR score): {average_diversity}")
    
    return NDCG / predict_num, HT / predict_num


def compute_diversity(item_embeddings, predicted_items, diversity_weight=0.5):
    """
    Computes the diversity score using MMR.
    
    :param item_embeddings: List of item embeddings.
    :param predicted_items: List of predicted item indices.
    :param diversity_weight: Weight for diversity score calculation.
    :return: Calculated diversity score.
    """
    if len(predicted_items) < 2:
        return 0  # No diversity for a single item

    # Get embeddings for the predicted items
    selected_embeddings = [item_embeddings[item_id] for item_id in predicted_items]
    
    # Compute cosine similarity between all pairs of selected items
    similarity_matrix = cosine_similarity(selected_embeddings)
    
    # Sum the negative of similarities between each pair of items to compute diversity
    diversity_score = 0
    num_pairs = 0
    for i in range(len(predicted_items)):
        for j in range(i + 1, len(predicted_items)):
            diversity_score += 1 - similarity_matrix[i][j]  # We want lower similarity to increase diversity
            num_pairs += 1

    # Normalize the diversity score by the number of pairs
    return (diversity_score / num_pairs) * diversity_weight


if __name__ == "__main__":
    inferenced_file_path = './recommendation_output.txt'
    answers, llm_predictions = get_answers_predictions(inferenced_file_path)
    print(len(answers), len(llm_predictions))
    assert(len(answers) == len(llm_predictions))
    
    # Assuming you have item embeddings available for diversity calculation
    item_embeddings = np.load('item_embeddings.npy')  # Example: Precomputed item embeddings
    
    # Evaluate at k=1 with both relevance (NDCG and HT) and diversity (MMR)
    ndcg, ht = evaluate(answers, llm_predictions, k=1, item_embeddings=item_embeddings, diversity_weight=0.5)
    
    print(f"ndcg at 1: {ndcg}")
    print(f"hit at 1: {ht}")
