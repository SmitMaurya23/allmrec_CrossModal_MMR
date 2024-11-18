import numpy as np

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
                    llm_prediction = llm_prediction[start+1:end]
                except Exception:
                    pass
                llm_predictions.append(llm_prediction)
    return answers, llm_predictions

def mmr(relevance_scores, similarity_matrix, lambda_val=0.5, top_k=10):
    """
    Implements Maximal Marginal Relevance (MMR).
    
    Args:
        relevance_scores (list): Relevance scores for items.
        similarity_matrix (2D array): Pairwise similarity between items.
        lambda_val (float): Balance between relevance and diversity.
        top_k (int): Number of items to recommend.
    
    Returns:
        list: Indices of selected items.
    """
    selected = []
    candidates = list(range(len(relevance_scores)))

    for _ in range(top_k):
        mmr_scores = []
        for i in candidates:
            sim_to_selected = [similarity_matrix[i][j] for j in selected] if selected else [0]
            max_similarity = max(sim_to_selected)
            mmr_score = lambda_val * relevance_scores[i] - (1 - lambda_val) * max_similarity
            mmr_scores.append((i, mmr_score))
        
        # Select item with highest MMR score
        best_candidate = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_candidate)
        candidates.remove(best_candidate)
    
    return selected

def intra_list_diversity(similarity_matrix, predictions):
    """
    Compute the intra-list diversity (ILD) of recommendations.
    Args:
        similarity_matrix (np.ndarray): Pairwise similarity matrix.
        predictions (list of lists): List of recommended items for each user.

    Returns:
        float: Average ILD across all users.
    """
    diversities = []
    for pred in predictions:
        if len(pred) > 1:
            diversity = 1 - np.mean([
                similarity_matrix[i][j]
                for i in pred for j in pred if i != j
            ])
            diversities.append(diversity)
    return np.mean(diversities) if diversities else 0.0

def evaluate(answers, llm_predictions, relevance_scores, similarity_matrix, k=5, lambda_val=0.5):
    NDCG = 0.0
    HT = 0.0
    predict_num = len(answers)
    diversified_predictions = []

    for answer, prediction in zip(answers, llm_predictions):
        # Apply MMR to diversify recommendations
        mmr_indices = mmr(relevance_scores, similarity_matrix, lambda_val=lambda_val, top_k=k)
        diversified_predictions.append([prediction[i] for i in mmr_indices])
        
        if answer in diversified_predictions[-1]:
            rank = diversified_predictions[-1].index(answer) + 1
            NDCG += 1 / np.log2(rank + 1)
            HT += 1

    # Compute intra-list diversity
    ild = intra_list_diversity(similarity_matrix, diversified_predictions)
                
    return NDCG / predict_num, HT / predict_num, ild, diversified_predictions

if __name__ == "__main__":
    inferenced_file_path = './recommendation_output.txt'
    answers, llm_predictions = get_answers_predictions(inferenced_file_path)
    print(len(answers), len(llm_predictions))
    assert len(answers) == len(llm_predictions)
    
    # Placeholder for relevance scores and similarity matrix
    # Replace with actual data or computation
    relevance_scores = np.random.rand(len(llm_predictions))
    similarity_matrix = np.random.rand(len(llm_predictions), len(llm_predictions))
    
    # Evaluate with MMR
    ndcg, ht, ild, diversified_predictions = evaluate(
        answers, llm_predictions, relevance_scores, similarity_matrix, k=5, lambda_val=0.5
    )
    
    print(f"NDCG@5: {ndcg}")
    print(f"Hit Rate@5: {ht}")
    print(f"Intra-List Diversity: {ild}")

    # Save diversified predictions
    with open('./output/diversified_recommendations.txt', 'w') as f:
        for preds in diversified_predictions:
            f.write(', '.join(preds) + '\n')
    print("Diversified recommendations saved to './output/diversified_recommendations.txt'.")
