import numpy as np
import json


def get_answers_predictions(file_path):
    """
    Parses the recommendation output file to extract answers and predictions.

    Args:
        file_path (str): Path to the recommendation output file.

    Returns:
        list, list: Extracted answers and predictions.
    """
    answers = []
    llm_predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Answer:'):
                answer = line.replace('Answer:', '').strip()[1:-1].lower()
                answers.append(answer)
            elif line.startswith('LLM:'):
                llm_prediction = line.replace('LLM:', '').strip()
                try:
                    if llm_prediction.startswith('[') and llm_prediction.endswith(']'):
                        llm_prediction = json.loads(llm_prediction)
                    else:
                        llm_prediction = llm_prediction.strip('"[]').split(',')
                except Exception as e:
                    print(f"Error parsing LLM predictions: {e} | Content: {llm_prediction}")
                    llm_prediction = []
                llm_predictions.append(llm_prediction)

    # Debugging Outputs
    print("Sample Answers:", answers[:5])
    print("Sample Predictions:", llm_predictions[:5])

    return answers, llm_predictions


def mmr(relevance_scores, similarity_matrix, lambda_val=0.5, top_k=10):
    """
    Implements Maximal Marginal Relevance (MMR).
    
    Args:
        relevance_scores (list or np.ndarray): Relevance scores for items.
        similarity_matrix (np.ndarray): Pairwise similarity between items.
        lambda_val (float): Balance between relevance and diversity.
        top_k (int): Number of items to recommend.
    
    Returns:
        list: Indices of selected items.
    """
    relevance_scores = np.array(relevance_scores).flatten()
    selected = []
    candidates = list(range(len(relevance_scores)))

    for _ in range(top_k):
        mmr_scores = []
        for i in candidates:
            if i >= similarity_matrix.shape[0]:
                print(f"Skipping index {i} - out of bounds for similarity matrix.")
                continue

            # Compute similarity to already selected items
            if selected:
                sim_to_selected = [
                    similarity_matrix[i][j]
                    for j in selected
                    if j < similarity_matrix.shape[1]
                ]
                max_similarity = max(sim_to_selected) if sim_to_selected else 0
            else:
                max_similarity = 0

            mmr_score = lambda_val * relevance_scores[i] - (1 - lambda_val) * max_similarity
            mmr_scores.append((i, mmr_score))
        
        if not mmr_scores:
            print("No valid candidates remaining for MMR.")
            break

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
                for i in pred for j in pred if i != j and i < similarity_matrix.shape[0] and j < similarity_matrix.shape[1]
            ])
            diversities.append(diversity)
    return np.mean(diversities) if diversities else 0.0


def evaluate(answers, llm_predictions, relevance_scores, similarity_matrix, k=5, lambda_val=0.5):
    """
    Evaluate the performance of the recommendations.

    Args:
        answers (list): Ground truth answers.
        llm_predictions (list): Predictions made by the model.
        relevance_scores (list): Relevance scores for items.
        similarity_matrix (np.ndarray): Pairwise similarity matrix.
        k (int): Number of top recommendations to consider.
        lambda_val (float): Balance parameter for MMR.

    Returns:
        tuple: NDCG, Hit Rate, ILD, and diversified predictions.
    """
    NDCG = 0.0
    HT = 0.0
    predict_num = len(answers)
    diversified_predictions = []

    for idx, (answer, prediction) in enumerate(zip(answers, llm_predictions)):
        if not prediction:
            print(f"Skipping empty prediction for answer: {answer}")
            diversified_predictions.append([])
            continue

        # Apply MMR
        mmr_indices = mmr(relevance_scores, similarity_matrix, lambda_val=lambda_val, top_k=k)
        print(f"Answer: {answer}, Prediction: {prediction}, MMR Indices: {mmr_indices}")

        # Append diversified predictions
        diversified_predictions.append([prediction[i] for i in mmr_indices if i < len(prediction)])

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
    print(f"Loaded {len(answers)} answers and {len(llm_predictions)} predictions.")

    # Validate answers and predictions
    if len(answers) == 0 or len(llm_predictions) == 0:
        print("Error: No answers or predictions found. Exiting.")
        exit()

    # Placeholder for relevance scores and similarity matrix
    try:
        relevance_scores = np.load('./output/relevance_scores.npy', allow_pickle=True)
        similarity_matrix = np.load('./output/embeddings.npy', allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error loading relevance scores or similarity matrix: {e}")
        exit()

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
