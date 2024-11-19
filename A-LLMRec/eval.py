import numpy as np
import json


def get_answers_predictions(file_path):
    answers = []
    llm_predictions = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
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
                    except json.JSONDecodeError as e:
                        print(f"Error parsing LLM predictions: {e} | Content: {llm_prediction}")
                        llm_prediction = []
                    llm_predictions.append(llm_prediction)

        print(f"Total Answers: {len(answers)}")
        print(f"Total Predictions: {len(llm_predictions)}")
        print("Sample Answers:", answers[:5])
        print("Sample Predictions:", llm_predictions[:5])

    except FileNotFoundError:
        print(f"Error: File not found at path {file_path}")
        exit()

    return answers, llm_predictions


def mmr(relevance_scores, similarity_matrix, lambda_val=0.3, top_k=10):
    relevance_scores = np.array(relevance_scores).flatten()
    selected = []
    candidates = list(range(min(len(relevance_scores), similarity_matrix.shape[0])))

    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Number of candidates: {len(candidates)}")

    for _ in range(top_k):
        mmr_scores = []
        for i in candidates:
            if i >= similarity_matrix.shape[0]:
                print(f"Skipping index {i} - out of bounds for similarity matrix rows.")
                continue

            if selected:
                sim_to_selected = [
                    np.mean(similarity_matrix[i, :, j])
                    for j in selected
                    if j < similarity_matrix.shape[2]
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
        if best_candidate < similarity_matrix.shape[0]:
            selected.append(best_candidate)
            candidates.remove(best_candidate)
        else:
            print(f"Invalid candidate selected: {best_candidate}. Skipping.")

        print(f"Selected: {selected}")
        print(f"MMR Scores (Top 5): {mmr_scores[:5]}")

    return selected


def intra_list_diversity(similarity_matrix, predictions):
    """
    Compute the intra-list diversity (ILD) of recommendations.
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


def evaluate(answers, llm_predictions, relevance_scores, similarity_matrix, k=5, lambda_val=0.3):
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Relevance scores shape: {relevance_scores.shape}")
    print(f"Number of answers: {len(answers)}")

    # Normalize similarity matrix
    similarity_matrix = (similarity_matrix - np.min(similarity_matrix)) / (np.max(similarity_matrix) - np.min(similarity_matrix))
    print(f"Normalized Similarity Matrix. Min: {np.min(similarity_matrix)}, Max: {np.max(similarity_matrix)}")

    # Normalize relevance scores
    relevance_scores = (relevance_scores - np.min(relevance_scores)) / (np.max(relevance_scores) - np.min(relevance_scores))
    print(f"Normalized Relevance Scores. Min: {np.min(relevance_scores)}, Max: {np.max(relevance_scores)}")

    NDCG = 0.0
    HT = 0.0
    predict_num = len(answers)
    diversified_predictions = []

    for idx, (answer, prediction) in enumerate(zip(answers, llm_predictions)):
        print(f"\nProcessing answer {idx + 1}/{len(answers)}: {answer}")
        print(f"Predictions for current answer: {prediction}")

        if not prediction or prediction in [[''], ['?'], ['!'], ['_________']]:
            print(f"Skipping invalid prediction for answer: {answer}")
            diversified_predictions.append([])
            continue

        mmr_indices = mmr(relevance_scores, similarity_matrix, lambda_val=lambda_val, top_k=k)
        print(f"Answer: {answer}, Prediction: {prediction}, MMR Indices: {mmr_indices}")

        recommended = [prediction[i] for i in mmr_indices if i < len(prediction)]
        diversified_predictions.append(recommended)
        print(f"Final Recommendations: {recommended}")

        if answer in recommended:
            rank = recommended.index(answer) + 1
            NDCG += 1 / np.log2(rank + 1)
            HT += 1

    ild = intra_list_diversity(similarity_matrix, diversified_predictions)
    return NDCG / predict_num, HT / predict_num, ild, diversified_predictions


if __name__ == "__main__":
    inferenced_file_path = './recommendation_output.txt'
    answers, llm_predictions = get_answers_predictions(inferenced_file_path)

    print(f"Loaded {len(answers)} answers and {len(llm_predictions)} predictions.")
    if len(answers) == 0 or len(llm_predictions) == 0:
        print("Error: No answers or predictions found. Exiting.")
        exit()

    try:
        relevance_scores = np.load('./output/relevance_scores.npy', allow_pickle=True)
        similarity_matrix = np.load('./output/embeddings.npy', allow_pickle=True)

        if len(relevance_scores) > similarity_matrix.shape[0]:
            print("Error: Relevance scores exceed similarity matrix rows. Truncating.")
            relevance_scores = relevance_scores[:similarity_matrix.shape[0]]

        print("Relevance scores loaded:", relevance_scores.shape)
        print("Similarity matrix loaded:", similarity_matrix.shape)

    except FileNotFoundError as e:
        print(f"Error loading relevance scores or similarity matrix: {e}")
        exit()

    ndcg, ht, ild, diversified_predictions = evaluate(
        answers, llm_predictions, relevance_scores, similarity_matrix, k=5, lambda_val=0.5
    )

    print(f"\nEvaluation Results:")
    print(f"NDCG@5: {ndcg}")
    print(f"Hit Rate@5: {ht}")
    print(f"Intra-List Diversity: {ild}")

    with open('./output/diversified_recommendations.txt', 'w') as f:
        for preds in diversified_predictions:
            f.write(', '.join(preds) + '\n')
    print("Diversified recommendations saved to './output/diversified_recommendations.txt'.")
