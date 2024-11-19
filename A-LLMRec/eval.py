import numpy as np
from scipy.linalg import det

def get_answers_predictions(file_path):
    answers = []
    llm_predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            if 'Answer:' == line[:len('Answer:')]:
                answer = line.replace('Answer:', '').strip()[1:-1].lower()
                answers.append(answer)
            if 'LLM:' == line[:len('LLM:')]:
                llm_prediction = line.replace('LLM:', '').strip().lower()
                try:
                    llm_prediction = llm_prediction.replace("\"item title\" : ", '')
                    start = llm_prediction.find('"')
                    end = llm_prediction.rfind('"')

                    if (start + end < start) or (start + end < end):
                        print(1/0)
                        
                    llm_prediction = llm_prediction[start+1:end]
                except Exception as e:
                    print()
                    
                llm_predictions.append(llm_prediction)
                
    return answers, llm_predictions

def compute_dpp_likelihood(kernel_matrix, item_set):
    sub_matrix = kernel_matrix[np.ix_(item_set, item_set)]
    return det(sub_matrix)

def evaluate(answers, llm_predictions, kernel_matrix, k=1):
    NDCG = 0.0
    HT = 0.0
    Diversity = 0.0
    predict_num = len(answers)
    print(predict_num)
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

        # Compute DPP likelihood for diversity
        item_indices = [answers.index(answer)]  # Example for single item, adjust as needed
        Diversity += compute_dpp_likelihood(kernel_matrix, item_indices)
                
    return NDCG / predict_num, HT / predict_num, Diversity / predict_num

if __name__ == "__main__":
    inferenced_file_path = './recommendation_output.txt'
    kernel_matrix_path = './kernel_matrix.npy'  # Example path to your kernel matrix file
    
    answers, llm_predictions = get_answers_predictions(inferenced_file_path)
    print(len(answers), len(llm_predictions))
    assert(len(answers) == len(llm_predictions))
    
    kernel_matrix = np.load(kernel_matrix_path)  # Load the kernel matrix
    
    ndcg, ht, diversity = evaluate(answers, llm_predictions, kernel_matrix, k=1)
    print(f"NDCG at 1: {ndcg}")
    print(f"Hit at 1: {ht}")
    print(f"Diversity: {diversity}")
