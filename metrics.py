import numpy as np

#accurcay metrics

def precision_at_k_1d(y_pred, relevant, k):
    # Get the indices of the top-k predictions
    pred_index = np.argsort(y_pred)[::-1][:k]
    # Count how many of the top-k are relevant
    relevant_count = sum([relevant[i] for i in pred_index])
    # Compute precision
    return relevant_count / k

def recall_at_k_1d(y_pred, relevant, k):
    # Get the indices of the top-k predictions
    pred_index = np.argsort(y_pred)[::-1][:k]
    # Count how many of the top-k are relevant
    relevant_count = sum([relevant[i] for i in pred_index])
    # Compute recall
    return relevant_count / sum(relevant) if sum(relevant) > 0 else 0

def precision_at_k(y_pred, relevant, k):
    return np.mean([precision_at_k_1d(y, r, k) for y, r in zip(y_pred, relevant)])

def recall_at_k(y_pred, relevant, k):
    return np.mean([recall_at_k_1d(y, r, k) for y, r in zip(y_pred, relevant)])

def mrr(y_pred, relevant):
    arr = []
    for y, r in zip(y_pred, relevant):
        # Rank indices based on predicted scores (descending)
        ranked_indices = np.argsort(y)[::-1]
        
        # Find the rank of the first relevant item
        for i, idx in enumerate(ranked_indices):
            if r[idx] > 0:  # Check if the item is relevant
                arr.append(1 / (i + 1))  # MRR is the reciprocal of the rank (1-based)
                continue
        
        # If no relevant items are found, return 0
        arr.append(0)
    
    return np.mean(arr)

def ndcg_at_k_1d(y_pred, relevant, k):
    if int(relevant.sum()) == 0:
        return 0
    # Get indices of top-k predictions
    pred_index = np.argsort(y_pred)[::-1][:k]
    # Rank the relevance scores based on predicted order
    ranked_relevance = relevant[pred_index]
    # Compute DCG
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ranked_relevance))
    # Compute IDCG (ideal DCG)
    ideal_relevance = np.sort(relevant)[::-1][:k]  # Ideal ranking
    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
    # Return NDCG
    return dcg / idcg if idcg > 0 else 0

def ndcg_at_k(y_pred, relevant, k):
    return np.mean([ndcg_at_k_1d(y, r, k) for y, r in zip(y_pred, relevant)])


# behavior metrics

def avg_popularity_at_k(y_pred, k, item_popularity):
    return np.mean([np.mean([item_popularity[i] for i in y[:k]]) for y in y_pred])

