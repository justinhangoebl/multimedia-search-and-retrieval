import numpy as np
import ast
from collections import Counter

#accurcay metrics

def precision_at_k_1d(y_pred, relevant, k):
    # Get indices of the top-k items based on scores
    top_k_indices = np.argsort(y_pred)[-k:][::-1]  # Sort descending
    # Count how many of the top-k items are relevant
    relevant_count = np.sum(relevant[top_k_indices])
        
    # Compute precision
    return relevant_count / k

def recall_at_k_1d(y_pred, relevant, k):
    # Get the indices of the top-k predictions
    top_k_indices  = np.argsort(y_pred)[::-1][:k]
    # Count how many of the top-k are relevant
    relevant_count = np.sum(relevant[top_k_indices])
    total_relevant = np.sum(relevant)
    # Compute recall
    return relevant_count / total_relevant if total_relevant > 0 else 0

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
                break
        
    
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

def avg_popularity_at_k(y_pred, item_popularity, k=10):
    top_k_items = np.argsort(y_pred, axis=1)[:, ::-1][:, :k]
    return np.mean([np.mean([item_popularity[i] for i in y]) for y in top_k_items])/np.mean(item_popularity)

def avg_coverage_at_k(y_pred, total_items, k=10):
    top_k_items = np.argsort(y_pred, axis=1)[:, ::-1][:, :k]
    
    # Find unique items in the recommendations
    unique_items = np.unique(top_k_items)
    
    # Calculate coverage as a proportion of total items
    return len(unique_items) / total_items


def intra_list_diversity_at_k(y_pred, item_similarity, k=10):
    top_k_items = np.argsort(y_pred, axis=1)[:, ::-1][:, :k]
    
    ild_scores = []
    for items in top_k_items:
        pairwise_diversity = [
            1 - item_similarity[i, j] for i in items for j in items if i != j
        ]
        avg_diversity = np.mean(pairwise_diversity)
        ild_scores.append(avg_diversity)
    
    return np.mean(ild_scores)


def novelty_at_k(y_pred, item_popularity, k=10):
    # Normalize popularity to avoid log(0)
    popularity_scores = np.clip(item_popularity, 1e-10, None)
    
    top_k_items = np.argsort(y_pred, axis=1)[:, ::-1][:, :k]
    
    novelty_scores = []
    for items in top_k_items:
        novelty = np.mean([np.log2(1 / popularity_scores[i]) for i in items])
        novelty_scores.append(novelty)
    
    return np.mean(novelty_scores)

def calculate_tag_entropy(tags_list):
    """
    Calculate Shannon entropy of tag distribution.
    Higher values indicate more diversity.
    
    Args:
        tags_list (list): List of tag sets for each item
    """
    if not tags_list:
        return 0
        
    # Flatten all tags and count frequencies
    all_tags = [tag for tags in tags_list for tag in tags]
    tag_counts = Counter(all_tags)
    total_tags = len(all_tags)
    
    # Calculate entropy
    entropy = 0
    for count in tag_counts.values():
        probability = count / total_tags
        entropy -= probability * np.log2(probability)
        
    return entropy / np.log2(total_tags) if total_tags > 0 else 0
    
def filter_tags(tags, threshold):
    tags_dict = ast.literal_eval(tags)
    return {tag: weight for tag, weight in tags_dict.items() if weight > threshold}

def diversity_at_k(y_pred, tags, genre_tags, k=10, threshold=74, filter=True):
    tags = [filter_tags(tag, threshold) for tag in tags['(tag, weight)']] if filter else tags['(tag, weight)']
    tags_list = [list(d.keys()) for d in tags]

    genre_tags = genre_tags['genre'].values
    result = [
        list(set(row1) - set(row2)) for row1, row2 in zip(tags_list, genre_tags)
    ]

    # Convert back to a 2D array
    tags_array = np.array(result, dtype=object)

    top_k_items = np.argsort(y_pred, axis=1)[:, ::-1][:, :k]
    
    entr_list = []
    for retrieved_for_query in top_k_items:
        list_of_tags = []
        for tags in tags_array[retrieved_for_query]:
            list_of_tags.append(tags)
        entr = calculate_tag_entropy(list_of_tags)
        entr_list.append(entr)
        
    return np.mean(entr_list)