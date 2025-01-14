import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Set, Tuple
from experimental_metrics import *
import numpy as np
from typing import List, Tuple

def get_diverse_recommendations_batch(
    query_similarities: np.ndarray,
    similarity_matrix: np.ndarray,
    k: int,
    lambda_diversity: float = 0.5,
    exclude_self: bool = True
) -> List[int]:
    n_items = len(query_similarities)
    selected_items = []
    available_mask = np.ones(n_items, dtype=bool)
    
    if exclude_self:
        query_index = np.argmax(query_similarities)
        available_mask[query_index] = False
    
    # Pre-allocate arrays for scores
    relevance_scores = query_similarities.copy()
    
    for _ in range(k):
        if not np.any(available_mask):
            break
            
        if selected_items:
            # Calculate diversity scores for all items at once
            similarities_to_selected = similarity_matrix[:, selected_items]
            diversity_scores = 1.0 - np.mean(similarities_to_selected, axis=1)
        else:
            diversity_scores = np.ones(n_items)
        
        # Calculate combined scores for all available items
        combined_scores = (1 - lambda_diversity) * relevance_scores + \
                         lambda_diversity * diversity_scores
        
        # Mask out unavailable items
        combined_scores[~available_mask] = float('-inf')
        
        # Select best item
        best_item = np.argmax(combined_scores)
        selected_items.append(best_item)
        available_mask[best_item] = False
    
    return selected_items

def get_diverse_recommendations(
    query_similarities: np.ndarray,
    similarity_matrix: np.ndarray,
    k: int,
    lambda_diversity: float = 0.5,
    exclude_self: bool = True
) -> List[int]:
    n_items = len(query_similarities)
    selected_items = []
    available_items = set(range(n_items))

    if exclude_self:
        # Exclude the query item itself
        query_index = np.argmax(query_similarities)
        available_items.remove(query_index)
    
    # Iteratively select remaining items
    for _ in range(k):
        if not available_items:
            break
        best_score = float('-inf')
        best_item = None
        
        # Calculate score for each candidate item
        for candidate in available_items:
            # Relevance score (from query similarities)
            relevance_score = query_similarities[candidate]
            
            if selected_items:
                diversity_score = 1.0 - np.mean(similarity_matrix[candidate, selected_items])
            else:
                diversity_score = 1.0  # No diversity penalty for the first selection

            
            # Combined score with trade-off parameter
            combined_score = (1 - lambda_diversity) * relevance_score + \
                           lambda_diversity * diversity_score
            
            # Update best item if this candidate has higher score
            if combined_score > best_score:
                best_score = combined_score
                best_item = candidate
        
        if best_item is not None:
            selected_items.append(best_item)
            available_items.remove(best_item)
    
    return selected_items

def evaluate_diversity(
    similarity_matrix: np.ndarray,
    recommended_items: List[int]
) -> Tuple[float, float]:
    n_items = len(recommended_items)
    if n_items < 2:
        return 0.0, 0.0
    
    similarities = []
    for i in range(n_items):
        for j in range(i + 1, n_items):
            item1, item2 = recommended_items[i], recommended_items[j]
            similarities.append(similarity_matrix[item1][item2])
    
    avg_similarity = np.mean(similarities)
    diverse_pairs = sum(1 for s in similarities if s < 0.5)
    prop_diverse = diverse_pairs / len(similarities)
    
    return avg_similarity, prop_diverse

def analyze_tradeoff(
    similarity_matrix: np.ndarray,
    query_similarities: np.ndarray,
    tags_df: List,
    genre_df: List[Set[str]],
    k: int = 10,
    lambda_range: List[float] = None,
    tag_threshold: float = 74
) -> dict:
    if lambda_range is None:
        lambda_range = np.arange(0.1, 1.0, 0.01)
    
    results = {
        'lambda_values': [],
        'tag_entropy': [],
        'unique_tag_ratio': [],
        'avg_relevance': [],
        'diversity_at_k': []
    }

    tags_list = [ast.literal_eval(x) for x in tags_df.drop(columns=['id'])['(tag, weight)'].tolist()]
    genre_tags = [ast.literal_eval(x) for x in genre_df.drop(columns=['id'])['genre'].tolist()]
    
    # Analyze each lambda value
    for lambda_val in tqdm(lambda_range, desc="Analyzing λ values"):
        # Get recommendations
        recommended_items = get_diverse_recommendations_tags(
            similarity_matrix=similarity_matrix,
            query_similarities=query_similarities,
            tags_list=tags_list,
            genre_tags=genre_tags,
            k=k,
            lambda_diversity=lambda_val,
            tag_threshold=tag_threshold
        )
        
        # Calculate metrics
        diversity_metrics = evaluate_tag_diversity(
            recommended_items=recommended_items,
            tags_list=tags_list,
            genre_tags=genre_tags,
            tag_threshold=tag_threshold
        )
        
        # Calculate average relevance
        avg_relevance = np.mean([query_similarities[i] for i in recommended_items])
        
        # Calculate diversity@k
        div_k = diversity_at_k(
            y_pred=query_similarities.reshape(1, -1),
            tags=tags_df,
            genre_tags=genre_df,
            k=k,
            threshold=tag_threshold
        )
        
        # Store results
        results['lambda_values'].append(lambda_val)
        results['tag_entropy'].append(diversity_metrics['tag_entropy'])
        results['unique_tag_ratio'].append(diversity_metrics['unique_tag_ratio'])
        results['avg_relevance'].append(avg_relevance)
        results['diversity_at_k'].append(div_k)
    
    return results

def plot_tradeoff(results: dict, save_path: str = None):
    """
    Create visualizations for diversity-relevance trade-off analysis.
    
    Args:
        results: Dictionary containing analysis results
        save_path: Optional path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Tag Entropy vs Relevance
    ax1.scatter(results['avg_relevance'], results['tag_entropy'], 
               c=results['lambda_values'], cmap='viridis')
    ax1.set_xlabel('Average Relevance')
    ax1.set_ylabel('Tag Entropy')
    ax1.set_title('Tag Entropy vs Relevance Trade-off')
    
    # Plot 2: Metrics vs Lambda
    ax2.plot(results['lambda_values'], results['tag_entropy'], 
             label='Tag Entropy', marker='o')
    ax2.plot(results['lambda_values'], results['avg_relevance'], 
             label='Avg Relevance', marker='s')
    ax2.set_xlabel('λ Value')
    ax2.set_ylabel('Score')
    ax2.set_title('Metrics vs λ Value')
    ax2.legend()
    
    # Plot 3: Diversity@k vs Lambda
    ax3.plot(results['lambda_values'], results['diversity_at_k'], 
             color='red', marker='o')
    ax3.set_xlabel('λ Value')
    ax3.set_ylabel('Diversity@k')
    ax3.set_title('Diversity@k vs λ Value')
    
    # Plot 4: Unique Tag Ratio vs Lambda
    ax4.plot(results['lambda_values'], results['unique_tag_ratio'], 
             color='green', marker='o')
    ax4.set_xlabel('λ Value')
    ax4.set_ylabel('Unique Tag Ratio')
    ax4.set_title('Unique Tag Ratio vs λ Value')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def find_optimal_lambda(results: dict) -> float:
    """
    Find the optimal lambda value based on the trade-off analysis.
    Uses a simple weighted sum of normalized metrics.
    
    Returns:
        Optimal lambda value
    """
    # Normalize all metrics to [0,1] range
    norm_entropy = (np.array(results['tag_entropy']) - min(results['tag_entropy'])) / \
                  (max(results['tag_entropy']) - min(results['tag_entropy']))
    norm_relevance = (np.array(results['avg_relevance']) - min(results['avg_relevance'])) / \
                    (max(results['avg_relevance']) - min(results['avg_relevance']))
    norm_diversity = (np.array(results['diversity_at_k']) - min(results['diversity_at_k'])) / \
                    (max(results['diversity_at_k']) - min(results['diversity_at_k']))
    
    # Calculate combined score (equal weights)
    combined_score = (norm_entropy + norm_relevance + norm_diversity) / 3
    
    # Return lambda value with highest combined score
    optimal_idx = np.argmax(combined_score)
    return results['lambda_values'][optimal_idx]

# Example usage with the previous sample data creation
def main():
    
    print("Loading data...")
    late_fusion = np.loadtxt('./predictions/full_late_fusion_matrix.csv', delimiter=",")
    tags_list = pd.read_csv('./dataset/id_tags_dict.tsv', sep='\t')
    genre_tags = pd.read_csv('./dataset/id_genres_mmsr.tsv', sep='\t')

    query_similarities = late_fusion[0]
    
    print("Running trade-off analysis...")
    # Analyze trade-off
    results = analyze_tradeoff(
        similarity_matrix=late_fusion,
        query_similarities=query_similarities,
        tags_list=tags_list,
        genre_tags=genre_tags,
        k=10,
        lambda_range=np.arange(0.1, 1.0, 0.1)
    )

    print("Plotting results...")    
    # Plot results
    plot_tradeoff(results, save_path='tradeoff_analysis.png')
    
    # Find optimal lambda
    optimal_lambda = find_optimal_lambda(results)
    print(f"\nOptimal λ value: {optimal_lambda:.2f}")

if __name__ == "__main__":
    main()