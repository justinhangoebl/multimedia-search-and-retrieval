import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from typing import Callable, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class MultiModalEarlyFusionRetrieval:
    def __init__(self):
        self.features = {}
        self.fused_features = None
        self.track_ids = None
        
    def load_and_normalize_features(self, feature_files):
        """
        Load and normalize features from different modalities
        
        :param feature_files: dict mapping feature names to file paths
        """
        for feature_name, file_path in feature_files.items():
            try:
                # Load feature matrix, assuming first column is ID
                df = pd.read_csv(file_path, sep='\t', index_col=0)
                
                # Normalize numerical features
                scaler = StandardScaler()
                normalized_features = scaler.fit_transform(df)
                
                self.features[feature_name] = {
                    'normalized_features': normalized_features,
                    'ids': df.index.tolist()
                }
                    
            except Exception as e:
                print(f"Error loading {feature_name} features: {e}")
    
    def fuse_features(self, feature_weights):
        """
        Perform early fusion of features using provided weights
        
        :param feature_weights: list of tuples [(feature_name, weight)]
        """
        # Verify all features exist
        for feature_name, _ in feature_weights:
            if feature_name not in self.features:
                raise ValueError(f"Feature '{feature_name}' not found in loaded features")
        
        # Get common track IDs across all modalities
        common_ids = set.intersection(*[
            set(self.features[feature_name]['ids']) 
            for feature_name, _ in feature_weights
        ])
        
        # Align features and apply weights
        weighted_features = []
        for feature_name, weight in feature_weights:
            feature_data = self.features[feature_name]
            # Select features for common IDs
            aligned_features = np.array([
                feature_data['normalized_features'][feature_data['ids'].index(id)]
                for id in common_ids
            ])
            # Apply weight
            weighted_features.append(aligned_features * weight)
        
        # Combine weighted features
        self.fused_features = np.hstack(weighted_features)
        self.track_ids = list(common_ids)
    
    def find_similar_tracks(self, query_id, top_k=5):
        """
        Find similar tracks based on fused features
        
        :param query_id: ID of the query track
        :param top_k: Number of similar tracks to return
        :return: List of (track_id, similarity_score) tuples
        """
        if query_id not in self.track_ids:
            return []
        
        query_idx = self.track_ids.index(query_id)
        query_features = self.fused_features[query_idx].reshape(1, -1)
        
        # Compute similarities
        similarities = cosine_similarity(query_features, self.fused_features)[0]
        
        # Get top-k similar tracks (excluding the query track)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        retrieved = []
        
        for idx in similar_indices:
            track_id = self.track_ids[idx]
            similarity = similarities[idx]
            retrieved.append({
                "source_id": query_id,
                "target_id": track_id, 
                "similarity": similarity
                })

        # Return track IDs and similarity scores
        return retrieved

    def batch_similarity_search(self, query_ids, top_k=5):
        """
        :param query_ids: List of track IDs to query
        :param top_k: Number of similar tracks to return per query
        :return: Dict mapping query IDs to lists of similar tracks
        """
        n_jobs = max(1, os.cpu_count() // 2)
        print(f"Using {n_jobs} cores for batch similarity search.")
        # Define a function to perform the similarity search for a single query
        def find_similar_tracks_parallel(query_id):
            return self.find_similar_tracks(query_id, top_k)
        
        with tqdm_joblib(total=len(query_ids), desc="Performing Similarity Search", dynamic_ncols=True) as pbar:
            # Use Parallel to perform the similarity search for all query_ids
            query_results = Parallel(n_jobs=n_jobs)(delayed(find_similar_tracks_parallel)(query_id) for query_id in query_ids)

            pbar.update(len(query_ids))  # Update progress bar after each result is processed

        return [item for sublist in query_results for item in sublist]