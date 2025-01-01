import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

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
        
        # Return track IDs and similarity scores
        return [
            (self.track_ids[idx], similarities[idx])
            for idx in similar_indices
        ]

    def batch_similarity_search(self, query_ids, top_k=5):
        """
        Perform similarity search for multiple query tracks
        
        :param query_ids: List of track IDs to query
        :param top_k: Number of similar tracks to return per query
        :return: Dict mapping query IDs to lists of similar tracks
        """
        results = {}
        for query_id in tqdm(query_ids):
            results[query_id] = self.find_similar_tracks(query_id, top_k)
        return results