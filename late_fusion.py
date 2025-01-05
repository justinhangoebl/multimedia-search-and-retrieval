import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from typing import Callable, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class MultiModalLateFusionRetrieval:
    def __init__(self):
        self.features = {}
        self.similarities = {}
        self.track_ids = None
        self._similarities_computed = False
    
    def load_and_normalize_features(self, feature_files: Dict[str, str]):
        """
        Load and normalize features from different modalities
        
        :param feature_files: dict mapping feature names to file paths
        """
        self.features.clear()
        self.similarities.clear()
        self._similarities_computed = False
        self.track_ids = None
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
    
    def compute_modality_similarities(self):
        """
        Compute similarity matrices for each modality separately
        """
        if not self.features:
            raise ValueError("No features loaded. Call load_and_normalize_features first.")
            
        # Find common track IDs across all modalities
        common_ids = set.intersection(*[
            set(feature_data['ids']) 
            for feature_data in self.features.values()
        ])
        
        if not common_ids:
            raise ValueError("No common track IDs found across modalities")
            
        self.track_ids = list(common_ids)
        
        # Compute similarities for each modality
        for feature_name, feature_data in self.features.items():
            # Get indices for common IDs
            indices = [feature_data['ids'].index(id) for id in common_ids]
            
            # Extract aligned features
            aligned_features = feature_data['normalized_features'][indices]
            
            # Compute similarity matrix
            self.similarities[feature_name] = cosine_similarity(aligned_features)
        
        self._similarities_computed = True

    def predefined_fusion_methods(self) -> Dict[str, Callable]:
        """
        Returns a dictionary of predefined fusion methods
        """
        return {
            'average': lambda similarities: np.mean(list(similarities.values()), axis=0),
            'maximum': lambda similarities: np.maximum.reduce(list(similarities.values())),
            'minimum': lambda similarities: np.minimum.reduce(list(similarities.values())),
            'weighted_average': lambda similarities, weights: np.average(
                list(similarities.values()), 
                weights=list(weights.values()), 
                axis=0
            )
        }
    
    def fuse_similarities(self, 
                         fusion_method: str = 'average',
                         weights: Dict[str, float] = None,
                         custom_fusion_func: Callable = None) -> np.ndarray:
        """
        :param fusion_method: String identifying the fusion method or 'custom'
        :param weights: Dictionary of weights for weighted fusion
        :param custom_fusion_func: Custom fusion function that takes dict of similarity matrices
        :return: Fused similarity matrix
        """
        if not self._similarities_computed:
            self.compute_modality_similarities()
            
        predefined_methods = self.predefined_fusion_methods()
        
        if custom_fusion_func is not None:
            return custom_fusion_func(self.similarities)
        elif fusion_method == 'weighted_average' and weights is not None:
            return predefined_methods[fusion_method](self.similarities, weights)
        elif fusion_method in predefined_methods:
            return predefined_methods[fusion_method](self.similarities)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def find_similar_tracks(self, 
                          query_id: str,
                          fused_similarities: np.ndarray = None,
                          top_k: int = 10,
                          fusion_method: str = 'average',
                          weights: Dict[str, float] = None,
                          custom_fusion_func: Callable = None) -> List[Tuple[str, float]]:
        """
        :param query_id: ID of the query track
        :param top_k: Number of similar tracks to return
        :param fusion_method: Fusion method to use
        :param weights: Weights for weighted fusion
        :param custom_fusion_func: Custom fusion function
        :return: List of (source_id, target_id, similarity) tuples
        """
        if not self._similarities_computed:
            self.compute_modality_similarities()
        if query_id not in self.track_ids:
            return []
        
        if(fused_similarities is None):
            fused_similarities = self.fuse_similarities(
                fusion_method=fusion_method,
                weights=weights,
                custom_fusion_func=custom_fusion_func
            )
        
        # Get query track index
        query_idx = self.track_ids.index(query_id)
        
        # Get similarities for query track
        similarities = fused_similarities[query_idx]
        
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

    def batch_similarity_search(self,
                                query_ids: List[str],
                                top_k: int = 10,
                                fusion_method: str = 'average',
                                weights: Dict[str, float] = None,
                                custom_fusion_func: Callable = None) -> Dict[str, List[Tuple[str, float]]]:
        # Define the number of jobs (cores) for parallel processing
        n_jobs = max(1, os.cpu_count() // 2)
        print(f"Using {n_jobs} cores for batch similarity search.")
        fused_similarities = self.fuse_similarities(
                fusion_method=fusion_method,
                weights=weights,
                custom_fusion_func=custom_fusion_func
            )
        
        def process_query(query_id):
            # Perform the similarity search for each query_id
            return self.find_similar_tracks(
                query_id,
                fused_similarities=fused_similarities,
                top_k=top_k,
                fusion_method=fusion_method,
                weights=weights,
                custom_fusion_func=custom_fusion_func
            )

        # Use joblib for parallel execution with progress bar
        with tqdm(total=len(query_ids), desc="Processing batch similarity search") as pbar:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_query)(query_id) for query_id in query_ids
            )

            # Update the progress bar as each job completes
            pbar.update(len(query_ids))

        return [item for sublist in results for item in sublist]