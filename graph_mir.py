import networkx as nx
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class MusicRetrievalSystem:
    def __init__(self, feature_files):
        self.feature_files = feature_files
        self.features = {}
        self.graph = nx.Graph()
        
    def load_features(self):
        for feature_name, file_path in self.feature_files.items():
            try:
                # Load feature matrix, assuming first column is ID
                df = pd.read_csv(file_path, sep='\t', index_col=0)
                
                # Normalize numerical features
                scaler = StandardScaler()
                normalized_features = scaler.fit_transform(df)
                self.features[feature_name] = {
                    'raw_data': df,
                    'normalized_features': normalized_features,
                    'ids': df.index.tolist()
                }
                    
            except Exception as e:
                print(f"Error loading {feature_name} features: {e}")
        
    def create_similarity_graph(self, similarity_threshold=0.75):
        """
        Create a graph based on feature similarities
        
        :param similarity_threshold: Minimum similarity to create an edge
        """
        # Combine features from different modalities
        combined_features = []
        combined_ids = []
        
        # Align features across different modalities
        for _, feature_data in self.features.items():
            combined_features.append(feature_data['normalized_features'])
            combined_ids.append(feature_data['ids'])
        
        # Ensure all feature sets have the same IDs
        common_ids = set.intersection(*[set(ids) for ids in combined_ids])
        
        # Reduce dimensionality using UMAP for multi-modal feature fusion
        combined_features_aligned = []
        for feature_data in self.features.values():
            aligned_subset = [
                feature_data['normalized_features'][feature_data['ids'].index(id)]
                for id in common_ids if id in feature_data['ids']
            ]
            combined_features_aligned.append(aligned_subset)
        
        # Concatenate features across modalities
        fused_features = np.hstack(combined_features_aligned)
        
        # Dimensionality reduction
        reducer = umap.UMAP(n_components=10, random_state=42)
        reduced_features = reducer.fit_transform(fused_features)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(reduced_features)
        similarity_matrix = cosine_similarity(fused_features)
        
        # Create graph based on similarity
        for i, id_i in tqdm(enumerate(common_ids), total=len(common_ids), desc="Creating graph"):
            for j, id_j in enumerate(common_ids):
                if i != j and similarity_matrix[i, j] > similarity_threshold:
                    self.graph.add_edge(id_i, id_j, weight=similarity_matrix[i, j])
        
    def recommend_similar_tracks(self, track_id, top_k=5):
        if track_id not in self.graph:
            return []
        
        # Get neighbors sorted by edge weight
        neighbors = sorted(
            self.graph.neighbors(track_id), 
            key=lambda x: self.graph[track_id][x].get('weight', 0), 
            reverse=True
        )
        
        return neighbors[:top_k]
    
    def extract_multimodal_features(self):
        # Combine features from different modalities
        multimodal_features = {}
        
        # Audio features
        audio_features = [
            'mfcc_bow', 'mfcc_stats',
            'blf_spectral', 'blf_spectralcontrast', 'blf_correlation', 'blf_deltaspectral', 'blf_logfluc', 'blf_vardeltaspectral',
            'musicnn',
            'ivec256', 'ivec512', 'ivec1024'
        ]
        
        # Metadata features
        metadata_features = [
            'lyrics_bert', 'lyrics_word2vec', 'lyrics_tf-idf',
            'total_listens', 
            'tags_dict', # tags_dict is not yet implement needs an number encoder
            'metadata',  # metadata is not yet implement needs an number encoder

        ]
        
        # Video features
        video_features = [
            'vgg19', 
            'resnet',
            'incp' #inception
            ]
        
        # Method to combine features
        def combine_features(feature_types):
            combined = []
            for feature_type in feature_types:
                matching_features = [
                    f for f in self.features.keys() 
                    if feature_type in f.lower()
                ]
                for feature in matching_features:
                    combined.append(self.features[feature]['normalized_features'])
            return np.hstack(combined) if combined else None
        
        multimodal_features = {
            'audio': combine_features(audio_features),
            'metadata': combine_features(metadata_features),
            'video': combine_features(video_features)
        }
        
        return multimodal_features
