import numpy as np
import pandas as pd
import ast
import os
import networkx as nx
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from collections import defaultdict

class GraphMetaData:
    def __init__(self):
        self.graph = nx.MultiGraph()
        
        self.infos = pd.read_csv('dataset/id_information_mmsr.tsv', sep='\t')
        self.metadata = pd.read_csv('dataset/id_metadata_mmsr.tsv', sep='\t')
        tags = pd.read_csv('dataset/id_tags_dict.tsv', sep='\t')
        genres_base = pd.read_csv('dataset/top_genres.tsv', sep='\t')

        self.genres = genres_base.copy()
        self.genres['top_genre'] = genres_base['top_genre'].apply(ast.literal_eval)

        self.valence =      self.metadata['valence'].values
        self.tempos =       self.metadata['tempo'].values
        self.energy =       self.metadata['energy'].values
        self.danceability = self.metadata['danceability'].values
        self.key =          self.metadata['key'].values

        self.__generate_adj_matrix()

    def __get_neighbors_with_weights(self, node):
        neighbors = []
        for neighbor, attrs in self.graph[node].items():  # Iterate through neighbors and edge attributes
            weight = attrs[0].get("weight", 1)  # Default weight is 1 if not specified
            neighbors.append((neighbor, weight))
        return neighbors

    def __bin_scores(self, scores, bins=20):
        bins_array = np.linspace(0, 1, bins + 1)  # Renamed to avoid confusion with 'bins'
        return np.digitize(scores, bins_array, right=False) - 1

    def __bin_quantile_based(self, scores, bins=20):
        return pd.qcut(scores, q=bins, labels=[f"{i}" for i in range(bins)])
        
    def __create_adjacency_matrix(self, arr):
        # Create a mapping of elements to indices
        unique_elements = np.unique(arr)
        element_index = {element: idx for idx, element in enumerate(unique_elements)}
        
        # Map the string array to indices
        indices = np.array([element_index[element] for element in arr])
        
        # Create an adjacency matrix using broadcasting
        adj_matrix = np.equal(indices[:, None], indices).astype(int)
        
        return adj_matrix
    
    def __generate_adj_matrix(self):
        self.sorted_genres = []
        sorted_danceability = []
        sorted_energy = []
        sorted_valence = []
        sorted_tempo = []
        sorted_key = []

        for _, row in self.infos.iterrows():
            self.sorted_genres.append(  self.genres  [self.genres['id'] == row['id']]['top_genre'].values[0])
            sorted_danceability.append( self.metadata[self.metadata['id'] == row['id']]['danceability'].values[0])
            sorted_energy.append(       self.metadata[self.metadata['id'] == row['id']]['energy'].values[0])
            sorted_valence.append(      self.metadata[self.metadata['id'] == row['id']]['valence'].values[0])
            sorted_tempo.append(        self.metadata[self.metadata['id'] == row['id']]['tempo'].values[0])
            sorted_key.append(          self.metadata[self.metadata['id'] == row['id']]['key'].values[0])

        binned_danceability_scores =    self.__bin_scores(sorted_danceability,  bins=20)
        binned_energy_scores =          self.__bin_scores(sorted_energy,        bins=20)
        binned_valence_scores =         self.__bin_scores(sorted_valence,       bins=20)
        binned_tempo_scores =           self.__bin_scores(sorted_tempo,         bins=20)
        
        self.adj_danceability =  self.__create_adjacency_matrix(binned_danceability_scores)
        self.adj_energy =        self.__create_adjacency_matrix(binned_energy_scores)
        self.adj_valence =       self.__create_adjacency_matrix(binned_valence_scores)
        self.adj_tempo =         self.__create_adjacency_matrix(binned_tempo_scores)
        self.adj_key =           self.__create_adjacency_matrix(sorted_key)

        self.adj_artist = np.zeros((len(self.infos), len(self.infos)), dtype=int)

        # Populate the matrix
        for i in range(len(self.infos)):
            for j in range(len(self.infos)):
                if self.infos.loc[i, "artist"] == self.infos.loc[j, "artist"]:
                    self.adj_artist[i, j] = 1

    def generate_multigraph(self, amount=5148):
        for (node_id, genres) in zip(self.infos['id'].values[:amount], self.sorted_genres[:amount]):
            self.graph.add_node(node_id, label=genres)

        for i, source_node_id in tqdm(enumerate(self.infos['id'].values[:amount]),total=amount):
            for j, target_node_id in enumerate(self.infos['id'].values[:amount]):
                if i >= j:
                    continue
                if(self.adj_artist[i, j] == 1):
                    self.graph.add_edge(source_node_id, target_node_id, **{"feature": 'artist'})
                if(self.adj_danceability[i, j] == 1):
                    self.graph.add_edge(source_node_id, target_node_id, **{"feature": 'danceability'})
                if(self.adj_energy[i, j] == 1):
                    self.graph.add_edge(source_node_id, target_node_id, **{"feature": 'energy'})
                if(self.adj_valence[i, j] == 1):
                    self.graph.add_edge(source_node_id, target_node_id, **{"feature": 'valence'})
                if(self.adj_tempo[i, j] == 1):
                    self.graph.add_edge(source_node_id, target_node_id, **{"feature": 'tempo'})
                if(self.adj_key[i, j] == 1): 
                    self.graph.add_edge(source_node_id, target_node_id, **{"feature": 'key'})

    def generate_graph(self, 
                      amount=5148,
                      weight={'artist': 1, 'danceability': 0.2, 'energy': 0.2, 'valence': 0.2, 'tempo': 0.2, 'key': 0.2}, th=0.5):
        for (node_id, genres) in zip(self.infos['id'].values[:amount], self.sorted_genres[:amount]):
            self.graph.add_node(node_id, label=genres)

        for i, source_node_id in tqdm(enumerate(self.infos['id'].values[:amount]),total=amount):
            for j, target_node_id in enumerate(self.infos['id'].values[:amount]):
                if i >= j:
                    continue
                dance = self.adj_danceability[i, j]
                energy = self.adj_energy[i, j]
                valence = self.adj_valence[i, j]
                tempo = self.adj_tempo[i, j]
                key = self.adj_key[i, j]
                artist = self.adj_artist[i, j]

                weight_sum = weight['artist']*artist + weight['danceability'] * dance + weight['energy'] * energy + weight['valence'] * valence + weight['tempo'] * tempo + weight['key'] * key
                if(weight_sum > th):
                    self.graph.add_edge(source_node_id, target_node_id, weight=weight_sum)

    def save_graph(self, filename):
        nx.write_gexf(self.graph, filename)

    def load_graph(self, filename):
        self.graph = nx.read_gexf(filename)

    def nearest_neighbors_search(self, song_id, k=10, weighted=True):
        if song_id not in self.graph:
            return []
        
        neighbors = defaultdict(int)
        for neighbor in self.__get_neighbors_with_weights(song_id):
            if(weighted):
                neighbors[neighbor[0]] = neighbor[1]
            else:
                neighbors[neighbor] = len(self.graph.get_edge_data(song_id, neighbor))
        
        similar_songs = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{"source_id": song_id, "target_id": target, "similarity": sim} 
                for target, sim in similar_songs]

    def community_based_search(self, song_id, k=10):
        if song_id not in self.graph:
            return []
        
        weighted_G = nx.Graph()
        for u, v, data in self.graph.edges(data=True):
            if weighted_G.has_edge(u, v):
                weighted_G[u][v]['weight'] += 1
            else:
                weighted_G.add_edge(u, v, weight=1)
        
        communities = nx.community.louvain_communities(weighted_G)
        
        song_community = None
        for community in communities:
            if song_id in community:
                song_community = community
                break
        
        if not song_community:
            return []
        
        similar_songs = defaultdict(int)
        for neighbor in song_community:
            if neighbor != song_id and self.graph.has_edge(song_id, neighbor):
                similar_songs[neighbor] = len(self.graph.get_edge_data(song_id, neighbor))
        
        results = sorted(similar_songs.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{"source_id": song_id, "target_id": target, "similarity": sim} 
                for target, sim in results]
    
    def precompute_community_data(self):
        weighted_G = nx.Graph()
        for u, v, data in self.graph.edges(data=True):
            if weighted_G.has_edge(u, v):
                weighted_G[u][v]['weight'] += 1
            else:
                weighted_G.add_edge(u, v, weight=1)
        
        communities = nx.community.louvain_communities(weighted_G)
        node_to_community = {}
        for community in communities:
            for node in community:
                node_to_community[node] = community
                
        return weighted_G, communities, node_to_community

    def batch_community_search(self, song_ids, topK=10, precomputed_data=None):
        if precomputed_data is None:
            _, _, node_to_community = self.precompute_community_data()
        else:
            node_to_community = precomputed_data[2]
        
        results = []
        for song_id in tqdm(song_ids, desc="Processing for UI", total=len(song_ids)):
            if song_id not in self.graph:
                continue
                
            song_community = node_to_community.get(song_id)
            if not song_community:
                continue
            
            similar_songs = defaultdict(int)
            for neighbor in song_community:
                if neighbor != song_id and self.graph.has_edge(song_id, neighbor):
                    similar_songs[neighbor] = len(self.graph.get_edge_data(song_id, neighbor))
            
            top_k = sorted(similar_songs.items(), key=lambda x: x[1], reverse=True)[:topK]
            results.extend([
                {"source_id": song_id, "target_id": target, "similarity": sim} 
                for target, sim in top_k
            ])
        
        return pd.DataFrame(results)

    def batch_nearest_neighbor(self, infos, topK=10, single=True):
        nn = []
        for _, row in tqdm(infos.iterrows(), desc="Processing for UI", total = len(infos)):
            nn_search = self.nearest_neighbors_search(row["id"], topK, weighted=single)

            nn.extend(nn_search)
        return pd.DataFrame(nn)