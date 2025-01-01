import numpy as np

def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def euclidian_sim(u, v):
    return 1/(1+np.linalg.norm(u - v))

def manhattan_sim(u, v):
    return 1/(1+np.sum(np.abs(u - v)))

def jaccard_similarity(u, v):
    return len(set(u).intersection(set(v))) / len(set(u).union(set(v)))