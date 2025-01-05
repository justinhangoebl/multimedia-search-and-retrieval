import numpy as np

def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def euclidian_sim(u, v):
    return 1/(1+np.linalg.norm(u - v))

def manhattan_sim(u, v):
    return 1/(1+np.sum(np.abs(u - v)))

def jaccard_similarity(u, v):
    return len(set(u).intersection(set(v))) / len(set(u).union(set(v)))

def kl_divergence(p, q, epsilon=1e-10):
    # Add a small constant to avoid log(0) issues
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    # Normalize histograms to make them probability distributions
    p /= np.sum(p)
    q /= np.sum(q)
    # Compute KL Divergence
    return np.sum(p * np.log(p / q))

def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)