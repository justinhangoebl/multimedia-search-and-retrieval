import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from tqdm_joblib import tqdm_joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from itertools import combinations

def plot_all_pairwise_gaussians(song_id, precomputed_data, num_features=13):
    """
    Plots a 2D projection of the Gaussian for all pairs of features in the dataset.
    
    Args:
        song_id: The ID of the song to plot.
        precomputed_data: Dictionary mapping song IDs to (mean, covariance matrix).
        num_features: The number of features in the feature space (default: 14).
    """
    # Extract mean and covariance matrix for the song
    mean, cov_matrix = precomputed_data[song_id]
    
    # Ensure the number of features matches
    if len(mean) != num_features or cov_matrix.shape[0] != num_features or cov_matrix.shape[1] != num_features:
        raise ValueError(f"Mean vector and covariance matrix must have {num_features} dimensions.")
    
    # Generate all combinations of 2 features
    feature_combinations = list(combinations(range(num_features), 2))
    
    # Create subplots for each pair of features
    ncols = 4  # Number of columns for subplots
    nrows = int(np.ceil(len(feature_combinations) / ncols))  # Number of rows for subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 6))
    axes = axes.flatten()  # Flatten the axes array to easily index
    
    for i, (f1, f2) in enumerate(feature_combinations):
        ax = axes[i]
        
        # Project to 2D using the specified feature combination
        mean_2d = mean[[f1, f2]]
        cov_matrix_2d = cov_matrix[np.ix_([f1, f2], [f1, f2])]
        
        # Create a grid of points for the 2D space
        x, y = np.linspace(mean_2d[0] - 3, mean_2d[0] + 3, 100), np.linspace(mean_2d[1] - 3, mean_2d[1] + 3, 100)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        
        # Compute Gaussian PDF
        rv = multivariate_normal(mean_2d, cov_matrix_2d)
        Z = rv.pdf(pos)
        
        # Plot the Gaussian distribution as a contour plot
        cs = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        fig.colorbar(cs, ax=ax, label="Probability Density")
        ax.scatter(mean_2d[0], mean_2d[1], color='red', label="Mean", zorder=5)
        ax.set_title(f"Features {f1} vs {f2}")
        ax.set_xlabel(f"Feature {f1}")
        ax.set_ylabel(f"Feature {f2}")
        ax.legend()
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Function to construct a covariance matrix from the upper triangular values
def construct_covariance_matrix(cov_values, dim=13):
    covariance_matrix = np.zeros((dim, dim))
    upper_triangle_indices = np.triu_indices(dim)
    covariance_matrix[upper_triangle_indices] = cov_values
    covariance_matrix = covariance_matrix + covariance_matrix.T - np.diag(covariance_matrix.diagonal())
    return covariance_matrix

# KL Divergence between two Gaussian distributions
def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    dim = len(mu1)  # Dimensionality
    sigma2_inv = np.linalg.inv(sigma2)
    
    term1 = np.trace(sigma2_inv @ sigma1)
    term2 = (mu2 - mu1).T @ sigma2_inv @ (mu2 - mu1)
    term3 = np.log2(np.linalg.det(sigma2) / np.linalg.det(sigma1))
    kl_div = 0.5 * (term1 + term2 - dim + term3)
    return kl_div

def jensen_shannon_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    mu_m = 0.5 * (mu1 + mu2)
    kl1 = kl_divergence_gaussian(mu1, sigma1, mu_m, 0.5 * (sigma1 + sigma2))
    kl2 = kl_divergence_gaussian(mu2, sigma2, mu_m, 0.5 * (sigma1 + sigma2))
    return 1 - 0.5 * (kl1 + kl2)

def compute_similarity_for_song(song_base, infos, precomputed_data, similarity_fn, topK):
    base_means, base_cov_matrix = precomputed_data[song_base['id']]
    rets_i = np.zeros(len(infos))
    
    for j, song_target in infos.iterrows():
        target_means, target_cov_matrix = precomputed_data[song_target['id']]
        sim = similarity_fn(base_means, base_cov_matrix, target_means, target_cov_matrix)
        rets_i[j] = sim
    
    # Get top K recommendations
    indices = np.argsort(rets_i)[::-1][:topK]
    scores = np.sort(rets_i)[::-1][:topK]
    return indices, scores, song_base.name  # Returning the index for the song to update rets matrix


def main():
    infos = pd.read_csv("dataset/id_information_mmsr.tsv", sep="\t")
    topK = 10
    similarity_fn = jensen_shannon_divergence_gaussian
    vector = pd.read_csv(f"dataset/id_mfcc_stats_mmsr.tsv", sep="\t")
    degress = 13

    rets = np.zeros((len(infos), len(infos)))
    # Convert to NumPy arrays for faster access
    song_ids = infos['id'].values
    vector_data = vector.values

    # Precompute means and covariance matrices
    precomputed_data = {}
    for i, song_id in tqdm(enumerate(song_ids), total=len(song_ids), desc="Precomputing data"):
        song_vector = vector_data[i]
        means = song_vector[1:14]
        cov_values = song_vector[14:]
        cov_matrix = construct_covariance_matrix(cov_values, dim=degress)
        precomputed_data[song_id] = (means, cov_matrix)

    # Number of CPU cores to use (half of available cores for parallelism)
    n_jobs = max(1, os.cpu_count() // 2)
    print(f"Using {n_jobs} cores for similarity computation.")

    # Parallelize similarity computation
    with tqdm_joblib(total=len(infos)) as pbar:
        results = Parallel(n_jobs=n_jobs)(delayed(compute_similarity_for_song)(song_base, infos, precomputed_data, similarity_fn, topK) 
                                          for _, song_base in infos.iterrows())
        
        # Populate the rets matrix with the results
        for indices, scores, song_index in results:
            rets[song_index, indices] = scores
            pbar.update(1)  # Update progress bar

    # Save the results to a CSV file
    np.savetxt(f"./predictions/rets_mfcc_stats_gaussian_jensen_matrix_10.csv", rets, delimiter=",")

if __name__ == "__main__":
    main()

