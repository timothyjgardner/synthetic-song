"""
Levina-Bickel Intrinsic Dimension Estimation Demo

This script demonstrates how the MLE intrinsic dimension estimator behaves
on a 1D closed loop (circle) embedded in higher dimensions, with varying
levels of added noise.

Key insight: The estimated dimension is scale-dependent when noise is present.
- Small k (local scale): noise dominates → dimension approaches ambient noise dimension
- Large k (global scale): manifold structure dominates → dimension approaches 1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def generate_noisy_circle(n_points, ambient_dim=10, noise_std=0.0, seed=42):
    """
    Generate points on a unit circle embedded in high-dimensional space with added noise.
    
    Parameters:
    -----------
    n_points : int
        Number of points to sample
    ambient_dim : int
        Dimension of the ambient space (must be >= 2)
    noise_std : float
        Standard deviation of Gaussian noise added to all dimensions
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    X : ndarray of shape (n_points, ambient_dim)
        The noisy circle data
    """
    np.random.seed(seed)
    
    # Sample angles uniformly
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    # Create circle in first two dimensions
    X = np.zeros((n_points, ambient_dim))
    X[:, 0] = np.cos(theta)
    X[:, 1] = np.sin(theta)
    
    # Add Gaussian noise to all dimensions
    if noise_std > 0:
        X += np.random.normal(0, noise_std, X.shape)
    
    return X


def levina_bickel_estimator(X, k):
    """
    Compute the Levina-Bickel MLE intrinsic dimension estimator.
    
    Parameters:
    -----------
    X : ndarray of shape (n_points, n_features)
        The data matrix
    k : int
        Number of nearest neighbors to use
    
    Returns:
    --------
    m_hat : float
        Estimated intrinsic dimension (averaged over all points)
    m_hat_per_point : ndarray
        Dimension estimate at each point
    """
    n = X.shape[0]
    
    # Compute all pairwise distances
    distances = cdist(X, X)
    
    # For each point, get sorted distances to neighbors (excluding self)
    m_hat_per_point = np.zeros(n)
    
    for i in range(n):
        # Sort distances and exclude self (distance 0)
        dists = np.sort(distances[i, :])[1:k+1]  # T_1, T_2, ..., T_k
        
        # Levina-Bickel formula (Eq. 8 in the paper):
        # m_hat_k(x) = [ (1/(k-1)) * sum_{j=1}^{k-1} log(T_k / T_j) ]^{-1}
        T_k = dists[-1]  # k-th nearest neighbor distance
        
        # Compute sum of log ratios (exclude j=k since log(T_k/T_k) = 0)
        log_ratios = np.log(T_k / dists[:-1])  # T_1 to T_{k-1}
        
        # Avoid division by zero for very small neighborhoods
        mean_log_ratio = np.mean(log_ratios)
        if mean_log_ratio > 1e-10:
            m_hat_per_point[i] = 1.0 / mean_log_ratio
        else:
            m_hat_per_point[i] = np.nan
    
    # Average over all points (excluding any NaN values)
    m_hat = np.nanmean(m_hat_per_point)
    
    return m_hat, m_hat_per_point


def run_experiment():
    """
    Run the full experiment showing dimension estimates vs k for various noise levels.
    """
    # Parameters
    n_points = 1000
    ambient_dim = 10
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    k_values = np.arange(5, 101, 5)
    
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors for different noise levels
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(noise_levels)))
    
    # Left plot: Dimension estimate vs k for different noise levels
    ax1 = axes[0]
    
    for noise_std, color in zip(noise_levels, colors):
        X = generate_noisy_circle(n_points, ambient_dim, noise_std)
        
        dim_estimates = []
        for k in k_values:
            m_hat, _ = levina_bickel_estimator(X, k)
            dim_estimates.append(m_hat)
        
        label = f'σ = {noise_std}' if noise_std > 0 else 'No noise'
        ax1.plot(k_values, dim_estimates, 'o-', color=color, label=label, 
                 markersize=4, linewidth=1.5)
    
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='True dim = 1')
    ax1.axhline(y=2, color='gray', linestyle=':', alpha=0.5, label='Noise dim = 2')
    ax1.set_xlabel('k (number of neighbors)', fontsize=12)
    ax1.set_ylabel('Estimated dimension', fontsize=12)
    ax1.set_title('Levina-Bickel Dimension Estimate vs k\n(Circle in 10D with Gaussian noise)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0.5, 3.0)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Visualize the data (projected to 2D) for a few noise levels
    ax2 = axes[1]
    
    noise_to_show = [0.0, 0.05, 0.2]
    markers = ['o', 's', '^']
    
    for noise_std, marker, color in zip(noise_to_show, markers, [colors[0], colors[2], colors[4]]):
        X = generate_noisy_circle(200, ambient_dim, noise_std)  # Fewer points for clarity
        label = f'σ = {noise_std}' if noise_std > 0 else 'No noise'
        ax2.scatter(X[:, 0], X[:, 1], c=[color], marker=marker, alpha=0.6, 
                    s=30, label=label, edgecolors='none')
    
    ax2.set_xlabel('x₁', fontsize=12)
    ax2.set_ylabel('x₂', fontsize=12)
    ax2.set_title('Data Visualization\n(First 2 dimensions)', fontsize=12)
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('levina_bickel_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Results saved to levina_bickel_results.png")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY: Estimated dimension at k=10 vs k=50")
    print("="*60)
    print(f"{'Noise (σ)':<12} {'k=10':<12} {'k=50':<12} {'k=100':<12}")
    print("-"*60)
    
    for noise_std in noise_levels:
        X = generate_noisy_circle(n_points, ambient_dim, noise_std)
        m_10, _ = levina_bickel_estimator(X, 10)
        m_50, _ = levina_bickel_estimator(X, 50)
        m_100, _ = levina_bickel_estimator(X, 100)
        label = f'{noise_std}' if noise_std > 0 else '0 (none)'
        print(f"{label:<12} {m_10:<12.2f} {m_50:<12.2f} {m_100:<12.2f}")
    
    print("-"*60)
    print("\nKey observations:")
    print("• No noise: dimension ≈ 1 for all k (correct)")
    print("• With noise: small k gives higher dimension (noise dominates)")
    print("• With noise: large k approaches 1 (manifold structure emerges)")
    print("• Higher noise requires larger k to 'see through' to the manifold")


if __name__ == "__main__":
    run_experiment()
