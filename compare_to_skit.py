import numpy as np
#from sklearn.decomposition import PCA as SklearnPCA
import scipy.linalg
from jax import random
import jax.numpy as jnp
from jax_pca import PCA

# Define JAX PCA class (reuse the implementation provided earlier)
# Import or copy-paste the JAX PCA implementation here

# Generate synthetic data
key = random.PRNGKey(42)
num_samples, num_features = 1000, 10
X_jax = random.normal(key, (num_samples, num_features))  # Data for JAX
X_scipy = np.array(X_jax)  # Convert JAX data to NumPy for SciPy

# Parameters
num_components = 5

# --- JAX PCA ---
jax_pca = PCA(num_components)
X_jax_transformed = jax_pca.fit_transform(X_jax)
X_jax_recovered = jax_pca.recover(X_jax_transformed)

# --- SciPy PCA ---
# Compute mean
mean_scipy = np.mean(X_scipy, axis=0)
X_scipy_centered = X_scipy - mean_scipy

# Covariance matrix and eigendecomposition
cov_matrix_scipy = np.cov(X_scipy_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix_scipy)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Project data to top components
eigenvectors_selected = eigenvectors[:, :num_components]
X_scipy_transformed = X_scipy_centered @ eigenvectors_selected

# Reconstruct the data
X_scipy_recovered = X_scipy_transformed @ eigenvectors_selected.T + mean_scipy

# --- Compare Results ---
print("Comparison Results:\n")

# Compare variance explained
explained_variance_jax = jnp.var(X_jax_transformed, axis=0)
explained_variance_scipy = np.var(X_scipy_transformed, axis=0)

print("Explained Variance (JAX):", explained_variance_jax)
print("Explained Variance (SciPy):", explained_variance_scipy)

# Compare reconstruction
reconstruction_error_jax = jnp.mean((X_jax - X_jax_recovered) ** 2)
reconstruction_error_scipy = np.mean((X_scipy - X_scipy_recovered) ** 2)

print("\nReconstruction Error (JAX):", reconstruction_error_jax)
print("Reconstruction Error (SciPy):", reconstruction_error_scipy)

# Compare computation times
import time

# JAX Timing
jax_start = time.time()
for _ in range(10):
    jax_pca.fit_transform(X_jax)
jax_end = time.time()
jax_time = (jax_end - jax_start) / 10

# SciPy Timing
scipy_start = time.time()
for _ in range(10):
    _ = X_scipy_centered @ eigenvectors_selected
scipy_end = time.time()
scipy_time = (scipy_end - scipy_start) / 10

print("\nComputation Time (JAX):", jax_time, "seconds")
print("Computation Time (SciPy):", scipy_time, "seconds")

# --- Compare Outputs ---
print("\nDo Transformed Results Match?")
print("Max Absolute Difference in Transformed Data:",
      np.max(np.abs(np.array(X_jax_transformed) - X_scipy_transformed)))

print("\nDo Reconstructed Results Match?")
print("Max Absolute Difference in Reconstructed Data:",
      np.max(np.abs(np.array(X_jax_recovered) - X_scipy_recovered)))
