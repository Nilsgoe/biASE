import jax.numpy as jnp
from jax import jit

class PCA:
    def __init__(self, n_components):
        """
        Initialize PCA with the number of components.

        Parameters:
        - num_components: int, number of principal components to retain.
        """
        self.num_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.components_ = None  # This will store the principal components

    #@jit
    def fit(self, X):
        """
        Fit the PCA model on the data.

        Parameters:
        - X: jnp.ndarray of shape (n_samples, n_features), the input data.

        Returns:
        - self: the fitted PCA instance.
        """
        # Step 1: Compute the mean and center the data
        self.mean = jnp.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Compute the covariance matrix
        cov_matrix = jnp.cov(X_centered, rowvar=False)

        # Step 3: Eigen decomposition of the covariance matrix
        eigenvalues, eigenvectors = jnp.linalg.eigh(cov_matrix)

        # Step 4: Sort eigenvalues and eigenvectors in descending order
        sorted_indices = jnp.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Keep only the top components
        self.explained_variance = eigenvalues[:self.num_components]
        self.components = eigenvectors[:, :self.num_components]
        total_variance = jnp.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance / total_variance
        # Store the components for easy access (similar to sklearn)
        self.components_ = self.components.T
        self.mean_=self.mean
        return self

    #@jit
    def transform(self, X):
        """
        Transform data into the PCA space using the fitted model.

        Parameters:
        - X: jnp.ndarray of shape (n_samples, n_features), the new data.

        Returns:
        - X_transformed: jnp.ndarray of shape (n_samples, num_components).
        """
        if self.mean is None or self.components is None:
            raise ValueError("PCA has not been fitted yet.")

        # Center the data
        X_centered = X - self.mean

        # Project the data into the principal component space
        X_transformed = X_centered @ self.components
        return X_transformed

    #@jit
    def fit_transform(self, X):
        """
        Fit the PCA model and transform the data in one step.

        Parameters:
        - X: jnp.ndarray of shape (n_samples, n_features), the input data.

        Returns:
        - X_transformed: jnp.ndarray of shape (n_samples, num_components).
        """
        self.fit(X)
        return self.transform(X)

    #@jit
    def inverse_transform(self, X_transformed):
        """
        Recover data from the PCA space back to the original feature space.

        Parameters:
        - X_transformed: jnp.ndarray of shape (n_samples, num_components).

        Returns:
        - X_recovered: jnp.ndarray of shape (n_samples, n_features).
        """
        if self.mean is None or self.components is None:
            raise ValueError("PCA has not been fitted yet.")

        # Reconstruct the data from the principal components
        X_recovered = X_transformed @ self.components.T + self.mean
        return X_recovered

    #@jit
    def recover(self, X_transformed):
        """
        Alias for inverse_transform (for compatibility).
        """
        return self.inverse_transform(X_transformed)

    #@jit
    def get_covariance(self):
        """
        Compute the covariance matrix of the original data using the fitted model.

        Returns:
        - cov_matrix: jnp.ndarray of shape (n_features, n_features).
        """
        if self.components is None or self.explained_variance is None:
            raise ValueError("PCA has not been fitted yet.")

        # Covariance is reconstructed using the top components
        cov_matrix = (self.components *
                      self.explained_variance) @ self.components.T
        return cov_matrix



def kpca_transform(X_new, X_train, kernel, components, eigenvalues, gamma=1.0, degree=3, coef0=1):
    """
    Transform new data points using kernel PCA.

    Parameters
    ----------
    X_new : array-like, shape (n_samples, n_features)
        New data points to transform.
    X_train : array-like, shape (n_train_samples, n_features)
        Training data used to compute the kernel PCA.
    kernel : str
        Kernel type ('linear', 'rbf', 'poly').
    components : array-like, shape (n_train_samples, n_components)
        Eigenvectors of the kernel PCA.
    eigenvalues : array-like, shape (n_components,)
        Eigenvalues associated with the eigenvectors.
    gamma : float, optional
        Parameter for RBF and polynomial kernels (default: 1.0).
    degree : int, optional
        Degree for polynomial kernel (default: 3).
    coef0 : float, optional
        Independent term in polynomial kernel (default: 1).

    Returns
    -------
    X_new_transformed : array, shape (n_samples, n_components)
        Transformed data points in the kernel PCA space.
    """
    # Define kernel functions
    def compute_kernel(X1, X2):
        if kernel == 'linear':
            return jnp.dot(X1, X2.T)
        elif kernel == 'rbf':
            sq_dist = jnp.sum(X1**2, axis=1, keepdims=True) - 2 * jnp.dot(X1, X2.T) + jnp.sum(X2**2, axis=1)
            return jnp.exp(-gamma * sq_dist)
        elif kernel == 'poly':
            return (gamma * jnp.dot(X1, X2.T) + coef0) ** degree
        else:
            raise ValueError("Unsupported kernel type. Use 'linear', 'rbf', or 'poly'.")

    # Compute the kernel matrix between X_new and X_train
    K = compute_kernel(X_new, X_train)
    
    # Center the kernel matrix
    n_train = X_train.shape[0]
    n_new = X_new.shape[0]
    one_n_train = jnp.ones((n_train, n_train)) / n_train
    one_n_new = jnp.ones((n_new, n_train)) / n_train
    K_centered = K - jnp.dot(one_n_new, compute_kernel(X_train, X_train)) - jnp.dot(K, one_n_train) + jnp.dot(one_n_new, jnp.dot(compute_kernel(X_train, X_train), one_n_train))

    # Scale the eigenvectors
    non_zeros = eigenvalues > 1e-12  # Handle numerical precision
    scaled_components = components[:, non_zeros] / jnp.sqrt(eigenvalues[non_zeros])

    # Transform the new data
    return jnp.dot(K_centered, scaled_components)
