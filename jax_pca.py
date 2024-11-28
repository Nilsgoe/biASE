import jax.numpy as jnp
from jax import jit

class PCA:
    def __init__(self, num_components):
        """
        Initialize PCA with the number of components.

        Parameters:
        - num_components: int, number of principal components to retain.
        """
        self.num_components = num_components
        self.mean = None
        self.components = None
        self.explained_variance = None

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
