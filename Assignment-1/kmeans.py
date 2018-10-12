import numpy as np


class KMeans:

    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.dim = None

    def initialize_centroids(self):
        self.centroids = np.random.randn(self.n_clusters, self.dim) * 255

    def fit(self, data):
        # Be sure data is give in np.ndarray
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Get vector dimension
        self.dim = data.shape[1]

        n_samples = data.shape[0]

        # Start with random samples
        labels = np.random.multinomial(1, np.ones(self.n_clusters)/self.n_clusters, n_samples)

        # Initialize for stop conditions
        old_centroids = None
        iter_ = 0

        # Train part
        while not self.should_stop(old_centroids, iter_):
            old_centroids = self.centroids
            iter_ += 1

            self.update_centroids(data, labels)
            labels = self.labels(data)

        # Return trained model
        return self

    def should_stop(self, old_centroids, iter_):
        # Check centroids and old centroids values are assigned
        if (self.centroids is None) or (old_centroids is None):
            return False
        return (iter_ > self.max_iter) or np.allclose(old_centroids.ravel(), self.centroids.ravel())

    def labels(self, data):
        # Calculate distance matrix which is n_samples x n_clusters
        dist_mtx = calculate_dists(data, self.centroids)
        return np.eye(self.n_clusters)[dist_mtx.argmin(axis=1)]

    def predict(self, data):
        # Get labels
        labels = self.labels(data)

        # Get cluster ids from 0 to n_clusters - 1
        return labels.argmax(axis=1)

    def update_centroids(self, data, labels):
        # Formula comes from Matrix Factorization approach for k-Means algorithm
        self.centroids = np.linalg.pinv(labels.T.dot(labels)).dot(labels.T).dot(data)


def calculate_dists(X, M):
    # Note: these are not real distances
    # X^2 is not calculated, because for each row
    # same element would be used, and that does not
    # affect the result of np.argmin(distance_matrix, axis=1)
    # function
    return -2 * X.dot(M.T) + (M*M).sum(axis=1)[np.newaxis, :]
