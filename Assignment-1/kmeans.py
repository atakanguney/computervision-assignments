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

        # Return centroids
        return self

    def should_stop(self, old_centroids, iter_):
        if (self.centroids is None) or (old_centroids is None):
            return False
        return (iter_ > self.max_iter) or np.allclose(old_centroids.ravel(), self.centroids.ravel())

    def labels(self, data):
        dist_mtx = calculate_dists(data, self.centroids)
        return np.eye(self.n_clusters)[dist_mtx.argmin(axis=1)]

    def predict(self, data):
        labels = self.labels(data)
        return labels.argmax(axis=1)

    def update_centroids(self, data, labels):
        self.centroids = np.linalg.pinv(labels.T.dot(labels)).dot(labels.T).dot(data)


def calculate_dists(X, M):
    return -2 * X.dot(M.T) + (M*M).sum(axis=1)[np.newaxis, :]
