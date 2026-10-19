import math

import torch

from util import load_data


def inertia(X, centroids):
    eucl_squared = (X - centroids) ** 2
    return eucl_squared.sum()


class KMeans:
    def __init__(self, n_clusters=5, tol=0.01, n_init=3) -> None:
        self.n_clusters = n_clusters
        self.tol = tol
        self.n_init = n_init

    def fit(self, X):
        lowest_inertia = math.inf
        best_centroids = None
        for _ in range(self.n_init):
            indices = torch.randperm(len(X))[: self.n_clusters]
            centers = X[indices]
            skipped = self.n_clusters
            count = 0
            while skipped > 0 and count < 30:
                skipped = self.n_clusters
                dist = torch.cdist(X, centers)
                closest = torch.argmin(dist, dim=1)
                for i in range(self.n_clusters):
                    selected = X[closest == i]
                    avg = selected.mean(dim=0)
                    if abs(centers[i] - avg) >= self.tol and len(
                        selected
                    ):  # only works for 1
                        centers[i] = selected.mean(dim=0)
                    else:
                        skipped -= 1
                centroids = centers[closest]
                inert = inertia(X, centroids)
                if inert < lowest_inertia:
                    best_centroids = centroids
                    lowest_inertia = inert
                count += 1
        return best_centroids


def main():
    X, _, _, _ = load_data()
    kmeans = KMeans(5, 0.01)
    kmeans.fit(X)


if __name__ == "__main__":
    main()
