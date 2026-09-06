from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from meles.common import Embeddings


class Classifier:
    model: NearestNeighbors
    labels: Optional[NDArray]

    def __init__(self, n_neighbors: int = 1, metric: str = "cosine"):
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.labels = None

    def fit(self, embeddings: Embeddings, labels: Sequence):
        """
        Fit the nearest-neighbour gallery and remember the label of every embedding.
        :param embeddings: The gallery embeddings with shape (num_embeddings, embedding_dim)
        :param labels: The label (e.g. identity) of every gallery embedding
        """
        self.model.fit(np.asarray(embeddings, dtype=float))
        self.labels = np.asarray(labels)

    def classify(self, embeddings: Embeddings, n_neighbors: Optional[int] = None) -> tuple[NDArray, NDArray]:
        """
        Return the ranked candidate labels and their distances for every embedding.
        :param embeddings: The query embeddings with shape (num_embeddings, embedding_dim)
        :param n_neighbors: The number of neighbours to return (defaults to the fitted value)
        :return: A tuple (labels, distances), each with shape (num_embeddings, n_neighbors), where the
                 neighbours are ordered from the closest gallery neighbour to the farthest.
        """
        if self.labels is None:
            raise RuntimeError("The classifier has to be fitted before it can classify embeddings.")
        distances, indices = self.model.kneighbors(
            np.asarray(embeddings, dtype=float),
            n_neighbors=n_neighbors,
            return_distance=True,
        )
        return self.labels[indices], distances

    def predict(self, embeddings: Embeddings) -> NDArray:
        """
        Predict the label of every embedding using its closest gallery neighbour.
        :param embeddings: The query embeddings with shape (num_embeddings, embedding_dim)
        :return: The predicted labels with shape (num_embeddings,)
        """
        neighbor_labels, _ = self.classify(embeddings, n_neighbors=1)
        return neighbor_labels[:, 0]
