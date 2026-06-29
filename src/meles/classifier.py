from typing import Any

from sklearn.neighbors import NearestNeighbors
from meles.common import Embeddings


class Classifier:
    model: NearestNeighbors

    def __init__(self, n_neighbors: int = 1, metric: str = "cosine"):
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)

    def fit(self, embeddings: Embeddings):
        self.model.fit(embeddings.cpu().numpy())

    # TODO set correct return type!
    def classify(self, embeddings: Embeddings) -> Any:
        return self.model.kneighbors(embeddings.cpu().numpy(), return_distance=True)
