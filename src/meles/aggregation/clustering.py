from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering

from meles.aggregation.aggregator import Aggregator, Embeddings
from meles.common import Frames
from meles.recognizer.recognizer import Recognizer


class ClusteringAggregator(Aggregator):
    """
    The ClusteringAggregator groups the embeddings into one or more clusters and reduces each cluster to a single
    representative embedding via a column-wise reduction (e.g. mean or median). Unlike the pooling aggregators it does
    not collapse all embeddings into a single embedding: a set of frames showing distinct appearances (or containing
    outliers) yields one embedding per discovered cluster.

    Clustering is performed with agglomerative clustering. The number of clusters is not fixed in advance: instead a
    ``distance_threshold`` controls how far apart embeddings may be before they are split into separate clusters, so
    the aggregator discovers "one or more" clusters from the data. The ``min_size`` drops any cluster that has less than
    that number of embeddings associated with it.
    """

    def __init__(
        self,
        recognizer: Recognizer,
        distance_threshold: Union[float, None] = 1.0,
        n_clusters: Union[int, None] = None,
        min_size: int = 0
    ):
        super().__init__(recognizer)
        self.distance_threshold = distance_threshold
        self.n_clusters = n_clusters
        self.min_size = min_size

    def _reduce(self, cluster: NDArray) -> NDArray:
        """
        Reduce the embeddings belonging to a single cluster to one representative embedding.
        :param cluster: The embeddings of a single cluster with shape (num_cluster_frames, embedding_dim)
        :return: The representative embedding with shape (embedding_dim,)
        """
        raise NotImplementedError

    def _cluster_labels(self, embeddings: NDArray) -> NDArray:
        """
        Assign each embedding to a cluster and return the per-embedding cluster labels.
        :param embeddings: The embeddings to cluster with shape (num_frames, embedding_dim)
        :return: The cluster label of each embedding with shape (num_frames,)
        """
        # A single embedding cannot be clustered; it forms its own trivial cluster.
        if len(embeddings) < 2:
            return np.zeros(len(embeddings), dtype=int)

        metric = "cosine" if self.recognizer.metric() == "cosine" else "euclidean"
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            distance_threshold=self.distance_threshold,
            metric=metric,
            linkage="average",
        )
        return clustering.fit_predict(embeddings)

    def embed(self, frames: Frames) -> Embeddings:
        embeddings = np.asarray(self.recognizer.embed(frames), dtype=float)
        labels = self._cluster_labels(embeddings)

        centroids = [
            self._reduce(subset)
            for label in np.unique(labels)
            if (subset := embeddings[labels == label]).__len__() >= self.min_size
        ]
        return np.stack(centroids).tolist()

    def name(self) -> str:
        return "cluster"


class MeanClusteringAggregator(ClusteringAggregator):
    """
    The MeanClusteringAggregator clusters the embeddings and returns the column-wise mean of each cluster.
    """

    def _reduce(self, cluster: NDArray) -> NDArray:
        return np.mean(cluster, axis=0)

    def name(self) -> str:
        return "cluster_mean"


class MedianClusteringAggregator(ClusteringAggregator):
    """
    The MedianClusteringAggregator clusters the embeddings and returns the column-wise median of each cluster.
    """

    def _reduce(self, cluster: NDArray) -> NDArray:
        return np.median(cluster, axis=0)

    def name(self) -> str:
        return "cluster_median"
