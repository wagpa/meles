from abc import abstractmethod
from typing import TypeAlias

# The input type used by the aggregator. It is a list of embeddings.
AggregationInput: TypeAlias = torch.Tensor

# The output type used by the aggregator. It is a single embedding or alist.
AggregationOutput: TypeAlias = torch.Tensor


# The abstract class for aggregators. It aggregates a list of embeddings into a single embedding. In general, only the
# NoAggregator class should return a list of embeddings (baseline).
class Aggregator:
    @abstractmethod
    def __call__(self, embeddings: AggregationInput) -> AggregationOutput:
        # embeddings: shape (num_frames, dim) -> returns (dim,) or (num_frames, dim)
        pass


# The NoAggregator class does not aggregate the embeddings. It simply returns the embeddings as is. This will be used as
# the experiment baseline.
class NoAggregator(Aggregator):
    def __call__(self, embeddings: AggregationInput) -> AggregationOutput:
        return embeddings
