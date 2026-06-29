from typing import TypeAlias, Any, Sequence, List

from numpy.typing import NDArray

Frames: TypeAlias = Sequence[NDArray[Any]]

Embeddings: TypeAlias = List[List[float]]
