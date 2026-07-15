from typing import TypeAlias, Any, Sequence, List, Union, IO

from numpy.typing import NDArray

Frame: TypeAlias = Union[str, NDArray[Any], IO[bytes]]

Frames: TypeAlias = Sequence[Frame]

Embeddings: TypeAlias = List[Union[List[float], None]]
