from typing import TypeAlias, Any, Sequence, List, Union, IO

from numpy.typing import NDArray

Frames: TypeAlias = Sequence[Union[str, NDArray[Any], IO[bytes]]]

Embeddings: TypeAlias = List[List[float]]
