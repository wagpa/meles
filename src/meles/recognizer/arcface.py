import hashlib

import numpy as np
import diskcache as dc
from deepface import DeepFace

from meles.common import Frame, Frames, Embeddings
from meles.recognizer.recognizer import Recognizer

# Unique sentinel so we can distinguish a cache miss from a cached ``None``
# (a frame in which no face was found). ``diskcache`` returns this exact object
# on a miss, so identity comparison is reliable.
_MISS = object()


class ArcFaceRecognizer(Recognizer):
    model_name: str = "ArcFace"

    def __init__(
        self,
        detector_backend: str = "opencv",
        enforce_detection: bool = False,
        cache_dir: str = "arcface",
    ):
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        # Instance-owned cache so different recognizers don't share state.
        self.cache = dc.Cache(cache_dir)

    def metric(self) -> str:
        return "cosine"

    def _key(self, frame: Frame) -> str:
        """
        Build a stable, content-based cache key for a frame.

        ``hash()`` cannot be used: it raises on numpy arrays and is salted per
        process for strings, so it never persists across runs. We hash the
        frame content together with the parameters that affect the embedding
        (model + detection settings), so changing a backend doesn't return
        stale vectors.
        """
        h = hashlib.sha256()
        if isinstance(frame, np.ndarray):
            h.update(str(frame.shape).encode())
            h.update(str(frame.dtype).encode())
            h.update(np.ascontiguousarray(frame).tobytes())
        elif isinstance(frame, str):
            # A path/URL/base64 string; the string itself is the stable identity.
            h.update(frame.encode("utf-8"))
        elif hasattr(frame, "read"):
            # A file-like object; hash the bytes without consuming the stream.
            pos = frame.tell()
            h.update(frame.read())
            frame.seek(pos)
        else:
            raise TypeError(f"Unsupported frame type for caching: {type(frame)!r}")

        h.update(self.model_name.encode())
        h.update(self.detector_backend.encode())
        h.update(str(self.enforce_detection).encode())
        return h.hexdigest()

    def embed(self, frames: Frames) -> Embeddings:
        keys = [self._key(frame) for frame in frames]

        # Look every frame up; missing entries hold the _MISS sentinel.
        results = [self.cache.get(key, default=_MISS) for key in keys]

        # Embed only the frames we don't have yet.
        miss_indices = [i for i, r in enumerate(results) if r is _MISS]
        if miss_indices:
            misses = [frames[i] for i in miss_indices]
            representations = DeepFace.represent(
                misses,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
            )
            # A single-element input returns the flat List[Dict]; normalise it
            # to the batched List[List[Dict]] form so indexing lines up.
            if len(misses) == 1:
                representations = [representations]

            for i, faces in zip(miss_indices, representations):
                embedding = faces[0]["embedding"] if faces else None
                results[i] = embedding
                # Only cache found faces so misses are retried next time.
                if embedding is not None:
                    self.cache.set(keys[i], embedding)

        # Any position still holding the sentinel had no face detected.
        return [None if r is _MISS else r for r in results]
