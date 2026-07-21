from deepface import DeepFace
from tqdm.auto import tqdm

from meles.common import Frames, Embeddings
from meles.recognizer.recognizer import Recognizer


class ArcFaceRecognizer(Recognizer):
    model_name: str = "ArcFace"

    def __init__(
        self,
        detector_backend: str = "opencv",
        enforce_detection: bool = False,
        batch_size: int = 64,
    ):
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        self.batch_size = batch_size

    def metric(self) -> str:
        return "cosine"

    def embed(self, frames: Frames) -> Embeddings:
        # Embed frames in batches so DeepFace keeps its batching efficiency
        # while progress stays observable. The tqdm bar advances by one per
        # frame and writes to stderr, which Snakemake captures in the per-job
        # log, giving live feedback on long embedding runs.
        embeddings: Embeddings = []
        with tqdm(
            total=len(frames),
            desc=f"Embedding ({self.model_name})",
            unit="frame",
        ) as progress:
            for start in range(0, len(frames), self.batch_size):
                batch = frames[start : start + self.batch_size]
                # A list input yields List[List[Dict]] — one face list per frame.
                representations = DeepFace.represent(
                    batch,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=self.enforce_detection,
                )
                # noinspection PyTypeChecker
                embeddings.extend(
                    faces[0]["embedding"] if faces else None
                    for faces in representations
                )
                progress.update(len(batch))
        return embeddings
