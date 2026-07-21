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
    ):
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection

    def metric(self) -> str:
        return "cosine"

    def embed(self, frames: Frames) -> Embeddings:
        # Embed one frame at a time so progress is observable. tqdm writes to
        # stderr, which Snakemake captures in the per-job log, giving live
        # feedback on long embedding runs. `unit="frame"` and the description
        # make the bar self-explanatory in the log output.
        embeddings: Embeddings = []
        for frame in tqdm(
            frames,
            total=len(frames),
            desc=f"Embedding ({self.model_name})",
            unit="frame",
        ):
            faces = DeepFace.represent(
                frame,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
            )
            # noinspection PyTypeChecker
            embeddings.append(faces[0]["embedding"] if faces else None)
        return embeddings
