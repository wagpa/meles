from deepface import DeepFace

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
        representations = DeepFace.represent(
            frames,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            enforce_detection=self.enforce_detection,
        )
        # noinspection PyTypeChecker
        return [faces[0]["embedding"] for faces in representations]
