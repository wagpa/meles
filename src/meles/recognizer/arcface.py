from meles.recognizer.recognizer import Recognizer, Frames, Embeddings
from deepface import DeepFace


class ArcFaceRecognizer(Recognizer):
    model_name: str = "ArcFace"
    detector_backend: str

    def __init__(self, detector_backend: str = "opencv"):
        self.detector_backend = detector_backend

    def metric(self) -> str:
        return "cosine"

    def embed(self, frames: Frames) -> Embeddings:
        embeddings = DeepFace.represent(
            frames,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
        )
        return [embedding["embedding"] for embedding in embeddings]
