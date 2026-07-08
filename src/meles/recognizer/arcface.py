from meles.recognizer.recognizer import Recognizer, Frames, Embeddings
from deepface import DeepFace


class ArcFaceRecognizer(Recognizer):
    model_name: str = "ArcFace"
    detector_backend: str
    enforce_detection: bool = False

    def __init__(self, detector_backend: str = "opencv"):
        self.detector_backend = detector_backend

    def metric(self) -> str:
        return "cosine"

    def embed(self, frames: Frames) -> Embeddings:
        embeddings = DeepFace.represent(
            frames,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            enforce_detection=self.enforce_detection,
        )
        # TODO show warning in case multiple faces were detected?
        # Use the fist face that was detected (skip if none were detected)
        return [embedding[0]["embedding"] for embedding in embeddings if embedding.__len__() > 0]
