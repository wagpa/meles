import numpy as np
import torch
from meles.recognizer.recognizer import Recognizer, Frames, Embeddings
from insightface.app import FaceAnalysis
from insightface.utils import face_align


class ArcFaceRecognizer(Recognizer):
    model: FaceAnalysis

    def __init__(self, name: str = "buffalo_l"):
        self.model = FaceAnalysis(name=name)
        self.model.prepare(ctx_id=0)

    def metric(self) -> str:
        return "cosine"

    def embed(self, frames: Frames) -> Embeddings:
        # Prepare the frames to the expected format
        images = frames.cpu().numpy()
        aligned_crops = []
        for img_bgr in images:
            faces = self.model.get(img_bgr)
            if faces:
                crop = face_align.norm_crop(img_bgr, faces[0].kps)
                aligned_crops.append(crop)

        # Embed the prepared frames
        rec = self.model.models["recognition"]
        embeddings = rec.get_feat(np.stack(aligned_crops))
        return torch.from_numpy(embeddings).to(frames.device)
