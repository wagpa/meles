"""
The frame-level recognition models the experiment compares.

Every preprocessing constant here was determined by measurement rather than from
documentation, because none of the three models documents its own completely:

- SFace: verified against ``cv2.FaceRecognizerSF.feature``, the reference implementation for
  this exact file. Raw 0-255 RGB reproduces it to a cosine of 1.000000; the graph applies
  ``(x - 127.5) / 128`` itself, in its leading Sub/Mul nodes. Note DeepFace fed this model
  BGR, so the embeddings here are not merely different weights -- they are less wrong.
- ArcFace: the preprocessing InsightFace's own ``ArcFaceONNX`` applies to this file.
- GhostFaceNet: chosen empirically. RGB with ``(x - 127.5) / 128`` separated identities best
  (mean same-identity cosine minus mean different-identity cosine of 0.760 over 72 frames of
  six identities, against 0.158 for the plain 0-1 scaling), and it agrees with the
  quantisation range the upstream model card states.
"""

from meles import weights
from meles.recognizer.frame import EmbedderSpec

# The OpenCV model zoo stores '*.onnx' with git-lfs, so 'raw.githubusercontent.com' would
# serve a pointer file rather than the model -- hence the 'media' host.
_OPENCV_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"

SFACE = EmbedderSpec(
    name="SFace",
    weights=weights.WeightSpec(
        file_name="face_recognition_sface_2021dec.onnx",
        url=f"{_OPENCV_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
    output_dim=128,
    # The graph normalises internally, so it takes raw pixels.
    mean=0.0,
    std=1.0,
    swap_rb=True,
    layout="NCHW",
)

# The recognition model out of InsightFace's buffalo_l pack: a ResNet50 trained on WebFace600K.
# Stronger than the ResNet34 DeepFace shipped as "ArcFace". Its weights are published for
# non-commercial research use only, which this project is.
ARCFACE = EmbedderSpec(
    name="ArcFace",
    weights=weights.WeightSpec(
        file_name="w600k_r50.onnx",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
        # The archive members are flat, with no 'buffalo_l/' prefix.
        archive_member="w600k_r50.onnx",
    ),
    output_dim=512,
    mean=127.5,
    std=127.5,
    swap_rb=True,
    layout="NCHW",
)

# GhostFaceNet has no first-party export that avoids TensorFlow -- every release asset in
# HamadYA/GhostFaceNets is a Keras '.h5'. This is a third-party ONNX conversion which reports
# 99.683% on LFW, and that figure matches none of the published upstream checkpoints, so treat
# it as "a GhostFaceNetV1-class model" rather than as a specific citable checkpoint.
GHOSTFACENET = EmbedderSpec(
    name="GhostFaceNet",
    weights=weights.WeightSpec(
        file_name="ghostface_fp32.onnx",
        url=("https://raw.githubusercontent.com/andestech/ModelZoo/master/GhostFaceNet/Model/ghostface_fp32.onnx"),
        sha256="e8c60ea19ec09ae0b1d30460426fac783c290b4e76c5c7e582e159c8906480ce",
    ),
    output_dim=512,
    mean=127.5,
    std=128.0,
    swap_rb=True,
    # Exported from Keras, so it takes NHWC and transposes internally.
    layout="NHWC",
)

# The models keyed by the name used in 'workflow/config.yaml'.
SPECS = {
    "sface": SFACE,
    "arcface": ARCFACE,
    "ghostfacenet": GHOSTFACENET,
}


def prefetch() -> None:
    """
    Download every model, including the detector, so that a pipeline run does not have to.

    Useful before a long run, and before running the tests offline.
    """
    from meles.detection.detector import YUNET

    weights.resolve(YUNET)
    for spec in SPECS.values():
        weights.resolve(spec.weights)
