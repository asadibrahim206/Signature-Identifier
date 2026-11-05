from __future__ import annotations

import hashlib
import io
import json
import sys
import threading
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from app.settings import ROOT, Settings

# The vendored implementation uses the upstream absolute `yolox` package imports.
sys.path.insert(0, str(ROOT / "vendor"))
from yolox.data import ValTransform  # noqa: E402
from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead  # noqa: E402
from yolox.utils import postprocess  # noqa: E402


class InvalidDocument(ValueError):
    def __init__(self, detail: str, status_code: int = 415):
        super().__init__(detail)
        self.status_code = status_code


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class Prediction:
    png: bytes
    count: int
    width: int
    height: int
    inference_ms: float


def decode_document(payload: bytes, max_pixels: int) -> Image.Image:
    if not payload:
        raise InvalidDocument("The uploaded file is empty.", 400)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}:
                    raise InvalidDocument("Use a JPEG, PNG, WEBP, BMP, or single-page TIFF image.")
                if image.width * image.height > max_pixels:
                    raise InvalidDocument(f"Image exceeds the {max_pixels:,}-pixel limit.", 413)
                if getattr(image, "n_frames", 1) != 1:
                    raise InvalidDocument(
                        "Upload one document page per request; animated/multipage images are unsupported.",
                        422,
                    )
                image.load()
                oriented = ImageOps.exif_transpose(image)
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    result = Image.new("RGB", rgba.size, "white")
                    result.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    result = oriented.convert("RGB")
                result.info.clear()
                return result
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise InvalidDocument("Image dimensions exceed the decoding safety limit.", 413) from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        if isinstance(error, InvalidDocument):
            raise
        raise InvalidDocument("The file is not a readable supported image.") from error


def restore_detections(
    output: torch.Tensor | None,
    width: int,
    height: int,
    input_size: tuple[int, int],
    limit: int,
) -> list[Detection]:
    if output is None:
        return []
    ratio = min(input_size[0] / height, input_size[1] / width)
    rows = output.detach().cpu().numpy()
    scores = rows[:, 4] * rows[:, 5]
    detections = []
    for index in np.argsort(-scores, kind="stable")[:limit]:
        x1, y1, x2, y2 = rows[index, :4].astype(float) / ratio
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 > x1 and y2 > y1 and np.isfinite([x1, y1, x2, y2, scores[index]]).all():
            detections.append(
                Detection((float(x1), float(y1), float(x2), float(y2)), float(scores[index]))
            )
    return detections


def annotate(image: Image.Image, detections: list[Detection]) -> bytes:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    line_width = max(2, min(6, round(min(result.size) / 300)))
    font = ImageFont.load_default(size=max(12, min(28, round(result.width / 65))))
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        draw.rectangle(
            (
                min(x1, result.width - 1),
                min(y1, result.height - 1),
                min(x2, result.width - 1),
                min(y2, result.height - 1),
            ),
            outline=(220, 38, 38),
            width=line_width,
        )
        label = f"signature {detection.score:.2f}"
        bounds = draw.textbbox((0, 0), label, font=font)
        label_width, label_height = bounds[2] - bounds[0] + 6, bounds[3] - bounds[1] + 6
        left = max(0, min(x1, result.width - label_width))
        top = max(0, y1 - label_height)
        draw.rectangle((left, top, left + label_width, top + label_height), fill=(220, 38, 38))
        draw.text((left + 3 - bounds[0], top + 3 - bounds[1]), label, fill="white", font=font)
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()


class SignatureDetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
        required = {
            "architecture": "yolox-tiny",
            "depth": 0.33,
            "width": 0.375,
            "num_classes": 1,
            "class_names": ["signature"],
            "input_size": [416, 416],
        }
        if any(self.metadata.get(key) != value for key, value in required.items()):
            raise RuntimeError(
                "Serving metadata is incompatible with the trained single-class YOLOX-Tiny."
            )
        with settings.model_path.open("rb") as stream:
            actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_digest != self.metadata.get("weights_sha256"):
            raise RuntimeError(
                "Model checksum mismatch. Restore matching model.pth and metadata.json."
            )
        self.sha256 = actual_digest
        requested = settings.device
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Use cpu or install matched CUDA PyTorch wheels."
            )
        self.device = torch.device(
            "cuda"
            if requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        torch.set_num_threads(settings.torch_threads)
        self.input_size = (416, 416)
        self.transform = ValTransform(legacy=False)
        backbone = YOLOPAFPN(depth=0.33, width=0.375, act="silu")
        head = YOLOXHead(num_classes=1, width=0.375, act="silu")
        self.model = YOLOX(backbone, head)
        # Match the training experiment's BatchNorm epsilon; it is not a state-dict tensor.
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.03
        state = torch.load(settings.model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, *self.input_size, device=self.device))

    def predict(self, payload: bytes, confidence: float, nms_iou: float) -> Prediction:
        # Serial inference bounds memory use and protects YOLOX's internal cached grids.
        with self.lock:
            image = decode_document(payload, self.settings.max_image_pixels)
            bgr = np.asarray(image)[:, :, ::-1].copy()
            inputs, _ = self.transform(bgr, None, self.input_size)
            tensor = torch.from_numpy(inputs).unsqueeze(0).to(self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = postprocess(
                    self.model(tensor), num_classes=1, conf_thre=confidence, nms_thre=nms_iou
                )[0]
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            detections = restore_detections(output, image.width, image.height, self.input_size, 100)
            return Prediction(
                annotate(image, detections), len(detections), image.width, image.height, elapsed
            )
