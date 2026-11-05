from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def positive_integer(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True)
class Settings:
    model_path: Path = ROOT / "signature_yolox/api_model/model.pth"
    metadata_path: Path = ROOT / "signature_yolox/api_model/metadata.json"
    device: str = "cpu"
    torch_threads: int = 2
    max_upload_bytes: int = 20 * 1024 * 1024
    max_image_pixels: int = 40_000_000

    @classmethod
    def from_environment(cls) -> Settings:
        model = Path(os.getenv("SIGNATURE_MODEL_PATH", str(cls.model_path))).resolve()
        metadata = Path(
            os.getenv("SIGNATURE_MODEL_METADATA", str(model.with_name("metadata.json")))
        )
        device = os.getenv("SIGNATURE_DEVICE", "cpu").lower()
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError("SIGNATURE_DEVICE must be cpu, cuda, or auto.")
        return cls(
            model_path=model,
            metadata_path=metadata.resolve(),
            device=device,
            torch_threads=positive_integer("SIGNATURE_TORCH_THREADS", 2),
            max_upload_bytes=positive_integer("SIGNATURE_MAX_UPLOAD_MB", 20) * 1024 * 1024,
            max_image_pixels=positive_integer("SIGNATURE_MAX_IMAGE_PIXELS", 40_000_000),
        )
