"""Package local official YOLOX source and export the trusted best checkpoint for serving.

Run once with `uv run python scripts/prepare_runtime.py`. A checkout that includes
vendor/ and signature_yolox/api_model/ does not need the training workspace or this step.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "signature_yolox/YOLOX-0.3.0"
CHECKPOINT = ROOT / "signature_yolox/outputs/signature_yolox_tiny/best_ckpt.pth"
EXPECTED_CHECKPOINT = "9a0bef4d37c1b458ab5ab0b75d3ccb2c01fc7fc33692d1bc6e5399b56c7ad8f3"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    if not CHECKPOINT.is_file() or digest(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError("Expected the exact validated best checkpoint from the completed run.")

    destination = ROOT / "vendor/yolox"
    files = [
        "models/darknet.py",
        "models/losses.py",
        "models/network_blocks.py",
        "models/yolo_head.py",
        "models/yolo_pafpn.py",
        "models/yolox.py",
        "utils/boxes.py",
        "utils/compat.py",
        "data/data_augment.py",
    ]
    hashes = {}
    for relative in files:
        original = SOURCE / "yolox" / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        hashes[relative] = digest(target)
    shutil.copyfile(SOURCE / "LICENSE", destination / "LICENSE")
    # Only package exports are narrowed; the original model, preprocessing and NMS code is unchanged.
    initializers = {
        "__init__.py": '__version__ = "0.3.0"\n',
        "models/__init__.py": (
            "from .yolo_head import YOLOXHead\nfrom .yolo_pafpn import YOLOPAFPN\n"
            "from .yolox import YOLOX\n"
        ),
        "utils/__init__.py": (
            "from .boxes import bboxes_iou, postprocess, xyxy2cxcywh\n"
            "from .compat import meshgrid\n"
        ),
        "data/__init__.py": "from .data_augment import ValTransform\n",
    }
    for relative, content in initializers.items():
        notice = "# Modified for this project: expose only the YOLOX inference imports.\n"
        (destination / relative).write_text(notice + content, encoding="utf-8")
    (ROOT / "vendor/source_manifest.json").write_text(
        json.dumps(
            {
                "upstream": "https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0",
                "tag": "0.3.0",
                "unchanged_source_sha256": hashes,
                "modified_files": list(initializers),
                "modification": "Package initializers expose inference-only imports.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # This input is the user's trusted local training checkpoint. The server never loads it directly.
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = {key: tensor.detach().cpu().contiguous() for key, tensor in checkpoint["model"].items()}
    output = ROOT / "signature_yolox/api_model"
    output.mkdir(parents=True, exist_ok=True)
    weights = output / "model.pth"
    torch.save(state, weights)
    loaded = torch.load(weights, map_location="cpu", weights_only=True)
    if loaded.keys() != state.keys() or not all(torch.equal(state[k], loaded[k]) for k in state):
        raise RuntimeError("Exported tensors differ from the trained checkpoint.")
    metadata = {
        "architecture": "yolox-tiny",
        "source_tag": "0.3.0",
        "depth": 0.33,
        "width": 0.375,
        "num_classes": 1,
        "class_names": ["signature"],
        "input_size": [416, 416],
        "default_confidence": 0.25,
        "default_nms_iou": 0.65,
        "max_detections": 100,
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT,
        "source_checkpoint_epoch": int(checkpoint["start_epoch"]),
        "weights_sha256": digest(weights),
        "weights_bytes": weights.stat().st_size,
        "tensor_count": len(state),
        "format": "PyTorch tensor-only state dictionary",
        "held_out_test_metrics": {
            "map_50": 0.9502923526790285,
            "map_50_95": 0.6550610933055143,
            "precision": 0.9281210592686002,
            "recall": 0.9387755102040817,
            "f1": 0.9334178820545338,
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "weights": str(weights),
                "bytes": weights.stat().st_size,
                "sha256": metadata["weights_sha256"],
                "trained_epoch": metadata["source_checkpoint_epoch"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
