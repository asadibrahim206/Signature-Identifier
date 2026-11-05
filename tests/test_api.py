from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pathspec import GitIgnoreSpec
from PIL import Image

from app.detector import Detection, InvalidDocument, annotate, decode_document, restore_detections
from app.main import create_app
from app.settings import ROOT, Settings


def png(size=(320, 180), mode="RGB", color="white"):
    stream = io.BytesIO()
    Image.new(mode, size, color).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as connection:
        yield connection


def test_swagger_describes_binary_image_upload_and_response(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/", follow_redirects=False).headers["location"] == "/docs"
    spec = client.get("/openapi.json").json()
    operation = spec["paths"]["/predict"]["post"]
    assert "multipart/form-data" in operation["requestBody"]["content"]
    assert "image/png" in operation["responses"]["200"]["content"]
    assert {item["name"] for item in operation["parameters"]} == {"confidence", "nms_iou"}


def test_health_loads_the_packaged_real_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    health = response.json()
    metadata = json.loads((ROOT / "signature_yolox/api_model/metadata.json").read_text())
    assert health["status"] == "ready"
    assert health["device"] == "cpu"
    assert health["model_sha256"] == metadata["weights_sha256"]


def test_predict_returns_full_size_png_with_real_inference(client):
    response = client.post("/predict", files={"file": ("page.png", png(), "image/png")})
    assert response.status_code == 200, response.text[:200]
    assert response.headers["content-type"] == "image/png"
    assert int(response.headers["X-Detection-Count"]) >= 0
    assert float(response.headers["X-Inference-Time-Ms"]) > 0
    with Image.open(io.BytesIO(response.content)) as image:
        image.load()
        assert image.size == (320, 180)
        assert image.mode == "RGB"


@pytest.mark.parametrize(
    "payload,status", [(b"", 400), (b"%PDF-1.7 fake", 415), (b"not an image", 415)]
)
def test_invalid_documents_get_useful_errors(client, payload, status):
    result = client.post("/predict", files={"file": ("bad.bin", payload)})
    assert result.status_code == status
    assert "detail" in result.json()


def test_missing_file_and_out_of_range_thresholds(client):
    assert client.post("/predict").status_code == 422
    for query in ("confidence=0", "confidence=2", "nms_iou=-1"):
        result = client.post(f"/predict?{query}", files={"file": ("x.png", png())})
        assert result.status_code == 422


def test_oversized_content_length_rejected_before_parsing(client):
    result = client.post("/predict", content=b"", headers={"Content-Length": str(22 * 1024 * 1024)})
    assert result.status_code == 413


def test_chunked_body_limit_cannot_be_bypassed():
    with TestClient(create_app(Settings(max_upload_bytes=64))) as small_client:

        def chunks():
            yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="x.png"\r\n\r\n'
            for _ in range(80):
                yield b"x" * 4096
            yield b"\r\n--boundary--\r\n"

        response = small_client.post(
            "/predict",
            content=chunks(),
            headers={"content-type": "multipart/form-data; boundary=boundary"},
        )
        assert response.status_code == 413, response.text


def test_image_dimension_limit_and_transparency():
    with pytest.raises(InvalidDocument) as error:
        decode_document(png((10, 10)), max_pixels=99)
    assert error.value.status_code == 413
    transparent = decode_document(png((10, 10), "RGBA", (0, 0, 0, 0)), max_pixels=100)
    assert transparent.getpixel((0, 0)) == (255, 255, 255)


def test_multiframe_tiff_is_not_silently_truncated(client):
    stream = io.BytesIO()
    Image.new("RGB", (10, 10)).save(
        stream, format="TIFF", save_all=True, append_images=[Image.new("RGB", (10, 10))]
    )
    result = client.post(
        "/predict", files={"file": ("pages.tiff", stream.getvalue(), "image/tiff")}
    )
    assert result.status_code == 422


def test_exif_orientation_is_applied():
    stream = io.BytesIO()
    image = Image.new("RGB", (40, 20), "white")
    exif = image.getexif()
    exif[274] = 6
    image.save(stream, format="JPEG", exif=exif)
    assert decode_document(stream.getvalue(), max_pixels=1000).size == (20, 40)


def test_boxes_use_training_letterbox_scale_and_stay_in_original_bounds():
    # A 1000x500 image scales by 0.416 with TOP-LEFT padding, without a centered-padding offset.
    output = torch.tensor(
        [[41.6, 20.8, 208.0, 104.0, 0.8, 0.5, 0.0], [-20.0, -10.0, 500.0, 500.0, 0.7, 0.5, 0.0]]
    )
    boxes = restore_detections(output, 1000, 500, (416, 416), 100)
    assert np.allclose(boxes[0].box, (100.0, 50.0, 500.0, 250.0), atol=1e-4)
    assert boxes[0].score == pytest.approx(0.4)
    assert boxes[1].box == (0.0, 0.0, 1000.0, 500.0)
    assert restore_detections(None, 1000, 500, (416, 416), 100) == []


def test_annotation_draws_prediction_boxes_and_zero_detection_is_unmarked():
    original = Image.new("RGB", (320, 180), "white")
    marked = Image.open(io.BytesIO(annotate(original, [Detection((20, 40, 180, 100), 0.91)])))
    assert marked.getpixel((20, 60)) == (220, 38, 38)
    unmarked = Image.open(io.BytesIO(annotate(original, [])))
    assert np.array_equal(np.asarray(original), np.asarray(unmarked))
    # Subpixel boxes clipped to the right/bottom edge must remain drawable.
    assert annotate(original, [Detection((319.8, 179.8, 320, 180), 0.8)])


def test_only_serving_artifacts_are_unignored():
    # Verify patterns without invoking any Git command.
    spec = GitIgnoreSpec.from_lines((ROOT / ".gitignore").read_text().splitlines())
    for path in (
        "signature_yolox/api_model/model.pth",
        "signature_yolox/api_model/metadata.json",
        "app/main.py",
        "vendor/yolox/models/yolox.py",
        "uv.lock",
    ):
        assert not spec.match_file(path), path
    for path in (
        "signature_yolox/outputs/signature_yolox_tiny/best_ckpt.pth",
        "signature_yolox/merged_coco/train2017/image.png",
        "Datasets/image.png",
        "signature_yolox/YOLOX-0.3.0/yolox/models/yolox.py",
        ".venv/file",
        "signature_yolox/api_model/accidental-training-output.pth",
    ):
        assert spec.match_file(path), path


def test_runtime_works_without_the_training_workspace(tmp_path):
    for folder in ("app", "vendor", "signature_yolox/api_model"):
        shutil.copytree(
            ROOT / folder, tmp_path / folder, ignore=shutil.ignore_patterns("__pycache__")
        )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.detector import SignatureDetector; from app.settings import Settings; "
            "print(SignatureDetector(Settings()).sha256)",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip()) == 64
