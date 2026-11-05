"""Exercise a running API and save its annotated PNG without any training files.

uv run python scripts/smoke_test.py --image C:/path/to/document.jpg
If --image is omitted, a generated blank page exercises the request/response path.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import httpx
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".artifacts/prediction.png"))
    args = parser.parse_args()

    if args.image:
        payload, filename = args.image.read_bytes(), args.image.name
    else:
        buffer = io.BytesIO()
        Image.new("RGB", (800, 1000), "white").save(buffer, format="PNG")
        payload, filename = buffer.getvalue(), "blank-page.png"

    with httpx.Client(base_url=args.url.rstrip("/"), timeout=120, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        docs = client.get("/docs")
        docs.raise_for_status()
        assert "SwaggerUIBundle" in docs.text, "Swagger UI was not served"
        schema = client.get("/openapi.json")
        schema.raise_for_status()
        assert (
            "image/png" in schema.json()["paths"]["/predict"]["post"]["responses"]["200"]["content"]
        )

        response = client.post("/predict", files={"file": (filename, payload)})
        if response.is_error:
            raise RuntimeError(f"Prediction failed: HTTP {response.status_code}: {response.text}")
        assert response.headers["content-type"] == "image/png"
        with Image.open(io.BytesIO(response.content)) as image:
            image.load()
            dimensions = image.size
            assert image.format == "PNG"

        invalid = client.post("/predict", files={"file": ("invalid.txt", b"not an image")})
        assert invalid.status_code == 415, invalid.text
        empty = client.post("/predict", files={"file": ("empty.png", b"")})
        assert empty.status_code == 400, empty.text

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(response.content)
    print(
        json.dumps(
            {
                "url": args.url,
                "health": health.json(),
                "input": filename,
                "output": str(args.output.resolve()),
                "output_dimensions": dimensions,
                "detections": int(response.headers["X-Detection-Count"]),
                "inference_ms": float(response.headers["X-Inference-Time-Ms"]),
                "swagger": "passed",
                "invalid_upload": invalid.status_code,
                "empty_upload": empty.status_code,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
