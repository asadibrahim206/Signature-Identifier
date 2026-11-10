# Signature Identifier

Detect handwritten signatures in document images with the trained official YOLOX-Tiny model. The API returns a PNG of the uploaded page with signature boxes and confidence labels.

## Deploy with GitHub Actions

The [CI/CD setup guide](docs/github-actions-deployment.md) explains how GitHub tests and builds the API, publishes the tested image to GHCR, and deploys it to this Windows laptop through a self-hosted runner. Follow its browser setup steps after pushing the new workflow and deployment files. Deployment stays disabled until you configure the repository variables and runner.

## Run with Docker

From this directory, with Docker Desktop running, build with the memory-limited builder used on this laptop. Create the builder only once; **skip the first command if `signature-api-step1` already exists** (it does on this laptop).

```powershell
docker buildx create --name signature-api-step1 --driver docker-container --driver-opt memory=256m --driver-opt memory-swap=2g --driver-opt cpuset-cpus=0-1
docker buildx build --builder signature-api-step1 --load --provenance=false --tag signature-detection-api:local .
if ($LASTEXITCODE -ne 0) { throw "Docker image build failed." }
docker buildx stop signature-api-step1
docker compose up --no-build -d
docker compose ps
```

The image is already built on this laptop. For later starts, just run `docker compose up --no-build -d`. Rebuild after API, dependency, or model changes. See [Docker instructions](docs/README.md#153-start-with-docker-desktop) for details and a CLI-path fallback.

Open **[Swagger UI](http://localhost:8000/docs)**. Expand `POST /predict`, select **Try it out**, upload a document image in `file`, and select **Execute**. The response is the annotated image.

```powershell
curl.exe --fail --request POST "http://localhost:8000/predict?confidence=0.25&nms_iou=0.65" --form "file=@C:/path/to/document.jpg" --output prediction.png
```

The trained model is included in `signature_yolox/api_model/`; no training, dataset download, or full training workspace is needed. The Docker image uses CPU inference.

## Run locally with uv

```powershell
uv sync --frozen
uv run --frozen uvicorn app.main:app --host 127.0.0.1 --port 8000
```

This creates a separate Python 3.12 environment with locked CPU dependencies. Use [Swagger UI](http://localhost:8000/docs) to test it. Stop the Docker service first if it already occupies port 8000: `docker compose down`.

## Verify

```powershell
uv run --frozen pytest -q
uv run --frozen python scripts/smoke_test.py --image "C:/path/to/document.jpg" --output .artifacts/prediction.png
```

See **[the complete project and API documentation](docs/README.md#15-document-image-api-and-docker)** for testing steps, response headers, settings, model packaging, and the training/evaluation report. The original notebook is [Signature_Detection_YOLOX_Tiny.ipynb](Signature_Detection_YOLOX_Tiny.ipynb).

Live container checks on 5 November 2026 returned **nine detections** on a held-out document and **zero** on a blank page. Swagger HTML, OpenAPI, both UI assets, and upload error responses passed HTTP checks. The annotated PNG was visually inspected. See [verification results and limits](docs/README.md#1510-completed-container-checks-5-november-2026).
