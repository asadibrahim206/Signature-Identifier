from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.middleware import Middleware

from app.detector import InvalidDocument, SignatureDetector
from app.settings import Settings

LOGGER = logging.getLogger(__name__)


class BodyLimitMiddleware:
    """Limit multipart bodies before the parser can spool unbounded uploads to disk."""

    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        length = headers.get(b"content-length")
        if length is not None:
            try:
                too_large = int(length) > self.max_bytes
            except ValueError:
                return await JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)(
                    scope, receive, send
                )
            if too_large:
                return await JSONResponse(
                    {"detail": "Upload exceeds the request size limit."}, status_code=413
                )(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI preserves HTTPException while translating arbitrary
                    # receive errors into a generic multipart parsing error.
                    raise HTTPException(413, "Upload exceeds the request size limit.")
            return message

        await self.app(scope, limited_receive, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.detector = SignatureDetector(settings)
        yield
        del application.state.detector

    application = FastAPI(
        title="Signature Detection API",
        version="1.0.0",
        description=(
            "Upload a document image to locate handwritten signatures using the trained "
            "official YOLOX-Tiny model. The response is the annotated PNG at the document's "
            "original, EXIF-oriented dimensions. This detects signatures; it does not verify identity."
        ),
        lifespan=lifespan,
        middleware=[
            Middleware(BodyLimitMiddleware, max_bytes=settings.max_upload_bytes + 256 * 1024)
        ],
        swagger_ui_parameters={"displayRequestDuration": True},
    )

    @application.get("/", include_in_schema=False)
    async def index():
        return RedirectResponse("/docs")

    @application.get("/health", tags=["Service"], summary="Model readiness and upload limits")
    async def health(request: Request):
        detector = request.app.state.detector
        return {
            "status": "ready",
            "model": "yolox-tiny",
            "classes": ["signature"],
            "device": str(detector.device),
            "input_size": list(detector.input_size),
            "model_sha256": detector.sha256,
            "max_upload_bytes": settings.max_upload_bytes,
            "max_image_pixels": settings.max_image_pixels,
        }

    @application.post(
        "/predict",
        tags=["Detection"],
        summary="Return a document image with signature predictions",
        response_class=Response,
        responses={
            200: {
                "description": "Annotated PNG. Zero detections returns an unmarked document PNG.",
                "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}},
                "headers": {
                    "X-Detection-Count": {"schema": {"type": "integer"}},
                    "X-Inference-Time-Ms": {"schema": {"type": "number"}},
                },
            },
            400: {"description": "Empty upload or malformed request."},
            413: {"description": "Upload or decoded image exceeds the configured limit."},
            415: {"description": "Unsupported or unreadable image."},
            422: {"description": "Missing file, invalid threshold, or multipage image."},
            503: {"description": "Inference resources are unavailable."},
        },
    )
    def predict(
        request: Request,
        file: Annotated[
            UploadFile,
            File(
                description="JPEG, PNG, WEBP, BMP, or single-page TIFF document. Default limit: 20 MiB."
            ),
        ],
        confidence: Annotated[
            float,
            Query(ge=0.01, le=1.0, description="Minimum objectness × class-confidence score."),
        ] = 0.25,
        nms_iou: Annotated[
            float,
            Query(
                ge=0.01,
                le=1.0,
                description="Overlap threshold for suppressing duplicate predictions.",
            ),
        ] = 0.65,
    ):
        payload = file.file.read(settings.max_upload_bytes + 1)
        if len(payload) > settings.max_upload_bytes:
            raise HTTPException(413, "Uploaded image exceeds the configured file size limit.")
        try:
            prediction = request.app.state.detector.predict(payload, confidence, nms_iou)
        except InvalidDocument as error:
            raise HTTPException(error.status_code, str(error)) from error
        except RuntimeError as error:
            LOGGER.exception("Model inference failed")
            raise HTTPException(
                503, "Model inference is unavailable; check the server log."
            ) from error
        return Response(
            content=prediction.png,
            media_type="image/png",
            headers={
                "Content-Disposition": 'inline; filename="prediction.png"',
                "X-Detection-Count": str(prediction.count),
                "X-Inference-Time-Ms": f"{prediction.inference_ms:.2f}",
                "X-Image-Width": str(prediction.width),
                "X-Image-Height": str(prediction.height),
                "Cache-Control": "no-store",
            },
        )

    return application


app = create_app()
