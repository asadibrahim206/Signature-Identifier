# Architecture diagram

Open **architecture.svg** for scalable viewing or **architecture.png** for a ready-to-insert image. The diagram is editable through **architecture.dot** (Graphviz source) or **architecture.mmd** (Mermaid source).

The left side describes the offline pipeline: three local COCO datasets, image/annotation cleaning, related-image grouping, the 80:10:10 split, YOLOX-Tiny fine-tuning, validation-based checkpoint selection and model export. The held-out test split is used for final evaluation, not model selection.

The right side describes the online Docker service: request validation, orientation correction, RGB-to-BGR conversion, 416-pixel letterboxing, YOLOX inference, score filtering, NMS, coordinate restoration and PNG annotation. Swagger is a client of the same HTTP endpoint. The image returns at the original oriented dimensions.

- Solid arrows: data or request-processing flow.
- Dashed green arrow: serving weights copied into the runtime package at build time.
- Dotted green arrow: startup integrity verification and model loading.
- Dotted blue line: additional health endpoint, not a stage in image prediction.

The offline training workspace is not a runtime dependency. The packaged tensor-only model and metadata are required. The serving process uses one Uvicorn worker, two PyTorch CPU threads and a serial processing lock. Uploads are request-local; there is no image database in this application.

Evidence: [project report](../../docs/README.md), [main.py](../../app/main.py), [detector.py](../../app/detector.py), [Dockerfile](../../Dockerfile), [compose.yaml](../../compose.yaml), and the saved test evaluation. The SHA-256 check validates artifact integrity, not model accuracy. The related-image grouping is heuristic; a successful group-crossing check does not prove all possible document relationships were identified.

Generated and visually reviewed on 9 November 2026. The SVG and PNG represent the same detailed diagram; Mermaid provides a simpler editable overview.
