# Official YOLOX inference source

This directory contains the inference-required subset of Megvii's official
[YOLOX 0.3.0](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0), under the
Apache 2.0 license in [yolox/LICENSE](yolox/LICENSE).

The model files, box postprocessing and `ValTransform` are copied unchanged.
Package `__init__.py` files expose only inference imports, avoiding unrelated
training, COCO evaluator, profiler and logging dependencies. Per-file upstream
hashes and the modified-file list are recorded in `source_manifest.json`.

`scripts/prepare_runtime.py` creates this subset from the existing local source
and exports the validated best checkpoint as tensor-only serving weights. It is
an artifact-preparation utility, not a training or model-download script.
