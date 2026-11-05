# Signature Detection Using YOLOX-Tiny

Project documentation and completed-run report, prepared **4 November 2026**.

The executable workflow is [Signature_Detection_YOLOX_Tiny.ipynb](../Signature_Detection_YOLOX_Tiny.ipynb). This document describes its implementation and the user's completed local training and evaluation. Measurements come from saved notebook outputs, dataset manifests, audit records, training logs, and evaluation files. Preparing this document did not rerun preprocessing, training, or inference.

## 1. Project purpose and completed result

The system locates handwritten signatures within document images. Given an image, it predicts zero or more rectangular bounding boxes, the class `signature`, and a confidence score for each detection. It supports pages containing multiple signatures and pages with no annotated signatures.

This is **signature detection**. It does not identify the signer, compare signatures against an enrolled reference, determine authenticity, detect forgery, or transcribe handwriting. PDF-to-image conversion, a deployment application, and production monitoring are outside the implemented notebook workflow.

Three existing Roboflow COCO exports were inspected, cleaned, grouped to reduce leakage, and merged into a dataset containing **3,970 images and 7,859 signature boxes**. Official YOLOX-Tiny was initialized from pretrained COCO weights and trained locally for **30 total epochs**, including a resumed run after seven completed epochs.

The validation-selected best checkpoint produced the following results on the held-out test split:

| Measurement | Test result |
|---|---:|
| Images | 397 |
| Annotated signatures | 784 |
| Precision at confidence 0.25 and matching IoU 0.5 | **92.81%** |
| Recall at the same thresholds | **93.88%** |
| F1-score at the same thresholds | **93.34%** |
| COCO mAP@0.5 | **95.03%** |
| COCO mAP@0.5:0.95 | **65.51%** |
| True-positive detections | 736 |
| False-positive detections | 57 |
| Missed ground-truth signatures | 48 |

These are object-detection measurements on this dataset split, not a general-purpose classification accuracy or a guarantee for unseen document collections. The authoritative full-precision results are in [metrics.json](../signature_yolox/reports/evaluation_test/metrics.json).

## 2. Project structure

```text
Signature Identifier/
├── Signature_Detection_YOLOX_Tiny.ipynb
├── Datasets/
│   ├── signature-yolo.v6i.coco/
│   ├── Signature Detection.v3i.coco/
│   └── signatures-xc8up-ytnch-qzw1d-pmbl.v1i.coco/
├── docs/
│   └── README.md
└── signature_yolox/
    ├── YOLOX-0.3.0/                 # Official source, obtained through a ZIP archive
    ├── clean_yolo/
    │   ├── images/                 # Lossless RGB PNG exports
    │   ├── labels/                 # Normalized single-class YOLO TXT
    │   └── class_names.json
    ├── merged_coco/
    │   ├── annotations/
    │   │   ├── instances_train2017.json
    │   │   ├── instances_val2017.json
    │   │   └── instances_test2017.json
    │   ├── train2017/
    │   ├── val2017/
    │   ├── test2017/
    │   └── yolo_labels/{train2017,val2017,test2017}/
    ├── experiments/
    │   ├── signature_yolox_tiny.py
    │   └── signature_train_config.json
    ├── weights/
    ├── outputs/signature_yolox_tiny/
    │   ├── best_ckpt.pth
    │   ├── latest_ckpt.pth
    │   ├── last_epoch_ckpt.pth
    │   ├── last_mosaic_epoch_ckpt.pth
    │   ├── train_log.txt
    │   └── tensorboard/
    └── reports/
        ├── audit.csv
        ├── merged_manifest.json
        ├── split_manifest.csv
        ├── validation.json
        ├── batch_benchmark.json
        ├── training_run.json
        └── evaluation_test/
```

The `train2017`, `val2017`, and `test2017` directory names are conventions required by the configured YOLOX COCO adapter. They do not indicate that this signature dataset is the original COCO 2017 benchmark.

Figures below link to existing report images. Keep the project directory structure intact when viewing this document.

## 3. Datasets and provenance

### 3.1 Sources used

All three datasets were read from the existing `Datasets/` folder. The current notebook contains no dataset-download workflow, requires no Roboflow SDK or API key, and does not contact Roboflow during dataset preparation.

| Source key | Roboflow project | Version | Local folder | Images | Boxes | Declared negative images |
|---|---|---:|---|---:|---:|---:|
| `bondaruk` | [lukasz-bondaruk/signature-yolo](https://universe.roboflow.com/lukasz-bondaruk/signature-yolo) | 6 | `signature-yolo.v6i.coco` | 810 | 4,148 | 56 |
| `tech` | [tech-ysdkk/signature-detection-hlx8j](https://universe.roboflow.com/tech-ysdkk/signature-detection-hlx8j) | 3 | `Signature Detection.v3i.coco` | 2,819 | 3,252 | 260 |
| `rf100` | [rf100-vl/signatures-xc8up-ytnch-qzw1d-pmbl](https://universe.roboflow.com/rf100-vl/signatures-xc8up-ytnch-qzw1d-pmbl) | 1 | `signatures-xc8up-ytnch-qzw1d-pmbl.v1i.coco` | 341 | 459 | 0 |
| **Total** | | | | **3,970** | **7,859** | **316** |

A declared negative is an image registered in COCO JSON with no annotations. Retaining these images assumes that the source labeling is exhaustive. An image missing from COCO registration is treated differently and is not automatically accepted as a negative.

Source paths and version metadata are recorded in [local_datasets.json](../signature_yolox/reports/local_datasets.json). The RF100 export README uses the older workspace spelling `rf-100-vl`; the configured project provenance uses the supplied `rf100-vl` URL. Local file loading does not depend on this workspace spelling.

### 3.2 Original export splits

| Source | Original split | Images | Signature boxes |
|---|---|---:|---:|
| bondaruk | train | 750 | 4,088 |
| bondaruk | valid | 60 | 60 |
| tech | train | 1,980 | 2,330 |
| tech | valid | 420 | 455 |
| tech | test | 419 | 467 |
| rf100 | train | 238 | 323 |
| rf100 | valid | 68 | 89 |
| rf100 | test | 35 | 47 |

These original partitions were recorded as provenance and then replaced by a new grouped split. Consequently, this report does not measure performance on the original Roboflow benchmark partitions.

### 3.3 Actual annotation structures and class mappings

Every inspected source used split-level `_annotations.coco.json` files with COCO `images`, `annotations`, and `categories` arrays. Bounding boxes used absolute-pixel `[x, y, width, height]` coordinates.

| Source | Category metadata observed | Category IDs actually used by annotations |
|---|---|---|
| bondaruk | IDs 0 and 1 named `signatures` | 1 |
| tech | IDs 0 and 1 named `signature` | 1 |
| rf100 | ID 0 named after the project; ID 1 named `signature` | 1 |

The importer checks the schema and used category IDs at runtime instead of inferring the annotation format from the folder name. It normalizes case and whitespace and maps only `signature` and `signatures` to the output class. Unused parent categories are reported without becoming detection classes. An unknown used category stops processing for review.

Final class conventions are:

- Human-readable class name: `signature`.
- YOLO TXT/model class index: `0`.
- COCO category ID: `1`, mapped internally by YOLOX to class index `0`.

The recorded per-file schema evidence is in [schemas.json](../signature_yolox/reports/schemas.json).

### 3.4 Image characteristics and pre-existing processing

All 3,970 referenced input images decoded as JPEGs.

| Source | Observed dimensions |
|---|---|
| bondaruk | All images 1,190 × 1,683 pixels |
| tech | All images 640 × 640 pixels |
| rf100 | 218 distinct width/height pairs; widths 184–3,648 and heights 131–4,200 pixels |

The RF100 width and height ranges are independent extrema, not the dimensions of one particular image.

The tech export's `README.roboflow.txt` describes auto-orientation, stretch resizing to 640 × 640, and offline augmentation: quarter-turn rotations, additional rotation up to ±10°, horizontal/vertical shear, brightness/exposure changes, blur, and salt-and-pepper noise. Its description states that five versions were generated per source image; this is source-export metadata, not a newly calculated count of independent originals. The other two export READMEs report no applied image augmentation. Existing stretch distortion and compression cannot be undone by this pipeline.

The export README license fields are CC BY 4.0 for bondaruk, MIT for rf100, and `undefined` for tech. This project does not assign a license to the tech data; retain the source READMEs and establish the applicable permissions before redistribution.

![Raw dataset statistics](../signature_yolox/reports/raw_statistics.png)

## 4. Inspection, cleaning, and preprocessing

### 4.1 Inspection and failure handling

The notebook implements checks for:

1. COCO schema structure; missing, non-integer, or duplicate IDs; dangling annotation references; ambiguous or unsafe image paths; and unknown used categories.
2. Missing image files, files not registered in COCO, and split image files without a corresponding source annotation registration.
3. Full Pillow decoding/verification and independent OpenCV decoding, including decoded dimensions, image format, multiframe ambiguity, and EXIF orientation.
4. COCO dimension metadata that disagrees with decoded pixels.
5. Non-finite coordinates, malformed bounding boxes, nonpositive box sizes, out-of-bounds boxes, and crowd annotations.
6. Exact byte duplicates, exact decoded RGB-pixel duplicates, and candidate perceptual similarity.

Structural problems stop preparation. Image/annotation problems that can be isolated are recorded in a quarantine manifest. Raw source files are never deleted or edited. The workflow therefore retains the evidence needed to correct a problem rather than silently removing useful data.

### 4.2 Cleaning rules

| Rule | Implemented behavior |
|---|---|
| Class normalization | Convert accepted source names to the single `signature` class. |
| Internal box format | Convert COCO pixel `xywh` into floating-point pixel `xyxy`. |
| Minor boundary overshoot | Clip a box only when no coordinate correction exceeds 2 pixels; log before/after values. |
| Invalid box or crowd annotation | Quarantine the entire image, retaining valid proposed boxes for review; do not leave the invalid signature as an unlabeled training object. |
| Stored annotation area | Recompute area from the cleaned box dimensions. |
| Repeated identical boxes | Remove exact repeated box entries with an audit record. |
| Negative images | Retain COCO-declared negatives because `KEEP_DECLARED_NEGATIVES=True`. |
| Tiny boxes | Preserve source annotations; report projected sides at or below one pixel at the configured base input size. |
| Dimensions/orientation mismatch | Require review rather than guessing how to transform the boxes. |

### 4.3 What cleaning actually changed in this run

| Audit/result | Recorded outcome |
|---|---:|
| Category-mapping inspection records | 8 |
| Bounding-box area recomputations | 1,831 |
| Minor boundary-overshoot repairs | 4 |
| Quarantined images/events | 0 |
| Structural errors | 0 |
| Exact duplicate image files removed | 0 |
| Exact RGB-pixel duplicates removed | 0 |
| Conflicting duplicate annotation groups | 0 |
| Images retained | 3,970 |
| Boxes retained | 7,859 |
| Projected boxes with a side ≤1 pixel at the base 416 input size | 0 |

The repair counts refer to annotation operations, not necessarily distinct images. Recomputing stored `area` does not mean 1,831 box locations were changed. Only four minor boundary repairs were logged. The maximum number of original boxes in one image was **17**.

The audit contains no repeated-box removal events or quarantines. The pipeline supported these cases, but this particular export did not trigger them. See [audit.csv](../signature_yolox/reports/audit.csv), [quarantined_records.json](../signature_yolox/reports/quarantined_records.json), and [duplicate_conflicts.json](../signature_yolox/reports/duplicate_conflicts.json).

### 4.4 Exact duplicate handling

SHA-256 hashes are calculated for both file bytes and decoded RGB pixels plus image dimensions. Pixel hashing can detect identical decoded content even when file metadata or container bytes differ.

If identical pixels have matching box sets, one representative is retained and all member provenance is attached to it. If box sets conflict, preparation stops until an authoritative annotation is chosen explicitly; the system does not automatically union conflicting labels. No such duplicates or conflicts were found in this run, so all input images were retained.

### 4.5 Export formats and resizing

Clean images are written as RGB PNG files without a new lossy encoding step. This preserves the decoded source pixels but cannot recover information previously lost in JPEG compression. Unique image identifiers are derived from the source key and relative path to prevent filename collisions.

Each image also gets a YOLO TXT file with one line per object:

```text
0 center_x center_y width height
```

Coordinates in TXT are normalized by that image's actual width and height. For a pixel box `(x1, y1, x2, y2)` in an image of width `W` and height `H`:

```text
center_x = (x1 + x2) / (2 × W)
center_y = (y1 + y2) / (2 × H)
width    = (x2 - x1) / W
height   = (y2 - y1) / H
```

An accepted negative has an empty TXT file. Final COCO JSON is regenerated with unique IDs, one category, cleaned absolute-pixel boxes, recomputed areas, and `iscrowd=0`.

**The official trainer reads COCO JSON.** The normalized YOLO TXT files are an additional validated export, not the format consumed by this YOLOX training configuration.

Native dimensions are retained in the prepared dataset. At loading time YOLOX scales images with aspect ratio preserved and pads the input canvas. Its default evaluation transform uses BGR pixel values, a top-left image placement, and padding value 114; preprocessing is taken directly from the official implementation rather than replaced by an unrelated YOLO preprocessing convention.

## 5. Related-image grouping and leakage control

Exact deduplication is insufficient when multiple exports contain resized, rotated, or otherwise augmented versions of the same document. These related images must remain in one split, even when their file and pixel hashes differ.

The implemented grouping combines:

1. **Filename/origin families:** remove Roboflow `.rf.<hash>` suffixes and encoded image extensions. Related PDF page names are grouped by document prefix. Family matching is also applied across datasets.
2. **Optional manual origin groups:** a CSV can specify shared origins across source paths when naming conventions are insufficient. No manual grouping file was configured for this run.
3. **Perceptual candidates:** a BK-tree searches 64-bit pHashes for four rotations, within Hamming distance **6**.
4. **Content verification:** a pHash candidate is accepted only after ORB descriptor matching, RANSAC geometric alignment, distributed inlier coverage, sufficient page overlap, and aligned foreground-ink agreement.
5. **Connected components:** accepted relationships are joined transitively before split assignment. Near duplicates are retained with their own images and annotations; grouping does not delete them.

The initial hash-only approach had joined 2,643 images into one component, including unrelated documents. The final implementation uses pHash only to propose candidates and applies content verification before accepting those links.

| Verification setting | Value |
|---|---:|
| Minimum geometric inliers | 20 |
| Minimum inlier fraction | 0.55 |
| Minimum inlier spatial coverage | 0.10 |
| Minimum aligned page overlap | 0.70 |
| Minimum foreground-ink F1 | 0.85 |

The verifier uses 512 × 512 thumbnails, ignores an outer margin, and allows a one-pixel foreground matching tolerance. These are verification-only operations; they do not modify training images or labels.

| Grouping result | Value |
|---|---:|
| Perceptual candidate pairs | 8,219 |
| Verified perceptual pairs | 1,037 |
| Candidate pairs not accepted | 7,182 |
| Recorded accepted relationship rows, including filename families | 3,189 |
| Final image groups | 1,751 |
| Largest group | 60 images, approximately 1.51% of the dataset |
| Giant-group guard | 25% of images |

Relationship rows can include redundant links; they are not counts of removed images or guaranteed independent original documents. The final groups passed the size guard.

Evidence is available in [grouping_diagnostics.json](../signature_yolox/reports/grouping_diagnostics.json), [near_duplicate_verification.csv](../signature_yolox/reports/near_duplicate_verification.csv), [similarity_edges.csv](../signature_yolox/reports/similarity_edges.csv), and [family_members.csv](../signature_yolox/reports/family_members.csv).

This procedure prevents the **detected** related-image groups from crossing splits. It is still heuristic: heavily edited or cropped duplicates may evade matching, and similar form templates or ambiguous filenames may be grouped conservatively. Passing these checks does not prove that every semantic overlap has been identified.

## 6. Final merged dataset and validation

A seeded greedy assignment balances image totals and source membership while keeping entire groups together. The seed is **42**, and the requested image ratio is **80:10:10**.

| Split | Images | Share of images | Signature boxes | Groups | Negative images |
|---|---:|---:|---:|---:|---:|
| Training | 3,176 | 80% | 6,188 | 1,396 | 254 |
| Validation | 397 | 10% | 887 | 177 | 22 |
| Test | 397 | 10% | 784 | 178 | 40 |
| **Total** | **3,970** | **100%** | **7,859** | **1,751** | **316** |

The exact image ratio was achieved in this run. Box counts and group counts do not need to follow that ratio. The manifest is the authority for membership; directory globbing is not used to include arbitrary image files in training.

Before training, the full validator checked every output image and corresponding label, decoded pixel identity, dimensions, COCO references, single-class mapping, finite in-bounds positive-area boxes, normalized TXT bounds, and COCO/TXT coordinate round trips. It also checked that detected group relationships stayed within one split and that each split contained positives. The saved result was `ready=true`, with **zero stale unlisted files** and **zero quarantine events**.

For the optimized resumed run, the existing full validation was reused after checking annotation SHA-256 hashes, prepared manifest identity where available, image/label file presence, and counts. This cached path avoids decoding every image again. It assumes prepared file contents have not been manually edited since full validation; changed data require full validation again.

See [validation.json](../signature_yolox/reports/validation.json), the validation snapshot inside [training_run.json](../signature_yolox/reports/training_run.json), and [split_manifest.csv](../signature_yolox/reports/split_manifest.csv).

![Final image and annotation distributions](../signature_yolox/reports/merged_distributions.png)

Source sample visualizations are available for [bondaruk](../signature_yolox/reports/source_bondaruk.png), [tech](../signature_yolox/reports/source_tech.png), and [rf100](../signature_yolox/reports/source_rf100.png). Final annotated examples are saved for [training](../signature_yolox/reports/merged_train.png), [validation](../signature_yolox/reports/merged_val.png), and [test](../signature_yolox/reports/merged_test.png).

## 7. Model design and transfer learning

### 7.1 Architecture

The model is the official **YOLOX-Tiny**, from source tag **0.3.0**. It uses depth multiplier **0.33**, width multiplier **0.375**, and a one-class detection head. It has not been substituted with another YOLO architecture.

Its main components are a CSPDarknet backbone for image features, a path-aggregation feature pyramid for combining multiple feature scales, and decoupled detection branches for box regression, objectness, and classification. It is an anchor-free detector, with the official dynamic assignment/matching procedure used during training.

The training log reports approximately **5.03 million parameters** and **6.44 GFLOPs** at the configured input size. These are the official model-summary estimates, not measured execution latency.

### 7.2 Initialization

The notebook obtains official source code through a tagged ZIP archive, without Git commands, and uses the official COCO-pretrained Tiny checkpoint:

[YOLOX-Tiny pretrained weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth).

The pretrained model originally predicts COCO classes. For this task, **456 compatible tensors were loaded** and **six class-prediction weight/bias tensors** across the three detection levels were initialized for the new single-class output. Shape checks ensure that unexpected architecture mismatches stop the pipeline.

The transfer cell prepares an initial one-class checkpoint. During the optimized continuation, the actual trainer loads the previously trained `latest_ckpt.pth` and its optimizer state; it does not restart from that freshly prepared initialization.

### 7.3 Official implementation and compatibility changes

The generated experiment inherits the official Tiny experiment and uses its model, dataloaders, augmentation classes, optimizer, schedule, and training loop. Two source compatibility edits are recorded:

- Use standard `pycocotools` evaluation rather than building the optional fast COCO extension.
- Make trusted checkpoint loading explicit with `weights_only=False` for modern PyTorch.

A small `SignatureTrainer` subclass applies the revised learning rate before the first resumed update and enables the configured cuDNN performance mode. It retains the official training loop and checkpoint behavior. Checkpoints loaded through this full serialization path must be trusted.

Provenance and hashes are recorded in [source_manifest.json](../signature_yolox/reports/source_manifest.json) and [transfer_learning.json](../signature_yolox/reports/transfer_learning.json).

## 8. Local environment and runtime optimization

The run used the user's **Dell G15 5515 with an RTX 3060 Laptop GPU and 6 GB VRAM**, under Windows. The GPU name is also recorded by the notebook; laptop model and VRAM capacity were supplied by the user.

| Component | Recorded version |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.14.1+cu126 |
| torchvision | 0.29.1+cu126 |
| CUDA reported by PyTorch | 12.6 |
| NumPy | 2.2.6 |
| Pillow | 12.3.0 |
| OpenCV headless | 4.14.0.94 |
| pycocotools | 2.0.11 |
| pandas | 2.3.3 |
| ImageHash | 4.3.2 |

These versions describe the saved environment, not a claim of compatibility with every future installation. Full dependency evidence is in [environment.json](../signature_yolox/reports/environment.json) and [requirements-runtime.txt](../signature_yolox/reports/requirements-runtime.txt).

The initial configuration targeted 100 epochs. To shorten local runtime, the target was reduced to 30 total epochs, validation frequency was reduced initially, per-epoch history saving was disabled, and the already validated dataset was reused. Input resolution and mixed precision were retained.

A temporary-model benchmark compared two warmup and six timed forward/backward passes for batches 8 and 16. It performed no optimizer updates and changed no checkpoint weights.

| Batch | Status | Images/second | Peak reserved GPU memory |
|---|---|---:|---:|
| 8 | Passed | 39.20 | 592 MiB |
| 16 | Passed | 35.97 | 1,154 MiB |

**Batch 8 was selected because it was faster in this benchmark.** The larger batch fit the memory checks but did not improve measured throughput. These are short compute benchmarks, not end-to-end training or deployment throughput figures. See [batch_benchmark.json](../signature_yolox/reports/batch_benchmark.json).

## 9. Training process and completed configuration

### 9.1 Actual run sequence

1. The initial run trained from the one-class pretrained initialization and completed seven epochs under the original 100-epoch plan.
2. It was interrupted during epoch eight. Completed epoch checkpoints remained available; the partial epoch was not preserved as a completed epoch.
3. The optimized run restored the latest model and SGD optimizer state, resumed at epoch eight, and completed through epoch 30.
4. The revised cosine schedule used the shorter total horizon. Warmup was not restarted from epoch zero. This continuation is not equivalent to training the original 100-epoch schedule to completion, or to an uninterrupted fresh 30-epoch run.
5. EMA-backed model weights were evaluated on validation data, and the best checkpoint was selected using validation COCO AP@0.5:0.95.
6. The best checkpoint was evaluated separately on the held-out test set.

The resumed-run metadata records a start at **3 November 2026, 23:16:10 PKT** and finish at **4 November 2026, 00:09:11 PKT**, approximately **53 minutes 2 seconds**. This covers the resumed training segment, including its setup and validation; it excludes the first seven epochs, the interrupted partial epoch, prior preparation, the benchmark, and final test evaluation. UTC timestamps remain available in `training_run.json`.

### 9.2 Final configuration

| Parameter | Completed continuation |
|---|---|
| Architecture / classes | YOLOX-Tiny / 1 |
| Input / evaluation canvas | 416 × 416 |
| Batch size | 8 |
| Epoch target | 30 total; resumed after 7 |
| Peak total learning rate | 0.00125 |
| Optimizer | SGD with Nesterov momentum |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Schedule | Official YOLOX warmup/cosine schedule, recomputed for 30 epochs |
| Configured warmup epochs | 3 in the resumed configuration; original completed phase used 5 |
| Configured `no_aug_epochs` | 5 |
| Validation interval | 5 initially; 1 in the official final phase |
| Mixed precision | Enabled |
| EMA | Enabled through the official default |
| Data workers | 2 |
| Seed | 42 |
| Fast cuDNN | Enabled; strict deterministic kernels disabled |
| Multi-scale range | 0; fixed base input size |
| Padded label capacity | 300 per image |
| New per-epoch history saving | Disabled during continuation |

The label capacity exceeds the maximum original count of 17 and the four-image mosaic requirement, avoiding truncation at the smaller official default capacities.

### 9.3 Augmentation and losses

| Augmentation setting | Value |
|---|---:|
| Mosaic probability | 0.5 |
| Mosaic scale range | 0.5–1.5 |
| Mixup | Disabled |
| Horizontal flip probability | 0 |
| HSV augmentation probability | 0.2 |
| Rotation range | ±5° |
| Shear range | ±1° |
| Translation factor | 0.05 |

These are online training augmentations, separate from augmentation already present in the tech export. Validation/test preprocessing does not apply these random training augmentations.

The official objective combines box IoU loss with a factor of 5, objectness loss, class loss, and an additional L1 box loss in the final phase. SGD uses backpropagation and mixed-precision gradient scaling; the official loop maintains EMA weights and validation/checkpoint updates.

**Configured versus observed final phase:** although `no_aug_epochs` is configured as 5, this official trainer's condition activates at the start of epoch `30 − 5 = 25`. The logs show mosaic closing and L1 loss enabling at epoch 25, so the observed final phase spans epochs **25–30**, with validation after each. This distinction is retained here to describe the actual run accurately.

### 9.4 Validation progression

The following values are rounded as printed in the training log:

| Completed epoch | Validation AP@0.5 | Validation AP@0.5:0.95 |
|---|---:|---:|
| 6, initial run | 87.2% | 48.4% |
| 7, initial run | 39.7% | 15.6% |
| 10, continuation | 7.0% | 2.1% |
| 15 | 92.8% | 59.8% |
| 20 | 94.9% | 63.9% |
| 25 | 94.4% | 64.3% |
| 26 | 94.2% | 64.6% |
| 30 | 94.6% | 64.2% |

The log reports the best validation AP@0.5:0.95 as **64.56%**. The large early fluctuations show that validation did not improve monotonically. The later values were relatively stable; these observations alone do not prove that a longer schedule could not improve performance.

### 9.5 Checkpoints and logs

- `best_ckpt.pth`: highest validation AP@0.5:0.95; used for the reported test evaluation.
- `latest_ckpt.pth`: latest completed training epoch; intended for resume.
- `last_epoch_ckpt.pth`: the latest validation-save checkpoint in the official implementation.
- `last_mosaic_epoch_ckpt.pth`: saved around the augmentation-phase transition.
- `epoch_1_ckpt.pth` through `epoch_7_ckpt.pth`: retained history from the earlier run; later per-epoch history was disabled.
- `train_log.txt`, TensorBoard events, and [training_console.log](../signature_yolox/reports/training_console.log): recorded training and validation behavior.

The checkpoint includes more than inference weights, including optimizer state. The best checkpoint is 40,578,275 bytes; this file size is not the model's activation-memory requirement. Resume does not restore every random-generator or AMP-scaler detail, and non-deterministic kernels may change exact repeat-run results.

## 10. Evaluation protocol

The final evaluator loads the one-class model and the best checkpoint with strict state-dictionary matching. It uses the official evaluation dataloader, `ValTransform`, model decoding, and YOLOX postprocessing. Predicted boxes are mapped back to original image dimensions and clipped to image boundaries.

| Evaluation setting | Value |
|---|---:|
| Split | Held-out test |
| Images / batches | 397 / 50 at batch size 8 |
| Model input size | 416 × 416 |
| Inference score floor | 0.001 |
| NMS IoU threshold | 0.65 |
| Maximum detections per image | 100 |
| Operating confidence threshold | 0.25 |
| Operating matching IoU | 0.5 |
| COCO AP IoU thresholds | 0.50, 0.55, …, 0.95 |

A detection score is objectness multiplied by the class confidence. NMS removes overlapping predictions according to its own threshold; that threshold is separate from the IoU used to match predictions to ground truth.

COCO AP is calculated with standard `pycocotools`, using detections down to the low inference score floor. Operating precision, recall, and F1 use only predictions scoring at least 0.25. Predictions are processed in score order, and each ground-truth box can be matched once. An extra detection on an already matched signature is a false positive.

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
IoU       = intersection area / union area of predicted and ground-truth boxes
```

With one class, the reported mAP is the AP for `signature` under the specified COCO averaging protocol. COCO AP@0.5:0.95 is not the same as F1 or the fixed-threshold recall. In particular, the console's COCO average recall across IoU thresholds should not be confused with the **93.88% recall at IoU 0.5 and confidence 0.25**.

The notebook includes both an empirical score-threshold PR curve and the COCO-interpolated PR curve at IoU 0.5. True negatives are not defined as a finite count of background boxes in object detection, so the object-level confusion matrix marks its background/background cell as N/A.

## 11. What evaluation showed

### 11.1 Object-level results

The test set contains 784 annotated signatures. At the operating thresholds, the detector produced **793 predictions**, of which 736 matched ground truth and 57 were unmatched; 48 ground-truth signatures were missed.

| Ground truth / predicted | Signature | Background / missed |
|---|---:|---:|
| Signature | 736 TP | 48 FN |
| Background / unmatched | 57 FP | N/A |

The exact saved values are precision **0.9281210593**, recall **0.9387755102**, F1 **0.9334178821**, mAP@0.5 **0.9502923527**, and mAP@0.5:0.95 **0.6550610933**.

The notebook's COCO summary additionally reports AP@0.75 of approximately **76.3%**. AP@0.5 is high, while AP under stricter overlap requirements is lower. This supports the interpretation that the model detects most signatures at moderate overlap, with tighter box localization remaining an area for improvement. It does not by itself identify the exact causes of individual errors.

The full console summary is shown below. These percentages are rounded from the printed three-decimal COCO values; use the full-precision JSON for the two headline AP measurements.

| COCO summary measurement | Test result |
|---|---:|
| AP, IoU 0.50:0.95, all areas, max 100 detections | 65.5% |
| AP, IoU 0.50, all areas, max 100 detections | 95.0% |
| AP, IoU 0.75, all areas, max 100 detections | 76.3% |
| AP, IoU 0.50:0.95, small objects | 64.6% |
| AP, IoU 0.50:0.95, medium objects | 65.6% |
| AP, IoU 0.50:0.95, large objects | 65.4% |
| Average recall, IoU 0.50:0.95, all areas, max 1 detection | 30.8% |
| Average recall, IoU 0.50:0.95, all areas, max 10 detections | 69.3% |
| Average recall, IoU 0.50:0.95, all areas, max 100 detections | 70.0% |
| Average recall, IoU 0.50:0.95, small objects, max 100 detections | 65.0% |
| Average recall, IoU 0.50:0.95, medium objects, max 100 detections | 70.3% |
| Average recall, IoU 0.50:0.95, large objects, max 100 detections | 69.7% |

COCO small/medium/large categories use pixel areas in the prepared images' original coordinate systems, not physical signature size or the 416-pixel model canvas. Different source resolutions therefore affect area categories. These COCO recall summaries average over overlap thresholds and use the low-score-floor prediction set; they are separate from the operating-point recall of 93.88%.

### 11.2 Per-image error analysis

The following counts were derived from the saved [per_image_counts.csv](../signature_yolox/reports/evaluation_test/per_image_counts.csv), without rerunning the model:

| Test-image outcome at confidence 0.25 / IoU 0.5 | Images |
|---|---:|
| At least one missed signature | 38 |
| At least one false-positive detection | 52 |
| Neither FP nor FN under the matching protocol | 326 |
| Declared negative test images | 40 |
| Declared negatives with at least one false-positive detection | 5 |

The first two categories overlap. The 326 error-free matching cases are an image-level descriptive count, not the project's primary accuracy metric. Five false-alarm images among 40 negatives also show why strong aggregate AP does not eliminate the need to inspect negative documents.

### 11.3 Curves and visual examples

The saved PR curve stays high through much of the recall range before dropping near the highest recall levels. The marked confidence-0.25 operating point corresponds to the precision/recall values reported above. Choosing another threshold changes the precision/recall tradeoff and should be done on validation data, not by repeatedly optimizing against this test set.

![Precision-recall curves and detection confusion matrix](../signature_yolox/reports/evaluation_test/precision_recall_confusion.png)

The seeded prediction montage shows detections on scanned letters, noisy or rotated pages, and pages containing multiple signatures. Green boxes are ground truth; red boxes are model predictions. These selected examples illustrate behavior but do not represent a complete error audit or establish performance on all rotations and document types.

![Saved test predictions](../signature_yolox/reports/evaluation_test/sample_predictions.png)

### 11.4 Timing and interpretation

The saved test inference progress reports approximately **42 seconds** for the 397-image split. The final training-time validation summary reports approximately **5.28 ms forward time plus 4.26 ms NMS time**, or **9.54 ms per-image inference time** under that evaluator's timing convention.

These timings cover different scopes. The full test pass includes startup, data handling, and inference across batches; the official validation timing reports selected model/NMS work. Neither figure is a complete production latency benchmark including document ingestion, PDF rasterization, application overhead, or concurrent users.

## 12. Limitations and practical interpretation

- **Detection scope:** a box labeled `signature` says where the model found a signature-like region. It does not establish the signer's identity or document authenticity.
- **Dataset dependence:** results apply to the reconstructed grouped split. Generalization to different scanners, handwriting styles, languages, forms, or document domains has not been established by this test alone.
- **Residual overlap uncertainty:** grouping is stronger than random splitting by filename, but heuristic near-duplicate detection cannot guarantee that every related page was found. Existing augmented data reduce the effective independence of samples.
- **Ground-truth assumptions:** declared negatives are treated as exhaustively labeled; missed source annotations can affect both training and apparent false positives. Bounds validation cannot establish semantic label completeness.
- **Localization:** the gap between AP@0.5 and AP@0.5:0.95 indicates sensitivity to stricter localization requirements. Annotation consistency and original image quality also affect achievable overlap.
- **Resolution:** none of the source boxes projected below one pixel at the base input size, but this does not guarantee that small signatures remain easy after augmentation or resizing. The source tech export was already stretched to a square.
- **Training history:** the completed model used seven epochs of the initial schedule followed by a shortened continuation. It should not be described as a fresh, uninterrupted 30-epoch experiment.
- **Reproducibility:** split assignment uses a fixed seed. CUDA kernels, Windows workers, resume state, and the changed schedule can still affect model-level reproducibility.
- **Benchmark scope:** batch selection used a small fixed-batch compute comparison; it does not prove a universal throughput advantage for batch 8.
- **Cached validation:** the optimized resume path verifies hashes and presence but does not decode every prepared image again. Manually edited or corrupted prepared image contents need a fresh full validation.
- **Deployment:** confidence thresholds, acceptable miss/false-alarm rates, and performance on representative new documents require separate validation before operational use.

Possible future experiments include a clean fresh-run comparison, validation-only threshold tuning, focused review of false positives/misses and box alignment, and higher-resolution or document-tile experiments for small objects. These have not been performed as part of the reported run.

## 13. Reproducing or continuing the workflow

### Full preparation and a new training run

1. Keep the three local source folders available and configure `LOCAL_DATASET_BASE` or the `SIGNATURE_DATASET_BASE` environment variable.
2. Use a CUDA-compatible environment with matching PyTorch/torchvision binaries. The notebook supports local GPU execution and Google Colab setup.
3. For a genuinely new experiment, set `TRAIN['resume']=False` and choose a new experiment name. The notebook currently defaults to resuming the existing experiment.
4. Run the notebook sequentially. Review schemas, cleaning reports, related-image verification, annotated samples, and final validation before training.
5. Stage 7 writes the experiment, prepares transfer weights, optionally benchmarks batches, and checks dataloaders. Stage 8 invokes the official trainer. Stage 9 evaluates the selected checkpoint.

### Reusing this prepared dataset

With the existing kernel, rerun configuration and Stages 7–9 rather than repeating Stages 2–6. After a kernel restart, run configuration, dependencies, and official-source setup, then the Stage 7 resume path. The saved manifests, COCO annotations, validation evidence, prepared image files, and checkpoint must remain available.

The completed checkpoint has reached the current 30-epoch target. Attempting to resume toward the same target intentionally stops in preflight. To continue training, choose a higher total target and review the revised schedule. To evaluate again, skip training, load the needed setup/experiment definitions, and run Stage 9 against the saved best checkpoint. No further training is required merely to use or evaluate that checkpoint.

Recreating the transfer initialization after training does not overwrite `best_ckpt.pth`; it only affects the separate initialization artifact. The completed trained checkpoint is identified by the hash below.

## 14. Evidence and artifact index

| Artifact | Contents / purpose |
|---|---|
| [Notebook](../Signature_Detection_YOLOX_Tiny.ipynb) | Full executable workflow and saved user-run outputs |
| [Raw inventory](../signature_yolox/reports/raw_inventory.csv) | Source paths, dimensions, annotations, hashes and inspection outcomes |
| [Schemas](../signature_yolox/reports/schemas.json) | Actual export structures, categories and used IDs |
| [Audit](../signature_yolox/reports/audit.csv) | Repairs, inspection events and exclusion reasons |
| [Clean manifest](../signature_yolox/reports/clean_manifest.json) | Clean boxes, hashes and provenance before grouped splitting |
| [Merged manifest](../signature_yolox/reports/merged_manifest.json) | Retained records, provenance, groups and final splits |
| [Split manifest](../signature_yolox/reports/split_manifest.csv) | Compact split/group membership and box counts |
| [Grouping diagnostics](../signature_yolox/reports/grouping_diagnostics.json) | Group sizes, candidate counts and verification thresholds |
| [Validation](../signature_yolox/reports/validation.json) | Original full validation outcome |
| [Batch benchmark](../signature_yolox/reports/batch_benchmark.json) | Measured batch candidates and selected batch |
| [Training configuration](../signature_yolox/experiments/signature_train_config.json) | Actual saved continuation settings and paths |
| [Experiment source](../signature_yolox/experiments/signature_yolox_tiny.py) | Official-Tiny subclass and resume LR/performance adjustments |
| [Training run](../signature_yolox/reports/training_run.json) | Command, configuration, validation snapshot, timestamps, completion and checkpoint paths |
| [Training console log](../signature_yolox/reports/training_console.log) | Initial/interrupted and resumed run output |
| [Best checkpoint](../signature_yolox/outputs/signature_yolox_tiny/best_ckpt.pth) | Model used for the reported test result |
| [Latest checkpoint](../signature_yolox/outputs/signature_yolox_tiny/latest_ckpt.pth) | Latest completed training state for resume |
| [Test metrics](../signature_yolox/reports/evaluation_test/metrics.json) | Full-precision headline metrics, thresholds and evaluated checkpoint hash |
| [COCO predictions](../signature_yolox/reports/evaluation_test/predictions_coco.json) | Retained low-score-floor detections for test images |
| [Per-image outcomes](../signature_yolox/reports/evaluation_test/per_image_counts.csv) | TP, FP, FN and annotation counts for every test image |
| [PR data](../signature_yolox/reports/evaluation_test/precision_recall_by_score.csv) | Score-threshold precision/recall measurements |

Some notebook cells were revised after earlier preparation and therefore have cleared execution counts. The retained reports and completed training/evaluation outputs provide the run evidence; an empty output in a revised cell does not invalidate an existing saved report.

### Reproducibility identifiers

| Item | SHA-256 |
|---|---|
| Evaluated `best_ckpt.pth` | `9a0bef4d37c1b458ab5ab0b75d3ccb2c01fc7fc33692d1bc6e5399b56c7ad8f3` |
| Official source archive | `50b484d2fe08dd8d90fe66d41b71072eb10c7065a661086364b1482d98413bf6` |
| Official pretrained Tiny checkpoint | `9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0` |
| Training COCO annotations | `dc96b0fbef4ef3da1a78c071bf0fdd64f35e1686581680833b860ceb25b62d5b` |
| Validation COCO annotations | `6f92b654e77ab303f203ad0e92eb2194af90bd36d20e4c3d597a400dbbb62d63` |
| Test COCO annotations | `0eeb50261ea3282a90073a443080749ceea7059c82b4ffd8d265ad91f02810a9` |

Technical behavior can be traced directly to the local [official Tiny experiment](../signature_yolox/YOLOX-0.3.0/exps/default/yolox_tiny.py), [base experiment](../signature_yolox/YOLOX-0.3.0/yolox/exp/yolox_base.py), [trainer](../signature_yolox/YOLOX-0.3.0/yolox/core/trainer.py), [COCO adapter](../signature_yolox/YOLOX-0.3.0/yolox/data/datasets/coco.py), and [detection head](../signature_yolox/YOLOX-0.3.0/yolox/models/yolo_head.py). The upstream repository is [Megvii-BaseDetection/YOLOX, tag 0.3.0](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0).
