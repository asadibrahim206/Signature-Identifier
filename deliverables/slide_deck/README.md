# Two-slide presentation

- **Signature_Detection.pptx** — exactly two slides in 16:9 format. Text, evaluation table and architecture flow remain editable.
- **Signature_Detection.pdf** — the same two slides as a visual PDF companion. Edit the PowerPoint file to change the content; the PDF uses rendered slide images.

Slide 1 introduces the document-to-annotated-image workflow, local datasets, cleaning, grouped split and GPU training. It includes an actual API prediction on a held-out nine-signature page, cropped for presentation visibility.

Slide 2 presents test-set metrics and the online prediction flow. Precision, recall and F1 use confidence 0.25 and matching IoU 0.50; AP is evaluated across the score curve. CPU deployment is distinct from GPU training.

## Suggested short talk track

**Slide 1:** “The system locates signatures in document images and returns the original page with bounding boxes and confidence scores. We combined three local datasets, validated the annotations, retained negative pages and grouped related images before creating the training, validation and test splits. YOLOX-Tiny was fine-tuned with COCO transfer learning on the RTX 3060 laptop. The example on the right is an actual API response with nine detected signatures.”

**Slide 2:** “On the held-out 397-image test split, mAP at IoU 0.50 was 95.03%, while the stricter 0.50–0.95 average was 65.51%. Precision and recall were about 93%. The trained model is packaged in Docker and served by FastAPI. Swagger submits a document to the prediction endpoint, which applies preprocessing, YOLOX inference and non-maximum suppression before returning the annotated PNG. These results establish performance on this test split; new document collections need further validation.”

## Sources and review

Source details are embedded in both slides' speaker notes. Principal evidence: [project report](../../docs/README.md), the notebook's saved outputs, [test metrics](../../signature_yolox/reports/evaluation_test/metrics.json), [API implementation](../../app/main.py), [detector](../../app/detector.py), Docker configuration and the saved `.artifacts/docker-prediction.png` verification image.

The deck passed package, slide-count, font and layout checks. Both slides were rendered and visually reviewed. This is not a claim of native Microsoft PowerPoint or Google Slides testing. The API image and verification evidence are from 5 November 2026; this presentation was prepared on 9 November 2026.
