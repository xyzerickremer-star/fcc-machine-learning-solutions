# Transfer Note: Cat/Dog Image Classifier → AuditAgent / Bachelorarbeit

## freeCodeCamp concept

The project demonstrates image classification with a convolutional neural network (CNN). The solution uses a pretrained MobileNetV2 model and maps its ImageNet outputs to the binary labels `cat` and `dog`.

Core ideas:

- Classification: assign an input object to a known category.
- Feature extraction: CNN layers detect visual features such as shapes, textures, and object parts.
- Transfer learning: reuse a model trained on a large generic dataset for a narrower task.
- Confidence/probability: output can be interpreted as a likelihood score, not only as a hard label.

## AuditAgent analogy

AuditAgent probably does not need cat/dog image classification directly, but the same classification pattern is highly relevant:

1. **Document classification**
   - invoice vs. contract vs. bank statement vs. delivery note vs. PBC response
2. **Evidence classification**
   - sufficient vs. incomplete vs. wrong document type
3. **Transaction classification**
   - routine booking vs. unusual/risky booking
4. **Attachment triage**
   - route uploaded evidence to the correct audit procedure or PBC request

The important product idea is not the image domain itself, but the pipeline:

> input object → feature extraction → category prediction → confidence score → human review.

## Bachelorarbeit relevance

Useful framing:

> Transfer-learning approaches show how pretrained models can reduce implementation effort in specialized audit contexts by reusing generic feature representations and adapting them to domain-specific classification tasks.

This supports a realistic AuditAgent thesis angle: rather than claiming a full custom model from scratch, the system can be designed around pretrained models plus human-in-the-loop validation.

## Product note

For AuditAgent, a practical first implementation could use embeddings or pretrained document models instead of training from scratch:

- input: uploaded files and extracted text/OCR
- model: pretrained embedding/document classifier
- output: predicted document/evidence class plus confidence
- auditor control: confirm, correct, or reject classification
- feedback: corrections become labeled examples for later fine-tuning
