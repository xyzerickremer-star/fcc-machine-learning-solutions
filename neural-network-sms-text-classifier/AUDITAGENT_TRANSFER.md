# AuditAgent / Bachelorarbeit Transfer Note

## What transfers

The SMS classifier is a compact analogue for AuditAgent triage: short, noisy text is mapped to an actionable class with a confidence score.

- **SMS ham/spam** -> **PBC/document/email relevant vs. non-relevant**
- **Spam probability** -> **triage/routing confidence**
- **Keyword and phrase features** -> **audit cues such as invoice, contract, bank statement, deadline, missing evidence, reminder**
- **False positives/negatives** -> **review workload vs. missed audit evidence risk**

## AuditAgent use cases

- Prioritize incoming PBC request emails by urgency and evidence type.
- Route attachments or document snippets to audit workstreams.
- Flag likely noise, duplicates, marketing mail, or unrelated correspondence before auditor review.
- Surface confidence scores so low-confidence items stay in a human-in-the-loop queue.

## Bachelorarbeit framing

This project demonstrates a baseline text-classification pipeline for audit automation: labeled examples, feature extraction, probabilistic classification, validation metrics, and operational thresholds. In the thesis, the same pattern can be extended from SMS messages to PBC metadata, document OCR text, and email bodies, with governance requirements for explainability, sampling, reviewer override, and audit-trail logging.
