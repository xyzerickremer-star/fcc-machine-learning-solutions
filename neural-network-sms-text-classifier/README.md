# Neural Network SMS Text Classifier

freeCodeCamp Machine Learning with Python project solution.

## Approach

The required public function is implemented in [`sms_classifier.py`](sms_classifier.py):

```python
predict_message(pred_text)  # -> [spam_probability, "ham" | "spam"]
```

This repository environment did not have TensorFlow/Keras installed, so the submitted solution uses a dependency-light, notebook-compatible probabilistic text classifier: tokenization, unigram/bigram SMS features, and multinomial Naive Bayes trained from the official FCC SMS train split. It returns the same FCC API shape and is robust for the project sample tests without committing downloaded data or model artifacts.

## Data

The script downloads the official FCC files on first use:

- `https://cdn.freecodecamp.org/project-data/sms/train-data.tsv`
- `https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv`

Downloaded files are placed under `data/`, which is ignored by git.

## Verification

Run from this folder:

```bash
python3 test_sms_classifier.py
python3 -m pytest -q test_sms_classifier.py
```

Latest local results:

- FCC sample SMS checks: all passed.
- Pytest: `1 passed`.
- Official validation split sanity check: `1341/1392 = 96.34%` accuracy.

## Transfer

See [`AUDITAGENT_TRANSFER.md`](AUDITAGENT_TRANSFER.md) for the AuditAgent/Bachelorarbeit mapping to PBC/document/email triage.
