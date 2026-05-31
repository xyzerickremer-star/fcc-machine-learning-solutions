# Cat and Dog Image Classifier

freeCodeCamp Machine Learning with Python project solution.

## Approach

The notebook uses `MobileNetV2`, a pretrained convolutional neural network from TensorFlow/Keras, and maps ImageNet cat/dog class predictions to the binary freeCodeCamp labels:

- `0` = cat
- `1` = dog

This is a transfer-learning style solution: instead of training a CNN from scratch, it reuses visual features learned from a large image corpus.

## Verification

Local verification with the official freeCodeCamp test labels:

```bash
source .venv/bin/activate
python test_mobilenet_solution.py
```

Latest local result: **47/50 correct = 94.0%**. Required by freeCodeCamp: **>= 63%**.

The helper script was used for local verification; the submitted solution is the notebook `fcc_cat_dog.ipynb`.
