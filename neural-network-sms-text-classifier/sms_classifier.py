"""freeCodeCamp Neural Network SMS Text Classifier solution.

The public entry point is ``predict_message(pred_text)``, which returns
``[spam_probability, "ham"|"spam"]`` as required by the FCC project tests.

This implementation is intentionally notebook-friendly and dependency-light:
it downloads the official FCC SMS data and trains a multinomial Naive Bayes
text classifier in pure Python. It behaves like a classic sklearn text
pipeline (tokenization + n-gram counts + probabilistic classifier), so it runs
quickly in local shells and Colab notebooks without storing model artifacts.
"""

from __future__ import annotations

import csv
import math
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

TRAIN_URL = "https://cdn.freecodecamp.org/project-data/sms/train-data.tsv"
VALID_URL = "https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv"
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_FILE = DATA_DIR / "train-data.tsv"
VALID_FILE = DATA_DIR / "valid-data.tsv"

RANDOM_SEED = 42
ALPHA = 0.35
SPAM_THRESHOLD = 0.50

_classifier: Optional["SmsNaiveBayes"] = None
_TOKEN_RE = re.compile(r"[a-z0-9£$€]+")


def _download_if_needed() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for url, path in ((TRAIN_URL, TRAIN_FILE), (VALID_URL, VALID_FILE)):
        if not path.exists() or path.stat().st_size == 0:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())


def _load_tsv(path: Path) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                continue
            label, text = row
            labels.append(1 if label.strip().lower() == "spam" else 0)
            texts.append(text)
    return texts, labels


def _features(text: str) -> List[str]:
    """Extract word/phrase features useful for short SMS spam detection."""
    text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    feats: List[str] = []

    # Unigrams and bigrams cover most SMS cues: call, txt, won, prize, mobile.
    feats.extend(f"w={token}" for token in tokens)
    feats.extend(f"b={a}_{b}" for a, b in zip(tokens, tokens[1:]))

    # Compact metadata features make the model robust on very short messages.
    digit_count = sum(ch.isdigit() for ch in text)
    if digit_count:
        feats.append("HAS_DIGIT")
    if digit_count >= 5:
        feats.append("MANY_DIGITS")
    if any(symbol in text for symbol in ("£", "$", "€")):
        feats.append("HAS_MONEY_SYMBOL")
    if "!" in text:
        feats.append("HAS_EXCLAMATION")
    if len(tokens) <= 4:
        feats.append("VERY_SHORT")

    return feats


class SmsNaiveBayes:
    """Small multinomial Naive Bayes classifier for SMS ham/spam."""

    def __init__(self, alpha: float = ALPHA) -> None:
        self.alpha = alpha
        self.class_doc_counts = [0, 0]
        self.feature_counts = [Counter(), Counter()]
        self.total_feature_counts = [0, 0]
        self.vocabulary: set[str] = set()

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "SmsNaiveBayes":
        for text, label in zip(texts, labels):
            klass = int(label)
            self.class_doc_counts[klass] += 1
            counts = Counter(_features(text))
            self.feature_counts[klass].update(counts)
            self.total_feature_counts[klass] += sum(counts.values())
            self.vocabulary.update(counts)
        return self

    def predict_proba_spam(self, text: str) -> float:
        vocab_size = max(len(self.vocabulary), 1)
        total_docs = sum(self.class_doc_counts)
        log_probs = []
        counts = Counter(_features(text))

        for klass in (0, 1):
            prior = (self.class_doc_counts[klass] + self.alpha) / (total_docs + 2 * self.alpha)
            log_prob = math.log(prior)
            denom = self.total_feature_counts[klass] + self.alpha * vocab_size
            for feature, count in counts.items():
                numerator = self.feature_counts[klass].get(feature, 0) + self.alpha
                log_prob += count * math.log(numerator / denom)
            log_probs.append(log_prob)

        # Stable sigmoid of spam-vs-ham log-odds.
        log_odds = log_probs[1] - log_probs[0]
        if log_odds >= 0:
            return 1.0 / (1.0 + math.exp(-log_odds))
        exp_value = math.exp(log_odds)
        return exp_value / (1.0 + exp_value)


def train_model() -> SmsNaiveBayes:
    """Download FCC data if needed, train the classifier, and cache it."""
    global _classifier
    if _classifier is not None:
        return _classifier

    random.seed(RANDOM_SEED)
    _download_if_needed()
    train_texts, train_labels = _load_tsv(TRAIN_FILE)
    _classifier = SmsNaiveBayes().fit(train_texts, train_labels)
    return _classifier


def predict_message(pred_text: str):
    """Return ``[probability, label]`` for one SMS message.

    ``probability`` is the estimated probability that the message is spam. The
    returned label is ``"spam"`` when that probability is at least 0.5 and
    ``"ham"`` otherwise.
    """
    classifier = train_model()
    probability = float(classifier.predict_proba_spam(pred_text))
    label = "spam" if probability >= SPAM_THRESHOLD else "ham"
    return [probability, label]


if __name__ == "__main__":
    for message in (
        "how are you doing today",
        "sale today! to stop texts call 98912460 4 08712460324",
        "you have won £1000 cash! call to claim your prize.",
    ):
        print(message, "=>", predict_message(message))
