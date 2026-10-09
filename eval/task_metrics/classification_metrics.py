"""
Classification metrics for specialist tool outputs (task-specific metrics: CONCH, BiomedCLIP).

Reports accuracy (= micro F1 for single-label prediction) and macro / weighted
precision, recall and F1 over the ground-truth classes.

Input JSON: a list of records with a ground-truth field and a prediction field.
Labels are compared case-insensitively; a list prediction uses its first element,
and "{LABEL}" is read as "LABEL".

Usage:
    python classification_metrics.py --input outputs/task_metrics/conch_crc100k_pred.json
    python classification_metrics.py --input BiomedClip_all.json --output biomedclip_metrics.json
"""
import argparse
import json
from collections import Counter


def normalize_label(value):
    if isinstance(value, list):
        value = value[0] if value else ""
    value = (value or "").strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    return value.strip().lower()


def compute_metrics(gts, preds):
    n = len(gts)
    correct = sum(g == p for g, p in zip(gts, preds))
    labels = sorted(set(gts))
    support = Counter(gts)

    per_class = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gts, preds))
        fp = sum(g != label and p == label for g, p in zip(gts, preds))
        fn = sum(g == label and p != label for g, p in zip(gts, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support[label]}

    def macro(key):
        return sum(c[key] for c in per_class.values()) / len(per_class)

    def weighted(key):
        return sum(c[key] * c["support"] for c in per_class.values()) / n

    accuracy = correct / n if n else 0.0
    return {
        "samples": n,
        "correct": correct,
        "accuracy": accuracy,
        "micro_f1": accuracy,
        "num_gt_classes": len(labels),
        "num_pred_classes": len(set(preds)),
        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "macro_f1": macro("f1"),
        "weighted_precision": weighted("precision"),
        "weighted_recall": weighted("recall"),
        "weighted_f1": weighted("f1"),
        "per_class": per_class,
    }


def main():
    p = argparse.ArgumentParser(description="Accuracy and macro / weighted P, R, F1 for classification outputs.")
    p.add_argument("--input", required=True, help="JSON list of records with ground truth and prediction.")
    p.add_argument("--gt-key", default="gt")
    p.add_argument("--pred-key", default="output")
    p.add_argument("--output", default=None, help="Optional path to save the metrics JSON.")
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)
    gts = [normalize_label(r.get(args.gt_key)) for r in records]
    preds = [normalize_label(r.get(args.pred_key)) for r in records]
    metrics = compute_metrics(gts, preds)

    for key, value in metrics.items():
        if key == "per_class":
            continue
        print(f"{key:20s}: {value:.4f}" if isinstance(value, float) else f"{key:20s}: {value}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
