"""
Evaluate classifier accuracy by comparing output folders against ground-truth folders.

Expected layout for both --ground_truth and --predicted:
    root/
    ├── people/     (files that contain a full human body)
    └── nopeople/   (everything else)

Usage:
    python evaluate.py --ground_truth ./gt --predicted ./output
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

LABELS = ["people", "nopeople"]


def _collect_pairs(gt_root: Path, pred_root: Path) -> Tuple[List[str], List[str]]:
    y_true: List[str] = []
    y_pred: List[str] = []
    missing = 0

    for true_label in LABELS:
        gt_dir = gt_root / true_label
        if not gt_dir.exists():
            continue
        for file in sorted(gt_dir.iterdir()):
            if not file.is_file():
                continue
            y_true.append(true_label)

            # Look in each predicted label folder
            found_label = None
            for pred_label in LABELS:
                if (pred_root / pred_label / file.name).exists():
                    found_label = pred_label
                    break

            if found_label is not None:
                y_pred.append(found_label)
            else:
                missing += 1
                y_pred.append("missing")

    if missing:
        print(f"Warning: {missing} ground-truth file(s) not found in any predicted folder.")

    # Drop unresolved entries so metrics stay binary
    paired = [(t, p) for t, p in zip(y_true, y_pred) if p != "missing"]
    if not paired:
        return [], []
    y_t, y_p = zip(*paired)
    return list(y_t), list(y_p)


def evaluate(gt_root: Path, pred_root: Path, output_png: str = "confusion_matrix.png") -> None:
    y_true, y_pred = _collect_pairs(gt_root, pred_root)

    if not y_true:
        print("No matching files found for evaluation.")
        return

    print(f"\nEvaluated {len(y_true)} files\n")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"Confusion matrix saved to {output_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classifier output against ground truth.")
    parser.add_argument("--ground_truth", required=True, help="Root folder with people/ and nopeople/ sub-dirs (ground truth labels)")
    parser.add_argument("--predicted", required=True, help="Root folder with people/ and nopeople/ sub-dirs (classifier output)")
    parser.add_argument("--output", default="confusion_matrix.png", help="Where to save the confusion matrix image")
    args = parser.parse_args()
    evaluate(Path(args.ground_truth), Path(args.predicted), args.output)
