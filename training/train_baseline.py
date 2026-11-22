"""
TechnoHome Customer Support Ticket Router
Training Script: TF-IDF + Logistic Regression Baseline
"""

import os
import sys
import json
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.baseline_tfidf import TFIDFBaseline


def load_dataset(data_path):
    """Load prepared_dataset.json and return splits."""
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + LogReg baseline model")
    parser.add_argument("--data-path", type=str,
                        default=str(config.DATA_DIR / "prepared_dataset.json"),
                        help="Path to prepared_dataset.json")
    parser.add_argument("--output-path", type=str,
                        default=str(config.MODEL_DIR / "tfidf_logreg.joblib"),
                        help="Path to save trained model")
    parser.add_argument("--max-features", type=int, default=config.TFIDF_MAX_FEATURES,
                        help="Max features for TF-IDF vectorizer")
    parser.add_argument("--C", type=float, default=config.LOGREG_C,
                        help="Regularization parameter for LogReg")
    parser.add_argument("--max-iter", type=int, default=config.LOGREG_MAX_ITER,
                        help="Max iterations for LogReg")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("TechnoHome TF-IDF + LogReg Baseline Training")
    print("=" * 60)

    print(f"\nLoading dataset from: {args.data_path}")
    dataset = load_dataset(args.data_path)

    train_texts = [ex["text"] for ex in dataset["train"]]
    train_labels = [ex["department"] for ex in dataset["train"]]
    val_texts = [ex["text"] for ex in dataset["validation"]]
    val_labels = [ex["department"] for ex in dataset["validation"]]

    print(f"  Train samples: {len(train_texts)}")
    print(f"  Validation samples: {len(val_texts)}")
    print(f"  Departments: {len(set(train_labels))}")

    # ── Train model ────────────────────────────────────────────────────────
    print(f"\nTraining TF-IDF + LogReg baseline...")
    print(f"  max_features={args.max_features}, C={args.C}, max_iter={args.max_iter}")

    model = TFIDFBaseline(
        max_features=args.max_features,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        C=args.C,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    model.train(train_texts, train_labels)
    print("  Training complete.")

    # ── Evaluate on validation set ─────────────────────────────────────────
    print("\nEvaluating on validation set...")
    correct = 0
    total = len(val_texts)
    for text, true_label in zip(val_texts, val_labels):
        pred_label, _ = model.predict(text)
        if pred_label == true_label:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"  Validation Accuracy: {accuracy:.4f} ({correct}/{total})")

    # ── Save model ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    model.save(args.output_path)
    print(f"\nModel saved to: {args.output_path}")

    # Print model size
    model_size_mb = os.path.getsize(args.output_path) / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")
    print("\nDone!")


if __name__ == "__main__":
    main()
