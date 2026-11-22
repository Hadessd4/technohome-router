"""
TechnoHome Customer Support Ticket Router
Training Script: BERT Fine-tuning (BERTurk)
"""

import os
import sys
import json
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.bert_classifier import BERTIntentClassifier


def load_dataset(data_path):
    """Load prepared_dataset.json."""
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for intent classification")
    parser.add_argument("--data-path", type=str,
                        default=str(config.DATA_DIR / "prepared_dataset.json"),
                        help="Path to prepared_dataset.json")
    parser.add_argument("--output-dir", type=str,
                        default=str(config.MODEL_DIR / "bert_intent"),
                        help="Directory to save fine-tuned model")
    parser.add_argument("--model-name", type=str, default=config.BERT_MODEL_NAME,
                        help="Pre-trained BERT model name")
    parser.add_argument("--max-length", type=int, default=config.BERT_MAX_LENGTH,
                        help="Max token sequence length")
    parser.add_argument("--epochs", type=int, default=config.BERT_EPOCHS,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=config.BERT_BATCH_SIZE,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=config.BERT_LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=config.BERT_WARMUP_STEPS,
                        help="Warmup steps for scheduler")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("TechnoHome BERT Fine-Tuning")
    print("=" * 60)

    print(f"\nLoading dataset from: {args.data_path}")
    dataset = load_dataset(args.data_path)

    train_texts = [ex["text"] for ex in dataset["train"]]
    train_labels = [ex["department"] for ex in dataset["train"]]
    val_texts = [ex["text"] for ex in dataset["validation"]]
    val_labels = [ex["department"] for ex in dataset["validation"]]

    print(f"  Train samples: {len(train_texts)}")
    print(f"  Validation samples: {len(val_texts)}")
    print(f"  Departments: {sorted(set(train_labels))}")

    # ── Initialize classifier ──────────────────────────────────────────────
    print(f"\nInitializing BERTIntentClassifier...")
    print(f"  Model: {args.model_name}")
    print(f"  Num classes: {config.NUM_DEPARTMENTS}")
    print(f"  Max length: {args.max_length}")

    classifier = BERTIntentClassifier(
        model_name=args.model_name,
        num_labels=config.NUM_DEPARTMENTS,
        max_length=args.max_length,
        label_map=config.DEPT_TO_IDX,
    )

    # ── Fine-tune ──────────────────────────────────────────────────────────
    print(f"\nFine-tuning BERT...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Seed: {args.seed}")

    classifier.fine_tune(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # ── Save model ─────────────────────────────────────────────────────────
    print(f"\nSaving model to: {args.output_dir}")
    classifier.save(args.output_dir)

    # Print model directory size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(args.output_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    size_mb = total_size / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")

    # ── Quick validation check ─────────────────────────────────────────────
    print("\nRunning quick validation check...")
    sample_texts = val_texts[:5]
    sample_labels = val_labels[:5]
    for text, true_label in zip(sample_texts, sample_labels):
        pred_label, confidence = classifier.predict(text)
        status = "✓" if pred_label == true_label else "✗"
        print(f"  {status} [{confidence:.3f}] {true_label:>25} → {pred_label:<25} | {text[:60]}...")

    print("\nDone!")


if __name__ == "__main__":
    main()
