"""
TechnoHome Customer Support Ticket Router
Training Script: Knowledge Distillation (BERT → DistilBERT)
"""

import os
import sys
import json
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.bert_classifier import BERTIntentClassifier
from models.distill_and_quantize import DistilledQuantizedClassifier


def load_dataset(data_path):
    """Load prepared_dataset.json."""
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation from BERT to DistilBERT")
    parser.add_argument("--data-path", type=str,
                        default=str(config.DATA_DIR / "prepared_dataset.json"),
                        help="Path to prepared_dataset.json")
    parser.add_argument("--teacher-dir", type=str,
                        default=str(config.MODEL_DIR / "bert_intent"),
                        help="Path to trained BERT teacher model")
    parser.add_argument("--output-dir", type=str,
                        default=str(config.MODEL_DIR / "student_model"),
                        help="Directory to save distilled student model")
    parser.add_argument("--student-model", type=str, default=config.STUDENT_MODEL_NAME,
                        help="Pre-trained student model name")
    parser.add_argument("--epochs", type=int, default=config.DISTILL_EPOCHS,
                        help="Number of distillation epochs")
    parser.add_argument("--batch-size", type=int, default=config.BERT_BATCH_SIZE,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=config.DISTILL_LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--temperature", type=float, default=config.DISTILL_TEMPERATURE,
                        help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=config.DISTILL_ALPHA,
                        help="Weight for distillation loss vs hard loss")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("TechnoHome Knowledge Distillation (BERT → DistilBERT)")
    print("=" * 60)

    print(f"\nLoading dataset from: {args.data_path}")
    dataset = load_dataset(args.data_path)

    train_texts = [ex["text"] for ex in dataset["train"]]
    train_labels = [ex["department"] for ex in dataset["train"]]
    val_texts = [ex["text"] for ex in dataset["validation"]]
    val_labels = [ex["department"] for ex in dataset["validation"]]

    print(f"  Train samples: {len(train_texts)}")
    print(f"  Validation samples: {len(val_texts)}")

    # ── Load teacher model ─────────────────────────────────────────────────
    print(f"\nLoading BERT teacher from: {args.teacher_dir}")
    teacher = BERTIntentClassifier(
        model_name=config.BERT_MODEL_NAME,
        num_labels=config.NUM_DEPARTMENTS,
        max_length=config.BERT_MAX_LENGTH,
        label_map=config.DEPT_TO_IDX,
    )
    teacher.load(args.teacher_dir)
    print("  Teacher loaded successfully.")

    # ── Initialize student ─────────────────────────────────────────────────
    print(f"\nInitializing DistilBERT student: {args.student_model}")
    student = DistilledQuantizedClassifier(
        student_model_name=args.student_model,
        num_labels=config.NUM_DEPARTMENTS,
        label_map=config.DEPT_TO_IDX,
    )

    # ── Distillation ───────────────────────────────────────────────────────
    print(f"\nStarting knowledge distillation...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Alpha: {args.alpha}")

    student.distill(
        teacher_model=teacher,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        alpha=args.alpha,
        seed=args.seed,
    )

    # ── Save student model ─────────────────────────────────────────────────
    print(f"\nSaving student model to: {args.output_dir}")
    student.save(args.output_dir)

    # Print model size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(args.output_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    size_mb = total_size / (1024 * 1024)
    print(f"Student model size: {size_mb:.2f} MB")

    # ── Quick validation ───────────────────────────────────────────────────
    print("\nQuick validation check on student model...")
    correct = 0
    n_check = min(100, len(val_texts))
    for text, true_label in zip(val_texts[:n_check], val_labels[:n_check]):
        pred_label, confidence = student.predict(text)
        if pred_label == true_label:
            correct += 1
    acc = correct / n_check if n_check > 0 else 0.0
    print(f"  Student accuracy on first {n_check} val samples: {acc:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
