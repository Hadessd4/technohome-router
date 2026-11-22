"""
TechnoHome Customer Support Ticket Router
Script: INT8 Dynamic Quantization of Student Model
"""

import os
import sys
import json
import argparse

import torch
import torch.quantization

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.distill_and_quantize import DistilledQuantizedClassifier


def get_dir_size_mb(path):
    """Calculate total size of all files in a directory in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Quantize the distilled student model (INT8)")
    parser.add_argument("--student-dir", type=str,
                        default=str(config.MODEL_DIR / "student_model"),
                        help="Path to trained student model")
    parser.add_argument("--output-dir", type=str,
                        default=str(config.MODEL_DIR / "quantized_model"),
                        help="Directory to save quantized model")
    args = parser.parse_args()

    # ── Load student model ─────────────────────────────────────────────────
    print("=" * 60)
    print("TechnoHome INT8 Dynamic Quantization")
    print("=" * 60)

    print(f"\nLoading student model from: {args.student_dir}")
    student = DistilledQuantizedClassifier(
        student_model_name=args.student_dir,
        num_labels=config.NUM_DEPARTMENTS,
        label_map=config.DEPT_TO_IDX,
    )
    student.load(args.student_dir)
    print("  Student model loaded.")

    # Measure original size
    original_size_mb = get_dir_size_mb(args.student_dir)
    print(f"  Original model size: {original_size_mb:.2f} MB")

    # ── Apply INT8 dynamic quantization ────────────────────────────────────
    print("\nApplying INT8 dynamic quantization...")
    print("  Quantizing torch.nn.Linear layers to qint8...")

    student.model = torch.quantization.quantize_dynamic(
        student.model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    print("  Quantization complete.")

    # ── Save quantized model ───────────────────────────────────────────────
    print(f"\nSaving quantized model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save quantized model state
    quantized_model_path = os.path.join(args.output_dir, "quantized_model.pt")
    torch.save(student.model.state_dict(), quantized_model_path)

    # Save tokenizer alongside
    if hasattr(student, "tokenizer") and student.tokenizer is not None:
        student.tokenizer.save_pretrained(args.output_dir)

    # Save label map and config info
    meta = {
        "student_model_name": config.STUDENT_MODEL_NAME,
        "num_labels": config.NUM_DEPARTMENTS,
        "label_map": config.DEPT_TO_IDX,
        "quantized": True,
        "dtype": "qint8",
    }
    meta_path = os.path.join(args.output_dir, "quantize_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ── Print size comparison ──────────────────────────────────────────────
    quantized_size_mb = get_dir_size_mb(args.output_dir)

    print(f"\n{'─' * 40}")
    print(f"  Size Comparison:")
    print(f"    Original (student):  {original_size_mb:.2f} MB")
    print(f"    Quantized (INT8):    {quantized_size_mb:.2f} MB")
    if original_size_mb > 0:
        reduction = (1 - quantized_size_mb / original_size_mb) * 100
        print(f"    Reduction:           {reduction:.1f}%")
    print(f"{'─' * 40}")

    print("\nDone!")


if __name__ == "__main__":
    main()
