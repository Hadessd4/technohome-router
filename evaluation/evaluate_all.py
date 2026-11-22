"""
TechnoHome Customer Support Ticket Router
Evaluation: Compare All 4 Models on the Same Test Set
"""

import os
import sys
import json
import time
import argparse

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def load_dataset(data_path):
    """Load prepared_dataset.json."""
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def get_dir_size_mb(path):
    """Calculate total size of all files in a directory (or single file) in MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)


def compute_metrics(y_true, y_pred, departments):
    """Compute accuracy, macro-F1, per-class accuracy, OOS recall."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    per_class = {}
    class_precision = {}
    class_recall = {}
    class_f1 = {}

    for dept in departments:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == dept and p == dept)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != dept and p == dept)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == dept and p != dept)
        total = sum(1 for t in y_true if t == dept)

        class_acc = tp / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[dept] = {
            "accuracy": round(class_acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": total,
        }
        class_precision[dept] = precision
        class_recall[dept] = recall
        class_f1[dept] = f1

    macro_f1 = np.mean(list(class_f1.values())) if class_f1 else 0.0
    oos_recall = class_recall.get("human_agent", 0.0)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(macro_f1), 4),
        "oos_recall": round(oos_recall, 4),
        "per_class": per_class,
    }


def measure_latency(predict_fn, test_inputs, n_runs=100):
    """Measure average prediction latency in ms."""
    for text in test_inputs[:5]:
        predict_fn(text)

    sample = test_inputs[:n_runs] if len(test_inputs) >= n_runs else test_inputs

    start = time.perf_counter()
    for text in sample:
        predict_fn(text)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / len(sample)) * 1000
    return round(avg_ms, 2)


def classification_report_dict(y_true, y_pred, departments):
    """Generate a classification report as a dict."""
    report = {}
    for dept in departments:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == dept and p == dept)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != dept and p == dept)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == dept and p != dept)
        total = sum(1 for t in y_true if t == dept)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report[dept] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1, 4),
            "support": total,
        }
    return report


def extract_label(result):
    """Extract label string from predict result (dict or tuple)."""
    if isinstance(result, dict):
        return result.get("department", "human_agent")
    elif isinstance(result, (tuple, list)):
        return result[0]
    return str(result)


def load_all_models():
    """Load all 4 trained models and return as a list of (name, predict_fn, model_path)."""
    models = []

    # 1. TF-IDF + LogReg
    try:
        from models.baseline_tfidf import TFIDFBaseline
        tfidf_path = str(config.MODEL_DIR / "tfidf_logreg.joblib")
        tfidf_model = TFIDFBaseline()
        tfidf_model.load(tfidf_path)

        def tfidf_predict(text):
            return tfidf_model.predict_raw(text)

        models.append(("TF-IDF + LogReg", tfidf_predict, tfidf_path))
        print("  ✓ TF-IDF + LogReg loaded")
    except Exception as e:
        print(f"  ✗ TF-IDF + LogReg failed to load: {e}")

    # 2. FastText + FFN
    try:
        from models.feedforward_nn import FFNClassifier

        ffn_path = str(config.MODEL_DIR / "ffn_model.pt")
        ffn_model = FFNClassifier(
            input_dim=config.FASTTEXT_DIM,
            hidden_dim=config.FFN_HIDDEN_DIM,
            num_classes=config.NUM_DEPARTMENTS,
            dropout=config.FFN_DROPOUT,
        )
        ffn_model.load(ffn_path)

        def ffn_predict(text):
            return ffn_model.predict(text)

        models.append(("FastText + FFN", ffn_predict, ffn_path))
        print("  ✓ FastText + FFN loaded")
    except Exception as e:
        print(f"  ✗ FastText + FFN failed to load: {e}")

    # 3. BERT (fine-tuned)
    try:
        from models.bert_classifier import BERTIntentClassifier
        bert_dir = str(config.MODEL_DIR / "bert_intent")
        bert_model = BERTIntentClassifier(
            model_name=config.BERT_MODEL_NAME,
            num_labels=config.NUM_DEPARTMENTS,
            label_map=config.DEPT_TO_IDX,
        )
        bert_model.load(bert_dir)

        def bert_predict(text):
            return bert_model.predict_raw(text)

        models.append(("BERT (BERTurk)", bert_predict, bert_dir))
        print("  ✓ BERT (BERTurk) loaded")
    except Exception as e:
        print(f"  ✗ BERT (BERTurk) failed to load: {e}")

    # 4. Quantized Student (DistilBERT)
    try:
        from models.distill_and_quantize import DistilledQuantizedClassifier
        quantized_dir = str(config.MODEL_DIR / "quantized_model")
        student_model = DistilledQuantizedClassifier(
            student_model_name=config.STUDENT_MODEL_NAME,
            num_labels=config.NUM_DEPARTMENTS,
            label_map=config.DEPT_TO_IDX,
        )
        student_model.load(quantized_dir)

        def quant_predict(text):
            return student_model.predict_raw(text)

        models.append(("DistilBERT (Quantized)", quant_predict, quantized_dir))
        print("  ✓ DistilBERT (Quantized) loaded")
    except Exception as e:
        print(f"  ✗ DistilBERT (Quantized) failed to load: {e}")

    return models


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models on the same test set")
    parser.add_argument("--data-path", type=str,
                        default=str(config.DATA_DIR / "prepared_dataset.json"),
                        help="Path to prepared_dataset.json")
    parser.add_argument("--latency-runs", type=int, default=config.LATENCY_RUNS,
                        help="Number of runs for latency measurement")
    args = parser.parse_args()

    print("=" * 70)
    print("TechnoHome Model Evaluation — All Models Comparison")
    print("=" * 70)

    # ── Load test data ─────────────────────────────────────────────────────
    print(f"\nLoading test data from: {args.data_path}")
    dataset = load_dataset(args.data_path)

    test_texts = [ex["text"] for ex in dataset["test"]]
    test_labels = [ex["department"] for ex in dataset["test"]]
    print(f"  Test samples: {len(test_texts)}")

    departments = config.DEPARTMENTS

    # ── Load all models ────────────────────────────────────────────────────
    print("\nLoading models...")
    models = load_all_models()

    if not models:
        print("\n[ERROR] No models could be loaded. Please train models first.")
        return

    # ── Evaluate each model ────────────────────────────────────────────────
    results = []
    all_reports = {}

    for model_name, predict_fn, model_path in models:
        print(f"\n{'─' * 60}")
        print(f"Evaluating: {model_name}")
        print(f"{'─' * 60}")

        # Get predictions
        y_pred = []
        for text in test_texts:
            try:
                pred_label, _ = predict_fn(text)
                y_pred.append(pred_label)
            except Exception as e:
                y_pred.append("human_agent")

        # Compute metrics
        metrics = compute_metrics(test_labels, y_pred, departments)
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Macro-F1:   {metrics['macro_f1']:.4f}")
        print(f"  OOS Recall: {metrics['oos_recall']:.4f}")

        # Model size
        size_mb = get_dir_size_mb(model_path)
        print(f"  Size:       {size_mb:.2f} MB")

        # Latency
        latency_ms = measure_latency(predict_fn, test_texts, n_runs=args.latency_runs)
        print(f"  Latency:    {latency_ms:.2f} ms (avg over {args.latency_runs} runs)")

        results.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "OOS Recall": metrics["oos_recall"],
            "Macro-F1": metrics["macro_f1"],
            "Size (MB)": round(size_mb, 2),
            "Latency (ms)": latency_ms,
        })

        report = classification_report_dict(test_labels, y_pred, departments)
        all_reports[model_name] = report

    # ── Print results table ────────────────────────────────────────────────
    print(f"\n{'=' * 85}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 85}")
    header = f"{'Model':<25} {'Accuracy':>10} {'OOS Recall':>12} {'Macro-F1':>10} {'Size (MB)':>10} {'Latency (ms)':>13}"
    print(header)
    print("-" * 85)
    for r in results:
        row = (f"{r['Model']:<25} {r['Accuracy']:>10.4f} {r['OOS Recall']:>12.4f} "
               f"{r['Macro-F1']:>10.4f} {r['Size (MB)']:>10.2f} {r['Latency (ms)']:>13.2f}")
        print(row)
    print("-" * 85)

    # ── Save results ───────────────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    results_path = os.path.join(config.RESULTS_DIR, "results_table.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults table saved to: {results_path}")

    reports_path = os.path.join(config.RESULTS_DIR, "classification_reports.json")
    with open(reports_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"Classification reports saved to: {reports_path}")

    # ── Print per-class breakdown for each model ───────────────────────────
    for model_name, report in all_reports.items():
        print(f"\n{'─' * 60}")
        print(f"Classification Report: {model_name}")
        print(f"{'─' * 60}")
        print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 65)
        for dept in departments:
            if dept in report:
                r = report[dept]
                print(f"{dept:<25} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                      f"{r['f1-score']:>10.4f} {r['support']:>10}")
        print("-" * 65)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
