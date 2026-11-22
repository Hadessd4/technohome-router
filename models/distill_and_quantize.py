"""
TechnoHome Customer Support Ticket Router — Knowledge Distillation & Quantization
Teacher: fine-tuned BERTurk  →  Student: DistilBERT (multilingual)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import config
from models.model_utils import Prediction


# ─── Custom Dataset (mirrors bert_classifier.IntentDataset) ──────────────────

class _DistillDataset(Dataset):
    """Tokenised dataset for distillation training containing both student and teacher encodings."""

    def __init__(self, texts, labels, student_tokenizer, teacher_tokenizer, max_length=config.BERT_MAX_LENGTH):
        self.s_enc = student_tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.t_enc = teacher_tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "s_input_ids": self.s_enc["input_ids"][idx],
            "s_attention_mask": self.s_enc["attention_mask"][idx],
            "t_input_ids": self.t_enc["input_ids"][idx],
            "t_attention_mask": self.t_enc["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ─── Distillation loss ───────────────────────────────────────────────────────

def distillation_loss_fn(student_logits, teacher_logits, labels,
                         T=config.DISTILL_TEMPERATURE,
                         alpha=config.DISTILL_ALPHA):
    """Combined knowledge-distillation loss.

    ``L = α · KL(softmax(teacher/T) || softmax(student/T)) · T²
          + (1 − α) · CE(student, labels)``
    """
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)

    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss


# ─── Main Classifier ────────────────────────────────────────────────────────

class DistilledQuantizedClassifier:
    """DistilBERT student trained by distilling a fine-tuned BERTurk teacher,
    with optional dynamic quantization (qint8).
    """

    def __init__(self, student_model_name=config.STUDENT_MODEL_NAME,
                 num_labels=config.NUM_DEPARTMENTS,
                 label_map=None):
        if student_model_name == config.STUDENT_MODEL_NAME:
            local_student = config.MODEL_DIR / "student_model"
            if local_student.exists():
                student_model_name = str(local_student)
        self.student_model_name = student_model_name
        self.num_labels = num_labels
        self.label_map = label_map or config.DEPT_TO_IDX
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            student_model_name, num_labels=num_labels,
        ).to(self.device)

    # ── Distillation training ────────────────────────────────────────────────

    def distill(self, teacher_model, train_texts, train_labels,
                val_texts=None, val_labels=None,
                epochs=config.DISTILL_EPOCHS,
                batch_size=config.BERT_BATCH_SIZE,
                learning_rate=None, lr=None,
                temperature=config.DISTILL_TEMPERATURE,
                alpha=config.DISTILL_ALPHA,
                seed=None):
        """Train the student by distilling from *teacher_model*.

        teacher_model can be a BERTIntentClassifier instance or a raw
        transformers PreTrainedModel.
        """
        actual_lr = learning_rate or lr or config.DISTILL_LEARNING_RATE

        # Get the underlying torch model from teacher
        if hasattr(teacher_model, 'model'):
            teacher_torch = teacher_model.model
        else:
            teacher_torch = teacher_model

        teacher_torch.eval()
        teacher_device = next(teacher_torch.parameters()).device

        # Get teacher tokenizer
        if hasattr(teacher_model, 'tokenizer'):
            teacher_tokenizer = teacher_model.tokenizer
        else:
            teacher_tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)

        # Convert string labels to indices if needed
        if train_labels and isinstance(train_labels[0], str):
            train_labels = [self.label_map[l] for l in train_labels]

        # Build student dataset with both tokenizers
        student_dataset = _DistillDataset(
            train_texts, train_labels, self.tokenizer, teacher_tokenizer
        )
        loader = DataLoader(student_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=actual_lr)

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            correct = 0
            total = 0
            for batch in loader:
                s_input_ids = batch["s_input_ids"].to(self.device)
                s_attention_mask = batch["s_attention_mask"].to(self.device)
                t_input_ids = batch["t_input_ids"].to(teacher_device)
                t_attention_mask = batch["t_attention_mask"].to(teacher_device)
                labels = batch["labels"].to(self.device)

                student_out = self.model(
                    input_ids=s_input_ids,
                    attention_mask=s_attention_mask,
                )
                student_logits = student_out.logits

                with torch.no_grad():
                    teacher_out = teacher_torch(
                        input_ids=t_input_ids,
                        attention_mask=t_attention_mask,
                    )
                teacher_logits = teacher_out.logits.to(self.device)

                loss = distillation_loss_fn(
                    student_logits, teacher_logits, labels,
                    T=temperature, alpha=alpha
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * s_input_ids.size(0)
                correct += (student_logits.argmax(dim=1) == labels).sum().item()
                total += s_input_ids.size(0)

            avg_loss = total_loss / total
            accuracy = correct / total
            print(f"[Distill] Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  acc={accuracy:.4f}")

    # ── Quantization ─────────────────────────────────────────────────────────

    @staticmethod
    def quantize(model):
        """Apply dynamic quantization (qint8) to Linear layers."""
        model = model.cpu()
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        return quantized

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, text):
        """Predict the department for a single *text*.

        Returns
        -------
        dict with 'department' and 'confidence' keys (for app.py compatibility)
        """
        self.model.eval()
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=config.BERT_MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze(0)
            idx = int(torch.argmax(probs))
            confidence = float(probs[idx])

        label = config.IDX_TO_DEPT[idx]
        return Prediction(label, confidence)

    def predict_raw(self, text):
        """Predict returning (label, confidence) tuple."""
        result = self.predict(text)
        return result["department"], result["confidence"]

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path=None):
        """Save student model + tokenizer to a directory at *path*."""
        if path is None:
            path = str(config.MODEL_DIR / "quantized_model")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path=None):
        """Load student model + tokenizer from a directory at *path*."""
        if path is None:
            path = str(config.MODEL_DIR / "quantized_model")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        quantized_pt = os.path.join(path, "quantized_model.pt")
        meta_json = os.path.join(path, "quantize_meta.json")

        if os.path.exists(quantized_pt) and os.path.exists(meta_json):
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.student_model_name, num_labels=self.num_labels
            ).cpu()
            quantized_model = torch.quantization.quantize_dynamic(
                base_model, {nn.Linear}, dtype=torch.qint8
            )
            state_dict = torch.load(quantized_pt, map_location="cpu", weights_only=True)
            quantized_model.load_state_dict(state_dict)
            self.model = quantized_model
            self.device = torch.device("cpu")
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, num_labels=self.num_labels,
            ).to(self.device)


# Alias for app.py compatibility
QuantizedClassifier = DistilledQuantizedClassifier
