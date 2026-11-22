"""
TechnoHome Customer Support Ticket Router — BERT (BERTurk) Fine-Tuned Classifier
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import config
from models.model_utils import Prediction


# ─── Custom Dataset ──────────────────────────────────────────────────────────

class IntentDataset(Dataset):
    """Tokenised dataset for intent / department classification."""

    def __init__(self, texts, labels, tokenizer, max_length=config.BERT_MAX_LENGTH):
        self.encodings = tokenizer(
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
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ─── Helper: compute_metrics for Trainer ─────────────────────────────────────

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}


# ─── Classifier ──────────────────────────────────────────────────────────────

class BERTIntentClassifier:
    """Fine-tune and use a BERT model for department classification.

    Default backbone: ``dbmdz/bert-base-turkish-cased`` (BERTurk).
    """

    def __init__(self, model_name=config.BERT_MODEL_NAME,
                 num_labels=config.NUM_DEPARTMENTS,
                 max_length=None, label_map=None):
        if model_name == config.BERT_MODEL_NAME:
            local_bert = config.MODEL_DIR / "bert_intent"
            if local_bert.exists():
                model_name = str(local_bert)
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length or config.BERT_MAX_LENGTH
        self.label_map = label_map or config.DEPT_TO_IDX
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels,
        ).to(self.device)

    # ── Fine-tuning ──────────────────────────────────────────────────────────

    def fine_tune(self, train_texts, train_labels, val_texts, val_labels,
                  output_dir=None, epochs=None, batch_size=None,
                  learning_rate=None, warmup_steps=None, weight_decay=0.01,
                  seed=None):
        """Fine-tune with the HuggingFace Trainer API.

        Labels can be string department names or integer indices.
        """
        if output_dir is None:
            output_dir = str(config.MODEL_DIR / "bert_checkpoints")
        epochs = epochs or config.BERT_EPOCHS
        batch_size = batch_size or config.BERT_BATCH_SIZE
        learning_rate = learning_rate or config.BERT_LEARNING_RATE
        warmup_steps = warmup_steps or config.BERT_WARMUP_STEPS
        seed = seed or config.RANDOM_SEED

        # Convert string labels to indices if needed
        if train_labels and isinstance(train_labels[0], str):
            train_labels = [self.label_map[l] for l in train_labels]
            val_labels = [self.label_map[l] for l in val_labels]

        train_dataset = IntentDataset(train_texts, train_labels,
                                       self.tokenizer, self.max_length)
        val_dataset = IntentDataset(val_texts, val_labels,
                                     self.tokenizer, self.max_length)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_steps=50,
            seed=seed,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=_compute_metrics,
        )

        trainer.train()
        self.model = trainer.model.to(self.device)

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
            max_length=self.max_length,
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
        """Save model + tokenizer to a directory at *path*."""
        if path is None:
            path = str(config.MODEL_DIR / "bert_intent")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path=None):
        """Load model + tokenizer from a directory at *path*."""
        if path is None:
            path = str(config.MODEL_DIR / "bert_intent")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=self.num_labels,
        ).to(self.device)


# Alias for app.py compatibility
BERTClassifier = BERTIntentClassifier
