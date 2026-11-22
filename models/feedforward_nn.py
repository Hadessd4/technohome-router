"""
TechnoHome Customer Support Ticket Router — Feed-Forward Neural Network (fastText → FFN)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from models.model_utils import Prediction


# ─── PyTorch Module ──────────────────────────────────────────────────────────

class IntentFFN(nn.Module):
    """Three-layer feed-forward network for intent classification.

    Architecture
    ------------
    Linear(embedding_dim, 256) → ReLU → Dropout(0.3)
    Linear(256, 128)           → ReLU → Dropout(0.3)
    Linear(128, num_classes)
    """

    def __init__(self, embedding_dim=config.FASTTEXT_DIM,
                 hidden_dim=config.FFN_HIDDEN_DIM,
                 num_classes=config.NUM_DEPARTMENTS,
                 dropout=config.FFN_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),   # 256 → 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─── Text → Embedding Helper ────────────────────────────────────────────────

class TextToEmbedding:
    """Convert raw text to a fixed-size embedding by averaging fastText word vectors.

    Uses gensim's ``KeyedVectors`` to load pre-trained word vectors.
    """

    def __init__(self, fasttext_path=None):
        """Load fastText vectors. If path is None, uses random embeddings."""
        self.kv = None
        self.dim = config.FASTTEXT_DIM
        if fasttext_path and os.path.exists(fasttext_path):
            from gensim.models import KeyedVectors
            binary = fasttext_path.endswith(".bin")
            self.kv = KeyedVectors.load_word2vec_format(fasttext_path, binary=binary)
            self.dim = self.kv.vector_size

    def embed(self, text):
        """Return a numpy array of shape ``(dim,)`` for *text*."""
        tokens = text.lower().split()
        if self.kv is not None:
            vectors = [self.kv[w] for w in tokens if w in self.kv]
            if vectors:
                return np.mean(vectors, axis=0).astype(np.float32)
        return np.zeros(self.dim, dtype=np.float32)

    def embed_batch(self, texts):
        """Return a numpy array of shape ``(len(texts), dim)``."""
        return np.stack([self.embed(t) for t in texts])


# ─── Wrapper / Trainer ───────────────────────────────────────────────────────

class FFNClassifier(nn.Module):
    """High-level wrapper around :class:`IntentFFN`.

    Handles training with SGD + CrossEntropyLoss, single-sample prediction,
    and save / load via ``torch.save`` / ``torch.load``.
    """

    def __init__(self, input_dim=config.FASTTEXT_DIM,
                 embedding_dim=None,  # alias for input_dim
                 hidden_dim=config.FFN_HIDDEN_DIM,
                 num_classes=config.NUM_DEPARTMENTS,
                 dropout=config.FFN_DROPOUT):
        super().__init__()
        actual_dim = input_dim if embedding_dim is None else embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = IntentFFN(
            embedding_dim=actual_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        ).to(self.device)
        self.num_classes = num_classes
        self._embedding_dim = actual_dim
        self._wv = None

    def _get_word_vectors(self):
        if self._wv is None:
            w2v_path = config.MODEL_DIR / "word2vec.model"
            if os.path.exists(w2v_path):
                try:
                    from gensim.models import Word2Vec
                    w2v = Word2Vec.load(str(w2v_path))
                    self._wv = w2v.wv
                except Exception:
                    pass
        return self._wv

    def text_to_embedding(self, text):
        wv = self._get_word_vectors()
        words = text.lower().split()
        if wv is not None:
            vectors = [wv[w] for w in words if w in wv]
            if vectors:
                return np.mean(vectors, axis=0).astype(np.float32)
        # Deterministic fallback per word if word2vec not trained yet
        if words:
            vecs = []
            for w in words:
                np.random.seed(hash(w) % (2**31))
                vecs.append(np.random.randn(self._embedding_dim).astype(np.float32))
            return np.mean(vecs, axis=0).astype(np.float32)
        return np.zeros(self._embedding_dim, dtype=np.float32)

    def forward(self, x):
        return self.model(x)

    def parameters(self):
        return self.model.parameters()

    def train_model(self, embeddings, labels,
                    epochs=config.FFN_EPOCHS,
                    batch_size=config.FFN_BATCH_SIZE,
                    lr=config.FFN_LEARNING_RATE,
                    momentum=config.FFN_MOMENTUM):
        """Train the network using SGD with momentum and CrossEntropyLoss."""
        X = torch.tensor(embeddings, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=momentum
        )

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += xb.size(0)

            avg_loss = total_loss / total
            accuracy = correct / total
            print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  acc={accuracy:.4f}")

    # ── nn.Module training/eval mode forwarding ──────────────────────────────

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, text_or_embedding):
        """Predict from text string or embedding vector.

        Returns
        -------
        Prediction(label, confidence)
        """
        self.model.eval()

        # If input is a string, convert to embedding
        if isinstance(text_or_embedding, str):
            embedding = self.text_to_embedding(text_or_embedding)
            x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        elif isinstance(text_or_embedding, np.ndarray):
            x = torch.tensor(text_or_embedding, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            x = x.to(self.device)
        else:
            x = text_or_embedding
            if x.dim() == 1:
                x = x.unsqueeze(0)
            x = x.to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            idx = int(torch.argmax(probs))
            confidence = float(probs[idx])

        label = config.IDX_TO_DEPT[idx]
        return Prediction(label, confidence)

    def predict_raw(self, embedding):
        """Predict returning (label, confidence) tuple."""
        result = self.predict(embedding)
        return result["department"], result["confidence"]

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path=None):
        """Save model weights to *path*."""
        if path is None:
            path = str(config.MODEL_DIR / "ffn_model.pt")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path=None):
        """Load model weights from *path*."""
        if path is None:
            path = str(config.MODEL_DIR / "ffn_model.pt")
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)


# Alias for app.py compatibility
FastTextFFNClassifier = FFNClassifier
