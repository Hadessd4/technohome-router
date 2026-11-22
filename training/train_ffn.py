"""
TechnoHome Customer Support Ticket Router
Training Script: FastText Embeddings + Feed-Forward Neural Network
"""

import os
import sys
import json
import argparse
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.feedforward_nn import FFNClassifier, TextToEmbedding


def load_dataset(data_path):
    """Load prepared_dataset.json."""
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def load_embeddings(embedding_dim, train_texts=None, seed=42):
    """
    Train or load in-domain Word2Vec embeddings on the training set.
    """
    w2v_path = config.MODEL_DIR / "word2vec.model"
    if os.path.exists(w2v_path):
        try:
            from gensim.models import Word2Vec
            w2v = Word2Vec.load(str(w2v_path))
            print(f"  Loaded existing Word2Vec embeddings from {w2v_path} ({len(w2v.wv)} words)")
            return w2v.wv
        except Exception:
            pass

    if train_texts is not None:
        from gensim.models import Word2Vec
        print("  Training Word2Vec embeddings on in-domain corpus...")
        sentences = [t.lower().split() for t in train_texts]
        w2v = Word2Vec(
            sentences=sentences,
            vector_size=embedding_dim,
            window=5,
            min_count=1,
            workers=4,
            epochs=30,
            seed=seed,
        )
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        w2v.save(str(w2v_path))
        print(f"  Saved in-domain Word2Vec to {w2v_path} ({len(w2v.wv)} words, dim={embedding_dim})")
        return w2v.wv

    return None


def texts_to_embeddings(texts, wv, embedding_dim):
    """
    Convert a list of texts to averaged word embeddings.
    If wv is None, uses random embeddings as fallback.
    """
    embeddings = []
    for text in texts:
        words = text.lower().split()
        if wv is not None:
            word_vecs = []
            for word in words:
                if word in wv:
                    word_vecs.append(wv[word])
            if word_vecs:
                embeddings.append(np.mean(word_vecs, axis=0))
            else:
                embeddings.append(np.zeros(embedding_dim))
        else:
            # Random fallback: deterministic per-text hash
            np.random.seed(hash(text) % (2**31))
            embeddings.append(np.random.randn(embedding_dim).astype(np.float32))
    return np.array(embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train FastText + FFN model")
    parser.add_argument("--data-path", type=str,
                        default=str(config.DATA_DIR / "prepared_dataset.json"),
                        help="Path to prepared_dataset.json")
    parser.add_argument("--output-path", type=str,
                        default=str(config.MODEL_DIR / "ffn_model.pt"),
                        help="Path to save trained model")
    parser.add_argument("--embedding-dim", type=int, default=config.FASTTEXT_DIM,
                        help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=config.FFN_HIDDEN_DIM,
                        help="FFN hidden layer dimension")
    parser.add_argument("--dropout", type=float, default=config.FFN_DROPOUT,
                        help="Dropout rate")
    parser.add_argument("--lr", type=float, default=config.FFN_LEARNING_RATE,
                        help="Learning rate for SGD")
    parser.add_argument("--momentum", type=float, default=config.FFN_MOMENTUM,
                        help="SGD momentum")
    parser.add_argument("--epochs", type=int, default=config.FFN_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.FFN_BATCH_SIZE,
                        help="Training batch size")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("TechnoHome FastText + FFN Training")
    print("=" * 60)

    print(f"\nLoading dataset from: {args.data_path}")
    dataset = load_dataset(args.data_path)

    train_texts = [ex["text"] for ex in dataset["train"]]
    train_labels = [config.DEPT_TO_IDX[ex["department"]] for ex in dataset["train"]]
    val_texts = [ex["text"] for ex in dataset["validation"]]
    val_labels = [config.DEPT_TO_IDX[ex["department"]] for ex in dataset["validation"]]

    print(f"  Train samples: {len(train_texts)}")
    print(f"  Validation samples: {len(val_texts)}")

    # ── Load embeddings ────────────────────────────────────────────────────
    print("\nLoading word embeddings...")
    wv = load_embeddings(args.embedding_dim, train_texts=train_texts, seed=args.seed)

    actual_dim = wv.vector_size if wv is not None else args.embedding_dim

    print("\nConverting texts to embeddings...")
    train_embeddings = texts_to_embeddings(train_texts, wv, actual_dim)
    val_embeddings = texts_to_embeddings(val_texts, wv, actual_dim)
    print(f"  Train embeddings shape: {train_embeddings.shape}")
    print(f"  Validation embeddings shape: {val_embeddings.shape}")

    # ── Prepare data loaders ───────────────────────────────────────────────
    train_X = torch.tensor(train_embeddings, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    val_X = torch.tensor(val_embeddings, dtype=torch.float32)
    val_y = torch.tensor(val_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # ── Initialize model ───────────────────────────────────────────────────
    print(f"\nInitializing FFNClassifier...")
    print(f"  input_dim={actual_dim}, hidden_dim={args.hidden_dim}, "
          f"num_classes={config.NUM_DEPARTMENTS}, dropout={args.dropout}")

    model = FFNClassifier(
        input_dim=actual_dim,
        hidden_dim=args.hidden_dim,
        num_classes=config.NUM_DEPARTMENTS,
        dropout=args.dropout,
    )

    # ── Training with SGD ──────────────────────────────────────────────────
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")

    print(f"\nTraining for {args.epochs} epochs (SGD, lr={args.lr}, momentum={args.momentum})...")
    print(f"{'Epoch':>6} | {'Loss':>10} | {'Train Acc':>10} | {'Val Acc':>10}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_X.size(0)

        avg_loss = total_loss / total_train
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X.to(device))
            val_preds = val_logits.argmax(dim=1).cpu()
            val_acc = (val_preds == val_y).float().mean().item()

        print(f"{epoch:>6} | {avg_loss:>10.4f} | {train_acc:>10.4f} | {val_acc:>10.4f}")

    # ── Save model ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    model.save(args.output_path)
    print(f"\nModel saved to: {args.output_path}")

    model_size_mb = os.path.getsize(args.output_path) / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")
    print("\nDone!")


if __name__ == "__main__":
    main()
