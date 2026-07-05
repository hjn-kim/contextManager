"""
ATLAS Eviction Policy (Component 3)
=====================================
Lightweight MLP: 396-d -> 128 -> 64 -> 1 -> Sigmoid
Scores each summary in buffer. Low score = evict.

Input features per summary (396-d):
  [0:384]  sentence embedding (all-MiniLM-L6-v2, frozen)
  [384]    position: i / t
  [385]    recency: (t - i) / t
  [386]    speaker: 0=provider, 1=patient
  [387]    token count
  [388]    medical entity count (scispaCy)
  [389:396] type one-hot (agenda_item, detail, medication, social_history, follow_up, question_unanswered, question)
"""

import torch
import torch.nn as nn
import random
from typing import List
from config import PolicyConfig, DEFAULT_CONFIG


class EvictionMLP(nn.Module):
    """
    Small MLP scorer.
    ~1M params. Runs in <10ms on CPU.

    NOTE: outputs raw logits, NOT probabilities.
    Use torch.sigmoid() at inference time for scores in [0, 1].
    This allows BCEWithLogitsLoss with pos_weight during training.
    """

    def __init__(self, config: PolicyConfig = None):
        super().__init__()
        config = config or DEFAULT_CONFIG.policy

        self.network = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_1),
            nn.ReLU(),
            nn.Linear(config.hidden_1, config.hidden_2),
            nn.ReLU(),
            nn.Linear(config.hidden_2, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


TYPE_TO_INDEX = {
    "agenda_item": 0,
    "detail": 1,
    "medication": 2,
    "social_history": 3,
    "follow_up": 4,
    "question_unanswered": 5,
    "question": 6,
}


def extract_features(buffer_item, current_step: int, encoder=None) -> List[float]:
    if encoder:
        embedding = encoder.encode(buffer_item.summary).tolist()
    else:
        embedding = [0.0] * 384

    t = max(current_step, 1)
    position = buffer_item.timestamp / t
    recency = (t - buffer_item.timestamp) / t
    speaker = 1.0 if buffer_item.speaker == "patient" else 0.0
    token_count = float(buffer_item.token_count)
    entity_count = 0.0

    type_onehot = [0.0] * 7  # 7 types (added question)
    type_idx = TYPE_TO_INDEX.get(buffer_item.type, 1)
    type_onehot[type_idx] = 1.0

    # 384 + 5 scalar + 7 onehot = 396
    features = embedding + [position, recency, speaker, token_count, entity_count] + type_onehot
    return features


class LearnedEvictionStrategy:
    def __init__(self, model_path: str = None, config: PolicyConfig = None):
        self.config = config or DEFAULT_CONFIG.policy
        self.model = EvictionMLP(self.config)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self._encoder = None

    def __call__(self, items: List) -> List:
        if not items:
            return items
        current_step = max(item.timestamp for item in items)
        encoder = self._get_encoder()
        features = []
        for item in items:
            feat = extract_features(item, current_step, encoder)
            features.append(feat)
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            logits = self.model(x).squeeze(-1)
            scores = torch.sigmoid(logits)
        for item, score in zip(items, scores):
            item.score = score.item()
        return items

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                pass
        return self._encoder


def train_policy(data_path: str, output_path: str, config: PolicyConfig = None,
                 val_ratio: float = 0.15):
    import json
    from torch.utils.data import DataLoader, TensorDataset

    config = config or DEFAULT_CONFIG.policy

    features, labels, conv_ids = [], [], []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            features.append(d["features"])
            labels.append(d["label"])
            conv_ids.append(d.get("conversation_id", "unknown"))

    if len(features) == 0:
        print(f"ERROR: no training examples found in {data_path}.")
        return None

    unique_convs = list(set(conv_ids))
    random.shuffle(unique_convs)

    if len(unique_convs) <= 2:
        print(f"WARNING: only {len(unique_convs)} conversation(s). No validation split.")
        train_idx = list(range(len(features)))
        val_idx = []
        n_val = 0
    else:
        n_val = max(1, int(len(unique_convs) * val_ratio))
        val_convs = set(unique_convs[:n_val])
        train_idx = [i for i, c in enumerate(conv_ids) if c not in val_convs]
        val_idx = [i for i, c in enumerate(conv_ids) if c in val_convs]

    X_all = torch.tensor(features, dtype=torch.float32)
    y_all = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])

    model = EvictionMLP(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=config.batch_size, shuffle=True)

    has_val = len(val_idx) > 0
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()
            monitor_loss = val_loss
        else:
            val_loss = float("nan")
            monitor_loss = train_loss

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            if has_val:
                print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} (no val set)")

    loss_type = "val_loss" if has_val else "train_loss"
    print(f"Best model saved to {output_path} ({loss_type}={best_loss:.4f})")
    return model
