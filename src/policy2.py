"""
ATLAS Eviction Policy v2 -- pairwise attention regression
==========================================================
MLP: 780-d -> 128 -> 64 -> 1, trained with MSE against an attention score.

How this differs from policy.py
-------------------------------
policy.py scores ONE summary (396-d) and is trained as binary classification
against a hindsight relevance label. policy2 scores a (source, target) PAIR and
regresses the attention the model's output at that step paid to that summary.

  [0:384]     source summary embedding   -- the buffered summary being scored
  [384:768]   target summary embedding   -- what the model is writing right now
  [768]       position     source_step / POSITION_SCALE (fixed 256, absolute position)
  [769]       recency      (current_step - source_step) / current_step
  [770]       speaker      1 patient, 0 provider (source utterance)
  [771]       token_count  source summary
  [772]       entity_count source summary
  [773:780]   type one-hot source summary

The trailing 12 keep policy.py's ordering, so the source half of the vector is
positionally identical to the old 396-d layout.

Output is a raw linear score, not a logit: the training target may be a
percentile (bounded) or a z-score (unbounded), and only the ORDER matters for
eviction, so no squashing is applied. Do not call sigmoid on it.

Why the target is a within-step rank by default
-----------------------------------------------
See build_eviction_attention_dataset.py: raw attention share sums to 1 across
the buffer, so it shrinks as the buffer grows while buffer size is not an input
feature. The default label is attention_rank_pct, the percentile within the step.

Read the recency baseline before trusting a good-looking score
-------------------------------------------------------------
On this data attention is heavily recency-driven (age 1-2 sit far above average,
age 3+ is nearly flat). Evaluation therefore reports what plain recency alone
achieves on the same metric. A model that does not clearly beat that number has
learned recency with extra steps.
"""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DIM = 780
EMBED_DIM = 384
POSITION_IDX = 768
RECENCY_IDX = 769
SPEAKER_IDX = 770

# position is normalised by a FIXED constant (not current_step) so it encodes
# absolute conversation position rather than 1 - recency. MUST match
# build_eviction_attention_dataset.POSITION_SCALE, or inference feeds the model
# a different vector than it was trained on (silently wrong scores, no error).
POSITION_SCALE = 256


@dataclass
class Policy2Config:
    """Separate from PolicyConfig so the 396-d pipeline keeps working."""
    feature_dim: int = FEATURE_DIM
    hidden_1: int = 128
    hidden_2: int = 64
    output_dim: int = 1
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    patience: int = 10
    weight_decay: float = 1e-4


DEFAULT_POLICY2_CONFIG = Policy2Config()


class PairEvictionMLP(nn.Module):
    """780-d pair scorer. Raw linear output -- higher = more attended."""

    def __init__(self, config: Policy2Config = None):
        super().__init__()
        config = config or DEFAULT_POLICY2_CONFIG
        self.feature_dim = config.feature_dim
        self.network = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_1),
            nn.ReLU(),
            nn.Linear(config.hidden_1, config.hidden_2),
            nn.ReLU(),
            nn.Linear(config.hidden_2, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =============================================================================
# METRICS
# =============================================================================

def _rank(values: List[float]) -> List[float]:
    """Average ranks, so ties do not create a spurious ordering."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for pos in range(i, j + 1):
            ranks[order[pos]] = avg
        i = j + 1
    return ranks


def _pearson(a: List[float], b: List[float]) -> Optional[float]:
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0:
        return None
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / (va * vb) ** 0.5


def within_step_spearman(groups, preds, labels) -> Optional[float]:
    """
    Mean Spearman correlation computed WITHIN each step.

    This is the metric that matches what eviction does -- rank the summaries
    held at one moment -- unlike a global correlation, which is inflated by the
    easy across-step variation.
    """
    buckets = defaultdict(list)
    for g, p, y in zip(groups, preds, labels):
        buckets[g].append((p, y))

    rhos = []
    for rows in buckets.values():
        if len(rows) < 2:
            continue
        rho = _pearson(_rank([r[0] for r in rows]), _rank([r[1] for r in rows]))
        if rho is not None:
            rhos.append(rho)
    return sum(rhos) / len(rhos) if rhos else None


# =============================================================================
# DATA
# =============================================================================

def load_dataset(path: Path):
    features, labels, groups = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            features.append(row["features"])
            labels.append(row["label"])
            groups.append((row["conversation_id"], row["target_step"]))

    if not features:
        raise SystemExit(f"No examples in {path}")
    dim = len(features[0])
    if dim != FEATURE_DIM:
        raise SystemExit(f"{path} has {dim}-d features, expected {FEATURE_DIM}-d.")
    return features, labels, groups


# =============================================================================
# TRAINING
# =============================================================================

def train_policy2(train_path: Path, valid_path: Optional[Path], output_path: Path,
                  config: Policy2Config = None, seed: int = 0):
    from torch.utils.data import DataLoader, TensorDataset

    config = config or DEFAULT_POLICY2_CONFIG
    torch.manual_seed(seed)
    random.seed(seed)

    X_tr, y_tr, g_tr = load_dataset(train_path)
    print(f"train: {len(X_tr)} rows <- {train_path.name}")

    has_val = valid_path is not None and valid_path.exists()
    if has_val:
        X_va, y_va, g_va = load_dataset(valid_path)
        print(f"valid: {len(X_va)} rows <- {valid_path.name}")
        overlap = {c for c, _ in g_tr} & {c for c, _ in g_va}
        if overlap:
            print(f"  WARNING: {len(overlap)} conversation(s) appear in BOTH splits "
                  f"-- validation is leaking.")
    else:
        print("valid: none (early stopping will monitor train loss)")

    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    if has_val:
        Xva = torch.tensor(X_va, dtype=torch.float32)
        yva = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)

    model = PairEvictionMLP(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(Xtr, ytr),
                        batch_size=config.batch_size, shuffle=True)

    best_loss = float("inf")
    patience = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        for bx, by in loader:
            pred = model(bx)
            loss = criterion(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / len(loader)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xva), yva).item()
            monitor = val_loss
        else:
            val_loss = float("nan")
            monitor = train_loss

        if monitor < best_loss:
            best_loss = monitor
            patience = 0
            torch.save(model.state_dict(), output_path)
        else:
            patience += 1
            if patience >= config.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            if has_val:
                print(f"  epoch {epoch + 1:>3}: train_mse={train_loss:.5f} "
                      f"val_mse={val_loss:.5f}")
            else:
                print(f"  epoch {epoch + 1:>3}: train_mse={train_loss:.5f}")

    print(f"\nSaved -> {output_path} "
          f"({'val' if has_val else 'train'}_mse={best_loss:.5f})")

    # --- ranking quality, plus the baseline that matters --------------------
    model.load_state_dict(torch.load(output_path, map_location="cpu"))
    model.eval()

    for name, X, y, g in ([("train", Xtr, y_tr, g_tr)] +
                          ([("valid", Xva, y_va, g_va)] if has_val else [])):
        with torch.no_grad():
            preds = model(X).squeeze(-1).tolist()
        rho = within_step_spearman(g, preds, y)

        # Recency alone, negated: the most recent summary has the smallest
        # recency value but the highest attention, so -recency is the naive
        # "keep the newest" scorer this model has to beat.
        recency = [-row[RECENCY_IDX] for row in X.tolist()]
        rho_recency = within_step_spearman(g, recency, y)

        print(f"  {name}: within-step Spearman = "
              f"{'n/a' if rho is None else f'{rho:.4f}'}   "
              f"(recency-only baseline = "
              f"{'n/a' if rho_recency is None else f'{rho_recency:.4f}'})")

    return model


# =============================================================================
# INFERENCE
# =============================================================================

class LearnedPairEvictionStrategy:
    """
    Eviction strategy backed by PairEvictionMLP.

    Call contract matches baselines.py:
        strategy(items: List[BufferItem], **kwargs) -> List[BufferItem]

    The target half of each feature vector is the summary produced at the
    CURRENT step. harness.ContextBuffer.evict() runs after the new summaries
    were appended, so those are exactly the items whose timestamp equals the
    current step; they form the target and are not scored as sources.

    On a turn that produced no summary there is no target to embed -- a state
    absent from the training data, which only contains steps that emitted one.
    Rather than invent a target embedding, existing scores are left untouched.
    """

    def __init__(self, model_path: str = None, config: Policy2Config = None,
                 use_scispacy: bool = True, encoder=None):
        self.config = config or DEFAULT_POLICY2_CONFIG
        self.use_scispacy = use_scispacy
        self.model = PairEvictionMLP(self.config)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self._encoder = encoder

    @staticmethod
    def _current_step(items, kwargs) -> int:
        """
        Index of the utterance being processed right now.

        max(timestamp) is only the current step when this turn actually added a
        summary; on a turn that added nothing it points at some earlier step,
        which would mislabel old summaries as the target. So prefer the id of
        the utterance the harness passes in (line_0011 -> 11) and fall back to
        max(timestamp) only when it cannot be parsed.
        """
        utt = kwargs.get("current_utterance") or {}
        uid = utt.get("id") if isinstance(utt, dict) else None
        if isinstance(uid, str):
            digits = "".join(ch for ch in uid if ch.isdigit())
            if digits:
                return int(digits)
        return max(item.timestamp for item in items)

    def __call__(self, items: List, **kwargs) -> List:
        if not items:
            return items

        current_step = self._current_step(items, kwargs)
        targets = [i for i in items if i.timestamp == current_step]
        sources = [i for i in items if i.timestamp != current_step]

        # No summary was produced this turn -> no target to embed, a state the
        # training data never contains. Leave the existing scores alone rather
        # than score against an invented target.
        if not targets or not sources:
            return items

        encoder = self._get_encoder()
        target_vec = self._target_embedding(targets, encoder)

        features = [self._pair_features(item, target_vec, current_step, encoder)
                    for item in sources]
        with torch.no_grad():
            scores = self.model(torch.tensor(features, dtype=torch.float32)).squeeze(-1)

        for item, score in zip(sources, scores):
            item.score = score.item()

        # Never evict what the model is writing right now: put this step's own
        # summaries above every scored source.
        top = max((i.score for i in sources), default=0.0)
        for item in targets:
            item.score = top + 1.0

        return items

    def _target_embedding(self, targets, encoder):
        """Mean of this step's summary embeddings, re-normalised -- matches
        build_eviction_attention_dataset.target_embedding()."""
        vecs = [encoder.encode(t.summary, normalize_embeddings=True) for t in targets]
        if len(vecs) == 1:
            return vecs[0].tolist()
        mean = sum(vecs) / len(vecs)
        norm = float((mean ** 2).sum() ** 0.5)
        return (mean / norm).tolist() if norm > 0 else mean.tolist()

    def _pair_features(self, item, target_vec, current_step, encoder) -> List[float]:
        from policy import TYPE_TO_INDEX, get_entity_count_cached, _token_count_rough

        src = encoder.encode(item.summary, normalize_embeddings=True).tolist()

        t = max(current_step, 1)
        position = item.timestamp / POSITION_SCALE
        recency = (t - item.timestamp) / t
        speaker = 1.0 if item.speaker == "patient" else 0.0
        token_count = _token_count_rough(item.summary)
        entity_count = get_entity_count_cached(item.summary,
                                               use_scispacy=self.use_scispacy)

        onehot = [0.0] * 7
        onehot[TYPE_TO_INDEX.get(item.type, 1)] = 1.0

        return (src + list(target_vec)
                + [position, recency, speaker, token_count, entity_count]
                + onehot)

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as exc:
                # Zero embeddings would silently wreck the scores; fail loudly,
                # same as policy.LearnedEvictionStrategy.
                raise RuntimeError(
                    "LearnedPairEvictionStrategy requires the all-MiniLM-L6-v2 "
                    "encoder used to build the dataset. "
                    f"Reason: {exc}"
                ) from exc
        return self._encoder


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Train the 780-d pair eviction policy.")
    p.add_argument("--train", type=Path,
                   default=PROJECT_ROOT / "data" / "eviction_policy_attention_train.jsonl")
    p.add_argument("--valid", type=Path,
                   default=PROJECT_ROOT / "data" / "eviction_policy_attention_valid.jsonl")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "eviction_mlp" / "eviction_mlp_attention_780.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = Policy2Config(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    train_policy2(args.train, args.valid, args.output, config, seed=args.seed)


if __name__ == "__main__":
    main()
