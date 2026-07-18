"""
Assemble the 780-d attention-regression dataset for the eviction policy.

Inputs
------
  data/summary_embedding_dict.jsonl    sentence -> 393-d, from build_summary_embedding_dict.py
  output/oracle_attention_pairs.jsonl  (source summary -> target summary) pairs
                                       with attention labels, from
                                       extract_oracle_attention.py

Feature layout (780-d)
----------------------
  [0:384]     source summary embedding
  [384:768]   target summary embedding
  [768]       position      source_step / current_step
  [769]       recency       (current_step - source_step) / current_step
  [770]       speaker       1 patient, 0 provider -- of the SOURCE utterance
  [771]       token_count   of the source summary
  [772]       entity_count  of the source summary
  [773:780]   type one-hot  of the source summary

The trailing 12 dims keep the exact order the existing 396-d pipeline uses
(context_manager_labels.extract_features_cached), so a model trained here reads
its source features in the same positions as the current policy does. position,
recency and speaker come from the pair row -- they describe the occurrence, not
the sentence -- while the remaining 9 come straight from the dictionary.

Label
-----
Default is attention_rank_pct: the source summary's percentile rank among the
summaries buffered at that same step, 0 = least attended .. 1 = most attended.
Raw attention_share is not a sound regression target because it sums to 1 across
the buffer, so identical importance yields wildly different values depending on
buffer size -- and buffer size is not among the 780 inputs. Ranking is also the
quantity eviction actually needs: which of the summaries I hold right now is
least used. --label lets you pick a different column.

Splitting
---------
Train/valid are split BY CONVERSATION. Splitting by row would put rows from the
same conversation -- often the very same source summary at adjacent timesteps --
on both sides, which leaks and makes validation loss meaningless.

Usage
-----
    python build_eviction_attention_dataset.py
    python build_eviction_attention_dataset.py --label attention_z --valid-frac 0.2
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DICT = PROJECT_ROOT / "data" / "summary_embedding_dict.jsonl"
DEFAULT_PAIRS = PROJECT_ROOT / "output" / "oracle_attention_pairs.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data"
DEFAULT_PREFIX = "eviction_policy_attention"

EMBED_DIM = 384
# Same separator extract_oracle_attention.py joins a step's summaries with.
JOINER = " | "
LABEL_CHOICES = ["attention_rank_pct", "attention_z", "attention_share",
                 "attention_log_share", "attention_mean"]


def load_dict(path: Path) -> dict:
    table = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            table[row["summary"]] = row["features"]
    if not table:
        raise SystemExit(f"Empty dictionary: {path}")
    dim = len(next(iter(table.values())))
    if dim != 393:
        raise SystemExit(f"Expected 393-d dictionary features, got {dim}-d in {path}")
    return table


def target_embedding(row, table):
    """
    Embedding of what the model was writing at this step.

    149 steps emit more than one summary, and attention was measured over that
    step's whole output, so the target is all of them together. The joined
    string is not an annotated sentence and is absent from the dictionary, so
    the members are looked up individually and averaged, then re-normalised to
    unit length to match every other embedding in the feature vector.

    Returns None if any member is missing from the dictionary.
    """
    # Pair files written before target_summaries was added carry only the
    # joined string. Splitting it back is exact here: no annotated sentence
    # contains " | " (verified against the dictionary), so this recovers those
    # rows instead of forcing a re-run of the GPU extraction.
    members = row.get("target_summaries")
    if not members:
        members = [s for s in row["target_summary"].split(JOINER) if s]
    vectors = []
    for text in members:
        feats = table.get(text)
        if feats is None:
            return None
        vectors.append(feats[:EMBED_DIM])

    if len(vectors) == 1:
        return vectors[0]

    mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(EMBED_DIM)]
    norm = sum(x * x for x in mean) ** 0.5
    return [x / norm for x in mean] if norm > 0 else mean


def build_features(src_feats, tgt_feats, row) -> list:
    """780-d: source embedding + target embedding + 12 occurrence/source dims."""
    t = max(row["current_step"], 1)
    position = row["source_step"] / t
    recency = (row["current_step"] - row["source_step"]) / t
    speaker = 1.0 if row["source_speaker"] == "patient" else 0.0

    # Dictionary layout: [0:384] embedding, [384] token_count,
    # [385] entity_count, [386:393] type one-hot.
    token_count = src_feats[384]
    entity_count = src_feats[385]
    type_onehot = src_feats[386:]

    return (
        src_feats[:EMBED_DIM]
        + tgt_feats
        + [position, recency, speaker, token_count, entity_count]
        + type_onehot
    )


def main():
    p = argparse.ArgumentParser(
        description="Build the 780-d attention-regression dataset.")
    p.add_argument("--dict", type=Path, default=DEFAULT_DICT)
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--prefix", default=DEFAULT_PREFIX,
                   help="Output files are <prefix>_train.jsonl / _valid.jsonl")
    p.add_argument("--label", choices=LABEL_CHOICES, default="attention_rank_pct")
    p.add_argument("--valid-frac", type=float, default=0.15,
                   help="Fraction of CONVERSATIONS held out (0 = no split).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-unrankable", action="store_true",
                   help="Keep steps holding a single summary. Their share is "
                        "always 1.0 and there is nothing to rank, so by default "
                        "they are dropped as label noise.")
    args = p.parse_args()

    for path in (args.dict, args.pairs):
        if not path.exists():
            raise SystemExit(f"Not found: {path}")

    print(f"dictionary <- {args.dict}")
    table = load_dict(args.dict)
    print(f"  {len(table)} sentences")

    print(f"pairs      <- {args.pairs}")
    examples, missing_src, missing_tgt, dropped_unrankable = [], set(), set(), 0

    with args.pairs.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if not args.keep_unrankable and not row.get("rankable", True):
                dropped_unrankable += 1
                continue

            src = table.get(row["source_summary"])
            if src is None:
                missing_src.add(row["source_summary"])
                continue

            tgt = target_embedding(row, table)
            if tgt is None:
                missing_tgt.add(row["target_summary"])
                continue

            examples.append({
                "conversation_id": row["conversation_id"],
                "source_summary_id": row["source_summary_id"],
                "source_utterance_id": row["source_utterance_id"],
                "target_utterance_id": row["target_utterance_id"],
                "target_step": row["target_step"],
                "source_summary": row["source_summary"],
                "target_summary": row["target_summary"],
                "features": build_features(src, tgt, row),
                "label": row[args.label],
                "label_name": args.label,
            })

    print(f"  {len(examples)} examples built, label = {args.label}")
    if dropped_unrankable:
        print(f"  dropped {dropped_unrankable} unrankable rows (buffer of 1)")

    # A target summary is produced at the same step it is scored, so it is a
    # genuine sentence too -- a miss here means the dictionary is out of date.
    if missing_src or missing_tgt:
        print(f"  WARNING: {len(missing_src)} source and {len(missing_tgt)} target "
              f"sentences absent from the dictionary; those rows were skipped.")
        print(f"           Rebuild it with build_summary_embedding_dict.py.")
        for s in list(missing_src | missing_tgt)[:3]:
            print(f"             {s[:80]}")

    if not examples:
        raise SystemExit("No examples built.")

    dim = len(examples[0]["features"])
    if dim != 780:
        raise SystemExit(f"Expected 780-d features, built {dim}-d.")

    # --- split by conversation ---------------------------------------------
    conv_ids = sorted({e["conversation_id"] for e in examples})
    rng = random.Random(args.seed)
    rng.shuffle(conv_ids)
    n_valid = int(round(len(conv_ids) * args.valid_frac))
    valid_convs = set(conv_ids[:n_valid])

    train = [e for e in examples if e["conversation_id"] not in valid_convs]
    valid = [e for e in examples if e["conversation_id"] in valid_convs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, rows in (("train", train), ("valid", valid)):
        if not rows:
            continue
        path = args.out_dir / f"{args.prefix}_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        written.append((path, rows))

    print()
    print(f"feature dim : {dim}")
    print(f"conversations: {len(conv_ids)} total, {len(valid_convs)} held out")
    for path, rows in written:
        labels = [r["label"] for r in rows]
        n = len(labels)
        mean = sum(labels) / n
        var = sum((x - mean) ** 2 for x in labels) / n
        print(f"  {path.name:<45} {n:>6} rows  "
              f"label mean={mean:.4f} std={var ** 0.5:.4f} "
              f"min={min(labels):.4f} max={max(labels):.4f}")

    types = Counter()
    for e in examples:
        onehot = e["features"][773:780]
        types[onehot.index(1.0) if 1.0 in onehot else -1] += 1
    print(f"  source type distribution (one-hot index): {dict(sorted(types.items()))}")

    print()
    print("Next: PolicyConfig.feature_dim 396 -> 780, and train with a regression "
          "loss (MSE) instead of BCEWithLogitsLoss -- policy.train_policy() is "
          "written for binary labels.")


if __name__ == "__main__":
    main()
