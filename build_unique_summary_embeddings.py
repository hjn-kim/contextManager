"""
Deduplicate the eviction-policy feature file down to one row per summary.

Why this exists
---------------
data/eviction_policy_bertscore_0p75.jsonl has one row per (summary, timestep):
63,192 rows for only 1,932 distinct summary sentences, because every summary is
re-emitted at each later timestep it is still buffered at. The 384-d embedding
is byte-identical across those repeats (verified: 0 conflicting embeddings), so
carrying all 63k copies is pure duplication -- 556 MB for ~17 MB of content.

What is dropped, and why
------------------------
396-d layout: [0:384] embedding, 384 position, 385 recency, 386 speaker,
387 token_count, 388 entity_count, 389:396 type one-hot.

  recency   dropped (requested). = (t - timestamp)/t, so it is a property of the
            (summary, timestep) pair, not of the summary.
  position  ALSO dropped. = timestamp/t -- equally timestep-dependent, and it
            genuinely varies across repeats for 1951 of 2015 summaries. Keeping
            it would silently freeze one arbitrary timestep's value into a row
            that claims to describe the sentence itself.
  label     dropped. Hindsight relevance is per-timestep too: 937 summaries
            carry both 0 and 1 across their repeats, so there is no single
            correct value to keep.
  speaker   dropped. It describes the UTTERANCE a summary came from, not the
            sentence, so it is not a dictionary-lookup property at all: the
            same sentence is legitimately spoken by either party in different
            conversations ("Depression" as an agenda_item, etc.). Take it from
            the row you are looking up instead -- extract_oracle_attention.py
            already emits source_speaker per row.

Everything left is a pure function of the sentence text, giving a 393-d vector:
  [0:384] embedding, 384 token_count, 385 entity_count, 386:393 type one-hot.

Usage
-----
    python build_unique_summary_embeddings.py \
        --input data/eviction_policy_bertscore_0p75.jsonl \
        --output data/summary_embeddings_unique.jsonl
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT = PROJECT_ROOT / "data" / "eviction_policy_bertscore_0p75.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "summary_embeddings_unique.jsonl"

EMBED_END = 384
POSITION_IDX = 384
RECENCY_IDX = 385
SPEAKER_IDX = 386
TAIL_START = 387          # token_count, entity_count, then type one-hot


def dedup(input_path: Path, output_path: Path, key_mode: str) -> None:
    # OrderedDict so output order follows first appearance in the source file,
    # which keeps runs diffable.
    rows = OrderedDict()
    n_lines = 0
    embed_conflicts = 0
    tail_conflicts = 0

    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_lines += 1

            feats = row["features"]
            summary = row["summary"]

            key = (summary if key_mode == "text"
                   else (row["conversation_id"], row["summary_utterance_id"], summary))

            embedding = feats[:EMBED_END]
            tail = feats[TAIL_START:]          # token_count, entity_count, one-hot

            existing = rows.get(key)
            if existing is None:
                rows[key] = {
                    "summary": summary,
                    "type": row["type"],
                    "conversation_ids": [row["conversation_id"]],
                    "summary_utterance_ids": [row["summary_utterance_id"]],
                    "occurrences": 1,
                    # 393-d: embedding + token_count + entity_count + type one-hot
                    "features": embedding + tail,
                }
                continue

            existing["occurrences"] += 1
            if row["conversation_id"] not in existing["conversation_ids"]:
                existing["conversation_ids"].append(row["conversation_id"])
            if row["summary_utterance_id"] not in existing["summary_utterance_ids"]:
                existing["summary_utterance_ids"].append(row["summary_utterance_id"])

            # The dedup is only sound if the kept dims really are constant.
            # Count any disagreement instead of silently keeping the first.
            if existing["features"][:EMBED_END] != embedding:
                embed_conflicts += 1
            if existing["features"][EMBED_END:] != tail:
                tail_conflicts += 1
                existing["tail_conflict"] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    feat_len = len(next(iter(rows.values()))["features"]) if rows else 0
    conflicted = sum(1 for r in rows.values() if r.get("tail_conflict"))

    print(f"read    : {n_lines} rows from {input_path}")
    print(f"wrote   : {len(rows)} unique summaries -> {output_path}")
    print(f"features: {feat_len}-d "
          f"(384 embedding + token_count + entity_count + 7 type one-hot)")
    print(f"dropped : position (384), recency (385), speaker (386), label")
    print()
    print(f"embedding conflicts across repeats : {embed_conflicts} "
          f"({'OK' if embed_conflicts == 0 else 'INVESTIGATE'})")
    print(f"summaries with conflicting tail    : {conflicted} "
          f"({'OK' if conflicted == 0 else 'INVESTIGATE — kept dims should be '
             'a pure function of the sentence text'})")


def main():
    p = argparse.ArgumentParser(
        description="Deduplicate eviction-policy features to one row per summary.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--key", choices=["text", "occurrence"], default="text",
                   help="'text': one row per distinct sentence (default). "
                        "'occurrence': per (conversation, utterance, sentence), "
                        "which keeps the same sentence separate when it is "
                        "annotated in more than one conversation.")
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    dedup(args.input, args.output, args.key)


if __name__ == "__main__":
    main()
