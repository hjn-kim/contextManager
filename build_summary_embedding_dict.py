"""
Build a sentence -> feature dictionary from the annotated conversations.

Why not reuse data/eviction_policy_bertscore_0p75.jsonl
------------------------------------------------------
That file (built 2026-07-05) can be deduplicated into the same shape, but it is
stale: data/train/*.json was re-annotated on 2026-07-18, and 7 conversations
changed (D2N022, D2N034, D2N038, D2N051, D2N056, D2N057, D2N059). 111 of the
sentences the current annotations produce are simply absent from it, so using
it as a lookup table raises KeyError on those. This script encodes the current
annotations directly, so the dictionary always matches the data on disk.

Layout (393-d, a pure function of the sentence text)
----------------------------------------------------
  [0:384]   sentence embedding      all-MiniLM-L6-v2, L2-normalised
  [384]     token_count             same rough char//4 estimate as training
  [385]     entity_count            scispaCy en_core_sci_sm entity count
  [386:393] type one-hot            7 types, policy.TYPE_TO_INDEX order

Deliberately absent -- these describe an occurrence, not a sentence, and belong
to the row you are looking up rather than to the dictionary:
  position, recency  timestep-dependent; derive from source_step / current_step
  speaker            the same sentence is legitimately spoken by either party
                     in different conversations; extract_oracle_attention.py
                     emits source_speaker per row
  label              hindsight relevance is per-timestep

Usage
-----
    python build_summary_embedding_dict.py
    python build_summary_embedding_dict.py --compare data/eviction_policy_bertscore_0p75.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from extract_oracle_attention import load_conversations  # noqa: E402

DEFAULT_DATA = PROJECT_ROOT / "data" / "train"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "summary_embedding_dict.jsonl"

EMBED_MODEL = "all-MiniLM-L6-v2"
SCISPACY_MODEL = "en_core_sci_sm"


def collect_summaries(data_dir: Path):
    """{summary_text: {"type": str, "conversation_ids": [...], "occurrences": n}}"""
    found = {}
    for conv in load_conversations(data_dir):
        for utt in conv["utterances"]:
            for detail in utt["gold_details"]:
                text = detail["summary"]
                entry = found.get(text)
                if entry is None:
                    found[text] = {
                        "type": detail["type"],
                        "conversation_ids": [conv["id"]],
                        "occurrences": 1,
                        "type_conflict": False,
                    }
                    continue
                entry["occurrences"] += 1
                if conv["id"] not in entry["conversation_ids"]:
                    entry["conversation_ids"].append(conv["id"])
                # Same sentence annotated with a different type in another
                # conversation -> the one-hot is not well defined. Flag it.
                if detail["type"] != entry["type"]:
                    entry["type_conflict"] = True
    return found


def build(data_dir: Path, output_path: Path, compare_path: Path = None) -> None:
    print(f"Reading {data_dir} ...")
    summaries = collect_summaries(data_dir)
    texts = sorted(summaries)
    print(f"  {len(texts)} distinct summary sentences")

    # --- embeddings ---------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    print(f"Encoding with {EMBED_MODEL} ...")
    encoder = SentenceTransformer(EMBED_MODEL)
    embeddings = encoder.encode(
        texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    )

    # --- entity counts ------------------------------------------------------
    entity_counts = None
    try:
        import spacy
        print(f"Counting entities with {SCISPACY_MODEL} ...")
        nlp = spacy.load(SCISPACY_MODEL)
        entity_counts = [float(len(doc.ents)) for doc in nlp.pipe(texts, batch_size=64)]
    except Exception as exc:
        # Refuse to write a silently-zero column: checkmodel.py treats an
        # all-zero entity_count as a hard failure, and a dictionary that looks
        # complete but has a dead feature is worse than none.
        raise SystemExit(
            f"Could not load {SCISPACY_MODEL} ({type(exc).__name__}: {exc}).\n"
            f"entity_count would be 0.0 for every sentence. Install the model "
            f"or rerun once it is available."
        )

    # --- type one-hot, matching the training feature layout -----------------
    from policy import TYPE_TO_INDEX
    from context_manager_labels import get_type_onehot_dim, token_count_rough

    onehot_dim = get_type_onehot_dim()
    print(f"  type one-hot dim = {onehot_dim}")

    rows = []
    type_conflicts = 0
    for i, text in enumerate(texts):
        meta = summaries[text]

        onehot = [0.0] * onehot_dim
        idx = TYPE_TO_INDEX.get(meta["type"], TYPE_TO_INDEX.get("detail", 0))
        if idx >= onehot_dim:
            idx = min(TYPE_TO_INDEX.get("detail", 0), onehot_dim - 1)
        onehot[idx] = 1.0

        if meta["type_conflict"]:
            type_conflicts += 1

        rows.append({
            "summary": text,
            "type": meta["type"],
            "conversation_ids": meta["conversation_ids"],
            "occurrences": meta["occurrences"],
            "type_conflict": meta["type_conflict"],
            "features": (
                embeddings[i].tolist()
                + [float(token_count_rough(text)), entity_counts[i]]
                + onehot
            ),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dim = len(rows[0]["features"])
    print()
    print(f"Wrote {len(rows)} rows ({dim}-d) -> {output_path}")
    print(f"  [0:384] embedding  [384] token_count  [385] entity_count  "
          f"[386:{dim}] type one-hot")
    print(f"  sentences typed inconsistently across conversations: {type_conflicts}")
    zero_ents = sum(1 for r in rows if r["features"][385] == 0.0)
    print(f"  entity_count == 0 for {zero_ents}/{len(rows)} sentences "
          f"({'OK' if zero_ents < len(rows) else 'scispaCy NOT working'})")

    if compare_path:
        compare(rows, compare_path)


def compare(rows, compare_path: Path) -> None:
    """Cross-check the shared sentences against the old feature file."""
    import numpy as np

    print(f"\nComparing against {compare_path.name} ...")
    old = {}
    with compare_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            old.setdefault(r["summary"], r["features"])

    new = {r["summary"]: r["features"] for r in rows}
    shared = set(new) & set(old)
    print(f"  sentences: new={len(new)}  old={len(old)}  shared={len(shared)}")
    print(f"  only in new (were missing before): {len(set(new) - set(old))}")
    print(f"  only in old (stale annotations)  : {len(set(old) - set(new))}")

    if not shared:
        return

    emb_delta, tok_mismatch, ent_mismatch = [], 0, 0
    for text in shared:
        n, o = new[text], old[text]
        emb_delta.append(float(np.abs(np.array(n[:384]) - np.array(o[:384])).max()))
        if n[384] != o[387]:      # token_count: new dim 384 vs old dim 387
            tok_mismatch += 1
        if n[385] != o[388]:      # entity_count: new dim 385 vs old dim 388
            ent_mismatch += 1

    print(f"  embedding max|diff| : {max(emb_delta):.2e} "
          f"({'identical encoder' if max(emb_delta) < 1e-4 else 'ENCODER DIFFERS'})")
    print(f"  token_count mismatches : {tok_mismatch}/{len(shared)}")
    print(f"  entity_count mismatches: {ent_mismatch}/{len(shared)}")


def main():
    p = argparse.ArgumentParser(
        description="Build a sentence -> 393-d feature dictionary from data/train.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--compare", type=Path, default=None,
                   help="Old feature jsonl to cross-check shared sentences against.")
    args = p.parse_args()
    build(args.data, args.output, args.compare)


if __name__ == "__main__":
    main()
