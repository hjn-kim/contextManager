"""
ATLAS Context Manager Label Generator
=====================================

Generates MLP training data for Component 3 / Context Manager.

Input:
  Annotated conversation JSON file(s)

Output:
  JSONL rows with:
    - features: fixed-length vector for one summary in the context buffer
    - label: hindsight relevance label, 1=keep, 0=evict

Main improvements in this version:
  1. BERTScorer is loaded once and reused.
  2. Cosine/BERTScore similarity matrix is precomputed once per conversation.
  3. entity_count is computed with cached scispaCy when available.
  4. Examples are streamed directly to JSONL instead of being accumulated first.

Feature dimension:
  This file automatically follows DEFAULT_CONFIG.policy.feature_dim.

  If config.py has:
    feature_dim = 396
  then features are:
    384 embedding + 5 scalar features + 7 type one-hot

  If config.py has:
    feature_dim = 395
  then features are:
    384 embedding + 5 scalar features + 6 type one-hot

This is intentional so the label generator stays consistent with policy.py/config.py.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from config import DEFAULT_CONFIG
from harness import BufferItem
from policy import TYPE_TO_INDEX


# =============================================================================
# CONSTANTS / GLOBAL CACHES
# =============================================================================

EMBEDDING_DIM = 384
SCALAR_FEATURE_DIM = 5  # position, recency, speaker, token_count, entity_count

SUMMARY_EMBED_CACHE: Dict[str, np.ndarray] = {}
ENTITY_COUNT_CACHE: Dict[str, float] = {}

_NLP = None
_NLP_LOAD_ATTEMPTED = False
_FEATURE_DIM_WARNING_PRINTED = False


# =============================================================================
# FILE / BASIC HELPERS
# =============================================================================

def iter_json_files(input_path: str) -> Iterable[Path]:
    """
    Yield JSON files from either a single file path or a directory path.
    Directory traversal is recursive and sorted for deterministic processing.
    """
    path = Path(input_path)

    if path.is_file():
        yield path
        return

    for p in sorted(path.rglob("*.json")):
        yield p


def token_count_rough(text: str) -> int:
    """
    Rough token-count convention matching the current harness.py placeholder.

    Final budget-sensitive experiments should eventually replace this with
    the actual LLM tokenizer.
    """
    return max(len(text) // 4, 1)


def normalize_speaker(speaker: str) -> str:
    """
    Normalize speaker labels to the feature convention:
      provider -> 0
      patient  -> 1
    """
    s = (speaker or "").strip().lower()

    if s in {"patient", "pt", "participant"}:
        return "patient"

    return "provider"


def get_type_onehot_dim() -> int:
    """
    Derive one-hot dimension from config.py.

    feature_dim = 384 embedding + 5 scalar + type_onehot_dim
    """
    feature_dim = getattr(DEFAULT_CONFIG.policy, "feature_dim", 396)
    onehot_dim = feature_dim - EMBEDDING_DIM - SCALAR_FEATURE_DIM

    if onehot_dim <= 0:
        onehot_dim = len(TYPE_TO_INDEX)

    return onehot_dim


def warn_once_feature_dim(actual_dim: int):
    """
    Warn once if generated feature length does not match config.py.
    """
    global _FEATURE_DIM_WARNING_PRINTED

    expected_dim = getattr(DEFAULT_CONFIG.policy, "feature_dim", actual_dim)

    if actual_dim != expected_dim and not _FEATURE_DIM_WARNING_PRINTED:
        print(
            f"WARNING: generated feature_dim={actual_dim}, "
            f"but DEFAULT_CONFIG.policy.feature_dim={expected_dim}. "
            f"Check config.py and policy.py consistency.",
            file=sys.stderr,
            flush=True,
        )
        _FEATURE_DIM_WARNING_PRINTED = True


# =============================================================================
# EMBEDDING / ENTITY FEATURES
# =============================================================================

def get_summary_embedding(summary: str, encoder) -> np.ndarray:
    """
    Return cached sentence embedding for one summary.

    Uses normalized all-MiniLM-L6-v2 embeddings by default.
    Shape: (384,)
    """
    key = (summary or "").strip()

    if key not in SUMMARY_EMBED_CACHE:
        emb = encoder.encode(
            [key],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        SUMMARY_EMBED_CACHE[key] = emb.astype(np.float32)

    return SUMMARY_EMBED_CACHE[key]


def _get_scispacy_nlp():
    """
    Lazy-load scispaCy model.

    If scispaCy or en_core_sci_sm is unavailable, returns None and the code
    falls back to entity_count = 0.0.
    """
    global _NLP, _NLP_LOAD_ATTEMPTED

    if _NLP_LOAD_ATTEMPTED:
        return _NLP

    _NLP_LOAD_ATTEMPTED = True

    try:
        import spacy
        _NLP = spacy.load("en_core_sci_sm")
        print("Loaded scispaCy model: en_core_sci_sm", flush=True)
    except Exception as exc:
        _NLP = None
        print(
            "WARNING: scispaCy model en_core_sci_sm is unavailable. "
            "entity_count will be 0.0. "
            f"Reason: {exc}",
            file=sys.stderr,
            flush=True,
        )

    return _NLP


def get_entity_count_cached(text: str, use_scispacy: bool = True) -> float:
    """
    Return cached medical entity count.

    If use_scispacy=False or scispaCy is unavailable, returns 0.0.
    """
    if not use_scispacy:
        return 0.0

    key = (text or "").strip()

    if key in ENTITY_COUNT_CACHE:
        return ENTITY_COUNT_CACHE[key]

    nlp = _get_scispacy_nlp()

    if nlp is None:
        ENTITY_COUNT_CACHE[key] = 0.0
        return 0.0

    doc = nlp(key)
    count = float(len(doc.ents))
    ENTITY_COUNT_CACHE[key] = count
    return count


def extract_features_cached(
    buffer_item: BufferItem,
    current_step: int,
    encoder,
    use_scispacy: bool = True,
) -> List[float]:
    """
    Extract feature vector for one BufferItem.

    Feature layout:
      [0:384]    sentence embedding
      [384]      position = item.timestamp / current_step
      [385]      recency = (current_step - item.timestamp) / current_step
      [386]      speaker = 0 provider, 1 patient
      [387]      token_count
      [388]      entity_count
      [389:]     type one-hot

    The type one-hot length is derived from DEFAULT_CONFIG.policy.feature_dim.
    """
    # The only per-timestep-varying features are position and recency (2 scalars).
    # Everything else (embedding, speaker, token_count, entity_count, type one-hot)
    # is constant for this item, so build it once and cache it on the item.
    const = getattr(buffer_item, "_feat_const", None)

    if const is None:
        embedding = get_summary_embedding(buffer_item.summary, encoder).tolist()
        speaker = 1.0 if buffer_item.speaker == "patient" else 0.0
        token_count = float(buffer_item.token_count)
        entity_count = get_entity_count_cached(
            buffer_item.summary, use_scispacy=use_scispacy
        )

        onehot_dim = get_type_onehot_dim()
        type_onehot = [0.0] * onehot_dim

        raw_type_idx = TYPE_TO_INDEX.get(buffer_item.type, TYPE_TO_INDEX.get("detail", 0))

        if raw_type_idx >= onehot_dim:
            # Example: config is 395-d / 6 types, but policy.TYPE_TO_INDEX includes "question": 6.
            # Fall back to "detail" rather than crashing.
            raw_type_idx = min(TYPE_TO_INDEX.get("detail", 0), onehot_dim - 1)

        type_onehot[raw_type_idx] = 1.0

        const = {
            "embedding": embedding,
            "tail": [speaker, token_count, entity_count] + type_onehot,
        }
        setattr(buffer_item, "_feat_const", const)

    t = max(current_step, 1)
    position = float(buffer_item.timestamp) / t
    recency = float(t - buffer_item.timestamp) / t

    # Order preserved: embedding + [position, recency, speaker, token, entity] + onehot
    features = const["embedding"] + [position, recency] + const["tail"]

    warn_once_feature_dim(len(features))
    return features


# =============================================================================
# ANNOTATION PARSING
# =============================================================================

def annotation_details(annotation: dict) -> List[Dict[str, str]]:
    """
    Convert supported annotation shapes into a list of {type, summary} pairs.

    Supported shapes:
      1. {"relevant": bool, "type": str, "summary": str}
      2. {"relevant": bool, "detail_list": [{"type": str, "summary": str}, ...]}
    """
    if not annotation.get("relevant", False):
        return []

    if annotation.get("detail_list"):
        out = []

        for d in annotation["detail_list"]:
            summary = (d.get("summary") or "").strip()

            if not summary:
                continue

            out.append({
                "type": d.get("type") or "detail",
                "summary": summary,
            })

        return out

    summary = (annotation.get("summary") or "").strip()

    if not summary:
        return []

    return [{
        "type": annotation.get("type") or "detail",
        "summary": summary,
    }]


def build_annotation_entries(
    conversation: dict,
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    """
    Flatten conversation annotations into entry rows.

    Each entry corresponds to one clinically relevant summary.
    If one utterance has multiple detail_list entries, each becomes one entry.

    Returns:
      entries:
        sorted list of annotation entries
      by_utterance:
        utterance_id -> list of entries
    """
    utterances = conversation.get("utterances", [])

    utt_index = {
        u["id"]: i
        for i, u in enumerate(utterances)
        if "id" in u
    }

    speaker_by_id = {
        u["id"]: normalize_speaker(u.get("speaker", ""))
        for u in utterances
        if "id" in u
    }

    entries: List[dict] = []
    by_utterance: Dict[str, List[dict]] = defaultdict(list)

    for ann in conversation.get("annotations", []):
        uid = ann.get("utterance_id")

        if uid not in utt_index:
            continue

        for d in annotation_details(ann):
            entry = {
                "conversation_id": conversation.get("id", "unknown"),
                "utterance_id": uid,
                "t": utt_index[uid],
                "speaker": speaker_by_id.get(uid, "provider"),
                "type": d["type"],
                "summary": d["summary"],
            }
            entries.append(entry)

    entries.sort(key=lambda x: (x["t"], x["utterance_id"], x["summary"]))

    # Assign stable entry index after sorting.
    for idx, entry in enumerate(entries):
        entry["entry_index"] = idx
        by_utterance[entry["utterance_id"]].append(entry)

    return entries, by_utterance


# =============================================================================
# SIMILARITY COMPUTATION
# =============================================================================

def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix.

    Returns:
      shape = (len(a), len(b))
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)

    return a @ b.T


def build_cosine_similarity_matrix(entries: List[dict], encoder) -> np.ndarray:
    """
    Precompute all entry-to-entry cosine similarities once per conversation.
    """
    if not entries:
        return np.zeros((0, 0), dtype=np.float32)

    embs = []

    for e in entries:
        emb = get_summary_embedding(e["summary"], encoder)
        e["embedding"] = emb
        embs.append(emb)

    entry_embs = np.stack(embs, axis=0)
    sim = cosine_matrix(entry_embs, entry_embs)
    return sim.astype(np.float32)


def create_bertscorer():
    """
    Create BERTScorer once.

    This avoids repeatedly loading microsoft/deberta-xlarge-mnli.
    """
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise RuntimeError(
            "bert-score is not installed. "
            "Install it or use --label-method cosine."
        ) from exc

    scorer = BERTScorer(
        model_type="microsoft/deberta-xlarge-mnli",
        lang="en",
        rescale_with_baseline=False,
    )

    # bert-score passes tokenizer.model_max_length straight into the Rust
    # tokenizer's enable_truncation(). DeBERTa ships a huge sentinel
    # model_max_length (~1e30), which overflows newer `tokenizers`
    # ("OverflowError: int too big to convert"). Clamp it to DeBERTa's real
    # max positions (512). Summaries are far shorter, so no score changes.
    tok = getattr(scorer, "_tokenizer", None)
    if tok is not None:
        max_len = getattr(tok, "model_max_length", None)
        if max_len is None or max_len > 1_000_000:
            tok.model_max_length = 512

    return scorer


def build_bertscore_similarity_matrix(
    entries: List[dict],
    bertscorer,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Precompute all entry-to-entry BERTScore F1 values once per conversation.

    sim[i, j] = BERTScore_F1(entries[i].summary, entries[j].summary)

    This is much faster than calling bert_score.score() once per
    buffer item per timestep, because the DeBERTa scorer is reused and
    all pairs are batched.
    """
    n = len(entries)

    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    candidates: List[str] = []
    references: List[str] = []
    pair_indices: List[Tuple[int, int]] = []

    # Only pairs (i, j) with entries[j].t > entries[i].t are ever queried:
    # an item is labeled only at timesteps AFTER it is added (current_t > t_i),
    # and lookups use future entries (t_j >= current_t), so t_j > t_i always.
    # Skipping the diagonal and past pairs roughly halves the DeBERTa forward
    # passes without changing any label (unqueried cells stay 0.0).
    for i, cand_entry in enumerate(entries):
        cand = cand_entry["summary"]
        t_i = cand_entry["t"]

        for j, ref_entry in enumerate(entries):
            if ref_entry["t"] <= t_i:
                continue
            candidates.append(cand)
            references.append(ref_entry["summary"])
            pair_indices.append((i, j))

    sim = np.zeros((n, n), dtype=np.float32)

    for start in range(0, len(candidates), batch_size):
        end = start + batch_size

        batch_cands = candidates[start:end]
        batch_refs = references[start:end]
        batch_pairs = pair_indices[start:end]

        _, _, f1 = bertscorer.score(
            batch_cands,
            batch_refs,
            batch_size=batch_size,
            verbose=False,
        )

        f1_values = f1.detach().cpu().numpy().astype(np.float32)

        for (i, j), value in zip(batch_pairs, f1_values):
            sim[i, j] = value

    return sim


def build_similarity_state(
    entries: List[dict],
    encoder,
    label_method: str,
    bertscorer=None,
    bertscore_batch_size: int = 16,
) -> dict:
    """
    Build per-conversation similarity state.

    Returns:
      {
        "entry_ts": np.ndarray shape (n,),
        "sim_matrix": np.ndarray shape (n, n),
      }
    """
    entry_ts = np.array([e["t"] for e in entries], dtype=np.int32)

    if label_method == "bertscore":
        if bertscorer is None:
            raise ValueError("bertscorer must be provided when label_method='bertscore'.")

        sim_matrix = build_bertscore_similarity_matrix(
            entries=entries,
            bertscorer=bertscorer,
            batch_size=bertscore_batch_size,
        )
    else:
        sim_matrix = build_cosine_similarity_matrix(
            entries=entries,
            encoder=encoder,
        )

    return {
        "entry_ts": entry_ts,
        "sim_matrix": sim_matrix,
    }


def max_future_similarity(
    item: BufferItem,
    sim_matrix: np.ndarray,
    future_indices: np.ndarray,
) -> float:
    """
    Compute:
      max similarity between buffer item summary and all future annotation summaries.

    Uses precomputed sim_matrix instead of recomputing cosine/BERTScore.

    future_indices depends only on the current timestep, not on the item, so it
    is computed once per timestep by the caller and passed in here rather than
    being recomputed for every buffer item.
    """
    if sim_matrix.size == 0:
        return 0.0

    item_idx = getattr(item, "entry_index", None)

    if item_idx is None:
        raise RuntimeError(
            "BufferItem is missing entry_index. "
            "This should be set when adding annotation entries to the buffer."
        )

    if len(future_indices) == 0:
        return 0.0

    return float(sim_matrix[item_idx, future_indices].max())


# =============================================================================
# EXAMPLE GENERATION
# =============================================================================

def make_buffer_item_from_entry(entry: dict) -> BufferItem:
    """
    Convert one annotation entry into a BufferItem.
    Attaches entry_index dynamically for similarity matrix lookup.
    """
    item = BufferItem(
        summary=entry["summary"],
        type=entry["type"],
        speaker=entry["speaker"],
        utterance_id=entry["utterance_id"],
        timestamp=entry["t"],
        token_count=token_count_rough(entry["summary"]),
    )

    # BufferItem dataclass has no entry_index field, but it does not use slots.
    # Attaching it here avoids modifying harness.py.
    setattr(item, "entry_index", entry["entry_index"])

    return item


def generate_examples_for_conversation_iter(
    conversation: dict,
    encoder,
    threshold: float,
    label_method: str,
    bertscorer=None,
    bertscore_batch_size: int = 16,
    use_scispacy: bool = True,
) -> Iterator[dict]:
    """
    Stream examples for one conversation.

    Important order:
      1. At timestep t, label all prior summaries already in buffer.
      2. Then add current utterance's ground-truth annotations to buffer.

    This matches hindsight-supervision replay semantics.
    """
    entries, by_utterance = build_annotation_entries(conversation)

    similarity_state = build_similarity_state(
        entries=entries,
        encoder=encoder,
        label_method=label_method,
        bertscorer=bertscorer,
        bertscore_batch_size=bertscore_batch_size,
    )

    buffer: List[BufferItem] = []
    conversation_id = conversation.get("id", "unknown")

    sim_matrix = similarity_state["sim_matrix"]
    entry_ts = similarity_state["entry_ts"]

    for t, utt in enumerate(conversation.get("utterances", [])):
        current_uid = utt.get("id")

        # future_indices depends only on t (not on the item), so compute once
        # per timestep and reuse it for every buffer item below.
        future_indices = np.where(entry_ts >= t)[0]

        # Label every prior summary BEFORE adding current annotation.
        for item in buffer:
            features = extract_features_cached(
                buffer_item=item,
                current_step=t,
                encoder=encoder,
                use_scispacy=use_scispacy,
            )

            max_sim = max_future_similarity(
                item=item,
                sim_matrix=sim_matrix,
                future_indices=future_indices,
            )

            label = 1 if max_sim > threshold else 0

            yield {
                "conversation_id": conversation_id,
                "current_utterance_id": current_uid,
                "summary_utterance_id": item.utterance_id,
                "timestep": t,
                "summary": item.summary,
                "type": item.type,
                "max_future_similarity": max_sim,
                "features": features,
                "label": label,
            }

        # Add current ground-truth annotations to buffer after labeling.
        for entry in by_utterance.get(current_uid, []):
            buffer.append(make_buffer_item_from_entry(entry))


# =============================================================================
# RESUME SUPPORT
# =============================================================================

def load_resume_state(output_path: Path) -> Tuple[set, str]:
    """
    Resume behavior:
      - If output exists, find the last conversation_id.
      - Remove all rows for that last conversation_id because it may be partial.
      - Mark earlier conversation_ids as processed.
      - Return processed_conversation_ids and file mode "a".

    If resume cannot be applied, returns empty set and "w".
    """
    processed_conversation_ids = set()

    if not output_path.exists():
        return processed_conversation_ids, "w"

    existing_lines: List[str] = []

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                existing_lines.append(line)

    resume_id = None

    for line in reversed(existing_lines):
        try:
            cid = json.loads(line).get("conversation_id")
            if cid:
                resume_id = cid
                break
        except Exception:
            pass

    if not resume_id:
        print("Resume: output file is empty or unreadable. Starting fresh.", flush=True)
        return processed_conversation_ids, "w"

    cleaned_lines: List[str] = []
    resume_count = 0

    for line in existing_lines:
        try:
            d = json.loads(line)
            cid = d.get("conversation_id")

            if cid == resume_id:
                resume_count += 1
                continue

            cleaned_lines.append(line)

            if cid:
                processed_conversation_ids.add(cid)

        except Exception:
            # Keep unreadable lines conservatively.
            cleaned_lines.append(line)

    with output_path.open("w", encoding="utf-8") as f:
        for line in cleaned_lines:
            f.write(line + "\n")

    print(
        f"Resume: removed {resume_count} rows for conversation_id={resume_id}; "
        "reprocessing from that conversation.",
        flush=True,
    )
    print(
        f"Resume: {len(processed_conversation_ids)} conversation_id(s) already done; skipping.",
        flush=True,
    )

    return processed_conversation_ids, "a"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate hindsight labels for ATLAS context manager MLP."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Annotated conversation JSON file or directory",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path, e.g. ..\\data\\training\\eviction_policy.jsonl",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIG.policy.relevance_threshold,
        help="Relevance threshold. Default comes from DEFAULT_CONFIG.policy.relevance_threshold.",
    )

    parser.add_argument(
        "--label-method",
        choices=["cosine", "bertscore"],
        default="bertscore",
        help="Labeling method. Use bertscore for document-faithful labels, cosine for speed.",
    )

    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer embedding model for 384-d summary embeddings.",
    )

    parser.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=16,
        help="Batch size for BERTScore matrix computation.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing output by removing the last conversation_id and reprocessing from there.",
    )

    parser.add_argument(
        "--flush-every",
        type=int,
        default=100,
        help="Flush output file every N examples. Use 1 for maximum crash safety.",
    )

    parser.add_argument(
        "--no-scispacy",
        action="store_true",
        help="Disable scispaCy entity_count and use 0.0 for all summaries.",
    )

    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {args.embedding_model}", flush=True)
    encoder = SentenceTransformer(args.embedding_model)

    bertscorer = None

    if args.label_method == "bertscore":
        print("Loading BERTScorer once: microsoft/deberta-xlarge-mnli", flush=True)
        bertscorer = create_bertscorer()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        processed_conversation_ids, mode = load_resume_state(output_path)
    else:
        processed_conversation_ids = set()
        mode = "w"

    n_files = 0
    n_examples = 0
    n_pos = 0
    n_neg = 0

    use_scispacy = not args.no_scispacy

    with output_path.open(mode, encoding="utf-8") as out:
        for json_path in iter_json_files(args.input):
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    conversation = json.load(f)
            except Exception as exc:
                print(f"SKIP {json_path}: {exc}", flush=True)
                continue

            if "utterances" not in conversation or "annotations" not in conversation:
                print(f"SKIP {json_path}: missing utterances or annotations", flush=True)
                continue

            conv_id = conversation.get("id", "unknown")

            if args.resume and conv_id in processed_conversation_ids:
                print(
                    f"SKIP already processed conversation_id={conv_id} file={json_path}",
                    flush=True,
                )
                continue

            print(
                f"Processing conversation_id={conv_id} file={json_path}",
                flush=True,
            )

            conv_examples = 0
            conv_pos = 0

            try:
                for ex in generate_examples_for_conversation_iter(
                    conversation=conversation,
                    encoder=encoder,
                    threshold=args.threshold,
                    label_method=args.label_method,
                    bertscorer=bertscorer,
                    bertscore_batch_size=args.bertscore_batch_size,
                    use_scispacy=use_scispacy,
                ):
                    out.write(json.dumps(ex, ensure_ascii=False) + "\n")

                    n_examples += 1
                    conv_examples += 1

                    if ex["label"] == 1:
                        n_pos += 1
                        conv_pos += 1
                    else:
                        n_neg += 1

                    if args.flush_every > 0 and n_examples % args.flush_every == 0:
                        out.flush()

                out.flush()
                n_files += 1

                conv_neg = conv_examples - conv_pos
                conv_rate = conv_pos / max(conv_examples, 1)

                print(
                    f"Done conversation_id={conv_id}: "
                    f"examples={conv_examples}, "
                    f"positive={conv_pos}, "
                    f"negative={conv_neg}, "
                    f"positive_rate={conv_rate:.3f}",
                    flush=True,
                )

            except Exception as exc:
                print(
                    f"ERROR while processing conversation_id={conv_id} file={json_path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

    positive_rate = n_pos / max(n_examples, 1)

    print(
        f"Wrote {n_examples} examples from {n_files} conversation file(s) to {output_path}",
        flush=True,
    )
    print(
        f"Labels: positive={n_pos}, negative={n_neg}, positive_rate={positive_rate:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()