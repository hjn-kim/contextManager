"""
ATLAS Evaluation Metrics
=========================
Model-level + System-level + Hardware-level.
"""

from typing import List, Dict, Optional


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            for key in scores:
                scores[key].append(s[key].fmeasure)
        return {k: sum(v) / max(len(v), 1) for k, v in scores.items()}
    except ImportError:
        print("WARNING: rouge-score not installed")
        return {}


_BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
_BERTSCORER = None
_BERTSCORER_UNAVAILABLE = False


def _get_bertscorer():
    """Lazily build a single cached BERTScorer and reuse it everywhere.

    Two reasons this is a function and not a bare bert_score.score() call:

    1. Correctness. bert-score passes tokenizer.model_max_length as the
       truncation length. deberta-xlarge-mnli leaves model_max_length at the
       huge default sentinel (~1e30); under transformers 5.x the *fast*
       tokenizer forwards it to the Rust enable_truncation(), which overflows
       with "OverflowError: int too big to convert". Capping model_max_length
       to the model's real 512 fixes it.

    2. Speed. Building the scorer loads a 1.6GB model; caching it avoids
       reloading on every call (compute_agenda_completeness otherwise reloads
       it once per reference item).

    Returns None if bert-score can't be initialized, so callers can fall back.
    """
    global _BERTSCORER, _BERTSCORER_UNAVAILABLE
    if _BERTSCORER is not None:
        return _BERTSCORER
    if _BERTSCORER_UNAVAILABLE:
        return None
    try:
        from bert_score import BERTScorer
        scorer = BERTScorer(model_type=_BERTSCORE_MODEL)
        tok = getattr(scorer, "_tokenizer", None)
        mml = getattr(tok, "model_max_length", None) if tok is not None else None
        if tok is not None and (mml is None or mml > 100_000):
            tok.model_max_length = 512  # deberta-xlarge-mnli's real context
        _BERTSCORER = scorer
        return _BERTSCORER
    except Exception as e:
        print(f"WARNING: could not initialize bert-score ({type(e).__name__}: {e})")
        _BERTSCORER_UNAVAILABLE = True
        return None


def compute_bertscore(predictions: List[str], references: List[str]) -> float:
    if not predictions or not references:
        return 0.0
    scorer = _get_bertscorer()
    if scorer is None:
        return 0.0
    try:
        P, R, F1 = scorer.score(predictions, references)
        return F1.mean().item()
    except Exception as e:
        # Don't let a single scoring failure crash the whole evaluation.
        print(f"WARNING: bert-score failed at runtime ({type(e).__name__}: {e}).")
        return 0.0


def compute_detection_metrics(pred_labels: List[str],
                               true_labels: List[str]) -> Dict[str, float]:
    from collections import Counter

    ALL_CLASSES = [
        "agenda_item", "detail", "medication", "social_history",
        "follow_up", "question", "question_unanswered", "irrelevant"
    ]

    tp, fp, fn = Counter(), Counter(), Counter()

    for pred, true in zip(pred_labels, true_labels):
        if pred == true:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    results = {}
    for t in ALL_CLASSES:
        p = tp[t] / max(tp[t] + fp[t], 1)
        r = tp[t] / max(tp[t] + fn[t], 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        results[t] = {"precision": p, "recall": r, "f1": f1}

    avg_p = sum(results[t]["precision"] for t in ALL_CLASSES) / len(ALL_CLASSES)
    avg_r = sum(results[t]["recall"] for t in ALL_CLASSES) / len(ALL_CLASSES)
    avg_f1 = sum(results[t]["f1"] for t in ALL_CLASSES) / len(ALL_CLASSES)
    results["macro_avg"] = {"precision": avg_p, "recall": avg_r, "f1": avg_f1}

    return results


def compute_agenda_completeness(system_summaries: List[str],
                                 reference_items: List[str],
                                 threshold: float = 0.85) -> float:
    if not reference_items:
        return 1.0
    if not system_summaries:
        return 0.0

    scorer = _get_bertscorer()
    if scorer is not None:
        try:
            found = 0
            for ref in reference_items:
                refs_repeated = [ref] * len(system_summaries)
                P, R, F1 = scorer.score(system_summaries, refs_repeated)
                if F1.max().item() > threshold:
                    found += 1
            return found / len(reference_items)
        except Exception as e:
            print(f"WARNING: agenda_completeness bert-score failed "
                  f"({type(e).__name__}: {e}); falling back to sentence-transformers.")

    # bert-score unavailable OR failing at runtime. Fall back to
    # sentence-transformers cosine.
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("all-MiniLM-L6-v2")
        ref_embs = model.encode(reference_items)
        sys_embs = model.encode(system_summaries)
        found = 0
        for ref_emb in ref_embs:
            sims = np.dot(sys_embs, ref_emb) / (
                np.linalg.norm(sys_embs, axis=1) * np.linalg.norm(ref_emb) + 1e-8)
            if sims.max() > threshold:
                found += 1
        return found / len(reference_items)
    except Exception:
        print("WARNING: agenda_completeness unavailable (no working "
              "bert-score or sentence-transformers backend).")
        return -1.0


def compute_linking_accuracy(system_links: List[Optional[str]],
                              gold_links: List[Optional[str]]) -> float:
    correct = sum(1 for s, g in zip(system_links, gold_links) if s == g)
    return correct / max(len(system_links), 1)


def compute_first_mention_accuracy(system_flags: List[bool],
                                    gold_flags: List[bool]) -> float:
    correct = sum(1 for s, g in zip(system_flags, gold_flags) if s == g)
    return correct / max(len(system_flags), 1)


def compute_resolution_accuracy(system_statuses: List[str],
                                 gold_statuses: List[str]) -> float:
    correct = sum(1 for s, g in zip(system_statuses, gold_statuses) if s == g)
    return correct / max(len(system_statuses), 1)


def parse_tegrastats_log(log_path: str) -> Dict:
    return {
        "peak_gpu_mem_mb": 0,
        "peak_cpu_mem_mb": 0,
        "avg_gpu_util": 0,
    }


def compute_kv_cache_memory(n_layers: int, n_heads: int, head_dim: int,
                             seq_len: int, dtype_bytes: int = 2) -> int:
    return 2 * n_layers * n_heads * head_dim * seq_len * dtype_bytes


def _detail_entries(annotation: dict) -> List[dict]:
    """Normalize legacy flat outputs and current detail_list outputs."""
    if not annotation or not annotation.get("relevant"):
        return []

    detail_list = annotation.get("detail_list")
    if detail_list:
        return [
            {
                "type": d.get("type", "detail"),
                "summary": d.get("summary"),
                "eval_only": d.get("eval_only"),
            }
            for d in detail_list
            if d.get("summary")
        ]

    if annotation.get("summary"):
        return [{
            "type": annotation.get("type", "detail"),
            "summary": annotation.get("summary"),
            "eval_only": annotation.get("eval_only"),
        }]

    return []


def _append_detection_pairs(pred_entries: List[dict], gold_entries: List[dict],
                            pred_types: List[str], gold_types: List[str]):
    """Align detail entries order-independently.

    A single utterance may carry several details, and the model can emit them
    in any order. Index-based pairing would then mislabel correct predictions,
    so we match entries by type instead:
      1. Greedily match pred/gold entries that share the same type (these count
         as correct regardless of position).
      2. Pair the leftover mismatches with each other.
      3. Pad whatever remains against 'irrelevant'.
    The (pred, gold) pairing order among leftovers does not affect the
    aggregated precision/recall in compute_detection_metrics.
    """
    from collections import Counter

    if not pred_entries and not gold_entries:
        pred_types.append("irrelevant")
        gold_types.append("irrelevant")
        return

    pred_labels = [e.get("type", "irrelevant") for e in pred_entries]
    gold_labels = [e.get("type", "irrelevant") for e in gold_entries]

    # 1. Match same-type entries first (order-independent).
    gold_pool = Counter(gold_labels)
    leftover_pred: List[str] = []
    for label in pred_labels:
        if gold_pool.get(label, 0) > 0:
            gold_pool[label] -= 1
            pred_types.append(label)
            gold_types.append(label)      # same type on both sides -> correct
        else:
            leftover_pred.append(label)

    # 2. Remaining unmatched gold entries.
    leftover_gold = list(gold_pool.elements())

    # 3. Pair leftover mismatches, padding the shorter side with 'irrelevant'.
    for i in range(max(len(leftover_pred), len(leftover_gold))):
        pred_types.append(leftover_pred[i] if i < len(leftover_pred) else "irrelevant")
        gold_types.append(leftover_gold[i] if i < len(leftover_gold) else "irrelevant")


def generate_report(harness_result, gold_annotations: List[dict] = None,
                    utterances: List[dict] = None,
                    eval_annotations: List[dict] = None) -> Dict:
    report = {
        "conversation_id": harness_result.conversation_id,
        "metadata": harness_result.metadata,
        "latency": {
            "avg_ms": harness_result.metadata.get("avg_latency_ms", 0),
            "max_ms": max(harness_result.latencies) if harness_result.latencies else 0,
            "min_ms": min(harness_result.latencies) if harness_result.latencies else 0,
        },
    }

    if gold_annotations and utterances:
        gold_by_id = {}
        for ann in gold_annotations:
            gold_by_id[ann["utterance_id"]] = ann

        aligned_preds = []
        aligned_golds = []
        pred_types = []
        gold_types = []

        for i, utt in enumerate(utterances):
            uid = utt["id"]
            pred = harness_result.llm_outputs[i] if i < len(harness_result.llm_outputs) else {"relevant": False}
            gold = gold_by_id.get(uid, {"relevant": False})

            pred_entries = _detail_entries(pred)
            gold_entries = _detail_entries(gold)
            _append_detection_pairs(pred_entries, gold_entries, pred_types, gold_types)

            for pred_entry, gold_entry in zip(pred_entries, gold_entries):
                if pred_entry.get("summary") and gold_entry.get("summary"):
                    aligned_preds.append(pred_entry["summary"])
                    aligned_golds.append(gold_entry["summary"])

        if aligned_preds and aligned_golds:
            report["rouge"] = compute_rouge(aligned_preds, aligned_golds)

        report["detection"] = compute_detection_metrics(pred_types, gold_types)

        system_summaries = [
            entry["summary"]
            for output in harness_result.llm_outputs
            for entry in _detail_entries(output)
            if entry.get("summary")
        ]
        gold_agenda_items = [
            entry["summary"]
            for annotation in gold_annotations
            for entry in _detail_entries(annotation)
            if entry.get("type") == "agenda_item"
        ]
        if gold_agenda_items:
            report["agenda_completeness"] = compute_agenda_completeness(
                system_summaries, gold_agenda_items)

    if eval_annotations and harness_result.tracker_states:
        last_tracker = harness_result.tracker_states[-1]
        tracker_preds = last_tracker.get("predictions", [])

        if tracker_preds:
            eval_by_key = {}
            for ann in eval_annotations:
                uid = ann["utterance_id"]
                entries = _detail_entries(ann)
                if entries:
                    for i, entry in enumerate(entries):
                        if entry.get("eval_only"):
                            eval_by_key[(uid, i)] = entry["eval_only"]
                elif ann.get("eval_only"):
                    eval_by_key[(uid, 0)] = ann["eval_only"]

            sys_links = []
            gold_links = []
            sys_first = []
            gold_first = []
            sys_resolution = []
            gold_resolution = []

            for pred in tracker_preds:
                uid = pred["utterance_id"]
                detail_index = pred.get("detail_index")
                if detail_index is None:
                    continue

                key = (uid, detail_index)
                if key in eval_by_key and pred.get("relevant"):
                    gold_eval = eval_by_key[key]

                    if "linked_agenda_item_id" in gold_eval:
                        sys_links.append(pred.get("linked_agenda_item_id"))
                        gold_links.append(gold_eval.get("linked_agenda_item_id"))

                    if pred["first_mention"] is not None:
                        sys_first.append(pred["first_mention"])
                        gold_first.append(gold_eval.get("first_mention"))

                    if pred["resolution_status"] is not None:
                        sys_resolution.append(pred["resolution_status"])
                        gold_resolution.append(gold_eval.get("resolution_status"))

            system_metrics = {}
            if sys_links:
                system_metrics["linking_accuracy"] = compute_linking_accuracy(sys_links, gold_links)
            if sys_first:
                system_metrics["first_mention_accuracy"] = compute_first_mention_accuracy(sys_first, gold_first)
            if sys_resolution:
                system_metrics["resolution_accuracy"] = compute_resolution_accuracy(sys_resolution, gold_resolution)

            if system_metrics:
                report["system_level"] = system_metrics

    return report
