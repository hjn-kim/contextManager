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


def compute_bertscore(predictions: List[str], references: List[str]) -> float:
    try:
        from bert_score import score
        P, R, F1 = score(predictions, references,
                         model_type="microsoft/deberta-xlarge-mnli",
                         verbose=False)
        return F1.mean().item()
    except ImportError:
        print("WARNING: bert-score not installed")
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

    try:
        from bert_score import score
        found = 0
        for ref in reference_items:
            refs_repeated = [ref] * len(system_summaries)
            P, R, F1 = score(system_summaries, refs_repeated,
                             model_type="microsoft/deberta-xlarge-mnli",
                             verbose=False)
            if F1.max().item() > threshold:
                found += 1
        return found / len(reference_items)
    except ImportError:
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
        except ImportError:
            print("WARNING: neither bert-score nor sentence-transformers installed.")
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

            pred_type = pred.get("type", "irrelevant") if pred.get("relevant") else "irrelevant"
            gold_type = gold.get("type", "irrelevant") if gold.get("relevant") else "irrelevant"
            pred_types.append(pred_type)
            gold_types.append(gold_type)

            if pred.get("relevant") and pred.get("summary") and gold.get("relevant") and gold.get("summary"):
                aligned_preds.append(pred["summary"])
                aligned_golds.append(gold["summary"])

        if aligned_preds and aligned_golds:
            report["rouge"] = compute_rouge(aligned_preds, aligned_golds)

        report["detection"] = compute_detection_metrics(pred_types, gold_types)

        system_summaries = [o["summary"] for o in harness_result.llm_outputs
                          if o.get("relevant") and o.get("summary")]
        gold_agenda_items = [a["summary"] for a in gold_annotations
                           if a.get("type") == "agenda_item"]
        if gold_agenda_items:
            report["agenda_completeness"] = compute_agenda_completeness(
                system_summaries, gold_agenda_items)

    if eval_annotations and harness_result.tracker_states:
        last_tracker = harness_result.tracker_states[-1]
        tracker_preds = last_tracker.get("predictions", [])

        if tracker_preds:
            eval_by_id = {}
            for ann in eval_annotations:
                if ann.get("eval_only"):
                    eval_by_id[ann["utterance_id"]] = ann["eval_only"]

            sys_links = []
            gold_links = []
            sys_first = []
            gold_first = []
            sys_resolution = []
            gold_resolution = []

            for pred in tracker_preds:
                uid = pred["utterance_id"]
                if uid in eval_by_id and pred.get("relevant"):
                    gold_eval = eval_by_id[uid]

                    if pred["linked_agenda_item_id"] is not None:
                        sys_links.append(pred["linked_agenda_item_id"])
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
