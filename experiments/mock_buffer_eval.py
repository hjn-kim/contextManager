r"""
mock_buffer_eval.py  (all-in-one)
=================================

buffer_metrics.py + eval_mock_buffer.py + plot 를 하나로 합친 파일.

하는 일:
  1. MockLLM 로 각 대화를 streaming 재생하며 전략(learned/fifo/sliding/random/
     oldest/oracle) × 예산(B) 조합의 buffer keep/evict 결정을 계산
  2. context-manager hindsight 라벨(eviction_policy*.jsonl)과 비교해
     buffer_keep_f1 / coverage / token_efficiency 산출 → CSV 저장
  3. buffer_keep_f1 만 예산(x축) 그래프로 PNG 저장 (x축 눈금 = 예산 값)

실행 (프로젝트 루트에서):
  cd c:\Users\mphk0\Documents\context_manager
  .venv311\Scripts\python.exe experiments\mock_buffer_eval.py `
      --input data\valid_aci `
      --policy-labels eviction_policy\eviction_policy_bertscore_valid_0p75.jsonl `
      --mlp eviction_mlp\eviction_mlp_bertscore_0p75.pt `
      --budgets 100 128 256 512 1024
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# src/ 를 import 경로에 추가 (이 파일은 experiments/ 안에 있음)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import AtlasConfig                                  # noqa: E402
from harness import MockLLM, ContextBuffer, BufferItem          # noqa: E402
from baselines import get_strategy                              # noqa: E402
from policy import LearnedEvictionStrategy                      # noqa: E402


# =============================================================================
# PART 1 — buffer_metrics
# =============================================================================

def iter_json_files(input_path: str) -> Iterable[Path]:
    path = Path(input_path)
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*.json")):
        yield p


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def load_policy_label_index(policy_jsonl_path: str):
    """index[(conv_id, current_uid)][(summary_uid, norm_summary)] = label"""
    index = defaultdict(dict)
    with open(policy_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            step_key = (str(row["conversation_id"]), str(row["current_utterance_id"]))
            item_key = (str(row["summary_utterance_id"]), normalize_text(row["summary"]))
            index[step_key][item_key] = int(row["label"])
    return index


def empty_counts() -> Dict[str, float]:
    return {
        "tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0,
        "kept_tokens": 0.0, "future_relevant_kept_tokens": 0.0,
        "evaluated_steps": 0.0, "candidate_items": 0.0, "kept_items_with_gold": 0.0,
    }


def add_counts(dst: Dict[str, float], src: Dict[str, float]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + float(v)


def update_counts_for_conversation(counts, conversation, item_snapshots, policy_index) -> None:
    """buffer after turn t 를 turn t+1 의 라벨과 비교."""
    conversation_id = str(conversation["id"])
    utterances = conversation.get("utterances", [])
    if not utterances:
        return

    for t, snapshot in enumerate(item_snapshots):
        next_t = t + 1
        if next_t >= len(utterances):
            continue

        step_key = (conversation_id, str(utterances[next_t]["id"]))
        gold_items = policy_index.get(step_key)
        if not gold_items:
            continue

        counts["evaluated_steps"] += 1
        counts["candidate_items"] += len(gold_items)

        kept_keys = {(str(it["utterance_id"]), normalize_text(it["summary"])) for it in snapshot}

        for item_key, gold_label in gold_items.items():
            pred_keep = item_key in kept_keys
            gold_keep = int(gold_label) == 1
            if pred_keep and gold_keep:
                counts["tp"] += 1
            elif pred_keep and not gold_keep:
                counts["fp"] += 1
            elif (not pred_keep) and gold_keep:
                counts["fn"] += 1
            else:
                counts["tn"] += 1

        for item in snapshot:
            item_key = (str(item["utterance_id"]), normalize_text(item["summary"]))
            if item_key not in gold_items:
                continue
            token_count = float(item.get("token_count", 0))
            counts["kept_items_with_gold"] += 1
            counts["kept_tokens"] += token_count
            if int(gold_items[item_key]) == 1:
                counts["future_relevant_kept_tokens"] += token_count


def finalize_metrics(counts: Dict[str, float]) -> Dict[str, float]:
    tp, fp, fn = counts.get("tp", 0.0), counts.get("fp", 0.0), counts.get("fn", 0.0)
    tn = counts.get("tn", 0.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # evict 기준(positive=label 0): tp=tn, fp=fn, fn=fp
    evict_precision = tn / max(tn + fn, 1.0)
    evict_recall = tn / max(tn + fp, 1.0)
    evict_f1 = 2 * evict_precision * evict_recall / max(evict_precision + evict_recall, 1e-8)

    # keep F1 과 evict F1 의 평균
    mean_f1 = (f1 + evict_f1) / 2.0

    token_efficiency = counts["future_relevant_kept_tokens"] / max(counts["kept_tokens"], 1.0)
    return {
        "buffer_keep_precision": precision,
        "buffer_keep_recall": recall,
        "buffer_keep_f1": f1,
        "buffer_evict_f1": evict_f1,
        "buffer_mean_f1": mean_f1,
        "future_relevant_coverage": recall,
        "token_efficiency": token_efficiency,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "evaluated_steps": counts.get("evaluated_steps", 0.0),
        "candidate_items": counts.get("candidate_items", 0.0),
        "kept_items_with_gold": counts.get("kept_items_with_gold", 0.0),
        "kept_tokens": counts.get("kept_tokens", 0.0),
        "future_relevant_kept_tokens": counts.get("future_relevant_kept_tokens", 0.0),
    }


# =============================================================================
# PART 2 — eval_mock_buffer
# =============================================================================

def build_mock_annotations(conversation: dict) -> Dict[str, dict]:
    annotations = {}
    for ann in conversation.get("annotations", []):
        uid = ann.get("utterance_id")
        if uid:
            annotations[uid] = ann
    return annotations


def output_to_details(output) -> List[tuple]:
    details = []
    if output.detail_list:
        for d in output.detail_list:
            summary = (d.get("summary") or "").strip()
            if summary:
                details.append((d.get("type") or "detail", summary))
    elif output.summary:
        details.append((output.type or "detail", output.summary))
    return details


def serialize_buffer_item(item: BufferItem) -> dict:
    return {
        "summary": item.summary, "type": item.type, "speaker": item.speaker,
        "utterance_id": item.utterance_id, "timestamp": item.timestamp,
        "token_count": item.token_count, "score": item.score,
    }


def run_mock_conversation_with_item_snapshots(conversation, config, eviction_strategy) -> List[List[dict]]:
    annotations = build_mock_annotations(conversation)
    llm = MockLLM(config, annotations)
    buffer = ContextBuffer(config)
    item_snapshots: List[List[dict]] = []

    for i, utt in enumerate(conversation.get("utterances", [])):
        context = buffer.get_context_string()
        output = llm.predict(context=context, speaker=utt["speaker"],
                             text=utt["text"], utterance_id=utt["id"])
        if output.relevant:
            for dtype, dsummary in output_to_details(output):
                buffer.add(BufferItem(
                    summary=dsummary, type=dtype, speaker=utt["speaker"],
                    utterance_id=utt["id"], timestamp=i,
                    token_count=llm.get_token_count(dsummary),
                ))
            if eviction_strategy:
                buffer.evict(eviction_strategy)
        item_snapshots.append([serialize_buffer_item(it) for it in buffer.items])

    return item_snapshots


def make_strategy(strategy_name, mlp_path, sliding_k) -> Optional[Callable]:
    if strategy_name == "oracle":
        return None
    if strategy_name == "learned":
        if not mlp_path:
            raise ValueError("--mlp is required when strategy includes 'learned'")
        learned = LearnedEvictionStrategy(model_path=mlp_path)

        def _learned(items, **kwargs):   # harness 가 넘기는 kwargs 흡수
            return learned(items)
        return _learned
    if strategy_name == "sliding":
        return get_strategy("sliding", window_size=sliding_k)
    return get_strategy(strategy_name)


def evaluate_condition(conversation_paths, policy_index, strategy_name, budget, mlp_path, sliding_k) -> dict:
    config = AtlasConfig()
    config.buffer.budget = 999999 if strategy_name == "oracle" else budget
    strategy = make_strategy(strategy_name, mlp_path, sliding_k)

    total_counts = empty_counts()
    for path in conversation_paths:
        with open(path, "r", encoding="utf-8") as f:
            conversation = json.load(f)
        if "utterances" not in conversation or "annotations" not in conversation:
            print(f"SKIP {path}: missing utterances or annotations")
            continue
        snapshots = run_mock_conversation_with_item_snapshots(conversation, config, strategy)
        conv_counts = empty_counts()
        update_counts_for_conversation(conv_counts, conversation, snapshots, policy_index)
        add_counts(total_counts, conv_counts)

    metrics = finalize_metrics(total_counts)
    return {
        "strategy": strategy_name,
        "budget": "unlimited" if strategy_name == "oracle" else budget,
        **metrics,
    }


CSV_FIELDS = [
    "strategy", "budget", "buffer_keep_precision", "buffer_keep_recall",
    "buffer_keep_f1", "buffer_evict_f1", "buffer_mean_f1",
    "future_relevant_coverage", "token_efficiency",
    "tp", "fp", "tn", "fn", "evaluated_steps", "candidate_items",
    "kept_items_with_gold", "kept_tokens", "future_relevant_kept_tokens",
]


def write_csv(rows: List[dict], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =============================================================================
# PART 3 — plot (buffer_keep_f1 only)
# =============================================================================

HIGHLIGHT = "learned"


def plot_f1(rows: List[dict], budgets: List[int], out_png: str, no_show: bool) -> None:
    import matplotlib
    if no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = []
    for r in rows:
        if r["strategy"] not in strategies:
            strategies.append(r["strategy"])

    budgets_sorted = sorted(budgets)
    xpos = list(range(len(budgets_sorted)))  # 균등 간격 배치 (100/128 이 안 겹치도록)

    plt.figure(figsize=(8, 5.5))
    for strat in strategies:
        by_budget = {r["budget"]: r["buffer_keep_f1"] for r in rows if r["strategy"] == strat}
        if strat == "oracle":
            val = by_budget.get("unlimited")
            if val is not None:
                plt.axhline(val, color="gray", ls="--", lw=1, alpha=0.7, label="oracle (unlim)")
            continue
        ys = [by_budget.get(b) for b in budgets_sorted]
        is_hl = strat == HIGHLIGHT
        plt.plot(xpos, ys, marker="o",
                 lw=3 if is_hl else 1.5,
                 zorder=5 if is_hl else 3,
                 label=strat + (" (MLP)" if is_hl else ""))

    plt.xticks(xpos, [str(b) for b in budgets_sorted])
    plt.xlabel("Token budget B")
    plt.ylabel("buffer_keep_f1")
    plt.title("Buffer keep F1 vs budget")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"png        : {out_png}")
    if not no_show:
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    default_csv = os.path.join(_HERE, "results", "mock_buffer_eval.csv")
    default_png = os.path.join(_HERE, "results", "mock_buffer_f1.png")

    parser = argparse.ArgumentParser(description="MockLLM buffer eval + F1 plot (all-in-one).")
    parser.add_argument("--input", required=True, help="대화 JSON 파일/폴더 (예: data\\valid_aci)")
    parser.add_argument("--policy-labels", required=True, help="context-manager 라벨 JSONL")
    parser.add_argument("--mlp", default=None, help="learned 전략용 MLP .pt")
    parser.add_argument("--budgets", nargs="+", type=int, default=[100, 128, 256, 512, 1024])
    parser.add_argument("--strategies", nargs="+",
                        default=["learned", "fifo", "sliding", "random", "oracle"],
                        choices=["learned", "fifo", "sliding", "random", "oldest", "oracle"])
    parser.add_argument("--sliding-k", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output", default=default_csv, help="CSV 저장 경로")
    parser.add_argument("--plot", default=default_png, help="F1 그래프 PNG 저장 경로")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    random.seed(args.random_seed)

    conversation_paths = list(iter_json_files(args.input))
    if not conversation_paths:
        raise RuntimeError(f"No JSON files found under {args.input}")
    print(f"conversations: {len(conversation_paths)}")
    print(f"policy labels: {args.policy_labels}")

    policy_index = load_policy_label_index(args.policy_labels)

    rows = []
    for strategy_name in args.strategies:
        if strategy_name == "oracle":
            r = evaluate_condition(conversation_paths, policy_index,
                                   strategy_name, 999999, args.mlp, args.sliding_k)
            rows.append(r)
            print(f"[oracle] keep_f1={r['buffer_keep_f1']:.4f} evict_f1={r['buffer_evict_f1']:.4f} "
                  f"keep/evict_f1={r['buffer_mean_f1']:.4f}")
            continue
        for budget in args.budgets:
            row = evaluate_condition(conversation_paths, policy_index,
                                     strategy_name, budget, args.mlp, args.sliding_k)
            rows.append(row)
            print(f"[B={budget:>5}] {strategy_name:<8} "
                  f"keep_f1={row['buffer_keep_f1']:.4f} evict_f1={row['buffer_evict_f1']:.4f} "
                  f"keep/evict_f1={row['buffer_mean_f1']:.4f}")

    write_csv(rows, args.output)
    print(f"csv        : {args.output}")

    plot_f1(rows, args.budgets, args.plot, args.no_show)


if __name__ == "__main__":
    main()
