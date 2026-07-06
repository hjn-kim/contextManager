"""
ATLAS Baseline Evaluation
==========================
Runs all baseline eviction strategies across a data split
and produces a comparison table.

Usage:
  python eval_baselines.py                         # valid split (default)
  python eval_baselines.py --split test1
  python eval_baselines.py --splits valid test1 test2 test3
  python eval_baselines.py --split valid --output results.json
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import AtlasConfig
from harness import StreamingHarness, MockLLM
from baselines import get_strategy

DATA_DIR = Path(__file__).parent / "data"
SPLITS = ["aci", "eval", "valid", "test1", "test2", "test3"]

EXPERIMENTS = [
    {"name": "oracle",       "strategy": "oracle",  "budget": None, "window": None},
    {"name": "fifo-256",     "strategy": "fifo",    "budget": 256,  "window": None},
    {"name": "fifo-512",     "strategy": "fifo",    "budget": 512,  "window": None},
    {"name": "fifo-1024",    "strategy": "fifo",    "budget": 1024, "window": None},
    {"name": "fifo-2048",    "strategy": "fifo",    "budget": 2048, "window": None},
    {"name": "sliding-5",    "strategy": "sliding", "budget": None, "window": 5},
    {"name": "sliding-10",   "strategy": "sliding", "budget": None, "window": 10},
    {"name": "random-1024",  "strategy": "random",  "budget": 1024, "window": None},
    {"name": "oldest-1024",  "strategy": "oldest",  "budget": 1024, "window": None},
]


def load_split(split: str) -> list[dict]:
    folder = DATA_DIR / split
    convs = []
    for f in sorted(folder.glob("*.json")):
        with open(f, encoding="utf-8") as fp:
            convs.append(json.load(fp))
    return convs


def run_experiment(convs: list[dict], exp: dict) -> list[dict]:
    config = AtlasConfig()

    if exp["strategy"] == "oracle":
        config.buffer.budget = 999999
        strategy = None
    else:
        config.buffer.budget = exp["budget"] or 1024
        if exp["strategy"] == "sliding" and exp.get("window"):
            strategy = get_strategy("sliding", window_size=exp["window"])
        else:
            strategy = get_strategy(exp["strategy"])

    results = []
    for conv in convs:
        annotations = {ann["utterance_id"]: ann for ann in conv.get("annotations", [])}
        llm = MockLLM(config, annotations)
        harness = StreamingHarness(config)
        harness.load_conversation_dict(conv)
        result = harness.run(llm, eviction_strategy=strategy)

        # total_relevant counts utterances, but buffer adds one item per detail_list entry.
        # Use actual items added for a meaningful retention rate.
        total_items_added = sum(
            len(o.get("detail_list") or [])
            for o in result.llm_outputs if o["relevant"]
        )
        result.metadata["total_items_added"] = total_items_added
        results.append(result.metadata)

    return results


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    avg_relevant = sum(r["total_relevant"] for r in results) / n
    avg_buffer   = sum(r["final_buffer_size"] for r in results) / n
    avg_tokens   = sum(r["final_buffer_tokens"] for r in results) / n
    avg_latency  = sum(r["avg_latency_ms"] for r in results) / n

    retention_rates = [
        r["final_buffer_size"] / r["total_items_added"]
        for r in results if r["total_items_added"] > 0
    ]
    retention = sum(retention_rates) / len(retention_rates) if retention_rates else 0.0

    return {
        "conversations": n,
        "avg_relevant":  round(avg_relevant, 1),
        "avg_buffer":    round(avg_buffer, 1),
        "avg_tokens":    round(avg_tokens, 1),
        "retention_pct": round(retention * 100, 1),
        "avg_latency_ms": round(avg_latency, 3),
    }


def budget_label(exp: dict) -> str:
    if exp["strategy"] == "oracle":
        return "∞"
    if exp["strategy"] == "sliding":
        return f"K={exp['window']}"
    return str(exp["budget"])


def print_table(split: str, rows: list[dict]):
    print(f"\n{'='*82}")
    print(f"  Split: {split}  ({rows[0]['agg']['conversations']} conversations)")
    print(f"{'='*82}")
    print(f"  {'Strategy':<16} {'Budget':>7} {'Avg Relevant':>13} {'Avg Buffer':>11} {'Retention':>10} {'Tokens':>8}")
    print(f"  {'-'*78}")
    for row in rows:
        exp, agg = row["exp"], row["agg"]
        print(
            f"  {exp['name']:<16} {budget_label(exp):>7} "
            f"{agg['avg_relevant']:>13.1f} {agg['avg_buffer']:>11.1f} "
            f"{agg['retention_pct']:>9.1f}% {agg['avg_tokens']:>8.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--split",  choices=SPLITS, help="단일 split 평가")
    group.add_argument("--splits", nargs="+", choices=SPLITS, help="여러 split 평가")
    parser.add_argument("--output", default=None, help="결과를 JSON으로 저장")
    args = parser.parse_args()

    splits = args.splits or [args.split or "valid"]

    all_results = {}
    for split in splits:
        folder = DATA_DIR / split
        if not folder.exists():
            print(f"[skip] {split}/ 폴더 없음")
            continue

        convs = load_split(split)
        if not convs:
            print(f"[skip] {split}/ 에 JSON 파일 없음")
            continue

        print(f"\n[{split}] {len(convs)}개 대화 로드됨")
        rows = []
        for exp in EXPERIMENTS:
            print(f"  {exp['name']:<16} ...", end=" ", flush=True)
            agg = aggregate(run_experiment(convs, exp))
            rows.append({"exp": exp, "agg": agg})
            print(f"retention={agg['retention_pct']}%")

        if rows:
            print_table(split, rows)
        all_results[split] = rows

    if args.output:
        out = {
            split: [{"name": r["exp"]["name"], **r["agg"]} for r in rows]
            for split, rows in all_results.items()
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n결과 저장됨: {args.output}")


if __name__ == "__main__":
    main()
