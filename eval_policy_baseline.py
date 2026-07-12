"""
ATLAS Baseline Eviction-Strategy Evaluation
==============================================
Runs the "baseline" LLM (meta-llama/Llama-3.2-3B-Instruct, zero-shot, no
LoRA adapter) through the streaming harness under a fixed set of eviction
baselines — fifo, sliding, random, attention — plus an oracle (unlimited
budget, keeps every sentence, an upper bound rather than a real strategy),
scoring against gold annotations with every metric in src/metrics.py.

This is the reference/baseline half of eval_policy_extrinsic.py's
baseline-vs-model comparison. Run this first to cheaply pre-fill the
"baseline" rows in the shared CSV; eval_policy_extrinsic.py then only
needs to run "model" (skipping anything already cached here) and appends
its rows below.

Results are cached in the same CSV, keyed by (strategy, model, budget) —
combos already present are skipped, so reruns only fill in what's missing.

Usage:
  python eval_policy_baseline.py
  python eval_policy_baseline.py --force-rerun   # ignore cache, recompute everything

Requirements (RunPod GPU box): same as eval_policy_extrinsic.py.
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from baselines import get_strategy
from eval_policy_extrinsic import (
    DEFAULT_BASE_MODEL,
    DEFAULT_BUDGETS,
    DEFAULT_CSV,
    load_conversations,
    build_llm,
    free_llm,
    run_model_across_budgets,
    flatten_metrics,
    load_cache,
    append_rows,
)

# Real-environment data location.
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "aci"

# Fixed baseline set. "oracle" is the unlimited-budget upper bound (no
# eviction), not a real eviction strategy — handled specially below.
STRATEGIES_TO_COMPARE = ["fifo", "sliding", "random", "attention", "oracle"]

# Buffer budget large enough that eviction never triggers -> oracle keeps all.
ORACLE_BUDGET = 999_999


def build_eviction_strategy(strategy_name: str, window_size: int):
    """Return the eviction callable for a baseline, or None for the oracle
    (no eviction — the harness keeps every item)."""
    if strategy_name == "oracle":
        return None
    if strategy_name == "sliding":
        return get_strategy("sliding", window_size=window_size)
    return get_strategy(strategy_name)


def main():
    parser = argparse.ArgumentParser(
        description="ATLAS baseline evaluation across fifo/sliding/random/attention "
                     "eviction baselines plus the keep-everything oracle (no LoRA adapter)"
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--window-size", type=int, default=10, help="sliding-window K")

    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--limit", type=int, default=None, help="limit number of conversations (debug)")

    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--skip-bertscore", action="store_true")

    parser.add_argument("--csv", default=str(DEFAULT_CSV),
                         help="cache/result CSV (strategy, model, budget, metric, value) — "
                              "shared with eval_policy_extrinsic.py")
    parser.add_argument("--force-rerun", action="store_true",
                         help="ignore the cache and recompute every (strategy, budget) combo")
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    csv_path = Path(args.csv)
    _, done = ([], set()) if args.force_rerun else load_cache(csv_path)

    convs = load_conversations(Path(args.data_dir), limit=args.limit)
    if not convs:
        print(f"No conversation JSON files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(convs)} conversations from {args.data_dir}")

    new_rows = []
    for strategy_name in STRATEGIES_TO_COMPARE:
        eviction_strategy = build_eviction_strategy(strategy_name, args.window_size)

        missing_budgets = [b for b in args.budgets if (strategy_name, "baseline", b) not in done]
        if not missing_budgets:
            print(f"[{strategy_name}/baseline] all budgets cached in {csv_path}, skipping")
            continue

        print(f"\n=== strategy={strategy_name} model=baseline budgets={missing_budgets} ===")
        llm = build_llm("baseline", args.base_model, None, args.base_model,
                         dtype, args.load_in_4bit, args.hf_token, args.max_new_tokens)

        if strategy_name == "oracle":
            # Oracle keeps everything regardless of budget, so LLM behavior is
            # identical across budgets — run once, replicate to every budget so
            # the CSV still has an oracle reference row per budget.
            one = run_model_across_budgets(
                "baseline", llm, convs, [ORACLE_BUDGET], eviction_strategy, args.skip_bertscore
            )
            report = next(iter(one.values()))
            per_budget = {b: report for b in missing_budgets}
        else:
            per_budget = run_model_across_budgets(
                "baseline", llm, convs, missing_budgets, eviction_strategy, args.skip_bertscore
            )
        free_llm(llm)

        for budget, report in per_budget.items():
            for metric, value in flatten_metrics(report).items():
                new_rows.append({
                    "strategy": strategy_name, "model": "baseline", "budget": budget,
                    "metric": metric, "value": value,
                })
            done.add((strategy_name, "baseline", budget))

    if new_rows:
        append_rows(csv_path, new_rows)
        print(f"\nAppended {len(new_rows)} rows to {csv_path}")
    else:
        print(f"\nNothing new to run — everything already cached in {csv_path}.")


if __name__ == "__main__":
    main()
