"""
ATLAS Baseline Eviction-Strategy Evaluation (fixed LoRA model)
===============================================================
Runs the FIXED LoRA Agenda LLM (--model-path, default the C1 adapter
tria-hongik/atlas-c1-v2-llama-3.2-3b) through the streaming harness under
every baseline eviction strategy — fifo, sliding, random, attention — plus
an oracle (unlimited budget, keeps every sentence; an upper bound rather
than a real strategy), scoring against gold annotations with every metric
in src/metrics.py.

This is the baseline half of eval_policy_extrinsic.py's baseline-vs-learned
comparison. Both scripts hold the Agenda LLM fixed to the same LoRA model
and vary only the eviction strategy. Run this first to pre-fill the
baseline-strategy rows in the shared CSV; eval_policy_extrinsic.py then only
needs to run the trained "learned" policy (skipping anything already cached
here) and plots baselines vs learned.

Results are cached in the same CSV, keyed by (strategy, budget) — combos
already present are skipped, so reruns only fill in what's missing.

Usage:
  python eval_policy_baseline.py
  python eval_policy_baseline.py --model-path someuser/some-hub-adapter
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

from eval_policy_extrinsic import (
    DEFAULT_BASE_MODEL,
    DEFAULT_C1_ADAPTER,
    DEFAULT_DATA_DIR,
    DEFAULT_BUDGETS,
    DEFAULT_CSV,
    BASELINE_STRATEGIES,
    ORACLE_BUDGET,
    load_conversations,
    resolve_model_spec,
    build_llm,
    free_llm,
    log_cache_stats,
    build_eviction_strategy,
    run_model_across_budgets,
    flatten_metrics,
    load_cache,
    append_rows,
)


def main():
    parser = argparse.ArgumentParser(
        description="ATLAS baseline evaluation on a FIXED LoRA model across "
                     "fifo/sliding/random/attention eviction baselines plus the oracle"
    )
    parser.add_argument("--model-path", default=DEFAULT_C1_ADAPTER,
                         help="LoRA adapter dir/hub-id or full model dir for the fixed Agenda LLM "
                              "(default: C1 hub adapter) — must match eval_policy_extrinsic.py")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--window-size", type=int, default=10, help="sliding-window K")

    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--limit", type=int, default=None, help="limit number of conversations (debug)")

    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--skip-bertscore", action="store_true",
                         help="skip all deberta-xlarge bert-score metrics — both summary "
                              "bertscore AND agenda_completeness (which is bert-score based)")

    parser.add_argument("--csv", default=str(DEFAULT_CSV),
                         help="cache/result CSV (strategy, budget, metric, value) — "
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

    # Fixed LoRA Agenda LLM, built lazily (and once) only if some strategy is
    # actually missing from the cache.
    llm = None
    # Shared across every strategy/budget so identical trajectories score once.
    report_cache = {}
    new_rows = []
    for strategy_name in BASELINE_STRATEGIES:
        missing_budgets = [b for b in args.budgets if (strategy_name, b) not in done]
        if not missing_budgets:
            print(f"[{strategy_name}] all budgets cached in {csv_path}, skipping")
            continue

        if llm is None:
            base_for_load, adapter_path, tok_source = resolve_model_spec(
                args.model_path, args.base_model, args.hf_token
            )
            llm = build_llm("model", base_for_load, adapter_path, tok_source,
                             dtype, args.load_in_4bit, args.hf_token, args.max_new_tokens)

        # policy_model is unused for baselines (no "learned" here).
        eviction_strategy = build_eviction_strategy(strategy_name, args.window_size, None, True)

        print(f"\n=== strategy={strategy_name} budgets={missing_budgets} ===")
        if strategy_name == "oracle":
            # Oracle keeps everything regardless of budget -> run once, replicate.
            one = run_model_across_budgets(
                "model", llm, convs, [ORACLE_BUDGET], eviction_strategy, args.skip_bertscore,
                report_cache=report_cache,
            )
            report = next(iter(one.values()))
            per_budget = {b: report for b in missing_budgets}
        else:
            per_budget = run_model_across_budgets(
                "model", llm, convs, missing_budgets, eviction_strategy, args.skip_bertscore,
                report_cache=report_cache,
            )

        for budget, report in per_budget.items():
            for metric, value in flatten_metrics(report).items():
                new_rows.append({
                    "strategy": strategy_name, "budget": budget,
                    "metric": metric, "value": value,
                })
            done.add((strategy_name, budget))

    if llm is not None:
        log_cache_stats(llm)
        free_llm(llm)

    if new_rows:
        append_rows(csv_path, new_rows)
        print(f"\nAppended {len(new_rows)} rows to {csv_path}")
    else:
        print(f"\nNothing new to run — everything already cached in {csv_path}.")


if __name__ == "__main__":
    main()
