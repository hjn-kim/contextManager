r"""
Agenda-Completeness C3 Sweep
============================
Measures agenda completeness of the FINAL RETAINED BUFFER for:

  C1 (LoRA Agenda LLM) + C2 (SystemTracker) + C3 in {
      none,                       # no eviction (upper bound)
      fifo, random, oldest,       # token-budget baselines
      learned,                    # the trained MLP policy
  }

across token budgets B in {100, 128, 256, 512}.

WHY IT IS CHEAP
---------------
C1's per-utterance output does NOT depend on C3 or the budget, so we run the
LoRA exactly ONCE per conversation (Phase A) and cache its predictions. The
whole (strategy x budget) grid is then replayed on CPU using those cached
predictions (Phase B) — no further GPU/LLM calls.

USAGE (on RunPod, from project root)
------------------------------------
  # Phase A runs the LoRA once per conversation and caches to experiments/predictions/;
  # Phase B sweeps the grid and writes CSV + PNG.
  python experiments/agenda_completeness_sweep.py \
      --input data/valid_aci \
      --mlp eviction_mlp/eviction_mlp_bertscore_0p75.pt \
      --budgets 100 128 256 512 \
      --strategies none fifo random oldest learned \
      --output experiments/results/agenda_sweep.csv \
      --plot   experiments/results/agenda_sweep.png --no-show

  # Sanity-check the pipeline with gold annotations instead of the LoRA (no GPU):
  python experiments/agenda_completeness_sweep.py --input data/valid_aci --mock \
      --strategies none fifo learned --mlp eviction_mlp/eviction_mlp_bertscore_0p75.pt

Re-running is instant: cached predictions are reused unless --refresh is given.
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Make src/ importable (this file lives in experiments/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import AtlasConfig                                    # noqa: E402
from harness import StreamingHarness, MockLLM                     # noqa: E402
from state_tracker import SystemTracker                           # noqa: E402
from harness_integration import (                                 # noqa: E402
    LoraAgendaLLM, build_eviction, _gold_agenda_items, DEFAULT_ADAPTER,
)


# =============================================================================
# Shared sentence encoder (loaded ONCE) + fast agenda completeness
# =============================================================================

_ENCODER = None


def _get_encoder():
    """Single cached SentenceTransformer, shared across the whole sweep.

    metrics.compute_agenda_completeness reloads the model on every call, which
    is far too slow for a grid sweep. We load it once here.
    """
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
    return _ENCODER


def agenda_completeness(system_summaries: List[str],
                        gold_items: List[str],
                        threshold: float = 0.85) -> Optional[float]:
    """Fraction of gold agenda items covered by a retained summary (cosine >= th).

    Same definition/threshold as metrics.compute_agenda_completeness, but reuses
    one cached encoder and skips the broken bert-score path entirely.
    """
    if not gold_items:
        return None
    if not system_summaries:
        return 0.0

    import numpy as np
    enc = _get_encoder()
    gold_embs = enc.encode(gold_items, normalize_embeddings=True)
    sys_embs = enc.encode(system_summaries, normalize_embeddings=True)
    sims = np.asarray(gold_embs) @ np.asarray(sys_embs).T   # (n_gold, n_sys)
    covered = (sims.max(axis=1) >= threshold).sum()
    return float(covered) / len(gold_items)


# =============================================================================
# Phase A — run C1 (LoRA) once per conversation, cache predictions
# =============================================================================

def _predicted_annotations(conv: dict, result) -> List[dict]:
    """Turn one harness run's llm_outputs into annotation records keyed by uid."""
    anns = []
    for utt, out in zip(conv["utterances"], result.llm_outputs):
        rec = {"utterance_id": utt["id"], "relevant": bool(out.get("relevant"))}
        if not out.get("relevant"):
            anns.append(rec)
            continue
        detail_list = out.get("detail_list")
        if detail_list:
            rec["detail_list"] = [
                {"type": d["type"], "summary": d["summary"]}
                for d in detail_list if d.get("type") and d.get("summary")
            ]
        elif out.get("summary"):
            rec["type"] = out.get("type") or "detail"
            rec["summary"] = out["summary"]
        anns.append(rec)
    return anns


def cache_predictions(conv_paths: List[Path], llm, pred_dir: Path,
                      use_mock: bool, refresh: bool) -> None:
    """Ensure every conversation has cached C1 predictions on disk."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    config = AtlasConfig()
    config.buffer.budget = 999999  # no eviction during prediction capture

    for path in conv_paths:
        with open(path, encoding="utf-8") as f:
            conv = json.load(f)
        cache_file = pred_dir / f"{conv['id']}.json"
        if cache_file.exists() and not refresh:
            continue

        if use_mock:
            gold = {a["utterance_id"]: a for a in conv.get("annotations", [])}
            conv_llm = MockLLM(config, gold)
        else:
            conv_llm = llm  # shared LoRA

        harness = StreamingHarness(config)
        harness.load_conversation_dict(conv)
        result = harness.run(conv_llm, tracker=None, eviction_strategy=None)

        anns = _predicted_annotations(conv, result)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"conversation_id": conv["id"], "annotations": anns},
                      f, ensure_ascii=False)
        print(f"  [cache] {conv['id']}: {len(anns)} predictions", flush=True)


# =============================================================================
# Phase B — replay grid on CPU using cached predictions
# =============================================================================

class ReplayLLM(MockLLM):
    """Replays cached C1 predictions, but counts tokens with the REAL tokenizer
    so token budgets B are meaningful (MockLLM's default is char-based)."""

    def __init__(self, config, annotations, tokenizer):
        super().__init__(config, annotations)
        self._tok = tokenizer

    def get_token_count(self, text: str) -> int:
        if not text:
            return 1
        if self._tok is None:
            return super().get_token_count(text)
        return max(len(self._tok(text, add_special_tokens=False)["input_ids"]), 1)


def _load_tokenizer(adapter_dir: str):
    """Load ONLY the tokenizer (cheap, CPU, no peft) for token counting."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(adapter_dir)
    except Exception as e:
        print(f"  [warn] tokenizer load failed ({e}); using char-based counts.")
        return None


def run_condition(conv_paths: List[Path], pred_dir: Path, tokenizer,
                  strategy_name: str, budget: int, mlp_path: Optional[str],
                  window: int) -> Dict:
    """Replay all conversations for one (strategy, budget) and average completeness."""
    config = AtlasConfig()
    config.buffer.budget = 999999 if strategy_name == "none" else budget
    strategy = build_eviction(strategy_name, mlp_path, window)

    per_conv = []
    for path in conv_paths:
        with open(path, encoding="utf-8") as f:
            conv = json.load(f)
        cache_file = pred_dir / f"{conv['id']}.json"
        if not cache_file.exists():
            continue
        with open(cache_file, encoding="utf-8") as f:
            pred = json.load(f)
        annotations = {a["utterance_id"]: a for a in pred["annotations"]}

        llm = ReplayLLM(config, annotations, tokenizer)
        tracker = SystemTracker(config.tracker)
        harness = StreamingHarness(config)
        harness.load_conversation_dict(conv)
        result = harness.run(llm, tracker=tracker, eviction_strategy=strategy)

        final_buffer = result.buffer_snapshots[-1] if result.buffer_snapshots else []
        gold_agenda = _gold_agenda_items(conv)
        c = agenda_completeness(final_buffer, gold_agenda)
        if c is not None:
            per_conv.append(c)

    avg = sum(per_conv) / len(per_conv) if per_conv else None
    return {
        "strategy": strategy_name,
        "budget": "unlimited" if strategy_name == "none" else budget,
        "agenda_completeness": avg,
        "conversations": len(per_conv),
    }


# =============================================================================
# Output — CSV + plot
# =============================================================================

def write_csv(rows: List[dict], out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "budget",
                                          "agenda_completeness", "conversations"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"csv  : {out_path}")


def plot_rows(rows: List[dict], budgets: List[int], out_png: str, no_show: bool) -> None:
    import matplotlib
    if no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = []
    for r in rows:
        if r["strategy"] not in strategies:
            strategies.append(r["strategy"])

    budgets_sorted = sorted(budgets)
    xpos = list(range(len(budgets_sorted)))  # equal spacing (100/128 don't overlap)

    plt.figure(figsize=(8, 5.5))
    for strat in strategies:
        by_b = {r["budget"]: r["agenda_completeness"] for r in rows
                if r["strategy"] == strat}
        if strat == "none":
            val = by_b.get("unlimited")
            if val is not None:
                plt.axhline(val, color="gray", ls="--", lw=1, alpha=0.8,
                            label="none (no C3, unlim)")
            continue
        ys = [by_b.get(b) for b in budgets_sorted]
        is_mlp = strat == "learned"
        plt.plot(xpos, ys, marker="o",
                 lw=3 if is_mlp else 1.5, zorder=5 if is_mlp else 3,
                 label=strat + (" (MLP)" if is_mlp else ""))

    plt.xticks(xpos, [str(b) for b in budgets_sorted])
    plt.xlabel("Token budget B")
    plt.ylabel("agenda completeness (final buffer)")
    plt.title("Agenda completeness vs budget (C1+C2, swap C3)")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"png  : {out_png}")
    if not no_show:
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    default_csv = os.path.join(_HERE, "results", "agenda_sweep.csv")
    default_png = os.path.join(_HERE, "results", "agenda_sweep.png")

    p = argparse.ArgumentParser(description="Agenda-completeness C3 sweep over budgets.")
    p.add_argument("--input", required=True, help="Conversation folder (e.g. data/valid_aci)")
    p.add_argument("--adapter", default=str(DEFAULT_ADAPTER), help="LoRA adapter dir")
    p.add_argument("--base-model", default=None)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--system-prompt-file", default=None,
                   help="System prompt matching SFT training")
    p.add_argument("--max-new-tokens", type=int, default=256)

    p.add_argument("--budgets", nargs="+", type=int, default=[100, 128, 256, 512])
    p.add_argument("--strategies", nargs="+",
                   default=["none", "fifo", "random", "oldest", "learned"],
                   choices=["none", "fifo", "sliding", "random", "oldest",
                            "learned", "attention"])
    p.add_argument("--mlp", default=None, help="MLP .pt (required if 'learned' in --strategies)")
    p.add_argument("--window", type=int, default=10, help="Sliding window K")

    p.add_argument("--mock", action="store_true",
                   help="Use gold annotations instead of the LoRA (no GPU) for Phase A")
    p.add_argument("--predictions-dir", default=None,
                   help="Where to cache C1 predictions (default: experiments/predictions/<input>)")
    p.add_argument("--refresh", action="store_true", help="Recompute cached predictions")
    p.add_argument("--limit", type=int, default=None, help="Only first N conversations")
    p.add_argument("--output", default=default_csv, help="CSV path")
    p.add_argument("--plot", default=default_png, help="PNG path")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    if "learned" in args.strategies and not args.mlp:
        raise SystemExit("--mlp <path> is required when 'learned' is in --strategies")

    conv_paths = sorted(Path(args.input).rglob("*.json"))
    if args.limit:
        conv_paths = conv_paths[:args.limit]
    if not conv_paths:
        raise SystemExit(f"No conversation JSON under {args.input}")

    # Predictions cache dir keyed by the input folder name.
    if args.predictions_dir:
        pred_dir = Path(args.predictions_dir)
    else:
        tag = ("mock_" if args.mock else "lora_") + Path(args.input).name
        pred_dir = Path(_HERE) / "predictions" / tag

    print(f"conversations : {len(conv_paths)}")
    print(f"strategies    : {args.strategies}")
    print(f"budgets       : {args.budgets}")
    print(f"predictions   : {pred_dir}  (mock={args.mock})")

    # ---- Phase A: capture C1 predictions once ----
    llm = None
    need_llm = not all((pred_dir / f"{p.stem}.json").exists() for p in conv_paths) or args.refresh
    if need_llm and not args.mock:
        system_prompt = None
        if args.system_prompt_file:
            system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
        llm = LoraAgendaLLM(
            AtlasConfig(), adapter_dir=args.adapter, base_model=args.base_model,
            device=args.device, system_prompt=system_prompt,
            max_new_tokens=args.max_new_tokens,
        )
    print("\n[Phase A] caching C1 predictions ...")
    cache_predictions(conv_paths, llm, pred_dir, use_mock=args.mock, refresh=args.refresh)

    # ---- Phase B: replay the grid on CPU ----
    tokenizer = _load_tokenizer(args.adapter)
    print("\n[Phase B] sweeping (strategy x budget) ...")
    rows = []
    for strat in args.strategies:
        if strat == "none":
            r = run_condition(conv_paths, pred_dir, tokenizer, "none",
                              999999, args.mlp, args.window)
            rows.append(r)
            print(f"  none (no C3)          completeness={_fmt(r['agenda_completeness'])}")
            continue
        for b in args.budgets:
            r = run_condition(conv_paths, pred_dir, tokenizer, strat, b,
                              args.mlp, args.window)
            rows.append(r)
            print(f"  {strat:<8} B={b:<5} completeness={_fmt(r['agenda_completeness'])}")

    write_csv(rows, args.output)
    plot_rows(rows, args.budgets, args.plot, args.no_show)


def _fmt(v):
    return " n/a " if v is None else f"{v*100:5.1f}%"


if __name__ == "__main__":
    main()
