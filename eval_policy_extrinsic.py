"""
ATLAS Extrinsic Evaluation (Component 1: Agenda LLM)
======================================================
Runs REAL LLM inference (HF transformers + PEFT LoRA, meant for a GPU
box such as a RunPod pod pulled from GitHub) through the streaming
harness and scores the output against gold annotations using every
metric in src/metrics.py (ROUGE, BERTScore, detection P/R/F1, agenda
completeness, linking/first-mention/resolution accuracy).

For every eviction strategy in STRATEGIES_TO_COMPARE (sliding, random,
oldest, attention, learned), compares:
  - "baseline" : meta-llama/Llama-3.2-3B-Instruct, no adapter (zero-shot)
  - "model"    : --model-path (default: tria-hongik/atlas-c1-v2-llama-3.2-3b,
                 the C1 LoRA adapter on HF hub)

Results are cached in a CSV keyed by (strategy, model, budget) — combos
already present are skipped, so reruns only fill in what's missing. Run
eval_policy_baseline.py first to pre-fill the "baseline" rows cheaply;
this script then only needs to run "model" (plus any baseline+learned
combo eval_policy_baseline.py didn't cover) and appends those rows below.

Usage:
  python eval_policy_extrinsic.py --policy-model models/policy/eviction_mlp_bertscore_0p75_5e-3.pt
  python eval_policy_extrinsic.py --policy-model ... --model-path someuser/some-hub-adapter
  python eval_policy_extrinsic.py --policy-model ... --force-rerun   # ignore cache, recompute everything

Outputs (output/ by default):
  - strategy_comparison.csv                      long-format cache/result
                                                   (strategy, model, budget, metric, value)
  - plots_strategy_comparison/<metric>/budgetN_bar.png          baseline vs model bars, x=strategy
  - plots_strategy_comparison/<metric>/budgetN_improvement.png  % improvement of model over baseline, x=strategy

Requirements (RunPod GPU box):
  pip install torch transformers peft accelerate bitsandbytes \
              rouge-score bert-score sentence-transformers matplotlib
"""

import argparse
import csv
import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from config import AtlasConfig
from harness import StreamingHarness, AgendaLLM, LLMOutput
from baselines import get_strategy
from state_tracker import SystemTracker
from metrics import (
    compute_rouge,
    compute_bertscore,
    compute_detection_metrics,
    compute_agenda_completeness,
    compute_linking_accuracy,
    compute_first_mention_accuracy,
    compute_resolution_accuracy,
)
from prompts import REALTIME_SYSTEM, REALTIME_USER

# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_C1_ADAPTER = "tria-hongik/atlas-c1-v2-llama-3.2-3b"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "aci"
DEFAULT_BUDGETS = [100, 256, 512]
DEFAULT_CSV = PROJECT_ROOT / "output" / "strategy_comparison.csv"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "output" / "plots_strategy_comparison"

STRATEGIES_TO_COMPARE = ["sliding", "random", "oldest", "attention", "learned"]
CSV_FIELDS = ["strategy", "model", "budget", "metric", "value"]

# meta fields tracked in every report but not real metrics.py values —
# excluded from the per-metric comparison plots.
META_KEYS = {"n_conversations", "elapsed_sec"}


# =============================================================================
# REAL AGENDA LLM (transformers + optional PEFT LoRA adapter)
# =============================================================================

class HFAgendaLLM(AgendaLLM):
    """Agenda LLM backed by a real HF transformers model (+ optional LoRA)."""

    def __init__(self, config: AtlasConfig, tokenizer, model, label: str):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.model = model
        self.label = label
        self.device = next(model.parameters()).device

    def get_token_count(self, text: str) -> int:
        # Real tokenizer, not harness.py's char/4 placeholder.
        return max(len(self.tokenizer.encode(text, add_special_tokens=False)), 1)

    def predict(self, context: str, speaker: str, text: str) -> LLMOutput:
        messages = [
            {"role": "system", "content": REALTIME_SYSTEM},
            {"role": "user", "content": REALTIME_USER.format(context=context, speaker=speaker, text=text)},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.config.llm.max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        raw = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> LLMOutput:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return LLMOutput(relevant=False)
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return LLMOutput(relevant=False)

        if not obj.get("relevant"):
            return LLMOutput(relevant=False)

        detail_list = [
            d for d in (obj.get("detail_list") or [])
            if isinstance(d, dict) and d.get("type") and d.get("summary")
        ]
        if not detail_list:
            return LLMOutput(relevant=False)

        return LLMOutput(
            relevant=True,
            detail_list=detail_list,
            type=detail_list[0]["type"],
            summary=detail_list[0]["summary"],
        )


def resolve_model_spec(spec: str, base_model_id: str, hf_token: str):
    """
    Figure out whether `spec` is a LoRA adapter (local dir or hub repo with
    adapter_config.json) or a standalone full model. Returns
    (base_model_for_loading, adapter_path_or_None, tokenizer_source).
    """
    local_path = Path(spec)
    adapter_config_path = None

    if local_path.is_dir() and (local_path / "adapter_config.json").exists():
        adapter_config_path = local_path / "adapter_config.json"
    elif not local_path.exists():
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id=spec, filename="adapter_config.json", token=hf_token)
            adapter_config_path = Path(downloaded)
        except Exception:
            adapter_config_path = None

    if adapter_config_path is not None:
        with open(adapter_config_path, encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        base = adapter_cfg.get("base_model_name_or_path") or base_model_id
        return base, spec, spec

    return spec, None, spec


def build_llm(label: str, base_for_load: str, adapter_path, tokenizer_source: str,
              dtype: torch.dtype, load_in_4bit: bool, hf_token: str, max_new_tokens: int) -> HFAgendaLLM:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  [{label}] tokenizer <- {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4",
        )

    print(f"  [{label}] base model <- {base_for_load}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_for_load, dtype=dtype, device_map="auto",
            quantization_config=quantization_config, token=hf_token,
        )
    except TypeError:
        # older transformers: kwarg is `torch_dtype`, not `dtype`
        model = AutoModelForCausalLM.from_pretrained(
            base_for_load, torch_dtype=dtype, device_map="auto",
            quantization_config=quantization_config, token=hf_token,
        )

    if adapter_path:
        from peft import PeftModel
        print(f"  [{label}] LoRA adapter <- {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    llm_config = AtlasConfig()
    llm_config.llm.max_tokens = max_new_tokens
    return HFAgendaLLM(llm_config, tokenizer, model, label=label)


def free_llm(llm: HFAgendaLLM):
    del llm.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# DATA
# =============================================================================

def load_conversations(data_dir: Path, limit=None):
    convs = []
    for f in sorted(data_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fp:
            convs.append(json.load(fp))
    if limit:
        convs = convs[:limit]
    return convs


# =============================================================================
# REPORT BUILDING (uses metrics.py functions directly — generate_report()
# assumes flat type/summary annotations, but this dataset uses detail_list,
# so predictions and gold are flattened here before scoring)
# =============================================================================

def build_report(result, conv: dict, skip_bertscore: bool) -> dict:
    utterances = conv["utterances"]
    gold_by_uid = {ann["utterance_id"]: ann for ann in conv.get("annotations", [])}
    pred_outputs = result.llm_outputs

    pred_types, gold_types = [], []
    aligned_pred_summaries, aligned_gold_summaries = [], []
    all_pred_summaries = []
    all_gold_agenda_summaries = []

    for i, utt in enumerate(utterances):
        pred = pred_outputs[i] if i < len(pred_outputs) else {"relevant": False}
        gold = gold_by_uid.get(utt["id"], {"relevant": False})

        pred_details = pred.get("detail_list") or []
        gold_details = gold.get("detail_list") or []

        all_pred_summaries.extend(d["summary"] for d in pred_details if d.get("summary"))
        all_gold_agenda_summaries.extend(
            d["summary"] for d in gold_details if d.get("type") == "agenda_item" and d.get("summary")
        )

        # Index-paired within the utterance so compute_detection_metrics
        # (which expects two same-length 1:1 label lists) gets sensible input.
        n = max(len(pred_details), len(gold_details), 1)
        for j in range(n):
            p = pred_details[j] if j < len(pred_details) else None
            g = gold_details[j] if j < len(gold_details) else None
            pred_types.append(p["type"] if p else "irrelevant")
            gold_types.append(g["type"] if g else "irrelevant")
            if p and g and p.get("summary") and g.get("summary"):
                aligned_pred_summaries.append(p["summary"])
                aligned_gold_summaries.append(g["summary"])

    report = {}

    if aligned_pred_summaries:
        report["rouge"] = compute_rouge(aligned_pred_summaries, aligned_gold_summaries)
        if not skip_bertscore:
            report["bertscore"] = compute_bertscore(aligned_pred_summaries, aligned_gold_summaries)

    report["detection"] = compute_detection_metrics(pred_types, gold_types)

    if all_gold_agenda_summaries:
        report["agenda_completeness"] = compute_agenda_completeness(
            all_pred_summaries, all_gold_agenda_summaries
        )

    # System-level (linking / first-mention / resolution) — only populated
    # when gold `_eval_only` fields exist in the source data.
    tracker_state = result.tracker_states[-1] if result.tracker_states else {}
    tracker_preds = tracker_state.get("predictions", [])

    eval_by_id = {}
    for utt in utterances:
        gold = gold_by_uid.get(utt["id"], {})
        for d in (gold.get("detail_list") or []):
            if d.get("eval_only"):
                eval_by_id[utt["id"]] = d["eval_only"]
                break

    sys_links, gold_links = [], []
    sys_first, gold_first = [], []
    sys_res, gold_res = [], []
    for pred in tracker_preds:
        uid = pred["utterance_id"]
        if uid in eval_by_id and pred.get("relevant"):
            ge = eval_by_id[uid]
            if pred["linked_agenda_item_id"] is not None:
                sys_links.append(pred["linked_agenda_item_id"])
                gold_links.append(ge.get("linked_agenda_item_id"))
            if pred["first_mention"] is not None:
                sys_first.append(pred["first_mention"])
                gold_first.append(ge.get("first_mention"))
            if pred["resolution_status"] is not None:
                sys_res.append(pred["resolution_status"])
                gold_res.append(ge.get("resolution_status"))

    system_level = {}
    if sys_links:
        system_level["linking_accuracy"] = compute_linking_accuracy(sys_links, gold_links)
    if sys_first:
        system_level["first_mention_accuracy"] = compute_first_mention_accuracy(sys_first, gold_first)
    if sys_res:
        system_level["resolution_accuracy"] = compute_resolution_accuracy(sys_res, gold_res)
    if system_level:
        report["system_level"] = system_level

    report["hardware"] = {
        "avg_latency_ms": result.metadata.get("avg_latency_ms", 0.0),
        "total_relevant": result.metadata.get("total_relevant", 0),
        "final_buffer_size": result.metadata.get("final_buffer_size", 0),
        "final_buffer_tokens": result.metadata.get("final_buffer_tokens", 0),
    }
    return report


def average_reports(reports: list) -> dict:
    """Recursively average numeric leaves across per-conversation reports.
    Missing keys in a given conversation are skipped, not treated as 0."""
    buckets = {}

    def collect(d, prefix):
        for k, v in d.items():
            path = prefix + (k,)
            if isinstance(v, dict):
                collect(v, path)
            elif isinstance(v, (int, float)):
                buckets.setdefault(path, []).append(float(v))

    for r in reports:
        collect(r, ())

    avg = {}
    for path, vals in buckets.items():
        node = avg
        for p in path[:-1]:
            node = node.setdefault(p, {})
        node[path[-1]] = sum(vals) / len(vals)
    return avg


# =============================================================================
# EVICTION STRATEGY (baseline, or the trained Component 3 policy)
# =============================================================================

def build_eviction_strategy(strategy_name: str, window_size: int, policy_model: str, use_scispacy: bool):
    """
    "learned" runs the trained EvictionMLP (src/policy.py, Component 3)
    instead of a baseline. Built once and reused across every model/budget —
    the eviction policy is orthogonal to which Agenda LLM is under test.
    """
    if strategy_name == "learned":
        if not policy_model:
            raise ValueError("strategy 'learned' requires --policy-model <path to a trained .pt checkpoint>")
        from policy import LearnedEvictionStrategy
        print(f"Loading learned eviction policy from {policy_model} (use_scispacy={use_scispacy})")
        return LearnedEvictionStrategy(model_path=policy_model, use_scispacy=use_scispacy)

    if strategy_name == "sliding":
        return get_strategy("sliding", window_size=window_size)

    return get_strategy(strategy_name)


# =============================================================================
# MAIN EVAL LOOP
# =============================================================================

def run_model_across_budgets(label, llm, convs, budgets, strategy, skip_bertscore):
    per_budget = {}
    for budget in budgets:
        config = AtlasConfig()
        config.buffer.budget = budget

        reports = []
        t0 = time.time()
        for ci, conv in enumerate(convs):
            tracker = SystemTracker()
            harness = StreamingHarness(config).load_conversation_dict(conv)
            result = harness.run(llm, tracker=tracker, eviction_strategy=strategy)
            reports.append(build_report(result, conv, skip_bertscore))
            print(f"    [{label}] budget={budget} conv {ci+1}/{len(convs)} "
                  f"({conv.get('id', '?')}) done", flush=True)

        per_budget[budget] = {
            "n_conversations": len(reports),
            "elapsed_sec": round(time.time() - t0, 1),
            **average_reports(reports),
        }
    return per_budget


def flatten_metrics(d: dict, prefix: str = "") -> dict:
    """Nested report dict -> {"rouge.rouge1": 0.42, "detection.detail.f1": 0.7, ...}"""
    flat = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_metrics(v, path))
        elif isinstance(v, (int, float)):
            flat[path] = float(v)
    return flat


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


# =============================================================================
# CSV CACHE — (strategy, model, budget) already computed -> skip re-running.
# Lets eval_policy_baseline.py pre-fill the "baseline" rows separately, and
# this script only fills in whatever is still missing (normally "model").
# =============================================================================

def load_cache(csv_path: Path):
    rows = []
    done = set()
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add((r["strategy"], r["model"], int(r["budget"])))
    return rows, done


def append_rows(csv_path: Path, rows: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# COMPARISON PLOTS — baseline vs model, x-axis = strategy
# =============================================================================

def write_plots(rows: list, plot_dir: Path):
    """One pair of PNGs per (metric, budget): a grouped bar (baseline vs
    model, x=strategy) and an improvement-% bar ((model-baseline)/|baseline|,
    x=strategy). Only strategies with both a baseline and a model row are
    plotted."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping plot generation "
              "(pip install matplotlib)", file=sys.stderr)
        return

    # pivot[metric][budget][strategy][model] = value
    pivot = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        metric = r["metric"]
        if metric in META_KEYS:
            continue
        pivot[metric][int(r["budget"])][r["strategy"]][r["model"]] = float(r["value"])

    plot_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    for metric, by_budget in sorted(pivot.items()):
        metric_dir = plot_dir / _safe_filename(metric)
        metric_dir.mkdir(parents=True, exist_ok=True)

        for budget, by_strategy in sorted(by_budget.items()):
            strategies = [
                s for s in STRATEGIES_TO_COMPARE
                if "baseline" in by_strategy.get(s, {}) and "model" in by_strategy.get(s, {})
            ]
            if not strategies:
                continue

            baseline_vals = [by_strategy[s]["baseline"] for s in strategies]
            model_vals = [by_strategy[s]["model"] for s in strategies]

            # (a) grouped bar: baseline vs model per strategy
            x = range(len(strategies))
            width = 0.35
            fig, ax = plt.subplots(figsize=(max(5, len(strategies) * 1.5), 4))
            ax.bar([i - width / 2 for i in x], baseline_vals, width, label="baseline")
            ax.bar([i + width / 2 for i in x], model_vals, width, label="model")
            ax.set_xticks(list(x))
            ax.set_xticklabels(strategies)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} — baseline vs model (budget={budget})")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            fig.savefig(metric_dir / f"budget{budget}_bar.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            n_saved += 1

            # (b) improvement %: (model - baseline) / |baseline| * 100
            improvements = [
                ((m - b) / abs(b) * 100) if b != 0 else float("nan")
                for b, m in zip(baseline_vals, model_vals)
            ]
            fig, ax = plt.subplots(figsize=(max(5, len(strategies) * 1.5), 4))
            colors = ["#2ca02c" if v >= 0 else "#d62728" for v in improvements]
            ax.bar(strategies, improvements, color=colors)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel(f"{metric} improvement (%)")
            ax.set_title(f"{metric} — model improvement over baseline (budget={budget})")
            ax.grid(True, axis="y", alpha=0.3)
            fig.savefig(metric_dir / f"budget{budget}_improvement.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            n_saved += 1

    print(f"Saved {n_saved} comparison plots to {plot_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ATLAS extrinsic (Component 1) evaluation: baseline vs model, "
                     "across sliding/random/oldest/attention/learned eviction strategies"
    )
    parser.add_argument("--model-path", default=DEFAULT_C1_ADAPTER,
                         help="LoRA adapter dir/hub-id or full model dir for the 'model' side "
                              "(default: C1 hub adapter)")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--policy-model", required=True,
                         help="path to a trained EvictionMLP .pt checkpoint, used for the 'learned' strategy")
    parser.add_argument("--policy-no-scispacy", action="store_true",
                         help="disable scispaCy entity_count in the learned policy's features "
                              "(only safe if the checkpoint was also trained without it)")
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
                         help="cache/result CSV (strategy, model, budget, metric, value)")
    parser.add_argument("--plot-dir", default=str(DEFAULT_PLOT_DIR))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--force-rerun", action="store_true",
                         help="ignore the cache and recompute every (strategy, model, budget) combo")
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    csv_path = Path(args.csv)
    cached_rows, done = ([], set()) if args.force_rerun else load_cache(csv_path)

    convs = load_conversations(Path(args.data_dir), limit=args.limit)
    if not convs:
        print(f"No conversation JSON files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(convs)} conversations from {args.data_dir}")

    model_specs = {"baseline": None, "model": args.model_path}

    new_rows = []
    for strategy_name in STRATEGIES_TO_COMPARE:
        eviction_strategy = build_eviction_strategy(
            strategy_name, args.window_size, args.policy_model, not args.policy_no_scispacy
        )
        for label, spec in model_specs.items():
            missing_budgets = [b for b in args.budgets if (strategy_name, label, b) not in done]
            if not missing_budgets:
                print(f"[{strategy_name}/{label}] all budgets cached in {csv_path}, skipping")
                continue

            print(f"\n=== strategy={strategy_name} model={label} budgets={missing_budgets} ===")
            if spec is None:
                base_for_load, adapter_path, tok_source = args.base_model, None, args.base_model
            else:
                base_for_load, adapter_path, tok_source = resolve_model_spec(spec, args.base_model, args.hf_token)

            llm = build_llm(label, base_for_load, adapter_path, tok_source,
                             dtype, args.load_in_4bit, args.hf_token, args.max_new_tokens)

            per_budget = run_model_across_budgets(
                label, llm, convs, missing_budgets, eviction_strategy, args.skip_bertscore
            )
            free_llm(llm)

            for budget, report in per_budget.items():
                for metric, value in flatten_metrics(report).items():
                    new_rows.append({
                        "strategy": strategy_name, "model": label, "budget": budget,
                        "metric": metric, "value": value,
                    })
                done.add((strategy_name, label, budget))

    if new_rows:
        append_rows(csv_path, new_rows)
        print(f"\nAppended {len(new_rows)} rows to {csv_path}")
    else:
        print(f"\nNothing new to run — everything already cached in {csv_path}.")

    if not args.no_plots:
        write_plots(cached_rows + new_rows, Path(args.plot_dir))


if __name__ == "__main__":
    main()
