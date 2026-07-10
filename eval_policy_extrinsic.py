"""
ATLAS Extrinsic Evaluation (Component 1: Agenda LLM)
======================================================
Runs REAL LLM inference (HF transformers + PEFT LoRA, meant for a GPU
box such as a RunPod pod pulled from GitHub) through the streaming
harness and scores the output against gold annotations using every
metric in src/metrics.py (ROUGE, BERTScore, detection P/R/F1, agenda
completeness, linking/first-mention/resolution accuracy).

Compares, at budgets 100 / 256 / 512:
  - "baseline" : meta-llama/Llama-3.2-3B-Instruct, no adapter (zero-shot)
  - "C1"       : the same base model + AAA_atlas_c1_lora_adapter_checkpoint100
  - every model passed via --model (local LoRA adapter dir, local full
    model dir, or a HF hub id — auto-detected)

Only --model is meant to be supplied by hand; everything else is a
fixed, overridable default.

Usage:
  python eval_policy_extrinsic.py
  python eval_policy_extrinsic.py --model runs/my_lora_ckpt200 runs/my_lora_ckpt300
  python eval_policy_extrinsic.py --model someuser/some-hub-adapter

  # Use the trained Component 3 eviction policy instead of the fifo default:
  python eval_policy_extrinsic.py --strategy learned --policy-model models/policy/eviction_mlp_bertscore_0p75.pt

Outputs (output/ by default):
  - eval_policy_extrinsic_results.json  full nested results
  - eval_policy_extrinsic_results.csv   long-format (model, budget, metric, value)
  - plots/<metric>/<model>.png          one PNG per (metric, model): budget on
                                         the x-axis. baseline gets its own image
                                         too, so every model (incl. baseline) is
                                         directly comparable file-by-file.

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
DEFAULT_C1_ADAPTER = PROJECT_ROOT / "AAA_atlas_c1_lora_adapter_checkpoint100"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "challenge_data" / "valid" / "eval"
DEFAULT_BUDGETS = [100, 256, 512]
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "eval_policy_extrinsic_results.json"
DEFAULT_CSV = PROJECT_ROOT / "output" / "eval_policy_extrinsic_results.csv"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "output" / "plots"

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
            raise ValueError("--strategy learned requires --policy-model <path to a trained .pt checkpoint>")
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


def print_report(label, budget, report):
    print(f"\n--- {label} | budget={budget} | "
          f"{report.get('n_conversations', 0)} conversations | "
          f"{report.get('elapsed_sec', 0)}s ---")
    printable = {k: v for k, v in report.items() if k not in ("n_conversations", "elapsed_sec")}
    print(json.dumps(printable, indent=2, ensure_ascii=False))


# =============================================================================
# CSV + PLOT EXPORT
# =============================================================================

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


def write_csv(all_results: dict, csv_path: Path):
    """Long-format CSV: one row per (model, budget, metric) — includes every
    value in the JSON output (metrics.py results + hardware/meta fields)."""
    rows = []
    for label, per_budget in all_results.items():
        for budget, report in per_budget.items():
            for metric, value in flatten_metrics(report).items():
                rows.append({"model": label, "budget": budget, "metric": metric, "value": value})

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "budget", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to {csv_path} ({len(rows)} rows)")


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def write_plots(all_results: dict, plot_dir: Path):
    """One PNG per (metric, model): x=budget, single line for that model.
    Saved as plot_dir/<metric>/<model>.png. Baseline gets its own image too
    (like every other model), so it's directly comparable file-by-file."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping plot generation "
              "(pip install matplotlib)", file=sys.stderr)
        return

    # pivot: metric -> label -> {budget: value}
    pivot = {}
    for label, per_budget in all_results.items():
        for budget, report in per_budget.items():
            for metric, value in flatten_metrics(report).items():
                if metric in META_KEYS:
                    continue
                pivot.setdefault(metric, {}).setdefault(label, {})[budget] = value

    plot_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    for metric, by_label in sorted(pivot.items()):
        metric_dir = plot_dir / _safe_filename(metric)
        metric_dir.mkdir(parents=True, exist_ok=True)

        for label, budget_values in by_label.items():
            if not budget_values:
                continue
            budgets = sorted(budget_values)
            values = [budget_values[b] for b in budgets]

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(budgets, values, marker="o", label=label)
            ax.set_xlabel("Budget (tokens)")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} — {label}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(metric_dir / f"{_safe_filename(label)}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            n_saved += 1

    print(f"Saved {n_saved} (metric, model) plots to {plot_dir}")


def main():
    parser = argparse.ArgumentParser(description="ATLAS extrinsic (Component 1) evaluation")
    parser.add_argument("--model", nargs="+", default=[],
                         help="Additional model(s) to evaluate: local LoRA adapter dir, "
                              "local full model dir, or HF hub id.")

    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--strategy", default="fifo",
                         choices=["fifo", "sliding", "random", "oldest", "attention", "learned"],
                         help="eviction strategy applied to the context buffer. "
                              "'learned' uses the trained Component 3 EvictionMLP (see --policy-model).")
    parser.add_argument("--window-size", type=int, default=10, help="only used when --strategy sliding")
    parser.add_argument("--policy-model", default=None,
                         help="path to a trained EvictionMLP .pt checkpoint (required when --strategy learned)")
    parser.add_argument("--policy-no-scispacy", action="store_true",
                         help="disable scispaCy entity_count in the learned policy's features "
                              "(only safe if the checkpoint was also trained without it)")

    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--c1-adapter", default=str(DEFAULT_C1_ADAPTER))
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-c1", action="store_true")

    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--hf-token", default=None)

    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="limit number of conversations (debug)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSON output path")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="CSV output path")
    parser.add_argument("--plot-dir", default=str(DEFAULT_PLOT_DIR), help="directory for per-metric PNG plots")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.strategy == "learned" and not args.policy_model:
        parser.error("--strategy learned requires --policy-model <path to a trained .pt checkpoint>")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    strategy = build_eviction_strategy(
        args.strategy, args.window_size, args.policy_model, not args.policy_no_scispacy
    )

    data_dir = Path(args.data_dir)
    convs = load_conversations(data_dir, limit=args.limit)
    if not convs:
        print(f"No conversation JSON files found in {data_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(convs)} conversations from {data_dir}")

    model_jobs = []  # (label, spec_or_None)
    if not args.skip_baseline:
        model_jobs.append(("baseline", None))
    if not args.skip_c1:
        model_jobs.append(("C1", args.c1_adapter))
    for m in args.model:
        model_jobs.append((Path(m).name or m, m))

    all_results = {}
    for label, spec in model_jobs:
        print(f"\n=== Model: {label} ===")
        if spec is None:
            base_for_load, adapter_path, tok_source = args.base_model, None, args.base_model
        else:
            base_for_load, adapter_path, tok_source = resolve_model_spec(spec, args.base_model, args.hf_token)

        llm = build_llm(label, base_for_load, adapter_path, tok_source,
                         dtype, args.load_in_4bit, args.hf_token, args.max_new_tokens)

        per_budget = run_model_across_budgets(
            label, llm, convs, args.budgets, strategy, args.skip_bertscore
        )
        for budget, report in per_budget.items():
            print_report(label, budget, report)

        all_results[label] = per_budget
        free_llm(llm)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full results to {output_path}")

    if not args.no_csv:
        write_csv(all_results, Path(args.csv))

    if not args.no_plots:
        write_plots(all_results, Path(args.plot_dir))


if __name__ == "__main__":
    main()
