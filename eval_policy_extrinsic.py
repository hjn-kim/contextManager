"""
ATLAS Extrinsic Evaluation (Component 1 Agenda LLM, fixed LoRA)
================================================================
Runs REAL LLM inference (HF transformers + PEFT LoRA, meant for a GPU
box such as a RunPod pod pulled from GitHub) through the streaming
harness and scores the output against gold annotations using every
metric in src/metrics.py (ROUGE, BERTScore, detection P/R/F1, agenda
completeness, linking/first-mention/resolution accuracy).

The Agenda LLM is held FIXED to the LoRA model (--model-path, default
tria-hongik/atlas-c1-v2-llama-3.2-3b). What is compared is the EVICTION
STRATEGY on top of that one model:
  - baseline strategies : fifo, sliding, random, attention, oracle
  - learned             : the trained EvictionMLP policy (Component 3)

Results are cached in a CSV keyed by (strategy, budget) — combos already
present are skipped, so reruns only fill in what's missing. Run
eval_policy_baseline.py first to pre-fill the baseline-strategy rows; this
script then only needs to run "learned" (plus any baseline strategy that
wasn't cached) and appends those rows, then plots baselines vs learned.

Usage:
  python eval_policy_extrinsic.py --policy-model models/policy/eviction_mlp_bertscore_0p75_5e-3.pt
  python eval_policy_extrinsic.py --policy-model ... --model-path someuser/some-hub-adapter
  python eval_policy_extrinsic.py --policy-model ... --force-rerun   # ignore cache, recompute everything

Outputs (output/ by default):
  - strategy_comparison.csv                      long-format cache/result
                                                   (strategy, budget, metric, value)
  - plots_strategy_comparison/<metric>/budgetN_bar.png          metric value per strategy (fixed LoRA model)
  - plots_strategy_comparison/<metric>/budgetN_improvement.png  learned's % improvement over each baseline

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

# The Agenda LLM (Component 1) is held FIXED to the LoRA model for the whole
# comparison. What varies is the eviction strategy: the baseline strategies
# (run by eval_policy_baseline.py) vs the trained "learned" policy (Component 3).
BASELINE_STRATEGIES = ["fifo", "sliding", "random", "attention", "oracle"]
STRATEGIES_TO_COMPARE = BASELINE_STRATEGIES + ["learned"]

# Buffer budget large enough that eviction never triggers -> oracle keeps all.
ORACLE_BUDGET = 999_999

CSV_FIELDS = ["strategy", "budget", "metric", "value"]

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
        # Greedy decoding (do_sample=False) is deterministic, so identical
        # (context, speaker, text) inputs always produce the same output. The
        # eviction sweep re-invokes the LLM on the same conversation prefixes
        # across every strategy and budget (buffers only diverge after a budget
        # is first exceeded), so caching removes the bulk of redundant
        # generations. This cache lives on the LLM instance, which is built once
        # and reused across the whole run.
        self._predict_cache = {}
        self._cache_hits = 0

    def get_token_count(self, text: str) -> int:
        # Real tokenizer, not harness.py's char/4 placeholder.
        return max(len(self.tokenizer.encode(text, add_special_tokens=False)), 1)

    def predict(self, context: str, speaker: str, text: str) -> LLMOutput:
        key = (context, speaker, text)
        cached = self._predict_cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        messages = [
            {"role": "system", "content": REALTIME_SYSTEM},
            {"role": "user", "content": REALTIME_USER.format(context=context, speaker=speaker, text=text)},
        ]
        # transformers 5.x makes apply_chat_template return a BatchEncoding
        # (return_dict defaults to True), not a bare tensor — passing that
        # straight into generate() blows up on inputs_tensor.shape. Ask for the
        # dict explicitly and unpack it (also supplies attention_mask).
        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **enc,
                max_new_tokens=self.config.llm.max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        raw = self.tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
        output = self._parse(raw)
        self._predict_cache[key] = output
        return output

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


def log_cache_stats(llm: HFAgendaLLM):
    """Report how many LLM generations were served from the deterministic
    prediction cache — i.e. how much cross-strategy/budget work was redundant."""
    hits = getattr(llm, "_cache_hits", 0)
    unique = len(getattr(llm, "_predict_cache", {}))
    total = hits + unique
    if total:
        print(f"LLM prediction cache: {hits}/{total} calls served from cache "
              f"({100 * hits / total:.1f}% redundant), {unique} unique generations.")


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

    # agenda_completeness is itself a bert-score (deberta-xlarge) metric, so
    # --skip-bertscore skips it too — otherwise "skip bertscore" still pays the
    # single most expensive metric on every conversation.
    if all_gold_agenda_summaries and not skip_bertscore:
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
    "oracle" keeps everything (no eviction, unlimited-budget upper bound).
    "learned" runs the trained EvictionMLP (src/policy.py, Component 3)
    instead of a baseline. Built once and reused across every budget —
    the eviction policy is orthogonal to which Agenda LLM is under test.
    """
    if strategy_name == "oracle":
        return None  # unlimited-budget upper bound — the harness never evicts

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

def _result_signature(conv, result):
    """Everything build_report's output depends on, EXCEPT latency (noise).

    Two runs with the same signature produce the same rouge/detection/
    bertscore/agenda/system-level and the same buffer-size/token counts, so the
    report can be reused. Different (strategy, budget) combos frequently yield
    identical trajectories — e.g. any conversation that never exceeds a budget,
    or sliding at budgets that don't bind the window — and this skips
    recomputing their (often deberta-backed) metrics."""
    outs = tuple(
        (
            bool(o.get("relevant")),
            tuple((d.get("type"), d.get("summary")) for d in (o.get("detail_list") or [])),
        )
        for o in result.llm_outputs
    )
    tracker = result.tracker_states[-1] if result.tracker_states else {}
    preds = tuple(
        (
            p.get("utterance_id"),
            bool(p.get("relevant")),
            p.get("linked_agenda_item_id"),
            p.get("first_mention"),
            p.get("resolution_status"),
        )
        for p in (tracker.get("predictions") or [])
    )
    meta = result.metadata or {}
    return (
        conv.get("id"),
        outs,
        preds,
        meta.get("final_buffer_size"),
        meta.get("final_buffer_tokens"),
    )


_SHARED_TRACKER_ENCODER = None
_SHARED_ENCODER_FAILED = False


def _shared_tracker_encoder():
    """Load the SystemTracker sentence encoder once and reuse it everywhere.

    run_model_across_budgets builds a fresh SystemTracker per conversation (each
    needs its own agenda state), and SystemTracker lazily loads a
    SentenceTransformer on first link — so without sharing, that model is
    reloaded for EVERY conversation (the repeated 'Loading weights' spam and a
    big slowdown). Returns None if it can't be built, in which case callers fall
    back to per-tracker lazy loading."""
    global _SHARED_TRACKER_ENCODER, _SHARED_ENCODER_FAILED
    if _SHARED_TRACKER_ENCODER is not None or _SHARED_ENCODER_FAILED:
        return _SHARED_TRACKER_ENCODER
    try:
        from sentence_transformers import SentenceTransformer
        model_name = SystemTracker().config.embedding_model
        _SHARED_TRACKER_ENCODER = SentenceTransformer(model_name)
    except Exception as e:
        print(f"WARNING: could not preload tracker encoder ({type(e).__name__}: {e})")
        _SHARED_ENCODER_FAILED = True
    return _SHARED_TRACKER_ENCODER


def run_model_across_budgets(label, llm, convs, budgets, strategy, skip_bertscore,
                             report_cache=None):
    # report_cache is keyed by trajectory signature and shared across the whole
    # run, so identical (strategy, budget, conv) outcomes score their metrics
    # only once. Pass the same dict across every call to dedupe across budgets
    # AND strategies. Note: reused reports keep the FIRST run's avg_latency_ms —
    # latency is not meaningful once predictions are cached anyway.
    if report_cache is None:
        report_cache = {}

    per_budget = {}
    for budget in budgets:
        config = AtlasConfig()
        config.buffer.budget = budget

        reports = []
        t0 = time.time()
        for ci, conv in enumerate(convs):
            tracker = SystemTracker()
            encoder = _shared_tracker_encoder()
            if encoder is not None:
                tracker._encoder = encoder  # reuse one model across all trackers
            harness = StreamingHarness(config).load_conversation_dict(conv)
            result = harness.run(llm, tracker=tracker, eviction_strategy=strategy)

            sig = _result_signature(conv, result)
            report = report_cache.get(sig)
            cached = report is not None
            if not cached:
                report = build_report(result, conv, skip_bertscore)
                report_cache[sig] = report
            reports.append(report)
            print(f"    [{label}] budget={budget} conv {ci+1}/{len(convs)} "
                  f"({conv.get('id', '?')}) {'cached' if cached else 'scored'}", flush=True)

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
# CSV CACHE — (strategy, budget) already computed -> skip re-running.
# Lets eval_policy_baseline.py pre-fill the baseline-strategy rows separately,
# and this script only fills in whatever is still missing (normally "learned").
# =============================================================================

def _csv_header(csv_path: Path):
    """Return the header row of an existing CSV, or None if absent/empty."""
    if not csv_path.exists():
        return None
    with open(csv_path, encoding="utf-8", newline="") as f:
        return next(csv.reader(f), None)


def load_cache(csv_path: Path):
    rows = []
    done = set()
    header = _csv_header(csv_path)
    if header is None:
        return rows, done
    if header != CSV_FIELDS:
        # An old/foreign schema (e.g. the previous strategy,model,budget,...
        # layout). Reading 4-field rows against a 5-field header shifts columns
        # and blows up int(budget). Ignore it as cache; append_rows() will back
        # it up and start a fresh CSV with the current schema.
        print(f"WARNING: {csv_path} has an incompatible schema {header} "
              f"(expected {CSV_FIELDS}); ignoring it as cache and rewriting fresh.")
        return rows, done
    with open(csv_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                budget = int(r["budget"])
            except (TypeError, ValueError):
                continue  # skip any stray malformed row rather than crash
            rows.append(r)
            done.add((r["strategy"], budget))
    return rows, done


def append_rows(csv_path: Path, rows: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = _csv_header(csv_path)
    if header is not None and header != CSV_FIELDS:
        # Don't append current-schema rows into a file with an incompatible
        # header — that is exactly what produces mixed/misaligned CSVs. Back the
        # old file up and start clean.
        backup = csv_path.with_suffix(csv_path.suffix + ".bak")
        print(f"WARNING: {csv_path} has an incompatible header; backing it up to "
              f"{backup} and starting a fresh CSV.")
        csv_path.replace(backup)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# COMPARISON PLOTS — eviction strategy on the x-axis (fixed LoRA model)
# =============================================================================

def write_plots(rows: list, plot_dir: Path):
    """Per (metric, budget), two PNGs — all on the fixed LoRA model:
      (a) bar of the metric per eviction strategy (baselines + oracle +
          learned), colored so 'baseline vs learned' reads at a glance;
      (b) learned's % improvement over each baseline strategy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping plot generation "
              "(pip install matplotlib)", file=sys.stderr)
        return

    # pivot[metric][budget][strategy] = value
    pivot = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        metric = r["metric"]
        if metric in META_KEYS:
            continue
        pivot[metric][int(r["budget"])][r["strategy"]] = float(r["value"])

    def color_for(s):
        if s == "learned":
            return "#1f77b4"   # highlight — the trained policy
        if s == "oracle":
            return "#7f7f7f"   # grey — upper bound
        return "#ff7f0e"       # baselines

    plot_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    for metric, by_budget in sorted(pivot.items()):
        metric_dir = plot_dir / _safe_filename(metric)
        metric_dir.mkdir(parents=True, exist_ok=True)

        for budget, by_strategy in sorted(by_budget.items()):
            strategies = [s for s in STRATEGIES_TO_COMPARE if s in by_strategy]
            if not strategies:
                continue
            values = [by_strategy[s] for s in strategies]
            colors = [color_for(s) for s in strategies]

            # (a) absolute metric value per strategy
            fig, ax = plt.subplots(figsize=(max(5, len(strategies) * 1.3), 4))
            ax.bar(strategies, values, color=colors)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} by eviction strategy (LoRA model, budget={budget})")
            ax.grid(True, axis="y", alpha=0.3)
            fig.savefig(metric_dir / f"budget{budget}_bar.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            n_saved += 1

            # (b) learned's improvement over each baseline strategy
            if "learned" in by_strategy:
                learned = by_strategy["learned"]
                base_strats = [s for s in strategies if s != "learned"]
                improvements = [
                    ((learned - by_strategy[s]) / abs(by_strategy[s]) * 100)
                    if by_strategy[s] != 0 else float("nan")
                    for s in base_strats
                ]
                fig, ax = plt.subplots(figsize=(max(5, len(base_strats) * 1.3), 4))
                bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in improvements]
                ax.bar(base_strats, improvements, color=bar_colors)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_ylabel(f"{metric}: learned improvement (%)")
                ax.set_title(f"{metric} — learned vs each baseline (budget={budget})")
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
        description="ATLAS extrinsic (Component 1) evaluation on a FIXED LoRA model: "
                     "compares baseline eviction strategies vs the trained learned policy"
    )
    parser.add_argument("--model-path", default=DEFAULT_C1_ADAPTER,
                         help="LoRA adapter dir/hub-id or full model dir for the fixed Agenda LLM "
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
    parser.add_argument("--skip-bertscore", action="store_true",
                         help="skip all deberta-xlarge bert-score metrics — both summary "
                              "bertscore AND agenda_completeness (which is bert-score based)")

    parser.add_argument("--csv", default=str(DEFAULT_CSV),
                         help="cache/result CSV (strategy, budget, metric, value) — "
                              "shared with eval_policy_baseline.py")
    parser.add_argument("--plot-dir", default=str(DEFAULT_PLOT_DIR))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--force-rerun", action="store_true",
                         help="ignore the cache and recompute every (strategy, budget) combo")
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    csv_path = Path(args.csv)
    cached_rows, done = ([], set()) if args.force_rerun else load_cache(csv_path)

    convs = load_conversations(Path(args.data_dir), limit=args.limit)
    if not convs:
        print(f"No conversation JSON files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(convs)} conversations from {args.data_dir}")

    # Agenda LLM is fixed to the LoRA model, built lazily (and once) the first
    # time an uncached strategy actually needs it — so a fully-cached rerun that
    # only needs to plot never pays the model-load cost.
    llm = None
    # Shared across every strategy/budget so identical trajectories score once.
    report_cache = {}
    new_rows = []
    for strategy_name in STRATEGIES_TO_COMPARE:
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

        eviction_strategy = build_eviction_strategy(
            strategy_name, args.window_size, args.policy_model, not args.policy_no_scispacy
        )

        print(f"\n=== strategy={strategy_name} budgets={missing_budgets} ===")
        if strategy_name == "oracle":
            # Oracle keeps everything regardless of budget -> identical across
            # budgets. Run once, replicate to each budget so plots still have an
            # oracle reference per budget.
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

    if not args.no_plots:
        write_plots(cached_rows + new_rows, Path(args.plot_dir))


if __name__ == "__main__":
    main()
