"""
ATLAS Real-Component Integration Harness
========================================
Wires the *real* three components together and evaluates them:

  utterance
    → Agenda LLM   : LoRA adapter (Llama-3.2-3B-Instruct + PEFT)  [Component 1]
    → SystemTracker: rule-based agenda tracker                    [Component 2]
    → ContextBuffer: eviction strategy (default: none / unlimited) [Component 3]
    → metrics.generate_report                                     [Evaluation]

Difference from harness.py:
  - harness.py uses MockLLM (replays gold annotations). This file drives the
    fine-tuned LoRA model for genuine predictions.
  - The eviction policy is SELECTABLE. Default is `none` (no eviction,
    unlimited budget). Plug in the MLP with `--eviction learned --mlp <path>`,
    or any baseline (fifo/sliding/random/oldest/attention).

Runtime deps (install where you actually run this, e.g. RunPod):
  pip install peft accelerate        # peft is REQUIRED to load the LoRA adapter
  # torch / transformers / sentence-transformers are already expected.

CPU works but is slow; a CUDA GPU is auto-detected and used when present.

Examples (from project root):
  # Real LLM + tracker, NO eviction (default), evaluate on the valid split
  python src/harness_integration.py --input data/valid_aci

  # Same, but plug in the MLP eviction policy at budget B=1024
  python src/harness_integration.py --input data/valid_aci \
      --eviction learned --mlp eviction_mlp/eviction_mlp_bertscore_0p75.pt \
      --budget 1024

  # A single conversation with a FIFO baseline
  python src/harness_integration.py --input data/valid_aci/D2N068.json \
      --eviction fifo --budget 512
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict

# Make `src/` importable whether run as a module or a script.
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import AtlasConfig, DEFAULT_CONFIG
# Reuse the shared streaming loop and data structures — do NOT re-implement them.
from harness import (
    StreamingHarness,
    MockLLM,
    AgendaLLM,
    LLMOutput,
    BufferItem,
    ContextBuffer,
)
from state_tracker import SystemTracker
from policy import LearnedEvictionStrategy
from baselines import get_strategy
from metrics import generate_report


# Default adapter directory (relative to project root).
_PROJECT_ROOT = Path(_SRC).parent
DEFAULT_ADAPTER = _PROJECT_ROOT / "AAA_atlas_c1_lora_adapter_checkpoint100"
DEFAULT_MLP = _PROJECT_ROOT / "eviction_mlp" / "eviction_mlp_bertscore_0p75.pt"


# =============================================================================
# AGENDA LLM — real LoRA backend (Component 1)
# =============================================================================

# The Agenda LLM is a text->JSON extractor. This system prompt describes the
# task. IT SHOULD MATCH WHATEVER PROMPT THE LoRA WAS FINE-TUNED WITH — if your
# SFT used a different system prompt, override it with --system-prompt-file.
DEFAULT_SYSTEM_PROMPT = (
    "You are a clinical scribe assistant. For each new utterance in a "
    "doctor-patient visit, decide whether it contains clinically relevant "
    "information worth remembering. If it does, extract one or more concise "
    "summaries, each with a type from: agenda_item, detail, medication, "
    "social_history, follow_up, question, question_unanswered.\n"
    "Respond ONLY with a JSON object of the form:\n"
    '{"relevant": true, "detail_list": [{"type": "<type>", '
    '"summary": "<concise summary>"}]}\n'
    'If the utterance is not clinically relevant, respond exactly with '
    '{"relevant": false}.'
)

# The user turn format mirrors the convention used elsewhere in the repo
# (baselines._build_attention_prompt): "[Context]: ...\n[Utterance]: [Speaker] ..."
_VALID_TYPES = {
    "agenda_item", "detail", "medication", "social_history",
    "follow_up", "question", "question_unanswered", "plan",
}


class LoraAgendaLLM(AgendaLLM):
    """
    Agenda LLM backed by the fine-tuned LoRA adapter.

    Loads base `meta-llama/Llama-3.2-3B-Instruct` and applies the PEFT adapter.
    Produces LLMOutput(relevant, type, summary, detail_list) by generating JSON
    and parsing it. Exposes `.model`, `.tok`, and `.system_prompt` so the
    attention eviction baseline can reuse the same backend.
    """

    def __init__(
        self,
        config: AtlasConfig,
        adapter_dir: str,
        base_model: Optional[str] = None,
        device: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__(config)
        self.adapter_dir = str(adapter_dir)
        self.base_model = base_model
        self.device = device
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self.model = None
        self.tok = None
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers/torch are required for the LoRA Agenda LLM."
            ) from e
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "peft is required to load the LoRA adapter. "
                "Install it in your run environment: pip install peft accelerate"
            ) from e

        # Resolve base model name from the adapter config if not given.
        if self.base_model is None:
            cfg_path = Path(self.adapter_dir) / "adapter_config.json"
            with open(cfg_path, encoding="utf-8") as f:
                self.base_model = json.load(f)["base_model_name_or_path"]

        # Device / dtype.
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        # Tokenizer: prefer the one bundled with the adapter (has chat template).
        self.tok = AutoTokenizer.from_pretrained(self.adapter_dir)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        model_kwargs = {"torch_dtype": dtype}
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        print(f"[LoraAgendaLLM] loading base '{self.base_model}' "
              f"on {self.device} ({dtype})...", flush=True)
        base = AutoModelForCausalLM.from_pretrained(self.base_model, **model_kwargs)
        print(f"[LoraAgendaLLM] applying adapter '{self.adapter_dir}'...", flush=True)
        self.model = PeftModel.from_pretrained(base, self.adapter_dir)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _build_prompt(self, context: str, speaker: str, text: str) -> str:
        speaker_label = speaker.capitalize() if speaker else "Speaker"
        user_text = (
            f"[Context]: {context or 'Start of visit'}\n"
            f"[Utterance]: [{speaker_label}] {text}"
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def predict(self, context: str, speaker: str, text: str) -> LLMOutput:
        import torch

        prompt = self._build_prompt(context, speaker, text)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tok.pad_token_id,
            )
        gen = out[0, enc["input_ids"].shape[1]:]
        completion = self.tok.decode(gen, skip_special_tokens=True)
        return self._parse(completion)

    @staticmethod
    def _parse(completion: str) -> LLMOutput:
        """Extract the first JSON object from the completion and normalize it."""
        obj = _extract_json(completion)
        if not obj or not obj.get("relevant"):
            return LLMOutput(relevant=False)

        detail_list = obj.get("detail_list")
        if isinstance(detail_list, list):
            cleaned = [
                {"type": d.get("type"), "summary": d.get("summary")}
                for d in detail_list
                if isinstance(d, dict) and d.get("type") and d.get("summary")
            ]
            if not cleaned:
                return LLMOutput(relevant=False)
            return LLMOutput(
                relevant=True,
                detail_list=cleaned,
                type=cleaned[0]["type"],
                summary=cleaned[0]["summary"],
            )

        # Legacy single-summary format.
        if obj.get("summary"):
            return LLMOutput(
                relevant=True,
                type=obj.get("type") or "detail",
                summary=obj.get("summary"),
            )
        return LLMOutput(relevant=False)

    # ------------------------------------------------------------------
    # Token counting — real tokenizer (replaces the char-based estimate)
    # ------------------------------------------------------------------
    def get_token_count(self, text: str) -> int:
        if not text:
            return 1
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return max(len(ids), 1)


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of the first JSON object from model output."""
    if not text:
        return None
    s = text.strip()
    # Strip common code fences.
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    # Find the first balanced {...} span.
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


# =============================================================================
# EVICTION STRATEGY — selectable (Component 3)
# =============================================================================

# All choices. "none" = no eviction (unlimited budget), the DEFAULT.
EVICTION_CHOICES = [
    "none", "learned", "fifo", "sliding", "random", "oldest", "attention",
]


def build_eviction(name: str, mlp_path: Optional[str], window: int):
    """
    Return an eviction strategy callable (or None for 'none').

    The strategy is called by the harness as
        strategy(items, llm=..., current_utterance=..., context=...)
    so learned/MLP is wrapped to absorb those keyword args.
    """
    if name == "none":
        return None

    if name == "learned":
        if not mlp_path:
            raise ValueError("--mlp <path> is required when --eviction learned")
        if not Path(mlp_path).exists():
            raise FileNotFoundError(f"MLP checkpoint not found: {mlp_path}")
        learned = LearnedEvictionStrategy(model_path=str(mlp_path))

        def _learned(items, **kwargs):  # absorb harness kwargs
            return learned(items)
        return _learned

    if name == "sliding":
        return get_strategy("sliding", window_size=window)

    # fifo / random / oldest / attention already accept **kwargs.
    return get_strategy(name)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_conversation(
    conv: dict,
    llm: AgendaLLM,
    config: AtlasConfig,
    eviction_strategy,
) -> Dict:
    """Run one conversation through the full pipeline and score it."""
    tracker = SystemTracker(config.tracker)
    harness = StreamingHarness(config)
    harness.load_conversation_dict(conv)
    result = harness.run(llm, tracker=tracker, eviction_strategy=eviction_strategy)

    try:
        report = generate_report(
            result,
            gold_annotations=conv.get("annotations"),
            utterances=conv.get("utterances"),
            eval_annotations=conv.get("annotations"),
        )
    except Exception as e:
        # Never let one conversation's metric backend abort a whole batch run.
        print(f"  [warn] metrics failed for {conv.get('id')}: {e}")
        report = {
            "conversation_id": result.conversation_id,
            "metadata": result.metadata,
            "latency": {
                "avg_ms": result.metadata.get("avg_latency_ms", 0),
                "max_ms": max(result.latencies) if result.latencies else 0,
                "min_ms": min(result.latencies) if result.latencies else 0,
            },
        }
    return report


def _iter_conversations(input_path: str) -> List[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    return sorted(p.rglob("*.json"))


def _aggregate(reports: List[Dict]) -> Dict:
    """Average the headline metrics across conversations."""
    def _avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    agg = {
        "conversations": len(reports),
        "avg_latency_ms": _avg([r["latency"]["avg_ms"] for r in reports]),
        "detection_macro_f1": _avg([
            r.get("detection", {}).get("macro_avg", {}).get("f1")
            for r in reports
        ]),
        "rougeL": _avg([r.get("rouge", {}).get("rougeL") for r in reports]),
        "agenda_completeness": _avg([r.get("agenda_completeness") for r in reports]),
        "linking_accuracy": _avg([
            r.get("system_level", {}).get("linking_accuracy") for r in reports
        ]),
        "first_mention_accuracy": _avg([
            r.get("system_level", {}).get("first_mention_accuracy") for r in reports
        ]),
        "resolution_accuracy": _avg([
            r.get("system_level", {}).get("resolution_accuracy") for r in reports
        ]),
    }
    return agg


def _fmt(v, pct=False):
    if v is None:
        return "  n/a"
    return f"{v*100:5.1f}%" if pct else f"{v:6.3f}"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ATLAS real-component integration harness (LoRA LLM + "
                    "SystemTracker + selectable eviction) with evaluation.")
    parser.add_argument("--input", required=True,
                        help="Conversation JSON file or folder (e.g. data/valid_aci)")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER),
                        help="LoRA adapter directory")
    parser.add_argument("--base-model", default=None,
                        help="Override base model (default: read from adapter_config.json)")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: auto-detect)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--system-prompt-file", default=None,
                        help="Text file with the system prompt (must match SFT training)")

    parser.add_argument("--eviction", default="none", choices=EVICTION_CHOICES,
                        help="Eviction policy (default: none = no eviction). "
                             "Use 'learned' with --mlp to plug in the MLP.")
    parser.add_argument("--mlp", default=None,
                        help=f"MLP .pt checkpoint (for --eviction learned). "
                             f"e.g. {DEFAULT_MLP}")
    parser.add_argument("--budget", type=int, default=1024,
                        help="Token budget B (used by all eviction policies except 'none')")
    parser.add_argument("--window", type=int, default=10,
                        help="Sliding window size K (for --eviction sliding)")

    parser.add_argument("--mock", action="store_true",
                        help="Use MockLLM (replay gold annotations) instead of the LoRA — "
                             "handy for sanity-checking the tracker/eviction wiring.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N conversations")
    parser.add_argument("--output", default=None, help="Save full per-conversation reports as JSON")
    args = parser.parse_args()

    # Config + budget.
    config = AtlasConfig()
    if args.eviction == "none":
        config.buffer.budget = 999999
    else:
        config.buffer.budget = args.budget

    # Eviction strategy.
    strategy = build_eviction(args.eviction, args.mlp, args.window)

    # System prompt override.
    system_prompt = None
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")

    # Conversations.
    conv_paths = _iter_conversations(args.input)
    if args.limit:
        conv_paths = conv_paths[:args.limit]
    if not conv_paths:
        raise SystemExit(f"No conversation JSON found under {args.input}")

    # Build the Agenda LLM once and reuse it across conversations.
    if args.mock:
        print("[integration] MockLLM mode (no LoRA loaded).")
        llm = None  # built per-conversation below (needs that conv's annotations)
    else:
        attn_impl = "eager" if args.eviction == "attention" else None
        llm = LoraAgendaLLM(
            config,
            adapter_dir=args.adapter,
            base_model=args.base_model,
            device=args.device,
            system_prompt=system_prompt,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=attn_impl,
        )

    print(f"\n{'='*72}")
    print(f"  Agenda LLM : {'MockLLM' if args.mock else 'LoRA ' + args.adapter}")
    print(f"  Tracker    : SystemTracker (rule-based)")
    print(f"  Eviction   : {args.eviction}"
          + (f"  (mlp={args.mlp})" if args.eviction == 'learned' else "")
          + (f"  B={args.budget}" if args.eviction != 'none' else "  (unlimited)"))
    print(f"  Convos     : {len(conv_paths)}")
    print(f"{'='*72}\n")

    reports = []
    for path in conv_paths:
        with open(path, encoding="utf-8") as f:
            conv = json.load(f)
        if "utterances" not in conv or "annotations" not in conv:
            print(f"  [skip] {path.name}: missing utterances/annotations")
            continue

        if args.mock:
            annotations = {a["utterance_id"]: a for a in conv.get("annotations", [])}
            conv_llm = MockLLM(config, annotations)
        else:
            conv_llm = llm

        report = evaluate_conversation(conv, conv_llm, config, strategy)
        reports.append(report)
        det = report.get("detection", {}).get("macro_avg", {}).get("f1")
        print(f"  {conv['id']:<10}  det_f1={_fmt(det)}  "
              f"buffer={report['metadata']['final_buffer_size']:>3}  "
              f"lat={report['latency']['avg_ms']:.0f}ms")

    # Aggregate.
    agg = _aggregate(reports)
    print(f"\n{'='*72}")
    print(f"  AGGREGATE  ({agg['conversations']} conversations)")
    print(f"{'='*72}")
    print(f"  detection macro-F1     : {_fmt(agg['detection_macro_f1'])}")
    print(f"  rougeL                 : {_fmt(agg['rougeL'])}")
    print(f"  agenda completeness    : {_fmt(agg['agenda_completeness'], pct=True)}")
    print(f"  linking accuracy       : {_fmt(agg['linking_accuracy'], pct=True)}")
    print(f"  first-mention accuracy : {_fmt(agg['first_mention_accuracy'], pct=True)}")
    print(f"  resolution accuracy    : {_fmt(agg['resolution_accuracy'], pct=True)}")
    print(f"  avg latency            : {agg['avg_latency_ms']:.0f} ms" if agg['avg_latency_ms'] else "")
    print(f"{'='*72}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"aggregate": agg, "reports": reports}, f,
                      indent=2, ensure_ascii=False)
        print(f"\nSaved reports to {args.output}")


if __name__ == "__main__":
    main()
