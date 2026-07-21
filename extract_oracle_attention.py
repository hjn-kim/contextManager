"""
Oracle-buffer attention extraction (summary -> summary, over time)
=================================================================

What this measures
------------------
At step t the Agenda LLM (C1) reads the *oracle* context buffer -- every
summary produced so far, nothing evicted (harness.py's oracle = budget
999999, an unlimited-buffer upper bound, not an answer-peeking policy) --
and emits its own relevance/summary JSON for the current utterance.

While it emits those output tokens we record how much attention each output
token paid to each *earlier summary* sitting in the buffer. The current
step's summary is appended to the buffer only after its output is finished,
so a summary never attends to itself and never to the future.

If A, B, C are summaries produced at successive steps, this yields rows

    source=A target=B      (attention while B was being produced)
    source=A target=C
    source=B target=C
    source=A target=<last> (attention while the final utterance was processed)

and the output is sorted so all of source=1 comes first, then source=2, ...
i.e. 1->2, 1->3, ..., 1->last, then 2->3, 2->4, ..., which is exactly the
"attention decay of one summary over time" curve, one block per summary.

This is the real-attention extraction that src/baselines.py's "attention"
strategy needs; it is computed offline here rather than inside the eviction
loop so the same numbers can be reused across budgets and analyses.

Data
----
Reads the annotated conversation JSONs in data/train/ (D2N001.json ... 67 files,
3743 utterances), each holding `utterances` (id, speaker, text) and
`annotations` (utterance_id, relevant, detail_list). Every emitted row therefore
carries real ids -- conversation_id "D2N001", source/target utterance_id
"line_0004" -- so a suspicious score can be traced straight back to the
transcript, and the rows join to other artifacts keyed the same way (e.g.
context_manager_labels.py's hindsight labels).

The oracle buffer is rebuilt here from the annotations rather than read off any
stored context string. Note that data/train.jsonl -- the flattened training file
built from these same conversations -- stores only the last 5 summaries in its
[Context] field (the old BufferConfig.max_context_items window; verified for
3743/3743 rows). So the C1 checkpoint was fine-tuned on <=5-summary contexts,
which makes a long oracle buffer out of distribution for it: the attention
numbers are real, but they are measured off-policy.

Model
-----
Defaults to the private LoRA adapter tria-hongik/atlas-c1-v2-llama-3.2-3b on
top of its recorded base model (same resolution logic as
eval_policy_extrinsic.py). Pass --hf-token or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN.

Two independent axes
--------------------
  --buffer-source   whose summaries sit in the buffer being attended TO
  --output-source   whose tokens the attention is measured FROM

The default (buffer=gold, output=generate) is the informative combination:
the context is fixed to the annotation -- so summary ids match the dataset and
a single bad model summary cannot poison every later step -- while the output
is what C1 genuinely produced. The scores then answer exactly "when really
writing this turn's output, how much did the model use each gold summary?"

  buffer  output    what it measures
  ------  --------  ------------------------------------------------------
  gold    generate  (default) real output usage of a fixed gold context
  gold    gold      fully teacher-forced; ~40x faster, good for smoke tests
  generate generate full C1 rollout; buffer drifts with the model's own errors

Usage (RunPod)
--------------
    export HF_TOKEN=hf_...
    python extract_oracle_attention.py \
        --data data/train \
        --out output/oracle_attention.jsonl

Memory note
-----------
Full attention matrices are never materialised for all layers at once: a
forward hook reduces each layer's weights to the per-summary statistics and
drops the tensor. Peak extra memory is one layer, ~heads x seq^2.
"""

import argparse
import gc
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from prompts import REALTIME_SYSTEM, REALTIME_USER  # noqa: E402

# NOTE: prompts.REALTIME_SYSTEM contains a literal "{TYPES}" (escaped as
# "{{TYPES}}" inside the f-string), so the 7 type definitions are not actually
# inlined. That is the prompt eval_policy_extrinsic.py already feeds this
# checkpoint, so it is reproduced verbatim here rather than "fixed" -- the
# attention numbers must come from the same prompt the rest of the pipeline uses.

DEFAULT_MODEL = "tria-hongik/atlas-c1-v2-llama-3.2-3b"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_DATA = PROJECT_ROOT / "data" / "train"
DEFAULT_OUT = PROJECT_ROOT / "output" / "oracle_attention.jsonl"
DEFAULT_PAIRS_OUT = PROJECT_ROOT / "output" / "oracle_attention_pairs.jsonl"

CONTEXT_EMPTY = "Start of visit"
JOINER = " | "


# =============================================================================
# DATA: annotated conversation JSON -> per-utterance stream
# =============================================================================

def detail_list_of(annotation: Optional[Dict]) -> List[Dict]:
    """The annotated (type, summary) pairs ([] when the turn is irrelevant)."""
    if not annotation or not annotation.get("relevant"):
        return []
    return [
        {"type": d["type"], "summary": d["summary"]}
        for d in (annotation.get("detail_list") or [])
        if isinstance(d, dict) and d.get("type") and d.get("summary")
    ]


def load_conversation(path: Path) -> Dict:
    """
    Read one annotated conversation JSON (data/train/D2N001.json and friends).

    Returns {"id": "D2N001", "utterances": [{utterance_id, speaker, text,
    gold_details}, ...]} -- one entry per utterance, in transcript order, with
    the annotation for that utterance already attached. Utterances with no
    annotation simply carry an empty gold_details.
    """
    with path.open(encoding="utf-8") as f:
        conv = json.load(f)

    by_uid = {a["utterance_id"]: a for a in conv.get("annotations", [])
              if a.get("utterance_id")}

    utterances = []
    for utt in conv.get("utterances", []):
        uid = utt.get("id")
        utterances.append({
            "utterance_id": uid,
            "speaker": (utt.get("speaker") or "").lower(),
            "text": utt.get("text") or "",
            "gold_details": detail_list_of(by_uid.get(uid)),
        })

    return {"id": conv.get("id") or path.stem, "utterances": utterances}


def load_conversations(path: Path, limit: Optional[int] = None) -> List[Dict]:
    """
    Load every conversation JSON under `path` (or a single file), sorted by
    filename so conversation order is reproducible across runs.
    """
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No conversation JSON files found under {path}")

    conversations = [load_conversation(f) for f in files]
    if limit:
        conversations = conversations[:limit]
    return conversations


# =============================================================================
# MODEL
# =============================================================================

def resolve_model_spec(spec: str, base_model_id: str, hf_token: Optional[str]):
    """(base_to_load, adapter_or_None, tokenizer_source) -- mirrors eval_policy_extrinsic."""
    local_path = Path(spec)
    adapter_config_path = None

    if local_path.is_dir() and (local_path / "adapter_config.json").exists():
        adapter_config_path = local_path / "adapter_config.json"
    elif not local_path.exists():
        try:
            from huggingface_hub import hf_hub_download
            adapter_config_path = Path(
                hf_hub_download(repo_id=spec, filename="adapter_config.json", token=hf_token)
            )
        except Exception:
            adapter_config_path = None

    if adapter_config_path is not None:
        with open(adapter_config_path, encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        return adapter_cfg.get("base_model_name_or_path") or base_model_id, spec, spec

    return spec, None, spec


def build_model(spec: str, base_model_id: str, dtype: torch.dtype,
                load_in_4bit: bool, hf_token: Optional[str]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_for_load, adapter, tok_source = resolve_model_spec(spec, base_model_id, hf_token)

    print(f"  tokenizer  <- {tok_source}")
    tokenizer = AutoTokenizer.from_pretrained(tok_source, token=hf_token)
    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required (offset mapping is used to locate each "
            "summary's token span inside the prompt)."
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4",
        )

    print(f"  base model <- {base_for_load}")
    # eager attention is mandatory: SDPA/flash kernels never materialise the
    # attention weights this whole script is about.
    kwargs = dict(device_map="auto", quantization_config=quantization_config,
                  token=hf_token, attn_implementation="eager")
    try:
        model = AutoModelForCausalLM.from_pretrained(base_for_load, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base_for_load, torch_dtype=dtype, **kwargs)

    if adapter:
        from peft import PeftModel
        print(f"  LoRA       <- {adapter}")
        model = PeftModel.from_pretrained(model, adapter, token=hf_token)

    model.eval()
    return tokenizer, model


def attention_modules(model) -> List:
    """Every self-attention submodule, in layer order."""
    for node in (model, getattr(model, "base_model", None)):
        if node is None:
            continue
        layers = getattr(getattr(node, "model", None), "layers", None)
        if layers is None:
            layers = getattr(node, "layers", None)
        if layers is None:
            inner = getattr(getattr(node, "model", None), "model", None)
            layers = getattr(inner, "layers", None)
        if layers is not None:
            return [layer.self_attn for layer in layers]
    raise RuntimeError("Could not locate decoder layers on this model.")


# =============================================================================
# PROMPT + SPAN BOOKKEEPING
# =============================================================================

def build_context(summaries: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """Join summaries with ' | ' and return each one's (start, end) char span."""
    if not summaries:
        return CONTEXT_EMPTY, []
    spans, cursor = [], 0
    for i, s in enumerate(summaries):
        if i:
            cursor += len(JOINER)
        spans.append((cursor, cursor + len(s)))
        cursor += len(s)
    return JOINER.join(summaries), spans


def build_prompt(tokenizer, context: str, speaker: str, text: str) -> str:
    messages = [
        {"role": "system", "content": REALTIME_SYSTEM},
        {"role": "user", "content": REALTIME_USER.format(
            context=context, speaker=speaker, text=text)},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def token_spans(offsets: List[Tuple[int, int]], char_spans: List[Tuple[int, int]],
                base_offset: int) -> List[List[int]]:
    """Map each summary's char span to the prompt token indices overlapping it."""
    out = []
    for start, end in char_spans:
        start += base_offset
        end += base_offset
        out.append([
            i for i, (a, b) in enumerate(offsets)
            if b > start and a < end and b > a
        ])
    return out


# =============================================================================
# ATTENTION CAPTURE
# =============================================================================

class AttentionReducer:
    """
    Forward hooks on every self-attention module that reduce the layer's
    attention weights to per-summary statistics in place, so the full
    [heads, seq, seq] tensor is freed immediately instead of being kept for
    all layers (which is what output_attentions=True would do).

    Per summary j it accumulates, averaged over heads and the selected output
    rows: the total attention mass on j's tokens, and the mean per token.
    finalize() then averages over layers.
    """

    def __init__(self, modules: List):
        self.modules = modules
        self.handles = []
        self.reset([], [])

    def reset(self, row_ids: List[int], summary_token_ids: List[List[int]]):
        self.row_ids = row_ids
        self.summary_token_ids = summary_token_ids
        self.mass_sum = torch.zeros(len(summary_token_ids), dtype=torch.float64)
        self.mass_mean = torch.zeros(len(summary_token_ids), dtype=torch.float64)
        self.layers_seen = 0
        self.captured = False

    def _hook(self, module, args, output):
        weights = None
        if isinstance(output, (tuple, list)):
            for item in output[1:]:
                if torch.is_tensor(item) and item.dim() == 4:
                    weights = item
                    break
        if weights is None or not self.row_ids or not self.summary_token_ids:
            return

        # [batch, heads, seq, seq] -> mean over heads and the selected rows
        rows = torch.as_tensor(self.row_ids, device=weights.device)
        attn = weights[0].index_select(1, rows).float().mean(dim=(0, 1)).double().cpu()

        for j, tok_ids in enumerate(self.summary_token_ids):
            if not tok_ids:
                continue
            mass = attn.index_select(0, torch.as_tensor(tok_ids))
            self.mass_sum[j] += float(mass.sum())
            self.mass_mean[j] += float(mass.mean())

        self.layers_seen += 1
        self.captured = True

    def __enter__(self):
        self.handles = [m.register_forward_hook(self._hook) for m in self.modules]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def finalize(self) -> Optional[Tuple[List[float], List[float]]]:
        """(per-summary mean attention, per-summary total mass), layer-averaged."""
        if not self.captured or self.layers_seen == 0:
            return None
        n = float(self.layers_seen)
        return (self.mass_mean / n).tolist(), (self.mass_sum / n).tolist()


# =============================================================================
# CORE STEP
# =============================================================================

def parse_generated(raw: str) -> List[Dict]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not obj.get("relevant"):
        return []
    return [
        {"type": d["type"], "summary": d["summary"]}
        for d in (obj.get("detail_list") or [])
        if isinstance(d, dict) and d.get("type") and d.get("summary")
    ]


@torch.no_grad()
def run_step(tokenizer, model, reducer: AttentionReducer, buffer: List[str],
             speaker: str, text: str, output_source: str, gold_details: List[Dict],
             max_new_tokens: int):
    """
    One oracle step: build the prompt over the full buffer, produce the output
    tokens, and measure their attention back onto each buffer summary.

    `output_source` controls only which tokens the attention is measured FROM
    (the model's own generation vs. the annotated summary). What gets appended
    to the buffer is a separate decision, made by the caller -- see
    process_conversation's `buffer_source`.

    Returns (mean_attn_per_summary, mass_per_summary, generated_details,
             n_output_tokens).
    """
    context, char_spans = build_context(buffer)
    prompt = build_prompt(tokenizer, context, speaker, text)

    context_start = prompt.rfind(context)
    if context_start < 0:
        raise RuntimeError("Context string not found verbatim in the rendered prompt.")

    enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True,
                    add_special_tokens=False)
    offsets = enc.pop("offset_mapping")[0].tolist()
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    # ---- output tokens: the model's own JSON, or the annotated one ----------
    # gen_details is None when nothing was generated (output_source == "gold").
    if output_source == "generate":
        gen = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        out_ids = gen[0, prompt_len:]
        gen_details = parse_generated(
            tokenizer.decode(out_ids, skip_special_tokens=True))
    else:
        payload = ({"relevant": True, "detail_list": gold_details}
                   if gold_details else {"relevant": False})
        out_ids = tokenizer(json.dumps(payload, ensure_ascii=False),
                            return_tensors="pt", add_special_tokens=False
                            )["input_ids"][0].to(device)
        gen_details = None

    if out_ids.numel() == 0:
        return None, None, gen_details, 0

    full_ids = torch.cat([input_ids[0], out_ids]).unsqueeze(0)
    total_len = full_ids.shape[1]

    # Rows that *produce* the output tokens: the last prompt position emits the
    # first output token, so rows run [prompt_len-1, total_len-2].
    row_ids = list(range(prompt_len - 1, total_len - 1))
    sum_tok_ids = token_spans(offsets, char_spans, context_start) if buffer else []

    if not sum_tok_ids:
        return None, None, gen_details, len(row_ids)

    reducer.reset(row_ids, sum_tok_ids)
    with reducer:
        model(input_ids=full_ids,
              attention_mask=torch.ones_like(full_ids),
              use_cache=False)

    stats = reducer.finalize()
    if stats is None:
        raise RuntimeError(
            "No attention weights captured. The model must be loaded with "
            "attn_implementation='eager' and expose per-layer self_attn modules."
        )
    mean_attn, mass = stats
    return mean_attn, mass, gen_details, len(row_ids)


# =============================================================================
# CONVERSATION LOOP
# =============================================================================

def process_conversation(conv_id: str, utterances: List[Dict], tokenizer, model,
                         reducer: AttentionReducer, output_source: str,
                         buffer_source: str, max_new_tokens: int) -> List[Dict]:
    """
    Oracle Context Manager: nothing is ever evicted, and the summaries produced
    at step t enter the buffer only after step t's output is finished.

    Two independent axes:
      buffer_source -- whose summaries sit in the buffer being attended TO.
        "gold" keeps the buffer identical to the annotation, so summary ids
        line up with the dataset and one bad model summary cannot poison every
        later step's context.
      output_source -- whose tokens the attention is measured FROM.
        "generate" means the model really wrote this turn's output, so the
        scores answer "how much did the real output actually use each summary".

    buffer=gold + output=generate is the useful hybrid: a fixed, trustworthy
    context measured against the model's genuine output behaviour.
    """
    buffer: List[str] = []          # summary texts, oracle order
    meta: List[Dict] = []           # per-summary provenance
    rows: List[Dict] = []
    n_utt = len(utterances)

    for t, utt in enumerate(utterances):
        is_last = (t == n_utt - 1)
        mean_attn, mass, gen_details, n_out = run_step(
            tokenizer, model, reducer, buffer,
            utt["speaker"], utt["text"], output_source,
            utt["gold_details"], max_new_tokens,
        )

        # What the buffer grows by, vs. what the model actually emitted. These
        # differ whenever buffer_source != output_source.
        gold_details = utt["gold_details"]
        new_details = gold_details if buffer_source == "gold" else (gen_details or [])

        # Summary ids created at this step (1-indexed, buffer order).
        target_ids = list(range(len(buffer) + 1, len(buffer) + 1 + len(new_details)))

        if mean_attn is not None:
            total_mass = sum(mass) or 1.0
            for j, m in enumerate(meta):
                rows.append({
                    "conversation_id": conv_id,
                    "source_summary_id": m["summary_id"],
                    "source_type": m["type"],
                    "source_summary": m["summary"],
                    "source_step": m["step"],
                    "source_utterance_id": m["utterance_id"],
                    "source_speaker": m["speaker"],
                    # target = what the model was producing while it attended back
                    "target_summary_ids": target_ids,
                    "target_step": t,
                    "target_utterance_id": utt["utterance_id"],
                    "target_summaries": [d["summary"] for d in new_details],
                    "target_label": ("last" if is_last
                                     else (target_ids[0] if target_ids else None)),
                    "is_last_utterance": is_last,
                    "target_relevant": bool(new_details),
                    # Did the model itself judge this turn relevant? Only
                    # meaningful when it actually generated (else None).
                    "model_relevant": (None if gen_details is None
                                       else bool(gen_details)),
                    "target_speaker": utt["speaker"],
                    "age_in_steps": t - m["step"],
                    "age_in_summaries": len(buffer) - j,
                    "attention_mean": mean_attn[j],   # mean over the summary's tokens
                    "attention_mass": mass[j],        # total mass on the summary
                    "attention_share": mass[j] / total_mass,  # normalised within this step
                    "buffer_size": len(buffer),
                    "n_output_tokens": n_out,
                })

        # Oracle append: only AFTER the current output is complete.
        for k, d in enumerate(new_details):
            buffer.append(d["summary"])
            meta.append({
                "summary_id": target_ids[k],
                "summary": d["summary"],
                "type": d["type"],
                "step": t,
                # Carried so the downstream feature builder can fill the
                # speaker / position / recency dims without re-reading the
                # source conversation.
                "utterance_id": utt["utterance_id"],
                "speaker": utt["speaker"],
            })

    return rows


# =============================================================================
# TRAINING-SET SHAPING
# =============================================================================

def add_within_step_labels(rows: List[Dict]) -> None:
    """
    Add step-normalised versions of the attention score, in place.

    Raw attention_share is not directly usable as a regression target: it sums
    to 1 across the buffer, so every value shrinks as the buffer grows. A score
    of 0.30 with 3 items and 0.03 with 30 items can mean the same thing, but
    buffer size is not one of the policy's input features -- the target would
    swing on information the model cannot see.

    What eviction actually needs is the ordering *within* one step ("of the
    summaries I am holding right now, which is least used?"), so the normalised
    columns below are the ones to train on:

      attention_rank_pct  percentile rank within the step, 0=lowest .. 1=highest.
                          Scale-free and buffer-size independent. Recommended.
      attention_z         z-score within the step. Keeps relative magnitude,
                          so it distinguishes "clearly ignored" from "just below
                          average" -- which the rank version flattens.
      attention_log_share log of the raw share; compresses the heavy tail if
                          you would rather regress on the raw quantity.

    Steps holding a single summary are degenerate (share is always 1.0 and
    there is nothing to rank); they get rank_pct=1.0, z=0.0 and are flagged
    with rankable=False so they can be dropped.
    """
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        groups[(row["conversation_id"], row["target_step"])].append(row)

    for group in groups.values():
        k = len(group)
        shares = [g["attention_share"] for g in group]

        mean = sum(shares) / k
        var = sum((s - mean) ** 2 for s in shares) / k
        std = var ** 0.5

        # Ascending rank; ties share the average rank so equal scores get equal
        # targets rather than an order-dependent split.
        order = sorted(range(k), key=lambda i: shares[i])
        ranks = [0.0] * k
        i = 0
        while i < k:
            j = i
            while j + 1 < k and shares[order[j + 1]] == shares[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0
            for pos in range(i, j + 1):
                ranks[order[pos]] = avg_rank
            i = j + 1

        for idx, row in enumerate(group):
            row["attention_rank_pct"] = (ranks[idx] / (k - 1)) if k > 1 else 1.0
            row["attention_z"] = ((shares[idx] - mean) / std) if std > 0 else 0.0
            row["attention_log_share"] = math.log(max(shares[idx], 1e-12))
            row["rankable"] = k > 1


def to_pair_rows(rows: List[Dict]) -> List[Dict]:
    """
    Project the full step log down to (source summary -> target summary) pairs,
    which is the shape the 780-d eviction dataset needs.

    Two things happen here:
      * Steps that produced no summary are dropped. 57% of rows sit on
        irrelevant turns, and they have no target summary to embed.
      * A step that produced several summaries stays ONE row, with the
        summaries joined. Attention was measured over that step's whole output,
        so there is no per-summary attribution to split it by; emitting one row
        each would duplicate a single label and double-weight the step.
    """
    pairs = []
    for row in rows:
        if not row["target_relevant"] or not row["target_summaries"]:
            continue
        pairs.append({
            "conversation_id": row["conversation_id"],
            # --- source: the buffered summary being scored -------------------
            "source_summary_id": row["source_summary_id"],
            "source_utterance_id": row["source_utterance_id"],
            "source_summary": row["source_summary"],
            "source_type": row["source_type"],
            "source_speaker": row["source_speaker"],
            "source_step": row["source_step"],
            # --- target: what the model was writing at this step -------------
            "target_summary_ids": row["target_summary_ids"],
            "target_utterance_id": row["target_utterance_id"],
            "target_summary": JOINER.join(row["target_summaries"]),
            # Kept separately because the joined string above is not itself an
            # annotated sentence: it will not be in a sentence-keyed embedding
            # dictionary. Downstream, embed these individually and average.
            "target_summaries": row["target_summaries"],
            "target_step": row["target_step"],
            # --- context needed to build position/recency --------------------
            "current_step": row["target_step"],
            "age_in_steps": row["age_in_steps"],
            "buffer_size": row["buffer_size"],
            "rankable": row["rankable"],
            # --- candidate labels --------------------------------------------
            "attention_share": row["attention_share"],
            "attention_rank_pct": row["attention_rank_pct"],
            "attention_z": row["attention_z"],
            "attention_log_share": row["attention_log_share"],
            "attention_mean": row["attention_mean"],
        })
    return pairs


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Extract summary->summary attention over time under an oracle "
                    "(never-evicting) Context Manager.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help="Directory of annotated conversation JSONs, or a single "
                        "one (default: %(default)s)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Full step log: every (buffered summary, step) row, "
                        "including steps that produced no summary. Keep this "
                        "for attention-decay analysis.")
    p.add_argument("--pairs-out", type=Path, default=DEFAULT_PAIRS_OUT,
                   help="Training-ready (source summary -> target summary) "
                        "pairs, written by default alongside --out.")
    p.add_argument("--no-pairs", action="store_true",
                   help="Skip the pair projection and write only --out.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="LoRA adapter repo/dir or full model (default: %(default)s)")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                   help="Fallback base model when the adapter config omits one.")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN")
                   or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                   help="Needed for the private repo. Defaults to $HF_TOKEN.")
    p.add_argument("--output-source", choices=["generate", "gold"], default="generate",
                   help="Whose tokens attention is measured FROM. 'generate': C1 "
                        "writes its own output (real behaviour, slow). 'gold': "
                        "teacher-force the annotated output (fast, deterministic).")
    p.add_argument("--buffer-source", choices=["gold", "generate"], default="gold",
                   help="Whose summaries sit in the buffer being attended TO. "
                        "'gold' (default) keeps the buffer identical to the "
                        "annotation so summary ids match the dataset and model "
                        "summary errors cannot compound. 'generate' requires "
                        "--output-source generate.")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="Only the first N conversations.")
    p.add_argument("--csv", type=Path, default=None, help="Also write a flat CSV here.")
    args = p.parse_args()

    if not args.data.exists():
        sys.exit(f"Data file not found: {args.data}")

    if args.buffer_source == "generate" and args.output_source != "generate":
        sys.exit("--buffer-source generate requires --output-source generate "
                 "(nothing is generated to put in the buffer otherwise).")
    print(f"buffer <- {args.buffer_source} summaries, "
          f"attention measured from {args.output_source} output tokens")

    print(f"Loading {args.data} ...")
    conversations = load_conversations(args.data, limit=args.limit)
    n_utt = sum(len(c["utterances"]) for c in conversations)
    print(f"  {len(conversations)} conversations, {n_utt} utterances")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print("Loading model ...")
    tokenizer, model = build_model(args.model, args.base_model, dtype,
                                   args.load_in_4bit, args.hf_token)
    reducer = AttentionReducer(attention_modules(model))
    print(f"  {len(reducer.modules)} attention layers hooked")

    all_rows: List[Dict] = []
    for ci, conv in enumerate(conversations):
        conv_id, utterances = conv["id"], conv["utterances"]
        print(f"[{ci + 1}/{len(conversations)}] {conv_id} "
              f"({len(utterances)} utterances) ...", flush=True)
        all_rows.extend(process_conversation(
            conv_id, utterances, tokenizer, model, reducer,
            args.output_source, args.buffer_source, args.max_new_tokens,
        ))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Step-normalised label columns must be computed over each step's whole
    # buffer, so this runs before any filtering or reordering.
    add_within_step_labels(all_rows)

    # Requested ordering: all rows for summary 1 (1->2, 1->3, ..., 1->last),
    # then summary 2 (2->3, 2->4, ...), and so on, per conversation.
    all_rows.sort(key=lambda r: (r["conversation_id"],
                                 r["source_summary_id"],
                                 r["target_step"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_rows)} rows (full step log) -> {args.out}")

    # Training-ready projection: summary -> summary pairs only.
    if not args.no_pairs:
        pairs = to_pair_rows(all_rows)
        args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
        with args.pairs_out.open("w", encoding="utf-8") as f:
            for row in pairs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_rankable = sum(1 for r in pairs if r["rankable"])
        print(f"Wrote {len(pairs)} pair rows ({n_rankable} rankable) "
              f"-> {args.pairs_out}")

    if args.csv:
        import csv
        fields = ["conversation_id", "source_summary_id", "source_utterance_id",
                  "target_step", "target_utterance_id", "target_label",
                  "is_last_utterance", "target_relevant", "model_relevant",
                  "age_in_steps", "age_in_summaries",
                  "attention_mean", "attention_mass", "attention_share",
                  "source_type", "buffer_size", "n_output_tokens"]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in all_rows:
                w.writerow({k: row[k] for k in fields})
        print(f"Wrote CSV -> {args.csv}")

    # Quick sanity read-out: mean attention share by age.
    by_age = defaultdict(list)
    for row in all_rows:
        by_age[row["age_in_steps"]].append(row["attention_share"])
    if by_age:
        print("\nmean attention share by age (steps):")
        for age in sorted(by_age)[:15]:
            vals = by_age[age]
            print(f"  age {age:>3}: {sum(vals) / len(vals):.5f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
