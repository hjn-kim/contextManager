"""
ATLAS Baseline Eviction Strategies
====================================

Spec baseline 5:
  1. Growing window (FIFO)  — token-budget driven, evict oldest when over B
  2. Sliding window          — count-based, always keep last K items only
  3. Random eviction         — random selection
  4. Oldest-first            — by original timestamp (differs from FIFO after reordering)
  5. Lowest-attention        — by real LLM attention weights

KEY DESIGN DECISION — why FIFO and sliding are different:

  FIFO is TOKEN-BUDGET driven:
    Buffer grows freely. When total tokens > B, remove oldest items
    one by one until budget met. If 12 items fit in B tokens, keep all 12.

  Sliding is COUNT-BASED:
    Always keep only the most recent K items. Even if budget allows more,
    old items are dropped. This matches the prior paper (Jang et al. 2025)
    where context_size = K summaries.

  Example with 12 items, budget=1024, total=200 tokens, K=5:
    FIFO:    keeps all 12 (200 < 1024, no eviction)
    Sliding: keeps only last 5 (always enforces window regardless of budget)

Oracle (unlimited budget) is NOT a baseline — handled at harness level
by setting budget=999999.

Strategy interface:
  strategy(items: List[BufferItem], **kwargs) -> List[BufferItem]
  Returns the items to KEEP (may be shorter than input).
  Remaining items must have .score set for tiebreaking if further
  eviction is needed.
"""

import math
import random
from collections.abc import Sequence
from typing import Callable, List, Optional


# =============================================================================
# STRATEGY 1: FIFO / Growing Window
# =============================================================================

def fifo_strategy(items: List, **kwargs) -> List:
    """
    Growing window / FIFO. Token-budget driven.

    Returns all items with scores assigned. Does NOT remove items itself —
    the eviction loop in ContextBuffer.evict() handles removal based on
    budget. Score = buffer position (oldest=0, newest=1).

    Behavior: buffer grows until overflow, then oldest items are removed
    one by one until budget is satisfied. Items from early in the
    conversation can survive if budget allows.
    """
    n = len(items)
    for i, item in enumerate(items):
        item.score = i / max(n - 1, 1)  # 0.0 (oldest) → 1.0 (newest)
    return items


# =============================================================================
# STRATEGY 2: Sliding Window (COUNT-BASED — different from FIFO)
# =============================================================================

class SlidingWindowStrategy:
    """
    Sliding window. Count-based — always keeps only the last K items.

    This is fundamentally different from FIFO:
    - FIFO keeps everything that fits in the token budget
    - Sliding keeps only the last K items regardless of budget

    After the window cut, if remaining items still exceed token budget,
    scores are assigned for further eviction (newest = highest score).

    The window size K defaults to 10 and can be configured.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    def __call__(self, items: List, **kwargs) -> List:
        if not items:
            return items

        # Sort by timestamp to identify most recent
        items.sort(key=lambda x: x.timestamp)

        # ALWAYS keep only the last K items (count-based window)
        if len(items) > self.window_size:
            items = items[-self.window_size:]

        # Assign scores for any further budget-based eviction
        n = len(items)
        for i, item in enumerate(items):
            item.score = i / max(n - 1, 1)

        return items


# =============================================================================
# STRATEGY 3: Random Eviction
# =============================================================================

def random_strategy(items: List, **kwargs) -> List:
    """
    Random eviction. Each item gets a random score.
    Eviction order is non-deterministic.
    """
    for item in items:
        item.score = random.random()
    return items


# =============================================================================
# STRATEGY 4: Oldest-First
# =============================================================================

def oldest_first_strategy(items: List, **kwargs) -> List:
    """
    Oldest-first. Score by original conversation timestamp.

    Difference from FIFO: FIFO scores by current buffer index (position
    in the list). Oldest-first scores by item.timestamp (original position
    in the conversation). These diverge when:
    - Previous eviction rounds removed intermediate items
    - Items were reinserted or reordered
    - Buffer was managed by a different strategy previously

    In a freshly-built chronological buffer they behave identically.
    The difference emerges in multi-strategy experiments and ablations.
    """
    if not items:
        return items

    min_t = min(item.timestamp for item in items)
    max_t = max(item.timestamp for item in items)
    spread = max(max_t - min_t, 1)

    for item in items:
        item.score = (item.timestamp - min_t) / spread

    return items


# =============================================================================
# STRATEGY 5: Lowest-Attention
# =============================================================================

def _as_score_list(scores, items: List) -> Optional[List[float]]:
    """
    Normalize several reasonable LLM-return formats into a score list.

    Accepted formats:
      - list/tuple aligned with current buffer items
      - dict keyed by item.summary
      - dict keyed by item.utterance_id
    """
    if scores is None:
        return None

    if isinstance(scores, dict):
        values = []
        for item in items:
            if item.summary in scores:
                values.append(scores[item.summary])
            elif item.utterance_id in scores:
                values.append(scores[item.utterance_id])
            else:
                return None
        return [float(v) for v in values]

    if isinstance(scores, Sequence) and not isinstance(scores, (str, bytes)):
        if len(scores) != len(items):
            return None
        return [float(v) for v in scores]

    return None


def _sanitize_scores(scores: List[float]) -> List[float]:
    clean = []
    for score in scores:
        if math.isfinite(score):
            clean.append(float(score))
        else:
            clean.append(0.0)
    return clean


def _build_attention_prompt(llm, context: str, current_utterance: dict) -> str:
    speaker = (current_utterance or {}).get("speaker", "")
    text = (current_utterance or {}).get("text", "")
    speaker_label = speaker.capitalize() if speaker else "Speaker"
    user_text = f"[Context]: {context or 'Start of visit'}\n[Utterance]: [{speaker_label}] {text}"

    tokenizer = getattr(llm, "tok", None) or getattr(llm, "tokenizer", None)
    system_prompt = getattr(llm, "system_prompt", None)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        except Exception:
            # Fall back to raw text below; the caller will still use true
            # attention, just with a simpler prompt wrapper.
            pass

    if system_prompt:
        return f"{system_prompt}\n\n{user_text}"
    return user_text


def _score_with_hf_attention(
    items: List,
    llm,
    current_utterance: dict,
    context: str,
) -> Optional[List[float]]:
    """
    Compute lowest-attention scores directly from a Hugging Face causal LM.

    This is intentionally conservative: if the backend cannot return
    attentions, we return None and the public strategy raises. That prevents
    reporting FIFO/oldest/similarity as an "attention" baseline.
    """
    model = getattr(llm, "model", None)
    tokenizer = getattr(llm, "tok", None) or getattr(llm, "tokenizer", None)
    if model is None or tokenizer is None:
        return None
    if not getattr(tokenizer, "is_fast", False):
        return None

    try:
        import torch
    except ImportError:
        return None

    raw_context = context or "Start of visit"
    prompt = _build_attention_prompt(llm, raw_context, current_utterance or {})

    context_start = prompt.find(raw_context)
    if context_start < 0:
        # If a chat template rewrites the text in an unexpected way, use a
        # plain prompt so character spans remain trustworthy.
        prompt = f"[Context]: {raw_context}\n[Utterance]: [{(current_utterance or {}).get('speaker', '').capitalize()}] {(current_utterance or {}).get('text', '')}"
        context_start = prompt.find(raw_context)

    enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)

    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        return None

    # Last prompt token predicts the first generated token. Its attention over
    # prior context is the baseline signal.
    attn = attentions[-1][0, :, -1, :].float().mean(dim=0).detach().cpu()

    scores: List[Optional[float]] = []
    search_from = 0
    for item in items:
        idx = raw_context.find(item.summary, search_from)
        if idx < 0:
            scores.append(None)
            continue

        start = context_start + idx
        end = start + len(item.summary)
        token_ids = [
            token_idx for token_idx, (tok_start, tok_end) in enumerate(offsets)
            if tok_end > start and tok_start < end
        ]
        if token_ids:
            scores.append(float(attn[token_ids].mean().item()))
        else:
            scores.append(0.0)
        search_from = idx + len(item.summary)

    known = [score for score in scores if score is not None]
    current_item_score = (max(known) + 1e-6) if known else 1.0
    return [
        current_item_score if score is None else score
        for score in scores
    ]


def lowest_attention_strategy(items: List, **kwargs) -> List:
    """
    Lowest-attention baseline from the spec.

    Uses LLM attention weights from the current inference context. Items with
    lower attention scores are evicted first by ContextBuffer.evict().

    Required input, passed by the harness:
      - llm: backend exposing either score_buffer_attention(...) or a HF
        causal-LM pair as .model plus .tok/.tokenizer
      - current_utterance/context: used to reconstruct the scored prompt

    If real attention weights are unavailable, this raises instead of silently
    falling back to oldest/FIFO. That keeps baseline numbers honest.
    """
    if not items:
        return items

    llm = kwargs.get("llm")
    current_utterance = kwargs.get("current_utterance") or {}
    context = kwargs.get("context") or "Start of visit"

    scores = None
    if llm is not None and hasattr(llm, "score_buffer_attention"):
        try:
            scores = llm.score_buffer_attention(
                items=items,
                current_utterance=current_utterance,
                context=context,
            )
        except NotImplementedError:
            scores = None

    score_list = _as_score_list(scores, items)
    if score_list is None and llm is not None:
        score_list = _score_with_hf_attention(items, llm, current_utterance, context)

    if score_list is None:
        raise RuntimeError(
            "attention baseline requires real LLM attention weights. "
            "Provide llm.score_buffer_attention(...) or a Hugging Face LLM "
            "with .model and fast .tok/.tokenizer loaded with attention output "
            "support, e.g. attn_implementation='eager'."
        )

    for item, score in zip(items, _sanitize_scores(score_list)):
        item.score = score

    return items


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================

# Default sliding window with K=10
_default_sliding = SlidingWindowStrategy(window_size=10)

STRATEGIES = {
    "fifo": fifo_strategy,
    "sliding": _default_sliding,
    "random": random_strategy,
    "oldest": oldest_first_strategy,
    "attention": lowest_attention_strategy,
}


def get_strategy(name: str, **kwargs) -> Callable:
    """
    Get eviction strategy by name.

    For sliding window, pass window_size:
        get_strategy("sliding", window_size=5)
    """
    if name == "sliding" and "window_size" in kwargs:
        return SlidingWindowStrategy(window_size=kwargs["window_size"])

    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]


def list_strategies() -> List[str]:
    """List all available strategy names."""
    return list(STRATEGIES.keys())
