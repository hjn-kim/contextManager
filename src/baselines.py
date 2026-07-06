"""
ATLAS Baseline Eviction Strategies
====================================

Spec baseline 5:
  1. Growing window (FIFO)  — token-budget driven, evict oldest when over B
  2. Sliding window          — count-based, always keep last K items only
  3. Random eviction         — random selection
  4. Oldest-first            — by original timestamp (differs from FIFO after reordering)
  5. Lowest-attention        — by LLM attention weights (deferred/placeholder)

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
  strategy(items: List[BufferItem]) → List[BufferItem]
  Returns the items to KEEP (may be shorter than input).
  Remaining items must have .score set for tiebreaking if further
  eviction is needed.
"""

import random
from typing import List, Callable


# =============================================================================
# STRATEGY 1: FIFO / Growing Window
# =============================================================================

def fifo_strategy(items: List) -> List:
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

    def __call__(self, items: List) -> List:
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

def random_strategy(items: List) -> List:
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

def oldest_first_strategy(items: List) -> List:
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
# STRATEGY 5: Lowest-Attention (PLACEHOLDER)
# =============================================================================

def lowest_attention_strategy(items: List) -> List:
    """
    Lowest-attention baseline. PLACEHOLDER — NOT YET IMPLEMENTED.

    Intended behavior: use LLM attention weights from the most recent
    inference call to score summaries. Items that received the least
    attention are evicted first.

    Requires llama.cpp attention weight extraction, which is not yet
    available. Currently falls back to oldest-first as a temporary
    stand-in. This MUST be replaced before this baseline can be
    included in final experiment results.

    TODO: integrate with LlamaCppLLM.get_attention_weights()
    """
    return oldest_first_strategy(items)


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
