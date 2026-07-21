"""
ATLAS Streaming Simulation Harness
====================================
Core loop:
  utterance → Agenda LLM → System Tracker → Context Manager → buffer → next

This file orchestrates the three components.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from pathlib import Path

from config import AtlasConfig, DEFAULT_CONFIG


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LLMOutput:
    """What the Agenda LLM produces per utterance.
    
    Supports two formats:
    1. Single: relevant + type + summary (legacy, spec v4)
    2. detail_list: relevant + detail_list (current annotation format)
       1 utterance can produce multiple (type, summary) pairs.
    """
    relevant: bool
    type: Optional[str] = None      # agenda_item, detail, medication, social_history, follow_up, question_unanswered, question
    summary: Optional[str] = None
    detail_list: Optional[List[Dict]] = None  # [{"type": str, "summary": str}, ...]


@dataclass
class BufferItem:
    """One summary sitting in the context buffer."""
    summary: str
    type: str
    speaker: str                     # "doctor" or "patient"
    utterance_id: str
    timestamp: int                   # position in conversation (0-indexed)
    token_count: int = 0
    score: float = 0.0               # assigned by eviction policy


@dataclass
class HarnessResult:
    """Full result of running one conversation through the harness."""
    conversation_id: str
    llm_outputs: List[Dict]          # per-utterance LLM outputs
    tracker_states: List[Dict]       # per-utterance tracker snapshots
    buffer_snapshots: List[List[str]]  # per-utterance buffer state
    latencies: List[float]           # per-utterance latency in ms
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# AGENDA LLM INTERFACE
# =============================================================================

class AgendaLLM:
    """
    Interface for the Agenda LLM.
    Subclass this for different backends (llama.cpp, mock, etc.)
    """

    def __init__(self, config: AtlasConfig):
        self.config = config

    def predict(self, context: str, speaker: str, text: str) -> LLMOutput:
        """
        Input:  [Context]: s1 | s2 | s3
                [Utterance]: [Speaker] text
        Output: LLMOutput(relevant, type, summary)
        """
        raise NotImplementedError

    def get_token_count(self, text: str) -> int:
        """
        Count tokens for budget tracking.

        WARNING: current implementation is a rough character-based estimate.
        For real experiments, this MUST be replaced with the actual LLM
        tokenizer (e.g. llama.cpp tokenizer). Budget sweep results are
        only valid with accurate token counts.

        TODO: override in LlamaCppLLM with actual tokenizer.
        """
        # Rough estimate: 1 token ≈ 4 characters (English text)
        # This is a PLACEHOLDER — not suitable for final experiments
        return max(len(text) // 4, 1)

    def score_buffer_attention(self, items, current_utterance: dict, context: str):
        """
        Optional hook for the spec's lowest-attention C3 baseline.

        Real LLM backends can override this and return one score per
        BufferItem. MockLLM deliberately does not implement it.
        """
        raise NotImplementedError


class MockLLM(AgendaLLM):
    """
    Mock LLM that uses pre-existing annotations.
    For testing harness without actual LLM inference.
    """

    def __init__(self, config: AtlasConfig, annotations: Dict[str, dict]):
        """
        annotations: {utterance_id: {"relevant": bool, "type": str, "summary": str}}
        """
        super().__init__(config)
        self.annotations = annotations

    @staticmethod
    def _model_detail_list(detail_list: List[Dict]) -> List[Dict]:
        """Return only fields the Agenda LLM is supposed to emit."""
        return [
            {"type": item["type"], "summary": item["summary"]}
            for item in detail_list or []
            if item.get("type") and item.get("summary")
        ]

    def predict(self, context: str, speaker: str, text: str,
                utterance_id: str = None) -> LLMOutput:
        if utterance_id and utterance_id in self.annotations:
            ann = self.annotations[utterance_id]
            if "detail_list" in ann:
                # New format: detail_list with multiple (type, summary) pairs
                detail_list = self._model_detail_list(ann["detail_list"])
                return LLMOutput(
                    relevant=ann["relevant"],
                    detail_list=detail_list,
                    # Also set first item as type/summary for backward compat
                    type=detail_list[0]["type"] if detail_list else None,
                    summary=detail_list[0]["summary"] if detail_list else None,
                )
            else:
                # Legacy format: single type/summary
                return LLMOutput(
                    relevant=ann["relevant"],
                    type=ann.get("type"),
                    summary=ann.get("summary")
                )
        return LLMOutput(relevant=False)


# =============================================================================
# CONTEXT BUFFER
# =============================================================================

class ContextBuffer:
    """
    Manages the pipe-delimited summary buffer.
    Delegates eviction decisions to a strategy (baseline or learned policy).
    """

    def __init__(self, config: AtlasConfig):
        self.config = config
        self.items: List[BufferItem] = []

    def add(self, item: BufferItem):
        """Add a new summary to the buffer."""
        self.items.append(item)

    def get_context_string(self) -> str:
        """
        Build pipe-delimited context for the LLM.

        Returns ALL items in buffer — the eviction policy already manages
        what's in the buffer to fit within B tokens. Do NOT additionally
        limit by max_context_items here, as that would make budget sweep
        experiments meaningless (LLM would see the same 5 summaries
        regardless of budget).
        """
        if not self.items:
            return "Start of visit"
        return " | ".join(item.summary for item in self.items)

    def get_total_tokens(self) -> int:
        """Total tokens in buffer."""
        return sum(item.token_count for item in self.items)

    def needs_eviction(self) -> bool:
        """Does the buffer exceed the budget?"""
        return self.get_total_tokens() > self.config.buffer.budget

    def evict(self, strategy, **strategy_kwargs):
        """
        Run eviction strategy, then enforce budget.

        Two-phase process:
        1. Strategy runs first — may return fewer items (e.g., sliding
           window trims to last K items regardless of budget)
        2. Budget check — if still over B tokens, evict lowest-scoring

        This means:
        - FIFO: strategy assigns scores, no items removed. Budget check
          then removes oldest if needed.
        - Sliding: strategy trims to K items. Budget check then removes
          more if K items still exceed budget.
        """
        if not self.items:
            return

        # Phase 1: strategy (may filter AND/OR score)
        self.items = strategy(self.items, **strategy_kwargs)

        # Phase 2: budget enforcement
        if self.needs_eviction():
            # Sort by score ascending (lowest = least important)
            self.items.sort(key=lambda x: x.score)

            # Remove lowest-scoring items until budget met
            while self.needs_eviction() and len(self.items) > 1:
                self.items.pop(0)

            # Edge case: single remaining item exceeds budget
            if self.needs_eviction() and len(self.items) == 1:
                import sys
                print(f"WARNING: single item ({self.items[0].token_count} tokens) "
                      f"exceeds budget ({self.config.buffer.budget}). "
                      f"Budget violated.", file=sys.stderr)

        # Restore chronological order
        self.items.sort(key=lambda x: x.timestamp)

    def snapshot(self) -> List[str]:
        """Current buffer state as list of summaries."""
        return [item.summary for item in self.items]


# =============================================================================
# MAIN HARNESS
# =============================================================================

class StreamingHarness:
    """
    Main simulation loop.

    Usage:
        harness = StreamingHarness(config)
        harness.load_conversation("path/to/D2N001.json")
        result = harness.run(llm, tracker, eviction_strategy)
    """

    def __init__(self, config: AtlasConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.conversation = None

    def load_conversation(self, path: str):
        """Load annotated conversation JSON."""
        with open(path) as f:
            self.conversation = json.load(f)
        return self

    def load_conversation_dict(self, data: dict):
        """Load from dict directly."""
        self.conversation = data
        return self

    def run(self, llm: AgendaLLM, tracker=None, eviction_strategy=None) -> HarnessResult:
        """
        Run the full simulation loop.

        For each utterance:
          1. Build context from buffer
          2. LLM predicts relevant/type/summary
          3. If relevant: tracker updates, summary added to buffer
          4. If buffer overflows: eviction strategy runs
          5. Log everything
        """
        if self.conversation is None:
            raise ValueError("No conversation loaded. Call load_conversation() first.")

        buffer = ContextBuffer(self.config)
        result = HarnessResult(
            conversation_id=self.conversation["id"],
            llm_outputs=[],
            tracker_states=[],
            buffer_snapshots=[],
            latencies=[]
        )

        utterances = self.conversation["utterances"]

        for i, utt in enumerate(utterances):
            t_start = time.perf_counter()

            # 1. Build context
            context = buffer.get_context_string()

            # 2. LLM prediction
            if isinstance(llm, MockLLM):
                output = llm.predict(context, utt["speaker"], utt["text"],
                                     utterance_id=utt["id"])
            else:
                output = llm.predict(context, utt["speaker"], utt["text"])

            # 3. If relevant, add to buffer
            if output.relevant:
                # Build list of (type, summary) pairs from detail_list or single
                details = []
                if output.detail_list:
                    details = [(d["type"], d["summary"]) for d in output.detail_list]
                elif output.summary:
                    details = [(output.type or "detail", output.summary)]

                for dtype, dsummary in details:
                    item = BufferItem(
                        summary=dsummary,
                        type=dtype,
                        speaker=utt["speaker"],
                        utterance_id=utt["id"],
                        timestamp=i,
                        token_count=llm.get_token_count(dsummary)
                    )
                    buffer.add(item)

            # Update tracker for every utterance, including irrelevant turns.
            if tracker:
                tracker.update(output, utt)

            # 4. Run eviction strategy every turn. Irrelevant turns do not add
            # items, but keeping this outside the relevance branch preserves the
            # loop contract.
            if eviction_strategy:
                buffer.evict(
                    eviction_strategy,
                    llm=llm,
                    current_utterance=utt,
                    context=context,
                )

            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000

            # 5. Log
            result.llm_outputs.append(asdict(output))
            result.buffer_snapshots.append(buffer.snapshot())
            result.latencies.append(latency_ms)
            if tracker:
                result.tracker_states.append(tracker.get_state())
            else:
                result.tracker_states.append({})

        # Finalize tracker (marks unresolved items, updates predictions)
        if tracker:
            tracker.finalize()
            # Update last tracker snapshot to reflect finalized state
            if result.tracker_states:
                result.tracker_states[-1] = tracker.get_state()

        result.metadata = {
            "total_utterances": len(utterances),
            "total_relevant": sum(1 for o in result.llm_outputs if o["relevant"]),
            "final_buffer_size": len(buffer.items),
            "final_buffer_tokens": buffer.get_total_tokens(),
            "avg_latency_ms": sum(result.latencies) / max(len(result.latencies), 1),
        }

        return result


# =============================================================================
# CONVENIENCE: run from command line
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ATLAS Streaming Harness")
    parser.add_argument("--input", required=True, help="Path to annotated conversation JSON")
    parser.add_argument("--budget", type=int, default=1024, help="Token budget B")
    parser.add_argument("--strategy", default="fifo",
                        choices=["fifo", "sliding", "random", "oldest", "attention", "oracle"],
                        help="Eviction strategy (oracle = unlimited budget upper bound)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Config
    config = AtlasConfig()

    # Oracle = unlimited budget (upper bound, not a baseline strategy)
    if args.strategy == "oracle":
        config.buffer.budget = 999999
        strategy = None  # No eviction needed
    else:
        config.buffer.budget = args.budget
        from baselines import get_strategy
        strategy = get_strategy(args.strategy)

    # Load conversation and build mock LLM from annotations
    with open(args.input) as f:
        conv = json.load(f)

    annotations = {}
    for ann in conv.get("annotations", []):
        annotations[ann["utterance_id"]] = ann

    llm = MockLLM(config, annotations)

    # Run
    harness = StreamingHarness(config)
    harness.load_conversation_dict(conv)
    result = harness.run(llm, eviction_strategy=strategy)

    # Output
    print(f"\n{'='*60}")
    print(f"Conversation: {result.conversation_id}")
    print(f"Strategy: {args.strategy} (budget={'unlimited' if args.strategy == 'oracle' else args.budget})")
    print(f"Total utterances: {result.metadata['total_utterances']}")
    print(f"Total relevant: {result.metadata['total_relevant']}")
    print(f"Final buffer: {result.metadata['final_buffer_size']} items, "
          f"{result.metadata['final_buffer_tokens']} tokens")
    print(f"Avg latency: {result.metadata['avg_latency_ms']:.2f} ms")
    print(f"{'='*60}")
    print(f"\nFinal buffer contents:")
    for s in result.buffer_snapshots[-1]:
        print(f"  • {s}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
