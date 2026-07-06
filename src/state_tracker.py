"""
ATLAS System Tracker (Component 2)
====================================
Deterministic, rule-based. No LLM calls. Runs on CPU.

Responsibilities:
  - Linking: associate summary with parent agenda item
  - First-mention detection: new info vs repeated
  - Resolution tracking: mentioned → discussed → resolved → unresolved
  - Nesting: details under agenda items
  - Visit phase: greeting → history → exam → planning → wrap_up
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from config import TrackerConfig, DEFAULT_CONFIG


@dataclass
class AgendaItem:
    """One agenda item with nested details."""
    id: str
    topic: str                          # summary text of the agenda item
    status: str = "mentioned"           # mentioned / discussed / resolved / unresolved
    first_mentioned: str = ""           # utterance_id
    speakers: List[str] = field(default_factory=list)  # who contributed
    details: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    follow_ups: List[str] = field(default_factory=list)
    unanswered: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None


class SystemTracker:
    """
    Maintains hierarchical agenda graph.
    Updated after each Agenda LLM output.
    """

    def __init__(self, config: TrackerConfig = None):
        self.config = config or DEFAULT_CONFIG.tracker
        self.agenda_items: List[AgendaItem] = []
        self.all_summaries: List[str] = []       # for first-mention detection
        self.visit_phase: str = "greeting"
        self._encoder = None                      # lazy-loaded sentence encoder
        self._item_counter = 0

        # Per-utterance prediction log (for evaluation against _eval_only)
        self.predictions: List[Dict] = []

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def update(self, llm_output, utterance: dict):
        """
        Process one LLM output.
        llm_output: LLMOutput with detail_list or single type/summary
        utterance: {"id", "speaker", "text"}
        """
        if not llm_output.relevant:
            self.predictions.append({
                "utterance_id": utterance["id"],
                "relevant": False,
                "linked_agenda_item_id": None,
                "first_mention": None,
                "resolution_status": None,
            })
            return

        speaker = utterance["speaker"]
        utt_id = utterance["id"]

        # Update visit phase
        self._update_visit_phase(utterance["text"])

        # Build list of (type, summary) pairs
        details = []
        if llm_output.detail_list:
            details = [(d["type"], d["summary"]) for d in llm_output.detail_list]
        elif llm_output.summary:
            details = [(llm_output.type or "detail", llm_output.summary)]

        # Process each detail
        for utt_type, summary in details:
            is_first = self._is_first_mention(summary)
            self.all_summaries.append(summary)

            linked_item_id = None
            if utt_type == "agenda_item":
                new_item = self._create_agenda_item(summary, utt_id, speaker)
                linked_item_id = new_item.id
            else:
                parent = self._find_parent(summary)
                if parent is None:
                    new_item = self._create_agenda_item(summary, utt_id, speaker)
                    linked_item_id = new_item.id
                else:
                    linked_item_id = parent.id
                    self._nest_under(parent, summary, utt_type, speaker)
                    self._update_resolution(parent, utt_type, speaker)

            self.predictions.append({
                "utterance_id": utt_id,
                "relevant": True,
                "linked_agenda_item_id": linked_item_id,
                "first_mention": is_first,
                "resolution_status": self._get_item_status(linked_item_id),
            })

    def finalize(self):
        """
        Call after conversation ends.
        1. Marks agenda items that were only 'mentioned' as 'unresolved'.
        2. Marks items with unanswered questions appropriately.
        3. Updates predictions list with final resolution statuses.
        """
        # Update agenda item statuses
        for item in self.agenda_items:
            if item.status == "mentioned":
                item.status = "unresolved"
            if item.unanswered:
                if item.status == "resolved":
                    item.status = "discussed"

        # Update predictions with final statuses
        for pred in self.predictions:
            if pred["linked_agenda_item_id"]:
                pred["resolution_status"] = self._get_item_status(
                    pred["linked_agenda_item_id"]
                )

    def get_state(self) -> dict:
        """
        Current tracker state as dict.
        Returns a deep copy to prevent snapshot aliasing —
        modifying this dict won't affect internal state.
        """
        import copy
        state = {
            "agenda_items": [
                {
                    "id": item.id,
                    "topic": item.topic,
                    "status": item.status,
                    "first_mentioned": item.first_mentioned,
                    "details": list(item.details),
                    "medications": list(item.medications),
                    "follow_ups": list(item.follow_ups),
                    "unanswered": list(item.unanswered),
                }
                for item in self.agenda_items
            ],
            "visit_phase": self.visit_phase,
            "predictions": copy.deepcopy(self.predictions),
        }
        return state

    def get_predictions(self) -> List[Dict]:
        """
        Per-utterance predictions for evaluation.
        Each entry: {utterance_id, relevant, linked_agenda_item_id,
                     first_mention, resolution_status}
        Compare against _eval_only fields in eval JSON.
        """
        return self.predictions

    def _get_item_status(self, item_id: str) -> Optional[str]:
        """Get current status of an agenda item by ID."""
        for item in self.agenda_items:
            if item.id == item_id:
                return item.status
        return None

    # -----------------------------------------------------------------
    # Linking
    # -----------------------------------------------------------------

    def _find_parent(self, summary: str) -> Optional[AgendaItem]:
        """
        Find the agenda item most similar to this summary.
        Returns None if no match above threshold.
        """
        if not self.agenda_items:
            return None

        encoder = self._get_encoder()
        if encoder is None:
            # Fallback: no encoder available, can't link
            return None

        summary_emb = encoder.encode(summary)
        best_item = None
        best_sim = -1

        for item in self.agenda_items:
            if item.embedding is None:
                item.embedding = encoder.encode(item.topic).tolist()
            sim = self._cosine_sim(summary_emb, item.embedding)
            if sim > best_sim:
                best_sim = sim
                best_item = item

        if best_sim >= self.config.linking_threshold:
            return best_item
        return None

    # -----------------------------------------------------------------
    # First-mention detection
    # -----------------------------------------------------------------

    def _is_first_mention(self, summary: str) -> bool:
        """
        Is this summary new information or a repeat?
        Returns True if first mention.
        """
        if not self.all_summaries:
            return True

        encoder = self._get_encoder()
        if encoder is None:
            return True  # Can't check, assume new

        summary_emb = encoder.encode(summary)
        for prior in self.all_summaries:
            prior_emb = encoder.encode(prior)
            sim = self._cosine_sim(summary_emb, prior_emb)
            if sim > self.config.repeat_threshold:
                return False  # Too similar → repeated

        return True

    # -----------------------------------------------------------------
    # Resolution tracking
    # -----------------------------------------------------------------

    def _update_resolution(self, item: AgendaItem, utt_type: str, speaker: str):
        """
        State machine:
          mentioned → discussed (both speakers contribute)
          discussed → resolved (follow_up references it)
        """
        if speaker not in item.speakers:
            item.speakers.append(speaker)

        if utt_type == "follow_up":
            item.status = "resolved"
        elif len(item.speakers) >= 2 and item.status == "mentioned":
            item.status = "discussed"

    # -----------------------------------------------------------------
    # Nesting
    # -----------------------------------------------------------------

    def _nest_under(self, parent: AgendaItem, summary: str,
                    utt_type: str, speaker: str):
        """Place summary under the parent agenda item by type."""
        if utt_type == "detail":
            parent.details.append(summary)
        elif utt_type == "medication":
            parent.medications.append(summary)
        elif utt_type == "follow_up":
            parent.follow_ups.append(summary)
        elif utt_type == "question_unanswered":
            parent.unanswered.append(summary)
        elif utt_type == "social_history":
            parent.details.append(summary)  # social_history nests as detail
        else:
            parent.details.append(summary)

    def _create_agenda_item(self, summary: str, utt_id: str, speaker: str) -> AgendaItem:
        """
        Create a new agenda item.
        ID = utterance_id where it was first mentioned (e.g., "line_0004").
        This matches gold _eval_only.linked_agenda_item_id format,
        enabling direct comparison in compute_linking_accuracy().
        """
        item = AgendaItem(
            id=utt_id,  # Use utterance_id, NOT synthetic A001
            topic=summary,
            first_mentioned=utt_id,
            speakers=[speaker],
        )
        self.agenda_items.append(item)
        return item

    # -----------------------------------------------------------------
    # Visit phase detection
    # -----------------------------------------------------------------

    PHASE_KEYWORDS = {
        "greeting": ["how are you", "hello", "hi ", "good morning", "nice to see"],
        "history_taking": ["how long", "when did", "any symptoms", "tell me about",
                          "how are you doing with", "have you been"],
        "examination": ["physical exam", "let me check", "blood pressure",
                       "heart sounds", "lungs", "examination"],
        "planning": ["i want to", "i'd like to", "we'll", "plan is",
                    "i'm going to order", "let's go ahead"],
        "wrap_up": ["any questions", "good to see you", "take care",
                   "finalize the note"],
    }

    def _update_visit_phase(self, text: str):
        """Rule-based keyword detection for visit phase."""
        text_lower = text.lower()
        for phase, keywords in self.PHASE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                # Phases only move forward
                phases = list(self.PHASE_KEYWORDS.keys())
                current_idx = phases.index(self.visit_phase)
                new_idx = phases.index(phase)
                if new_idx >= current_idx:
                    self.visit_phase = phase
                break

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _get_encoder(self):
        """Lazy-load sentence encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.config.embedding_model)
            except ImportError:
                print("WARNING: sentence-transformers not installed. Tracker linking disabled.")
                return None
        return self._encoder

    @staticmethod
    def _cosine_sim(a, b) -> float:
        """Cosine similarity between two vectors."""
        import numpy as np
        a = np.array(a).flatten()
        b = np.array(b).flatten()
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / max(norm, 1e-8))
