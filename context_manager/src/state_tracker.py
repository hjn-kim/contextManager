"""
ATLAS System Tracker (Component 2)
====================================
Deterministic, rule-based tracker for agenda state.

The tracker consumes model-facing outputs only:
  - relevant
  - type
  - summary / detail_list

It produces system-level fields used by eval_only metrics:
  - linked_agenda_item_id
  - first_mention
  - resolution_status

Design rule: agenda creation, child linking, and nesting are separate
decisions. A child detail that cannot be linked does not automatically create
a new agenda item.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import copy

from config import DEFAULT_CONFIG, TrackerConfig


@dataclass
class AgendaItem:
    """One agenda item with nested clinical summaries."""

    id: str
    topic: str
    status: str = "mentioned"
    first_mentioned: str = ""
    speakers: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    social_history: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    follow_ups: List[str] = field(default_factory=list)
    unanswered: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None


class SystemTracker:
    """
    Maintains a lightweight agenda graph.

    Agenda creation is separate from child attachment:
    - agenda_item details open agenda items directly
    - child details link to an existing agenda or remain unlinked
    - child details never create agenda items on their own
    """

    CHILD_TYPES = {
        "detail",
        "medication",
        "follow_up",
        "plan",
        "question",
        "question_unanswered",
    }
    GLOBAL_TYPES = {"social_history"}

    def __init__(self, config: TrackerConfig = None):
        self.config = config or DEFAULT_CONFIG.tracker
        self.agenda_items: List[AgendaItem] = []
        self.all_summaries: List[str] = []
        self.visit_phase: str = "greeting"
        self._encoder = None
        self._active_agenda_item: Optional[AgendaItem] = None
        self.global_items: Dict[str, List[str]] = {
            "social_history": [],
            "unlinked": [],
        }
        self.predictions: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, llm_output, utterance: dict):
        """Process one LLM output for one utterance."""
        speaker = utterance["speaker"]
        utt_id = utterance["id"]
        utterance_text = utterance.get("text", "")

        self._update_visit_phase(utterance_text)

        if not llm_output.relevant:
            self.predictions.append({
                "utterance_id": utt_id,
                "detail_index": None,
                "relevant": False,
                "type": "irrelevant",
                "summary": None,
                "linked_agenda_item_id": None,
                "first_mention": None,
                "resolution_status": None,
            })
            return

        details = self._iter_details(llm_output)
        current_utterance_parent: Optional[AgendaItem] = None

        for detail_index, (utt_type, summary) in enumerate(details):
            is_first = self._is_first_mention(summary)
            self.all_summaries.append(summary)

            state_item_id = None
            predicted_link_id = None
            predicted_status = self._resolution_majority_baseline()

            if utt_type == "agenda_item":
                parent = self._create_agenda_item(summary, utt_id, speaker)
                current_utterance_parent = parent
                state_item_id = parent.id

            elif utt_type in self.GLOBAL_TYPES:
                self.global_items.setdefault(utt_type, []).append(summary)

            else:
                parent = self._resolve_child_parent(
                    summary=summary,
                    utt_type=utt_type,
                    utterance=utterance,
                    current_utterance_parent=current_utterance_parent,
                )

                if parent is not None:
                    self._active_agenda_item = parent
                    self._nest_under(parent, summary, utt_type, speaker)
                    self._update_resolution(parent, utt_type, speaker)
                    state_item_id = parent.id
                    predicted_link_id = parent.id
                else:
                    self.global_items.setdefault("unlinked", []).append(summary)

            self.predictions.append({
                "utterance_id": utt_id,
                "detail_index": detail_index,
                "relevant": True,
                "type": utt_type,
                "summary": summary,
                "linked_agenda_item_id": predicted_link_id,
                "first_mention": is_first,
                "resolution_status": predicted_status,
            })

    def finalize(self):
        """
        Finalize agenda item state only.

        Per-detail predictions remain event-time predictions for eval_only
        comparison and are not overwritten with final agenda state.
        """
        for item in self.agenda_items:
            if item.status == "mentioned":
                item.status = "unresolved"
            if item.unanswered and item.status == "resolved":
                item.status = "discussed"

    def get_state(self) -> dict:
        """Return a copy of the current tracker state."""
        return {
            "agenda_items": [
                {
                    "id": item.id,
                    "topic": item.topic,
                    "status": item.status,
                    "first_mentioned": item.first_mentioned,
                    "details": list(item.details),
                    "questions": list(item.questions),
                    "social_history": list(item.social_history),
                    "medications": list(item.medications),
                    "follow_ups": list(item.follow_ups),
                    "unanswered": list(item.unanswered),
                }
                for item in self.agenda_items
            ],
            "global_items": copy.deepcopy(self.global_items),
            "visit_phase": self.visit_phase,
            "predictions": copy.deepcopy(self.predictions),
        }

    def get_predictions(self) -> List[Dict]:
        """Return per-detail tracker predictions."""
        return self.predictions

    # ------------------------------------------------------------------
    # Detail routing
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_details(llm_output) -> List[Tuple[str, str]]:
        if llm_output.detail_list:
            return [
                (d["type"], d["summary"])
                for d in llm_output.detail_list
                if d.get("type") and d.get("summary")
            ]
        if llm_output.summary:
            return [(llm_output.type or "detail", llm_output.summary)]
        return []

    def _resolve_child_parent(
        self,
        summary: str,
        utt_type: str,
        utterance: dict,
        current_utterance_parent: Optional[AgendaItem],
    ) -> Optional[AgendaItem]:
        if current_utterance_parent is not None:
            return current_utterance_parent

        semantic_parent = self._find_semantic_parent(summary)
        if semantic_parent is not None:
            return semantic_parent

        return None

    def _find_semantic_parent(self, summary: str) -> Optional[AgendaItem]:
        if not self.agenda_items:
            return None

        encoder = self._get_encoder()
        if encoder is None:
            return None

        summary_emb = encoder.encode(summary)
        best_item = None
        best_score = -1.0

        for item in self.agenda_items:
            topic_emb = self._item_topic_embedding(item, encoder)
            topic_score = self._cosine_sim(summary_emb, topic_emb)

            agenda_text = self._agenda_text(item)
            context_score = topic_score
            if agenda_text and agenda_text != item.topic:
                context_score = self._cosine_sim(summary_emb, encoder.encode(agenda_text))

            score = max(topic_score, context_score)
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None:
            return None

        if best_score >= self.config.linking_threshold:
            return best_item

        return None

    # ------------------------------------------------------------------
    # First mention
    # ------------------------------------------------------------------

    def _is_first_mention(self, summary: str) -> bool:
        if not self.all_summaries:
            return True

        encoder = self._get_encoder()
        if encoder is None:
            return True

        summary_emb = encoder.encode(summary)
        for prior in self.all_summaries:
            prior_emb = encoder.encode(prior)
            if self._cosine_sim(summary_emb, prior_emb) > self.config.repeat_threshold:
                return False
        return True

    # ------------------------------------------------------------------
    # Resolution and nesting
    # ------------------------------------------------------------------

    def _update_resolution(self, item: AgendaItem, utt_type: str, speaker: str):
        if speaker not in item.speakers:
            item.speakers.append(speaker)
        if utt_type == "follow_up":
            item.status = "resolved"

    @staticmethod
    def _resolution_majority_baseline() -> str:
        """
        Majority-class placeholder for per-detail event-time resolution.

        This is not a real resolution tracker. It prevents final agenda state
        from overwriting historical predictions, but it should be reported as a
        majority baseline until a measured resolution signal is added.
        """
        return "mentioned"

    def _nest_under(self, parent: AgendaItem, summary: str, utt_type: str, speaker: str):
        if utt_type == "detail" or utt_type == "plan":
            parent.details.append(summary)
        elif utt_type == "medication":
            parent.medications.append(summary)
        elif utt_type == "follow_up":
            parent.follow_ups.append(summary)
        elif utt_type == "question_unanswered":
            parent.unanswered.append(summary)
        elif utt_type == "question":
            parent.questions.append(summary)
        elif utt_type == "social_history":
            parent.social_history.append(summary)
        else:
            parent.details.append(summary)

    def _create_agenda_item(self, summary: str, utt_id: str, speaker: str) -> AgendaItem:
        item = AgendaItem(
            id=utt_id,
            topic=summary,
            first_mentioned=utt_id,
            speakers=[speaker],
        )
        self.agenda_items.append(item)
        self._active_agenda_item = item
        return item

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _get_item_status(self, item_id: str) -> Optional[str]:
        for item in self.agenda_items:
            if item.id == item_id:
                return item.status
        return None

    def _agenda_text(self, item: AgendaItem) -> str:
        parts = [
            item.topic,
            *item.details,
            *item.questions,
            *item.social_history,
            *item.medications,
            *item.follow_ups,
            *item.unanswered,
        ]
        return " ".join(part for part in parts if part)

    @staticmethod
    def _item_topic_embedding(item: AgendaItem, encoder):
        if item.embedding is None:
            item.embedding = encoder.encode(item.topic).tolist()
        return item.embedding

    # ------------------------------------------------------------------
    # Visit phase detection
    # ------------------------------------------------------------------

    PHASE_KEYWORDS = {
        "greeting": ["how are you", "hello", "hi ", "good morning", "nice to see"],
        "history_taking": [
            "how long",
            "when did",
            "any symptoms",
            "tell me about",
            "how are you doing with",
            "have you been",
        ],
        "examination": [
            "physical exam",
            "let me check",
            "blood pressure",
            "heart sounds",
            "lungs",
            "examination",
        ],
        "planning": [
            "i want to",
            "i'd like to",
            "we'll",
            "plan is",
            "i'm going to order",
            "let's go ahead",
        ],
        "wrap_up": ["any questions", "good to see you", "take care", "finalize the note"],
    }

    def _update_visit_phase(self, text: str):
        text_lower = text.lower()
        phases = list(self.PHASE_KEYWORDS.keys())
        current_idx = phases.index(self.visit_phase)
        for phase, keywords in self.PHASE_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                new_idx = phases.index(phase)
                if new_idx >= current_idx:
                    self.visit_phase = phase
                break

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _get_encoder(self):
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
        import numpy as np

        a = np.array(a).flatten()
        b = np.array(b).flatten()
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / max(norm, 1e-8))
