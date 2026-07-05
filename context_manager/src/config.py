"""
ATLAS Configuration
====================
All tunable parameters in one place.
Change values here, not in code.
"""
from dataclasses import dataclass, field
from typing import List
@dataclass
class TrackerConfig:
    """System Tracker thresholds — may be tuned later."""
    linking_threshold: float = 0.25      # cosine sim to link summary → agenda item
    repeat_threshold: float = 0.85       # cosine sim above which = repeated info
    embedding_model: str = "all-MiniLM-L6-v2"
@dataclass
class PolicyConfig:
    """Context Manager / Eviction Policy."""
    feature_dim: int = 396               # 384 embedding + 12 scalar/onehot features (7 types)
    hidden_1: int = 128
    hidden_2: int = 64
    output_dim: int = 1
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    patience: int = 10                   # early stopping
    weight_decay: float = 1e-4
    relevance_threshold: float = 0.75    # BERTScore threshold for hindsight labels
@dataclass
class BufferConfig:
    """Context buffer settings."""
    budget: int = 1024                   # token budget B
    # DEPRECATED: max_context_items is no longer used by harness.
    # The buffer shows ALL items to LLM (managed by eviction policy).
    # Kept only for annotation_pipeline --mode format compatibility.
    max_context_items: int = 5
    budget_sweep: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048, 4096])
@dataclass
class LLMConfig:
    """Agenda LLM inference settings."""
    model_path: str = "models/gguf/llama-3.2-3b-q4_k_m.gguf"
    n_ctx: int = 8192                    # context window (must be > max budget + utterance + output headroom)
    # NOTE: budget 4096 + utterance (~100 tok) + output (~50 tok) + system prompt
    # needs at least ~4500 tokens. n_ctx=8192 gives safe headroom.
    n_gpu_layers: int = -1               # -1 = all layers on GPU
    temperature: float = 0.0
    max_tokens: int = 256
@dataclass
class AtlasConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
# Global default config
DEFAULT_CONFIG = AtlasConfig()
