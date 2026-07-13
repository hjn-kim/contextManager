# Project Spec v4: Learned Context Management for Streaming Summarization

**Target:**  AAAI 2026 (~July deadline)
**Platform:** NVIDIA Jetson AGX Orin (JetPack 6.x, CUDA 12.x)

---

## Problem Statement

A conversation streams in as utterances. After each utterance, the agenda LLM produces a short summary. A buffer of prior summaries serves as context for the next utterance. The buffer has a fixed capacity of B tokens. When it overflows, a learned eviction policy scores every summary and evicts the lowest-scoring ones until the buffer fits.

We are building and evaluating this eviction policy against ad hoc baselines (FIFO, sliding window, random, attention-based).

---

## Architecture: Model vs. System vs. Context Manager

Three separate components. Do not conflate them.

```
Utterance arrives
      │
      ▼
┌──────────────────────────────────────────────────────┐
│ AGENDA LLM (3B, Q4_K_M, runs on GPU)                │
│                                                      │
│ Input:                                               │
│   [Context]: summary_1 | summary_2 | ... | summary_k │
│   [Utterance]: [Patient] I've had this cough for...  │
│                                                      │
│ Output:                                              │
│   {"relevant": true, "type": "detail",               │
│    "summary": "Persistent cough for three weeks"}    │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│ SYSTEM TRACKER (Python, deterministic, runs on CPU)  │
│                                                      │
│ • Links summary to parent agenda item (embedding sim)│
│ • Detects first mention (cosine sim against prior)   │
│ • Tracks resolution status (rule-based)              │
│ • Nests details under agenda items                   │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│ CONTEXT MANAGER / EVICTION POLICY (MLP, <10ms, CPU)  │
│                                                      │
│ • Adds new summary to buffer                         │
│ • If buffer > B tokens: score all, evict lowest      │
│ • Buffer feeds back as [Context] for next utterance   │
└──────────────────────────────────────────────────────┘
```

---

## Component 1: Agenda LLM — 3 Fields Only

The model sees the utterance and outputs three fields. 'type' and 'summary' is grouped in detail_list.

**Input:**

```
[Utterance]: [Doctor] "looking at your blood pressure today here in the office , it is a little elevated . show me the blood pressure readings. i'd like to see you take your medication a little bit more , so that we can get that under control a little bit better"
```

**Output (relevant):**

```json
{
  "utterance_id": "line_0026",
  "relevant": true,
  "detail_list": [
    {
      "type": "detail",
      "summary": "Doctor notes elevated blood pressure."
    },
    {
      "type": "follow_up",
      "summary": "Doctor advises the patient to take medication consistently to improve blood pressure control."
    }
  ]
}
```

**Output (not relevant):**

```json
{"relevant": false}
```

**Field definitions:**

| Field | Type | Values |
|-------|------|--------|
| `relevant` | bool | Does this utterance contain clinically relevant information? |
| `type` | string | `agenda_item`, `detail`, `medication`, `social_history`, `follow_up`, `question`,`question_unanswered` |
| `summary` | string | One concise clinical sentence per detail_list. Include specific values (e.g., "BP 140/90"). |

The model does NOT output: linking, resolution status, first mention, clinical category. Those are system tracker responsibilities.

**The 7 annotation types:**

| Type | When to use | Example summary |
|------|-------------|-----------------|
| `agenda_item` | Primary concern or goal for the visit. Most visits have 1–4. | Congestive heart failure |
| `detail` | Supporting information for an existing agenda item | Cough is predominantly dry with intermittent yellow sputum, worse in morning |
| `medication` | Drugs only when actual dosage, prescription, or administration regimen is explicitly stated | The doctor increases lisinopril to 40 mg daily and initiates Lasix 20 mg daily. |
| `social_history` | Lifestyle, occupation, habits | Former smoker, quit 5 years ago, 20 pack-year history |
| `follow_up` | Action items, referrals, tests, prescriptions | Plan: chest X-ray today, referral to pulmonology |
| `question` | Question related to clinical information. | Patient asks about weight concerns |
| `question_unanswered` | Question or concern raised but never addressed. **Highest priority.** | Patient asks about weight concerns — not addressed |


**Models:**

| Model | Format | Expected Mem | Expected Speed |
|-------|--------|-------------|----------------|
| Llama 3.2 1B | GGUF Q4_K_M | ~1GB | ~80–120 tok/s |
| Llama 3.2 3B | GGUF Q4_K_M | ~2GB | ~40–60 tok/s |
| Mistral 7B | GGUF Q4_K_M | ~4.5GB | ~15–25 tok/s |

Primary: Llama 3.2 3B. Others for generalization experiments.

**Fine-tuning:** QLoRA (rank 32–64). Three-phase curriculum:

| Phase | Data | Volume | LR | Purpose |
|-------|------|--------|----|---------|
| 1: Medical pre-train | Synthetic + NoteChat subset | ~50K examples | 1e-4 | General medical conversation understanding |
| 2: Task fine-tune | MTS-Dialog + ACI + PriMock + AMI | ~5K examples | 5e-5 | Agenda extraction with pipe-delimited context format |
| 3: Real calibration | 12 Penn Medicine train conversations | ~500 examples | 1e-5 | Align to real conversation patterns |

---

## Component 2: System Tracker (Deterministic Code)

The system tracker maintains a hierarchical agenda graph. Updated after each model output. Runs on CPU. No LLM calls.

| Task | Logic | Uses Model Output? |
|------|-------|--------------------|
| **Linking** | Embed new summary + each existing agenda item (all-MiniLM-L6-v2). Link to highest cosine similarity match above threshold 0.8. Create new agenda item if no match. | Uses `summary` text |
| **First-mention detection** | Cosine similarity of new summary against all prior summaries. If max sim > 0.85 → repeat. Otherwise → first mention. | Uses `summary` text |
| **Resolution tracking** | Starts at "mentioned." Both speakers contribute details → "discussed." A `follow_up` references it → "resolved." Conversation ends without either → "unresolved." | Uses `type` field |
| **Visit phase** | Rule-based keyword detection: greetings → history_taking → examination → planning → wrap_up. | Uses utterance text |
| **Nesting** | Place `detail`, `medication`, `follow_up`, `question_unanswered` under their parent `agenda_item` via the resolved link. | Uses linking output |

**Output state:**

```json
{
  "agenda_items": [
    {
      "id": "A001",
      "topic": "chronic cough",
      "status": "discussed",
      "first_mentioned": "line_005",
      "details": ["Cough is dry with yellow sputum", "Night sweats for ~1 week"],
      "medications": ["Taking OTC cough syrup, no improvement"],
      "follow_ups": [],
      "unanswered": ["Patient asks about chest X-ray"]
    }
  ],
  "visit_phase": "history_taking"
}
```

---

## Component 3: Context Manager / Eviction Policy

Separate model from the agenda LLM. Lightweight MLP (~1M params). Runs in <10ms on Jetson CPU.

**When it runs:** After the LLM produces a summary, before the next utterance arrives. No latency impact (utterances arrive every ~5 seconds).

**Architecture:**

```
Input features per summary in buffer (396-d):
  - Sentence embedding         (384-d, frozen all-MiniLM-L6-v2)
  - Position: i / t             (float, 0–1)
  - Recency: (t - i) / t        (float, 0–1)
  - Speaker                     (binary: 0=provider, 1=patient)
  - Token count                 (int)
  - Medical entity count         (int, via scispaCy en_core_sci_sm)
  - Type one-hot                 (7-d: agenda_item, detail, medication, social_history, follow_up, question, question_unanswered)

Total: 396-d

Network:
  Linear(396, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 1) → Sigmoid

Output: relevance score r_i ∈ [0, 1] per summary
```

**Decision logic:**
1. Score all summaries in buffer.
2. Sort by score ascending.
3. Evict lowest-scoring summaries until total tokens ≤ B.

No learned thresholds — just evict until budget is met.

**Optional extension (ablation condition):** Compress instead of evict for mid-range scores. Call LLM with compression prompt targeting ≤20 tokens. Acceptance: BERTScore > 0.80.

**Training — hindsight supervision:**

After a full conversation is processed, label each (summary, timestep) pair:

```python
for each summary s_i in buffer at time t:
    future_items = [a for a in all_annotations if a.utterance_id >= current_id]
    scores = [bertscore(s_i.summary, item.summary) for item in future_items]
    relevance_label = 1 if max(scores) > 0.75 else 0
```

Validate on 5 dialogues with leave-one-out perturbation (target Pearson r > 0.6).

**Training details:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 256 |
| Epochs | 50–100 (early stopping, patience=10) |
| Weight decay | 1e-4 |
| Class weights | Computed from label distribution |
| Expected training data | ~30,000 examples from ~50 conversations |
| Split | 70/15/15 by conversation (no leakage) |
| Training time | ~5–10 minutes on Jetson GPU |

**Intrinsic eval:** Spearman's ρ > 0.5 (ranking), F1 > 0.7 (binary keep/evict).

**Baselines (5 strategies):**

| Strategy | Logic |
|----------|-------|
| Growing window (FIFO) | Append until full, truncate oldest |
| Sliding window | Keep only most recent B tokens |
| Random eviction | Buffer full → evict random summary |
| Oldest-first | Buffer full → evict oldest summary |
| Lowest-attention | Use LLM attention weights; evict lowest-scoring |

---

## Component 4: Annotation Schema & Data Pipeline

### 4a. Three Data Products

The same annotated conversations produce three separate outputs:

**A. Agenda LLM training data** — condensed, 3-field format:

```jsonl
{"input": "[Context]: Patient presents with chronic cough | Cough produces yellow mucus\n[Utterance]: [Patient] Night sweats too, maybe for a week.", "output": "{\"relevant\": true, \"detail_list\":[{\"type\": \"detail\", \"summary\": \"Patient reports night sweats for approximately one week\"}]"}
```

**B. Context manager training data** — feature vectors + binary labels:

```json
{"features": [0.23, -0.14, ..., 0.87, 0.42, 1, 3, 0, 0, 1, 0, 0, 0], "label": 1}
```

**C. Evaluation data** — 3 model fields + `_eval_only` fields for system-level metrics:

```json
{
  "relevant": true, "type": "detail",
  "summary": "Patient reports night sweats for approximately one week",
  "_eval_only": {
    "linked_agenda_item_id": "line_0008",
    "first_mention": true,
    "resolution_status": "discussed",
    "clinical_category": "hpi"
  }
}
```

`_eval_only` fields are NEVER included in training data. They exist only for evaluating the system tracker.

| Eval-only field | What it evaluates | How the system computes it |
|-----------------|-------------------|---------------------------|
| `linked_agenda_item_id` | Linking accuracy | Embedding similarity; link to highest match > 0.8 |
| `first_mention` | First-mention detection | Cosine sim against prior summaries; new if max < 0.85 |
| `resolution_status` | Resolution tracking | Rule-based: mentioned → discussed → resolved |
| `clinical_category` | Category classification | Keyword classifier over summary text |

### 4b. Datasets

| Source | Conversations | Split | Purpose |
|--------|--------------|-------|---------|
| ACI-BENCH | 274 | 207 train / 67 eval | Full clinical encounters |


### 4c. Annotation Pipeline

```bash
# Step 1: Batch annotate with teacher model
python annotation_pipeline.py --mode batch --input data/raw/aci/ --output data/annotated/aci/

# Step 2: Realtime annotate (mirrors deployment — line-by-line with context window)
python annotation_pipeline.py --mode realtime --input data/raw/aci/ --output data/annotated/aci/

# Step 3: Generate synthetic conversations
python annotation_pipeline.py --mode generate --input data/cases.json --output data/synthetic/

# Step 4: ASR noise injection
python annotation_pipeline.py --mode noise --input data/annotated/aci/ --output data/noisy/ --wer 0.15

# Step 5: Quality validation (5% sample)
python annotation_pipeline.py --mode validate --input data/annotated/ --output data/validation/ --sample-rate 0.05

# Step 6: Format for LLM training (strips _eval_only fields)
python annotation_pipeline.py --mode format --input data/annotated/ --output data/training/llm_format_a.jsonl --format-type A

# Step 7: Generate context manager labels (separate pipeline)
python context_manager_labels.py --input data/annotated/ --output data/training/eviction_policy.jsonl
```

### 4d. AMI Corpus Conversion

| AMI Annotation | → Agenda Type | resolution_status (eval-only) |
|---------------|--------------|-------------------------------|
| Topic segment | `agenda_item` | "discussed" |
| Decision | `agenda_item` | "resolved" |
| Action item | `follow_up` | "mentioned" |
| Extractive summary item | `detail` | "discussed" |
| Unanswered Elicit dialogue act | `question_unanswered` | "unresolved" |

Speaker normalization: Project Manager → provider, all others → patient.

```bash
python ami_conversion.py --source ami_nxt --input /path/to/ami --output data/ami/
```

---

## Component 5: Evaluation Framework

Evaluation compares **system output** (model + tracker combined) against human annotations with `_eval_only` fields.

**Model-level metrics:**

| Metric | Implementation |
|--------|---------------|
| ROUGE-1/2/L | `rouge-score` package |
| BERTScore (F1) | `bert-score`, backbone: `microsoft/deberta-xlarge-mnli` |
| Detection P/R/F1 | 6-way type classification + irrelevant |
| Agenda completeness | Fraction of ground-truth agenda items found (BERTScore match > 0.85) |
| Temporal F1 | Item surfaced within {30s, 60s, 120s} of first mention |

**System-level metrics:**

| Metric | Implementation |
|--------|---------------|
| Linking accuracy | Compare system link vs. `_eval_only.linked_agenda_item_id` |
| First-mention rate | Compare system flag vs. `_eval_only.first_mention` |
| Resolution tracking | Compare system status vs. `_eval_only.resolution_status` |

**Hardware metrics (lightweight, Paper 1 "Practical Considerations"):**

| Measurement | Tool |
|-------------|------|
| Per-turn inference latency (ms) | `time.perf_counter()` around LLM call |
| Peak GPU/unified memory (MB) | `tegrastats --interval 100` |
| Tokens/sec at each budget | llama.cpp `--verbose` |
| KV-cache memory per budget | `2 × L × H × D × seq_len × dtype_bytes` |

Appears in paper as dual-axis Pareto plot and ~0.5-page subsection.

---

## Component 6: Ablation Study (at B=1024)

| Condition | Eviction | Compression | State Tracking |
|-----------|----------|-------------|----------------|
| Full system | Learned MLP | Yes | Hierarchical |
| Policy only | Learned MLP | No | Flat |
| No policy | FIFO | No | Flat |
| Compression only | FIFO | Yes | Flat |
| Hierarchical only | FIFO | No | Hierarchical |
| Policy + compression | Learned MLP | Yes | Flat |

Budget sweep for main comparison: B ∈ {256, 512, 1024, 2048, 4096}

---

## Week-by-Week Summary

### Week 1: Environment + Dataset + Baselines

- Flash Jetson, install all dependencies, download 3 models (GGUF Q4_K_M).
- Verify LLM inference + `tegrastats` logging.
- Obtain datasets (ACI-BENCH). Begin human annotation with full schema (including `_eval_only` fields).
- Set up teacher-model annotation pipeline (batch + realtime modes). Start ACI-BENCH.
- Implement streaming simulation harness: feed utterances sequentially, LLM produces 3-field output, buffer accumulates pipe-delimited summaries. Add `time.perf_counter()` from Day 1.
- Implement all 5 baseline eviction strategies.
- Run baselines at B = {512, 1024, 2048, 4096}. Compute ROUGE, BERTScore, agenda completeness, temporal F1. Run oracle (unlimited budget) as upper bound.
- Hardware baseline (~2 hours): profile each model × budget with `tegrastats`. Record peak memory, tok/s, time-to-first-token.

**Deliverables:** Working environment, 10–15 annotated dialogues, baseline results table, hardware profiling table, simulation harness.

---

### Week 2: Eviction Policy v1 + State Tracker + Annotation Push

- Generate context manager training labels using hindsight supervision (BERTScore method). Run `context_manager_labels.py` on annotated data.
- Validate labels against leave-one-out perturbation on 5 dialogues (target r > 0.6).
- Build training pipeline: conversation × timestep × summary → (396-d features, binary label).
- Implement MLP scorer. Train v1 (~5–10 min). Evaluate: Spearman's ρ, binary F1.
- Integrate eviction policy into harness. Preliminary comparison vs. baselines at B=1024.
- Implement system-level state tracker: linking (embedding similarity > 0.8), first-mention (cosine sim > 0.85 = repeat), resolution tracking (rule-based state machine).
- Convert detailed annotations → LLM training format (Step 6: strip `_eval_only`, output pipe-delimited context + 3-field target).
- Continue teacher annotation (ACI-BENCH, PriMock57). Begin AMI conversion.
- Continue Penn Medicine human annotation. Compute Cohen's kappa on completed dialogues.

**Deliverables:** Eviction policy v1, state tracker code, LLM training data + context manager labels, preliminary policy vs. baseline results, 25–30 annotated dialogues, IAA scores.

---

### Week 3: Compression + Integration + Policy Iteration

- Implement compression module (LLM prompt targeting ≤20 tokens). Test on 20 summaries. Verify BERTScore > 0.80 against originals.
- Implement extended eviction: evict lowest, optionally compress mid-range (ablation condition).
- Full integration test on 5 dialogues: utterance → LLM (3 fields) → state tracker (link, resolve, first-mention) → context manager (score, evict) → pipe-delimited buffer → next utterance.
- Retrain eviction policy on expanded dataset (30–40 dialogues). Try variants: add LLM attention features as input, try listwise ranking loss instead of BCE.
- Compare v1 vs. v2 on validation set. Pick best.
- Begin synthetic data generation (Stream A: case-report-conditioned).

**Deliverables:** Compression module, full integrated pipeline, eviction policy v2, synthetic generation started.

---

### Weeks 4–7: Summary

| Week | Focus | Key Output |
|------|-------|------------|
| 4 | Policy vs. 5 baselines (all budgets), ablation study (6 conditions at B=1024), dataset freeze at 50–60 dialogues | Results tables, Pareto plots, frozen dataset |
| 5 | Cross-model (1B, 7B), cross-domain (AMI), all figures + tables including dual-axis Pareto and latency plots | All paper figures/tables finalized |
| 6 | Write paper: intro (hardware-grounded), related work, method, experiments + "Practical Considerations," discussion | Draft v1 |
| 7 | Advisor review, revision, formatting, abstract (~Apr 29), submit (~May 3) | Submitted paper |

---

## File Structure

```
project/
├── data/
│   ├── raw/                        # Original transcripts
│   │   ├── penn_medicine/          # 16 real conversations
│   │   ├── aci_bench/              # ACI-BENCH download
│   │   ├── primock57/              # PriMock57 download
│   │   ├── mts_dialog/             # MTS-Dialog download
│   │   └── ami/                    # AMI corpus (NXT XML)
│   ├── annotated/                  # Full annotations (3 model fields + _eval_only)
│   │   ├── human/                  # Human-annotated (Penn Medicine)
│   │   ├── teacher/                # Teacher-model annotated (ACI, PriMock, synthetic)
│   │   └── converted/             # AMI converted
│   ├── training/
│   │   ├── llm_format_a.jsonl      # LLM training: pipe-delimited context → 3-field output
│   │   ├── llm_format_b.jsonl      # LLM training: two-stage (detect then summarize)
│   │   └── eviction_policy.jsonl   # Context manager: 396-d features + binary label
│   ├── synthetic/
│   │   ├── stream_a_case_reports/
│   │   ├── stream_b_note_conditioned/
│   │   └── stream_c_asr_degraded/
│   ├── noisy/                      # ASR-degraded variants
│   ├── validation/                 # Quality validation reports
│   └── splits/                     # Train/val/test splits
├── models/
│   ├── gguf/                       # Downloaded GGUF model files
│   └── policy/                     # Eviction policy MLP checkpoints
├── src/
│   ├── harness.py                  # Streaming simulation (utterance → LLM → tracker → manager → loop)
│   ├── baselines.py                # 5 baseline eviction strategies
│   ├── policy.py                   # MLP eviction policy network + training loop
│   ├── compression.py              # Compression module (LLM prompt, BERTScore check)
│   ├── state_tracker.py            # System tracker (linking, first-mention, resolution, nesting)
│   ├── metrics.py                  # Model-level + system-level eval metrics
│   ├── hardware.py                 # tegrastats parsing, KV-cache calculator
│   ├── annotation_prompts.py       # Teacher-model prompts (batch + realtime)
│   ├── annotation_pipeline.py      # Pipeline runner (batch, realtime, generate, noise, validate, format)
│   ├── context_manager_labels.py   # Hindsight supervision label generator
│   ├── ami_conversion.py           # AMI NXT XML → agenda schema converter
│   └── synthetic_gen.py            # Synthetic conversation generation
├── experiments/
│   ├── configs/                    # Experiment configs (model, budget, strategy)
│   ├── logs/                       # Raw logs + tegrastats dumps
│   └── results/                    # Processed results (CSV)
├── paper/
│   ├── figures/
│   └── tables/
└── README.md
```

---

## Dependencies

```bash
# System
JetPack 6.x (Ubuntu 22.04, CUDA 12.x)

# LLM inference
llama.cpp                        # Build with: cmake -DGGML_CUDA=ON

# Python
pip install torch                # Jetson wheel
pip install sentence-transformers # all-MiniLM-L6-v2 for embeddings
pip install scispacy
pip install en_core_sci_sm       # python -m spacy download en_core_sci_sm
pip install rouge-score
pip install bert-score
pip install lxml                 # AMI NXT XML parsing
pip install wandb                # Experiment tracking (optional)

# Monitoring
sudo apt install python3-jtop
# tegrastats pre-installed with JetPack
```
