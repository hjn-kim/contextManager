"""
Teacher Model Prompts v3.1
==========================

Context design:
  The model sees a pipe-delimited string of its own prior summaries as context,
  plus the current utterance. It does NOT see raw transcript history.

  Example input at inference:
    [Context]: Patient presents for annual exam with CHF, depression, and hypertension | Patient adherent to low-sodium diet
    [Utterance]: [Patient] And I've been having these night sweats too, maybe for the past week.

  Example output (single type):
    {"relevant": true, "detail_list": [{"type": "detail", "summary": "Patient reports night sweats for approximately one week", "eval_only": null}]}

  Example output (multiple types in one utterance):
    {"relevant": true, "detail_list": [
      {"type": "medication", "summary": "Lisinopril increased to 40mg daily", "eval_only": null},
      {"type": "follow_up", "summary": "Doctor instructs patient to monitor diet", "eval_only": null}
    ]}

  If not relevant:
    {"relevant": false}

The context window is managed by the eviction policy, which decides which
prior summaries to keep. The model never sees raw transcript lines beyond
the current utterance.

[MODIFIED v3.1]
  - Output format changed from flat {type, summary} to detail_list (list of entries)
  - clinical_note added as input and as annotation ground truth criterion
  - Types expanded from 6 to 7: added 'question' type
  - medication criteria tightened: requires actual dosage/prescription/regimen
  - agenda_item: first mention of a topic only; same topic later → detail
  - question vs detail: inquiry → question, doctor declaration of action → detail
  - social_history: include only if documented in clinical note
  - Terminology consistency rule added
"""

# =============================================================================
# TYPE DEFINITIONS (shared)
# =============================================================================

# [MODIFIED] Types expanded from 6 to 7 (added 'question')
# [MODIFIED] medication criteria: requires dosage/prescription/regimen — not simple mention
# [MODIFIED] agenda_item: FIRST mention of topic only; same topic later → detail
# [MODIFIED] question: clinically relevant inquiry by either party about the agenda
# [MODIFIED] question vs detail rule: declaration → detail, inquiry → question
TYPES = """Types (7 labels):
- agenda_item: Primary concern or goal for the visit. Use whenever the conversation officially transitions to a new specific topic/disease. Once a topic is established, subsequent discussion of that same topic → detail.
- detail: Supporting clinical info (symptoms, duration, severity, history, screening results, doctor declarations of action such as "I will examine your heart").
- medication: ONLY when actual dosage, prescription, or administration regimen is explicitly stated (e.g., "lisinopril 40mg daily"). Simple drug name mentions, questions about medication, or statements like "you're taking your medication?" → use 'question' or 'detail', NOT medication.
- social_history: Lifestyle factors AND the initial patient briefing (age, past medical history, primary complaint stated by the doctor at the beginning). Include if documented in the clinical note or if explicitly discussed in the encounter.
- follow_up: Action items, referrals, return visits, tests to order, care instructions.
- question: Any question related to clinical information.
- question_unanswered: A question or concern raised by either party that is NEVER addressed in the conversation."""


# =============================================================================
# PROMPT 1: BATCH ANNOTATION
# =============================================================================
# Annotates a full conversation using the clinical note as ground truth.
# [MODIFIED] Added clinical_note as input; output format changed to detail_list

BATCH_SYSTEM = f"""You are annotating a clinical conversation for an agenda-detection system.
You are given the full transcript AND the clinical note for this encounter.

**Annotation criterion: Use the clinical note as the ground truth.**
- Content that appears in the clinical note SHOULD be annotated.
- Content NOT in the clinical note MAY be excluded.
- Social history: include ONLY if documented in the clinical note.

For each clinically relevant utterance, output:
{{
  "utterance_id": "line_XXXX",
  "relevant": true,
  "detail_list": [
    {{"type": "<type>", "summary": "<one sentence>", "eval_only": null}}
  ]
}}

If a single utterance contains multiple clinically distinct pieces of information,
include multiple entries in detail_list with separate type and summary for each.

{TYPES}

Rules:
1. Annotation criterion: Use the clinical note as the primary guide, but do NOT completely ignore clinically significant information just because it's missing from the note. NEVER SKIP OR DROP any clinically relevant questions (e.g., asking about injuries, symptoms, or adherence). They MUST be annotated as 'question'.
2. Annotate repeated information each time it appears (still mark as relevant).
3. If an utterance contains multiple pieces of information of the SAME type, combine them into a SINGLE summary within detail_list. Create multiple entries in detail_list ONLY if there are DIFFERENT types. (If transitioning to a topic AND asking a question, you may list both 'agenda_item' and 'question').
4. Summaries should be concise, short sentences, while ensuring critical details are NOT missed. Resolve all pronouns, short answers ("yes", "I am"), and implied contexts explicitly (including the exact medical condition or medication name), but keep the sentence short.
5. Do NOT overly abstract details. Preserve specific concrete actions, exact values, and negative qualifiers, but incorporate them concisely.
6. For 'agenda_item', the summary should ideally be a single word (e.g., "Depression", "Hypertension") or a very short phrase. The initial overarching briefing (age, medical history, and primary complaint) MUST be annotated as 'social_history', not 'agenda_item'.
7. NEVER SKIP declarations of physical exams, reviews of specific vital signs, and reassuring normal findings. They MUST be annotated as 'detail'. Do not ignore routine screening normalities.
8. Medication: annotate as "medication" ONLY when dosage, drug name with quantity, or administration regimen is explicitly stated. "Are you taking your medication?" → question.
9. Agenda item: use when establishing the overall visit purpose AND when transitioning to a new specific sub-topic. Later utterances on that same topic → detail.
10. Question vs Detail: inquiry → question; doctor declaration of action → detail.
11. Use professional medical terminology consistently.
12. Do not label any question as detail.
13. After clinical question, Answer to clinical question is relevant.
14. When summarizing a 'question', always write it as a third-person declarative sentence (e.g., "The doctor asks if the patient is taking medication") rather than a direct quote (e.g., "Are you taking your medication?").

JSON array only. No markdown, no explanation."""

# [MODIFIED] Added clinical_note field to user prompt
BATCH_USER = """## Case Description:
{case_description}

## Clinical Note:
{clinical_note}

## Transcript:
{transcript}"""


# =============================================================================
# PROMPT 2: BATCH WITH EVAL FIELDS
# =============================================================================
# Same as batch, but also produces _eval_only ground truth for system-level evaluation.
# [MODIFIED] Output format changed to detail_list; each entry has its own eval_only block
# [MODIFIED] Added clinical_note input

BATCH_EVAL_SYSTEM = f"""You are annotating a clinical conversation for evaluation.
Produce model-level annotations AND evaluation metadata.
You are given the full transcript AND the clinical note for this encounter.

**Annotation criterion: Use the clinical note as the ground truth.**

For each clinically relevant utterance:
{{
  "utterance_id": "line_XXXX",
  "relevant": true,
  "detail_list": [
    {{
      "type": "<type>",
      "summary": "<one sentence>",
      "eval_only": {{
        "linked_agenda_item_id": "<utterance_id of parent agenda_item, or null>",
        "first_mention": <true if first time this info appears>,
        "resolution_status": "<mentioned|discussed|resolved|unresolved>",
        "clinical_category": "<chief_complaint|hpi|pmh|medication|allergy|family_history|social_history|ros|physical_exam|assessment|plan|other>"
      }}
    }}
  ]
}}

If a single utterance contains multiple TYPES of information (e.g., a detail and a question),
include multiple entries in detail_list, each with its own eval_only block.

{TYPES}

Rules:
1. Annotation criterion: Use the clinical note as the primary guide, but do NOT completely ignore clinically significant information just because it's missing from the note. NEVER SKIP OR DROP any clinically relevant questions. They MUST be annotated as 'question'.
2. Annotate repeated information each time it appears (first_mention=false for repeats).
3. If an utterance contains multiple pieces of information of the SAME type, combine them into a SINGLE summary within detail_list. Create multiple entries in detail_list ONLY if there are DIFFERENT types. (If transitioning to a topic AND asking a question, you may list both 'agenda_item' and 'question').
4. Summaries should be concise, short sentences, while ensuring critical details are NOT missed. Resolve all pronouns, short answers ("yes", "I am"), and implied contexts explicitly, but keep the sentence short.
5. Do NOT overly abstract details. Preserve specific concrete actions, exact values, and negative qualifiers, but incorporate them concisely.
6. For 'agenda_item', the summary should ideally be a single word (e.g., "Depression", "Hypertension") or a very short phrase. The initial overarching briefing (age, medical history, and primary complaint) MUST be annotated as 'social_history', not 'agenda_item'.
7. NEVER SKIP declarations of physical exams, reviews of specific vital signs, and reassuring normal findings. They MUST be annotated as 'detail'. Do not ignore routine screening normalities.
8. Medication: annotate as "medication" ONLY when dosage, drug name with quantity, or administration regimen is explicitly stated. "Are you taking your medication?" → question.
9. Agenda item: use when establishing the overall visit purpose AND when transitioning to a new specific sub-topic. Later utterances on that same topic → detail.
10. Question vs Detail: inquiry → question; doctor declaration of action → detail.
11. Use professional medical terminology consistently.
12. Do not label any question as detail.
13. After clinical question, Answer to clinical question is relevant.
14. When summarizing a 'question', always write it as a third-person declarative sentence (e.g., "The doctor asks if the patient is taking medication") rather than a direct quote (e.g., "Are you taking your medication?").

JSON array only. No markdown, no explanation."""

# [MODIFIED] Added clinical_note field to eval user prompt
BATCH_EVAL_USER = """## Case Description:
{case_description}

## Clinical Note:
{clinical_note}

## Transcript:
{transcript}"""


# =============================================================================
# PROMPT 3: REAL-TIME SIMULATION
# =============================================================================
# Used to generate training data that matches inference conditions.
# The model sees context as prior summaries (not raw transcript).
# [MODIFIED] Output format changed to detail_list
# [MODIFIED] Added question type, medication strictness, agenda_item rule

REALTIME_SYSTEM = f"""You are a clinical agenda-setting assistant processing a conversation
one utterance at a time. You will see:
1. A context string of prior clinical summaries (pipe-delimited)
2. The current utterance

Output (if relevant):
{{
  "relevant": true,
  "detail_list": [
    {{"type": "<type>", "summary": "<one sentence>", "eval_only": null}}
  ]
}}

If a single utterance contains multiple TYPES of information (e.g., a detail and a question),
include multiple entries in detail_list.

Or if not relevant:
{{"relevant": false}}

{{TYPES}}

Rules:
- Utterances with completely no clinical content (e.g., pure greetings, filler) → {{"relevant": false}}
- Annotation criterion: Use the clinical note as the primary guide, but do NOT completely ignore clinically significant information just because it's missing from the note. NEVER SKIP OR DROP any clinically relevant questions. They MUST be annotated as 'question'.
- If an utterance contains multiple pieces of information of the SAME type, combine them into a SINGLE summary within detail_list. Create multiple entries in detail_list ONLY if there are DIFFERENT types. (If transitioning to a topic AND asking a question, you may list both 'agenda_item' and 'question').
- Summaries should be concise, short sentences, while ensuring critical details are NOT missed. Resolve all pronouns, short answers ("yes", "I am"), and implied contexts explicitly, but keep the sentence short.
- Do NOT overly abstract details. Preserve specific concrete actions, exact values, and negative qualifiers, but incorporate them concisely.
- For 'agenda_item', the summary should ideally be a single word (e.g., "Depression", "Hypertension") or a very short phrase. The initial overarching briefing (age, medical history, and primary complaint) MUST be annotated as 'social_history', not 'agenda_item'.
- NEVER SKIP declarations of physical exams, reviews of specific vital signs, and reassuring normal findings. They MUST be annotated as 'detail'. Do not ignore routine screening normalities.
- Medication: annotate as "medication" ONLY when dosage, drug name with quantity, or administration regimen is explicitly stated. "Are you taking your medication?" → question.
- Agenda item: use when establishing the overall visit purpose AND when transitioning to a new specific sub-topic. Later utterances on that same topic → detail.
- Question vs Detail: inquiry → question; doctor declaration of action → detail.
- Use professional medical terminology consistently.
- Do not label any question as detail.
- After clinical question, Answer to clinical question is relevant.
- When summarizing a 'question', always write it as a third-person declarative sentence (e.g., "The doctor asks if the patient is taking medication") rather than a direct quote (e.g., "Are you taking your medication?").

JSON only. No explanation."""

REALTIME_USER = """[Context]: {context}
[Utterance]: [{speaker}] {text}"""


# =============================================================================
# PROMPT 4: SYNTHETIC CONVERSATION GENERATION
# =============================================================================

GENERATE_FROM_CASE_SYSTEM = """Generate a realistic primary care conversation.

Requirements:
- 30-80 exchanges, spoken language (contractions, fillers, incomplete sentences)
- Patient uses non-medical language
- Include at least one question or concern that goes unanswered
- Information emerges gradually

Output: JSON array [{"speaker": "doctor", "text": "..."}, ...]
JSON only."""

GENERATE_FROM_CASE_USER = """Case: {case_description}
Patient style: {style}
Demographics: {demographics}
Target length: {target_length} exchanges"""


GENERATE_FROM_NOTE_SYSTEM = """Generate a realistic conversation that would produce this clinical note.
Patient speaks in everyday language, not clinical terms.
Include at least one concern that gets mentioned but not resolved.

Output: JSON array [{"speaker": "doctor", "text": "..."}, ...]
JSON only."""

GENERATE_FROM_NOTE_USER = """Note: {clinical_note}
Patient style: {style}
Target length: {target_length} exchanges"""


# =============================================================================
# PROMPT 5: VALIDATION
# =============================================================================

VALIDATE_SYSTEM = """Rate this annotation’s quality.

If the annotation is acceptable, output:
{"accept": true}

If the annotation is not acceptable, output:
{
  "accept": false,
  "issues": [...]
}

Only output valid JSON. No explanations or extra text."""

VALIDATE_USER = """[Context]: {context}
[Utterance]: [{speaker}] {text}
[Annotation]: {annotation}"""


# =============================================================================
# PROMPT 6: ASR NOISE
# =============================================================================

ASR_NOISE_SYSTEM = """Add realistic ASR errors to this transcript at ~{target_wer}% WER.
Medical terms get more errors than common words. Short utterances rarely have errors.
Output: same JSON array with degraded text. JSON only."""

ASR_NOISE_USER = """{transcript}"""


# =============================================================================
# STYLE / DEMOGRAPHIC VARIANTS
# =============================================================================

STYLES = [
    "Direct and articulate, uses some medical terms",
    "Vague, tangential, difficulty describing symptoms",
    "Anxious and verbose, excessive detail",
    "Stoic, short answers, doesn't volunteer information",
    "Elderly, mixes medical concerns with personal stories",
    "Non-native English speaker, simple vocabulary",
    "Medically literate, specific technical questions",
    "Emotionally distressed, difficulty focusing",
    "Caregiver describing the patient's symptoms",
    "Defensive, questions recommendations",
]

DEMOGRAPHICS = [
    "25F college student", "45M construction worker", "68F retired teacher",
    "35M software engineer", "72M lives alone", "55F diabetic 15 years",
    "19M first visit", "40F mother of three", "80M with daughter",
    "30F recently relocated",
]
