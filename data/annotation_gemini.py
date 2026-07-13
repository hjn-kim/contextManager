"""
Annotation Pipeline v3 — Gemini Version
========================================

Modes:
  batch     — annotate full conversations (teacher sees full transcript)
  eval      — annotate with _eval_only ground truth fields
  realtime  — line-by-line with summary-based context (matches inference)
  generate  — synthetic conversations + annotation
  validate  — quality check
  noise     — ASR error injection
  format    — convert to training JSONL
"""

import json
import os
import re
import csv
import random
import time
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# =============================================================================
# DATA STRUCTURES
# =============================================================================

# [MODIFIED] detail_list 필드 추가 — 1 utterance에 여러 type/summary 지원 (prompts v3.1)
@dataclass
class Annotation:
    utterance_id: str
    relevant: bool
    detail_list: Optional[list] = None   # [{"type": ..., "summary": ..., "eval_only": ...}]
    # 하위 호환 필드 (단일 type/summary)
    type: Optional[str] = None
    summary: Optional[str] = None
    eval_only: Optional[dict] = None


@dataclass
class Conversation:
    id: str
    source: str
    utterances: list       # [{"id", "speaker", "text"}]
    annotations: list      # [Annotation]
    case_description: str = ""
    metadata: dict = field(default_factory=dict)


# =============================================================================
# TEACHER MODEL (Gemini)
# =============================================================================

class Teacher:
    def __init__(self, model: str = "gemini-3.1-flash-lite-preview", temperature: float = 0.0, retries: int = 3):
        self.model = model
        self.temperature = temperature
        self.retries = retries
        from google import genai
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    def _call(self, system: str, user: str) -> str:
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
            ),
        )
        return response.text

    def query(self, system: str, user: str):
        for attempt in range(self.retries):
            try:
                raw = self._call(system, user)
                raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
                raw = re.sub(r'\s*```$', '', raw.strip())
                return json.loads(raw)
            except Exception as e:
                if attempt == self.retries - 1:
                    print(f"  ERROR: {e}")
                    return None
                
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    time.sleep(30)
                else:
                    time.sleep(2 ** attempt)


# =============================================================================
# BATCH ANNOTATOR
# =============================================================================

def annotate_batch(teacher: Teacher, conversations: list[dict],
                   output_dir: str, include_eval: bool = False) -> list[Conversation]:
    from prompts import BATCH_SYSTEM, BATCH_USER, BATCH_EVAL_SYSTEM, BATCH_EVAL_USER

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, conv in enumerate(conversations):
        print(f"[{i+1}/{len(conversations)}] {conv['id']}")

        transcript = "\n".join(
            f"[line_{j:04d}] [{t['speaker'].capitalize()}] {t['text']}"
            for j, t in enumerate(conv["transcript"])
        )
        case = conv.get("case_description", "Not available")
        # [MODIFIED] clinical_note를 프롬프트에 전달 (prompts v3.1)
        clinical_note = conv.get("note", "Not available")
        sys_prompt = BATCH_EVAL_SYSTEM if include_eval else BATCH_SYSTEM
        usr_prompt = (BATCH_EVAL_USER if include_eval else BATCH_USER).format(
            case_description=case, clinical_note=clinical_note, transcript=transcript
        )

        result = teacher.query(sys_prompt, usr_prompt)
        annotations = []
        if result and isinstance(result, list):
            for item in result:
                # [MODIFIED] detail_list 형식으로 파싱 (prompts v3.1)
                annotations.append(Annotation(
                    utterance_id=item.get("utterance_id", "unknown"),
                    relevant=True,
                    detail_list=item.get("detail_list"),
                ))

        utterances = [
            {"id": f"line_{j:04d}", "speaker": t["speaker"], "text": t["text"]}
            for j, t in enumerate(conv["transcript"])
        ]

        out = Conversation(
            id=conv["id"], source=conv.get("source", "unknown"),
            utterances=utterances, annotations=annotations,
            case_description=case,
            metadata={"annotator": teacher.model, "include_eval": include_eval}
        )
        results.append(out)

        with open(os.path.join(output_dir, f"{conv['id']}.json"), "w") as f:
            json.dump(asdict(out, dict_factory=lambda x: {k: v for k, v in x if v is not None}), f, indent=2)

    return results


# =============================================================================
# REAL-TIME ANNOTATOR
# =============================================================================

def annotate_realtime(teacher: Teacher, conv: dict,
                      max_context_items: int = 5) -> list[Annotation]:
    """
    Annotate line-by-line with context = pipe-delimited prior summaries.
    This matches inference conditions exactly.
    """
    from prompts import REALTIME_SYSTEM, REALTIME_USER

    annotations = []
    summary_buffer = []  # The model's own prior summaries

    for j, turn in enumerate(conv["transcript"]):
        uid = f"line_{j:04d}"

        # Build context: last N summaries, pipe-delimited
        recent = summary_buffer[-max_context_items:]
        context = " | ".join(recent) if recent else "Start of visit"

        result = teacher.query(
            REALTIME_SYSTEM,
            REALTIME_USER.format(
                context=context,
                speaker=turn["speaker"].capitalize(),
                text=turn["text"]
            )
        )

        if result and result.get("relevant"):
            # [MODIFIED] detail_list 형식으로 파싱 (prompts v3.1)
            detail_list = result.get("detail_list")
            ann = Annotation(
                utterance_id=uid, relevant=True,
                detail_list=detail_list,
            )
            # detail_list 내 모든 summary를 context buffer에 추가
            if detail_list:
                for entry in detail_list:
                    s = entry.get("summary")
                    if s:
                        summary_buffer.append(s)
        else:
            ann = Annotation(utterance_id=uid, relevant=False)

        annotations.append(ann)

    return annotations


# =============================================================================
# SYNTHETIC GENERATOR
# =============================================================================

def generate_synthetic(teacher: Teacher, cases: list[dict],
                       output_dir: str, variants: int = 3) -> list[Conversation]:
    from prompts import (GENERATE_FROM_CASE_SYSTEM, GENERATE_FROM_CASE_USER,
                         GENERATE_FROM_NOTE_SYSTEM, GENERATE_FROM_NOTE_USER,
                         STYLES, DEMOGRAPHICS, BATCH_SYSTEM, BATCH_USER)

    os.makedirs(output_dir, exist_ok=True)
    results = []
    counter = 0

    for case in cases:
        for v in range(variants):
            style = random.choice(STYLES)
            demo = random.choice(DEMOGRAPHICS)
            length = random.randint(35, 70)

            if "description" in case:
                turns = teacher.query(
                    GENERATE_FROM_CASE_SYSTEM,
                    GENERATE_FROM_CASE_USER.format(
                        case_description=case["description"],
                        style=style, demographics=demo, target_length=length
                    )
                )
                desc = case["description"]
            else:
                turns = teacher.query(
                    GENERATE_FROM_NOTE_SYSTEM,
                    GENERATE_FROM_NOTE_USER.format(
                        clinical_note=case["clinical_note"],
                        style=style, target_length=length
                    )
                )
                desc = "From clinical note"

            if not turns or not isinstance(turns, list):
                continue

            cid = f"syn_{counter:05d}"
            counter += 1

            # Annotate
            transcript = "\n".join(
                f"[line_{j:04d}] [{t['speaker'].capitalize()}] {t['text']}"
                for j, t in enumerate(turns)
            )
            ann_result = teacher.query(
                BATCH_SYSTEM, BATCH_USER.format(case_description=desc, transcript=transcript)
            )

            annotations = []
            if ann_result and isinstance(ann_result, list):
                for item in ann_result:
                    annotations.append(Annotation(
                        utterance_id=item.get("utterance_id", "unknown"),
                        relevant=True, type=item.get("type"), summary=item.get("summary"),
                    ))

            utterances = [{"id": f"line_{j:04d}", "speaker": t["speaker"], "text": t["text"]}
                         for j, t in enumerate(turns)]

            out = Conversation(id=cid, source="synthetic", utterances=utterances,
                              annotations=annotations, case_description=desc,
                              metadata={"style": style, "demographics": demo})
            results.append(out)
            with open(os.path.join(output_dir, f"{cid}.json"), "w") as f:
                json.dump(asdict(out, dict_factory=lambda x: {k: v for k, v in x if v is not None}), f, indent=2)
            print(f"  {cid}: {len(turns)} turns, {len(annotations)} annotations")

    return results


# =============================================================================
# ASR NOISE
# =============================================================================

def inject_noise(teacher: Teacher, transcript: list[dict], wer: float = 0.15) -> list[dict]:
    from prompts import ASR_NOISE_SYSTEM, ASR_NOISE_USER
    result = teacher.query(
        ASR_NOISE_SYSTEM.format(target_wer=int(wer * 100)),
        ASR_NOISE_USER.format(transcript=json.dumps(transcript, indent=2))
    )
    return result if result and isinstance(result, list) else transcript


# =============================================================================
# VALIDATOR
# =============================================================================

def validate_sample(teacher: Teacher, conversations: list[Conversation],
                    sample_rate: float = 0.1) -> dict:
    from prompts import VALIDATE_SYSTEM, VALIDATE_USER

    candidates = []
    for conv in conversations:
        utt_map = {u["id"]: u for u in conv.utterances}
        ann_map = {a.utterance_id: a for a in conv.annotations}
        summaries_so_far = []
        for utt in conv.utterances:
            ann = ann_map.get(utt["id"])
            if ann and ann.relevant and ann.detail_list:
                context = " | ".join(summaries_so_far[-5:]) if summaries_so_far else "Start of visit"
                candidates.append((conv.id, utt, ann, context))
                for entry in ann.detail_list:
                    if entry.get("summary"):
                        summaries_so_far.append(entry["summary"])
    n = max(1, int(len(candidates) * sample_rate))
    sample = random.sample(candidates, min(n, len(candidates)))
    validations = []
    for conv_id, utt, ann, context in sample:
        result = teacher.query(
            VALIDATE_SYSTEM,
            VALIDATE_USER.format(
                context=context, speaker=utt["speaker"].capitalize(),
                text=utt["text"],
                annotation=json.dumps(ann.detail_list)
            )
        )
        if isinstance(result, dict):
            result["conversation_id"] = conv_id
            result["utterance_id"] = utt["id"]
            result["speaker"] = utt["speaker"]
            result["text"] = utt["text"]
            result["annotations"] = ann.detail_list
            validations.append(result)

    accepted = sum(1 for v in validations if v.get("accept") is True)
    rejected = sum(1 for v in validations if v.get("accept") is False)

    accepted_validations = [
    v for v in validations
    if v.get("accept") is False
    ]
    
    return {
        "total": len(validations),
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": accepted / max(len(validations), 1),
        "validations": accepted_validations,
    }


# =============================================================================
# TRAINING DATA FORMATTER
# =============================================================================

def format_training_data(conversations: list[Conversation], output_path: str,
                         format_type: str = "A", max_context: int = 5):
    """
    Convert annotated conversations to training JSONL.

    Context construction matches inference:
      - Replay the conversation sequentially
      - Accumulate ground-truth summaries as context
      - Each example: (context + utterance) → model output

    This means the model trains on the SAME context format it will see
    at inference (its own prior summaries), not raw transcript.

    Format A: single-step → {"relevant", "type", "summary"}
    Format B: two-stage → detect ("yes"/"no") then summarize
    """
    examples = []

    for conv in conversations:
        ann_map = {a.utterance_id: a for a in conv.annotations}
        summary_buffer = []  # Simulates the model's own output history

        for utt in conv.utterances:
            ann = ann_map.get(utt["id"])

            # Build context exactly as inference would
            recent = summary_buffer[-max_context:]
            context = " | ".join(recent) if recent else "Start of visit"

            if format_type == "A":
                input_text = f"[Context]: {context}\n[Utterance]: [{utt['speaker'].capitalize()}] {utt['text']}"

                if ann and ann.relevant and ann.detail_list:
                    output_text = json.dumps({
                        "relevant": True, "detail_list": ann.detail_list
                    })
                else:
                    output_text = '{"relevant": false}'

                examples.append({"input": input_text, "output": output_text})

            elif format_type == "B":
                input_base = f"[Context]: {context}\n[Utterance]: [{utt['speaker'].capitalize()}] {utt['text']}"

                # Stage 1: detection
                examples.append({
                    "stage": "detect",
                    "input": f"Is this clinically relevant?\n{input_base}\nAnswer:",
                    "output": "yes" if (ann and ann.relevant) else "no"
                })

                # Stage 2: summarization (relevant only)
                if ann and ann.relevant and ann.detail_list:
                    for entry in ann.detail_list:
                        examples.append({
                            "stage": "summarize",
                            "input": f"Summarize.\n{input_base}\nType: {entry['type']}\nSummary:",
                            "output": entry["summary"]
                        })

            # Update context buffer with ground-truth summary
            # At inference this would be the model's own output
            if ann and ann.relevant and ann.detail_list:
                for entry in ann.detail_list:
                    if entry.get("summary"):
                        summary_buffer.append(entry["summary"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # Stats
    relevant = sum(1 for e in examples if format_type == "A" and '"relevant": true' in e.get("output", ""))
    irrelevant = sum(1 for e in examples if format_type == "A" and '"relevant": false' in e.get("output", ""))
    print(f"Wrote {len(examples)} examples to {output_path}")
    if format_type == "A":
        print(f"  Relevant: {relevant}, Irrelevant: {irrelevant}, Ratio: {relevant/max(relevant+irrelevant,1):.1%}")


# =============================================================================
# HELPERS
# =============================================================================

def _parse_dialogue(dialogue_text: str) -> list[dict]:
    """
    ACI-BENCH dialogue column → list of {"speaker", "text"} dicts.

    Input:  "[doctor] hi , how are you ?\n[patient] i'm fine ..."
    Output: [{"speaker": "doctor", "text": "hi , how are you ?"},
             {"speaker": "patient", "text": "i'm fine ..."}]
    """
    utterances = []
    pattern = r'\[(doctor|patient)\]\s*'
    parts = re.split(pattern, dialogue_text)
    # parts = ["", "doctor", "text...", "patient", "text...", ...]
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i].strip()
        text = re.sub(r'\s*\n\s*', ' ', parts[i + 1].strip())
        if text:
            utterances.append({"speaker": speaker, "text": text})
        i += 2
    return utterances


def load_aci_bench_csv(csv_path: str, metadata_path: str = None) -> list[dict]:
    """
    ACI-BENCH CSV → list of conversation dicts ready for annotate_batch/realtime.

    CSV columns:     dataset, encounter_id, dialogue, note
    Metadata columns: dataset, encounter_id, id, ..., cc, 2nd_complaints

    Returns: [{"id": "D2N001", "source": "aci-bench",
               "transcript": [{"speaker", "text"}, ...],
               "case_description": "..."}]
    """
    meta = {}
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                eid = (row.get("encounter_id") or row.get("id", "")).strip()
                if eid:
                    meta[eid] = row

    conversations = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = (row.get("encounter_id") or row.get("id", "unknown")).strip()
            dialogue = row.get("dialogue", "")
            note = row.get("note", "")
            dataset = row.get("dataset", "aci-bench").strip()

            transcript = _parse_dialogue(dialogue)
            if not transcript:
                print(f"  WARNING: no utterances parsed for {eid}, skipping")
                continue

            # Build case description from metadata
            m = meta.get(eid, {})
            cc = m.get("cc", "").strip()
            complaints = m.get("2nd_complaints", "").strip()
            case_desc = cc
            if complaints:
                case_desc += f"; {complaints}"
            if not case_desc:
                # Fallback: extract from note heading
                for line in note.strip().split("\n"):
                    line = line.strip()
                    if line and line.upper() != "CHIEF COMPLAINT":
                        case_desc = line.rstrip(".")
                        break

            # [MODIFIED] note를 conv dict에 포함 — prompts에서 clinical_note로 사용
            conversations.append({
                "id": eid,
                "source": dataset,
                "transcript": transcript,
                "case_description": case_desc or "Not available",
                "note": note,
            })

    print(f"Loaded {len(conversations)} conversations from {csv_path}")
    return conversations

def _parse_mts_dialogue(dialogue_text: str) -> list[dict]:
    utterances = []
    pattern = r'(Doctor|Patient):\s*'
    parts = re.split(pattern, dialogue_text)
    i = 1
    while i < len(parts) - 1:
        speaker_raw = parts[i].strip()
        text = re.sub(r'\s*\n\s*', ' ', parts[i + 1].strip())
        if text:
            utterances.append({
                "speaker": "doctor" if speaker_raw == "Doctor" else "patient",
                "text": text
            })
        i += 2
    return utterances


def load_mts_dialog_csv(csv_path: str) -> list[dict]:
    conversations = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = f"mts_{row['ID']}"
            transcript = _parse_mts_dialogue(row.get("dialogue", ""))
            if not transcript:
                print(f"  WARNING: no utterances parsed for {eid}, skipping")
                continue
            conversations.append({
                "id": eid,
                "source": "mts-dialog",
                "transcript": transcript,
                "case_description": row.get("section_header", "Not available"),
                "note": row.get("section_text", "Not available"),
            })
    print(f"Loaded {len(conversations)} conversations from {csv_path}")
    return conversations


def _load_conversations(path, metadata=None, limit=None, ids=None):
    if path.endswith(".csv"):
        if "MTS" in os.path.basename(path):
            convs = load_mts_dialog_csv(path)
        else:
            convs = load_aci_bench_csv(path, metadata)
    else:
        convs = [json.load(open(f)) for f in sorted(Path(path).glob("*.json"))]
    if ids:
        convs = [c for c in convs if c["id"] in ids]
    elif limit:
        convs = convs[:limit]
    return convs


def _load_annotated(path):
    results = []
    for f in sorted(Path(path).glob("*.json")):
        d = json.load(open(f, encoding="utf-8"))
        results.append(Conversation(
            id=d["id"], source=d.get("source", ""),
            utterances=d["utterances"],
            annotations=[Annotation(**a) for a in d["annotations"]],
            case_description=d.get("case_description", ""),
            metadata=d.get("metadata", {})
        ))
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["batch", "eval", "realtime", "generate", "validate", "noise", "format"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", help="ACI-BENCH metadata CSV path (for csv input)")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--context-size", type=int, default=5)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--sample-rate", type=float, default=0.05)
    parser.add_argument("--format-type", choices=["A", "B"], default="A")
    parser.add_argument("--wer", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max conversations to process (for testing)")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="Specific encounter IDs to process (e.g. D2N007 D2N026)")
    args = parser.parse_args()

    teacher = Teacher(model=args.model)

    if args.mode == "batch":
        annotate_batch(teacher, _load_conversations(args.input, args.metadata, args.limit, args.ids), args.output)
    elif args.mode == "eval":
        annotate_batch(teacher, _load_conversations(args.input, args.metadata, args.limit, args.ids), args.output, include_eval=True)
    elif args.mode == "realtime":
        os.makedirs(args.output, exist_ok=True)
        for conv in _load_conversations(args.input, args.metadata, args.limit, args.ids):
            anns = annotate_realtime(teacher, conv, args.context_size)
            out = Conversation(
                id=conv["id"], source=conv.get("source", ""),
                utterances=[{"id": f"line_{j:04d}", "speaker": t["speaker"], "text": t["text"]}
                           for j, t in enumerate(conv["transcript"])],
                annotations=anns
            )
            with open(os.path.join(args.output, f"{conv['id']}.json"), "w") as f:
                json.dump(asdict(out, dict_factory=lambda x: {k: v for k, v in x if v is not None}), f, indent=2)
    elif args.mode == "generate":
        generate_synthetic(teacher, json.load(open(args.input)), args.output, args.variants)
    elif args.mode == "validate":
        report = validate_sample(teacher, _load_annotated(args.input), args.sample_rate)
        os.makedirs(args.output, exist_ok=True)
        report = {"source": os.path.basename(os.path.dirname(args.input)), **report}
        report_path = os.path.join(args.output, "report.json")
        existing = []
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.append(report)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        print(f"Acceptance: {report['acceptance_rate']:.1%}")
    elif args.mode == "noise":
        os.makedirs(args.output, exist_ok=True)
        for conv in _load_conversations(args.input, args.metadata, args.limit, args.ids):
            conv["transcript"] = inject_noise(teacher, conv["transcript"], args.wer)
            conv["id"] = conv["id"] + "_asr"
            with open(os.path.join(args.output, f"{conv['id']}.json"), "w") as f:
                json.dump(conv, f, indent=2)
    elif args.mode == "format":
        format_training_data(_load_annotated(args.input), args.output, args.format_type, args.context_size)

if __name__ == "__main__":
    main()
