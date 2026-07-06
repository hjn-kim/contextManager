"""
Validate labeled training jsonl files produced by context_manager_labels.py.

Checks per file:
  1. every row has all required keys (features, label, conversation_id, ...)
  2. features length == PolicyConfig.feature_dim (396)
  3. conversation_id is present and non-empty
  4. label distribution (positive/negative %) and no unexpected label values
  5. features contain no NaN/Inf, summary is non-empty, no duplicate rows,
     no malformed JSON lines
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

EXPECTED_FEATURE_DIM = 396

REQUIRED_KEYS = [
    "conversation_id",
    "current_utterance_id",
    "summary_utterance_id",
    "timestep",
    "summary",
    "type",
    "max_future_similarity",
    "features",
    "label",
]


def check_file(path: Path) -> bool:
    total = 0
    missing_key_counts = Counter()
    feature_len_counts = Counter()
    label_counts = Counter()
    non_finite_feature_rows = 0
    empty_summary_rows = 0
    missing_conv_id_rows = 0
    duplicate_rows = 0
    bad_json_lines = []
    decode_error_lines = []
    seen_keys = set()

    # Read as bytes and decode per line so one bad/truncated line (e.g. a
    # partial write, or a file still syncing from cloud storage) doesn't
    # crash the whole scan.
    with path.open("rb") as f:
        for lineno, raw_line in enumerate(f, start=1):
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                decode_error_lines.append((lineno, str(exc)))
                continue

            if not line:
                continue
            total += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                bad_json_lines.append((lineno, str(exc)))
                continue

            for key in REQUIRED_KEYS:
                if key not in row:
                    missing_key_counts[key] += 1

            if not row.get("conversation_id"):
                missing_conv_id_rows += 1

            features = row.get("features")
            if features is None:
                feature_len_counts["MISSING"] += 1
            else:
                feature_len_counts[len(features)] += 1
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in features):
                    non_finite_feature_rows += 1

            label_counts[row.get("label")] += 1

            if not (row.get("summary") or "").strip():
                empty_summary_rows += 1

            dedup_key = (
                row.get("conversation_id"),
                row.get("current_utterance_id"),
                row.get("summary_utterance_id"),
                row.get("timestep"),
                row.get("summary"),
            )
            if dedup_key in seen_keys:
                duplicate_rows += 1
            else:
                seen_keys.add(dedup_key)

    ok = True

    print(f"File: {path}")
    print(f"Total rows: {total}")
    print()

    print("[1] Required keys present")
    for key in REQUIRED_KEYS:
        missing = missing_key_counts.get(key, 0)
        status = "OK" if missing == 0 else "FAIL"
        ok = ok and missing == 0
        print(f"  {key:<24} missing in {missing} rows [{status}]")
    print()

    print(f"[2] features length (expected {EXPECTED_FEATURE_DIM})")
    for length, count in sorted(feature_len_counts.items(), key=lambda x: str(x[0])):
        is_ok = (length == EXPECTED_FEATURE_DIM)
        ok = ok and is_ok
        print(f"  length={length!s:<10} rows={count} [{'OK' if is_ok else 'FAIL'}]")
    print()

    print("[3] conversation_id presence")
    is_ok = missing_conv_id_rows == 0
    ok = ok and is_ok
    print(f"  rows missing/empty conversation_id: {missing_conv_id_rows} [{'OK' if is_ok else 'FAIL'}]")
    print()

    print("[4] label distribution")
    n_pos = label_counts.get(1, 0)
    n_neg = label_counts.get(0, 0)
    other_labels = {k: v for k, v in label_counts.items() if k not in (0, 1)}
    denom = max(n_pos + n_neg, 1)
    print(f"  positive (1): {n_pos} ({100 * n_pos / denom:.2f}%)")
    print(f"  negative (0): {n_neg} ({100 * n_neg / denom:.2f}%)")
    if other_labels:
        ok = False
        print(f"  unexpected label values: {other_labels} [FAIL]")
    else:
        print("  no unexpected label values [OK]")
    print()

    print("[5] data quality")
    is_ok = non_finite_feature_rows == 0
    ok = ok and is_ok
    print(f"  rows with NaN/Inf in features: {non_finite_feature_rows} [{'OK' if is_ok else 'FAIL'}]")

    is_ok = empty_summary_rows == 0
    ok = ok and is_ok
    print(f"  rows with empty summary text: {empty_summary_rows} [{'OK' if is_ok else 'FAIL'}]")

    is_ok = duplicate_rows == 0
    ok = ok and is_ok
    print(f"  duplicate rows (same conv/current_utt/summary_utt/timestep/summary): {duplicate_rows} [{'OK' if is_ok else 'FAIL'}]")

    is_ok = len(bad_json_lines) == 0
    ok = ok and is_ok
    print(f"  malformed JSON lines: {len(bad_json_lines)} [{'OK' if is_ok else 'FAIL'}]")
    for lineno, err in bad_json_lines[:5]:
        print(f"    line {lineno}: {err}")

    is_ok = len(decode_error_lines) == 0
    ok = ok and is_ok
    print(f"  lines with UTF-8 decode errors: {len(decode_error_lines)} [{'OK' if is_ok else 'FAIL'}]")
    for lineno, err in decode_error_lines[:5]:
        print(f"    line {lineno}: {err}")
    print()

    print("=" * 60)
    print("RESULT:", "PASS - file looks normal" if ok else "FAIL - issues found above")
    print("=" * 60)
    print()

    return ok


def main():
    parser = argparse.ArgumentParser(description="Validate labeled training jsonl output.")
    parser.add_argument("--input", required=True, nargs="+", help="Path(s) to jsonl file(s) to check.")
    args = parser.parse_args()

    overall_ok = True

    for input_path in args.input:
        path = Path(input_path)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            overall_ok = False
            continue
        overall_ok = check_file(path) and overall_ok

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
