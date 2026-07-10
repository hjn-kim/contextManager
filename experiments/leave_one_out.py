r"""
leave_one_out.py  (make + run 통합)
===================================

make_leave_one_out.py + run_leave_one_out.py 를 하나로 합친 파일.

두 가지 모드:
  1) 기본(full run): leave-one-conversation-out 로 분할 → EvictionMLP 학습
     (policy.train_policy) → eval_policy_intrinsic.py 로 평가 → CSV/요약 저장
  2) --splits-only: 분할 파일(train.jsonl/valid.jsonl)과 conversation_counts.csv 만
     생성하고 종료 (기존 make_leave_one_out.py 동작)

Full run 예:
  cd c:\Users\mphk0\Documents\context_manager
  $env:PYTHONPATH="src"
  .venv311\Scripts\python.exe experiments\leave_one_out.py `
      --input eviction_policy\eviction_policy_bertscore_train_0p7.jsonl `
      --outdir eviction_policy\leave_one_out_0p7 `
      --model-dir eviction_mlp\leave_one_out_0p7 `
      --leaveout D2N001 `
      --eval-script eval_policy_intrinsic.py `
      --src-dir src

Splits-only 예:
  .venv311\Scripts\python.exe experiments\leave_one_out.py `
      --input eviction_policy\eviction_policy_bertscore_train_0p7.jsonl `
      --outdir eviction_policy\leave_one_out_0p7 `
      --splits-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# 이 파일(experiments/)의 위치 기준 경로 — 실행 cwd 와 무관하게 동작.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_EVAL_SCRIPT = os.path.join(_HERE, "eval_policy_intrinsic.py")
_DEFAULT_SRC_DIR = os.path.join(_HERE, "..", "src")


# -----------------------------------------------------------------------------
# JSONL split utilities
# -----------------------------------------------------------------------------

def safe_dir_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()) or "unknown"


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if "conversation_id" not in row:
                raise ValueError(f"Missing conversation_id at line {line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in input file: {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def count_labels(rows: Sequence[dict]) -> Dict[str, int | float]:
    total = len(rows)
    positive = sum(1 for r in rows if int(r.get("label", 0)) == 1)
    negative = total - positive
    return {
        "rows": total,
        "positive": positive,
        "negative": negative,
        "positive_rate": positive / total if total else 0.0,
    }


def write_conversation_counts(outdir: Path, rows: Sequence[dict]) -> None:
    """대화별 row/label 분포를 conversation_counts.csv 로 저장 (make 기능)."""
    by_conv: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_conv[str(row["conversation_id"])].append(row)

    csv_path = outdir / "conversation_counts.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["conversation_id", "rows", "positive", "negative", "positive_rate"])
        writer.writeheader()
        for cid in sorted(by_conv):
            writer.writerow({"conversation_id": cid, **count_labels(by_conv[cid])})


def create_splits(input_path: Path, outdir: Path, leaveout_ids: Sequence[str] | None,
                  write_counts: bool = False) -> List[dict]:
    rows = read_jsonl(input_path)
    all_convs = sorted({str(r["conversation_id"]) for r in rows})

    if leaveout_ids:
        targets = [str(x) for x in leaveout_ids]
        missing = [x for x in targets if x not in all_convs]
        if missing:
            raise ValueError(
                "Requested leaveout conversation_id(s) not found: "
                + ", ".join(missing)
                + "\nAvailable conversation_id(s): "
                + ", ".join(all_convs)
            )
    else:
        targets = all_convs

    outdir.mkdir(parents=True, exist_ok=True)
    if write_counts:
        write_conversation_counts(outdir, rows)

    manifests: List[dict] = []
    for leaveout in targets:
        split_dir = outdir / safe_dir_name(leaveout)
        train_rows = [r for r in rows if str(r["conversation_id"]) != leaveout]
        valid_rows = [r for r in rows if str(r["conversation_id"]) == leaveout]

        if not train_rows:
            raise ValueError(f"No train rows after excluding leaveout={leaveout}")
        if not valid_rows:
            raise ValueError(f"No valid rows for leaveout={leaveout}")

        train_path = split_dir / "train.jsonl"
        valid_path = split_dir / "valid.jsonl"
        write_jsonl(train_path, train_rows)
        write_jsonl(valid_path, valid_rows)

        manifest = {
            "leaveout": leaveout,
            "split_dir": str(split_dir),
            "input_path": str(input_path),
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "train": count_labels(train_rows),
            "valid": count_labels(valid_rows),
            "train_conversations": sorted({str(r["conversation_id"]) for r in train_rows}),
            "valid_conversations": sorted({str(r["conversation_id"]) for r in valid_rows}),
        }
        with (split_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        manifests.append(manifest)

    with (outdir / "leave_one_out_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifests, f, ensure_ascii=False, indent=2)

    return manifests


# -----------------------------------------------------------------------------
# Evaluation parsing utilities
# -----------------------------------------------------------------------------

def parse_eval_stdout(stdout: str) -> Dict[str, str | float | int]:
    """Parse eval_policy_intrinsic.py text output into a flat metric dict."""
    metrics: Dict[str, str | float | int] = {}

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()

        if key == "tp/fp/tn/fn":
            nums = [x.strip() for x in value.split("/")]
            if len(nums) == 4 and all(x.lstrip("-").isdigit() for x in nums):
                metrics["tp"] = int(nums[0])
                metrics["fp"] = int(nums[1])
                metrics["tn"] = int(nums[2])
                metrics["fn"] = int(nums[3])
            continue

        if key == "model":
            metrics[key] = value
            continue

        try:
            if re.fullmatch(r"[-+]?\d+", value):
                metrics[key] = int(value)
            else:
                metrics[key] = float(value)
        except ValueError:
            metrics[key] = value

    return metrics


def run_eval(eval_script: Path, data_path: Path, model_path: Path, python_exe: str,
             src_dir: Path) -> subprocess.CompletedProcess:
    cmd = [python_exe, str(eval_script), "--data", str(data_path), "--model", str(model_path)]
    # eval 스크립트는 policy/config 를 import 하므로 PYTHONPATH 에 src 를 넣어준다.
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) + (os.pathsep + existing if existing else "")
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def write_results_csv(csv_path: Path, results: List[Dict[str, object]]) -> None:
    preferred = [
        "leaveout", "train_rows", "valid_rows", "valid_positive", "valid_negative",
        "valid_positive_rate", "model_path", "eval_returncode", "rows", "positive",
        "negative", "threshold", "accuracy", "precision", "recall", "f1",
        "tp", "fp", "tn", "fn", "spearman_rho_mean", "spearman_groups",
    ]

    all_keys = []
    for r in results:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)

    fieldnames = [k for k in preferred if k in all_keys] + [k for k in all_keys if k not in preferred]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-one-conversation-out: 분할(+옵션 학습/평가) 통합 러너.")
    parser.add_argument("--input", required=True, help="Input eviction policy JSONL file")
    parser.add_argument("--outdir", required=True, help="Output dir for LOO train/valid files and logs")
    parser.add_argument("--leaveout", nargs="*", default=None,
                        help="Conversation IDs to hold out. If omitted, all are used.")
    parser.add_argument("--splits-only", action="store_true",
                        help="분할 파일과 conversation_counts.csv 만 생성하고 종료 (학습/평가 없음).")
    # full-run 전용 옵션
    parser.add_argument("--model-dir", default=None, help="LOO .pt 저장 폴더 (full run 시 필요)")
    parser.add_argument("--model-prefix", default="eviction_mlp_bertscore_0p7",
                        help="저장 모델 파일명 접두사. 최종: <prefix>_minus_<leaveout>.pt")
    parser.add_argument("--eval-script", default=_DEFAULT_EVAL_SCRIPT)
    parser.add_argument("--src-dir", default=_DEFAULT_SRC_DIR, help="policy.py/config.py 가 있는 폴더")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--internal-val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train-if-model-exists", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    # -------- splits-only 모드 (make 동작) --------
    if args.splits_only:
        manifests = create_splits(input_path, outdir, args.leaveout, write_counts=True)
        print(f"Created {len(manifests)} leave-one-out split(s). (splits-only)")
        for m in manifests:
            print(f"- {m['leaveout']}: train_rows={m['train']['rows']} "
                  f"valid_rows={m['valid']['rows']} "
                  f"valid_pos_rate={m['valid']['positive_rate']:.3f}")
        return

    # -------- full run 모드 (분할 + 학습 + 평가) --------
    if not args.model_dir:
        parser.error("full run 에는 --model-dir 이 필요합니다 (또는 --splits-only 사용).")

    random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except Exception:
        pass

    src_dir = Path(args.src_dir).resolve()
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from policy import train_policy
    except Exception as exc:
        raise RuntimeError(
            "Could not import train_policy from policy.py. Check --src-dir and PYTHONPATH."
        ) from exc

    model_dir = Path(args.model_dir)
    eval_script = Path(args.eval_script)
    model_dir.mkdir(parents=True, exist_ok=True)

    manifests = create_splits(input_path, outdir, args.leaveout, write_counts=True)

    results: List[Dict[str, object]] = []
    for manifest in manifests:
        leaveout = manifest["leaveout"]
        split_dir = Path(manifest["split_dir"])
        train_path = Path(manifest["train_path"])
        valid_path = Path(manifest["valid_path"])
        model_path = model_dir / f"{args.model_prefix}_minus_{safe_dir_name(leaveout)}.pt"

        print("=" * 72, flush=True)
        print(f"LEAVEOUT: {leaveout}", flush=True)
        print(f"TRAIN: {train_path}", flush=True)
        print(f"VALID: {valid_path}", flush=True)
        print(f"MODEL: {model_path}", flush=True)

        if model_path.exists() and args.skip_train_if_model_exists:
            print(f"SKIP TRAIN: existing model found: {model_path}", flush=True)
        else:
            print("TRAIN START", flush=True)
            train_policy(str(train_path), str(model_path), val_ratio=args.internal_val_ratio)
            print("TRAIN DONE", flush=True)

        print("EVAL START", flush=True)
        completed = run_eval(eval_script, valid_path, model_path, args.python_exe, src_dir)
        print(completed.stdout, flush=True)
        if completed.stderr.strip():
            print(completed.stderr, file=sys.stderr, flush=True)

        (split_dir / "eval_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (split_dir / "eval_stderr.txt").write_text(completed.stderr, encoding="utf-8")

        parsed = parse_eval_stdout(completed.stdout)
        row: Dict[str, object] = {
            "leaveout": leaveout,
            "train_rows": manifest["train"]["rows"],
            "valid_rows": manifest["valid"]["rows"],
            "valid_positive": manifest["valid"]["positive"],
            "valid_negative": manifest["valid"]["negative"],
            "valid_positive_rate": manifest["valid"]["positive_rate"],
            "model_path": str(model_path),
            "eval_returncode": completed.returncode,
            **parsed,
        }
        results.append(row)

        with (split_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

        if completed.returncode != 0:
            print(f"WARNING: evaluation failed for leaveout={leaveout} "
                  f"returncode={completed.returncode}", file=sys.stderr)

    results_csv = outdir / "leave_one_out_results.csv"
    write_results_csv(results_csv, results)

    numeric_keys = ["accuracy", "precision", "recall", "f1", "keep_f1", "evict_f1",
                    "sum_f1", "spearman_rho_mean", "rows", "positive", "negative"]
    means: Dict[str, float] = {}
    for key in numeric_keys:
        values = [r.get(key) for r in results if isinstance(r.get(key), (int, float))]
        if values:
            means[key] = sum(float(v) for v in values) / len(values)

    summary_path = outdir / "leave_one_out_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"n_runs": len(results), "mean": means, "results_csv": str(results_csv)},
                  f, ensure_ascii=False, indent=2)

    print("=" * 72, flush=True)
    print(f"DONE: {len(results)} leave-one-out run(s)", flush=True)
    print(f"RESULTS CSV: {results_csv}", flush=True)
    print(f"SUMMARY JSON: {summary_path}", flush=True)
    if means:
        print("MEAN METRICS:", flush=True)
        for k, v in means.items():
            print(f"  {k}: {v:.4f}", flush=True)


if __name__ == "__main__":
    main()
