"""
Validate ATLAS eviction-policy artifacts: either the labeled training jsonl
(feature vectors) or a trained EvictionMLP checkpoint (.pt state_dict).

File type is picked automatically from the extension:
  .jsonl / .json  -> feature-vector checks (scispaCy usage, dead dims, etc.)
  .pt / .pth      -> model checkpoint checks (shapes, NaN/Inf, dead neurons)

--------------------------------------------------------------------------
jsonl feature layout (see PolicyConfig.feature_dim = 396):
  [0:384]    sentence embedding (normalize_embeddings=True -> L2 norm ~1)
  [384]      position
  [385]      recency
  [386]      speaker (0 or 1)
  [387]      token_count
  [388]      entity_count (scispaCy medical entity count)
  [389:396]  type one-hot (7 dims, exactly one 1.0 per row)

jsonl checks:
  1. scispaCy usage: entity_count (dim 388) should not be all-zero across the
     whole file. All-zero means scispaCy failed to load and the code silently
     fell back to 0.0 for every row (see _get_scispacy_nlp()).
  2. No feature dimension is constant-zero across every row (dead dimension).
  3. Embedding dims [0:384] have unit L2 norm.
  4. Type one-hot dims [389:396] sum to exactly 1.0 per row.
  5. Scalar sanity ranges: position/recency in [0,1], speaker in {0,1},
     token_count >= 0, entity_count >= 0.

--------------------------------------------------------------------------
.pt checkpoint checks (EvictionMLP: feature_dim -> hidden_1 -> hidden_2 -> output_dim):
  1. Loads as a state_dict compatible with EvictionMLP (matching PolicyConfig).
  2. No NaN/Inf in any parameter tensor.
  3. Layer shapes match PolicyConfig (feature_dim=396, hidden_1=128, hidden_2=64).
  4. No dead output neurons (weight row + bias all zero).
  5. No dead input dimensions in the first layer (weight column all zero across
     every hidden unit) -- e.g. a permanently-zero entity_count (scispaCy)
     column means the model never learned to use that feature.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

EXPECTED_FEATURE_DIM = 396
EMBED_START, EMBED_END = 0, 384
POSITION_IDX = 384
RECENCY_IDX = 385
SPEAKER_IDX = 386
TOKEN_COUNT_IDX = 387
ENTITY_COUNT_IDX = 388
TYPE_ONEHOT_START = 389


def describe_input_dim(i: int) -> str:
    if EMBED_START <= i < EMBED_END:
        return f"embedding[{i}]"
    if i == POSITION_IDX:
        return "position"
    if i == RECENCY_IDX:
        return "recency"
    if i == SPEAKER_IDX:
        return "speaker"
    if i == TOKEN_COUNT_IDX:
        return "token_count"
    if i == ENTITY_COUNT_IDX:
        return "entity_count (scispaCy)"
    return f"type_onehot[{i - TYPE_ONEHOT_START}]"


# =============================================================================
# jsonl feature-vector checks
# =============================================================================

def check_jsonl_file(path: Path, feature_dim: int = EXPECTED_FEATURE_DIM) -> bool:
    total = 0
    skipped = 0
    decode_errors = []
    json_errors = []

    nonzero_count = np.zeros(feature_dim, dtype=np.int64)

    entity_zero_rows = 0
    embed_norm_sum = 0.0
    embed_norm_min = float("inf")
    embed_norm_max = float("-inf")
    bad_onehot_rows = 0
    out_of_range_rows = {"position": 0, "recency": 0, "speaker": 0,
                          "token_count": 0, "entity_count": 0}

    # Read as bytes and decode per line so one bad/truncated line (e.g. a
    # partial write, or a file still syncing from cloud storage) doesn't
    # crash the whole scan.
    with path.open("rb") as f:
        for lineno, raw_line in enumerate(f, start=1):
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                decode_errors.append((lineno, str(exc)))
                continue

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                json_errors.append((lineno, str(exc)))
                continue

            features = row.get("features")

            if features is None or len(features) != feature_dim:
                skipped += 1
                continue

            total += 1
            arr = np.asarray(features, dtype=np.float64)

            nonzero_count += (arr != 0.0)

            if arr[ENTITY_COUNT_IDX] == 0.0:
                entity_zero_rows += 1

            embed = arr[EMBED_START:EMBED_END]
            norm = float(np.linalg.norm(embed))
            embed_norm_sum += norm
            embed_norm_min = min(embed_norm_min, norm)
            embed_norm_max = max(embed_norm_max, norm)

            onehot = arr[TYPE_ONEHOT_START:feature_dim]
            if not np.isclose(onehot.sum(), 1.0, atol=1e-6):
                bad_onehot_rows += 1

            position, recency, speaker, token_count, entity_count = arr[POSITION_IDX:ENTITY_COUNT_IDX + 1]
            if not (0.0 <= position <= 1.0):
                out_of_range_rows["position"] += 1
            if not (0.0 <= recency <= 1.0):
                out_of_range_rows["recency"] += 1
            if speaker not in (0.0, 1.0):
                out_of_range_rows["speaker"] += 1
            if token_count < 0:
                out_of_range_rows["token_count"] += 1
            if entity_count < 0:
                out_of_range_rows["entity_count"] += 1

    ok = True

    print(f"File: {path} (jsonl feature check)")
    print(f"Total valid rows: {total} (skipped malformed/wrong-length: {skipped})")
    print()

    is_ok = len(decode_errors) == 0
    ok = ok and is_ok
    print("[0] File encoding / JSON integrity")
    print(f"  lines with UTF-8 decode errors: {len(decode_errors)} [{'OK' if is_ok else 'FAIL'}]")
    for lineno, err in decode_errors[:5]:
        print(f"    line {lineno}: {err}")
    is_ok = len(json_errors) == 0
    ok = ok and is_ok
    print(f"  lines with malformed JSON: {len(json_errors)} [{'OK' if is_ok else 'FAIL'}]")
    for lineno, err in json_errors[:5]:
        print(f"    line {lineno}: {err}")
    print()

    if total == 0:
        print("No valid rows to check. [FAIL]")
        return False

    print("[1] scispaCy usage (entity_count, dim 388)")
    zero_ratio = entity_zero_rows / total
    if zero_ratio == 1.0:
        ok = False
        print("  entity_count is 0.0 for 100% of rows -> scispaCy NOT used "
              "(likely fell back, see _get_scispacy_nlp() warning in logs) [FAIL]")
    elif zero_ratio > 0.95:
        print(f"  entity_count is 0.0 for {100 * zero_ratio:.2f}% of rows -> "
              f"suspiciously high, verify scispaCy loaded correctly [WARN]")
    else:
        print(f"  entity_count is 0.0 for {100 * zero_ratio:.2f}% of rows -> scispaCy appears active [OK]")
    print()

    print(f"[2] Dead dimensions (constant zero across all {total} rows)")
    all_zero_dims = [i for i in range(feature_dim) if nonzero_count[i] == 0]
    if all_zero_dims:
        ok = False
        embed_dead = [i for i in all_zero_dims if EMBED_START <= i < EMBED_END]
        scalar_dead = [i for i in all_zero_dims if POSITION_IDX <= i <= ENTITY_COUNT_IDX]
        onehot_dead = [i for i in all_zero_dims if i >= TYPE_ONEHOT_START]
        print(f"  {len(all_zero_dims)} dead dimension(s) found [FAIL]")
        if embed_dead:
            print(f"    embedding dims dead: {embed_dead}")
        if scalar_dead:
            print(f"    scalar dims dead: {scalar_dead}")
        if onehot_dead:
            print(f"    type one-hot dims dead (that type never appears): {onehot_dead}")
    else:
        print("  none [OK]")
    print()

    print("[3] Embedding L2 norm (expected ~1.0, normalize_embeddings=True)")
    embed_norm_mean = embed_norm_sum / total
    is_ok = abs(embed_norm_mean - 1.0) < 0.05
    ok = ok and is_ok
    print(f"  mean={embed_norm_mean:.4f} min={embed_norm_min:.4f} max={embed_norm_max:.4f} "
          f"[{'OK' if is_ok else 'FAIL'}]")
    print()

    print(f"[4] Type one-hot validity (dims 389:{feature_dim}, must sum to 1.0)")
    is_ok = bad_onehot_rows == 0
    ok = ok and is_ok
    print(f"  rows with invalid one-hot (sum != 1): {bad_onehot_rows} [{'OK' if is_ok else 'FAIL'}]")
    print()

    print("[5] Scalar feature ranges")
    for name, count in out_of_range_rows.items():
        is_ok = count == 0
        ok = ok and is_ok
        print(f"  {name:<12} out-of-range rows: {count} [{'OK' if is_ok else 'FAIL'}]")
    print()

    print("=" * 60)
    print("RESULT:", "PASS - features look normal" if ok else "FAIL - issues found above")
    print("=" * 60)
    print()

    return ok


# =============================================================================
# .pt checkpoint checks
# =============================================================================

def check_pt_file(path: Path) -> bool:
    import torch
    from config import DEFAULT_CONFIG
    from policy import EvictionMLP

    ok = True
    cfg = DEFAULT_CONFIG.policy

    print(f"File: {path} (model checkpoint check)")
    print()

    print("[0] Load checkpoint")
    try:
        loaded = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"  failed to torch.load(): {exc} [FAIL]")
        print()
        print("=" * 60)
        print("RESULT: FAIL - issues found above")
        print("=" * 60)
        return False

    if hasattr(loaded, "state_dict") and not isinstance(loaded, dict):
        print("  file contains a full nn.Module, not just a state_dict -> extracting state_dict()")
        state_dict = loaded.state_dict()
    else:
        state_dict = loaded
    print(f"  loaded OK, {len(state_dict)} tensors [OK]")
    print()

    ref_model = EvictionMLP(cfg)
    ref_state = ref_model.state_dict()

    loaded_keys = set(state_dict.keys())
    ref_keys = set(ref_state.keys())
    missing_keys = sorted(ref_keys - loaded_keys)
    extra_keys = sorted(loaded_keys - ref_keys)

    print("[1] Expected layer keys (EvictionMLP: feature_dim -> hidden_1 -> hidden_2 -> output_dim)")
    is_ok = len(missing_keys) == 0
    ok = ok and is_ok
    print(f"  missing keys: {missing_keys if missing_keys else 'none'} [{'OK' if is_ok else 'FAIL'}]")
    is_ok = len(extra_keys) == 0
    ok = ok and is_ok
    print(f"  unexpected extra keys: {extra_keys if extra_keys else 'none'} [{'OK' if is_ok else 'FAIL'}]")
    print()

    print("[2] Layer shapes (expected feature_dim={}, hidden_1={}, hidden_2={}, output_dim={})".format(
        cfg.feature_dim, cfg.hidden_1, cfg.hidden_2, cfg.output_dim))
    shape_mismatches = []
    for key in sorted(loaded_keys & ref_keys):
        got = tuple(state_dict[key].shape)
        expected = tuple(ref_state[key].shape)
        if got != expected:
            shape_mismatches.append((key, got, expected))
    is_ok = len(shape_mismatches) == 0
    ok = ok and is_ok
    if shape_mismatches:
        print(f"  {len(shape_mismatches)} shape mismatch(es) [FAIL]")
        for key, got, expected in shape_mismatches:
            print(f"    {key}: got {got}, expected {expected}")
    else:
        for key in sorted(loaded_keys & ref_keys):
            print(f"    {key}: {tuple(state_dict[key].shape)} [OK]")
    print()

    print("[3] NaN / Inf in parameters")
    bad_tensors = []
    for key in sorted(loaded_keys):
        t = state_dict[key]
        if not torch.is_floating_point(t):
            continue
        if torch.isnan(t).any() or torch.isinf(t).any():
            bad_tensors.append(key)
    is_ok = len(bad_tensors) == 0
    ok = ok and is_ok
    print(f"  tensors with NaN/Inf: {bad_tensors if bad_tensors else 'none'} [{'OK' if is_ok else 'FAIL'}]")
    print()

    print("[4] Weight magnitude sanity")
    for key in sorted(loaded_keys):
        t = state_dict[key]
        if not torch.is_floating_point(t) or t.numel() == 0:
            continue
        print(f"  {key:<16} mean|w|={t.abs().mean().item():.4f}  max|w|={t.abs().max().item():.4f}")
    print()

    if shape_mismatches or missing_keys:
        print("[5] Dead neuron / dead input checks skipped (shape mismatch above).")
        print()
        print("=" * 60)
        print("RESULT:", "PASS - model looks normal" if ok else "FAIL - issues found above")
        print("=" * 60)
        return ok

    print("[5] Dead neurons (output row weights + bias all zero)")
    layer_pairs = [
        ("network.0.weight", "network.0.bias", "layer 1 (input -> hidden_1)"),
        ("network.2.weight", "network.2.bias", "layer 2 (hidden_1 -> hidden_2)"),
        ("network.4.weight", "network.4.bias", "layer 3 (hidden_2 -> output)"),
    ]
    any_dead_neuron = False
    for w_key, b_key, label in layer_pairs:
        if w_key not in state_dict or b_key not in state_dict:
            continue
        w = state_dict[w_key]
        b = state_dict[b_key]
        dead_rows = [i for i in range(w.shape[0])
                     if torch.all(w[i] == 0) and b[i] == 0]
        if dead_rows:
            any_dead_neuron = True
            print(f"  {label}: {len(dead_rows)}/{w.shape[0]} dead neuron(s) -> indices {dead_rows} [FAIL]")
        else:
            print(f"  {label}: 0/{w.shape[0]} dead neurons [OK]")
    ok = ok and not any_dead_neuron
    print()

    print("[6] Dead input dimensions (first layer, weight column all zero across every hidden unit)")
    w0 = state_dict["network.0.weight"]  # shape (hidden_1, feature_dim)
    dead_input_dims = [i for i in range(w0.shape[1]) if torch.all(w0[:, i] == 0)]
    is_ok = len(dead_input_dims) == 0
    ok = ok and is_ok
    if dead_input_dims:
        print(f"  {len(dead_input_dims)} dead input dim(s) [FAIL]")
        for i in dead_input_dims[:20]:
            print(f"    dim {i}: {describe_input_dim(i)}")
        if len(dead_input_dims) > 20:
            print(f"    ... and {len(dead_input_dims) - 20} more")
        if ENTITY_COUNT_IDX in dead_input_dims:
            print("  NOTE: entity_count (scispaCy) input is completely unused by this model.")
    else:
        print("  none [OK]")
    print()

    print("=" * 60)
    print("RESULT:", "PASS - model looks normal" if ok else "FAIL - issues found above")
    print("=" * 60)
    print()

    return ok


def check_file(path: Path, feature_dim: int = EXPECTED_FEATURE_DIM) -> bool:
    suffix = path.suffix.lower()
    if suffix in (".pt", ".pth"):
        return check_pt_file(path)
    return check_jsonl_file(path, feature_dim=feature_dim)


def main():
    parser = argparse.ArgumentParser(
        description="Validate ATLAS eviction-policy jsonl feature files or .pt model checkpoints.")
    parser.add_argument("--input", required=True, nargs="+",
                         help="Path(s) to .jsonl feature file(s) or .pt model checkpoint(s).")
    parser.add_argument("--feature-dim", type=int, default=EXPECTED_FEATURE_DIM,
                         help="Expected feature vector length for jsonl files (default: 396).")
    args = parser.parse_args()

    overall_ok = True

    for input_path in args.input:
        path = Path(input_path)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            overall_ok = False
            continue
        overall_ok = check_file(path, feature_dim=args.feature_dim) and overall_ok

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
