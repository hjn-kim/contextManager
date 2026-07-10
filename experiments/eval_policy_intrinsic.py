import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from policy import EvictionMLP
from config import DEFAULT_CONFIG


def safe_div(a, b):
    return a / b if b else 0.0


def spearman_corr(x, y):
    """
    scipy 없이 Spearman rho 계산.
    x, y 길이가 2 미만이거나 label이 전부 같으면 None 반환.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 2:
        return None
    if len(set(y.tolist())) < 2:
        return None

    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))

    rx = rx.astype(float)
    ry = ry.astype(float)

    if rx.std() == 0 or ry.std() == 0:
        return None

    return float(np.corrcoef(rx, ry)[0, 1])


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    rows = load_jsonl(args.data)

    X = torch.tensor([r["features"] for r in rows], dtype=torch.float32)
    y = np.array([int(r["label"]) for r in rows])

    model = EvictionMLP(DEFAULT_CONFIG.policy)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(X).squeeze(-1)
        scores = torch.sigmoid(logits).cpu().numpy()

    pred = (scores >= args.threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    acc = safe_div(tp + tn, len(y))

    print("=== Intrinsic Policy Evaluation ===")
    print(f"model      : {args.model}")
    print(f"rows       : {len(rows)}")
    print(f"positive   : {int(y.sum())}")
    print(f"negative   : {int((1-y).sum())}")
    print(f"threshold  : {args.threshold}")
    print(f"accuracy   : {acc:.4f}")
    print(f"precision  : {precision:.4f}")
    print(f"recall     : {recall:.4f}")
    print(f"f1         : {f1:.4f}")
    print(f"tp/fp/tn/fn: {tp}/{fp}/{tn}/{fn}")

    # Optional: Spearman by conversation_id + timestep/current_t if fields exist
    groups = defaultdict(list)
    for r, score, label in zip(rows, scores, y):
        conv = r.get("conversation_id", "unknown")
        t = r.get("timestep", r.get("current_t", r.get("current_step", None)))
        if t is not None:
            groups[(conv, t)].append((score, label))

    rhos = []
    for _, vals in groups.items():
        if len(vals) < 2:
            continue
        s = [v[0] for v in vals]
        l = [v[1] for v in vals]
        rho = spearman_corr(s, l)
        if rho is not None:
            rhos.append(rho)

    if rhos:
        print(f"spearman_rho_mean: {np.mean(rhos):.4f}")
        print(f"spearman_groups  : {len(rhos)}")
    else:
        print("spearman_rho_mean: N/A")
        print("reason           : jsonl에 timestep/current_t 정보가 없거나 group별 label 다양성이 부족함")


if __name__ == "__main__":
    main()