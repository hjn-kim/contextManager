import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from policy import EvictionMLP
from config import DEFAULT_CONFIG


def roc_curve_np(y_true, scores):
    """
    sklearn 없이 ROC 곡선 계산.
    반환: fpr, tpr (둘 다 (0,0)에서 시작, (1,1)에서 끝), auc
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    # 점수 내림차순으로 정렬하며 임계값을 낮춰간다.
    order = np.argsort(-scores)
    y_sorted = y_true[order]

    P = int(y_true.sum())
    N = int(len(y_true) - P)

    if P == 0 or N == 0:
        # 한 클래스만 있으면 ROC/AUC 정의 불가
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    tps = np.cumsum(y_sorted)              # 누적 true positive
    fps = np.cumsum(1 - y_sorted)          # 누적 false positive

    tpr = np.concatenate([[0.0], tps / P])
    fpr = np.concatenate([[0.0], fps / N])

    # 사다리꼴 적분으로 AUC
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def threshold_sweep(y_true, scores):
    """
    임계값을 후보 점수 전체로 훑어 각 지점의 precision/recall/f1/TPR/FPR을 구한다.
    반환: dict with best_f1 (F1 최대 지점), best_j (Youden's J = TPR-FPR 최대 지점)
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    P = int(y_true.sum())
    N = int(len(y_true) - P)

    # 후보 임계값: 실제 등장한 점수들(중복 제거) + 경계 여유값
    cand = np.unique(scores)
    cand = np.concatenate([cand, [cand.max() + 1e-6]])  # 아무것도 양성이 아닌 지점 포함

    best_f1 = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0,
               "tpr": 0.0, "fpr": 0.0}
    best_j = {"threshold": 0.5, "j": -1.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
              "tpr": 0.0, "fpr": 0.0}

    for t in cand:
        pred = scores >= t
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum(~pred & (y_true == 1)))

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        tpr = safe_div(tp, P)
        fpr = safe_div(fp, N)
        j = tpr - fpr

        if f1 > best_f1["f1"]:
            best_f1 = {"threshold": float(t), "f1": f1, "precision": precision,
                       "recall": recall, "tpr": tpr, "fpr": fpr}
        if j > best_j["j"]:
            best_j = {"threshold": float(t), "j": j, "f1": f1, "precision": precision,
                      "recall": recall, "tpr": tpr, "fpr": fpr}

    return {"best_f1": best_f1, "best_j": best_j}


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
    parser.add_argument("--threshold", type=float, default=None,
                        help="평가 임계값. 생략하면 F1이 최대가 되는 임계값을 자동으로 사용한다.")
    parser.add_argument("--roc-out", default=None,
                        help="ROC 곡선 PNG 저장 경로. 생략 시 모델 경로 옆에 <model>_roc.png 로 저장.")
    parser.add_argument("--no-show", action="store_true",
                        help="그래프 창을 띄우지 않고 PNG 저장만 한다.")
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

    # 임계값 선택: --threshold를 주면 그 값, 생략하면 F1이 최대인 임계값을 자동 사용.
    sweep = threshold_sweep(y, scores)
    both_classes = (int(y.sum()) > 0) and (int((1 - y).sum()) > 0)

    if args.threshold is not None:
        eval_threshold = args.threshold
        threshold_source = "user"
    elif both_classes:
        eval_threshold = sweep["best_f1"]["threshold"]
        threshold_source = "auto(best-F1)"
    else:
        eval_threshold = 0.5
        threshold_source = "default(단일 클래스)"

    pred = (scores >= eval_threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    acc = safe_div(tp + tn, len(y))

    # keep 기준(positive=label 1) = f1, evict 기준(positive=label 0): tp=tn, fp=fn, fn=fp
    keep_f1 = f1
    evict_precision = safe_div(tn, tn + fn)
    evict_recall = safe_div(tn, tn + fp)
    evict_f1 = safe_div(2 * evict_precision * evict_recall, evict_precision + evict_recall)
    sum_f1 = keep_f1 + evict_f1

    print("=== Intrinsic Policy Evaluation ===")
    print(f"model      : {args.model}")
    print(f"positive   : {int(y.sum())}")
    print(f"negative   : {int((1-y).sum())}")
    print(f"threshold  : {eval_threshold:.4f} [{threshold_source}]")
    print(f"accuracy   : {acc:.4f}")
    print(f"precision  : {precision:.4f}")
    print(f"recall     : {recall:.4f}")
    print(f"keep_f1    : {keep_f1:.4f}")
    print(f"evict_f1   : {evict_f1:.4f}")
    print(f"sum_f1     : {sum_f1:.4f}")
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

    # -------------------------------------------------------------------------
    # ROC 곡선
    # -------------------------------------------------------------------------
    fpr, tpr, auc = roc_curve_np(y, scores)
    print(f"roc_auc    : {auc:.4f}" if np.isfinite(auc) else "roc_auc    : N/A (한 클래스만 존재)")

    if not np.isfinite(auc):
        print("ROC 곡선을 그릴 수 없습니다: valid 데이터에 positive/negative 중 한 종류만 있습니다.")
        return

    # 임계값 스윕 결과(위에서 이미 계산): F1 최대 지점 / Youden's J 최대 지점
    bf1 = sweep["best_f1"]
    bj = sweep["best_j"]
    print("--- threshold sweep ---")
    print(f"best_f1        : {bf1['f1']:.4f} @ threshold={bf1['threshold']:.4f} "
          f"(precision={bf1['precision']:.4f}, recall={bf1['recall']:.4f})")
    print(f"youden_j_best  : J={bj['j']:.4f} @ threshold={bj['threshold']:.4f} "
          f"(f1={bj['f1']:.4f}, TPR={bj['tpr']:.4f}, FPR={bj['fpr']:.4f})")

    # 현재 threshold의 동작점 (FPR, TPR)
    op_fpr = safe_div(fp, fp + tn)
    op_tpr = safe_div(tp, tp + fn)

    if args.roc_out:
        roc_out = args.roc_out
    else:
        base, _ = os.path.splitext(args.model)
        roc_out = base + "_roc.png"

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")  # 창 없이 파일로만 저장
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="C0", lw=2, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="random")
    plt.scatter([op_fpr], [op_tpr], color="red", zorder=5,
                label=f"threshold={eval_threshold:.3f} (FPR={op_fpr:.3f}, TPR={op_tpr:.3f})")
    plt.scatter([bf1["fpr"]], [bf1["tpr"]], color="green", marker="*", s=140, zorder=6,
                label=f"best F1={bf1['f1']:.3f} @ t={bf1['threshold']:.3f}")
    plt.scatter([bj["fpr"]], [bj["tpr"]], color="purple", marker="D", s=55, zorder=6,
                label=f"Youden's J @ t={bj['threshold']:.3f}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve\n{os.path.basename(args.model)}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(roc_out, dpi=150)
    print(f"roc_png    : {roc_out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()