# -*- coding: utf-8 -*-
"""발표자료(presentation.pptx)용 차트를 생성한다.

저장된 최종 모델을 Test 데이터에 그대로 실행해 곡선·분포·누적 포착률을 그린다.
`evaluate_test.py`가 이미 확정한 평가 결과를 **시각화만** 하며,
임계값 탐색이나 모델 재선정은 하지 않는다 (Test 오염 방지).

    python src/make_deck_charts.py

출력:
    assets/images/deck/roc_curve.png
    assets/images/deck/pr_curve.png
    assets/images/deck/gain_curve.png
    assets/images/deck/prob_distribution.png
    artifacts/deck_chart_metrics.json   — 발표자료에 인용한 수치
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_config, load_model  # noqa: E402

OUT_IMG = PROJECT_ROOT / "assets" / "images" / "deck"
OUT_JSON = PROJECT_ROOT / "artifacts" / "deck_chart_metrics.json"

# 발표자료 팔레트 (presentation.pptx와 동일)
INK, RISK, BLUE = "#19202E", "#D65A54", "#4C72B0"
MUTED, GRID = "#6C7789", "#D8DEE7"

plt.rcParams.update({
    "font.family": "Malgun Gothic",   # 한글 폰트 (macOS: AppleGothic)
    "axes.unicode_minus": False,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "font.size": 11,
})


def _frame(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def _save(fig, name):
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_IMG / name, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("  saved", name)


def main() -> None:
    cfg = load_config()
    bundle = load_model()
    pipe, thr = bundle["pipeline"], bundle["threshold"]
    feats, target = bundle["feature_names"], cfg["target"]["column"]

    test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
    X, y = test[feats], test[target].astype(int)
    prob = pipe.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(y, prob)
    pr_auc = average_precision_score(y, prob)
    n, n_pos = len(y), int(y.sum())
    print(f"Test {n:,}명 · 이탈 {n_pos:,}명 · ROC-AUC {roc_auc:.4f} · PR-AUC {pr_auc:.4f}")

    # 1. ROC 곡선
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(5.3, 4.3))
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1.3, label="무작위 (AUC 0.500)")
    ax.plot(fpr, tpr, color=RISK, lw=2.6, label=f"최종 모델 (AUC {roc_auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.10, color=RISK)
    ax.set_xlabel("거짓 양성률 (FPR)")
    ax.set_ylabel("참 양성률 = Recall")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    _frame(ax)
    _save(fig, "roc_curve.png")

    # 2. PR 곡선 (주 지표)
    prec, rec, _ = precision_recall_curve(y, prob)
    base = n_pos / n
    fig, ax = plt.subplots(figsize=(5.3, 4.3))
    ax.axhline(base, ls="--", color=MUTED, lw=1.3, label=f"무작위 기준선 ({base:.3f})")
    ax.plot(rec, prec, color=BLUE, lw=2.6, label=f"최종 모델 (PR-AUC {pr_auc:.3f})")
    ax.fill_between(rec, prec, base, where=(prec >= base), alpha=0.10, color=BLUE)
    ax.set_xlabel("Recall (이탈 고객 탐지율)")
    ax.set_ylabel("Precision (예측 적중률)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    _frame(ax)
    _save(fig, "pr_curve.png")

    # 3. 누적 포착률 — 확률 높은 순으로 연락했을 때
    order = np.argsort(-prob)
    gain = np.cumsum(y.values[order]) / n_pos
    pct = np.arange(1, n + 1) / n
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1.3, label="무작위로 연락했을 때")
    ax.plot(np.r_[0, pct], np.r_[0, gain], color=RISK, lw=2.8,
            label="모델 확률 순으로 연락했을 때")
    ax.fill_between(np.r_[0, pct], np.r_[0, gain], np.r_[0, pct], alpha=0.12, color=RISK)
    marks = {}
    for q in (0.10, 0.20, 0.30, 0.50):
        g = float(gain[int(q * n) - 1])
        marks[q] = g
        ax.plot([q], [g], "o", color=INK, ms=7, zorder=5)
        ax.annotate(f"{g:.0%}", (q, g), textcoords="offset points", xytext=(6, -13),
                    fontsize=10.5, fontweight="bold", color=INK)
    ax.set_xlabel("연락한 고객 비율 (확률 높은 순)")
    ax.set_ylabel("포착한 이탈 고객 비율")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    _frame(ax)
    _save(fig, "gain_curve.png")

    # 4. 예측 확률 분포 — Precision이 낮은 이유
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bins = np.linspace(0, 1, 41)
    ax.hist(prob[y == 0], bins=bins, color=BLUE, alpha=0.72, label="실제 유지 고객")
    ax.hist(prob[y == 1], bins=bins, color=RISK, alpha=0.72, label="실제 이탈 고객")
    ax.axvline(thr, color=INK, lw=2.0, ls="--")
    ax.annotate(f"임계값 {thr}", (thr, ax.get_ylim()[1] * 0.93), xytext=(8, 0),
                textcoords="offset points", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel("모델이 예측한 이탈 확률")
    ax.set_ylabel("고객 수")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    _frame(ax)
    _save(fig, "prob_distribution.png")

    summary = {
        "generated_from": "models/histgradientboosting_without_retention_final.joblib",
        "test_n": n, "test_pos": n_pos,
        "roc_auc": round(float(roc_auc), 4), "pr_auc": round(float(pr_auc), 4),
        "threshold": thr,
        "contacted_at_threshold": int((prob >= thr).sum()),
        "contacted_pct": round(float((prob >= thr).mean()), 4),
        "cumulative_gain": {f"top{int(k * 100)}pct": round(v, 4) for k, v in marks.items()},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
