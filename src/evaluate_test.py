"""최종 Test 평가 — 프로젝트에서 딱 한 번만 실행하는 스크립트.

실행 (둘 중 아무거나 · PyCharm 실행 버튼도 가능):
    python src/evaluate_test.py
    python -m src.evaluate_test

[이 스크립트가 지키는 규율]
- 재학습하지 않는다. config.yaml이 가리키는 **저장된 최종 모델**을 그대로 불러온다.
- 임계값을 탐색하지 않는다. config.yaml의 model.threshold를 **고정**해서 적용한다.
  (임계값은 Validation에서 이미 결정했고, Test에서 다시 고르면 Test가 오염된다.)
- 따라서 이 스크립트에는 "성능이 더 잘 나오게 조정하는" 코드가 존재하지 않는다.

[Test 결과를 본 뒤 하면 안 되는 것]
- 결과가 기대보다 낮다고 모델·임계값·Feature를 다시 고르는 것.
  그렇게 하는 순간 Test는 더 이상 "한 번도 보지 않은 데이터"가 아니게 되고,
  발표에서 제시하는 성능은 실제 일반화 성능이 아니게 된다.
- 낮게 나왔다면 그대로 보고하고 원인 추정을 '한계'에 기록하는 것이 맞다.
"""
from __future__ import annotations

import json
from datetime import date

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:  # python -m src.evaluate_test 로 실행할 때
    from .predict import load_config, resolve_path
except ImportError:  # python src/evaluate_test.py 나 PyCharm 실행 버튼으로 직접 실행할 때
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.predict import load_config, resolve_path

TEST_PATH = "data/interim/test_without_retention.csv"
OUT_JSON = "artifacts/modeling/test_metrics.json"
OUT_MD = "reports/test_report.md"


def evaluate() -> dict:
    cfg = load_config()
    target = cfg["target"]["column"]
    threshold = float(cfg["model"]["threshold"])   # Validation에서 확정된 값, 여기서 변경 금지

    model_path = resolve_path(cfg["model"]["path"])
    test_path = resolve_path(TEST_PATH)
    for path, hint in ((model_path, "노트북 02를 실행해 최종 모델을 저장하세요."),
                       (test_path, "python src/data.py 를 실행해 분할 데이터를 생성하세요.")):
        if not path.exists():
            raise FileNotFoundError(f"파일이 없습니다: {path}\n{hint}")

    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    model = joblib.load(model_path)
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and expected != X_test.shape[1]:
        raise ValueError(
            f"모델이 기대하는 feature 수({expected})와 test 데이터 컬럼 수"
            f"({X_test.shape[1]})가 다릅니다. 같은 전처리 버전인지 확인하세요."
        )

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)      # 고정 임계값 적용
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    return {
        "evaluated_on": str(date.today()),
        "model_file": model_path.name,
        "test_size": int(len(test_df)),
        "churn_rate": round(float(y_test.mean()), 4),
        "threshold": threshold,
        # 임계값과 무관한 지표 (모델의 순위화 능력)
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        # 임계값에 따라 달라지는 지표
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def write_report(m: dict) -> None:
    """발표·결과서에 그대로 옮길 수 있는 Markdown을 생성한다."""
    detected = m["true_positive"] + m["false_negative"]
    lines = [
        "# 최종 Test 평가 결과",
        "",
        f"> 평가일 {m['evaluated_on']} · 모델 `{m['model_file']}` · "
        f"임계값 **{m['threshold']}** (Validation에서 확정, Test에 고정 적용)",
        "",
        "Test 데이터는 학습·검증·모델 선정 어디에도 사용하지 않았으며, "
        "이 평가가 **처음이자 마지막 사용**이다. 결과를 근거로 모델이나 임계값을 "
        "다시 조정하지 않는다.",
        "",
        "## 1. 평가 대상",
        "",
        "| 항목 | 값 |",
        "| --- | --- |",
        f"| Test 고객 수 | {m['test_size']:,}명 |",
        f"| 실제 이탈률 | {m['churn_rate']:.2%} |",
        "",
        "## 2. 성능 지표",
        "",
        "| 지표 | 값 | 비고 |",
        "| --- | --- | --- |",
        f"| **PR-AUC** | **{m['pr_auc']:.4f}** | 주 지표 — 불균형 데이터에서 이탈 고객 선별 능력 |",
        f"| ROC-AUC | {m['roc_auc']:.4f} | 임계값과 무관한 순위화 능력 |",
        f"| Recall | {m['recall']:.4f} | 실제 이탈 고객 중 탐지 비율 |",
        f"| Precision | {m['precision']:.4f} | 이탈 예측 중 실제 이탈 비율 |",
        f"| F1-score | {m['f1']:.4f} | Precision·Recall 균형 |",
        f"| Accuracy | {m['accuracy']:.4f} | 참고용 (불균형 데이터라 단독 해석 금지) |",
        "",
        "## 3. Confusion Matrix",
        "",
        "| 실제 \\ 예측 | 유지(0) | 이탈(1) |",
        "| --- | ---: | ---: |",
        f"| 유지(0) | {m['true_negative']:,} | {m['false_positive']:,} |",
        f"| 이탈(1) | {m['false_negative']:,} | {m['true_positive']:,} |",
        "",
        f"- 실제 이탈 고객 {detected:,}명 중 **{m['true_positive']:,}명 탐지** "
        f"(Recall {m['recall']:.1%}), {m['false_negative']:,}명 놓침.",
        f"- 유지 고객 {m['false_positive']:,}명을 이탈로 잘못 예측 — 캠페인 비용에 직결된다.",
        "",
        "## 4. 해석",
        "",
        "- (Validation 성능과 비교해 큰 차이가 없다면) 과적합 없이 일반화됐다고 볼 수 있다.",
        "- (차이가 크다면) 그 사실과 원인 추정을 '한계'에 기록한다. "
        "결과를 이유로 모델을 다시 고르지 않는다.",
        "",
    ]
    path = resolve_path(OUT_MD)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    m = evaluate()

    print("=" * 60)
    print("최종 Test 평가 (한 번만 실행)")
    print("=" * 60)
    print(f"모델      : {m['model_file']}")
    print(f"임계값    : {m['threshold']}  (config.yaml에서 읽음, 고정)")
    print(f"Test 규모 : {m['test_size']:,}명 (실제 이탈률 {m['churn_rate']:.2%})")
    print("-" * 60)
    print(f"PR-AUC    : {m['pr_auc']:.4f}   <- 주 지표")
    print(f"ROC-AUC   : {m['roc_auc']:.4f}")
    print(f"Recall    : {m['recall']:.4f}")
    print(f"Precision : {m['precision']:.4f}")
    print(f"F1-score  : {m['f1']:.4f}")
    print(f"Accuracy  : {m['accuracy']:.4f}")
    print("-" * 60)
    print(f"TN {m['true_negative']:,} | FP {m['false_positive']:,}")
    print(f"FN {m['false_negative']:,} | TP {m['true_positive']:,}")
    print("=" * 60)

    out = resolve_path(OUT_JSON)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(m)
    print(f"저장 완료: {OUT_JSON}")
    print(f"저장 완료: {OUT_MD}")


if __name__ == "__main__":
    main()
