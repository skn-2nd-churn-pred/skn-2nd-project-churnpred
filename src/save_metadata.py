"""모델 메타데이터 생성 — 가이드 "⑧ 최종 모델 저장과 추론 검증" 요구사항.

가이드가 모델과 함께 저장하라고 명시한 항목을 한 파일로 모은다.
  - 학습 날짜와 데이터 버전
  - Feature 이름과 순서
  - Target의 0/1 의미
  - 선택한 threshold
  - Validation/Test 지표
  - Python과 주요 패키지 버전

실행 (둘 중 아무거나):
    python src/save_metadata.py
    python -m src.save_metadata

산출물: artifacts/model_metadata.json
"""
from __future__ import annotations

import json
import platform
from datetime import date

import joblib
import pandas as pd

try:  # 모듈로 실행할 때
    from .predict import load_config, resolve_path
except ImportError:  # 파일을 직접 실행하거나 PyCharm 실행 버튼을 쓸 때
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.predict import load_config, resolve_path

OUT_PATH = "artifacts/model_metadata.json"
METRICS_LONG = "artifacts/modeling/model_metrics_long.csv"
TEST_METRICS = "artifacts/modeling/test_metrics.json"
FINAL_MODEL_NAME = "HistGradientBoosting"
FEATURE_SET = "without_retention"


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("pandas", "numpy", "scikit-learn", "joblib"):
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:  # noqa: BLE001 - 버전 조회 실패는 치명적이지 않다
            versions[name] = "unknown"
    return versions


def _validation_metrics() -> dict | None:
    """노트북 02가 만든 비교표에서 최종 모델의 validation 성능을 찾는다."""
    path = resolve_path(METRICS_LONG)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[(df["model"] == FINAL_MODEL_NAME) & (df["feature_set"] == FEATURE_SET)]
    if row.empty:
        return None
    row = row.iloc[0]
    keys = ("pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1")
    return {k: round(float(row[k]), 4) for k in keys if k in row.index}


def _test_metrics() -> dict | None:
    path = resolve_path(TEST_METRICS)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = ("pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1",
            "true_negative", "false_positive", "false_negative", "true_positive")
    return {k: data[k] for k in keys if k in data}


def build_metadata() -> dict:
    cfg = load_config()
    model_path = resolve_path(cfg["model"]["path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"모델이 없습니다: {model_path}\n노트북 02를 실행해 최종 모델을 저장하세요."
        )

    feature_source = resolve_path(cfg["model"]["feature_source"])
    target = cfg["target"]["column"]
    feature_names = [c for c in pd.read_csv(feature_source, nrows=0).columns if c != target]

    model = joblib.load(model_path)

    return {
        "trained_on": str(date.today()),
        "model": {
            "name": FINAL_MODEL_NAME,
            "file": model_path.name,
            "type": type(model).__name__,
            "feature_set": FEATURE_SET,
            "selection_reason": (
                "Retention 제외 버전 중 PR-AUC가 가장 높았다. 리텐션 컬럼은 접촉 시점을 "
                "확인할 수 없어 누수 여부를 검증할 방법이 없으므로 제외했다 "
                "(reports/modeling_report.md 5절)."
            ),
        },
        "data": {
            "source": cfg["data"]["raw_path"],
            "split": "Train 60% / Validation 20% / Test 20% (stratify=Churn, random_state=42)",
            "feature_source": cfg["model"]["feature_source"],
            "n_features": len(feature_names),
            "excluded_columns": {
                "CustomerID": "식별자 — 예측 정보 없음",
                "RetentionCalls / RetentionOffersAccepted / MadeCallToRetentionTeam":
                    "누수 의심 — 접촉 시점 컬럼이 없어 검증 불가",
            },
        },
        "target": {
            "column": target,
            "positive_label": cfg["target"]["positive_label"],
            "meaning": cfg["target"]["meaning"],
        },
        "threshold": {
            "value": float(cfg["model"]["threshold"]),
            "decided_on": "validation",
            "note": "Validation에서 결정하고 Test에는 고정 적용했다. "
                    "config.yaml의 model.threshold 한 곳에서 관리한다.",
        },
        "primary_metric": {
            "name": "PR-AUC",
            "reason": "불균형 데이터(이탈 28.8%)에서 소수 클래스인 이탈 고객을 "
                      "선별하는 것이 목적이므로 PR-AUC를 주 지표로 사용했다.",
        },
        "metrics": {
            "validation": _validation_metrics(),
            "test": _test_metrics(),
        },
        "feature_names": feature_names,
        "versions": _package_versions(),
    }


def main() -> None:
    meta = build_metadata()
    out = resolve_path(OUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"저장 완료: {OUT_PATH}")
    print(f"  모델      : {meta['model']['name']} ({meta['model']['file']})")
    print(f"  Feature   : {meta['data']['n_features']}개")
    print(f"  임계값    : {meta['threshold']['value']}")
    def summarize(metrics: dict | None) -> str:
        if not metrics:
            return "없음"
        parts = [f"{k.upper()} {metrics[k]:.4f}"
                 for k in ("pr_auc", "roc_auc", "recall") if k in metrics]
        return " · ".join(parts) if parts else "지표 없음"

    print(f"  Validation: {summarize(meta['metrics']['validation'])}")
    print(f"  Test      : {summarize(meta['metrics']['test'])}")

    if meta["metrics"]["validation"] and "pr_auc" not in meta["metrics"]["validation"]:
        print("\n[안내] validation 비교표에 PR-AUC가 없습니다. "
              "노트북 02를 다시 실행하면 갱신됩니다.")


if __name__ == "__main__":
    main()
