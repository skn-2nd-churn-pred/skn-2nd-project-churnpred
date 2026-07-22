"""모델 메타데이터 생성 — 가이드 "⑧ 최종 모델 저장과 추론 검증" 요구사항.

가이드가 모델과 함께 저장하라고 명시한 항목을 한 파일로 모은다.
  - 실제 학습 날짜와 metadata 생성 시각
  - 학습 데이터 fingerprint
  - Feature 이름과 순서
  - Target의 0/1 의미
  - 선택한 threshold
  - Validation/Test 지표
  - 학습 환경과 metadata 생성 환경의 Python/주요 패키지 버전

실행 (둘 중 아무거나):
    python src/save_metadata.py
    python -m src.save_metadata

산출물: artifacts/model_metadata.json
"""
from __future__ import annotations

import json
import math
import platform
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

try:  # 모듈로 실행할 때
    from .predict import load_config, load_model, resolve_path
except ImportError:  # 파일을 직접 실행하거나 PyCharm 실행 버튼을 쓸 때
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.predict import load_config, load_model, resolve_path

OUT_PATH = "artifacts/model_metadata.json"
METRICS_LONG = "artifacts/modeling/model_metrics_long.csv"
TEST_METRICS = "artifacts/modeling/test_metrics.json"

METRIC_KEYS = ("pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1")
CONFUSION_KEYS = (
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
)
TEST_INFO_KEYS = (
    "evaluated_on",
    "model_file",
    "test_size",
    "churn_rate",
    "threshold",
)
VERSION_KEYS = ("python", "pandas", "numpy", "scikit-learn", "joblib")


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in VERSION_KEYS[1:]:
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:  # noqa: BLE001 - 버전 조회 실패는 치명적이지 않다
            versions[name] = "unknown"
    return versions


def _file_sha256(path: Path) -> str:
    """파일 내용을 기준으로 재현 가능한 SHA-256 fingerprint를 만든다."""
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_config(cfg: dict[str, Any], key: str) -> Any:
    value: Any = cfg["model"].get(key)
    if value in (None, ""):
        raise ValueError(f"config.yaml의 model.{key} 값이 필요합니다.")
    return value


def _validation_metrics(model_name: str, feature_set: str) -> dict[str, float]:
    """노트북 02가 만든 비교표에서 최종 모델의 validation 성능을 찾는다."""
    path = resolve_path(METRICS_LONG)
    if not path.exists():
        raise FileNotFoundError(
            f"Validation 지표 파일이 없습니다: {path}\n"
            "노트북 02를 실행해 모델 비교표를 먼저 생성하세요."
        )

    df = pd.read_csv(path)
    required_columns = {"model", "feature_set", *METRIC_KEYS}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Validation 지표 파일에 필요한 컬럼이 없습니다: {missing_columns}"
        )

    rows = df[(df["model"] == model_name) & (df["feature_set"] == feature_set)]
    if rows.empty:
        raise ValueError(
            f"Validation 지표에서 model={model_name!r}, "
            f"feature_set={feature_set!r} 행을 찾지 못했습니다."
        )
    if len(rows) != 1:
        raise ValueError(
            f"최종 모델의 Validation 지표가 {len(rows)}행입니다. "
            "정확히 1행이어야 합니다."
        )

    row = rows.iloc[0]
    metrics = {key: round(float(row[key]), 4) for key in METRIC_KEYS}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Validation 지표에 NaN 또는 무한대 값이 있습니다.")
    return metrics


def _test_metrics(model_path: Path, threshold: float) -> dict[str, Any]:
    """Test 지표가 현재 모델·임계값과 일치하는지 확인한 뒤 반환한다."""
    path = resolve_path(TEST_METRICS)
    if not path.exists():
        raise FileNotFoundError(
            f"Test 지표 파일이 없습니다: {path}\n"
            "src/evaluate_test.py를 실행해 최종 Test 지표를 먼저 생성하세요."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    keys = (*TEST_INFO_KEYS, *METRIC_KEYS, *CONFUSION_KEYS)
    missing_keys = [key for key in keys if key not in data]
    if missing_keys:
        raise ValueError(f"Test 지표 파일에 필요한 항목이 없습니다: {missing_keys}")
    if data["model_file"] != model_path.name:
        raise ValueError(
            "Test 지표의 모델 파일이 현재 설정과 다릅니다: "
            f"{data['model_file']} != {model_path.name}"
        )
    if not math.isclose(
        float(data["threshold"]), threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "Test 지표의 임계값이 실제 추론 임계값과 다릅니다: "
            f"{data['threshold']} != {threshold}"
        )

    metrics = {key: data[key] for key in keys}
    numeric_metric_values = [float(metrics[key]) for key in METRIC_KEYS]
    if not all(math.isfinite(value) for value in numeric_metric_values):
        raise ValueError("Test 지표에 NaN 또는 무한대 값이 있습니다.")
    return metrics


def _estimator_type(model: Any) -> str:
    """Pipeline이면 마지막 estimator의 타입을, 아니면 객체 자체 타입을 반환한다."""
    steps = getattr(model, "steps", None)
    estimator = steps[-1][1] if steps else model
    return type(estimator).__name__


def build_metadata() -> dict:
    cfg = load_config()
    model_name = str(_required_config(cfg, "name"))
    feature_set = str(_required_config(cfg, "feature_set"))
    trained_on = str(_required_config(cfg, "trained_on"))
    selection_reason = str(_required_config(cfg, "selection_reason"))
    raw_training_versions = _required_config(cfg, "training_versions")
    if not isinstance(raw_training_versions, dict):
        raise ValueError("config.yaml의 model.training_versions는 key-value 형식이어야 합니다.")
    missing_versions = [key for key in VERSION_KEYS if key not in raw_training_versions]
    if missing_versions:
        raise ValueError(f"학습 환경 버전 정보가 부족합니다: {missing_versions}")
    training_versions = {
        key: str(raw_training_versions[key]) for key in VERSION_KEYS
    }
    try:
        date.fromisoformat(trained_on)
    except ValueError as error:
        raise ValueError(
            "config.yaml의 model.trained_on은 YYYY-MM-DD 형식이어야 합니다."
        ) from error

    model_path = resolve_path(cfg["model"]["path"])
    feature_source = resolve_path(cfg["model"]["feature_source"])
    raw_source = resolve_path(cfg["data"]["raw_path"])
    target = str(cfg["target"]["column"])

    bundle = load_model()
    required_bundle_keys = {"pipeline", "threshold", "feature_names"}
    missing_bundle_keys = sorted(required_bundle_keys - set(bundle))
    if missing_bundle_keys:
        raise ValueError(
            f"저장 모델 번들에 필요한 항목이 없습니다: {missing_bundle_keys}"
        )

    model = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    feature_names = list(bundle["feature_names"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"임계값은 0과 1 사이여야 합니다: {threshold}")

    config_threshold = float(cfg["model"]["threshold"])
    if not math.isclose(
        threshold, config_threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "저장 모델 번들의 임계값과 config.yaml의 임계값이 다릅니다: "
            f"{threshold} != {config_threshold}"
        )

    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and int(expected_features) != len(feature_names):
        raise ValueError(
            f"모델은 {expected_features}개 Feature를 기대하지만 "
            f"metadata 기준 목록은 {len(feature_names)}개입니다."
        )

    validation_metrics = _validation_metrics(model_name, feature_set)
    test_metrics = _test_metrics(model_path, threshold)
    excluded_columns = {"CustomerID": "식별자 — 예측 정보 없음"}
    if feature_set == "without_retention":
        excluded_columns[
            "RetentionCalls / RetentionOffersAccepted / MadeCallToRetentionTeam"
        ] = "누수 의심 — 접촉 시점 컬럼이 없어 검증 불가"

    return {
        "trained_on": trained_on,
        "metadata_generated_at": (
            datetime.now().astimezone().isoformat(timespec="seconds")
        ),
        "model": {
            "name": model_name,
            "file": model_path.name,
            "type": type(model).__name__,
            "pipeline_type": type(model).__name__,
            "estimator_type": _estimator_type(model),
            "feature_set": feature_set,
            "selection_reason": selection_reason,
        },
        "data": {
            "source": cfg["data"]["raw_path"],
            "source_sha256": _file_sha256(raw_source) if raw_source.exists() else None,
            "split": (
                "Train 60% / Validation 20% / Test 20% "
                f"(stratify={target}, random_state={cfg['project']['random_state']})"
            ),
            "feature_source": cfg["model"]["feature_source"],
            "feature_source_sha256": _file_sha256(feature_source),
            "n_features": len(feature_names),
            "excluded_columns": excluded_columns,
        },
        "target": {
            "column": target,
            "positive_label": cfg["target"]["positive_label"],
            "meaning": cfg["target"]["meaning"],
        },
        "threshold": {
            "value": threshold,
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
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "feature_names": feature_names,
        "versions": training_versions,
        "metadata_environment_versions": _package_versions(),
    }


def main() -> None:
    meta = build_metadata()
    out = resolve_path(OUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"저장 완료: {OUT_PATH}")
    print(f"  모델      : {meta['model']['name']} ({meta['model']['file']})")
    print(f"  Estimator : {meta['model']['estimator_type']}")
    print(f"  Feature   : {meta['data']['n_features']}개")
    print(f"  임계값    : {meta['threshold']['value']}")

    def summarize(metrics: dict[str, Any]) -> str:
        parts = [f"{k.upper()} {metrics[k]:.4f}"
                 for k in ("pr_auc", "roc_auc", "recall") if k in metrics]
        return " · ".join(parts) if parts else "지표 없음"

    print(f"  Validation: {summarize(meta['metrics']['validation'])}")
    print(f"  Test      : {summarize(meta['metrics']['test'])}")


if __name__ == "__main__":
    main()
