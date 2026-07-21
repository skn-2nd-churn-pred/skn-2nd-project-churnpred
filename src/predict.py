"""설정 로드 + 추론 (앱·노트북 공유). src의 유일한 모듈.

load_model()은 두 가지 저장 형식을 모두 지원한다.

1) 번들 dict — 모델·임계값·feature 목록을 함께 저장한 형태 (권장):
       joblib.dump({"pipeline": pipe, "threshold": thr, "feature_names": [...]}, path)

2) Pipeline 객체 단독 — 노트북 02가 현재 저장하는 형태:
       joblib.dump(pipe, path)
   이 경우 threshold는 config.yaml의 model.threshold에서,
   feature_names는 model.feature_source CSV의 컬럼 순서에서 복원한다.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _feature_names_from_source(cfg: dict[str, Any]) -> list[str]:
    """모델이 학습한 feature 이름·순서를 기준 CSV의 컬럼에서 복원한다."""
    source = resolve_path(cfg["model"]["feature_source"])
    if not source.exists():
        raise FileNotFoundError(
            f"feature 목록을 복원할 파일이 없습니다: {source}\n"
            "src/data.py를 실행해 data/processed 최종 데이터를 생성하세요."
        )
    target = cfg["target"]["column"]
    columns = pd.read_csv(source, nrows=0).columns.tolist()
    return [c for c in columns if c != target]


@lru_cache(maxsize=1)
def load_model() -> dict:
    """{pipeline, threshold, feature_names} 형태로 정규화해서 반환한다."""
    cfg = load_config()
    path = resolve_path(cfg["model"]["path"])
    if not path.exists():
        raise FileNotFoundError(f"모델이 없습니다: {path}\n노트북 02에서 학습해 저장하세요.")

    obj = joblib.load(path)
    if isinstance(obj, dict):        # 형식 1) 이미 번들이면 그대로 사용
        return obj

    # 형식 2) Pipeline 객체 단독 -> config와 기준 CSV로 나머지를 채워 번들을 만든다.
    feature_names = _feature_names_from_source(cfg)
    expected = getattr(obj, "n_features_in_", None)
    if expected is not None and expected != len(feature_names):
        raise ValueError(
            f"모델이 기대하는 feature 수({expected})와 "
            f"{cfg['model']['feature_source']}의 컬럼 수({len(feature_names)})가 다릅니다. "
            "config.yaml의 model.feature_source가 이 모델을 학습한 데이터와 같은지 확인하세요."
        )
    return {
        "pipeline": obj,
        "threshold": float(cfg["model"]["threshold"]),
        "feature_names": feature_names,
    }


def risk_grade(p: float) -> str:
    return "고위험" if p >= 0.7 else "중위험" if p >= 0.4 else "저위험"


def suggest_action(grade: str) -> str:
    return {
        "고위험": "즉시 전담 상담 · 할인/요금제 재설계 · 이탈 방어 쿠폰",
        "중위험": "맞춤 혜택 안내 · 요금제 재안내 · 만족도 설문",
        "저위험": "정기 뉴스레터 · 로열티 포인트 안내",
    }[grade]


def predict_df(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    bundle = bundle or load_model()
    pipe, thr, feat = bundle["pipeline"], bundle["threshold"], bundle["feature_names"]
    X = df.copy()
    for c in feat:
        if c not in X.columns:
            X[c] = pd.NA
    proba = pipe.predict_proba(X[feat])[:, 1]
    out = df.copy()
    out["churn_probability"] = proba.round(4)
    out["churn_prediction"] = (proba >= thr).astype(int)
    out["risk_grade"] = [risk_grade(p) for p in proba]
    out["suggested_action"] = out["risk_grade"].map(suggest_action)
    return out


def predict_one(customer: dict[str, Any], bundle: dict | None = None) -> dict[str, Any]:
    return predict_df(pd.DataFrame([customer]), bundle).iloc[0].to_dict()
