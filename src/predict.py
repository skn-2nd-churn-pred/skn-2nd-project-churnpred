"""설정 로드 + 추론 (앱·노트북 공유). src의 유일한 모듈.

노트북 02에서 아래 형식으로 저장하면 Streamlit이 그대로 동작한다:

    joblib.dump({"pipeline": pipe, "threshold": thr, "feature_names": [...]},
                "models/churn_pipeline.joblib")
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


@lru_cache(maxsize=1)
def load_model() -> dict:
    path = resolve_path(load_config()["output"]["model_path"])
    if not path.exists():
        raise FileNotFoundError(f"모델이 없습니다: {path}\n노트북 02에서 학습해 저장하세요.")
    return joblib.load(path)


def risk_grade(p: float) -> str:
    return "고위험" if p >= 0.7 else "중위험" if p >= 0.4 else "저위험"


def suggest_action(grade: str) -> str:
    return {
        "고위험": "즉시 전담 상담 · 할인/요금제 재설계 · 이탈 방어 쿠폰",
        "중위험": "맞춤 혜택 안내 · 온보딩 재안내 · 만족도 설문",
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
