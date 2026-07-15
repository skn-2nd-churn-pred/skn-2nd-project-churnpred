"""공통 정제 규칙 — 담당자 A가 확정·수정하고, B(모델링)·C(클러스터)는 import만 한다.

목적: "B와 C가 같은 기준으로 정제"를 말이 아니라 코드로 보장한다 (handoff 카드 1번).
노트북 01(EDA)에서 정제 방침이 확정되면 이 함수에 반영하고 팀에 공지한다.

주의: 여기서는 '잘못된 값 교정'만 한다. 결측 대치(imputation)·스케일링은
      각 파이프라인에서 Train에만 fit 해야 하므로 절대 여기서 하지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Cell2Cell 원본 공통 정제. 노트북 02·03은 read_csv 직후 반드시 이 함수를 거친다."""
    df = df.copy()

    # 1) HandsetPrice: 문자열 'Unknown'(약 56.8%) → 결측으로 변환 후 숫자화 (위장 결측)
    if "HandsetPrice" in df.columns:
        df["HandsetPrice"] = pd.to_numeric(df["HandsetPrice"], errors="coerce")

    # 2) AgeHH1/AgeHH2: 나이 0(약 27%)은 불가능한 값 → 결측 처리
    for c in ("AgeHH1", "AgeHH2"):
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)

    # 3) 음수가 불가능한 컬럼: 음수(월매출 3건, 단말사용일수 76건) → 결측 처리
    for c in ("MonthlyRevenue", "CurrentEquipmentDays"):
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    # TODO(담당자 A): EDA에서 추가 발견한 규칙을 여기에 반영하고 전원에게 공지
    return df
