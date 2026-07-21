"""
data.py
==============================================================
Cell2Cell 통신사 고객 이탈(Churn) 예측 - 로드·검증·데이터셋 생성
(팀 작업계획서 "2. 이 데이터에서 반드시 결정/처리할 것" 항목 반영)

[사용 방법 - PyCharm, 팀 프로젝트 구조 기준]
project/
├── data/
│   ├── raw/cell2celltrain.csv       <- Kaggle 원본 (51,047명, Target 있음). 수정 금지.
│   ├── interim/                      <- 리텐션 포함/제외 "두 실험 버전"이 여기 저장됨 (자동 생성)
│   └── processed/                    <- 최종 채택한 리텐션 제외 학습 테이블
├── artifacts/                        <- 전처리기(.joblib)·feature_schema.json (자동 생성)
└── src/data.py                       <- 이 파일

실행: python src/data.py
   -> data/interim/에 비교 실험용 두 버전(train/val/test × with/without_retention)이 생성된다.
   -> 최종 채택된 without_retention 버전만 data/processed/train.csv 등의 표준 이름으로 저장된다.

[이 스크립트가 하는 일]
1. cell2celltrain.csv를 Train/Val/Test로 먼저 stratify=y 3분할한다 (60/20/20).
2. ServiceArea(747개 고유값), HandsetPrice('Unknown' 포함 문자열), AgeHH1/2(구조적
   0-센티널 존재) 등 이 데이터 특유의 지저분한 컬럼을 정리한다.
3. 리텐션팀 관련 3개 컬럼(RetentionCalls, RetentionOffersAccepted,
   MadeCallToRetentionTeam)은 누수 의심 컬럼이라 포함/제외 두 버전을 모두 만들어
   data/interim/에 저장한다. 최종 운영 데이터에는 검증 결과에 따라 제외 버전을 사용한다.
4. 결측치 대체·인코딩·스케일링을 하나의 Pipeline으로 묶어 Train에만 fit한다.
5. SMOTE는 여기서 적용하지 않는다 - 클래스 불균형은 모델링 단계에서
   class_weight='balanced'로 우선 대응하기로 팀에서 정했다.
==============================================================
"""

import json
from pathlib import Path
from shutil import copy2
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------------
# 0. 경로 및 기본 설정
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT_DIR / "data" / "raw" / "cell2celltrain.csv"

# data/interim: 리텐션 포함/제외 "실험용" 두 버전을 여기에 저장한다 (아직 확정 아님)
INTERIM_DIR = ROOT_DIR / "data" / "interim"
# data/processed: 확정된 without_retention 버전만 표준 파일명(train/val/test)으로 저장한다.
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FINAL_VARIANT = "without_retention"
ARTIFACT_DIR = ROOT_DIR / "artifacts"

TARGET_COL = "Churn"          # "Yes"/"No" -> 1/0 으로 변환해서 사용
ID_COLUMNS = ["CustomerID"]   # 식별자 -> Feature 제외

# 리텐션팀 관련 컬럼은 "이탈 조짐을 보인 고객에게 리텐션팀이 먼저 연락한다"는
# 순환 논리가 있을 수 있는 누수 의심 컬럼이다.
# 실제로 데이터를 열어보면 접촉 이력이 있는 고객의 이탈률이 훨씬 높다
# (MadeCallToRetentionTeam=No -> 28.2%, Yes -> 45.0%). 이게 "접촉했더니 이탈이 늘었다"가
# 아니라 "이탈 위험 신호가 보인 고객을 리텐션팀이 먼저 골라 연락했다"는
# 반대 방향의 인과관계일 가능성이 높다. 그래서 포함/제외 두 버전을 모두 만들어
# 모델링 단계에서 성능·해석 두 측면을 비교하기로 했다.
RETENTION_COLS = ["RetentionCalls", "RetentionOffersAccepted", "MadeCallToRetentionTeam"]

# AgeHH1/AgeHH2는 이 데이터셋에서 0을 "해당 세대원 없음"이라는 의미로 이미
# 광범위하게 쓰고 있다(AgeHH1==0 이 13,917건, AgeHH2==0 이 26,087건 존재).
# 반면 진짜 결측(NaN, 909건 - AgeHH1/AgeHH2 항상 같이 결측)은 "정보 자체가
# 기록되지 않음"이라 성격이 다르다. 그런데도 데이터셋이 이미 0을
# "정보 없음/해당 없음"의 센티널로 쓰고 있으므로, 중앙값 같은 그럴듯한 나이를
# 채우기보다 기존 인코딩 관례에 맞춰 0으로 채우는 쪽이 더 일관적이라고 판단했다.
ZERO_FILL_COLS = ["AgeHH1", "AgeHH2"]

# 그 외 수치형 컬럼은 결측 비율이 0.3~1.8%로 낮고 특정 패턴이 없어 Train 중앙값으로 대체한다.
NUMERIC_COLS = [
    "MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge", "DirectorAssistedCalls",
    "OverageMinutes", "RoamingCalls", "PercChangeMinutes", "PercChangeRevenues",
    "DroppedCalls", "BlockedCalls", "UnansweredCalls", "CustomerCareCalls",
    "ThreewayCalls", "ReceivedCalls", "OutboundCalls", "InboundCalls",
    "PeakCallsInOut", "OffPeakCallsInOut", "DroppedBlockedCalls", "CallForwardingCalls",
    "CallWaitingCalls", "MonthsInService", "UniqueSubs", "ActiveSubs",
    "Handsets", "HandsetModels", "CurrentEquipmentDays", "AgeHH1", "AgeHH2",
    "ReferralsMadeBySubscriber", "IncomeGroup", "AdjustmentsToCreditRating",
    "HandsetPrice",  # 문자열 -> 숫자로 변환한 뒤 이 리스트에서 수치형으로 취급
]

CATEGORICAL_COLS = [
    "ChildrenInHH", "HandsetRefurbished", "HandsetWebCapable", "TruckOwner", "RVOwner",
    "Homeownership", "BuysViaMailOrder", "RespondsToMailOffers", "OptOutMailings",
    "NonUSTravel", "OwnsComputer", "HasCreditCard", "NewCellphoneUser",
    "NotNewCellphoneUser", "OwnsMotorcycle", "CreditRating", "PrizmCode",
    "Occupation", "MaritalStatus",
    "ServiceArea_Region",  # ServiceArea를 가공해서 만든 파생 컬럼
]

RARE_REGION_THRESHOLD = 50  # 지역 코드 앞 3자리 기준, Train에서 이 값 미만이면 'Other'로 묶음


# 1. 원본 로드 및 품질 점검
def load_train_data(path: Path = TRAIN_PATH) -> pd.DataFrame:
    """Target이 포함된 cell2celltrain.csv를 읽는다."""
    df = pd.read_csv(path)
    return df

def print_quality_report(df: pd.DataFrame) -> None:
    print(f"[품질 점검] shape: {df.shape}")
    print(f"[품질 점검] CustomerID 중복 수: {df['CustomerID'].duplicated().sum()}")
    print(f"[품질 점검] Churn 분포:\n{df['Churn'].value_counts(normalize=True).round(4)}")
    null_counts = df.isnull().sum()
    print("[품질 점검] 결측치가 있는 컬럼:")
    print(null_counts[null_counts > 0].sort_values(ascending=False))


# 2. Cell2Cell 데이터 특유의 지저분한 컬럼 정리
def engineer_service_area(df: pd.DataFrame, rare_regions: Optional[set] = None):
    """
    ServiceArea는 'NYCBRO917' 처럼 [3자리 대도시권 코드][3자리 세부지역][우편번호 앞자리]
    형태의 문자열이며 고유값이 747개나 돼서 그대로 원-핫 인코딩하면 컬럼이 폭발한다.

    앞 3자리(대도시권 코드, 예: NYC/LAX/SFR/DAL)만 잘라내면 고유값이 57개로 줄고,
    그중에서도 표본이 50건 미만인 소규모 지역(12개, 전체의 0.47%)은 'Other'로 묶어
    최종 카테고리 수를 관리 가능한 수준으로 낮췄다.

    rare_regions: Train에서 계산한 '희소 지역' 집합. Val/Test에는 이 집합을
    그대로 재사용해야 한다 (Train 기준을 벗어나면 데이터 누수).
    """
    df = df.copy()
    region3 = df["ServiceArea"].astype(str).str[:3]
    region3 = region3.where(df["ServiceArea"].notna(), "Unknown")

    if rare_regions is None:
        # Train에서만 호출되는 경로: 여기서 희소 지역 집합을 계산해서 반환
        counts = region3.value_counts()
        rare_regions = set(counts[counts < RARE_REGION_THRESHOLD].index)

    df["ServiceArea_Region"] = region3.where(~region3.isin(rare_regions), "Other")
    return df, rare_regions

def engineer_handset_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    HandsetPrice는 '10','100',...,'Unknown' 처럼 숫자와 문자열이 섞인 컬럼이다.
    'Unknown'을 그냥 버리면 정보 손실이라, 별도 표시(Flag) 컬럼으로 남겨두고
    숫자 부분은 결측(NaN) 처리해서 이후 수치형 파이프라인의 중앙값 대체를 타게 한다.
    """
    df = df.copy()
    df["HandsetPrice_Unknown"] = (df["HandsetPrice"] == "Unknown").astype(int)
    df["HandsetPrice"] = pd.to_numeric(df["HandsetPrice"], errors="coerce")
    return df

def clean_features(df: pd.DataFrame, rare_regions: Optional[set] = None):
    """위 두 정제 함수를 순서대로 적용. Target을 참조하지 않는 단순 파생/정리라
    Train/Val/Test 분리 이전이든 이후든 결과가 같지만, 희소 지역(rare_regions) 기준은
    Train에서만 계산해서 재사용하므로 반드시 분리 이후, Train에서 먼저 계산한다."""
    df, rare_regions = engineer_service_area(df, rare_regions)
    df = engineer_handset_price(df)
    return df, rare_regions



# 3. Train / Validation / Test 분리
def split_data(df: pd.DataFrame, seed: int = 42):
    """고객 1명 = 1행(스냅샷 데이터)이므로 stratify=y 랜덤 분리를 사용한다."""
    y = (df[TARGET_COL] == "Yes").astype(int)
    train_val_df, test_df, ytv, yte = train_test_split(
        df, y, test_size=0.20, stratify=y, random_state=seed
    )
    train_df, val_df, ytr, yval = train_test_split(
        train_val_df, ytv, test_size=0.25, stratify=ytv, random_state=seed
    )  # 0.25 * 0.8 = 0.2 -> 전체 기준 60/20/20
    return (
        train_df.reset_index(drop=True), ytr.reset_index(drop=True),
        val_df.reset_index(drop=True), yval.reset_index(drop=True),
        test_df.reset_index(drop=True), yte.reset_index(drop=True),
    )


# 4. 전처리 Pipeline
def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """
    [수치형] AgeHH1/AgeHH2는 constant=0으로, 나머지는 median으로 결측을 채운 뒤
    StandardScaler로 표준화한다 (근거는 상단 ZERO_FILL_COLS 주석 참고).
    [범주형] 최빈값 대체 + OneHotEncoder(handle_unknown="ignore").
    """
    zero_fill_present = [c for c in ZERO_FILL_COLS if c in numeric_cols]
    median_fill_present = [c for c in numeric_cols if c not in ZERO_FILL_COLS]

    transformers = []
    if median_fill_present:
        median_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num_median", median_pipe, median_fill_present))
    if zero_fill_present:
        zero_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num_zero", zero_pipe, zero_fill_present))

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers)

def get_feature_names(preprocessor: ColumnTransformer, numeric_cols, categorical_cols):
    names = []
    for name, _, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            names.extend(ohe.get_feature_names_out(categorical_cols))
        elif name in ("num_median", "num_zero"):
            names.extend(cols)
    return names



# 5. 데이터셋 하나(포함 or 제외 버전)를 처리하는 메인 로직
def process_variant(train_df, ytr, val_df, yval, test_df, yte,
                     numeric_cols, categorical_cols, variant_name: str):
    """variant_name: 'with_retention' 또는 'without_retention'.
    같은 분할(train/val/test)을 그대로 재사용하고, 컬럼 구성만 다르게 해서
    전처리기를 Train에 fit한다."""
    feature_cols = numeric_cols + categorical_cols
    X_train, X_val, X_test = train_df[feature_cols], val_df[feature_cols], test_df[feature_cols]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)
    feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)

    def to_df(X_raw, y):
        arr = preprocessor.transform(X_raw)
        out = pd.DataFrame(arr, columns=feature_names)
        out[TARGET_COL] = y.reset_index(drop=True)
        return out

    train_out = to_df(X_train, ytr)
    val_out = to_df(X_val, yval)
    test_out = to_df(X_test, yte)

    train_out.to_csv(INTERIM_DIR / f"train_{variant_name}.csv", index=False)
    val_out.to_csv(INTERIM_DIR / f"val_{variant_name}.csv", index=False)
    test_out.to_csv(INTERIM_DIR / f"test_{variant_name}.csv", index=False)
    joblib.dump(preprocessor, ARTIFACT_DIR / f"preprocessor_{variant_name}.joblib")

    print(f"[{variant_name}] Feature 수: {len(feature_names)} "
          f"(수치형 {len(numeric_cols)} + 범주형 원-핫 {len(feature_names) - len(numeric_cols)})")
    return feature_names



def promote_to_processed(variant_name: str = FINAL_VARIANT) -> None:
    """확정된 실험 버전만 data/processed의 최종 학습 테이블로 승격한다."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        source = INTERIM_DIR / f"{split}_{variant_name}.csv"
        if not source.exists():
            raise FileNotFoundError(f"승격할 실험 데이터가 없습니다: {source}")
        copy2(source, PROCESSED_DIR / f"{split}.csv")
    print(f"[최종 데이터 승격] {variant_name} -> data/processed/train.csv, val.csv, test.csv")

# 6. 메인 실행
def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_train_data()
    print_quality_report(df)

    # 1) 먼저 분리 (희소 지역 집합도 Train에서만 계산해서 재사용해야 하므로,
    #    ServiceArea 가공보다도 분리를 먼저 한다)
    train_df, ytr, val_df, yval, test_df, yte = split_data(df)
    print(f"\n[분리 결과] Train {len(train_df)} / Val {len(val_df)} / Test {len(test_df)}")
    print(f"[분리 결과] 이탈률 - Train {ytr.mean():.4f} / Val {yval.mean():.4f} / Test {yte.mean():.4f}")

    # 2) Train에서 희소 지역 집합을 계산하고, Val/Test에는 그 기준을 그대로 적용
    train_df, rare_regions = clean_features(train_df, rare_regions=None)
    val_df, _ = clean_features(val_df, rare_regions=rare_regions)
    test_df, _ = clean_features(test_df, rare_regions=rare_regions)
    print(f"\n[ServiceArea] Train 기준 희소 지역(< {RARE_REGION_THRESHOLD}건) {len(rare_regions)}개 -> 'Other'로 그룹화")

    # 3) 리텐션 컬럼 포함/제외 두 버전을 모두 생성 (팀 작업계획서 #2)
    cat_with = CATEGORICAL_COLS  # RETENTION_COLS 중 MadeCallToRetentionTeam은 범주형이라 별도 처리
    num_with = NUMERIC_COLS + ["RetentionCalls", "RetentionOffersAccepted"]
    cat_with_retention = CATEGORICAL_COLS + ["MadeCallToRetentionTeam"]

    print("\n=== [버전 1] 리텐션 컬럼 포함 ===")
    process_variant(train_df, ytr, val_df, yval, test_df, yte,
                     num_with, cat_with_retention, "with_retention")

    print("\n=== [버전 2] 리텐션 컬럼 제외 ===")
    process_variant(train_df, ytr, val_df, yval, test_df, yte,
                     NUMERIC_COLS, CATEGORICAL_COLS, "without_retention")

    # 팀이 최종 채택한 리텐션 제외 버전을 processed 폴더의 표준 파일명으로 저장한다.
    promote_to_processed()

    # 4) 메타데이터 저장
    metadata = {
        "target": TARGET_COL,
        "target_meaning": {"0": "유지(No)", "1": "이탈(Yes)"},
        "id_columns": ID_COLUMNS,
        "retention_columns_leakage_note": (
            "MadeCallToRetentionTeam=Yes 고객의 이탈률(45.0%)이 No 고객(28.2%)보다 "
            "훨씬 높음 -> 리텐션팀이 이탈 위험 신호를 보고 먼저 연락했을 가능성 "
            "(순환 논리). with_retention / without_retention 두 버전을 비교했으며, "
            "예측 시점의 불확실성과 성능 과대평가 위험을 고려해 without_retention을 최종 채택함."
        ),
        "service_area_note": (
            f"ServiceArea 원본 747개 고유값 -> 앞 3자리 대도시권 코드로 축약 후 "
            f"Train 기준 {RARE_REGION_THRESHOLD}건 미만 지역은 'Other'로 그룹화"
        ),
        "zero_fill_columns": ZERO_FILL_COLS,
        "zero_fill_note": "AgeHH1/AgeHH2는 0이 '해당 세대원 없음'의 기존 센티널 값이라 "
                           "결측치도 중앙값 대신 0으로 채움 (기존 인코딩 관례와 일관성 유지)",
        "class_imbalance_note": "이탈률 약 28.8%. 전처리 단계에서 SMOTE를 적용하지 않았고, "
                                 "모델링 단계에서 class_weight='balanced'를 우선 적용하기로 함.",
        "split": {
            "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df),
            "train_churn_rate": round(float(ytr.mean()), 4),
            "val_churn_rate": round(float(yval.mean()), 4),
            "test_churn_rate": round(float(yte.mean()), 4),
        },
        "rare_regions_grouped_to_other": sorted(rare_regions),
    }
    with open(ARTIFACT_DIR / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n[저장 완료] data/interim/*_with_retention.csv, *_without_retention.csv")
    print("[저장 완료] artifacts/preprocessor_with_retention.joblib, preprocessor_without_retention.joblib, feature_schema.json")


if __name__ == "__main__":
    main()
