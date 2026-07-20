"""Streamlit 시연 (단일 파일, 4개 탭).

실행: streamlit run streamlit_app/app.py
- 저장된 최종 모델(models/churn_pipeline.joblib)을 불러온다. 앱에서 재학습하지 않는다.
- 입력을 바꾸면 확률도 실제로 바뀐다.
- 원본 CSV가 없으면 data/interim의 전처리 데이터(train+val+test)로 대체한다.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import load_config, load_model, predict_one, resolve_path  # noqa: E402

st.set_page_config(page_title="고객 이탈 예측", page_icon="📉", layout="wide")
cfg = load_config()
GRADE_ICON = {"고위험": "🔴", "중위험": "🟡", "저위험": "🟢"}
MODELING_DIR = ROOT / "artifacts" / "modeling"
CLUSTER_MODEL_PATH = ROOT / "models" / "kmeans_pipeline.joblib"

CLUSTER_FEATURE_LABELS = {
    "MonthlyRevenue": "월 평균 요금($)",
    "MonthlyMinutes": "월 통화량(분)",
    "OverageMinutes": "초과 사용량(분)",
    "MonthsInService": "가입 개월 수",
    "CurrentEquipmentDays": "현재 단말 사용일수",
    "DroppedBlockedCalls": "끊김·차단 통화 수",
    "CustomerCareCalls": "고객센터 통화 수",
    "ActiveSubs": "활성 회선 수",
}

SEGMENT_GUIDE = {
    "Loyal Long-term Customers": {
        "title": "장기 충성 고객",
        "icon": "🏅",
        "description": "가입 기간과 단말기 사용 기간이 길고, 사용량과 고객센터 이용은 낮은 안정적인 장기 고객군입니다.",
        "actions": [
            "장기 고객 감사 보상 제공",
            "단말기 교체 할인 또는 업그레이드 혜택 제안",
            "멤버십 등급 및 장기 혜택 강화",
            "신규 서비스 체험 프로모션 제공",
        ],
    },
    "Premium Heavy Users": {
        "title": "프리미엄 헤비 유저",
        "icon": "💎",
        "description": "월 매출, 월 사용량, 초과 사용량이 매우 높은 핵심 고객군입니다. 사용 규모가 큰 만큼 품질과 지원 경험이 중요합니다.",
        "actions": [
            "VIP 전용 요금제 또는 전담 혜택 제안",
            "초과 사용량 패키지 제공",
            "프리미엄 고객 지원 및 우선 응대",
            "데이터·통화 무제한 상품 전환 제안",
            "장기 유지 보상 제공",
        ],
    },
    "Multi-line Customers": {
        "title": "다회선 고객",
        "icon": "👨‍👩‍👧‍👦",
        "description": "활성 회선 수가 많지만 회선당 이용량과 매출은 상대적으로 낮은 고객군으로, 가족 또는 복수 회선 고객일 가능성이 높습니다.",
        "actions": [
            "가족 결합 할인 제안",
            "추가 회선 혜택 제공",
            "회선별 사용 활성화 프로모션",
            "가족 단위 콘텐츠 및 부가서비스 추천",
        ],
    },
    "Regular Customers": {
        "title": "일반 고객",
        "icon": "🙂",
        "description": "주요 지표가 평균 또는 평균 이하이며 뚜렷한 극단 특성이 없는 일반적인 고객군입니다.",
        "actions": [
            "기본 유지 프로그램 적용",
            "관심 서비스 기반 업셀링",
            "사용량 증가 이벤트 제공",
            "개인화된 요금제 추천",
        ],
    },
    "High-Maintenance Customers": {
        "title": "집중 관리 고객",
        "icon": "🛠️",
        "description": "사용량이 높고 끊김·차단 통화와 고객센터 접촉이 매우 많은 고객군으로, 서비스 품질 문제와 지원 수요가 큽니다.",
        "actions": [
            "네트워크 및 통화 품질 우선 점검",
            "전담 상담 또는 우선 응대",
            "문제 발생 전 선제적 안내",
            "품질 이슈에 대한 적절한 보상",
            "반복 문의 원인 분석 및 이탈 방어",
        ],
    },
}

# 현황·예측 탭에서 사람이 해석할 핵심 수치 Feature (interim 컬럼명 그대로)
KEY_FEATURES = {
    "MonthlyRevenue": "월 매출($)",
    "MonthlyMinutes": "월 통화량(분)",
    "TotalRecurringCharge": "월 기본요금($)",
    "OverageMinutes": "초과 사용량(분)",
    "PercChangeMinutes": "통화량 증감(분)",
    "DroppedBlockedCalls": "끊김·차단 통화",
    "CustomerCareCalls": "고객센터 통화",
    "RetentionCalls": "리텐션팀 통화",
    "MonthsInService": "가입 개월 수",
    "ActiveSubs": "활성 회선 수",
    "Handsets": "보유 단말 수",
    "CurrentEquipmentDays": "단말 사용일수",
    "AgeHH1": "가구주 나이",
    "HandsetPrice": "단말 가격($)",
}
CREDIT_COLS = [
    "CreditRating_1-Highest", "CreditRating_2-High", "CreditRating_3-Good",
    "CreditRating_4-Medium", "CreditRating_5-Low", "CreditRating_6-VeryLow",
    "CreditRating_7-Lowest",
]


@st.cache_data
def load_customers() -> pd.DataFrame | None:
    """원본 CSV가 있으면 사용, 없으면 전처리 완료된 interim 3분할을 합쳐 사용."""
    raw = resolve_path(cfg["data"]["raw_path"])
    if raw.exists():
        d = cfg["data"]
        df = pd.read_csv(raw, encoding=d.get("encoding", "utf-8"), sep=d.get("sep", ","))
        tcol, pos = cfg["target"]["column"], str(cfg["target"]["positive_label"]).strip().lower()
        df["_churn"] = df[tcol].astype(str).str.strip().str.lower().eq(pos).astype(int)
        return df
    parts = [ROOT / "data" / "interim" / f"{s}_with_retention.csv"
             for s in ("train", "val", "test")]
    if not all(p.exists() for p in parts):
        return None
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df["_churn"] = df["Churn"]
    return df


@st.cache_data
def load_scoring_data() -> pd.DataFrame | None:
    """모델 입력용 전처리 완료 데이터 (학습 때와 같은 feature 이름·자료형)."""
    parts = [ROOT / "data" / "interim" / f"{s}_with_retention.csv"
             for s in ("train", "val", "test")]
    if not all(p.exists() for p in parts):
        return None
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df["_churn"] = df["Churn"]
    return df


@st.cache_data
def score_customers(df: pd.DataFrame) -> pd.Series:
    """전체 고객의 이탈 확률 (위험도 구간·예시 고객 선택에 사용)."""
    bundle = load_model()
    feat = bundle["feature_names"]
    X = df.copy()
    for c in feat:
        if c not in X.columns:
            X[c] = pd.NA
    return pd.Series(bundle["pipeline"].predict_proba(X[feat])[:, 1], index=df.index)


@st.cache_data
def load_scaler_stats() -> dict[str, tuple[float, float]]:
    """전처리기(StandardScaler)의 평균·표준편차 — 표준화 값을 실제 단위로 오가는 데 사용."""
    import joblib
    path = ROOT / "artifacts" / "preprocessor_with_retention.joblib"
    if not path.exists():
        return {}
    pre = joblib.load(path)
    stats: dict[str, tuple[float, float]] = {}
    for _, trans, cols in pre.transformers_:
        scaler = getattr(trans, "named_steps", {}).get("scaler")
        if scaler is not None and hasattr(scaler, "mean_"):
            for c, m, s in zip(cols, scaler.mean_, scaler.scale_):
                stats[c] = (float(m), float(s))
    return stats



@st.cache_resource
def load_cluster_bundle() -> dict:
    """저장된 K-Means 전처리·모델 번들을 불러온다."""
    import joblib

    if not CLUSTER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"군집 모델을 찾을 수 없습니다: {CLUSTER_MODEL_PATH}. "
            "03_cluster 노트북을 실행해 models/kmeans_pipeline.joblib을 생성하세요."
        )
    return joblib.load(CLUSTER_MODEL_PATH)


def predict_customer_segment(new_customer: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """학습 당시와 동일한 전처리를 적용해 신규 고객의 군집을 예측한다."""
    features = bundle["feature_names"]
    missing = [feature for feature in features if feature not in new_customer.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    transformed = pd.DataFrame(
        bundle["imputer"].transform(new_customer[features]),
        columns=features,
        index=new_customer.index,
    )
    transformed = transformed.clip(
        lower=pd.Series(bundle["lower_bounds"]),
        upper=pd.Series(bundle["upper_bounds"]),
        axis=1,
    )
    transformed_scaled = bundle["scaler"].transform(transformed)
    labels = bundle["kmeans"].predict(transformed_scaled)

    output = new_customer.copy()
    output["Cluster"] = labels
    output["ClusterName"] = pd.Series(labels, index=output.index).map(bundle["cluster_names"])
    return output


def cluster_default_values(bundle: dict) -> dict[str, float]:
    """입력 폼 기본값으로 군집 학습 당시 중앙값을 사용한다."""
    features = bundle["feature_names"]
    statistics = getattr(bundle["imputer"], "statistics_", None)
    if statistics is None:
        return {feature: 0.0 for feature in features}
    return {feature: float(value) for feature, value in zip(features, statistics)}

def get_bundle():
    try:
        return load_model(), None
    except FileNotFoundError as e:
        return None, str(e)


st.title("📉 고객 이탈 예측")
t1, t2, t3, t4 = st.tabs(["현황", "모델 성능", "이탈 예측", "고객 세그먼트"])

# ---------- 화면 1. 고객 현황 ----------
with t1:
    df = load_customers()
    if df is None:
        st.error("고객 데이터가 없습니다. data/raw 또는 data/interim에 CSV를 준비하세요.")
    else:
        y = df["_churn"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("전체 고객", f"{len(df):,}명")
        c2.metric("이탈 고객", f"{int(y.sum()):,}명")
        c3.metric("유지 고객", f"{int((1 - y).sum()):,}명")
        c4.metric("이탈률", f"{y.mean():.1%}")

        left, right = st.columns(2)
        with left:
            st.subheader("유지 / 이탈 분포")
            st.bar_chart(y.map({0: "유지", 1: "이탈"}).value_counts())
        with right:
            st.subheader("특성 구간별 이탈률")
            options = {k: v for k, v in KEY_FEATURES.items()
                       if k in df.columns and pd.api.types.is_numeric_dtype(df[k])}
            sel = st.selectbox("Feature 선택", list(options), format_func=options.get)
            binned = pd.qcut(df[sel], q=5, duplicates="drop")
            rate = y.groupby(binned, observed=True).mean()
            rate.index = rate.index.map(lambda iv: f"{iv.left:g}~{iv.right:g}")
            st.bar_chart(rate)
            st.caption("구간은 5분위 기준. 막대가 오른쪽으로 갈수록 값이 큰 고객군.")

        st.subheader("핵심 인사이트 (EDA)")
        st.markdown(
            "- **단말 사용일수(CurrentEquipmentDays)가 길수록 이탈률 상승** — "
            "오래된 단말을 쓰는 고객은 교체 시점에 타사로 이동할 위험이 크다.\n"
            "- **가입 초기(MonthsInService 낮음) 고객의 이탈률이 높음** — 온보딩 구간 관리가 필요하다.\n"
            "- **리텐션팀 통화 경험 고객의 이탈률(45.0%)이 미경험(28.2%)보다 높음** — "
            "이미 이탈 의사를 보인 신호이므로 모델은 retention 포함/제외 두 버전으로 비교해 확정했다."
        )

        bundle, err = get_bundle()
        st.subheader("위험도 구간별 고객 수")
        sdf = load_scoring_data()
        if bundle is None:
            st.warning(err)
        elif sdf is None:
            st.warning("전처리 데이터(data/interim)가 없어 위험도 구간을 계산할 수 없습니다.")
        else:
            proba = score_customers(sdf)
            grades = pd.cut(proba, bins=[0, 0.4, 0.7, 1.0],
                            labels=["저위험", "중위험", "고위험"], include_lowest=True)
            counts = grades.value_counts().reindex(["고위험", "중위험", "저위험"])
            g1, g2, g3 = st.columns(3)
            for col, grade in zip((g1, g2, g3), counts.index):
                col.metric(f"{GRADE_ICON[grade]} {grade}", f"{int(counts[grade]):,}명")
            st.bar_chart(counts)

# ---------- 화면 2. 모델 성능 ----------
with t2:
    st.subheader("모델별 성능 비교 (validation)")
    long_path = MODELING_DIR / "model_metrics_long.csv"
    if long_path.exists():
        cmp = pd.read_csv(long_path)
        cmp = cmp[cmp["feature_set"] == "with_retention"]
        show = cmp[["model", "roc_auc", "accuracy", "precision", "recall", "f1"]]
        st.dataframe(
            show.sort_values("roc_auc", ascending=False).reset_index(drop=True)
                .style.format({c: "{:.4f}" for c in show.columns[1:]})
                .highlight_max(subset=["roc_auc"], color="#1a7a3a"),
            use_container_width=True,
        )
        st.caption("Retention 포함 feature set 기준. 자세한 비교는 reports/modeling_report.md 참고.")
    else:
        st.info("노트북 02를 실행하면 모델 비교표가 여기에 표시됩니다.")

    bundle, err = get_bundle()
    if bundle is None:
        st.warning(err)
    else:
        m = bundle.get("metrics", {})
        st.subheader(f"최종 모델: {m.get('model_name', 'HistGradientBoosting')}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ROC-AUC", f"{m.get('roc_auc', 0):.4f}")
        c2.metric("PR-AUC", f"{m.get('pr_auc', 0):.4f}")
        c3.metric("Precision", f"{m.get('precision', 0):.4f}")
        c4.metric("Recall", f"{m.get('recall', 0):.4f}")
        c5.metric("F1", f"{m.get('f1', 0):.4f}")
        st.caption(f"평가: {m.get('eval_split', 'validation')} · 임계값 {bundle['threshold']}")

        cm = m.get("confusion_matrix")
        if cm:
            st.subheader("Confusion Matrix (validation)")
            st.dataframe(pd.DataFrame(
                [[f"{cm['tn']:,} (TN)", f"{cm['fp']:,} (FP)"],
                 [f"{cm['fn']:,} (FN)", f"{cm['tp']:,} (TP)"]],
                index=["실제 유지", "실제 이탈"], columns=["예측 유지", "예측 이탈"],
            ), use_container_width=True)
            st.caption(
                f"실제 이탈 {cm['fn'] + cm['tp']:,}명 중 {cm['tp']:,}명 탐지 "
                f"(Recall {m.get('recall', 0):.1%}). FP {cm['fp']:,}명은 캠페인 비용과 "
                "직결되므로 임계값으로 조정한다."
            )

    st.subheader("중요 Feature와 해석")
    fi = MODELING_DIR / "presentation_04_feature_importance.png"
    if fi.exists():
        st.image(str(fi), use_container_width=True)
    st.markdown(
        "- 단말 사용일수·가입 개월 수·통화량 증감 등 **이용 행태 변화** 지표가 상위권이다.\n"
        "- 리텐션팀 통화는 이탈 의사가 이미 드러난 신호로, 예측 기여도가 높지만 인과가 아니다.\n"
        "- 중요도는 예측 기여도이며 인과관계로 해석하지 않는다 (modeling_report 9장)."
    )
    roc = MODELING_DIR / "presentation_02_roc_curve.png"
    if roc.exists():
        with st.expander("ROC Curve 보기"):
            st.image(str(roc), use_container_width=True)

# ---------- 화면 3. 개별 고객 이탈 예측 ----------
with t3:
    bundle, err = get_bundle()
    if bundle is None:
        # st.stop()을 쓰면 이 탭뿐 아니라 스크립트 전체(이후 탭 포함)가 멈춘다.
        # 이 탭만 건너뛰고 다른 탭은 계속 렌더링되도록 if/else로 감싼다.
        st.warning(err)
    else:
        df = load_scoring_data()
        if df is None:
            st.error("입력폼 기준값을 만들 전처리 데이터(data/interim)가 없습니다.")
        else:
            proba_all = score_customers(df)
            examples = {
                "예시: 고위험 고객": int(proba_all.idxmax()),
                "예시: 중위험 고객": int((proba_all - 0.55).abs().idxmin()),
                "예시: 저위험 고객": int(proba_all.idxmin()),
                "직접 입력 (전체 중앙값에서 시작)": None,
            }
            choice = st.selectbox("예시 고객 선택", list(examples))
            idx = examples[choice]
            base = (df.loc[idx] if idx is not None else df.median(numeric_only=True)).copy()

            feat_names = bundle["feature_names"]
            form = {c: base.get(c) for c in feat_names}

            stats = load_scaler_stats()  # 실제 단위 <-> 표준화 값 변환용
            edited: dict[str, float] = {}
            with st.form("predict"):
                st.markdown("**주요 고객 정보** — 값을 바꾸면 예측 확률이 실제로 바뀝니다.")
                cols = st.columns(2)
                for i, (feat, label) in enumerate(KEY_FEATURES.items()):
                    if feat not in df.columns:
                        continue
                    v = base.get(feat)
                    v = 0.0 if pd.isna(v) else float(v)
                    if feat in stats:  # 화면에는 실제 단위로 보여준다
                        m, s = stats[feat]
                        v = v * s + m
                    with cols[i % 2]:
                        edited[feat] = st.number_input(f"{label} ({feat})", value=round(v, 1))
                c1, c2 = st.columns(2)
                with c1:
                    retention = st.selectbox(
                        "리텐션팀 통화 경험 (MadeCallToRetentionTeam)", ["No", "Yes"],
                        index=int(base.get("MadeCallToRetentionTeam_Yes", 0) == 1),
                    )
                with c2:
                    current = next(
                        (c for c in CREDIT_COLS if base.get(c, 0) == 1), CREDIT_COLS[2])
                    credit = st.selectbox(
                        "신용 등급 (CreditRating)",
                        CREDIT_COLS, index=CREDIT_COLS.index(current),
                        format_func=lambda c: c.split("_", 1)[1],
                    )
                go = st.form_submit_button("예측 실행", type="primary")

            if go:
                for feat, v in edited.items():  # 입력받은 실제 단위를 다시 표준화 값으로 변환
                    if feat in stats:
                        m, s = stats[feat]
                        form[feat] = (v - m) / s
                    else:
                        form[feat] = v
                form["MadeCallToRetentionTeam_Yes"] = float(retention == "Yes")
                form["MadeCallToRetentionTeam_No"] = float(retention == "No")
                for c in CREDIT_COLS:
                    form[c] = float(c == credit)
                try:
                    r = predict_one(form, bundle)
                except Exception as e:  # 입력값 이상 등
                    st.error(f"예측에 실패했습니다. 입력값을 확인하세요: {e}")
                else:
                    grade = r["risk_grade"]
                    c1, c2 = st.columns([1, 2])
                    c1.metric("이탈 확률", f"{r['churn_probability']:.1%}")
                    c1.metric("위험 등급", f"{GRADE_ICON[grade]} {grade}")
                    c2.progress(min(float(r["churn_probability"]), 1.0))
                    c2.info(f"제안 유지 활동: {r['suggested_action']}")
                    st.caption(
                        f"임계값 {bundle['threshold']} 기준 예측 = "
                        f"{'이탈' if r['churn_prediction'] else '유지'} · "
                        "입력하지 않은 나머지 특성은 선택한 예시 고객 값을 사용합니다."
                    )

# ---------- 화면 4. 신규 고객 세그먼트 분류 ----------
with t4:
    st.subheader("🧩 고객 세그먼트 안내")

    st.markdown("### 먼저, 고객은 다음 5가지 세그먼트로 분류됩니다.")

    cols = st.columns(5)
    for col, key in zip(cols, [
        "Loyal Long-term Customers",
        "Premium Heavy Users",
        "Multi-line Customers",
        "Regular Customers",
        "High-Maintenance Customers",
    ]):
        guide = SEGMENT_GUIDE[key]
        with col:
            st.markdown(f"## {guide['icon']}")
            st.markdown(f"**{guide['title']}**")
            st.caption(guide["description"])

    st.divider()

    st.subheader("🧩 신규 고객 세그먼트 분류")
    st.write(
        "고객의 이용·계약 정보를 입력하면 저장된 K-Means 모델이 고객군을 분류하고, "
        "해당 고객군에 적합한 대응 전략을 안내합니다."
    )
    st.caption(
        "군집 번호는 우열이나 위험 순위가 아닙니다. Churn은 군집 학습에 사용되지 않았으며, "
        "이 화면은 고객 행동 특성에 따른 관리 전략을 제안합니다."
    )

    try:
        cluster_bundle = load_cluster_bundle()
    except (FileNotFoundError, KeyError, ValueError) as e:
        st.error(str(e))
    else:
        features = cluster_bundle["feature_names"]
        defaults = cluster_default_values(cluster_bundle)
        entered: dict[str, float] = {}

        with st.form("cluster_predict_form"):
            st.markdown("**고객 정보 입력**")
            left, right = st.columns(2)
            for i, feature in enumerate(features):
                label = CLUSTER_FEATURE_LABELS.get(feature, feature)
                default = defaults.get(feature, 0.0)
                target_col = left if i % 2 == 0 else right
                with target_col:
                    entered[feature] = st.number_input(
                        f"{label} ({feature})",
                        min_value=0.0,
                        value=round(float(default), 1),
                        step=1.0,
                        help="모델 학습 데이터의 중앙값을 기본값으로 사용합니다.",
                    )
            classify = st.form_submit_button("고객군 분류", type="primary", use_container_width=True)

        if classify:
            new_customer = pd.DataFrame([entered], columns=features)
            try:
                result = predict_customer_segment(new_customer, cluster_bundle).iloc[0]
            except Exception as e:
                st.error(f"고객군 분류에 실패했습니다. 입력값과 모델 파일을 확인하세요: {e}")
            else:
                cluster_no = int(result["Cluster"])
                segment_name = str(result["ClusterName"])
                guide = SEGMENT_GUIDE.get(
                    segment_name,
                    {
                        "title": segment_name,
                        "icon": "👤",
                        "description": "저장된 군집 프로파일을 확인해 고객 특성을 해석하세요.",
                        "actions": ["군집 프로파일에 맞는 고객 관리 전략을 수립하세요."],
                    },
                )

                st.divider()
                result_col, detail_col = st.columns([1, 2])
                with result_col:
                    st.metric("분류된 Cluster", f"Cluster {cluster_no}")
                    st.success(f"{guide['icon']} **{guide['title']}**")
                    st.caption(segment_name)

                with detail_col:
                    st.markdown("### 고객군 특징")
                    st.info(guide["description"])
                    st.markdown("### 추천 고객 대응")
                    for action in guide["actions"]:
                        st.markdown(f"- {action}")

                with st.expander("입력한 고객 정보 확인"):
                    display_input = pd.DataFrame(
                        {
                            "항목": [CLUSTER_FEATURE_LABELS.get(f, f) for f in features],
                            "Feature": features,
                            "입력값": [entered[f] for f in features],
                        }
                    )
                    st.dataframe(display_input, use_container_width=True, hide_index=True)

                st.warning(
                    "이 결과는 고객 유형 분류이며 이탈 예측 결과가 아닙니다. "
                    "실제 유지 캠페인에서는 3번 탭의 이탈 위험도와 함께 활용하세요."
                )

