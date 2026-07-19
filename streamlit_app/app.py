"""Streamlit 시연 (단일 파일, 3개 탭).

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


def get_bundle():
    try:
        return load_model(), None
    except FileNotFoundError as e:
        return None, str(e)


st.title("📉 고객 이탈 예측")
t1, t2, t3 = st.tabs(["현황", "모델 성능", "이탈 예측"])

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
        st.warning(err)
        st.stop()
    df = load_scoring_data()
    if df is None:
        st.error("입력폼 기준값을 만들 전처리 데이터(data/interim)가 없습니다.")
        st.stop()

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
            st.stop()
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
