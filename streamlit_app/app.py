"""Streamlit 시연 (단일 파일, 3개 탭).

실행: streamlit run streamlit_app/app.py
- 저장된 최종 모델을 불러온다 (앱에서 재학습하지 않는다).
- 입력을 바꾸면 확률도 실제로 바뀐다.
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
tcol, pos = cfg["target"]["column"], str(cfg["target"]["positive_label"]).strip().lower()


@st.cache_data
def raw():
    d = cfg["data"]
    p = resolve_path(d["raw_path"])
    if not p.exists():
        return None
    df = pd.read_csv(p, encoding=d.get("encoding", "utf-8"), sep=d.get("sep", ","))
    df["_churn"] = df[tcol].astype(str).str.strip().str.lower().eq(pos).astype(int)
    return df


st.title("📉 고객 이탈 예측")
t1, t2, t3 = st.tabs(["현황", "모델 성능", "이탈 예측"])

# ---------- 현황 ----------
with t1:
    df = raw()
    if df is None:
        st.error(f"원본 데이터가 없습니다: {cfg['data']['raw_path']}")
    else:
        y = df["_churn"]
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 고객", f"{len(df):,}")
        c2.metric("이탈 고객", f"{int(y.sum()):,}")
        c3.metric("이탈률", f"{y.mean():.1%}")
        cats = [c for c in df.columns if c not in ("_churn", tcol)
                and not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 15]
        if cats:
            col = st.selectbox("특성별 이탈률", cats)
            st.bar_chart(y.groupby(df[col]).mean().sort_values(ascending=False))

# ---------- 모델 성능 ----------
with t2:
    mp = resolve_path(cfg["output"]["metrics_path"])
    if mp.exists():
        st.dataframe(pd.read_csv(mp, index_col=0), use_container_width=True)
    else:
        st.info("노트북 02에서 학습하면 모델 비교표가 여기에 표시됩니다.")

# ---------- 이탈 예측 ----------
with t3:
    try:
        bundle = load_model()
    except FileNotFoundError as e:
        st.warning(str(e))
        st.stop()
    df = raw()
    if df is None:
        st.error("입력폼 기준값을 만들 원본 CSV가 없습니다.")
        st.stop()

    form = {}
    with st.form("f"):
        cols = st.columns(2)
        for i, feat in enumerate(bundle["feature_names"]):
            if feat not in df.columns:
                continue
            s = df[feat]
            with cols[i % 2]:
                if pd.api.types.is_numeric_dtype(s):
                    form[feat] = st.number_input(feat, value=float(s.median()))
                else:
                    form[feat] = st.selectbox(feat, sorted(s.dropna().astype(str).unique()))
        go = st.form_submit_button("예측 실행", type="primary")

    if go:
        r = predict_one(form, bundle)
        grade = r["risk_grade"]
        icon = {"고위험": "🔴", "중위험": "🟡", "저위험": "🟢"}[grade]
        c1, c2 = st.columns([1, 2])
        c1.metric("이탈 확률", f"{r['churn_probability']:.1%}")
        c1.metric("위험 등급", f"{icon} {grade}")
        c2.progress(min(float(r["churn_probability"]), 1.0))
        c2.info(r["suggested_action"])
