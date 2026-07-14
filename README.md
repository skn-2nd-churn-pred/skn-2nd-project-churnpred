# 고객 이탈 예측 — 최소 골격

> 발표 2026-07-22 · 전처리·EDA·모델링은 **노트북에서 직접** 진행합니다.

## 작업 순서
1. 실제 CSV를 `data/raw/` 에 넣고 `config.yaml`(경로·target) 수정
2. `notebooks/01_eda.ipynb` — 점검·EDA
3. `notebooks/02_model.ipynb` — 분리→전처리→비교→선정→Test→**저장**
4. 시연: `streamlit run streamlit_app/app.py`

```bash
pip install -r requirements.txt
```

## 골격이 주는 것 (직접 짜면 틀리기 쉬운 부분만)
- **`src/predict.py`** — 설정 로드 + 추론 계약. 노트북 02가 이 형식으로 저장하면 앱이 그대로 동작:
  ```python
  joblib.dump({"pipeline": pipe, "threshold": thr, "feature_names": [...]},
              "models/churn_pipeline.joblib")
  ```
- **`streamlit_app/app.py`** — 단일 파일, 3탭(현황·성능·예측). 저장 모델 로드(재학습 X), 입력폼은 원본 CSV에서 자동 생성 → 학습/화면 Feature 일치.
- **문서 템플릿** — `docs/project_plan.md`(요구사항·데이터·검증), `reports/report.md`(전처리·학습 결과).

노트북 02에는 규율(먼저 분리 / Train에만 fit / Test 한 번 / Pipeline 저장 / 누수 제거)과 시작 코드·`TODO`가 있습니다. **최종 모델은 근거 대며 직접 선정**(자동 아님).

## 구조
```
config.yaml
src/predict.py          # 설정+추론 (유일 모듈)
notebooks/01_eda.ipynb 02_model.ipynb   # ★ 실제 작업
streamlit_app/app.py    # 3탭 시연
docs/project_plan.md  reports/report.md
data/raw  models  artifacts
```

| 산출물 | 위치 |
| --- | --- |
| 전처리·학습 결과서 | `reports/report.md` |
| 최종 모델 | `models/churn_pipeline.joblib` (노트북 02) |
| 시연 | `streamlit_app/app.py` |
| 발표자료 | `presentation.pdf` |
