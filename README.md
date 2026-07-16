# Cell2Cell 고객 이탈 예측

> 발표 2026-07-22 · **통신사 고객유지팀**이 이탈 방어 캠페인 대상자를 선정할 수 있도록
> 고객 요금·통화·단말 데이터로 **이탈 여부(Churn, Yes=1)** 를 예측하고 유지 활동을 제안한다.

- 데이터: [Cell2Cell (Kaggle)](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom) — 고객 51,047명 × 58컬럼, 이탈률 28.8%
- 작업 흐름: [docs/pipeline.png](docs/pipeline.png) · 팀 간 공유 시점: [docs/handoff.png](docs/handoff.png) · 상세 계획: [docs/team_plan.md](docs/team_plan.md)

## 시작하기

```bash
git clone https://github.com/skn-2nd-churn-pred/skn-2nd-project-churnpred.git
pip install -r requirements.txt
# 데이터 2개를 내려받아 data/raw/ 에 배치 (Git에는 올라가지 않음):
#   data/raw/cell2celltrain.csv   — 학습용 (Target 포함)
#   data/raw/cell2cellholdout.csv — ⚠️ 라벨 없음: Test 사용 금지, Streamlit 시연용만
```

모델 학습 후 시연:
```bash
streamlit run streamlit_app/app.py
```

## 작업 순서 (3인 분업)

| 담당 | 노트북 | 산출물 |
| --- | --- | --- |
| A. 전처리·EDA | `notebooks/01_eda.ipynb` | `reports/report.md` A파트 + **`src/clean.py` 정제 규칙 확정** |
| B. 모델링 | `notebooks/02_model.ipynb` | B파트 + `models/churn_pipeline.joblib` + `artifacts/metrics.csv` |
| C. 세그먼트(보조) | `notebooks/03_cluster.ipynb` | C파트 (Must 완료 후 착수, 예측 로직 미포함) |

## 골격이 잡아주는 것 (직접 짜면 틀리기 쉬운 부분)

- **`src/clean.py`** — 공통 정제 규칙(HandsetPrice 'Unknown', AgeHH1=0, 음수값). A가 관리하고 노트북 02·03이 import → **모델링과 클러스터가 항상 같은 기준으로 시작**. imputation은 여기서 하지 않음(Train에만 fit).
- **`src/predict.py`** — 설정 로드 + 추론 계약. 노트북 02가 이 형식으로 저장하면 앱이 그대로 동작:
  ```python
  joblib.dump({"pipeline": pipe, "threshold": thr, "feature_names": [...]},
              "models/churn_pipeline.joblib")
  ```
- **`streamlit_app/app.py`** — 단일 파일 3탭(현황·성능·예측). 저장 모델 로드(재학습 X), 입력폼은 원본 CSV에서 자동 생성 → 학습/화면 Feature 일치.

노트북 02에는 규율(먼저 분리 / Train에만 fit / 임계값은 Validation / Test 한 번 / Pipeline 저장)과 시작 코드·`TODO`가 있습니다. **최종 모델은 근거를 대며 직접 선정**(자동 아님).

## 이 데이터에서 주의할 것

- **리텐션 3컬럼**(`RetentionCalls`·`RetentionOffersAccepted`·`MadeCallToRetentionTeam`) — 누수 의심(통화 고객 이탈률 45% vs 28%). 포함/제외 실험 후 근거와 함께 결정.
- **holdout은 Test가 아님** — 라벨이 전부 결측. Train/Val/Test는 train CSV에서 3분할.
- `CustomerID` 제외, `ServiceArea`(고유값 747) 그룹화 필요.

## 구조

```
config.yaml             # 경로·Target (Churn / Yes=1)
src/clean.py            # ★ 공통 정제 규칙 (A 관리, B·C import)
src/predict.py          # 설정 로드 + 추론 (앱·노트북 공유)
notebooks/01_eda 02_model 03_cluster    # ★ 실제 작업 (담당자별)
streamlit_app/app.py    # 3탭 시연
docs/                   # team_plan · project_plan · pipeline.png · handoff.png
reports/report.md       # 결과서 (A/B/C파트)
data/raw  models  artifacts             # 원본(Git 제외) · 모델 · 지표
```

| 필수 산출물 | 위치 |
| --- | --- |
| 전처리·학습·세그먼트 결과서 | `reports/report.md` (A/B/C파트) |
| 최종 모델 | `models/churn_pipeline.joblib` (노트북 02에서 저장) |
| Streamlit 시연 | `streamlit_app/app.py` |
| 발표자료 | `presentation.pdf` |

> 발표 전 체크: README·발표자료의 성능 수치 = `artifacts/metrics.csv` 실제 수치와 일치할 것.

## 커밋 메시지 규칙

`YYYYMMDD-작업자-작업내용` 형식으로 작성한다.

```
20260716-홍길동-RandomForest, XGBoost 비교 추가
20260716-김영희-HandsetPrice Unknown 결측 처리 수정
20260717-이철수-군집 프로필 결과 report.md C파트에 반영
```

- 날짜는 커밋하는 당일 날짜(`YYYYMMDD`)
- 작업자는 본인 이름
- 작업내용은 한 줄로 간단히, 무엇을 했는지 알아볼 수 있게
