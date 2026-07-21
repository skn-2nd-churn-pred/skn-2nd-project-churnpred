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
| A. 전처리·EDA | `notebooks/01_eda.ipynb`, `00_data_check.ipynb` | `reports/preprocessing_report.md` + **`src/clean.py`/`src/data.py` 정제 규칙 확정** |
| B. 모델링 | `notebooks/02_model.ipynb` | `reports/modeling_report.md` + `models/*_final.joblib` |
| C. 세그먼트(보조) | `notebooks/03_cluster.ipynb` | `reports/clustering_report.md` (Must 완료 후 착수, 예측 로직 미포함) |

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
config.yaml                  # 경로·Target·최종 모델·임계값 (단일 출처)
src/
  clean.py                   # 공통 정제 규칙 (노트북 02·03이 import)
  data.py                    # 로드·검증·3분할·전처리 Pipeline
  predict.py                 # 설정 로드 + 추론 (앱·노트북·테스트 공유)
  evaluate_test.py           # 최종 Test 평가 (1회만 실행)
  save_metadata.py           # 모델 메타데이터 생성
notebooks/                   # 00_data_check · 01_eda · 02_model · 03_cluster
streamlit_app/app.py         # 4탭 시연 (현황·모델성능·이탈예측·고객세그먼트)
docs/                        # team_plan · project_plan · pipeline.png · handoff.png
reports/                     # preprocessing · modeling · clustering · test_report.md
tests/test_inference.py      # 모델 로드·신규 고객 예측 검증
data/  models/  artifacts/   # 원본·중간 데이터(Git 제외) · 모델 · 지표
```

### 가이드 표준 구조와의 차이

프로젝트 가이드의 권장 구조를 기준으로, 이 프로젝트에서 다르게 둔 부분과 이유입니다.

| 가이드 | 이 프로젝트 | 이유 |
| --- | --- | --- |
| `docs/requirements.md`<br>`data_dictionary.md`<br>`validation_plan.md` | `docs/project_plan.md`<br>`reports/preprocessing_report.md` | 요구사항·Data Card·검증 계획을 두 문서에 통합. 내용은 모두 포함되어 있고, 파일을 쪼개면 같은 내용을 이중 관리하게 되어 합쳤습니다. |
| `src/features.py`<br>`src/train_ml.py` | `src/data.py`<br>`notebooks/02_model.ipynb` | Feature 생성은 전처리와 분리되지 않아 `data.py`에 통합했습니다. 모델 학습은 비교·선정 과정을 보여줘야 해서 노트북에서 수행합니다(스크립트로 옮기면 노트북과 이중 관리). |
| `src/evaluate.py` | `src/evaluate_test.py` | 이름을 좁혀서 **Test 1회 평가 전용**임을 드러냈습니다. 임계값 탐색 코드를 의도적으로 넣지 않아, 구조적으로 Test 오염을 막습니다. |
| `streamlit_app/pages/` 3개 | `streamlit_app/app.py` 단일 파일 4탭 | 탭이 서로 같은 모델·데이터를 공유해 한 파일이 더 단순합니다. 군집 세그먼트 탭이 추가되어 4탭입니다. |
| `reports/training_report.md` | `reports/modeling_report.md`<br>`clustering_report.md`<br>`test_report.md` | 담당자별로 파일을 나눠 같은 파일을 동시에 편집하는 머지 충돌을 막았습니다. |
| 노트북 `01`~`03` | `00`~`03` (4개) | 데이터 점검과 EDA를 분리했고, 군집분석 노트북이 추가됐습니다. |

| 필수 산출물 | 위치 |
| --- | --- |
| 데이터 전처리 결과서 | `reports/preprocessing_report.md` |
| 모델 학습 결과서 | `reports/modeling_report.md` (+ `test_report.md` 최종 Test) |
| 학습된 최종 모델 | `models/histgradientboosting_without_retention_final.joblib`<br>메타데이터: `artifacts/model_metadata.json` |
| Streamlit 시연 | `streamlit_app/app.py` |
| 발표자료 | `presentation.pdf` |

### 검증·산출물 재생성

```bash
python tests/test_inference.py    # 모델 로드·신규 고객 예측 검증 (pytest 없이도 실행)
python src/save_metadata.py       # artifacts/model_metadata.json 갱신
python src/evaluate_test.py       # 최종 Test 평가 — 한 번만 실행
```

> 발표 전 체크: README·발표자료의 성능 수치 = `reports/test_report.md`·`artifacts/model_metadata.json`의 실제 수치와 일치할 것.

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
