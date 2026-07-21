# 팀 작업 계획서 — Cell2Cell 고객 이탈 예측 (4인)

> 발표: 2026-07-22 (화) · 작성: 2026-07-15
> 데이터: `cell2celltrain.csv` (51,047명, Target `Churn` 포함)

## 0. 프로젝트 한 문장

> **통신사 고객유지팀 담당자**가 **이탈 방어 캠페인 대상자 선정**을 할 수 있도록
> **고객 요금·통화·단말 데이터**로 **한 달 내 이탈 여부(Churn, Yes=1)** 를 예측하고
> **위험 등급별 유지 활동**을 제안한다.

- 이탈률 약 29% (Yes 14,711 / No 36,336) — 극단적 불균형 아님
- 우선 지표: **Recall** (놓친 이탈 고객 FN의 비용 > 잘못 경고한 FP의 비용) + F1·PR-AUC 병행
- 스냅샷 데이터(1고객 1행)이므로 분리는 `stratify=y` 3분할

### 기능 범위 (가이드 Must/Should/Could)
- **Must**: 이탈 예측(B/D), Streamlit 4탭
- **Should**: 임계값 조정, 리텐션 컬럼 포함/제외 근거
- **Could**: **고객 세그먼트 군집분석** — 이탈 예측과 **독립된 보조 분석**. Must 항목이 끝난 뒤 진행하며, 일정이 밀리면 가장 먼저 축소 대상

---

## 1. 역할 분배 (4인)

> 가이드 원칙: 역할을 완전히 분리하지 않고, **각 작업에 담당자 1명 + 검토자 1명**을 둔다.

| 역할 | 담당 | 주요 책임 |
| --- | --- | --- |
| **A. PM / 요구사항·문서** | 팀원 1 | 범위·일정 관리, `docs/project_plan.md` 완성, README·발표자료 통합, 산출물 최종 점검 |
| **B. 데이터 / 전처리** | 팀원 2 | Data Card 작성, 품질 점검(결측·이상값), 전처리 Pipeline 설계, `reports/preprocessing_report.md` |
| **C. EDA / Feature** | 팀원 3 | 노트북 01 (시각화 5~7개 + 인사이트 3개), Feature 결정 근거, 누수 의심 컬럼 판정 |
| **D. 모델링 / 평가** | 팀원 4 | 노트북 02 (분리→비교→선정→Test→저장), 임계값 결정, `reports/modeling_report.md` |
| **Streamlit / 통합** | A 주도 + D 지원 | 4탭 앱 연결, 저장 모델 호출 검증, 시연 리허설 |
| **군집분석 (보조, Could)** | C | 노트북 03 (`03_cluster.ipynb`), `reports/clustering_report.md`. **Must 완료 후 착수** |

### 상호 검토(리뷰) 짝
| 작업 | 담당자 | 검토자 |
| --- | --- | --- |
| 전처리 Pipeline | B | D (누수·fit 범위 검증) |
| EDA·Feature 결정 | C | B |
| 모델 비교·선정 | D | C (동일 조건 비교 확인) |
| Streamlit·문서 | A | 전원 (통합일 리허설) |
| 군집분석 | C | B (Target 미사용 여부·해석 과잉 확인) |

---

## 2. 이 데이터에서 반드시 결정/처리할 것 (분석 결과 반영)

| # | 항목 | 처리 방침 | 담당 |
| --- | --- | --- | --- |
| 1 | `CustomerID` | 식별자 → Feature 제외 (`config.yaml` `id_columns`) | B |
| 2 | `RetentionCalls` `RetentionOffersAccepted` `MadeCallToRetentionTeam` | **누수 의심**(이탈 조짐 → 리텐션팀 접촉의 순환 논리 소지). 포함/제외 **두 버전 실험** 후 근거와 함께 결정, 결과서에 기록 | C 판정 → D 실험 |
| 3 | `ServiceArea` (고유값 747) | 원-핫 불가 → 앞 3자리(지역) 추출 또는 빈도 상위 N + Other 그룹화 | B |
| 4 | 결측치 | 수치형 0.3~1.8% → median, `AgeHH1/2` 1.8% → median 또는 "미상" 플래그 | B |
| 5 | `HandsetPrice` | 문자형에 'Unknown' 포함 → 숫자 변환 + Unknown 처리 확인 | B |
| 6 | 클래스 비율 | 29% — SMOTE 없이 `class_weight='balanced'` 우선, 필요 시에만 추가 실험 | D |

---

## 3. 일정 (D-7)

| 날짜 | 마일스톤 | 산출물 | 담당 |
| --- | --- | --- | --- |
| **7/15 (수)** | 데이터 확정·환경 세팅. `config.yaml` 수정, `data/raw/` 배치, 저장소 클론·브랜치 생성 | config 반영, Data Card 초안 | 전원 |
| **7/16 (목)** | 노트북 01: 품질 점검 + EDA. 누수 의심 컬럼(위 표 #2) 판정 | 시각화 5~7개, 인사이트 3개 | C (검토 B) |
| **7/17 (금)** | 전처리 Pipeline 확정 + 노트북 02 전반: 3분할, Dummy~RF 기준 모델 | 분할 코드, 기준 성능표 | B·D |
| **7/18 (토)** | 부스팅 추가, 리텐션 컬럼 포함/제외 비교, 최종 모델 선정 + 임계값(Validation) | 모델 비교표, 선정 근거 | D (검토 C) |
| **7/19 (일)** | **최종 Test 1회** → 모델 저장(`joblib`) → Streamlit 연결 확인. **(C, 시간 되면) 노트북 03 군집분석 착수** | `churn_pipeline.joblib`, metrics.csv | D → A / C |
| **7/20 (월)** | 통합일: 전체 실행 리허설(클론→설치→실행), 결과서 완성. 군집분석은 **완료 시에만** 반영 | `reports/preprocessing_report.md`·`modeling_report.md` 완성 | 전원 |
| **7/21 (화)** | 발표자료 작성, README 수치 일치 확인, 시연 리허설 2회 | `presentation.pdf` | A (전원 리뷰) |
| **7/22 (수)** | **발표** | — | 전원 |

---

## 4. Git 운영 규칙

- `main` 직접 작업 금지 → `feature/eda`, `feature/preprocess`, `feature/model`, `feature/streamlit`, `feature/cluster`
- 결과서는 **담당자별 파일 분리**(`preprocessing_report.md`/`modeling_report.md`/`clustering_report.md`)로 관리 — 같은 파일을 여러 명이 동시에 채우지 않도록 머지 충돌 방지
- 정제 규칙은 **`src/clean.py`로만 공유** (A가 작성·수정, B·C는 import만 — 노트북에 복붙 금지)
- 하루 1회 이상 main에 통합하고 **전체 실행 확인** (7/20은 필수 통합일)
- 커밋 금지: `data/raw/*.csv`(대용량 원본), `.env`, 가상환경 — `.gitignore`가 이미 차단
- README 성능 수치 = 실제 `metrics.csv` 수치 일치 확인 후 발표

## 5. 완료 기준 체크리스트 (발표 전 전원 확인)

- [ ] 저장 모델을 새 프로세스에서 로드해 신규 고객 1명 예측 가능
- [ ] `streamlit run streamlit_app/app.py` 실행, 입력 바꾸면 확률이 실제로 변함
- [ ] Test는 최종 1회만 평가했고, 임계값은 Validation에서 결정함
- [ ] 리텐션 컬럼 포함/제외 실험 결과와 결정 근거가 결과서에 있음
- [ ] Train/Validation/Test를 `cell2celltrain.csv`에서 분리하고 각 역할을 지켰음
- [ ] README 2~3개 명령으로 설치→실행 재현 가능
- [ ] (선택) 군집분석을 했다면 Target을 학습에 쓰지 않았고, 군집별 이탈률로 사후 검증했음
