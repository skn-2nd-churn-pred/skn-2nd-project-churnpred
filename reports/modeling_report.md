# 고객 이탈 예측 모델링 상세 리포트

## 1. 프로젝트 목적

고객의 통화·요금·단말기·인구통계 정보를 활용해 고객 이탈 여부(`Churn`)를 예측한다.
특히 이탈 방어 활동(Retention) 정보를 포함했을 때와 제외했을 때의 성능 차이를 비교했다.

## 2. 데이터 및 실험 설계

| 구분 | Retention 포함 | Retention 제외 |
| --- | ---: | ---: |
| 학습 데이터 수 | 30,627 | 30,627 |
| 검증 데이터 수 | 10,210 | 10,210 |
| 입력 피처 수 | 130 | 126 |
| 타깃 변수 | Churn (1=이탈, 0=유지) | Churn (1=이탈, 0=유지) |

- 학습에는 `train_with_retention.csv`, `train_without_retention.csv`만 사용했다.
- 검증에는 `val_with_retention.csv`, `val_without_retention.csv`만 사용했다.
- Retention 포함 버전에는 `RetentionCalls`, `RetentionOffersAccepted`, 유지팀 연락 여부 관련 변수가 추가된다.
- test 데이터는 최종 모델을 확정한 후 최종 예측에만 사용한다.

## 3. 비교 모델과 평가 기준

비교 모델은 Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting이다.

- 주 평가 지표는 **ROC-AUC**다. 여러 임계값에서 이탈 고객을 유지 고객보다 높은 위험도로 구분하는 능력을 본다.
- Accuracy는 전체 정답 비율이다.
- Precision은 이탈이라고 예측한 고객 중 실제 이탈 고객의 비율이다.
- Recall은 실제 이탈 고객 중 모델이 찾아낸 비율이다.
- F1-score는 Precision과 Recall의 균형을 나타낸다.

## 4. 전체 검증 성능 비교

| model | feature_set | roc_auc | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| HistGradientBoosting | Retention 포함 | 0.6787 | 0.6206 | 0.4019 | 0.6492 | 0.4965 |
| Random Forest | Retention 포함 | 0.6701 | 0.6929 | 0.4587 | 0.3657 | 0.4070 |
| Decision Tree | Retention 포함 | 0.6334 | 0.5656 | 0.3628 | 0.6706 | 0.4708 |
| Logistic Regression | Retention 포함 | 0.6296 | 0.5933 | 0.3690 | 0.5795 | 0.4509 |
| HistGradientBoosting | Retention 제외 | 0.6738 | 0.6193 | 0.3996 | 0.6390 | 0.4917 |
| Random Forest | Retention 제외 | 0.6665 | 0.6916 | 0.4550 | 0.3555 | 0.3992 |
| Decision Tree | Retention 제외 | 0.6257 | 0.5603 | 0.3588 | 0.6683 | 0.4669 |
| Logistic Regression | Retention 제외 | 0.6228 | 0.5872 | 0.3650 | 0.5850 | 0.4495 |

## 5. 최종 모델 선정 결과

### 5-1. Retention 포함 모델

- 최종 모델: **HistGradientBoosting**
- ROC-AUC: **0.6787**
- Accuracy: 0.6206
- Precision: 0.4019
- Recall: 0.6492
- F1-score: **0.4965**
- 저장 파일: `models/histgradientboosting_with_retention.joblib`

### 5-2. Retention 제외 모델

- 최종 모델: **HistGradientBoosting**
- ROC-AUC: **0.6738**
- Accuracy: 0.6193
- Precision: 0.3996
- Recall: 0.6390
- F1-score: **0.4917**
- 저장 파일: `models/histgradientboosting_without_retention.joblib`

### 5-3. Retention 변수의 효과

- Retention 포함 모델의 ROC-AUC는 제외 모델보다 **0.0049** 높다.
- 임계값 0.5 기준으로, Retention 포함 모델은 실제 이탈 고객을 **30명 더 탐지**했다.
- 반면 유지 고객을 이탈로 잘못 예측한 고객도 **17명 더 많다.**

따라서 유지팀의 접촉 비용과 이탈 고객을 놓쳤을 때의 비용을 함께 고려해 운영 임계값을 조정할 수 있다.

## 6. Accuracy와 기준선 0.5 해석

> 최종 Retention 포함 모델의 Accuracy는 62.1%이다. 이는 이탈 고객을 최대한 놓치지 않도록 클래스 가중치를 적용하고, 기준선 0.5에서 Recall을 우선했기 때문에 일부 유지 고객도 이탈 위험군으로 분류한 결과이다.

현재 기준선 0.5는 이탈 확률이 50% 이상인 고객을 이탈 위험군으로 분류한다. 아래는 기준선을 바꾸었을 때의 검증 성능과 고객 접촉 규모 변화다.

| threshold | accuracy | precision | recall | f1 | predicted_churn_customers | true_positive | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3000 | 0.4032 | 0.3188 | 0.9422 | 0.4764 | 8695.0 | 2772.0 | 5923.0 | 170.0 |
| 0.4000 | 0.5145 | 0.3550 | 0.8382 | 0.4987 | 6947.0 | 2466.0 | 4481.0 | 476.0 |
| 0.5000 | 0.6206 | 0.4019 | 0.6492 | 0.4965 | 4752.0 | 1910.0 | 2842.0 | 1032.0 |
| 0.6000 | 0.7059 | 0.4847 | 0.3277 | 0.3910 | 1989.0 | 964.0 | 1025.0 | 1978.0 |
| 0.7000 | 0.7185 | 0.5904 | 0.0755 | 0.1338 | 376.0 | 222.0 | 154.0 | 2720.0 |

- 기준선을 낮추면 실제 이탈 고객을 더 많이 탐지할 수 있지만, 유지 고객에게도 더 많은 방어 활동을 하게 된다.
- 기준선을 높이면 접촉 대상과 오탐은 줄지만, 실제 이탈 고객을 더 많이 놓칠 수 있다.
- 최적 기준선은 캠페인 예산, 유지 연락 비용, 이탈 고객을 놓쳤을 때의 비용을 함께 고려해 결정한다.

## 7. 혼동행렬 해석

### Retention 포함 모델

| 실제값 / 예측값 | 유지(0) | 이탈(1) |
| --- | ---: | ---: |
| 유지(0) | 4,426 | 2,842 |
| 이탈(1) | 1,032 | 1,910 |

### Retention 제외 모델

| 실제값 / 예측값 | 유지(0) | 이탈(1) |
| --- | ---: | ---: |
| 유지(0) | 4,443 | 2,825 |
| 이탈(1) | 1,062 | 1,880 |

## 8. 전체 최적 모델의 주요 피처

순열 중요도(Permutation Importance) 기준 상위 10개 피처다. 변수를 무작위로 섞었을 때 ROC-AUC가 크게 떨어질수록 모델 예측에 중요한 정보라는 뜻이다.

1. `CurrentEquipmentDays` (현재 단말기 사용 일수): 0.0597
2. `MonthsInService` (서비스 이용 개월 수): 0.0346
3. `MonthlyMinutes` (월 통화 시간): 0.0288
4. `PercChangeMinutes` (통화 시간 변화): 0.0232
5. `TotalRecurringCharge` (월 정기 요금): 0.0124
6. `UniqueSubs` (전체 가입 회선 수): 0.0076
7. `RetentionCalls` (이탈 방어 연락 횟수): 0.0069
8. `OverageMinutes` (초과 사용 통화 시간): 0.0059
9. `CreditRating_5-Low` (신용등급 5-낮음 여부): 0.0057
10. `AgeHH1` (가구원 1 나이): 0.0047

피처 중요도는 모델 예측에 유용한 정도를 뜻하며, 해당 변수가 이탈의 직접 원인임을 증명하지는 않는다.

## 9. 운영 제안

1. **사전 이탈 예방**: 유지팀 접촉 정보가 아직 없는 시점이라면 `Retention 제외 모델`을 사용한다.
2. **접촉 이력 반영 위험도 판단**: 유지 활동 정보까지 확보된 경우 `Retention 포함 모델`을 사용한다.
3. 단말기 사용 기간, 서비스 이용 기간, 통화 시간 변화가 큰 고객을 우선 관리 후보로 검토한다.
4. 임계값 0.5는 기본값이므로, 캠페인 예산과 고객 접촉 비용에 맞춰 Recall과 Precision의 균형을 조정한다.

## 10. 한계 및 유의사항

- Retention 변수는 이탈 판정 시점 이후에 기록된 정보라면 데이터 누수가 될 수 있으므로 측정 시점을 반드시 확인해야 한다.
- 최종 발표 전 교차검증 또는 test 데이터로 성능을 추가 확인하여 신뢰도를 높인다.
- 피처 중요도는 인과관계가 아니라 예측 기여도다.

## 11. 생성 산출물

- 모델: `models/histgradientboosting_with_retention.joblib`
- 모델: `models/histgradientboosting_without_retention.joblib`
- 상세 리포트: `reports/modeling_report.md`
- 모델 성능 비교: `artifacts/modeling/presentation_00_retention_model_comparison.png`
- 성능 그래프: `artifacts/modeling/presentation_01_model_performance.png`
- ROC 곡선: `artifacts/modeling/presentation_02_roc_curve.png`
- 혼동행렬: `artifacts/modeling/presentation_03_confusion_matrix.png`
- 피처 중요도: `artifacts/modeling/presentation_04_feature_importance.png`
- 기준선 비교: `artifacts/modeling/presentation_05_threshold_tradeoff.png`
