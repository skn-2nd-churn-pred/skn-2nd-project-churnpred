# 고객 이탈 예측 모델링 상세 리포트

## 1. 프로젝트 목적

- 통신사 고객의 통화·요금·단말기·생활정보를 활용해 고객 이탈 여부(`Churn`)를 예측한다.
- 목표는 불균형 데이터에서 Accuracy나 Precision만 높이는 것이 아니라, 실제 이탈 고객을 효과적으로 선별하는 것이다.
- 즉 높은 정밀도만이 아니라, 임계값 조정 등을 고려하여 이탈 고객 탐지 성능과 경영 시 운영상의 최대 효율의 균형을 주안점에 둔 프로젝트이다.

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
- test 데이터는 모델 확정 후 최종 확인용으로 남겨 둔다.

## 3. 비교 모델과 평가 지표

사용 모델은 **Dummy (Most Frequent), Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting**이다.

- **ROC-AUC**: 여러 기준선에서 이탈 고객과 유지 고객을 구분하는 순위화 능력이다.
- **PR-AUC**: 이탈 고객(양성 클래스)이 약 29%인 불균형 데이터에서 이탈 고객을 실제로 얼마나 잘 선별하는지 확인하는 지표다. PR-AUC는 양성 클래스의 Precision과 Recall에 집중하므로 본 프로젝트의 **주 평가 지표**로 사용한다. ROC-AUC는 전체 고객의 위험도 구분 능력을 확인하는 보조 지표로 함께 사용한다.
- **Accuracy**: 전체 정답 비율이다.
- **Precision**: 이탈이라고 예측한 고객 중 실제 이탈 고객의 비율이다.(예측 정확도)
- **Recall**: 실제 이탈 고객 중 모델이 찾아낸 비율이다.(실제 이탈 고객 탐지율)
- **F1-score**: Precision과 Recall의 균형을 나타낸다.

### Dummy 기준 모델의 의미

`Dummy (Most Frequent)`는 모든 고객을 가장 많은 클래스인 유지 고객으로 예측한다. Retention 제외 검증 데이터에서 Accuracy는 **0.7119**이지만, Recall과 F1-score는 각각 **0.0000**, **0.0000**이다. 즉 Accuracy가 높아도 실제 이탈 고객을 찾지 못할 수 있다. 따라서 본 프로젝트는 Accuracy를 일부 포기하더라도 Recall, F1-score, ROC-AUC, PR-AUC를 함께 고려한다.

## 4. 전체 검증 성능 비교

### 4-1. Retention 포함 모델

| model | roc_auc | pr_auc | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| HistGradientBoosting | 0.6787 | 0.4494 | 0.6206 | 0.4019 | 0.6492 | 0.4965 |
| Random Forest | 0.6701 | 0.4354 | 0.6929 | 0.4587 | 0.3657 | 0.4070 |
| Decision Tree | 0.6334 | 0.3963 | 0.5656 | 0.3628 | 0.6706 | 0.4708 |
| Logistic Regression | 0.6296 | 0.3856 | 0.5933 | 0.3690 | 0.5795 | 0.4509 |
| Dummy (Most Frequent) | 0.5000 | 0.2881 | 0.7119 | 0.0000 | 0.0000 | 0.0000 |

### 4-2. Retention 제외 모델

| model | roc_auc | pr_auc | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| HistGradientBoosting | 0.6738 | 0.4453 | 0.6193 | 0.3996 | 0.6390 | 0.4917 |
| Random Forest | 0.6665 | 0.4280 | 0.6916 | 0.4550 | 0.3555 | 0.3992 |
| Decision Tree | 0.6257 | 0.3885 | 0.5603 | 0.3588 | 0.6683 | 0.4669 |
| Logistic Regression | 0.6228 | 0.3764 | 0.5872 | 0.3650 | 0.5850 | 0.4495 |
| Dummy (Most Frequent) | 0.5000 | 0.2881 | 0.7119 | 0.0000 | 0.0000 | 0.0000 |

## 5. 최종 채택안 및 근거

### 5-1. 데이터 버전: without_retention 채택

Retention 포함 모델의 최고 ROC-AUC는 **0.6787**, Retention 제외 모델의 최고 ROC-AUC는 **0.6738**로 차이는 **0.0049**다. 성능 차이는 작지만, Retention 팀 접촉 정보가 고객 이탈을 예측하려는 시점에 이미 확보되는 정보인지는 확인할 수 없다. 이 정보가 이탈 판정 이후에 기록된 것이라면 데이터 누수 위험이 생긴다.

따라서 사전 이탈 예방이라는 운영 목적과 정보 시점의 불확실성을 고려하여, **without_retention 버전**을 최종 데이터 버전으로 채택했다.

### 5-2. 모델: HistGradientBoosting 채택

Retention 제외 버전에서 **HistGradientBoosting**은 비교 모델 중 PR-AUC가 가장 높은 **0.4453**를 기록했다. ROC-AUC도 **0.6738**로 가장 높았으며, Recall은 **0.6390**, F1-score는 **0.4917**다.

전처리 단계에서 예상했던 것처럼 단일 변수 규칙이 아니라 여러 변수의 상호작용이 중요한 데이터이므로, Boosting 계열 모델이 좋은 성능을 보여주었다. 이에 따라 최종적으로 **without_retention 버전의 HistGradientBoosting 모델**을 채택했다.

- 최종 모델 파일: `models/histgradientboosting_without_retention_final.joblib`
- 기본 기준선: 0.5 (운영 비용과 이탈 손실을 고려해 조정 가능)

## 6. Accuracy와 기준선 해석

최종 모델의 Accuracy는 **61.9%**다. 이는 이탈 고객을 최대한 놓치지 않도록 클래스 가중치를 적용했기 때문에 일부 유지 고객도 이탈 위험군으로 분류한다.

기준선 0.5는 이탈 확률이 50% 이상인 고객을 이탈 위험군으로 분류한다. 아래는 최종 모델에서 기준선을 바꾸었을 때의 검증 성능과 고객 접촉 규모 변화다.

| threshold | accuracy | precision | recall | f1 | predicted_churn_customers | true_positive | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3000 | 0.4047 | 0.3191 | 0.9402 | 0.4765 | 8668.0 | 2766.0 | 5902.0 | 176.0 |
| 0.4000 | 0.5125 | 0.3539 | 0.8379 | 0.4976 | 6965.0 | 2465.0 | 4500.0 | 477.0 |
| 0.5000 | 0.6193 | 0.3996 | 0.6390 | 0.4917 | 4705.0 | 1880.0 | 2825.0 | 1062.0 |
| 0.6000 | 0.7027 | 0.4765 | 0.3209 | 0.3835 | 1981.0 | 944.0 | 1037.0 | 1998.0 |
| 0.7000 | 0.7195 | 0.5975 | 0.0812 | 0.1430 | 400.0 | 239.0 | 161.0 | 2703.0 |

- 기준선을 낮추면 실제 이탈 고객을 더 많이 탐지할 수 있지만, 유지 고객에게도 더 많은 방어 활동을 하게 된다.
- 기준선을 높이면 접촉 대상과 오탐은 줄지만, 실제 이탈 고객을 더 많이 놓칠 수 있다.
- 최적 기준선은 캠페인 예산, 유지 연락 비용, 이탈 고객을 놓쳤을 때의 비용을 함께 고려해 결정한다.

## 7. 혼동행렬 해석

### Retention 포함 HistGradientBoosting

| 실제값 / 예측값 | 유지(0) | 이탈(1) |
| --- | ---: | ---: |
| 유지(0) | 4,426 | 2,842 |
| 이탈(1) | 1,032 | 1,910 |

### 최종 채택: Retention 제외 HistGradientBoosting

| 실제값 / 예측값 | 유지(0) | 이탈(1) |
| --- | ---: | ---: |
| 유지(0) | 4,443 | 2,825 |
| 이탈(1) | 1,062 | 1,880 |

## 8. 최종 모델의 주요 피처

순열 중요도(Permutation Importance) 기준 상위 10개 피처다. 변수를 무작위로 섞었을 때 ROC-AUC가 크게 떨어질수록 모델 예측에 중요한 정보라는 뜻이다.

1. `CurrentEquipmentDays` (현재 단말기 사용 일수): 0.0556
2. `MonthsInService` (서비스 이용 개월 수): 0.0357
3. `MonthlyMinutes` (월 통화 시간): 0.0298
4. `PercChangeMinutes` (통화 시간 변화): 0.0241
5. `TotalRecurringCharge` (월 정기 요금): 0.0136
6. `CreditRating_5-Low` (신용등급 5-낮음 여부): 0.0064
7. `UniqueSubs` (전체 가입 회선 수): 0.0062
8. `OverageMinutes` (초과 사용 통화 시간): 0.0059
9. `AgeHH1` (가구원 1 나이): 0.0050
10. `HandsetRefurbished_No` (HandsetRefurbished_No): 0.0038

피처 중요도는 모델 예측에 유용한 정도이며, 해당 변수가 이탈의 직접 원인임을 증명하지는 않는다.

## 9. 운영 제안

1. 사전 이탈 예방에 최종 채택 모델인 `without_retention` HistGradientBoosting을 사용한다.
2. 이탈 확률이 높은 고객부터 유지 캠페인 대상자로 우선 검토한다.
3. 기준선은 예산, 고객 접촉 비용, 이탈 고객을 놓쳤을 때의 손실을 고려해 조정한다.

## 10. 한계 및 유의사항

- Retention 변수의 측정 시점이 불명확하므로 운영 모델에서는 제외했다.
- PR-AUC와 피처 중요도는 인과관계가 아니라 예측 성능과 예측 기여도를 설명한다.

## 11. 생성 산출물

- 최종 모델: `models/histgradientboosting_without_retention_final.joblib`
- 상세 리포트: `reports/modeling_report.md`
- 모델 성능 비교: `artifacts/modeling/presentation_00_retention_model_comparison.png`
- 성능 그래프: `artifacts/modeling/presentation_01_model_performance.png`
- ROC 곡선: `artifacts/modeling/presentation_02_roc_curve.png`
- PR 곡선(PR-AUC): `artifacts/modeling/presentation_03_pr_curve.png`
- 혼동행렬: `artifacts/modeling/presentation_04_confusion_matrix.png`
- 최종 모델 피처 중요도: `artifacts/modeling/presentation_05_feature_importance.png`
- 최종 모델 기준선 비교: `artifacts/modeling/presentation_06_threshold_tradeoff.png`
