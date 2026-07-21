"""추론 검증 테스트 — 가이드 완료 기준 자동 확인.

가이드의 완료 기준을 코드로 검증한다.
  1) 저장한 모델을 "새 프로세스"에서 다시 불러올 수 있다
  2) 신규 고객 1명의 데이터를 예측할 수 있다
  3) 이탈 여부와 이탈 확률을 출력한다
  4) 입력값을 바꾸면 예측 확률도 실제로 바뀐다 (고정 결과가 아님)

실행 (둘 중 아무거나):
    pytest -q                    # pytest가 설치된 경우
    python tests/test_inference.py   # pytest 없이도 동작
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import load_config, load_model, predict_one, resolve_path  # noqa: E402

cfg = load_config()
MODEL_PATH = resolve_path(cfg["model"]["path"])
FEATURE_SOURCE = resolve_path(cfg["model"]["feature_source"])
SKIP_REASON = (
    f"모델({MODEL_PATH.name}) 또는 기준 데이터가 없습니다. "
    "노트북 02와 python src/data.py 를 먼저 실행하세요."
)


def _ready() -> bool:
    return MODEL_PATH.exists() and FEATURE_SOURCE.exists()


def _sample_customer() -> dict:
    """기준 데이터의 첫 고객을 '신규 고객'처럼 사용한다 (Target 제외)."""
    df = pd.read_csv(FEATURE_SOURCE, nrows=1)
    return df.drop(columns=[cfg["target"]["column"]]).iloc[0].to_dict()


def test_model_loads():
    """1) 저장된 모델을 불러오고, 추론에 필요한 3요소가 모두 갖춰진다."""
    bundle = load_model()
    assert {"pipeline", "threshold", "feature_names"} <= set(bundle)
    assert 0.0 < bundle["threshold"] < 1.0, "임계값은 0과 1 사이여야 한다"
    assert len(bundle["feature_names"]) > 0
    # 저장 형식이 dict든 Pipeline이든 정규화돼서 나와야 한다
    assert hasattr(bundle["pipeline"], "predict_proba")


def test_feature_count_matches_model():
    """모델이 기대하는 feature 수와 복원된 feature 목록이 일치한다."""
    bundle = load_model()
    expected = getattr(bundle["pipeline"], "n_features_in_", None)
    if expected is not None:
        assert expected == len(bundle["feature_names"]), (
            f"모델은 {expected}개를 기대하는데 feature 목록은 "
            f"{len(bundle['feature_names'])}개입니다. "
            "config.yaml의 model.feature_source를 확인하세요."
        )


def test_predict_one_customer():
    """2)+3) 신규 고객 1명을 예측해 이탈 여부와 확률을 반환한다."""
    result = predict_one(_sample_customer())

    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["churn_prediction"] in (0, 1)
    assert result["risk_grade"] in ("저위험", "중위험", "고위험")
    assert result["suggested_action"]

    # 예측 라벨이 임계값과 일관되는지 확인
    threshold = load_model()["threshold"]
    expected_label = int(result["churn_probability"] >= threshold)
    assert result["churn_prediction"] == expected_label


def test_probability_changes_with_input():
    """4) 입력을 바꾸면 확률도 바뀐다 — 고정 결과를 반환하지 않는지 확인."""
    base = _sample_customer()

    # 이탈과 관련이 큰 변수를 크게 바꿔본다 (전처리 후 표준화 값 기준)
    low, high = dict(base), dict(base)
    key = "CurrentEquipmentDays" if "CurrentEquipmentDays" in base else next(iter(base))
    low[key], high[key] = -3.0, 3.0

    p_low = predict_one(low)["churn_probability"]
    p_high = predict_one(high)["churn_probability"]
    assert p_low != p_high, (
        f"'{key}'를 크게 바꿨는데 확률이 동일합니다({p_low}). "
        "모델이 입력을 반영하지 못하거나 고정 결과를 반환하는지 확인하세요."
    )


TESTS = [
    test_model_loads,
    test_feature_count_matches_model,
    test_predict_one_customer,
    test_probability_changes_with_input,
]


def main() -> int:
    """pytest 없이 직접 실행할 때 사용하는 간이 러너."""
    if not _ready():
        print(f"[SKIP] {SKIP_REASON}")
        return 0

    failed = 0
    for test in TESTS:
        name = test.__name__
        try:
            test()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {name}\n       {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] {name}\n        {type(e).__name__}: {e}")
        else:
            print(f"[PASS] {name}")

    print("-" * 50)
    print("전체 통과" if failed == 0 else f"{failed}건 실패")
    return 1 if failed else 0


# pytest로 실행할 때 모델이 없으면 건너뛴다
try:
    import pytest

    pytestmark = pytest.mark.skipif(not _ready(), reason=SKIP_REASON)
except ImportError:
    pass


if __name__ == "__main__":
    raise SystemExit(main())
