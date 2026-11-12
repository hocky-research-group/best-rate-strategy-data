"""The error-cost front holds only strategies that no other strategy beats on both cost and error,
and which ones it holds does not depend on the order the strategies arrive in: two estimators
fitted to the same runs cost exactly the same, and only the more accurate of them is on it."""
import sys
from itertools import permutations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tradeoff_analysis import pareto_front  # noqa: E402

# two estimators on the same runs at each of three sizes, plus a costlier, worse strategy
STRATEGIES = [
    dict(label="EATR n=5", cost_us=1.0, rmse=0.90),
    dict(label="iMetaD n=5", cost_us=1.0, rmse=0.80),
    dict(label="EATR n=10", cost_us=2.0, rmse=0.60),
    dict(label="iMetaD n=10", cost_us=2.0, rmse=0.70),
    dict(label="EATR n=20", cost_us=4.0, rmse=0.40),
    dict(label="iMetaD n=20", cost_us=4.0, rmse=0.40),
    dict(label="VES n=5", cost_us=5.0, rmse=0.50),
]


def test_equal_cost_keeps_only_the_lower_error_in_any_order():
    fronts = set()
    for order in permutations(range(len(STRATEGIES))):
        d = pd.DataFrame([STRATEGIES[i] for i in order], index=list(order))
        fronts.add(tuple(pareto_front(d).label))
    assert len(fronts) == 2, fronts          # only the exact tie at n=20 may resolve either way
    for f in fronts:
        assert f[:2] == ("iMetaD n=5", "EATR n=10")
        assert f[2] in ("EATR n=20", "iMetaD n=20")


def test_front_is_strictly_decreasing_in_error_and_drops_dominated_points():
    f = pareto_front(pd.DataFrame(STRATEGIES))
    assert (f.cost_us.diff().dropna() > 0).all()
    assert (f.rmse.diff().dropna() < 0).all()
    assert "VES n=5" not in set(f.label)
