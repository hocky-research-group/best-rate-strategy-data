"""The design rule accepts a protocol only when every one of its physical conditions holds, and
the error of a strategy is the root-mean-square over every subset fit, not over conditions."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bias_design_recipe as recipe  # noqa: E402
from tradeoff_analysis import pooled_rmse  # noqa: E402

# Q-like inputs: basin width 0.05, hill width 0.02, k0 = 1.54 us^-1, barrier 5.95 kT
Q = dict(sigma_a=0.05, sigma=0.02, k0_per_s=1.54e6, gamma=0.99)
BARRIER = 5.95
TAU_REL = 305.0


def _verdict(b, h0, pace_steps, gamma=Q["gamma"], barrier=BARRIER, tau_rel=TAU_REL):
    p = recipe.predict(Q["sigma_a"], Q["sigma"], Q["k0_per_s"], gamma, b, h0, pace_steps * recipe.DT_PS)
    return dict(**p, **recipe.verdict(p, gamma, barrier, pace_steps * recipe.DT_PS, tau_rel))


def test_rule_is_the_three_physical_conditions():
    """Residual barrier, pace over relaxation and a minimum boost; no upper boost cap."""
    assert recipe.RESIDUAL_MIN_KT == 2.0
    assert recipe.PACE_MIN_OVER_RELAX == 0.3
    assert recipe.BOOST_MIN == 3.0
    assert not hasattr(recipe, "BOOST_WINDOW")


def test_gentle_slow_protocol_is_recommended():
    v = _verdict(12, 0.35, 70000)          # the registered prospective Q protocol
    assert v["recommended"]
    assert v["residual_pred_kT"] >= recipe.RESIDUAL_MIN_KT
    assert v["tau_over_relax"] == pytest.approx(700.0 / TAU_REL)


def test_each_condition_rejects_on_its_own():
    """Every case passes the other two conditions and fails only the one named."""
    # too fast: the recommended protocol at pace 100 steps (1 ps against a 305 ps relaxation)
    fast = _verdict(12, 0.35, 100)
    assert fast["tau_over_relax"] < recipe.PACE_MIN_OVER_RELAX and not fast["recommended"]
    # barrier filled away: a slow, strong bias against a 4 kT barrier leaves under 1 kT
    filled = _verdict(20, 1.0, 100000, barrier=4.0)
    assert filled["tau_over_relax"] >= recipe.PACE_MIN_OVER_RELAX and filled["boost"] >= recipe.BOOST_MIN
    assert filled["residual_pred_kT"] < recipe.RESIDUAL_MIN_KT and not filled["recommended"]
    # too slow to be useful: hills too small to reach a boost of 3
    slow = _verdict(5, 0.02, 100000)
    assert slow["tau_over_relax"] >= recipe.PACE_MIN_OVER_RELAX and slow["residual_pred_kT"] >= recipe.RESIDUAL_MIN_KT
    assert slow["boost"] < recipe.BOOST_MIN and not slow["recommended"]


def test_no_escape_is_rejected_not_an_error():
    p = dict(boost=np.nan, V_escape_kT=np.nan, t_star_ns=np.inf, n_star=np.inf, h_eff=0.0)
    v = recipe.verdict(p, 0.9, BARRIER, 100.0, TAU_REL)
    assert not v["recommended"]


def test_pooled_rmse_weights_every_subset_fit():
    """Two conditions with different draw counts: the pooled value is the RMSE of all draws."""
    rng = np.random.default_rng(0)
    e1, e2 = rng.normal(0.3, 0.5, 10), rng.normal(-0.8, 0.2, 4)
    d = pd.DataFrame(dict(n_fits=[10, 4], rmse=[np.sqrt((e1 ** 2).mean()), np.sqrt((e2 ** 2).mean())],
                          mean_err=[e1.mean(), e2.mean()], sd=[e1.std(ddof=1), e2.std(ddof=1)]))
    exact = math.sqrt((np.concatenate([e1, e2]) ** 2).mean())
    assert pooled_rmse(d) == pytest.approx(exact)
    inflated = math.sqrt(np.mean(d.mean_err ** 2 + d.sd ** 2))
    assert inflated != pytest.approx(exact)
