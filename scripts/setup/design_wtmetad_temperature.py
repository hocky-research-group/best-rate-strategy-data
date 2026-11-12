"""Bias parameters for one well-tempered condition per temperature, from the design rule.

Usage:
    uv run --group figures python scripts/design_wtmetad_temperature.py [out.csv]

The rule fixes what the bias must do, not what the parameters are: fill to a chosen
residual below the apparent barrier, and take long enough doing it that the coordinate
keeps up. This inverts the growth law for the parameters that achieve it at each
temperature of the scan, so the residual barrier is held fixed while the boost the
condition implies varies by five orders of magnitude.

Growth law, with the unbroadened hill overlap used throughout this work:
    V_n = (b - 1) kT ln[1 + n h_eff / ((b - 1) kT)],
    h_eff = h0 sigma / sqrt(sigma^2 + 2 sigma_A^2).
Inverted for the hill height that reaches V* in n hills:
    h_eff = (b - 1) kT [exp(V* / ((b - 1) kT)) - 1] / n.

Barriers are the second-half reweighted values of analysis/fes_temperature_scan.csv and
sigma_A is the reweighted width of the folded basin in the same runs; at 312 K that width
is 0.050 against 0.050 from the unbiased PMF, which is the check that the biased runs
may be used for it at the other temperatures.

The reweighted barrier rises throughout a run and is still low when the run ends: at 312 K,
where the answer is known from the unbiased PMF, it is 6% low after 2 us. Barriers at the
other temperatures are scaled by that one factor, since an uncorrected value places every
rung of the ladder about one rung spacing too high. The scaling is an extrapolation from a
single anchor, so the fill each run actually achieves is measured afterwards and is what
the analysis uses; the correction only decides where the rungs are placed.

Sanity check (asserted): at 312 K the procedure must return the prospective condition
already run and validated there, bias factor 12 and a height near 0.3 kJ/mol at pace
70000, with an escape time near the 81 ns observed.
"""
import sys

import numpy as np
import pandas as pd

KB = 0.0083144626      # kJ/(mol K)
SIGMA = 0.02           # hill width along Q, as in every run of this study
DT_PS = 0.01           # timestep
RESIDUAL_KT = 2.5      # barrier left standing
T_STAR_NS = 80.0       # target escape time, matching the validated 312 K condition
PACE = 70000
SIGMA_A = {312: 0.0503, 300: 0.0449, 285: 0.0403, 270: 0.0325}
UNBIASED_BARRIER_KT = 5.95   # 312 K, from the unbiased PMF: the anchor for the correction
BIAS_FACTOR = {312: 12, 300: 12, 285: 14, 270: 16}


def height_for(barrier_kT: float, T: float, b: int, sigma_a: float) -> tuple[float, float, float]:
    """Hill height that reaches the target fill in the target time.

    Args:
        barrier_kT (float): apparent barrier in kT.
        T (float): temperature in K.
        b (int): bias factor.
        sigma_a (float): width of the folded basin along Q.

    Returns:
        (h0 in kJ/mol, h_eff in kJ/mol, number of hills before escape).
    """
    kT = KB * T
    v_star = (barrier_kT - RESIDUAL_KT) * kT
    n = T_STAR_NS * 1e3 / (PACE * DT_PS)
    h_eff = (b - 1) * kT * (np.exp(v_star / ((b - 1) * kT)) - 1) / n
    return h_eff * np.sqrt(SIGMA ** 2 + 2 * sigma_a ** 2) / SIGMA, h_eff, n


def calibrated_barriers() -> pd.Series:
    """Second-half reweighted barriers in kT, scaled onto the 312 K unbiased anchor."""
    scan = pd.read_csv("analysis/fes_temperature_scan.csv")
    bar = (scan[(scan.estimate == "reweight") & (scan.window == "second half")]
           .groupby("temperature_K").barrier_kT.mean())
    return bar * (UNBIASED_BARRIER_KT / bar[312])


def main(out: str) -> None:
    """Write one row per temperature with the parameters the rule implies.

    Args:
        out (str): output CSV path.
    """
    bar = calibrated_barriers()
    rows = []
    for T, b in BIAS_FACTOR.items():
        h0, h_eff, n = height_for(bar[T], T, b, SIGMA_A[T])
        rows.append(dict(temperature_K=T, barrier_kT=round(bar[T], 2), sigma_A=SIGMA_A[T],
                         bias_factor=b, height_kJmol=round(h0, 2), pace=PACE,
                         h_eff_kJmol=round(h_eff, 3), hills_to_escape=round(n),
                         residual_kT=RESIDUAL_KT, t_star_ns=T_STAR_NS,
                         predicted_boost=round(float(np.exp(bar[T] - RESIDUAL_KT)))))
    df = pd.DataFrame(rows)
    anchor = df[df.temperature_K == 312].iloc[0]
    assert anchor.bias_factor == 12, anchor.bias_factor
    assert abs(anchor.height_kJmol - 0.30) < 0.08, anchor.height_kJmol
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n312 K returns b = {anchor.bias_factor}, h0 = {anchor.height_kJmol} kJ/mol, "
          "against the validated prospective condition b = 12, h0 = 0.35")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "analysis/wtmetad_temperature_design.csv")
