"""Bias parameters for one well-tempered condition along R_ee per temperature of the scan.

Usage:
    uv run python scripts/design_wtmetad_temperature_ree.py [out.csv]

The Q conditions of the scan (design_wtmetad_temperature.py) hold the residual barrier
fixed. Along R_ee that target is not usable: with gamma near 0.4 the bias that leaves a
chosen residual, Delta F - gamma V, exceeds the apparent barrier itself. The R_ee recipe
instead holds the predicted escape time fixed at the value of the Q condition at the slow
pace, 84 hills of 4 ns (336 ns), and lets the growth law set the hill height:

    escape when  k0 exp(gamma V_n / kT) n tau = 1,
    V_n = (b - 1) kT ln[1 + n h_eff / ((b - 1) kT)],
    h_eff = h0 sigma / sqrt(sigma^2 + 2 sigma_A^2),

with gamma the 312 K OPES-flooding pilot value along R_ee, k0(T) the 312 K reference rate
lowered by the barrier change along Q (the parameter-free prediction the scan tests), the
bias factor of the Q condition at that temperature, sigma the hill width along R_ee and
sigma_A the width of the folded basin along R_ee at that temperature, reweighted from the
Q-biased scan runs (analysis/fes_temperature_scan_ree.csv, second half, mean of the two
seeds). The apparent barrier along R_ee from the same runs, scaled by the 312 K anchor as
the Q barriers are, gives the predicted residual, Delta F - gamma V_n, for the record; it
does not enter the height.

Sanity check (asserted): at 270 K the procedure must return the condition already run
there, h0 = 4.0 kJ/mol at bias factor 16, whose inputs are the notebook values of
2026-09-22 (basin width 0.093 nm, k0 = 60.8 /s, 84 hills, t* = 336 ns).
"""
import sys

import numpy as np
import pandas as pd

KB = 0.0083144626       # kJ/(mol K)
SIGMA = 0.06            # hill width along R_ee (nm), as in every R_ee run of this study
DT_PS = 0.01            # timestep
PACE = 400000           # the slow pace of the Q conditions, 4 ns
N_HILLS = 84            # hills before escape, matching the 270 K R_ee condition (t* = 336 ns)
GAMMA = 0.44            # OPES-flooding pilot along R_ee at 312 K
LN_K0_312_PER_S = 14.24614588110295          # reference rate, ln(k / s^-1)
D_BARRIER_Q_KT = {312: 0.0, 300: 2.31, 285: 5.05, 270: 10.14}   # Q barrier change, as in the scan
BIAS_FACTOR = {312: 12, 300: 12, 285: 14, 270: 16}
UNBIASED_BARRIER_REE_312 = 5.92              # kT, unbiased PMF along R_ee: the anchor
TEMPS = (300, 285)                           # the conditions to run; 270 K exists, 312 K has no single fill


def height_for(T: float, b: int, sigma_a: float, k0: float) -> tuple[float, float, float]:
    """Hill height for which the growth law predicts escape after N_HILLS hills.

    Args:
        T (float): temperature in K.
        b (int): bias factor.
        sigma_a (float): width of the folded basin along R_ee (nm).
        k0 (float): unbiased rate at T (1/s).

    Returns:
        (h0 in kJ/mol, h_eff in kJ/mol, gamma V_n at escape in kT).
    """
    kT = KB * T
    tau_s = PACE * DT_PS * 1e-12
    gv = -np.log(k0 * N_HILLS * tau_s)            # gamma V_n / kT at escape
    v = gv / GAMMA * kT
    h_eff = (b - 1) * kT * (np.exp(v / ((b - 1) * kT)) - 1) / N_HILLS
    return h_eff * np.sqrt(SIGMA ** 2 + 2 * sigma_a ** 2) / SIGMA, h_eff, gv


def scan_inputs() -> pd.DataFrame:
    """Second-half basin width and anchored barrier along R_ee per temperature."""
    s = pd.read_csv("analysis/fes_temperature_scan_ree.csv")
    s = s[s.window == "second half"].groupby("temperature_K")[["sigma_A", "barrier_kT"]].mean()
    s["barrier_kT"] *= UNBIASED_BARRIER_REE_312 / s.loc[312, "barrier_kT"]
    return s


def main(out: str) -> None:
    """Write one row per temperature to run, after the 270 K check.

    Args:
        out (str): output CSV path.
    """
    h0_270, _, _ = height_for(270, 16, 0.093, 60.8)
    assert abs(h0_270 - 4.0) < 0.05, h0_270
    inp = scan_inputs()
    rows = []
    for T in TEMPS:
        k0 = float(np.exp(LN_K0_312_PER_S - D_BARRIER_Q_KT[T]))
        h0, h_eff, gv = height_for(T, BIAS_FACTOR[T], inp.loc[T, "sigma_A"], k0)
        rows.append(dict(temperature_K=T, barrier_ree_kT=round(inp.loc[T, "barrier_kT"], 2),
                         sigma_A=round(inp.loc[T, "sigma_A"], 3), bias_factor=BIAS_FACTOR[T],
                         height_kJmol=round(h0, 1), pace=PACE, h_eff_kJmol=round(h_eff, 3),
                         hills_to_escape=N_HILLS, t_star_ns=N_HILLS * PACE * DT_PS / 1e3,
                         k0_per_s=round(k0, 1), gamma=GAMMA, gamma_V_kT=round(gv, 2),
                         residual_pred_kT=round(inp.loc[T, "barrier_kT"] - gv, 2),
                         predicted_boost=round(float(np.exp(gv)))))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n270 K check: h0 = {h0_270:.2f} kJ/mol against the 4.0 run there")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "analysis/wtmetad_temperature_design_ree.csv")
