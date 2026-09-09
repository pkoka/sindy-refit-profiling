"""
Sanity test for gate_cv.

Builds a synthetic library where collinearity is a single tunable knob, runs it
through the real ensemble_parallel so the actual dropout path is exercised, and
checks that the retention-conditioned CV separates the two cases.

The absolute CV values depend on N, n_features, N_DROP, THRESHOLD and ALPHA, so
they are not the thing to gate on. The SEPARATION RATIO is.

usage:
    python test_gate.py
"""

import numpy as np

from gate import gate_cv
from refit_parallel import ensemble_parallel

# true model: y depends on col 0 and col 2 only. col 1 is the planted duplicate.
TRUE_C0 = 1.5
TRUE_C2 = 0.8
N_SAMPLES = 8000
N_FEATURES = 6
N_MODELS = 20
NOISE = 1e-3


def make_synthetic(rho, seed=0):
    """
    rho is the target correlation between col1 and col0, directly.
      rho = 0.0  -> genuinely independent, col1 should get coefficient 0 and
                    fall out of the active set entirely
      rho -> 1.0 -> col1 is a copy of col0, library has a redundant direction

    this replaces the old eps knob, where eps=1 still left corr about 0.71 and
    the "independent" baseline had a real ambiguity in it
    """
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((N_SAMPLES, N_FEATURES))
    theta[:, 1] = rho * theta[:, 0] + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(N_SAMPLES)

    y = (TRUE_C0 * theta[:, 0] + TRUE_C2 * theta[:, 2]).reshape(-1, 1)
    y = y + NOISE * rng.standard_normal(y.shape)

    cond = float(np.linalg.cond(theta))
    return theta, y, cond


def run_case(name, eps):
    theta, y, cond = make_synthetic(eps)
    median, stacked, masks = ensemble_parallel(theta, y, n_models=N_MODELS, n_jobs=1)
    stat = gate_cv(stacked, masks, median=median)
    print(f"{name:16s} eps={eps:<8.4g} cond={cond:10.1f}  "
          f"n_active={stat['n_active']:2d}  "
          f"mean_cv={stat['mean_cv']:.5f}  median_cv={stat['median_cv']:.5f}")
    return stat


if __name__ == "__main__":
    indep = run_case("independent", 0.0)
    collin = run_case("near-collinear", 0.9999)

    print()
    for key in ("mean_cv", "median_cv"):
        a, b = indep[key], collin[key]
        ratio = b / a if a > 0 else np.inf
        print(f"{key:10s} separation ratio {ratio:8.1f}x  "
              f"({a:.5f} -> {b:.5f})")

    # gate on separation, not on absolute values. if the collinear case isn't at
    # least an order of magnitude worse, the conditioning is wrong somewhere.
    ok = (collin["median_cv"] / indep["median_cv"] > 10.0
          and indep["n_active"] >= 2
          and collin["n_active"] >= 2)
    print(f"-> {'PASS' if ok else 'FAIL'}")
