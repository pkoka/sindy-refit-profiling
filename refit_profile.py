"""
E-SINDy refit cost profiling on swing-up-shaped data.

Measures wall-clock cost of REFITTING the E-SINDy dynamics model
(not policy inference) as a function of:
    N  - buffer size
    d  - library dimension (set by polynomial degree)
    B  - ensemble count

Config mirrors SINDy-RL (Zolman et al. 2025) Table 5, swing-up row:
    quadratic library, STLRidge threshold 7e-3, alpha 5e-5,
    B = 20, median aggregation, discrete-time model.

Usage:
    python refit_profile.py            # full sweep
    python refit_profile.py --quick    # fast smoke test
"""
import argparse
import csv
import platform
import time

import numpy as np
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import EnsembleOptimizer, STLSQ

# swing-up observation: (x, cos(theta), sin(theta), xdot, thetadot)
N_STATE = 5
N_CTRL = 1          # force along x

THRESHOLD = 7e-3    # STLRidge threshold, Table 5
ALPHA = 5e-5        # ridge weight, Table 5

# Sweep grid. N=16000 is their effective buffer:
# 8000 offline + 8000-capped on-policy queue.
N_GRID = [1000, 2000, 4000, 8000, 16000, 32000]
DEGREE_GRID = [2, 3]
B_GRID = [5, 10, 20, 40]
N_REPEAT = 7

QUICK_N = [2000, 8000]
QUICK_DEGREE = [2]
QUICK_B = [20]


def make_buffer(n_samples, degree, seed=0, n_active=4):
    """Buffer whose target IS a sparse combination of library terms, so
    STLSQ retains a realistic active set instead of thresholding to near-zero."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, N_STATE))
    u = rng.uniform(-1.0, 1.0, (n_samples, N_CTRL))
    theta = build_theta(np.hstack([x, u]), degree)
    xi = np.zeros((theta.shape[1], N_STATE))
    for j in range(N_STATE):
        idx = rng.choice(theta.shape[1], n_active, replace=False)
        xi[idx, j] = rng.uniform(0.5, 2.0, n_active)
    y = theta @ xi + 0.01 * rng.standard_normal((n_samples, N_STATE))
    return theta, y


def build_theta(x_aug, degree):
    """Evaluate the candidate library. Column count is the library
    dimension d that drives regression cost."""
    lib = PolynomialLibrary(degree=degree, include_bias=False)
    return lib.fit_transform(x_aug)


def time_refit(theta, y, n_models, n_repeat=N_REPEAT):
    """Median wall-clock seconds for one full E-SINDy refit.

    EnsembleOptimizer with bagging + library_ensemble is E-SINDy:
    bootstrap resampling of rows plus dropout of library terms.
    Fits are sequential here, matching the published implementation.
    """
    times = []
    for _ in range(n_repeat):
        opt = EnsembleOptimizer(
            opt=STLSQ(threshold=THRESHOLD, alpha=ALPHA),
            bagging=True,
            library_ensemble=True,
            n_models=n_models,
            ensemble_aggregator=lambda c: np.median(c, axis=0),
        )
        start = time.perf_counter()
        opt.fit(theta, y)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="small grid for a fast smoke test")
    parser.add_argument("--out", default="refit_times.csv")
    args = parser.parse_args()

    n_grid = QUICK_N if args.quick else N_GRID
    degree_grid = QUICK_DEGREE if args.quick else DEGREE_GRID
    b_grid = QUICK_B if args.quick else B_GRID

    machine = f"{platform.platform()} | {platform.processor() or platform.machine()}"
    print(f"machine: {machine}\n")
    print(f"{'N':>7} {'deg':>4} {'d':>4} {'B':>4} {'refit_s':>9} {'Hz':>8}")

    rows = []
    for n_samples in n_grid:
        for degree in degree_grid:
            theta, y = make_buffer(n_samples, degree)
            d = theta.shape[1]
            for n_models in b_grid:
                seconds = time_refit(theta, y, n_models)
                hz = 1.0 / seconds if seconds > 0 else float("inf")
                print(f"{n_samples:>7} {degree:>4} {d:>4} {n_models:>4} "
                      f"{seconds:>9.3f} {hz:>8.2f}")
                rows.append({
                    "machine": machine,
                    "N": n_samples,
                    "degree": degree,
                    "d": d,
                    "B": n_models,
                    "refit_s": round(seconds, 6),
                    "hz": round(hz, 4),
                    "mode": "sequential",
                })

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
