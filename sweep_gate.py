"""
Threshold calibration for the refit acceptance gate.

v2: N and noise are the primary axes, not rho.

v1 swept only collinearity at N=8000, noise=1e-3 and every single run recovered
the true model exactly, so there was no "bad" class to calibrate a threshold
against. Collinearity alone does not break the fit when data is plentiful. The
pathology Larranaga studies is the ULTRA-LOW-DATA limit, so N has to come down
and noise has to come up before a redundant library direction actually costs
anything.

Records, per run, both the gate statistic and the ACTUAL coefficient error
against the known true model. The threshold is read off where CV starts
predicting unacceptable error.

Also sweeps library dimension, because the mean CV averages the one bad term
against however many clean ones there are, so the statistic's sensitivity is
library-size dependent. v1 showed d=27 CVs running 5-10x below d=6 at matched
rho.

usage:
    python sweep_gate.py
    python sweep_gate.py --out FILE.csv --err-tol 0.05
"""

import argparse
import csv

import numpy as np

from gate import gate_cv
from refit_parallel import ensemble_parallel

TRUE_C0 = 1.5
TRUE_C2 = 0.8
N_MODELS = 20


def make_synthetic(rho, n_features, n_samples, noise, seed=0):
    """col1 correlates with col0 at correlation rho. true model uses col0 and
    col2 only, so col1 is a planted redundant direction with true coefficient 0."""
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((n_samples, n_features))
    theta[:, 1] = rho * theta[:, 0] + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(n_samples)

    y = (TRUE_C0 * theta[:, 0] + TRUE_C2 * theta[:, 2]).reshape(-1, 1)
    y = y + noise * rng.standard_normal(y.shape)
    return theta, y


def true_coefs(n_features):
    c = np.zeros((1, n_features))
    c[0, 0] = TRUE_C0
    c[0, 2] = TRUE_C2
    return c


def run_point(rho, n_features, n_samples, noise, seed):
    theta, y = make_synthetic(rho, n_features, n_samples, noise, seed)
    median, stacked, masks = ensemble_parallel(
        theta, y, n_models=N_MODELS, n_jobs=1, base_seed=1000 * seed
    )
    stat = gate_cv(stacked, masks, median=median)
    truth = true_coefs(n_features)

    # relative L2 error of the ADOPTED model (the median, which is what gets
    # deployed). penalizes spurious terms too, a gate ignoring them isn't working
    err = float(np.linalg.norm(median - truth) / np.linalg.norm(truth))

    spurious = int(np.sum((median != 0) & (truth == 0)))
    missed = int(np.sum((median == 0) & (truth != 0)))

    return dict(
        rho=rho, n_features=n_features, n_samples=n_samples, noise=noise, seed=seed,
        cond=float(np.linalg.cond(theta)),
        mean_cv=stat["mean_cv"],
        median_cv=stat["median_cv"],
        max_cv=stat["max_cv"],
        n_active=stat["n_active"],
        coef_err=err,
        spurious=spurious,
        missed=missed,
    )


def sweep(rhos, feature_dims, sample_counts, noises, seeds, out):
    rows = []
    for d in feature_dims:
        for N in sample_counts:
            for noise in noises:
                for rho in rhos:
                    grp = []
                    for s in seeds:
                        r = run_point(rho, d, N, noise, s)
                        rows.append(r)
                        grp.append(r)
                    finite = [x["mean_cv"] for x in grp if np.isfinite(x["mean_cv"])]
                    mcv = np.median(finite) if finite else np.inf
                    print(f"d={d:3d} N={N:6d} noise={noise:<6.3g} rho={rho:<7.4f} "
                          f"cond={np.median([x['cond'] for x in grp]):8.1f}  "
                          f"mean_cv={mcv:9.5f}  "
                          f"err={np.median([x['coef_err'] for x in grp]):9.6f}  "
                          f"act={np.median([x['n_active'] for x in grp]):.1f}  "
                          f"spur={np.median([x['spurious'] for x in grp]):.1f}  "
                          f"miss={np.median([x['missed'] for x in grp]):.1f}")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out}")
    return rows


def report(rows, err_tol=0.05):
    """largest CV among acceptable runs vs smallest among bad ones. if those
    ranges don't overlap, any threshold between them works."""
    n_bad = sum(1 for r in rows if r["coef_err"] > err_tol)
    print(f"\nruns: {len(rows)}  bad (err>{err_tol}): {n_bad}  "
          f"good: {len(rows) - n_bad}")
    if n_bad == 0:
        print("no bad runs. push N lower or noise higher, nothing to calibrate against")
        return

    for key in ("mean_cv", "median_cv", "max_cv"):
        good = [r[key] for r in rows if r["coef_err"] <= err_tol and np.isfinite(r[key])]
        bad = [r[key] for r in rows if r["coef_err"] > err_tol and np.isfinite(r[key])]
        if not good or not bad:
            print(f"{key}: not enough of both classes")
            continue
        print(f"\n{key}")
        print(f"  good n={len(good):4d}  max={max(good):.4f}  p95={np.percentile(good, 95):.4f}")
        print(f"  bad  n={len(bad):4d}  min={min(bad):.4f}  p5={np.percentile(bad, 5):.4f}")
        if min(bad) > max(good):
            print(f"  cleanly separable -> threshold in ({max(good):.4f}, {min(bad):.4f})")
        else:
            # overlapping. report the threshold maximizing correct decisions
            cand = sorted(set(good + bad))
            best, best_acc = None, -1.0
            for t in cand:
                acc = (sum(1 for g in good if g <= t) + sum(1 for b in bad if b > t)) / (len(good) + len(bad))
                if acc > best_acc:
                    best, best_acc = t, acc
            print(f"  overlapping. best single threshold {best:.4f} "
                  f"at {best_acc:.1%} correct decisions")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="gate_calib.csv")
    p.add_argument("--err-tol", type=float, default=0.05)
    args = p.parse_args()

    rows = sweep(
        rhos=[0.0, 0.99, 0.999, 0.9999, 0.99999, 0.999999],
        feature_dims=[6, 27],
        sample_counts=[30, 50, 100, 250],
        noises=[0.01],
        seeds=range(8),
        out=args.out,
    )
    report(rows, err_tol=args.err_tol)