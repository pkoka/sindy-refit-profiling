"""
Analysis of a gate calibration CSV.

Reports LIFT over the majority-class baseline, not raw accuracy. A pooled
"80.6% correct" on a set that is 75% good runs is worth 5.6 points, not 80.6.
Several cells in the low-noise sweep turned out to have zero lift once the
baseline was accounted for.

Also reports a held-out threshold: chosen on even seeds, scored on odd. The
in-sample best-threshold numbers are optimistically biased because the threshold
is picked on the same runs it is graded against.

Three questions:
  1. within a fixed regime, is CV monotone in actual error? (rank correlation)
  2. within a fixed regime, does a threshold beat always-accept? (lift)
  3. does normalizing by a per-regime baseline recover pooled separability?

Q3 answered NO in both sweeps to date, and is kept only because the negative
result is cited in the writeup.

usage:
    python analyze_calib.py
    python analyze_calib.py --csv gate_calib_lownoise.csv --err-tol 0.05
"""

import argparse
import csv
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "rho": float(r["rho"]),
                "d": int(r["n_features"]),
                "N": int(r["n_samples"]),
                "noise": float(r["noise"]),
                "seed": int(r["seed"]),
                "cond": float(r["cond"]),
                "mean_cv": float(r["mean_cv"]),
                "median_cv": float(r["median_cv"]),
                "max_cv": float(r["max_cv"]),
                "n_active": int(r["n_active"]),
                "err": float(r["coef_err"]),
                "spurious": int(r["spurious"]),
            })
    return rows


def best_threshold(good, bad):
    """single threshold maximizing correct decisions. returns (thresh, accuracy)."""
    if not good or not bad:
        return None, None
    cand = sorted(set(good) | set(bad))
    best, best_acc = None, -1.0
    for t in cand:
        acc = (sum(1 for g in good if g <= t) + sum(1 for b in bad if b > t)) / (len(good) + len(bad))
        if acc > best_acc:
            best, best_acc = t, acc
    return best, best_acc


def majority_baseline(good, bad):
    """always-accept or always-reject, whichever wins. a threshold that doesn't
    beat this is doing nothing at all."""
    n = len(good) + len(bad)
    return max(len(good), len(bad)) / n if n else None


def score_at(threshold, good, bad):
    n = len(good) + len(bad)
    if n == 0 or threshold is None:
        return None
    return (sum(1 for g in good if g <= threshold)
            + sum(1 for b in bad if b > threshold)) / n


def split_classes(rows, key, err_tol):
    good = [r[key] for r in rows if r["err"] <= err_tol and np.isfinite(r[key])]
    bad = [r[key] for r in rows if r["err"] > err_tol and np.isfinite(r[key])]
    return good, bad


def heldout(rows, key, err_tol, fit_pred=lambda r: r["seed"] % 2 == 0):
    """
    pick the threshold on one half of the seeds, score it on the other half.
    seed-wise split so the two halves are independent draws of the same
    conditions rather than different conditions.
    """
    fit_rows = [r for r in rows if fit_pred(r)]
    test_rows = [r for r in rows if not fit_pred(r)]

    g_fit, b_fit = split_classes(fit_rows, key, err_tol)
    g_test, b_test = split_classes(test_rows, key, err_tol)
    if not (g_fit and b_fit and g_test and b_test):
        return None

    t, _ = best_threshold(g_fit, b_fit)
    acc = score_at(t, g_test, b_test)
    base = majority_baseline(g_test, b_test)
    return {"threshold": t, "acc": acc, "baseline": base, "lift": acc - base,
            "n_fit": len(g_fit) + len(b_fit), "n_test": len(g_test) + len(b_test)}


def spearman(cvs, errs):
    if HAVE_SCIPY:
        return float(spearmanr(cvs, errs).statistic)
    rc, re_ = np.argsort(np.argsort(cvs)), np.argsort(np.argsort(errs))
    return float(np.corrcoef(rc, re_)[0, 1])


def within_regime(rows, key, err_tol):
    """monotonicity and lift inside each (d, N, noise) cell"""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["d"], r["N"], r["noise"])].append(r)

    print(f"\n=== within-regime, {key} ===")
    print(f"{'d':>4} {'N':>6} {'noise':>7} {'n':>4} {'bad':>4} "
          f"{'spear':>7} {'base':>7} {'acc':>7} {'lift':>7}")
    lifts, sps = [], []
    for k in sorted(groups):
        g = [r for r in groups[k] if np.isfinite(r[key])]
        if len(g) < 4:
            continue
        sp = spearman([r[key] for r in g], [r["err"] for r in g])
        good, bad = split_classes(g, key, err_tol)
        base = majority_baseline(good, bad)
        t, acc = best_threshold(good, bad)

        if acc is None:
            # every run in this cell is one class. a gate can still be correct
            # here (reject everything), it just cannot RANK within the cell
            print(f"{k[0]:>4} {k[1]:>6} {k[2]:>7.3g} {len(g):>4} {len(bad):>4} "
                  f"{sp:>7.3f} {base:>7.1%} {'-':>7} {'one class':>7}")
            continue

        lift = acc - base
        lifts.append(lift)
        if np.isfinite(sp):
            sps.append(sp)
        print(f"{k[0]:>4} {k[1]:>6} {k[2]:>7.3g} {len(g):>4} {len(bad):>4} "
              f"{sp:>7.3f} {base:>7.1%} {acc:>7.1%} {lift:>+7.1%}")

    if sps:
        print(f"\nmedian within-regime spearman(CV, err): {np.median(sps):.3f}")
    if lifts:
        print(f"median within-regime lift over baseline: {np.median(lifts):+.1%}")
        print(f"cells with zero or negative lift:        "
              f"{sum(1 for l in lifts if l <= 0)}/{len(lifts)}")


def pooled(rows, key, err_tol):
    good, bad = split_classes(rows, key, err_tol)
    print(f"\n=== pooled, {key} ===")
    if not good or not bad:
        print("  only one class present")
        return
    base = majority_baseline(good, bad)
    t, acc = best_threshold(good, bad)
    print(f"  good n={len(good):4d}  bad n={len(bad):4d}  baseline {base:.1%}")
    print(f"  in-sample  threshold {t:8.4f}  acc {acc:.1%}  lift {acc - base:+.1%}")

    ho = heldout(rows, key, err_tol)
    if ho is None:
        print("  held-out: not enough of both classes in both seed halves")
    else:
        print(f"  held-out   threshold {ho['threshold']:8.4f}  acc {ho['acc']:.1%}  "
              f"lift {ho['lift']:+.1%}  (fit n={ho['n_fit']}, test n={ho['n_test']})")


def relative_gate(rows, key, err_tol):
    """
    normalize each run's CV by its regime's baseline (median CV of the rho=0
    runs in the same cell). NEGATIVE RESULT in both sweeps to date: this scores
    well below a pooled absolute threshold. kept because the writeup cites it.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[(r["d"], r["N"], r["noise"])].append(r)

    ratios_good, ratios_bad = [], []
    for k, g in groups.items():
        base_vals = [r[key] for r in g if r["rho"] == 0.0 and np.isfinite(r[key])]
        if not base_vals:
            continue
        base = float(np.median(base_vals))
        if base <= 0:
            continue
        for r in g:
            if not np.isfinite(r[key]):
                continue
            (ratios_bad if r["err"] > err_tol else ratios_good).append(r[key] / base)

    print(f"\n=== baseline-relative gate, {key} ===")
    if not ratios_good or not ratios_bad:
        print("  not enough of both classes")
        return
    mbase = majority_baseline(ratios_good, ratios_bad)
    t, acc = best_threshold(ratios_good, ratios_bad)
    print(f"  best relative threshold {t:.2f}x baseline  acc {acc:.1%}  "
          f"lift {acc - mbase:+.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="gate_calib.csv")
    p.add_argument("--err-tol", type=float, default=0.05)
    args = p.parse_args()

    rows = load(args.csv)
    print(f"loaded {len(rows)} runs from {args.csv}  (scipy: {HAVE_SCIPY})  "
          f"err_tol={args.err_tol}")

    for key in ("mean_cv", "median_cv"):
        within_regime(rows, key, args.err_tol)
        pooled(rows, key, args.err_tol)
        relative_gate(rows, key, args.err_tol)
