"""
Parallel E-SINDy refit: bootstrap members are independent, so the ensemble
loop is embarrassingly parallel. Day 1 showed cost scales as B^0.98, i.e.
almost exactly linear in ensemble count, so this is the right lever.

reuses the Day 1 harness for library construction and buffer generation

usage:
    python refit_parallel.py --check                      # correctness vs EnsembleOptimizer
    python refit_parallel.py --check-backends             # loky vs threading equivalence
    python refit_parallel.py --sweep --out FILE.csv       # n_jobs speedup sweep
"""

import argparse
import os
import platform
import time

import numpy as np
from joblib import Parallel, delayed
from pysindy.optimizers import EnsembleOptimizer, STLSQ

from refit_profile import (ALPHA, N_STATE, THRESHOLD, build_theta,
                           make_buffer, time_refit)


def fit_one_member(theta, y, seed, n_drop=1):
    """
    fit a single ensemble member
    mirrors EnsembleOptimizer._reduce:
      1. bootstrap rows: n_subset = n_samples, replace=True
      2. drop n_drop library columns, chosen without replacement, sorted
      3. fit a FRESH STLSQ (never share one across workers)
      4. scatter coefficients back into a full-width zero array
    returns array of shape (n_targets, n_features)
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = theta.shape
    rows = rng.integers(0, n_samples, n_samples)
    theta_boot, y_boot = theta[rows], y[rows]
    keep_inds = np.sort(rng.choice(n_features, n_features - n_drop, replace=False))
    theta_sub = theta_boot[:, keep_inds]
    opt = STLSQ(threshold=THRESHOLD, alpha=ALPHA)
    opt.fit(theta_sub, y_boot)
    coef = np.zeros((y.shape[1], n_features))
    coef[:, keep_inds] = opt.coef_
    return coef


def ensemble_parallel(theta, y, n_models=20, n_jobs=1, base_seed=0,
                      backend=None, batch_size="auto"):
    """
    run n_models bootstrap members under joblib and aggregate them

    each member gets distinct seed (base_seed + i) so bootstrap row draws and
    column dropouts differ across members

    backend:
        None       joblib default (loky, process based, pickles theta per task)
        "loky"     explicit processes
        "threading" threads. np.linalg.lstsq inside STLSQ releases the GIL in
                   LAPACK, so this may parallelize with zero pickling of theta.
                   only valid because members share no mutable state.
    batch_size: members per dispatched task. raise it to amortize dispatch cost
                when per member work is small relative to transfer.

    returns:
        median  : ndarray, shape (n_targets, n_features)
                  element-wise median across members, this is the refit estimate
        stacked : ndarray, shape (n_models, n_targets, n_features)
                  all member coefficients, retained for spread diagnostics
                  e.g. coefficient of variation over active terms, for use as a
                  refit-acceptance gate.
    """
    theta = np.asarray(theta).view(np.ndarray)
    y = np.asarray(y).view(np.ndarray)
    coefs = Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size)(
        delayed(fit_one_member)(theta, y, base_seed + i) for i in range(n_models)
    )
    stacked = np.stack(coefs, axis=0)
    return np.median(stacked, axis=0), stacked


def check_correctness(n_samples=8000, degree=2, n_models=20, n_trials=10):
    """
    compare parallel implementation against EnsembleOptimizer
    EnsembleOptimizer uses global np.random, so exact equality not available
    compare instead:
      - recovered support (which coefficients are nonzero) should match
      - nonzero values should agree to within few percent
    print PASS/FAIL
    don't proceed to timing till this passes
    """
    theta, y = make_buffer(n_samples, degree)
    n_samples = theta.shape[0]

    ref_coefs, ref_members = [], []
    for _ in range(n_trials):
        ens = EnsembleOptimizer(
            STLSQ(threshold=THRESHOLD, alpha=ALPHA),
            bagging=True,
            n_subset=n_samples,          # bootstrap at full size
            n_models=n_models,
            library_ensemble=True,
            n_candidates_to_drop=1,
        )
        ens.fit(theta, y)
        ref_coefs.append(ens.coef_.copy())
        ref_members.append(np.array(ens.coef_list))   # (n_models, n_targets, n_features)

    par_coefs, par_members = [], []
    for t in range(n_trials):
        C, S = ensemble_parallel(theta, y, n_models=n_models,
                                 n_jobs=1, base_seed=1000 * t)
        par_coefs.append(C)
        par_members.append(S)

    ref = np.stack(ref_coefs)     # (n_trials, n_targets, n_features)
    par = np.stack(par_coefs)

    # support: "stable" = nonzero in every trial; "any" = nonzero in at least one
    ref_stable, ref_any = np.all(ref != 0, axis=0), np.any(ref != 0, axis=0)
    par_stable, par_any = np.all(par != 0, axis=0), np.any(par != 0, axis=0)
    only_ref = ref_stable & ~par_any     # reference always finds it, i never do
    only_par = par_stable & ~ref_any     # i always find it, reference never does
    shared = ref_stable & par_stable

    if shared.sum() == 0:
        print("FAIL: no stable terms in common")
        return

    ref_mean, par_mean = ref.mean(axis=0), par.mean(axis=0)
    rel_diff = np.abs(par_mean[shared] - ref_mean[shared]) / np.abs(ref_mean[shared])

    # member-level spread, averaged over trials, this is what n_subset controls
    ref_spread = np.mean([m.std(axis=0) for m in ref_members], axis=0)[shared]
    par_spread = np.mean([m.std(axis=0) for m in par_members], axis=0)[shared]
    ratio = par_spread / ref_spread

    # negative control: n_candidates_to_drop=3 on the reference drives median spread ratio to 0.63 -> FAIL,
    # test is sensitive to variance-only defects, mean comparison alone isn't
    # --------------------------------
    # gate on median spread ratio only
    # per-term ratios have wide tails (p90 abt 2.0 even at n_trials=10) because library dropout
    # contributes discrete per-term hits, this doesn't avg out
    # requires n_trials >= 10, at n_trials=3 the median itself is unstable (observed 0.63-1.50 across runs of identical code)
    ok = (only_ref.sum() == 0 and only_par.sum() == 0
          and rel_diff.max() < 0.05
          and 0.7 < np.median(ratio) < 1.4)

    print(f"stable support     ref {ref_stable.sum():3d}  par {par_stable.sum():3d}")
    print(f"disagreement       ref-only {only_ref.sum()}  par-only {only_par.sum()}")
    print(f"rel difference     median {np.median(rel_diff):.4f}  max {rel_diff.max():.4f}")
    print(f"spread ratio       median {np.median(ratio):.3f} "
          f"p10 {np.percentile(ratio,10):.3f}  p90 {np.percentile(ratio,90):.3f}")
    print(f"-> {'PASS' if ok else 'FAIL'}")


def check_backends(n_samples=8000, degree=2, n_models=20, n_jobs=4):
    """threading vs loky must be bit identical. seeds are per member and members
    share no state, so backend choice cannot change the answer. if this fails,
    something is sharing state and the threading numbers are meaningless."""
    theta, y = make_buffer(n_samples, degree)
    _, a = ensemble_parallel(theta, y, n_models, n_jobs, backend="loky")
    _, b = ensemble_parallel(theta, y, n_models, n_jobs, backend="threading")
    ok = bool(np.array_equal(a, b))
    print(f"loky vs threading  identical {ok}  max abs diff {np.abs(a - b).max():.3e}")
    print(f"-> {'PASS' if ok else 'FAIL'}")


def sweep_njobs(configs, n_jobs_grid, out="parallel_times.csv",
                backends=("loky", "threading"), reps=7):
    """Time each config across n_jobs and backend. Report speedup vs n_jobs=1
    within the same backend.

    Two configs with different per-member work:
        small members: N=16000, degree=2, B=20   (the published anchor)
        large members: N=32000, degree=3, B=40   (worst case)

    Parallel efficiency should be noticeably better for the large one, since
    process overhead amortizes over longer fits.

    reports min and p90. min estimates what the machine can do, p90 is what
    decides whether a 10Hz deadline is actually met.
    """
    import csv

    n_cores = os.cpu_count()
    grid = [j for j in n_jobs_grid if j <= n_cores]
    machine = platform.machine() + "-" + platform.node()
    blas = os.environ.get("OMP_NUM_THREADS", "unset")
    rows = []

    for (N, degree, B) in configs:
        theta, y = make_buffer(N, degree)
        d = theta.shape[1]
        for backend in backends:
            baseline = None
            for nj in grid:
                ensemble_parallel(theta, y, n_models=B, n_jobs=nj, backend=backend)  # warm-up
                ts = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    ensemble_parallel(theta, y, n_models=B, n_jobs=nj, backend=backend)
                    ts.append(time.perf_counter() - t0)
                t_min = min(ts)
                t_p90 = float(np.percentile(ts, 90))
                if baseline is None:
                    baseline = t_min
                rows.append(dict(machine=machine, n_cores=n_cores, blas_threads=blas,
                                 backend=backend, N=N, degree=degree, d=d, B=B,
                                 n_jobs=nj,
                                 refit_s=round(t_min, 4), refit_s_p90=round(t_p90, 4),
                                 hz=round(1.0 / t_min, 3), hz_p90=round(1.0 / t_p90, 3),
                                 speedup=round(baseline / t_min, 3)))
                print(f"{backend:9s} N={N} d={d} B={B} n_jobs={nj:3d} "
                      f"min {t_min:7.3f}s  p90 {t_p90:7.3f}s  {baseline/t_min:5.2f}x")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--check-backends", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--out", default="parallel_times.csv")
    p.add_argument("--backends", default="loky,threading")
    args = p.parse_args()

    if args.check:
        check_correctness()
    if args.check_backends:
        check_backends()
    if args.sweep:
        sweep_njobs(
            configs=[(16000, 2, 20), (32000, 3, 40)],
            n_jobs_grid=[1, 2, 3, 4, 5, 6, 8, 10, 12, 15],
            out=args.out,
            backends=tuple(args.backends.split(",")),
        )